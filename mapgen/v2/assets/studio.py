"""AssetStudio: orchestrates asset generation with caching and post-processing.

Each AssetSpec is generated independently, cached by content hash, and
post-processed by kind: sprites/POIs are matted from their flat key to
RGBA, textures are verified/repaired for tileability. The style bible is
generated first and passed as a style reference to every other call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Protocol

from PIL import Image

from ..color import palette_distance
from ..types import AssetKind, AssetSpec, PlanDocument, StyleSpec
from .matting import KEY_SHIFT_THRESHOLD, detect_key_color, key_shift, key_to_alpha
from .textures import ensure_tileable

logger = logging.getLogger(__name__)

# Worst-decile ΔE76 above which an asset is flagged as a palette outlier.
# In-palette assets with normal shading variation score well under this;
# wholesale hue drift (the failure mode worth a human look) scores well over.
PALETTE_OUTLIER_THRESHOLD = 35.0


class AssetGenerator(Protocol):
    """Anything that can produce a raw image for an AssetSpec."""

    def generate(
        self,
        spec: AssetSpec,
        style: StyleSpec,
        style_reference: Optional[Image.Image] = None,
    ) -> Image.Image: ...


class AssetStudio:
    def __init__(self, generator: AssetGenerator, cache_dir: Path | str):
        self.generator = generator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def asset_path(self, spec: AssetSpec) -> Path:
        return self.cache_dir / f"{spec.id}.png"

    def _meta_path(self, spec: AssetSpec) -> Path:
        return self.cache_dir / f"{spec.id}.json"

    def _save_raw(self, spec: AssetSpec, raw: Image.Image) -> None:
        """Keep the unprocessed generation so matting/post-process changes
        can be re-applied later without spending another API call."""
        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        raw.save(raw_dir / f"{spec.id}.png")

    def is_cached(self, spec: AssetSpec) -> bool:
        meta = self._meta_path(spec)
        if not meta.exists() or not self.asset_path(spec).exists():
            return False
        try:
            return json.loads(meta.read_text()).get("hash") == spec.content_hash()
        except (json.JSONDecodeError, OSError):
            return False

    def generate_all(
        self,
        plan: PlanDocument,
        force: bool = False,
        only_ids: Optional[set[str]] = None,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, Path]:
        """Generate (or load from cache) every asset in the plan manifest.

        Returns a map of asset id -> file path. The style bible is always
        processed first so it can serve as the style reference.
        """
        specs = sorted(plan.manifest, key=lambda s: s.kind != AssetKind.STYLE_BIBLE)
        style_reference: Optional[Image.Image] = None
        out: dict[str, Path] = {}
        failures: list[str] = []
        total = len(specs)
        for i, spec in enumerate(specs):
            if progress:
                progress(spec.id, i, total)
            skip = only_ids is not None and spec.id not in only_ids
            if (skip or not force) and self.is_cached(spec):
                out[spec.id] = self.asset_path(spec)
                if spec.kind == AssetKind.STYLE_BIBLE:
                    style_reference = Image.open(out[spec.id]).convert("RGB")
                continue
            if skip:
                continue

            logger.info("Generating asset %s (%s)", spec.id, spec.kind.value)
            try:
                raw = self.generator.generate(spec, plan.style, style_reference)
                self._save_raw(spec, raw)
                processed = self._post_process(spec, raw)
                path = self.asset_path(spec)
                processed.save(path)
                self._write_meta(spec, raw, processed, plan.style)
            except Exception as exc:
                # One bad asset must not waste the rest of a paid batch --
                # except the style bible, which everything else references.
                if spec.kind == AssetKind.STYLE_BIBLE:
                    raise
                logger.error("Asset %s failed (%s); continuing with the rest", spec.id, exc)
                failures.append(spec.id)
                continue
            out[spec.id] = path
            if spec.kind == AssetKind.STYLE_BIBLE:
                style_reference = processed.convert("RGB")
        if progress:
            progress("done", total, total)
        if failures:
            raise RuntimeError(
                f"{len(failures)} asset(s) failed after retries: {', '.join(failures)}. "
                f"The other {len(out)} are cached; re-run the assets stage to retry the failures."
            )
        return out

    def _write_meta(
        self,
        spec: AssetSpec,
        raw: Image.Image,
        processed: Image.Image,
        style: StyleSpec,
        reprocessed: bool = False,
    ) -> None:
        score = palette_distance(processed, style.palette)
        meta = {
            "hash": spec.content_hash(),
            "kind": spec.kind.value,
            "subject": spec.subject,
            "palette_score": round(score, 2),
            "palette_outlier": score > PALETTE_OUTLIER_THRESHOLD,
        }
        if reprocessed:
            meta["reprocessed"] = True
        if spec.kind in (AssetKind.SPRITE_SHEET, AssetKind.POI_SPRITE, AssetKind.ORNAMENT, AssetKind.SHIELD):
            key = detect_key_color(raw)
            shift = key_shift(key)
            meta["detected_key"] = list(key)
            meta["key_shift"] = round(shift, 1)
            if shift > KEY_SHIFT_THRESHOLD:
                meta["key_shifted"] = True
                logger.warning(
                    "Asset %s came back on a palette-shifted key %s (shift %.0f); "
                    "matted via the key-family gate but worth a visual check",
                    spec.id,
                    key,
                    shift,
                )
        self._meta_path(spec).write_text(json.dumps(meta))

    def reprocess_all(
        self,
        plan: PlanDocument,
        only_ids: Optional[set[str]] = None,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, Path]:
        """Re-run post-processing from the saved raw generations -- no API
        spend. Used after matting/post-process fixes to repair assets in
        place. Assets without a saved raw are skipped."""
        specs = [s for s in plan.manifest if only_ids is None or s.id in only_ids]
        out: dict[str, Path] = {}
        for i, spec in enumerate(specs):
            raw_path = self.cache_dir / "raw" / f"{spec.id}.png"
            if not raw_path.exists():
                continue
            if progress:
                progress(spec.id, i, len(specs))
            raw = Image.open(raw_path)
            processed = self._post_process(spec, raw)
            path = self.asset_path(spec)
            processed.save(path)
            self._write_meta(spec, raw, processed, plan.style, reprocessed=True)
            out[spec.id] = path
        if progress:
            progress("done", len(specs), len(specs))
        return out

    def read_meta(self, asset_id: str) -> dict:
        meta = self.cache_dir / f"{asset_id}.json"
        if not meta.exists():
            return {}
        try:
            return json.loads(meta.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def set_flagged(self, asset_id: str, flagged: bool) -> dict:
        """Mark an asset for human-reviewed regeneration. The flag lives in
        the asset's meta file and is cleared naturally when the asset is
        regenerated (meta is rewritten)."""
        meta_path = self.cache_dir / f"{asset_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no asset meta for {asset_id!r}")
        meta = self.read_meta(asset_id)
        meta["flagged"] = flagged
        meta_path.write_text(json.dumps(meta))
        return meta

    def flagged_ids(self) -> set[str]:
        return {
            p.stem
            for p in self.cache_dir.glob("*.json")
            if self.read_meta(p.stem).get("flagged")
        }

    def _post_process(self, spec: AssetSpec, img: Image.Image) -> Image.Image:
        if img.size != (spec.width_px, spec.height_px):
            # Never squash: when the model returns a different aspect ratio,
            # scale to fit inside the requested box instead of distorting
            # (sprite sheets excepted -- their grid must match the spec).
            if spec.kind == AssetKind.SPRITE_SHEET:
                img = img.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
            else:
                scale = min(spec.width_px / img.width, spec.height_px / img.height)
                img = img.resize(
                    (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                    Image.Resampling.LANCZOS,
                )
        if spec.kind == AssetKind.TEXTURE:
            return ensure_tileable(img)
        if spec.kind in (AssetKind.SPRITE_SHEET, AssetKind.POI_SPRITE, AssetKind.ORNAMENT, AssetKind.SHIELD):
            # Key out the flat magenta background. Gemini returns opaque RGBA
            # PNGs (alpha channel present but all 255), so mode == "RGBA" does
            # NOT mean the background is already transparent -- only skip keying
            # when the image carries real (varying) transparency.
            if img.mode == "RGBA" and _has_real_transparency(img):
                keyed = img
            else:
                keyed = key_to_alpha(img)
            if spec.kind == AssetKind.SPRITE_SHEET:
                return keyed
            # Trim transparent margins so the compositor's aspect-fit works
            # against the subject's true bounds, not the generation canvas.
            from .matting import trim_to_content

            return trim_to_content(keyed)
        return img.convert("RGB")


def _has_real_transparency(img: Image.Image) -> bool:
    """True if the alpha channel actually varies (genuine cut-out), not a
    uniformly-opaque channel that merely happens to be present."""
    alpha = img.getchannel("A")
    lo, hi = alpha.getextrema()
    return lo < 250


def load_sprites_from_sheet(sheet_path: Path, grid: tuple[int, int]) -> list[Image.Image]:
    """Cut a sprite sheet into individual trimmed RGBA sprites.

    Image models don't reliably honour an exact NxM grid -- one sheet comes
    back as a 6x3 of saguaros, another as a single big house -- so a fixed-grid
    crop slices sprites mid-body and grabs background. Segment by content
    instead: key out the flat (magenta) background and treat each connected
    blob as one sprite. Fall back to the requested grid only if that finds
    nothing.
    """
    import numpy as np
    from scipy import ndimage

    from .matting import detect_key_color, key_to_alpha, trim_to_content

    sheet = Image.open(sheet_path).convert("RGBA")

    # Clean alpha: keep existing transparency if already matted, else key out
    # the (usually magenta) flat background.
    alpha0 = np.asarray(sheet.getchannel("A"))
    matted = sheet if (alpha0 < 30).mean() > 0.1 else key_to_alpha(sheet, key=detect_key_color(sheet))
    alpha = np.asarray(matted.getchannel("A"))

    mask = ndimage.binary_dilation(alpha > 40, iterations=3)  # bridge thin gaps
    labels, _ = ndimage.label(mask)
    total = mask.size
    found: list[tuple[int, int, Image.Image]] = []
    for box in ndimage.find_objects(labels):
        if box is None:
            continue
        ys, xs = box
        area = (ys.stop - ys.start) * (xs.stop - xs.start)
        if area < total * 0.004 or area > total * 0.6:  # drop noise + whole-sheet blobs
            continue
        cell = matted.crop((xs.start, ys.start, xs.stop, ys.stop))
        found.append((ys.start, xs.start, trim_to_content(cell)))

    if found:
        found.sort(key=lambda f: (f[0] // 64, f[1]))  # reading order, row-banded
        return [f[2] for f in found]

    # Fallback: the requested grid.
    cols, rows = grid
    cell_w, cell_h = sheet.width // cols, sheet.height // rows
    return [
        trim_to_content(sheet.crop((c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)))
        for r in range(rows)
        for c in range(cols)
    ]
