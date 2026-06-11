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

from ..types import AssetKind, AssetSpec, PlanDocument, StyleSpec
from .matting import key_to_alpha
from .textures import ensure_tileable

logger = logging.getLogger(__name__)


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
            raw = self.generator.generate(spec, plan.style, style_reference)
            self._save_raw(spec, raw)
            processed = self._post_process(spec, raw)
            path = self.asset_path(spec)
            processed.save(path)
            self._meta_path(spec).write_text(
                json.dumps({"hash": spec.content_hash(), "kind": spec.kind.value, "subject": spec.subject})
            )
            out[spec.id] = path
            if spec.kind == AssetKind.STYLE_BIBLE:
                style_reference = processed.convert("RGB")
        if progress:
            progress("done", total, total)
        return out

    def _post_process(self, spec: AssetSpec, img: Image.Image) -> Image.Image:
        if img.size != (spec.width_px, spec.height_px):
            img = img.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
        if spec.kind == AssetKind.TEXTURE:
            return ensure_tileable(img)
        if spec.kind in (AssetKind.SPRITE_SHEET, AssetKind.POI_SPRITE, AssetKind.ORNAMENT):
            # Key out the flat magenta background. Gemini returns opaque RGBA
            # PNGs (alpha channel present but all 255), so mode == "RGBA" does
            # NOT mean the background is already transparent -- only skip keying
            # when the image carries real (varying) transparency.
            if img.mode == "RGBA" and _has_real_transparency(img):
                return img
            return key_to_alpha(img)
        return img.convert("RGB")


def _has_real_transparency(img: Image.Image) -> bool:
    """True if the alpha channel actually varies (genuine cut-out), not a
    uniformly-opaque channel that merely happens to be present."""
    alpha = img.getchannel("A")
    lo, hi = alpha.getextrema()
    return lo < 250


def load_sprites_from_sheet(sheet_path: Path, grid: tuple[int, int]) -> list[Image.Image]:
    """Cut a sprite sheet into individual trimmed RGBA sprites."""
    from .matting import trim_to_content

    sheet = Image.open(sheet_path).convert("RGBA")
    cols, rows = grid
    cell_w, cell_h = sheet.width // cols, sheet.height // rows
    sprites = []
    for r in range(rows):
        for c in range(cols):
            cell = sheet.crop((c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h))
            sprites.append(trim_to_content(cell))
    return sprites
