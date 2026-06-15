"""Repaint engine: executes a plan over the guide render.

Loop: plan -> build template -> paint -> extract selection -> per-window
color normalization -> commit to the working canvas and store. Every raw
model response is cached so post-processing changes never cost another
API call (same policy as the asset studio), and `max_calls` is a hard
budget: when it runs out the engine stops cleanly and the store makes the
next run resume where this one stopped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
from PIL import Image

from ..types import StyleSpec, hex_to_rgb
from .color_norm import normalize_selection
from .grid import QuadrantGrid, QuadStatus, Selection
from .planner import plan_order
from .store import RepaintStore
from .template import build_template, extract_selection, selection_px_box
from .verify import STRUCTURE_CORR_THRESHOLD, structure_correlation

logger = logging.getLogger(__name__)

# A quadrant whose *blurred* guide pixels vary less than this (std over RGB)
# is featureless -- blank paper, haze, padding, or plain textured ground
# (open desert, open water): skipped, never painted, never used as seam
# context. Blurring first ignores brush-texture noise while preserving real
# structure (a thin road or shoreline survives the blur), so big textured-
# but-empty regions don't burn paint calls and can't invite invention.
SKIP_STD_THRESHOLD = 3.5
SKIP_BLUR_RADIUS = 5.0

# When the guide region is essentially featureless, structure correlation is
# meaningless; instead reject paints that introduce strong structure of
# their own (invented mesas/lakes on a flat desert window).
FLAT_GUIDE_STD = 6.0
INVENTION_STD_FACTOR = 2.5


class RepaintPainter(Protocol):
    """Anything that can repaint a 1024px template window."""

    def paint(
        self,
        template: Image.Image,
        style: StyleSpec,
        style_bible: Optional[Image.Image] = None,
    ) -> Image.Image: ...


class IdentityPainter:
    """Returns the window unchanged; proves the windowing/stitch machinery
    adds zero artifacts (the keystone round-trip test) and powers dry runs."""

    def paint(self, template, style, style_bible=None):
        return template


class GeminiPainter:
    """Zero-shot Gemini img2img against the red-boundary convention."""

    PROMPT = (
        "This image is a window cut from a large hand-illustrated map poster. "
        "Repaint the region enclosed by the red boundary lines (where a side has "
        "no red line, the region extends to the image edge; if there are no red "
        "lines at all, repaint the entire image). Enrich that region with "
        "hand-painted gouache texture and visible brushwork in exactly the style "
        "of the attached style reference and of the painting surrounding the "
        "region: {style}. CRITICAL: keep all content exactly in place -- every "
        "road, building, coastline, water edge and color stays exactly where and "
        "what it is; change only the paint texture. NEVER invent content: no new "
        "buildings, islands, terrain, boats, people, animals, clouds, or scenery "
        "that is not in the input. If part of the region is blank or a flat "
        "color, repaint it as the same flat surface with only subtle paper and "
        "brush texture. Remove the red boundary lines. Return everything outside "
        "the region pixel-identical. Output the full window at the same size. "
        "No text anywhere."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.1-flash-image-preview",
        max_retries: int = 3,
    ):
        from mapgen.services.gemini_service import GeminiService

        self._service = GeminiService(api_key=api_key, model=model)
        self.max_retries = max_retries

    def paint(self, template, style, style_bible=None):
        from google.genai import types as genai_types

        contents: list = []
        if style_bible is not None:
            contents.append("STYLE REFERENCE — copy only the palette, brushwork and mood:")
            contents.append(style_bible)
        contents.append(template)
        contents.append(self.PROMPT.format(style=style.description))

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._service.client.models.generate_content(
                    model=self._service.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                return self._service._extract_image_from_response(response)
            except Exception as exc:  # network/API errors: retry with backoff
                last_error = exc
                wait = 2**attempt
                logger.warning("Repaint attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError("Repaint window generation failed") from last_error


@dataclass
class RepaintResult:
    image: Image.Image  # stitched canvas at guide size (unpadded)
    completed: bool  # False when max_calls stopped the run early
    calls_made: int
    calls_planned: int


class RepaintEngine:
    def __init__(
        self,
        painter: RepaintPainter,
        store: RepaintStore,
        style: StyleSpec,
        style_bible: Optional[Image.Image] = None,
        water_mask: Optional[Image.Image] = None,
        raw_dir: Optional[Path] = None,
        normalize_strength: float = 1.0,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.painter = painter
        self.store = store
        self.style = style
        self.style_bible = style_bible
        self.water_mask = water_mask
        self.raw_dir = raw_dir
        self.normalize_strength = normalize_strength
        self.progress = progress

    def plan(self, guide: Image.Image) -> tuple[QuadrantGrid, list[Selection]]:
        """The selections a run would paint, given current store state."""
        grid = QuadrantGrid(guide.width, guide.height)
        padded = self._pad(guide, grid)
        painted = self.store.cells_with_status(QuadStatus.GENERATED)
        skipped = self._skip_cells(padded, grid, painted)
        return grid, plan_order(grid, painted, skipped)

    def run(self, guide: Image.Image, max_calls: Optional[int] = None) -> RepaintResult:
        grid, selections = self.plan(guide)
        padded_guide = self._pad(guide, grid)
        working = padded_guide.copy()
        padded_water = None
        if self.water_mask is not None:
            padded_water = Image.new("L", grid.padded_size, 0)
            padded_water.paste(self.water_mask.convert("L"), (0, 0))

        painted = self.store.cells_with_status(QuadStatus.GENERATED)
        for cell in painted:  # resume: restore previously painted pixels
            quad = self.store.load_quadrant(cell)
            if quad is not None:
                working.paste(quad, self.store_box(grid, cell)[:2])

        calls = 0
        completed = True
        for i, sel in enumerate(selections):
            if max_calls is not None and calls >= max_calls:
                logger.info("Repaint budget of %d calls reached; stopping (resumable)", max_calls)
                completed = False
                break
            if self.progress:
                self.progress(i, len(selections), str(sel))

            template, window_box, sel_box = build_template(working, sel, painted, grid)
            result = self.painter.paint(template, self.style, self.style_bible)
            calls += 1
            self._save_raw(sel, template, result)

            extracted = extract_selection(result, sel_box)
            abs_box = selection_px_box(sel, grid)
            guide_region = padded_guide.crop(abs_box)
            water_region = padded_water.crop(abs_box) if padded_water else None

            # Structure guard: a repaint that abandoned the guide's content
            # (hallucinated scenery on a featureless window) is rejected --
            # guide pixels stay, cells are flagged for review/redo.
            corr = structure_correlation(guide_region, extracted)
            if corr < STRUCTURE_CORR_THRESHOLD:
                logger.warning(
                    "Repaint of %s rejected (structure correlation %.2f); keeping guide",
                    sel, corr,
                )
                for cell in sel.cells():
                    self.store.set_status(
                        cell, QuadStatus.FLAGGED, notes=f"structure mismatch ({corr:.2f})"
                    )
                self.store.record_call(sel)
                continue

            # Invention guard: correlation can't protect a near-featureless
            # guide (there is no structure to correlate). If the paint
            # introduced strong structure onto a flat window, it invented
            # scenery -- keep the guide pixels and move on (GENERATED, not
            # FLAGGED: a redo would just invent again).
            guide_std = float(np.asarray(guide_region.convert("L"), dtype=np.float32).std())
            painted_std = float(np.asarray(extracted.convert("L"), dtype=np.float32).std())
            if guide_std < FLAT_GUIDE_STD and painted_std > guide_std * INVENTION_STD_FACTOR + 4.0:
                logger.warning(
                    "Repaint of %s invented structure on a flat window "
                    "(guide std %.1f -> painted std %.1f); keeping guide",
                    sel, guide_std, painted_std,
                )
                working.paste(guide_region, abs_box[:2])
                for cell in sel.cells():
                    self.store.save_quadrant(cell, working.crop(grid.cell_box(cell)))
                    painted.add(cell)
                    self.store.set_status(
                        cell,
                        QuadStatus.GENERATED,
                        notes=f"kept guide (invention guard {guide_std:.1f}->{painted_std:.1f})",
                    )
                self.store.record_call(sel)
                continue

            normalized = normalize_selection(
                extracted, guide_region, water_region, strength=self.normalize_strength
            )

            working.paste(normalized, abs_box[:2])
            for cell in sel.cells():
                self.store.save_quadrant(cell, working.crop(grid.cell_box(cell)))
                painted.add(cell)
            self.store.record_call(sel)

        if self.progress:
            self.progress(len(selections), len(selections), "stitching")
        image = working.crop((0, 0, guide.width, guide.height))
        return RepaintResult(image, completed, calls, len(selections))

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def store_box(grid: QuadrantGrid, cell) -> tuple[int, int, int, int]:
        return grid.cell_box(cell)

    def _pad(self, guide: Image.Image, grid: QuadrantGrid) -> Image.Image:
        pad_w, pad_h = grid.padded_size
        guide = guide.convert("RGB")
        if (guide.width, guide.height) == (pad_w, pad_h):
            return guide
        # Pad with paper, not the edge pixel: the spike showed a large flat
        # non-paper area (the plate's land color) reads as a blank canvas to
        # the model and invites invented scenery; paper reads as margin.
        paper = hex_to_rgb(self.style.palette.get("paper", "#f3e9d4"))
        padded = Image.new("RGB", (pad_w, pad_h), paper)
        padded.paste(guide, (0, 0))
        return padded

    def _skip_cells(self, padded_guide: Image.Image, grid: QuadrantGrid, painted) -> set:
        from PIL import ImageFilter

        skipped = set()
        blur = ImageFilter.GaussianBlur(SKIP_BLUR_RADIUS)
        for cell in grid.all_cells():
            if cell in painted:
                continue
            # Per-channel std of the blurred cell: texture noise is gone,
            # real features (roads, shorelines, sprites) survive. Each cell
            # is blurred in isolation so a structured neighbor can't bleed
            # over the boundary and keep a blank cell alive.
            crop = padded_guide.crop(grid.cell_box(cell)).filter(blur)
            cell_std = np.asarray(crop, dtype=np.float32).reshape(-1, 3).std(axis=0).max()
            if cell_std < SKIP_STD_THRESHOLD:
                skipped.add(cell)
                self.store.set_status(cell, QuadStatus.SKIPPED)
        return skipped

    def _save_raw(self, sel: Selection, template: Image.Image, result: Image.Image) -> None:
        if self.raw_dir is None:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{sel.x:03d}_{sel.y:03d}_{sel.w}x{sel.h}"
        template.save(self.raw_dir / f"{stem}_template.png")
        result.convert("RGB").save(self.raw_dir / f"{stem}_result.png")
