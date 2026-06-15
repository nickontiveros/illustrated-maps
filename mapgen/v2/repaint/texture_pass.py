"""Single-call texture pass: whole-base img2img + frequency split.

The default repaint mode, validated against the tiled spike's failures:
one generation over the entire below-labels base means there are no joints
to misalign and no windows to drift between, and whole-poster context
keeps the model faithful (the tiled spike's hallucinations happened on
context-starved windows). The model's output is then used only for what it
is reliable at: low/mid-frequency paint texture. Native high frequencies
(linework, edges) are kept by construction, so geometry stays pixel-exact.

The tiled engine (engine.py) remains for a future fine-tuned exact-infill
painter -- the only proven route to high-res tiled texture (Isometric NYC).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

import numpy as np
from PIL import Image, ImageFilter

from ..color import normalize_to_reference
from ..types import StyleSpec
from .verify import STRUCTURE_CORR_THRESHOLD, structure_correlation

logger = logging.getLogger(__name__)

MODEL_MAX_DIM = 2048
# Fraction of poster width below which texture comes from the AI repaint.
# ~28px at A1 print size -- the upscaled model output carries no real detail
# finer than that anyway, and native linework above it stays untouched.
TEXTURE_RADIUS_FRACTION = 0.004
MIN_TEXTURE_RADIUS = 4.0


class StructureRejection(RuntimeError):
    """The repaint abandoned the guide's content (hallucination guard)."""


class TexturePainter(Protocol):
    """Anything that can repaint the whole downscaled base for texture."""

    def repaint(
        self,
        base_small: Image.Image,
        style: StyleSpec,
        style_bible: Optional[Image.Image] = None,
    ) -> Image.Image: ...


class IdentityTexturePass:
    """No-op repaint: lets the blend/guard machinery be tested for free."""

    def repaint(self, base_small, style, style_bible=None):
        return base_small


class GeminiTexturePass:
    """One Gemini img2img call over the whole base (prompt validated live)."""

    PROMPT = (
        "Repaint this entire illustrated map poster as a single unified "
        "hand-painted gouache artwork with rich visible brushwork and paper "
        "texture, exactly in the style of the attached reference. CRITICAL: "
        "every road, building, coastline, water edge and color stays exactly "
        "where and what it is -- change only the paint texture. Never invent "
        "content. Output at the same size. No text."
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

    def repaint(self, base_small, style, style_bible=None):
        from google.genai import types as genai_types

        contents: list = []
        if style_bible is not None:
            contents.append("STYLE REFERENCE — copy only the palette, brushwork and mood:")
            contents.append(style_bible)
        contents.append(base_small)
        contents.append(self.PROMPT)

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
                logger.warning(
                    "Texture pass attempt %d failed (%s); retrying in %ds",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError("Texture pass generation failed") from last_error


def texture_repaint(
    base: Image.Image,
    painter: TexturePainter,
    style: StyleSpec,
    style_bible: Optional[Image.Image] = None,
    strength: float = 1.0,
    detail_radius: Optional[float] = None,
    water_mask: Optional[Image.Image] = None,
    raw_sink=None,
) -> Image.Image:
    """Blend the painter's low/mid frequencies over the native base.

    result = native + strength * (lowmid(repaint) - lowmid(native))

    where lowmid is a Gaussian at `detail_radius` (default scales with
    poster width). The repaint is Lab-normalized to the base (water
    excluded) first so palette drift can't ride in on the low band; the
    structure guard rejects a repaint that abandoned the content outright.
    `raw_sink(name, image)` optionally receives the raw model output so it
    can be cached beside other raw generations.
    """
    base = base.convert("RGB")
    if strength <= 0:
        return base

    small = base.copy()
    small.thumbnail((MODEL_MAX_DIM, MODEL_MAX_DIM), Image.Resampling.LANCZOS)
    logger.info("Texture pass: repainting at %dx%d", small.width, small.height)
    repainted = painter.repaint(small, style, style_bible).convert("RGB")
    if raw_sink is not None:
        raw_sink("texture_pass_raw", repainted)

    corr = structure_correlation(base, repainted)
    if corr < STRUCTURE_CORR_THRESHOLD:
        raise StructureRejection(
            f"texture pass rejected: structure correlation {corr:.2f} below "
            f"{STRUCTURE_CORR_THRESHOLD} (model did not follow the map content); "
            "re-run to retry"
        )

    if repainted.size != base.size:
        repainted = repainted.resize(base.size, Image.Resampling.LANCZOS)
    exclude = None
    if water_mask is not None:
        exclude = np.asarray(water_mask.convert("L")) > 127
    repainted = normalize_to_reference(repainted, base, exclude_mask=exclude)

    radius = detail_radius or max(MIN_TEXTURE_RADIUS, base.width * TEXTURE_RADIUS_FRACTION)
    native = np.asarray(base, dtype=np.float32)
    ai_low = np.asarray(repainted.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    native_low = np.asarray(base.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    out = np.clip(native + strength * (ai_low - native_low), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")
