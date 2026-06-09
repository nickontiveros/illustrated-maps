"""Frequency-split harmonization: AI mood at a frequency where it cannot
break anything.

The composed poster is downscaled to model resolution and repainted by an
image model for unified color and light. The repainted image is then
upscaled and ONLY its low-frequency component (a heavy Gaussian blur) is
blended over the native render: detail, linework and text stay exactly as
composed; global color grading comes from the AI. Misalignment between
the two images is invisible because only frequencies far below the
misalignment scale are taken from the AI image.

`strength` is a user dial; 0 disables the blend entirely.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np
from PIL import Image, ImageFilter

from ..types import StyleSpec

logger = logging.getLogger(__name__)

MODEL_MAX_DIM = 2048
# Blur radius as a fraction of poster width: ~40px at A1 scale.
LOW_FREQ_RADIUS_FRACTION = 0.006


class MoodPass(Protocol):
    """Anything that can repaint a small composite for global color/light."""

    def repaint(self, composite: Image.Image, style: StyleSpec) -> Image.Image: ...


class IdentityMoodPass:
    """No-op repaint (lets the blend math be tested in isolation)."""

    def repaint(self, composite: Image.Image, style: StyleSpec) -> Image.Image:
        return composite


class GeminiMoodPass:
    """Repaint via Gemini img2img at model resolution."""

    PROMPT = (
        "Repaint this illustrated map poster as a single unified hand-painted "
        "artwork: harmonize the colors, add soft global lighting and gentle "
        "color variation across the painting. {style}. "
        "CRITICAL: preserve every shape, road, label and building exactly "
        "where it is. Do not add, move or remove anything. Do not add text."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.1-flash-image-preview"):
        from mapgen.services.gemini_service import GeminiService

        self._service = GeminiService(api_key=api_key, model=model)

    def repaint(self, composite: Image.Image, style: StyleSpec) -> Image.Image:
        from google.genai import types as genai_types

        response = self._service.client.models.generate_content(
            model=self._service.model,
            contents=[composite, self.PROMPT.format(style=style.description)],
            config=genai_types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        return self._service._extract_image_from_response(response)


def low_frequency(img: Image.Image, radius: float) -> np.ndarray:
    blurred = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(blurred, dtype=np.float32)


def harmonize(
    poster: Image.Image,
    mood_pass: MoodPass,
    style: StyleSpec,
    strength: float = 0.5,
) -> Image.Image:
    """Blend the mood pass's low frequencies over the native render.

    result = native + strength * (lowfreq(mood) - lowfreq(native))

    High frequencies (detail, linework, text) are untouched by
    construction; at strength=0 the poster is returned unchanged.
    """
    if strength <= 0:
        return poster

    small = poster.copy()
    small.thumbnail((MODEL_MAX_DIM, MODEL_MAX_DIM), Image.Resampling.LANCZOS)
    logger.info("Harmonize: mood pass at %dx%d", small.width, small.height)
    repainted = mood_pass.repaint(small, style)
    if repainted.size != poster.size:
        repainted = repainted.resize(poster.size, Image.Resampling.LANCZOS)

    radius = max(8.0, poster.width * LOW_FREQ_RADIUS_FRACTION)
    native = np.asarray(poster.convert("RGB"), dtype=np.float32)
    delta = low_frequency(repainted, radius) - low_frequency(poster, radius)
    out = np.clip(native + strength * delta, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")
