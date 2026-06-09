"""Gemini-backed asset generator.

Builds asset prompts with the fixed camera/lighting/keying conventions
that make sprites composable (aerial-oblique view, NW sun, flat magenta
key background) and attaches the style-bible image to every call so all
assets share one visual language.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PIL import Image

from ..types import AssetKind, AssetSpec, StyleSpec

logger = logging.getLogger(__name__)

CAMERA_CLAUSE = (
    "aerial oblique view from the south at about 40 degrees, "
    "front facade and roof both visible, sunlight from the northwest"
)
KEY_CLAUSE = (
    "isolated on a perfectly flat, pure magenta background (#FF00FF), "
    "no shadows cast onto the background, no text anywhere"
)


def build_prompt(spec: AssetSpec, style: StyleSpec) -> str:
    style_clause = style.description
    if spec.kind == AssetKind.STYLE_BIBLE:
        return (
            f"A small corner of an imaginary illustrated tourist map: {style_clause}. "
            "Show a little coastline, a road, a few buildings and trees. "
            "This image defines the palette and brushwork for a whole map series. "
            "No text anywhere."
        )
    if spec.kind == AssetKind.TEXTURE:
        return (
            f"Seamless tileable texture, {style_clause}: {spec.prompt_hints or spec.subject}. "
            "Perfectly tileable: the left/right and top/bottom edges must wrap seamlessly. "
            "Even tone, no vignetting, no border, no text."
        )
    if spec.kind == AssetKind.SPRITE_SHEET:
        cols, rows = spec.sheet_grid or (3, 2)
        return (
            f"A sprite sheet of exactly {cols * rows} variations arranged in a strict "
            f"{cols}x{rows} grid, each cell containing one {spec.prompt_hints or spec.subject}, "
            f"{style_clause}, {CAMERA_CLAUSE}, each sprite {KEY_CLAUSE}. "
            "Keep every sprite fully inside its grid cell with margin."
        )
    if spec.kind == AssetKind.POI_SPRITE:
        photo_clause = (
            "Use the attached photograph as the architectural reference; keep the "
            "building recognizable with its iconic silhouette. "
            if spec.source_photo
            else ""
        )
        return (
            f"An illustrated landmark for a tourist map: {spec.subject}. {photo_clause}"
            f"{style_clause}, {CAMERA_CLAUSE}, slightly exaggerated charming proportions, "
            f"simplified but recognizable architectural details, {KEY_CLAUSE}."
        )
    # Ornament
    return (
        f"A decorative map ornament: {spec.subject}, {style_clause}, "
        f"flat-on view, {KEY_CLAUSE}."
    )


class GeminiAssetGenerator:
    """AssetGenerator backed by the Gemini image API (via V1's service)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.1-flash-image-preview", max_retries: int = 3):
        from mapgen.services.gemini_service import GeminiService

        self._service = GeminiService(api_key=api_key, model=model)
        self.max_retries = max_retries

    def generate(
        self,
        spec: AssetSpec,
        style: StyleSpec,
        style_reference: Optional[Image.Image] = None,
    ) -> Image.Image:
        from google.genai import types as genai_types

        prompt = build_prompt(spec, style)
        contents: list = []
        if style_reference is not None and spec.kind != AssetKind.STYLE_BIBLE:
            contents.append("STYLE REFERENCE — copy only the palette, brushwork and mood:")
            contents.append(style_reference)
        if spec.source_photo:
            photo = Image.open(spec.source_photo).convert("RGB")
            photo.thumbnail((1024, 1024))
            contents.append("ARCHITECTURAL REFERENCE PHOTO:")
            contents.append(photo)
        contents.append(prompt)

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
                logger.warning("Asset %s attempt %d failed (%s); retrying in %ds", spec.id, attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"Asset generation failed for {spec.id}") from last_error
