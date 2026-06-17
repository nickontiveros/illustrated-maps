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
    "isolated on a single flat solid magenta background: every pixel not covered "
    "by the subject must be pure magenta #FF00FF, edge to edge, with no other "
    "background of any kind (no parchment, no paper, no sky, no scene, no frame, "
    "no border), no shadows cast onto the background, no text anywhere; "
    "never use magenta, fuchsia, or hot pink in the subject itself"
)


def build_prompt(spec: AssetSpec, style: StyleSpec) -> str:
    style_clause = style.description
    if spec.kind == AssetKind.STYLE_BIBLE:
        return (
            f"A small corner of an imaginary illustrated tourist map: {style_clause}. "
            f"Show {style.scene}. "
            "This image defines the palette and brushwork for a whole map series. "
            "No text anywhere."
        )
    if spec.kind == AssetKind.TEXTURE:
        return (
            f"A flat painted surface swatch used to fill areas of an illustrated map: "
            f"{spec.prompt_hints or spec.subject}. Palette and brushwork of {style_clause}. "
            "This is an extreme close-up of one uniform painted wash -- only subtle "
            "brushstrokes and gentle tone variation across a single material. "
            "It is NOT a map and NOT a scene: absolutely no objects, no buildings, "
            "no roads, no rivers, no coastlines, no trees, no fields, no map features "
            "of any kind. Perfectly tileable: the left/right and top/bottom edges must "
            "wrap seamlessly. Even tone, no vignetting, no border, no text."
        )
    if spec.kind == AssetKind.SPRITE_SHEET:
        cols, rows = spec.sheet_grid or (3, 2)
        return (
            f"A sprite sheet of exactly {cols * rows} variations arranged in a strict "
            f"{cols}x{rows} grid, each cell containing one {spec.prompt_hints or spec.subject}, "
            f"{style_clause}, {CAMERA_CLAUSE}, each sprite {KEY_CLAUSE}. "
            "The background of the entire sheet, including every grid cell and the "
            "gaps between cells, must be pure magenta #FF00FF. "
            "Keep every sprite fully inside its grid cell with margin."
        )
    if spec.kind == AssetKind.POI_SPRITE:
        photo_clause = (
            "Use the attached photograph as the visual reference; keep the "
            "subject recognizable with its iconic silhouette. "
            if spec.source_photo
            else ""
        )
        if spec.prompt_hints:
            # Typed landmark (mountain, park, campus, ...): the hint says what
            # to actually paint; no building coercion.
            return (
                f"An illustrated landmark for a tourist map: {spec.subject}. {photo_clause}"
                f"Important: {spec.prompt_hints}. "
                f"{style_clause}, {CAMERA_CLAUSE}, slightly exaggerated charming "
                f"proportions, simplified but recognizable details. Draw the landmark "
                f"as ONE compact isolated vignette with nothing around it: no streets, "
                f"no sky, no surrounding map, no base or platform under it, no signage "
                f"text. The vignette must be {KEY_CLAUSE}."
            )
        return (
            f"An illustrated landmark for a tourist map: {spec.subject}. {photo_clause}"
            f"{style_clause}, {CAMERA_CLAUSE}, slightly exaggerated charming proportions, "
            f"simplified but recognizable architectural details. Draw exactly ONE "
            f"isolated structure with nothing around it: no streets, no water, no sky, "
            f"no ground plane, no neighboring buildings, no base or platform under it. "
            f"If the subject is a street, district, or area rather than a single "
            f"building, draw only its single most iconic building instead -- never "
            f"a scene, never a city block, never signage. The structure must be "
            f"{KEY_CLAUSE}."
        )
    if spec.kind == AssetKind.SHIELD:
        ref_clause = (
            "Use the attached image as the exact reference for the sign's shape, "
            "proportions, internal divisions and colors; reproduce it faithfully. "
            if spec.source_photo
            else ""
        )
        return (
            f"A highway route shield marker sign, painted in the style of {style_clause}. "
            f"{ref_clause}"
            "Flat-on view, as if the metal sign is facing the viewer. Render ONLY the "
            "blank sign itself -- its outline, background color and any inner border or "
            "banner -- as a hand-painted illustrated placard with gentle brushwork and "
            "a soft edge. CRITICAL: the sign must be completely BLANK -- absolutely no "
            "route number, letters, digits or text of any kind anywhere on it. "
            f"The shield must be {KEY_CLAUSE}."
        )
    # Ornament
    return (
        f"A decorative map ornament: {spec.subject}, {style_clause}, flat-on view. "
        "Draw ONLY the ornament itself: it is a small decorative element, not a map, "
        "not a scene, not a landscape, not a framed picture. If the ornament is a "
        "frame or cartouche, draw just the ornamental border with a completely empty "
        "center -- the background must show through the entire middle. "
        f"The ornament must be {KEY_CLAUSE}."
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
            # A missing/corrupt reference photo downgrades to a text-only
            # prompt -- it must never abort a paid batch.
            try:
                photo = Image.open(spec.source_photo).convert("RGB")
                photo.thumbnail((1024, 1024))
                contents.append("ARCHITECTURAL REFERENCE PHOTO:")
                contents.append(photo)
            except Exception as exc:
                logger.warning(
                    "Reference photo %r for %s unusable (%s); generating without it",
                    spec.source_photo,
                    spec.id,
                    exc,
                )
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
