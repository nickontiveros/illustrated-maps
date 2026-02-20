"""Gemini AI image generation service.

Uses the Google GenAI SDK for image generation with Gemini 2.0 Flash
or newer models that support native image output.
"""

import base64
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class GenerationResult:
    """Result from image generation."""

    image: Image.Image
    prompt_used: str
    model: str
    generation_time: float
    tokens_used: Optional[int] = None


class GeminiService:
    """Service for AI image generation using Google Gemini.

    Uses the newer google-genai SDK with native image generation support.
    See: https://ai.google.dev/gemini-api/docs/image-generation
    """

    # Default style prompts for different generation modes
    STYLE_PROMPTS = {
        "base_map": (
            "Transform this satellite and map reference into a hand illustrated tourist map. "
            "\n\n"
            "CRITICAL: You MUST follow the geography shown in the reference image exactly. "
            "Every road, coastline, water body, park, and building in your output must match "
            "the reference. Do not add, remove, or relocate any geographic features. "
            "\n\n"
            "Style guidelines:\n"
            "- Hand-painted illustration aesthetic with warm, muted colors\n"
            "- Roads: cream/beige paths clearly visible\n"
            "- Buildings: warm tan/terracotta with subtle shadows\n"
            "- Water: soft blue-green tones\n"
            "- Parks/vegetation: muted sage green\n"
            "- Maintain consistent color palette throughout\n"
            "\n\n"
            "DO NOT include: roller coasters, theme park rides, brick/cobblestone textures "
            "unless visible in reference, fantasy elements, text labels, or any features "
            "not present in the reference image.\n"
            "\n"
            "Create clean edges suitable for seamless tiling."
        ),
        "landmark": (
            "Transform this building photograph into an illustrated theme park map "
            "style, matching the vibrant hand-painted aesthetic of Disneyland maps. "
            "Create an isometric view of the building with simplified but recognizable "
            "architectural details. Use warm, saturated colors with subtle shadows. "
            "The illustration should look like it belongs on a theme park map, slightly "
            "exaggerated in scale and detail for visual appeal. Remove the background "
            "and create clean edges. Generate only the illustrated building, no text."
        ),
        "inpaint_seam": (
            "Seamlessly blend these two map sections together. Match the illustrated "
            "theme park map style with consistent colors, textures, and details across "
            "the seam. Ensure roads, paths, and features connect smoothly without any "
            "visible transition line. Generate only the blended image, no text."
        ),
    }

    # Model configurations
    MODELS = {
        "gemini-3-pro-image-preview": {
            "max_size": 2048,
            "supports_image_input": True,
            "supports_image_output": True,
        },
        "gemini-2.0-flash-exp": {
            "max_size": 2048,
            "supports_image_input": True,
            "supports_image_output": True,
        },
        "gemini-2.5-flash-preview-05-20": {
            "max_size": 2048,
            "supports_image_input": True,
            "supports_image_output": True,
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-image-preview",
    ):
        """
        Initialize Gemini service.

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model to use for generation (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy initialization of GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_base_tile(
        self,
        reference_image: Image.Image,
        style_prompt: Optional[str] = None,
        terrain_description: Optional[str] = None,
        tile_position: Optional[str] = None,
        style_reference: Optional[Image.Image] = None,
    ) -> GenerationResult:
        """
        Generate an illustrated map tile from a reference image.

        Args:
            reference_image: Base map render (satellite + OSM composite)
            style_prompt: Custom style prompt (uses default if None)
            terrain_description: Optional terrain description to include
            tile_position: Optional position info (e.g., "top-left corner")
            style_reference: Optional style reference image for visual consistency

        Returns:
            GenerationResult with generated image
        """
        from google.genai import types

        # Build prompt
        if style_reference:
            prompt = (
                "The first image is a satellite/map reference showing the geography to illustrate. "
                "The second image is a style reference - match its illustrated style, color palette, "
                "line quality, and artistic approach exactly. "
            )
            prompt += style_prompt or self.STYLE_PROMPTS["base_map"]
        else:
            prompt = style_prompt or self.STYLE_PROMPTS["base_map"]

        if terrain_description:
            prompt += f"\n\nTerrain characteristics: {terrain_description}"

        if tile_position:
            prompt += f"\n\nThis is the {tile_position} section of the map."

        # Enhanced road treatment hint
        if style_reference:
            prompt += (
                "\n\nIMPORTANT: Follow the road widths and positions shown in "
                "the reference image closely. Roads should be clearly visible "
                "paths with consistent width hierarchy."
            )

        # Resize image if needed (max 2048 for most models)
        max_size = self.MODELS.get(self.model, {}).get("max_size", 2048)
        if reference_image.width > max_size or reference_image.height > max_size:
            reference_image = reference_image.copy()
            reference_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Resize style reference to same max_size
        if style_reference:
            if style_reference.width > max_size or style_reference.height > max_size:
                style_reference = style_reference.copy()
                style_reference.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        start_time = time.time()

        # Build contents list
        if style_reference:
            contents = [reference_image, style_reference, prompt]
        else:
            contents = [reference_image, prompt]

        # Generate with image input using new SDK
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generation_time = time.time() - start_time

        # Extract image from response
        generated_image = self._extract_image_from_response(response)

        return GenerationResult(
            image=generated_image,
            prompt_used=prompt,
            model=self.model,
            generation_time=generation_time,
        )

    def stylize_landmark(
        self,
        photo: Image.Image,
        style_reference: Optional[Image.Image] = None,
        landmark_name: Optional[str] = None,
        style_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Transform a landmark photo into illustrated style.

        Args:
            photo: Landmark photograph
            style_reference: Optional reference image for style matching
            landmark_name: Name of the landmark for context
            style_prompt: Custom style prompt

        Returns:
            GenerationResult with illustrated landmark
        """
        from google.genai import types

        # Build prompt
        prompt = style_prompt or self.STYLE_PROMPTS["landmark"]

        if landmark_name:
            prompt += f"\n\nThis is the {landmark_name}."

        start_time = time.time()

        # If style reference provided, include it
        if style_reference:
            full_prompt = f"Transform the first image to match the illustrated style of the second image. {prompt}"
            response = self.client.models.generate_content(
                model=self.model,
                contents=[photo, style_reference, full_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[photo, prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

        generation_time = time.time() - start_time

        generated_image = self._extract_image_from_response(response)

        return GenerationResult(
            image=generated_image,
            prompt_used=prompt,
            model=self.model,
            generation_time=generation_time,
        )

    def inpaint_seam(
        self,
        seam_region: Image.Image,
        orientation: str = "horizontal",
    ) -> GenerationResult:
        """
        Inpaint a seam strip extracted from the assembled image.

        The strip is centered on the cell boundary and already shows the
        discontinuity that needs to be blended.

        Args:
            seam_region: Strip cropped from the assembled image around the seam
            orientation: "horizontal" or "vertical" seam

        Returns:
            GenerationResult with inpainted region
        """
        from google.genai import types

        prompt = self.STYLE_PROMPTS["inpaint_seam"]

        start_time = time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=[seam_region, prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generation_time = time.time() - start_time

        generated_image = self._extract_image_from_response(response)

        return GenerationResult(
            image=generated_image,
            prompt_used=prompt,
            model=self.model,
            generation_time=generation_time,
        )

    def generate_text_to_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> GenerationResult:
        """
        Generate an image from text prompt only.

        Args:
            prompt: Text description of desired image
            width: Output width
            height: Output height

        Returns:
            GenerationResult with generated image
        """
        from google.genai import types

        start_time = time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generation_time = time.time() - start_time

        generated_image = self._extract_image_from_response(response)

        # Resize if needed
        if generated_image.size != (width, height):
            generated_image = generated_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )

        return GenerationResult(
            image=generated_image,
            prompt_used=prompt,
            model=self.model,
            generation_time=generation_time,
        )

    def _extract_image_from_response(self, response) -> Image.Image:
        """Extract PIL Image from Gemini response.

        The new google-genai SDK returns image data in parts with inline_data.
        The data is already bytes, not base64 encoded.
        """
        # Handle new SDK response format
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data is not None:
                        # New SDK returns bytes directly
                        image_data = part.inline_data.data
                        if isinstance(image_data, bytes):
                            return Image.open(BytesIO(image_data)).convert("RGBA")
                        elif isinstance(image_data, str):
                            # Fall back to base64 decode if string
                            return Image.open(BytesIO(base64.b64decode(image_data))).convert("RGBA")

        # Try alternate response structures for older SDK compatibility
        if hasattr(response, "images") and response.images:
            image_data = response.images[0]
            if isinstance(image_data, bytes):
                return Image.open(BytesIO(image_data)).convert("RGBA")
            elif isinstance(image_data, str):
                return Image.open(BytesIO(base64.b64decode(image_data))).convert("RGBA")

        # Check if response has parts directly
        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    image_data = part.inline_data.data
                    if isinstance(image_data, bytes):
                        return Image.open(BytesIO(image_data)).convert("RGBA")

        raise ValueError(
            f"Could not extract image from Gemini response. "
            f"Response type: {type(response)}, "
            f"Has candidates: {hasattr(response, 'candidates')}"
        )

    def estimate_cost(self, num_tiles: int, num_landmarks: int) -> dict:
        """
        Estimate generation costs.

        Args:
            num_tiles: Number of base map tiles
            num_landmarks: Number of landmarks

        Returns:
            Cost estimate dictionary
        """
        # Approximate costs (these may change)
        cost_per_image = 0.13  # USD

        tile_cost = num_tiles * cost_per_image
        landmark_cost = num_landmarks * cost_per_image
        # Estimate seam repairs at ~20% of tiles
        seam_cost = int(num_tiles * 0.2) * cost_per_image

        return {
            "tile_cost": tile_cost,
            "landmark_cost": landmark_cost,
            "seam_cost": seam_cost,
            "total_cost": tile_cost + landmark_cost + seam_cost,
            "num_tiles": num_tiles,
            "num_landmarks": num_landmarks,
        }
