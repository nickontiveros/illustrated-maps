"""Gemini AI image generation service.

Uses the Google GenAI SDK for image generation with Gemini 2.0 Flash
or newer models that support native image output.
"""

import base64
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


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

    # Hierarchical generation prompts
    HIERARCHICAL_PROMPTS = {
        "overview": (
            "Create a hand-illustrated tourist map of this entire region. "
            "Establish a unified artistic style with warm, muted colors. "
            "Follow the geography shown in the reference image exactly — every "
            "coastline, highway, river, large park, and urban area must match. "
            "\n\n"
            "Style guidelines:\n"
            "- Hand-painted illustration aesthetic with a warm, cohesive color palette\n"
            "- Roads: cream/beige paths clearly visible\n"
            "- Buildings/urban areas: warm tan/terracotta clusters\n"
            "- Water: soft blue-green tones\n"
            "- Parks/vegetation: muted sage green\n"
            "- Consistent line quality and texture throughout\n"
            "\n"
            "Show major features at this zoom level: coastlines, highways, rivers, "
            "large parks, urban texture, major landmarks. "
            "DO NOT include text labels, fantasy elements, or features not in the reference. "
            "Create clean edges suitable for subdivision."
        ),
        "enhance_medium": (
            "ILLUSTRATION REFERENCE — match ONLY the color palette and artistic "
            "style from this image. It has been upscaled from a lower-resolution "
            "overview, so road widths and feature sizes are exaggerated. "
            "Use the SATELLITE image below for correct road widths, building "
            "sizes, and geographic scale. This shows the SAME geographic area:\n"
        ),
        "enhance_medium_instruction": (
            "Enhance this map section with more geographic detail while preserving "
            "the established color palette and artistic style. Add secondary roads, "
            "building clusters, park interiors, and neighborhood texture. "
            "\n\n"
            "CRITICAL SCALE RULES:\n"
            "- Road widths, building sizes, and feature proportions MUST match "
            "the SATELLITE image, NOT the illustration reference.\n"
            "- The illustration reference is upscaled from a lower-resolution "
            "overview — its roads and features appear thicker than they should be.\n"
            "- Copy ONLY colors, textures, and artistic approach from the "
            "illustration. Copy ALL geographic scale from the satellite.\n"
            "\n"
            "DO NOT change the color palette or artistic approach. "
            "DO NOT include text labels or features not visible in the satellite image."
        ),
        "enhance_fine": (
            "ILLUSTRATION REFERENCE — match ONLY the color palette and artistic "
            "style from this image. It is from the previous generation pass, so "
            "use the SATELLITE image below for correct road widths, building "
            "sizes, and geographic scale. This shows the SAME geographic area:\n"
        ),
        "enhance_fine_instruction": (
            "Add fine detail to this map section while preserving the established "
            "color palette and artistic style. Add individual buildings, small "
            "streets, landscape features, and architectural detail. "
            "\n\n"
            "CRITICAL SCALE RULES:\n"
            "- Road widths, building sizes, and feature proportions MUST match "
            "the SATELLITE image, NOT the illustration reference.\n"
            "- Copy ONLY colors, textures, and artistic approach from the "
            "illustration. Copy ALL geographic scale from the satellite.\n"
            "\n"
            "DO NOT change the color palette or artistic approach. "
            "DO NOT include text labels or features not visible in the satellite image. "
            "Create clean edges suitable for seamless tiling."
        ),
    }

    # Terrain-specific prompt modifiers appended when terrain is detected
    TERRAIN_PROMPT_MODIFIERS = {
        "desert": (
            "\n\nDesert terrain style: Render the landscape with Sonoran Desert characteristics. "
            "Include saguaro cacti silhouettes, dry sandy washes, mesas and buttes on the horizon, "
            "red-brown rocky outcrops, sparse creosote bush and palo verde trees. "
            "Use sun-washed warm tones — sand, terracotta, sienna, light adobe. "
            "DO NOT include: snow, dense forest, lush green meadows, or heavy rainfall features."
        ),
        "mountain": (
            "\n\nMountain terrain style: Emphasize elevation changes with dramatic shading. "
            "Show rocky peaks, ridgelines, and canyon walls. Use shadow and highlight to convey "
            "three-dimensional relief. Include subtle contour-like shading on slopes."
        ),
        "flat": (
            "\n\nFlat terrain style: Use subtle color gradients and texture variation to convey "
            "the open expanse. Avoid dramatic relief shading. Show gentle terrain through "
            "color shifts rather than sharp elevation features."
        ),
    }

    @staticmethod
    def detect_terrain_modifier(terrain_description: str) -> str:
        """Detect the appropriate terrain prompt modifier from a terrain description.

        Args:
            terrain_description: Text description from TerrainService.get_terrain_description()

        Returns:
            The matching modifier string, or empty string if no match.
        """
        if not terrain_description:
            return ""

        desc_lower = terrain_description.lower()

        # Desert detection: keywords or low-elevation arid characteristics
        desert_keywords = ["desert", "sand", "arid", "dune"]
        if any(kw in desc_lower for kw in desert_keywords):
            return GeminiService.TERRAIN_PROMPT_MODIFIERS["desert"]

        # Mountain detection
        mountain_keywords = ["mountainous", "steep slopes", "significant elevation"]
        if any(kw in desc_lower for kw in mountain_keywords):
            return GeminiService.TERRAIN_PROMPT_MODIFIERS["mountain"]

        # Flat terrain detection
        flat_keywords = ["flat terrain", "relatively flat", "mostly level"]
        if any(kw in desc_lower for kw in flat_keywords):
            return GeminiService.TERRAIN_PROMPT_MODIFIERS["flat"]

        # Desert-range elevation heuristic: 300-600m with gentle terrain in arid zones
        import re
        elev_match = re.search(r'(\d+)m to (\d+)m', terrain_description)
        if elev_match:
            min_elev = int(elev_match.group(1))
            max_elev = int(elev_match.group(2))
            if 200 <= min_elev <= 800 and max_elev - min_elev < 500:
                if "hilly" in desc_lower or "gently" in desc_lower or "level" in desc_lower:
                    return GeminiService.TERRAIN_PROMPT_MODIFIERS["desert"]

        return ""

    # Model configurations
    MODELS = {
        "gemini-3.1-flash-image-preview": {
            "max_size": 2048,
            "supports_image_input": True,
            "supports_image_output": True,
        },
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
        model: str = "gemini-3.1-flash-image-preview",
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

        # Build the style/instruction portion of the prompt
        base_style = style_prompt or self.STYLE_PROMPTS["base_map"]

        instruction_parts = [base_style]

        if terrain_description:
            instruction_parts.append(f"\n\nTerrain characteristics: {terrain_description}")
            modifier = self.detect_terrain_modifier(terrain_description)
            if modifier:
                instruction_parts.append(modifier)

        if tile_position:
            instruction_parts.append(f"\n\nThis is the {tile_position} section of the map.")

        instructions = "".join(instruction_parts)

        # Resize image if needed (max 2048 for most models)
        max_size = self.MODELS.get(self.model, {}).get("max_size", 2048)
        if reference_image.width > max_size or reference_image.height > max_size:
            reference_image = reference_image.copy()
            reference_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if style_reference:
            if style_reference.width > max_size or style_reference.height > max_size:
                style_reference = style_reference.copy()
                style_reference.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        start_time = time.time()

        # Build contents list.
        # When using a style reference, interleave labels BEFORE each image so
        # the model knows each image's role before it processes the pixels.
        if style_reference:
            prompt = (
                "GEOGRAPHY SOURCE — illustrate ONLY the geography (roads, buildings, "
                "coastlines, parks, terrain) shown in this satellite/map image:"
            )
            contents = [
                prompt,
                reference_image,
                (
                    "STYLE REFERENCE — copy ONLY the artistic style (color palette, "
                    "line quality, hand-illustrated look) from this image. "
                    "IGNORE all geography, roads, and features in this image — "
                    "it is from a completely different location:"
                ),
                style_reference,
                (
                    "Now generate an illustrated map tile.\n"
                    "CRITICAL RULES:\n"
                    "1. Geography MUST come from the satellite image above. "
                    "Every road, intersection, park, and building must match the satellite.\n"
                    "2. Style MUST come from the style reference. Match its colors, line "
                    "quality, and painted aesthetic.\n"
                    "3. Do NOT reproduce any roads, layouts, or features from the style "
                    "reference — its geography is from a different place.\n\n"
                ) + instructions,
            ]
        else:
            prompt = instructions
            contents = [reference_image, prompt]

        # Log details of the Gemini call for debugging
        text_parts = [c for c in contents if isinstance(c, str)]
        image_parts = [c for c in contents if isinstance(c, Image.Image)]
        logger.info(
            "Gemini call: model=%s, images=%d [%s], text_parts=%d, prompt_len=%d",
            self.model,
            len(image_parts),
            ", ".join(f"{img.size[0]}x{img.size[1]}" for img in image_parts),
            len(text_parts),
            sum(len(t) for t in text_parts),
        )
        for i, part in enumerate(text_parts):
            logger.debug("Gemini text part %d:\n%s", i, part[:500])

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
        logger.info(
            "Gemini response: %dx%d in %.1fs",
            generated_image.width, generated_image.height, generation_time,
        )

        return GenerationResult(
            image=generated_image,
            prompt_used=prompt,
            model=self.model,
            generation_time=generation_time,
        )

    def generate_overview(
        self,
        reference_image: Image.Image,
        style_prompt: Optional[str] = None,
        terrain_description: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a low-res overview illustration of the full region.

        This establishes global style, color palette, and composition for
        hierarchical generation. The overview is later subdivided to guide
        higher-resolution enhancement passes.

        Args:
            reference_image: Full-region satellite + OSM composite.
            style_prompt: Custom overview prompt (uses default if None).
            terrain_description: Optional terrain description.

        Returns:
            GenerationResult with overview illustration.
        """
        from google.genai import types

        prompt = style_prompt or self.HIERARCHICAL_PROMPTS["overview"]

        instruction_parts = [prompt]
        if terrain_description:
            instruction_parts.append(f"\n\nTerrain characteristics: {terrain_description}")
            modifier = self.detect_terrain_modifier(terrain_description)
            if modifier:
                instruction_parts.append(modifier)

        instructions = "".join(instruction_parts)

        max_size = self.MODELS.get(self.model, {}).get("max_size", 2048)
        if reference_image.width > max_size or reference_image.height > max_size:
            reference_image = reference_image.copy()
            reference_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        start_time = time.time()

        contents = [reference_image, instructions]

        logger.info(
            "Gemini overview call: model=%s, image=%dx%d, prompt_len=%d",
            self.model,
            reference_image.width, reference_image.height,
            len(instructions),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generation_time = time.time() - start_time
        generated_image = self._extract_image_from_response(response)

        logger.info(
            "Gemini overview response: %dx%d in %.1fs",
            generated_image.width, generated_image.height, generation_time,
        )

        return GenerationResult(
            image=generated_image,
            prompt_used=instructions,
            model=self.model,
            generation_time=generation_time,
        )

    def generate_enhanced_tile(
        self,
        illustration_crop: Image.Image,
        reference_image: Image.Image,
        level: str = "medium",
        terrain_description: Optional[str] = None,
        tile_position: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an enhanced tile guided by a lower-res illustration crop.

        Both images show the SAME geographic area — the illustration crop at
        lower resolution (style guidance) and the reference at higher resolution
        (geographic detail). This eliminates the geography-vs-style confusion
        of the flat pipeline.

        Args:
            illustration_crop: Crop from previous level's illustration (style guide).
            reference_image: High-res satellite + OSM composite (geography).
            level: Enhancement level — "medium" (L1) or "fine" (L2).
            terrain_description: Optional terrain description.
            tile_position: Optional position info (e.g., "top-left corner").

        Returns:
            GenerationResult with enhanced tile.
        """
        from google.genai import types

        if level == "fine":
            label = self.HIERARCHICAL_PROMPTS["enhance_fine"]
            instruction = self.HIERARCHICAL_PROMPTS["enhance_fine_instruction"]
        else:
            label = self.HIERARCHICAL_PROMPTS["enhance_medium"]
            instruction = self.HIERARCHICAL_PROMPTS["enhance_medium_instruction"]

        instruction_parts = [instruction]
        if terrain_description:
            instruction_parts.append(f"\n\nTerrain characteristics: {terrain_description}")
            modifier = self.detect_terrain_modifier(terrain_description)
            if modifier:
                instruction_parts.append(modifier)
        if tile_position:
            instruction_parts.append(f"\n\nThis is the {tile_position} section of the map.")

        full_instruction = "".join(instruction_parts)

        max_size = self.MODELS.get(self.model, {}).get("max_size", 2048)
        if reference_image.width > max_size or reference_image.height > max_size:
            reference_image = reference_image.copy()
            reference_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Upscale illustration crop to match reference dimensions so Gemini
        # gets a 1:1 pixel comparison (same geography, same size, different
        # detail level).  Without this, small crops (e.g. 647×765) sent
        # alongside 2048×2048 references confuse the model about target scale.
        if (illustration_crop.width, illustration_crop.height) != (reference_image.width, reference_image.height):
            logger.info(
                "Upscaling illustration crop from %dx%d to %dx%d",
                illustration_crop.width, illustration_crop.height,
                reference_image.width, reference_image.height,
            )
            illustration_crop = illustration_crop.resize(
                (reference_image.width, reference_image.height),
                Image.Resampling.LANCZOS,
            )

        start_time = time.time()

        contents = [
            label,
            illustration_crop,
            (
                "GEOGRAPHY SOURCE — use this satellite/map image for roads, "
                "buildings, parks, water, and all geographic detail:"
            ),
            reference_image,
            full_instruction,
        ]

        text_parts = [c for c in contents if isinstance(c, str)]
        image_parts = [c for c in contents if isinstance(c, Image.Image)]
        logger.info(
            "Gemini enhance call (%s): model=%s, images=%d [%s], prompt_len=%d",
            level, self.model,
            len(image_parts),
            ", ".join(f"{img.size[0]}x{img.size[1]}" for img in image_parts),
            sum(len(t) for t in text_parts),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generation_time = time.time() - start_time
        generated_image = self._extract_image_from_response(response)

        logger.info(
            "Gemini enhance response (%s): %dx%d in %.1fs",
            level, generated_image.width, generated_image.height, generation_time,
        )

        return GenerationResult(
            image=generated_image,
            prompt_used=full_instruction,
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
