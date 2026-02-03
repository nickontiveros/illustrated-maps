"""Outpainting service for filling empty regions after perspective transform.

After applying perspective transform to create the trapezoid-shaped map,
there are empty regions that need to be filled:
1. Horizon band - full-width strip above the map
2. Left/right triangles - gaps from perspective convergence

This service uses a single-generation + upscale approach:
1. Fill empty regions with grey
2. Downscale entire image to fit Gemini limits
3. Single Gemini call to complete the image
4. Upscale with Real-ESRGAN
5. Merge upscaled outpaint with original high-res content
"""

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..models.project import BoundingBox
from .gemini_service import GeminiService


# Outpainting prompt - style-neutral, no "theme park" reference
OUTPAINT_PROMPT = """Complete this illustrated map by filling ONLY the grey empty regions.
Continue the existing style exactly - match colors, textures, brushwork, and level of detail.
Do NOT modify or add clouds/sky to any area that already has map content.

For the TOP EDGE ONLY: Create a thin horizon line where terrain fades into soft sky and clouds.
Sky and clouds should appear ONLY at the very top edge of the image, nowhere else.
Use atmospheric perspective - terrain becomes lighter and hazier as it approaches the horizon.

For the SIDE REGIONS: Continue the map content naturally (buildings, streets, terrain).
These areas should show more map content fading out toward the edges - NO sky or clouds here.

Only fill the grey areas. Preserve all existing map content exactly as it is.

IMPORTANT: Do not add any text, labels, words, letters, numbers, titles, captions, watermarks,
signatures, logos, symbols, or any written content of any kind."""


class OutpaintingService:
    """Service for outpainting empty regions using single-generation + upscale.

    This approach produces consistent results without seams by:
    1. Using a single Gemini call (sees full context)
    2. Upscaling with Real-ESRGAN (maintains quality)
    3. Merging only the empty regions back to original
    """

    def __init__(
        self,
        convergence: float = 0.7,
        vertical_scale: float = 0.4,
        horizon_margin: float = 0.15,
        max_gemini_size: int = 2048,
        fill_color: tuple[int, int, int] = (128, 128, 128),
        gemini_service: Optional[GeminiService] = None,
    ):
        """
        Initialize outpainting service.

        Args:
            convergence: Perspective convergence factor (0.7 = 70% width at top)
            vertical_scale: Vertical compression at top (0.4 = 40% height at top)
            horizon_margin: Horizon band height as fraction of content height
            max_gemini_size: Maximum dimension for Gemini input
            fill_color: RGB color for empty regions (grey by default)
            gemini_service: Optional pre-configured Gemini service
        """
        self.convergence = convergence
        self.vertical_scale = vertical_scale
        self.horizon_margin = horizon_margin
        self.max_gemini_size = max_gemini_size
        self.fill_color = fill_color

        # Lazy-initialize services
        self._gemini = gemini_service
        self._upsampler = None

    @property
    def gemini(self) -> GeminiService:
        """Lazy initialization of Gemini service."""
        if self._gemini is None:
            self._gemini = GeminiService()
        return self._gemini

    @property
    def upsampler(self):
        """Lazy initialization of Real-ESRGAN upsampler."""
        if self._upsampler is None:
            self._upsampler = self._init_upsampler()
        return self._upsampler

    def _init_upsampler(self):
        """Initialize Real-ESRGAN upsampler."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Use RealESRGAN_x4plus model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            # Find model weights
            weights_path = self._find_model_weights()

            upsampler = RealESRGANer(
                scale=4,
                model_path=weights_path,
                model=model,
                tile=400,  # Process in tiles to save memory
                tile_pad=10,
                pre_pad=0,
                half=False,  # Use FP32 for better quality
            )
            return upsampler
        except ImportError:
            print("Warning: Real-ESRGAN not installed. Using Lanczos upscaling.")
            print("Install with: pip install realesrgan basicsr")
            return None

    def _find_model_weights(self) -> str:
        """Find Real-ESRGAN model weights."""
        # Common locations to check
        possible_paths = [
            Path("weights/RealESRGAN_x4plus.pth"),
            Path.home() / ".cache/realesrgan/RealESRGAN_x4plus.pth",
            Path("/tmp/RealESRGAN_x4plus.pth"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Download if not found
        print("Downloading Real-ESRGAN weights...")
        import urllib.request

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        weights_dir = Path.home() / ".cache/realesrgan"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "RealESRGAN_x4plus.pth"
        urllib.request.urlretrieve(url, weights_path)
        return str(weights_path)

    def outpaint_image(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Image.Image:
        """
        Outpaint empty regions using single-generation + upscale approach.

        Args:
            image: Input perspective-transformed map image
            bbox: Geographic bounding box of the visible map
            output_path: Optional path to save result
            progress_callback: Optional callback for progress updates

        Returns:
            Image with outpainted regions
        """
        if progress_callback:
            progress_callback("Creating empty region mask...", 0.1)

        # 1. Create mask of empty regions
        mask = self.create_empty_region_mask(image.size)

        if progress_callback:
            progress_callback("Preparing image for generation...", 0.15)

        # 2. Fill empty regions with grey
        prepared = self.prepare_for_generation(image, mask)

        # 3. Calculate downscale factor
        scale_factor = self.calculate_scale_factor(image.size)

        if progress_callback:
            progress_callback(f"Downscaling image (factor: {scale_factor:.3f})...", 0.2)

        # 4. Downscale for Gemini
        downscaled = self.downscale(prepared, scale_factor)

        if progress_callback:
            progress_callback("Generating outpaint with Gemini...", 0.3)

        # 5. Single Gemini generation
        outpainted = self.generate_outpaint(downscaled)

        if progress_callback:
            progress_callback("Upscaling with Real-ESRGAN...", 0.6)

        # 6. Upscale back to original resolution
        upscale_factor = 1.0 / scale_factor
        upscaled = self.upscale(outpainted, upscale_factor, image.size)

        if progress_callback:
            progress_callback("Merging with original...", 0.9)

        # 7. Merge upscaled outpaint with original (only in empty regions)
        result = self.merge_with_original(image, upscaled, mask)

        # Save if path provided
        if output_path:
            result.save(output_path)

        if progress_callback:
            progress_callback("Complete!", 1.0)

        return result

    def create_empty_region_mask(
        self,
        size: tuple[int, int],
    ) -> np.ndarray:
        """
        Create a mask of empty regions based on perspective parameters.

        The mask is True where content needs to be filled (empty regions).

        Args:
            size: (width, height) of the image

        Returns:
            Boolean numpy array where True = empty region
        """
        width, height = size

        # Calculate horizon band height
        # horizon_margin is fraction of content height, content = height / (1 + horizon_margin)
        content_height = height / (1 + self.horizon_margin)
        horizon_height = int(height - content_height)

        # Calculate trapezoid inset based on convergence
        # At top of content (y = horizon_height), inset = width * (1 - convergence) / 2
        top_inset = (width * (1 - self.convergence)) / 2

        # Create mask
        mask = np.zeros((height, width), dtype=bool)

        # Fill horizon band (everything above content)
        mask[:horizon_height, :] = True

        # Fill triangles
        # For each row from horizon_height to bottom, calculate the trapezoid edges
        for y in range(horizon_height, height):
            # How far down from horizon (0 at horizon, 1 at bottom)
            progress = (y - horizon_height) / (height - horizon_height)

            # Inset decreases linearly as we go down
            current_inset = top_inset * (1 - progress)

            # Left triangle
            left_edge = int(current_inset)
            mask[y, :left_edge] = True

            # Right triangle
            right_edge = width - int(current_inset)
            mask[y, right_edge:] = True

        return mask

    def prepare_for_generation(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """
        Fill empty regions with grey for Gemini to complete.

        Args:
            image: Original image
            mask: Boolean mask where True = empty region

        Returns:
            Image with empty regions filled with grey
        """
        # Convert to numpy
        img_array = np.array(image)

        # Fill empty regions with grey
        fill_rgba = (*self.fill_color, 255)
        img_array[mask] = fill_rgba

        return Image.fromarray(img_array)

    def calculate_scale_factor(self, size: tuple[int, int]) -> float:
        """
        Calculate downscale factor to fit within Gemini's limits.

        Args:
            size: (width, height) of original image

        Returns:
            Scale factor (< 1.0 means downscale)
        """
        width, height = size
        max_dim = max(width, height)

        if max_dim <= self.max_gemini_size:
            return 1.0

        return self.max_gemini_size / max_dim

    def downscale(
        self,
        image: Image.Image,
        scale_factor: float,
    ) -> Image.Image:
        """
        Downscale image by the given factor.

        Args:
            image: Image to downscale
            scale_factor: Factor to scale by

        Returns:
            Downscaled image
        """
        if scale_factor >= 1.0:
            return image

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def generate_outpaint(self, image: Image.Image) -> Image.Image:
        """
        Generate outpainted content with a single Gemini call.

        Args:
            image: Downscaled image with grey empty regions

        Returns:
            Image with empty regions filled
        """
        from google.genai import types

        result = self.gemini.client.models.generate_content(
            model=self.gemini.model,
            contents=[image, OUTPAINT_PROMPT],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        generated = self.gemini._extract_image_from_response(result)

        # Ensure output matches input size
        if generated.size != image.size:
            generated = generated.resize(image.size, Image.Resampling.LANCZOS)

        return generated

    def upscale(
        self,
        image: Image.Image,
        factor: float,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.

        Args:
            image: Image to upscale
            factor: Upscale factor
            target_size: Target (width, height) for final output

        Returns:
            Upscaled image at target_size
        """
        if self.upsampler is None:
            # Fall back to Lanczos if Real-ESRGAN not available
            return image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert PIL to numpy BGR (Real-ESRGAN expects BGR)
        img_array = np.array(image.convert("RGB"))
        img_bgr = img_array[:, :, ::-1]

        # Upscale
        try:
            output, _ = self.upsampler.enhance(img_bgr, outscale=factor)
            # Convert back to RGB
            output_rgb = output[:, :, ::-1]
            result = Image.fromarray(output_rgb)
        except Exception as e:
            print(f"Real-ESRGAN failed: {e}. Falling back to Lanczos.")
            result = image.resize(target_size, Image.Resampling.LANCZOS)

        # Ensure exact target size
        if result.size != target_size:
            result = result.resize(target_size, Image.Resampling.LANCZOS)

        return result.convert("RGBA")

    def merge_with_original(
        self,
        original: Image.Image,
        outpainted: Image.Image,
        mask: np.ndarray,
        feather: int = 50,
    ) -> Image.Image:
        """
        Merge outpainted regions with original, using only empty regions from outpaint.

        Args:
            original: Original high-res image
            outpainted: Upscaled outpainted image
            mask: Boolean mask where True = empty region (use outpaint)
            feather: Feather radius for smooth blending at edges

        Returns:
            Merged image
        """
        # Ensure same size
        if outpainted.size != original.size:
            outpainted = outpainted.resize(original.size, Image.Resampling.LANCZOS)

        # Create feathered mask for smooth blending
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        if feather > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

        # Convert to arrays
        orig_array = np.array(original).astype(np.float32)
        out_array = np.array(outpainted).astype(np.float32)
        blend_mask = np.array(mask_img).astype(np.float32) / 255.0

        # Expand mask to 4 channels
        blend_mask = blend_mask[:, :, np.newaxis]

        # Blend: result = original * (1 - mask) + outpainted * mask
        result_array = orig_array * (1 - blend_mask) + out_array * blend_mask

        # Convert back to image
        result_array = result_array.clip(0, 255).astype(np.uint8)
        return Image.fromarray(result_array)

    def estimate_cost(self) -> dict:
        """
        Estimate generation cost.

        Returns:
            Cost breakdown dictionary
        """
        cost_per_image = 0.13

        return {
            "gemini_generation": 1,
            "gemini_cost": cost_per_image,
            "upscale_cost": 0.00,  # Local processing
            "total_cost": cost_per_image,
        }
