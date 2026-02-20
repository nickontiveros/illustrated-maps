"""Atmospheric perspective and terrain enhancement service.

Adds depth cues via vertical gradient (contrast reduction, saturation loss,
blue haze toward top/distance) and improves terrain drama via DEM vertical
exaggeration.
"""

import math
from typing import Optional

import numpy as np
from PIL import Image

from ..models.atmosphere import AtmosphereSettings


class AtmosphereService:
    """Service for applying atmospheric perspective effects to maps."""

    def __init__(self, settings: Optional[AtmosphereSettings] = None):
        """Initialize atmosphere service.

        Args:
            settings: Atmosphere configuration. Uses defaults if None.
        """
        self.settings = settings or AtmosphereSettings()

    def apply_atmosphere(
        self,
        image: Image.Image,
        perspective_params: Optional[dict] = None,
    ) -> Image.Image:
        """Apply atmospheric perspective gradient to a map image.

        Creates depth cues by progressively reducing contrast, desaturating,
        and adding blue haze from bottom (near) to top (far). The gradient
        accounts for perspective compression if parameters are provided.

        Args:
            image: Map image to process.
            perspective_params: Optional dict with 'convergence', 'vertical_scale',
                'horizon_margin' for non-linear gradient mapping.

        Returns:
            New image with atmospheric perspective applied.
        """
        if not self.settings.enabled:
            return image

        has_alpha = image.mode == "RGBA"
        alpha_channel = None

        if has_alpha:
            img_array = np.array(image, dtype=np.float64)
            alpha_channel = img_array[:, :, 3].copy()
            rgb = img_array[:, :, :3]
        else:
            rgb = np.array(image.convert("RGB"), dtype=np.float64)

        h, w, _ = rgb.shape

        # Parse haze color
        haze_rgb = self._hex_to_rgb(self.settings.haze_color)

        # Build per-row atmosphere strength gradient
        gradient = self._build_gradient(h, perspective_params)

        # Apply effects per-row
        result = rgb.copy()

        for y in range(h):
            strength = gradient[y]

            if strength < 0.001:
                continue

            row = result[y]

            # 1. Reduce contrast (push toward mean)
            if self.settings.contrast_reduction > 0:
                mean_val = np.mean(row, axis=1, keepdims=True)
                contrast_factor = 1.0 - (self.settings.contrast_reduction * strength)
                row = mean_val + (row - mean_val) * contrast_factor

            # 2. Reduce saturation (push toward grayscale)
            if self.settings.saturation_reduction > 0:
                gray = np.mean(row, axis=1, keepdims=True)
                sat_factor = 1.0 - (self.settings.saturation_reduction * strength)
                row = gray + (row - gray) * sat_factor

            # 3. Add haze (blend toward haze color)
            if self.settings.haze_strength > 0:
                haze_amount = self.settings.haze_strength * strength
                haze_array = np.array(haze_rgb, dtype=np.float64)
                row = row * (1.0 - haze_amount) + haze_array * haze_amount

            result[y] = row

        result = np.clip(result, 0, 255).astype(np.uint8)

        if has_alpha and alpha_channel is not None:
            result_rgba = np.dstack([result, alpha_channel.astype(np.uint8)])
            return Image.fromarray(result_rgba, "RGBA")

        return Image.fromarray(result, "RGB")

    def _build_gradient(
        self,
        height: int,
        perspective_params: Optional[dict] = None,
    ) -> np.ndarray:
        """Build per-row atmosphere strength gradient.

        Strength is 0.0 at the bottom (near) and increases toward the top (far).
        The gradient curve exponent controls how quickly it ramps up.

        When perspective params are provided, the gradient accounts for the
        non-linear compression at the top of the image (more atmosphere effect
        per pixel in the compressed distance region).

        Args:
            height: Image height in pixels.
            perspective_params: Optional perspective parameters.

        Returns:
            1D array of shape (height,) with values 0.0-1.0.
        """
        # Linear normalized position: 0 = bottom, 1 = top
        y_positions = np.linspace(0, 1, height)
        # Flip: row 0 is top (far), row height-1 is bottom (near)
        distance_factor = 1.0 - y_positions  # 1.0 at top, 0.0 at bottom

        # Account for perspective horizon margin
        if perspective_params:
            horizon_margin = perspective_params.get("horizon_margin", 0.15)
            # The top portion (horizon margin) is sky/horizon - full atmosphere
            horizon_rows = int(height * horizon_margin / (1 + horizon_margin))
            if horizon_rows > 0:
                distance_factor[:horizon_rows] = 1.0

        # Apply curve exponent
        gradient = np.power(distance_factor, self.settings.gradient_curve)

        return gradient

    def generate_fog_layer(
        self,
        image_size: tuple[int, int],
        perspective_params: Optional[dict] = None,
    ) -> Image.Image:
        """Generate a standalone fog/haze overlay layer.

        Creates a transparent RGBA image with the haze color at varying
        opacity, suitable for compositing onto a map.

        Args:
            image_size: (width, height) of the output.
            perspective_params: Optional perspective parameters.

        Returns:
            RGBA Image with fog layer.
        """
        w, h = image_size
        haze_rgb = self._hex_to_rgb(self.settings.haze_color)

        gradient = self._build_gradient(h, perspective_params)

        # Create RGBA fog layer
        fog = np.zeros((h, w, 4), dtype=np.uint8)
        fog[:, :, 0] = haze_rgb[0]
        fog[:, :, 1] = haze_rgb[1]
        fog[:, :, 2] = haze_rgb[2]

        # Alpha varies by row
        for y in range(h):
            alpha = int(gradient[y] * self.settings.haze_strength * 255)
            fog[y, :, 3] = min(255, alpha)

        return Image.fromarray(fog, "RGBA")

    def compute_depth_map(
        self,
        image_size: tuple[int, int],
        perspective_params: Optional[dict] = None,
    ) -> np.ndarray:
        """Compute a depth map for prompt guidance.

        Returns a normalized depth map where 0.0 = near (bottom) and
        1.0 = far (top), accounting for perspective compression.

        Args:
            image_size: (width, height).
            perspective_params: Optional perspective parameters.

        Returns:
            2D array of shape (height, width) with values 0.0-1.0.
        """
        w, h = image_size
        gradient = self._build_gradient(h, perspective_params)

        # Expand to 2D
        depth_map = np.tile(gradient[:, np.newaxis], (1, w))
        return depth_map

    def get_tile_depth_hint(
        self,
        tile_row: int,
        total_rows: int,
    ) -> str:
        """Get a depth zone description for a tile's position.

        Used to add depth hints to Gemini prompts for individual tiles.

        Args:
            tile_row: Row index of the tile (0 = top).
            total_rows: Total number of tile rows.

        Returns:
            Depth zone description string.
        """
        if total_rows <= 1:
            return "middle distance zone"

        position = tile_row / (total_rows - 1)  # 0.0 = top, 1.0 = bottom

        if position < 0.33:
            return "far distance zone (reduce contrast, add slight blue haze, softer details)"
        elif position < 0.67:
            return "middle distance zone (moderate detail and contrast)"
        else:
            return "near/foreground zone (full contrast, sharp details, warm colors)"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )


def compute_enhanced_hillshade(
    dem: np.ndarray,
    resolution: tuple[float, float],
    vertical_exaggeration: float = 1.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    nodata_value: float = -9999,
    warm_color: tuple[int, int, int] = (255, 243, 224),
    cool_color: tuple[int, int, int] = (200, 215, 235),
) -> Image.Image:
    """Compute colorized hillshade with warm/cool directional lighting.

    Warm tones for sun-facing slopes, cool tones for shadow-facing slopes.
    This creates more dramatic terrain visualization than grayscale hillshade.

    Args:
        dem: Digital Elevation Model array.
        resolution: (x_res, y_res) in meters per pixel.
        vertical_exaggeration: Multiply elevations before gradient computation.
        azimuth: Light source azimuth in degrees.
        altitude: Light source altitude in degrees.
        nodata_value: Value used for missing data.
        warm_color: RGB color for sun-lit slopes.
        cool_color: RGB color for shadowed slopes.

    Returns:
        RGBA PIL Image with colorized hillshade.
    """
    res_x, res_y = resolution

    # Apply vertical exaggeration
    dem_exag = dem.copy().astype(np.float64)
    valid_mask = dem_exag != nodata_value
    if valid_mask.any():
        mean_elev = np.mean(dem_exag[valid_mask])
        dem_exag[~valid_mask] = mean_elev
        dem_exag[valid_mask] = (dem_exag[valid_mask] - mean_elev) * vertical_exaggeration + mean_elev
    else:
        return Image.new("RGBA", (dem.shape[1], dem.shape[0]), (0, 0, 0, 0))

    # Calculate gradients
    dx = np.gradient(dem_exag, res_x, axis=1)
    dy = np.gradient(dem_exag, res_y, axis=0)

    # Calculate slope and aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    azimuth_rad = math.radians(azimuth)
    altitude_rad = math.radians(altitude)

    # Standard hillshade
    hillshade = (
        np.sin(altitude_rad) * np.cos(slope)
        + np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )
    hillshade = np.clip(hillshade, 0, 1)

    # Directional component (positive = sun-facing, negative = shadow-facing)
    directional = np.cos(azimuth_rad - aspect) * np.sin(slope)
    directional = np.clip(directional, -1, 1)

    # Build colorized output
    h_height, h_width = dem.shape
    result = np.zeros((h_height, h_width, 4), dtype=np.uint8)

    warm = np.array(warm_color, dtype=np.float64)
    cool = np.array(cool_color, dtype=np.float64)

    # Blend warm/cool based on directional lighting
    # Positive directional = warm (sun-facing)
    # Negative directional = cool (shadow-facing)
    sun_factor = (directional + 1.0) / 2.0  # 0.0 = full shadow, 1.0 = full sun

    for c in range(3):
        color_value = cool[c] + (warm[c] - cool[c]) * sun_factor
        result[:, :, c] = np.clip(color_value * hillshade, 0, 255).astype(np.uint8)

    # Alpha based on slope magnitude (flat areas are transparent)
    slope_norm = np.clip(slope / (math.pi / 4), 0, 1)  # Normalize to 0-1
    result[:, :, 3] = (slope_norm * 180).astype(np.uint8)  # Max ~70% opacity

    # Set nodata to transparent
    result[~valid_mask, 3] = 0

    return Image.fromarray(result, "RGBA")
