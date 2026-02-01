"""Tile blending service for seamless map stitching."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class TileInfo:
    """Information about a generated tile."""

    image: Image.Image
    col: int  # Column index
    row: int  # Row index
    x_offset: int  # X offset in final image
    y_offset: int  # Y offset in final image
    # Grid bounds for determining edge tiles
    max_col: int = 0
    max_row: int = 0


class BlendingService:
    """Service for seamlessly blending map tiles."""

    def __init__(self, num_pyramid_levels: int = 5):
        """
        Initialize blending service.

        Args:
            num_pyramid_levels: Number of levels for Laplacian pyramid blending
        """
        self.num_pyramid_levels = num_pyramid_levels

    def blend_tiles(
        self,
        tiles: list[TileInfo],
        output_size: tuple[int, int],
        overlap: int = 256,
    ) -> Image.Image:
        """
        Blend multiple tiles into a single seamless image.

        Args:
            tiles: List of TileInfo objects
            output_size: (width, height) of output image
            overlap: Overlap between adjacent tiles in pixels

        Returns:
            Blended PIL Image
        """
        width, height = output_size

        # Determine grid bounds
        max_col = max(t.col for t in tiles)
        max_row = max(t.row for t in tiles)

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        for tile in tiles:
            tile_array = np.array(tile.image).astype(np.float32) / 255.0

            # Determine which edges need feathering (only internal edges)
            tile_height, tile_width = tile_array.shape[:2]
            feather_left = tile.col > 0
            feather_right = tile.col < max_col
            feather_top = tile.row > 0
            feather_bottom = tile.row < max_row

            # Create weight mask with selective edge feathering
            weight = self._create_weight_mask_selective(
                tile_width, tile_height, overlap,
                feather_left, feather_right, feather_top, feather_bottom,
            )

            # Calculate position in output
            y1 = tile.y_offset
            y2 = min(y1 + tile_height, height)
            x1 = tile.x_offset
            x2 = min(x1 + tile_width, width)

            # Calculate which portion of the tile to use
            ty1 = 0
            tx1 = 0
            ty2 = y2 - y1
            tx2 = x2 - x1

            # Add weighted tile to output
            tile_slice = tile_array[ty1:ty2, tx1:tx2]
            weight_slice = weight[ty1:ty2, tx1:tx2]

            for c in range(4):
                output[y1:y2, x1:x2, c] += tile_slice[:, :, c] * weight_slice
            weight_sum[y1:y2, x1:x2] += weight_slice

        # Normalize by weight sum
        mask = weight_sum > 0
        for c in range(4):
            output[:, :, c][mask] /= weight_sum[mask]

        # Convert back to uint8
        output = (output * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(output, mode="RGBA")

    def _create_weight_mask(
        self,
        width: int,
        height: int,
        overlap: int,
    ) -> np.ndarray:
        """
        Create a weight mask with feathered edges for blending.

        Args:
            width: Tile width
            height: Tile height
            overlap: Overlap size

        Returns:
            2D weight array
        """
        return self._create_weight_mask_selective(
            width, height, overlap,
            feather_left=True, feather_right=True,
            feather_top=True, feather_bottom=True,
        )

    def _create_weight_mask_selective(
        self,
        width: int,
        height: int,
        overlap: int,
        feather_left: bool,
        feather_right: bool,
        feather_top: bool,
        feather_bottom: bool,
    ) -> np.ndarray:
        """
        Create a weight mask with selectively feathered edges.

        Only feathers edges that will overlap with adjacent tiles.
        Edge tiles (at borders) keep sharp outer edges.

        Args:
            width: Tile width
            height: Tile height
            overlap: Overlap size
            feather_left: Whether to feather left edge
            feather_right: Whether to feather right edge
            feather_top: Whether to feather top edge
            feather_bottom: Whether to feather bottom edge

        Returns:
            2D weight array
        """
        # Create ramps for edges
        x_ramp = np.ones(width, dtype=np.float32)
        y_ramp = np.ones(height, dtype=np.float32)

        if overlap > 0:
            # Only feather edges that have adjacent tiles
            if feather_left:
                x_ramp[:overlap] = np.linspace(0, 1, overlap)
            if feather_right:
                x_ramp[-overlap:] = np.linspace(1, 0, overlap)
            if feather_top:
                y_ramp[:overlap] = np.linspace(0, 1, overlap)
            if feather_bottom:
                y_ramp[-overlap:] = np.linspace(1, 0, overlap)

        # Create 2D mask
        mask = np.outer(y_ramp, x_ramp)

        return mask

    def multiband_blend(
        self,
        img1: Image.Image,
        img2: Image.Image,
        mask: np.ndarray,
        levels: Optional[int] = None,
    ) -> Image.Image:
        """
        Perform multi-band (Laplacian pyramid) blending.

        This creates smoother transitions than simple alpha blending
        by blending different frequency bands separately.

        Args:
            img1: First image
            img2: Second image
            mask: Blend mask (0 = img1, 1 = img2)
            levels: Number of pyramid levels

        Returns:
            Blended PIL Image
        """
        if levels is None:
            levels = self.num_pyramid_levels

        # Convert to arrays
        arr1 = np.array(img1).astype(np.float32) / 255.0
        arr2 = np.array(img2).astype(np.float32) / 255.0

        # Ensure mask has same shape
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        # Build Laplacian pyramids for both images
        lap1 = self._build_laplacian_pyramid(arr1, levels)
        lap2 = self._build_laplacian_pyramid(arr2, levels)

        # Build Gaussian pyramid for mask
        mask_pyr = self._build_gaussian_pyramid(mask, levels)

        # Blend pyramids
        blended_pyr = []
        for l1, l2, m in zip(lap1, lap2, mask_pyr):
            blended = l1 * (1 - m) + l2 * m
            blended_pyr.append(blended)

        # Reconstruct from pyramid
        result = self._reconstruct_from_pyramid(blended_pyr)

        # Convert back to uint8
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(result, mode="RGBA")

    def _build_gaussian_pyramid(
        self,
        img: np.ndarray,
        levels: int,
    ) -> list[np.ndarray]:
        """Build Gaussian pyramid."""
        pyramid = [img]
        current = img

        for _ in range(levels - 1):
            # Downsample with Gaussian blur
            blurred = ndimage.gaussian_filter(current, sigma=1)
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)
            current = downsampled

        return pyramid

    def _build_laplacian_pyramid(
        self,
        img: np.ndarray,
        levels: int,
    ) -> list[np.ndarray]:
        """Build Laplacian pyramid."""
        gaussian = self._build_gaussian_pyramid(img, levels)
        laplacian = []

        for i in range(levels - 1):
            # Upsample next level
            h, w = gaussian[i].shape[:2]
            upsampled = np.zeros_like(gaussian[i])

            # Simple nearest-neighbor upsample
            small = gaussian[i + 1]
            for y in range(min(h, small.shape[0] * 2)):
                for x in range(min(w, small.shape[1] * 2)):
                    upsampled[y, x] = small[y // 2, x // 2]

            # Blur upsampled
            upsampled = ndimage.gaussian_filter(upsampled, sigma=1)

            # Laplacian = current - upsampled
            laplacian.append(gaussian[i] - upsampled)

        # Last level is just Gaussian
        laplacian.append(gaussian[-1])

        return laplacian

    def _reconstruct_from_pyramid(self, pyramid: list[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        result = pyramid[-1]

        for i in range(len(pyramid) - 2, -1, -1):
            # Upsample
            h, w = pyramid[i].shape[:2]
            upsampled = np.zeros((h, w) + result.shape[2:], dtype=result.dtype)

            for y in range(min(h, result.shape[0] * 2)):
                for x in range(min(w, result.shape[1] * 2)):
                    upsampled[y, x] = result[y // 2, x // 2]

            # Blur and add Laplacian
            upsampled = ndimage.gaussian_filter(upsampled, sigma=1)
            result = upsampled + pyramid[i]

        return result

    def create_gradient_mask(
        self,
        width: int,
        height: int,
        direction: str = "horizontal",
        center: float = 0.5,
        falloff: float = 0.3,
    ) -> np.ndarray:
        """
        Create a gradient mask for blending.

        Args:
            width: Mask width
            height: Mask height
            direction: "horizontal" or "vertical"
            center: Position of blend center (0-1)
            falloff: Width of gradient transition (0-1)

        Returns:
            2D gradient mask array
        """
        if direction == "horizontal":
            x = np.linspace(0, 1, width)
            gradient = np.clip((x - center + falloff) / (2 * falloff), 0, 1)
            mask = np.tile(gradient, (height, 1))
        else:
            y = np.linspace(0, 1, height)
            gradient = np.clip((y - center + falloff) / (2 * falloff), 0, 1)
            mask = np.tile(gradient[:, np.newaxis], (1, width))

        return mask
