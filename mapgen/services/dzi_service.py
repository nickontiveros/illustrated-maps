"""Deep Zoom Image (DZI) service for handling large images.

This service generates tile pyramids from large images (like assembled maps)
to enable smooth panning and zooming in the browser without loading the
entire image into memory.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class DZIInfo:
    """Information about a DZI image."""

    width: int
    height: int
    tile_size: int
    overlap: int
    format: str
    max_level: int

    @property
    def num_levels(self) -> int:
        return self.max_level + 1

    def get_level_dimensions(self, level: int) -> tuple[int, int]:
        """Get dimensions at a specific level."""
        scale = 2 ** (self.max_level - level)
        return (
            max(1, self.width // scale),
            max(1, self.height // scale),
        )

    def get_num_tiles(self, level: int) -> tuple[int, int]:
        """Get number of tiles at a specific level."""
        w, h = self.get_level_dimensions(level)
        return (
            math.ceil(w / self.tile_size),
            math.ceil(h / self.tile_size),
        )

    def to_dzi_xml(self) -> str:
        """Generate DZI descriptor XML."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{self.format}"
       Overlap="{self.overlap}"
       TileSize="{self.tile_size}">
    <Size Height="{self.height}" Width="{self.width}"/>
</Image>'''


class DZIService:
    """Service for generating and serving Deep Zoom Images."""

    def __init__(
        self,
        tile_size: int = 254,
        overlap: int = 1,
        format: str = "jpg",
        quality: int = 85,
    ):
        """
        Initialize DZI service.

        Args:
            tile_size: Size of each DZI tile (default 254)
            overlap: Overlap between tiles (default 1)
            format: Output format (jpg or png)
            quality: JPEG quality (1-100)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.format = format
        self.quality = quality

    def get_info(self, image_path: Path) -> DZIInfo:
        """Get DZI info for an image without generating tiles.

        Args:
            image_path: Path to source image

        Returns:
            DZIInfo with image dimensions and DZI parameters
        """
        with Image.open(image_path) as img:
            width, height = img.size

        max_level = math.ceil(math.log2(max(width, height)))

        return DZIInfo(
            width=width,
            height=height,
            tile_size=self.tile_size,
            overlap=self.overlap,
            format=self.format,
            max_level=max_level,
        )

    def generate_tiles(
        self,
        image_path: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> DZIInfo:
        """Generate DZI tile pyramid from a large image.

        Uses pyvips if available for memory-efficient processing,
        falls back to PIL for smaller images.

        Args:
            image_path: Path to source image
            output_dir: Directory to write tiles (will create {name}_files/ subdirectory)
            progress_callback: Optional callback(level, total_levels) for progress

        Returns:
            DZIInfo describing the generated pyramid
        """
        # Try to use pyvips for large images (memory efficient)
        try:
            return self._generate_with_pyvips(image_path, output_dir, progress_callback)
        except ImportError:
            # Fall back to PIL
            return self._generate_with_pil(image_path, output_dir, progress_callback)

    def _generate_with_pyvips(
        self,
        image_path: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> DZIInfo:
        """Generate tiles using pyvips (memory efficient)."""
        import pyvips

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        img = pyvips.Image.new_from_file(str(image_path))
        width, height = img.width, img.height
        max_level = math.ceil(math.log2(max(width, height)))

        info = DZIInfo(
            width=width,
            height=height,
            tile_size=self.tile_size,
            overlap=self.overlap,
            format=self.format,
            max_level=max_level,
        )

        # Write DZI descriptor
        dzi_path = output_dir / f"{image_path.stem}.dzi"
        dzi_path.write_text(info.to_dzi_xml())

        # Create tiles directory
        tiles_dir = output_dir / f"{image_path.stem}_files"
        tiles_dir.mkdir(exist_ok=True)

        # Generate each level
        for level in range(max_level + 1):
            if progress_callback:
                progress_callback(level, max_level + 1)

            level_dir = tiles_dir / str(level)
            level_dir.mkdir(exist_ok=True)

            # Calculate scale for this level
            scale = 2 ** (max_level - level)
            level_width = max(1, width // scale)
            level_height = max(1, height // scale)

            # Resize image for this level
            if scale > 1:
                level_img = img.resize(1.0 / scale)
            else:
                level_img = img

            # Generate tiles for this level
            cols = math.ceil(level_width / self.tile_size)
            rows = math.ceil(level_height / self.tile_size)

            for row in range(rows):
                for col in range(cols):
                    # Calculate tile bounds with overlap
                    x = col * self.tile_size
                    y = row * self.tile_size

                    # Add overlap (except at edges)
                    x_start = max(0, x - self.overlap)
                    y_start = max(0, y - self.overlap)
                    x_end = min(level_width, x + self.tile_size + self.overlap)
                    y_end = min(level_height, y + self.tile_size + self.overlap)

                    # Extract tile
                    tile = level_img.crop(
                        x_start,
                        y_start,
                        x_end - x_start,
                        y_end - y_start,
                    )

                    # Save tile
                    tile_path = level_dir / f"{col}_{row}.{self.format}"
                    if self.format == "jpg":
                        tile.jpegsave(str(tile_path), Q=self.quality)
                    else:
                        tile.pngsave(str(tile_path))

        return info

    def _generate_with_pil(
        self,
        image_path: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> DZIInfo:
        """Generate tiles using PIL (for smaller images or fallback)."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            max_level = math.ceil(math.log2(max(width, height)))

            info = DZIInfo(
                width=width,
                height=height,
                tile_size=self.tile_size,
                overlap=self.overlap,
                format=self.format,
                max_level=max_level,
            )

            # Write DZI descriptor
            dzi_path = output_dir / f"{image_path.stem}.dzi"
            dzi_path.write_text(info.to_dzi_xml())

            # Create tiles directory
            tiles_dir = output_dir / f"{image_path.stem}_files"
            tiles_dir.mkdir(exist_ok=True)

            # Generate each level (from highest to lowest resolution)
            current_img = img.copy()

            for level in range(max_level, -1, -1):
                if progress_callback:
                    progress_callback(max_level - level, max_level + 1)

                level_dir = tiles_dir / str(level)
                level_dir.mkdir(exist_ok=True)

                level_width, level_height = current_img.size

                # Generate tiles for this level
                cols = math.ceil(level_width / self.tile_size)
                rows = math.ceil(level_height / self.tile_size)

                for row in range(rows):
                    for col in range(cols):
                        # Calculate tile bounds with overlap
                        x = col * self.tile_size
                        y = row * self.tile_size

                        x_start = max(0, x - self.overlap)
                        y_start = max(0, y - self.overlap)
                        x_end = min(level_width, x + self.tile_size + self.overlap)
                        y_end = min(level_height, y + self.tile_size + self.overlap)

                        # Extract tile
                        tile = current_img.crop((x_start, y_start, x_end, y_end))

                        # Save tile
                        tile_path = level_dir / f"{col}_{row}.{self.format}"
                        if self.format == "jpg":
                            tile.save(tile_path, "JPEG", quality=self.quality)
                        else:
                            tile.save(tile_path, "PNG")

                # Resize for next level (half size)
                if level > 0:
                    new_width = max(1, level_width // 2)
                    new_height = max(1, level_height // 2)
                    current_img = current_img.resize(
                        (new_width, new_height),
                        Image.Resampling.LANCZOS,
                    )

        return info

    def get_tile_path(
        self,
        output_dir: Path,
        image_name: str,
        level: int,
        col: int,
        row: int,
    ) -> Path:
        """Get the path to a specific tile.

        Args:
            output_dir: DZI output directory
            image_name: Base name of the image (without extension)
            level: Zoom level
            col: Column index
            row: Row index

        Returns:
            Path to the tile file
        """
        return output_dir / f"{image_name}_files" / str(level) / f"{col}_{row}.{self.format}"

    def tile_exists(
        self,
        output_dir: Path,
        image_name: str,
        level: int,
        col: int,
        row: int,
    ) -> bool:
        """Check if a tile exists."""
        return self.get_tile_path(output_dir, image_name, level, col, row).exists()

    def is_generated(self, output_dir: Path, image_name: str) -> bool:
        """Check if DZI tiles have been generated for an image."""
        dzi_path = output_dir / f"{image_name}.dzi"
        tiles_dir = output_dir / f"{image_name}_files"
        return dzi_path.exists() and tiles_dir.exists()
