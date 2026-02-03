"""Outpainting region models for perspective map edge filling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image


class RegionType(Enum):
    """Type of region to outpaint."""

    HORIZON = "horizon"
    TRIANGLE_LEFT = "triangle_left"
    TRIANGLE_RIGHT = "triangle_right"


@dataclass
class OutpaintRegion:
    """A region of the image that needs to be outpainted.

    After perspective transform, the map becomes trapezoidal with empty areas:
    - Horizon band at the top
    - Triangular gaps on left and right from convergence
    """

    region_type: RegionType
    """Type of region (horizon, left triangle, right triangle)."""

    x: int
    """X position of region bounding box in image (pixels)."""

    y: int
    """Y position of region bounding box in image (pixels)."""

    width: int
    """Width of region bounding box (pixels)."""

    height: int
    """Height of region bounding box (pixels)."""

    polygon: Optional[list[tuple[int, int]]] = None
    """Polygon vertices for triangular regions. None for rectangular horizon."""

    mask: Optional[np.ndarray] = None
    """Binary mask where True = area to fill, False = existing content."""

    context_image: Optional[Image.Image] = None
    """Extracted context from adjacent visible map content."""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get (left, top, right, bottom) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def size(self) -> tuple[int, int]:
        """Get (width, height) of the region."""
        return (self.width, self.height)

    @property
    def area(self) -> int:
        """Total pixel area of the region."""
        return self.width * self.height

    def __str__(self) -> str:
        return f"OutpaintRegion[{self.region_type.value}] at ({self.x},{self.y}) {self.width}x{self.height}"


@dataclass
class HorizonTile:
    """A single tile within the horizon band for tiled outpainting.

    The horizon band is too wide for single Gemini generation (>4K pixels),
    so we divide it into overlapping tiles like the base map.
    """

    index: int
    """0-based tile index from left to right."""

    x: int
    """X position in the full image (pixels)."""

    y: int
    """Y position (always 0 for horizon band)."""

    width: int
    """Tile width (default 2048px)."""

    height: int
    """Tile height (same as horizon band height)."""

    overlap_left: int = 0
    """Overlap with previous tile (256px default, 0 for first tile)."""

    overlap_right: int = 0
    """Overlap with next tile (256px default, 0 for last tile)."""

    generated: Optional[Image.Image] = None
    """Generated outpaint result for this tile."""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get (left, top, right, bottom) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def content_x(self) -> int:
        """X position of non-overlapping content (for blending)."""
        return self.x + self.overlap_left

    @property
    def content_width(self) -> int:
        """Width of non-overlapping content."""
        return self.width - self.overlap_left - self.overlap_right

    def __str__(self) -> str:
        return f"HorizonTile[{self.index}] at x={self.x} width={self.width} overlap=({self.overlap_left},{self.overlap_right})"


@dataclass
class TerrainContext:
    """Geographic context about terrain north of the visible map.

    Used to build appropriate prompts for horizon outpainting.
    Fetched by querying OSM for features in the expanded bounding box.
    """

    primary_terrain: str = "urban"
    """Dominant terrain type: 'water', 'forest', 'mountain', 'urban', 'desert', 'plains'."""

    has_coastline: bool = False
    """Whether water (ocean/lake) is present to the north."""

    has_mountains: bool = False
    """Whether mountains/peaks are present to the north."""

    has_forest: bool = False
    """Whether significant forest/woodland is present."""

    has_water: bool = False
    """Whether any water bodies are present."""

    water_names: list[str] = field(default_factory=list)
    """Names of notable water bodies (e.g., 'Atlantic Ocean', 'Hudson River')."""

    mountain_names: list[str] = field(default_factory=list)
    """Names of notable peaks or ranges."""

    description: str = ""
    """Human-readable description for prompt building."""

    def build_description(self) -> str:
        """Build a natural language description of the terrain."""
        parts = []

        if self.has_coastline:
            if self.water_names:
                parts.append(f"coastline with {', '.join(self.water_names[:2])}")
            else:
                parts.append("coastline and water")
        elif self.has_water:
            if self.water_names:
                parts.append(f"water including {', '.join(self.water_names[:2])}")
            else:
                parts.append("water bodies")

        if self.has_mountains:
            if self.mountain_names:
                parts.append(f"mountains including {', '.join(self.mountain_names[:2])}")
            else:
                parts.append("mountainous terrain")

        if self.has_forest:
            parts.append("forested areas")

        if not parts:
            parts.append(f"{self.primary_terrain} landscape")

        self.description = "Terrain to the north includes " + ", ".join(parts) + "."
        return self.description

    def __str__(self) -> str:
        features = []
        if self.has_coastline:
            features.append("coast")
        if self.has_mountains:
            features.append("mtns")
        if self.has_forest:
            features.append("forest")
        if self.has_water and not self.has_coastline:
            features.append("water")
        feat_str = ", ".join(features) if features else "none"
        return f"TerrainContext[{self.primary_terrain}] features: {feat_str}"
