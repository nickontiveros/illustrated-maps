"""Seam data models for tile boundary repair."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SeamInfo:
    """Information about a seam between two adjacent tiles.

    A seam is the boundary region where two tiles meet and blend together.
    This class tracks the location and properties of each seam for repair.
    """

    orientation: Literal["horizontal", "vertical"]
    """Seam orientation - horizontal (tiles side by side) or vertical (tiles stacked)."""

    tile_a: tuple[int, int]
    """(col, row) coordinates of the first tile (left or top)."""

    tile_b: tuple[int, int]
    """(col, row) coordinates of the second tile (right or bottom)."""

    x: int
    """X position of seam region in assembled image (pixels)."""

    y: int
    """Y position of seam region in assembled image (pixels)."""

    width: int
    """Width of seam region (pixels)."""

    height: int
    """Height of seam region (pixels)."""

    @property
    def id(self) -> str:
        """Unique identifier for this seam."""
        return f"{self.tile_a[0]},{self.tile_a[1]}-{self.tile_b[0]},{self.tile_b[1]}"

    @property
    def description(self) -> str:
        """Human-readable description of the seam."""
        direction = "→" if self.orientation == "horizontal" else "↓"
        return f"Tile ({self.tile_a[0]},{self.tile_a[1]}) {direction} ({self.tile_b[0]},{self.tile_b[1]})"

    def __str__(self) -> str:
        return f"Seam[{self.id}] {self.orientation} at ({self.x},{self.y}) {self.width}x{self.height}"
