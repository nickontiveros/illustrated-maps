"""Project configuration models."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .landmark import Landmark


class BoundingBox(BaseModel):
    """Geographic bounding box for the map region."""

    north: float = Field(..., ge=-90, le=90, description="Northern latitude boundary")
    south: float = Field(..., ge=-90, le=90, description="Southern latitude boundary")
    east: float = Field(..., ge=-180, le=180, description="Eastern longitude boundary")
    west: float = Field(..., ge=-180, le=180, description="Western longitude boundary")

    @property
    def center(self) -> tuple[float, float]:
        """Return center point (lat, lon)."""
        return ((self.north + self.south) / 2, (self.east + self.west) / 2)

    @property
    def width_degrees(self) -> float:
        """Width in degrees longitude."""
        return abs(self.east - self.west)

    @property
    def height_degrees(self) -> float:
        """Height in degrees latitude."""
        return abs(self.north - self.south)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (north, south, east, west) tuple for OSMnx."""
        return (self.north, self.south, self.east, self.west)

    def to_osmnx_bbox(self) -> tuple[float, float, float, float]:
        """Return as (west, south, east, north) tuple for OSMnx graph_from_bbox."""
        return (self.west, self.south, self.east, self.north)


class OutputSettings(BaseModel):
    """Output image settings."""

    width: int = Field(default=7016, ge=100, description="Output width in pixels (A1 @ 300 DPI)")
    height: int = Field(default=9933, ge=100, description="Output height in pixels (A1 @ 300 DPI)")
    dpi: int = Field(default=300, ge=72, description="Output DPI")

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width / self.height


class StyleSettings(BaseModel):
    """Visual style settings for map generation."""

    perspective_angle: float = Field(
        default=35.264,
        ge=15,
        le=60,
        description="Isometric projection angle in degrees",
    )
    prompt: str = Field(
        default=(
            "Transform this map into a vibrant illustrated theme park map style, "
            "similar to Disneyland park maps. Hand-painted illustration aesthetic with "
            "warm, saturated colors. Isometric aerial view perspective. Maintain exact "
            "layout of roads, buildings, and features. Clean edges suitable for tiling."
        ),
        description="Base prompt for Gemini image generation",
    )
    color_palette: Optional[list[str]] = Field(
        default=None,
        description="Optional color palette to enforce (hex colors)",
    )


class TileSettings(BaseModel):
    """Tile generation settings for high-resolution output."""

    size: int = Field(default=2048, ge=512, le=4096, description="Tile size in pixels")
    overlap: int = Field(default=256, ge=64, le=512, description="Tile overlap in pixels")

    @property
    def effective_size(self) -> int:
        """Effective tile size after overlap."""
        return self.size - self.overlap

    def calculate_grid(self, width: int, height: int) -> tuple[int, int]:
        """Calculate number of tiles needed for given dimensions."""
        import math

        cols = math.ceil(width / self.effective_size)
        rows = math.ceil(height / self.effective_size)
        return (cols, rows)


class Project(BaseModel):
    """Main project configuration."""

    name: str = Field(..., min_length=1, description="Project name")
    region: BoundingBox
    output: OutputSettings = Field(default_factory=OutputSettings)
    style: StyleSettings = Field(default_factory=StyleSettings)
    tiles: TileSettings = Field(default_factory=TileSettings)
    landmarks: list[Landmark] = Field(default_factory=list, description="Landmarks to illustrate")

    # Paths (set when loading from file)
    project_dir: Optional[Path] = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "Project":
        """Load project from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        project = cls(**data)
        project.project_dir = path.parent
        return project

    def to_yaml(self, path: Path) -> None:
        """Save project to YAML file."""
        data = self.model_dump(exclude={"project_dir"})
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @property
    def landmarks_dir(self) -> Path:
        """Directory for landmark photos."""
        if self.project_dir is None:
            raise ValueError("Project not loaded from file")
        return self.project_dir / "landmarks"

    @property
    def logos_dir(self) -> Path:
        """Directory for logo PNGs."""
        if self.project_dir is None:
            raise ValueError("Project not loaded from file")
        return self.project_dir / "logos"

    @property
    def output_dir(self) -> Path:
        """Directory for generated output."""
        if self.project_dir is None:
            raise ValueError("Project not loaded from file")
        return self.project_dir / "output"

    def ensure_directories(self) -> None:
        """Create project directories if they don't exist."""
        if self.project_dir is None:
            raise ValueError("Project not loaded from file")
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)
        self.logos_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
