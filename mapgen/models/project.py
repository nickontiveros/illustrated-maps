"""Project configuration models."""

import math
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .atmosphere import AtmosphereSettings
from .border import BorderSettings
from .landmark import Landmark
from .narrative import NarrativeSettings
from .road_style import RoadStyleSettings
from .typography import TypographySettings


class CardinalDirection(str, Enum):
    """Cardinal direction for map orientation (which direction is 'up')."""

    NORTH = "north"  # Default - north is up
    SOUTH = "south"  # South is up (map rotated 180°)
    EAST = "east"    # East is up (map rotated 90° counter-clockwise)
    WEST = "west"    # West is up (map rotated 90° clockwise)

    @property
    def rotation_degrees(self) -> int:
        """Get the rotation angle in degrees (counter-clockwise from north-up)."""
        return {
            CardinalDirection.NORTH: 0,
            CardinalDirection.EAST: 90,
            CardinalDirection.SOUTH: 180,
            CardinalDirection.WEST: 270,
        }[self]


class DetailLevel(str, Enum):
    """Level of detail for OSM data extraction based on region size."""

    FULL = "full"  # All features (< 100 km²)
    SIMPLIFIED = "simplified"  # Major roads, notable buildings (100-1,000 km²)
    REGIONAL = "regional"  # Primary roads, landmarks only (1,000-50,000 km²)
    COUNTRY = "country"  # Motorways, major cities only (> 50,000 km²)


# Area thresholds for automatic detail level selection (in km²)
DETAIL_LEVEL_THRESHOLDS = {
    DetailLevel.FULL: 100,
    DetailLevel.SIMPLIFIED: 1_000,
    DetailLevel.REGIONAL: 50_000,
    # COUNTRY is for anything larger
}


def calculate_adjusted_dimensions(
    geo_aspect: float,
    target_area_pixels: int = 7016 * 9933,  # A1 area
    max_dimension: int = 12000,
) -> tuple[int, int]:
    """Calculate output dimensions matching geographic aspect ratio.

    Adjusts output dimensions to match the geographic aspect ratio while
    maintaining a similar total pixel area. This prevents distortion when
    the region's shape doesn't match the default A1 portrait dimensions.

    Args:
        geo_aspect: Geographic aspect ratio (width/height in degrees,
                   adjusted for latitude)
        target_area_pixels: Target total pixel area (default: A1 at 300 DPI)
        max_dimension: Maximum allowed dimension in pixels

    Returns:
        Tuple of (width, height) in pixels
    """
    # width * height = target_area
    # width / height = geo_aspect
    # Therefore: width = sqrt(target_area * geo_aspect)
    width = int(math.sqrt(target_area_pixels * geo_aspect))
    height = int(target_area_pixels / width)

    # Clamp to max dimension
    if width > max_dimension:
        width = max_dimension
        height = int(width / geo_aspect)
    if height > max_dimension:
        height = max_dimension
        width = int(height * geo_aspect)

    return width, height


def get_recommended_detail_level(area_km2: float) -> DetailLevel:
    """
    Get the recommended detail level for a given area.

    Args:
        area_km2: Area in square kilometers

    Returns:
        Recommended DetailLevel
    """
    if area_km2 < DETAIL_LEVEL_THRESHOLDS[DetailLevel.FULL]:
        return DetailLevel.FULL
    elif area_km2 < DETAIL_LEVEL_THRESHOLDS[DetailLevel.SIMPLIFIED]:
        return DetailLevel.SIMPLIFIED
    elif area_km2 < DETAIL_LEVEL_THRESHOLDS[DetailLevel.REGIONAL]:
        return DetailLevel.REGIONAL
    else:
        return DetailLevel.COUNTRY


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

    def calculate_area_km2(self) -> float:
        """
        Calculate the area of the bounding box in square kilometers.

        Uses the Haversine-based formula for calculating the area of a
        geographic rectangle on Earth's surface.

        Returns:
            Area in square kilometers
        """
        # Earth's radius in km
        R = 6371.0

        # Convert to radians
        lat1 = math.radians(self.south)
        lat2 = math.radians(self.north)
        lon1 = math.radians(self.west)
        lon2 = math.radians(self.east)

        # Handle longitude wrap-around
        delta_lon = lon2 - lon1
        if delta_lon < 0:
            delta_lon += 2 * math.pi

        # Calculate area using spherical geometry
        # Area = R² × |sin(lat2) - sin(lat1)| × |lon2 - lon1|
        area = (R**2) * abs(math.sin(lat2) - math.sin(lat1)) * abs(delta_lon)

        return area

    def get_recommended_detail_level(self) -> DetailLevel:
        """
        Get the recommended detail level for this bounding box.

        Returns:
            Recommended DetailLevel based on area
        """
        return get_recommended_detail_level(self.calculate_area_km2())

    @property
    def geographic_aspect_ratio(self) -> float:
        """Calculate the true geographic aspect ratio (width/height).

        Accounts for the fact that longitude degrees are shorter at higher
        latitudes. Uses the center latitude for the correction factor.
        """
        center_lat = (self.north + self.south) / 2
        lat_correction = math.cos(math.radians(center_lat))
        # Width in "corrected" degrees, height in latitude degrees
        corrected_width = self.width_degrees * lat_correction
        return corrected_width / self.height_degrees

    def expanded_for_rotation(self, degrees: float) -> "BoundingBox":
        """Compute an expanded bbox that covers the original after rotation.

        When the map is rotated by `degrees`, the corners of the original bbox
        sweep out a larger area. This returns a bbox that fully encloses the
        rotated original.

        Args:
            degrees: Clockwise rotation in degrees from north.

        Returns:
            Expanded BoundingBox.
        """
        if degrees % 360 == 0:
            return self

        center_lat, center_lon = self.center
        half_h = self.height_degrees / 2
        half_w = self.width_degrees / 2

        rad = math.radians(degrees)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))

        # Rotated bounding box half-extents
        new_half_w = half_w * cos_a + half_h * sin_a
        new_half_h = half_w * sin_a + half_h * cos_a

        return BoundingBox(
            north=min(90, center_lat + new_half_h),
            south=max(-90, center_lat - new_half_h),
            east=min(180, center_lon + new_half_w),
            west=max(-180, center_lon - new_half_w),
        )

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
    orientation: CardinalDirection = Field(
        default=CardinalDirection.NORTH,
        description="Which cardinal direction is 'up' on the map (legacy, prefer orientation_degrees)",
    )
    orientation_degrees: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        description="Arbitrary orientation angle in degrees clockwise from north (overrides orientation)",
    )
    prompt: str = Field(
        default=(
            "Transform this map into a hand illustrated tourist map style. "
            "Hand-painted illustration aesthetic with warm, muted colors. "
            "Aerial view perspective. Maintain exact layout of roads, buildings, "
            "and features. Clean edges suitable for tiling."
        ),
        description="Base prompt for Gemini image generation",
    )
    color_palette: Optional[list[str]] = Field(
        default=None,
        description="Optional color palette to enforce (hex colors)",
    )
    palette_preset: Optional[str] = Field(
        default=None,
        description="Named palette preset: vintage_tourist, modern_pop, ink_wash",
    )
    palette_enforcement_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Strength of post-generation palette clamping (0=off)",
    )
    color_consistency_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cross-tile color consistency strength (0=off, 1=full)",
    )
    typography: Optional[TypographySettings] = Field(
        default=None,
        description="Typography and labeling settings",
    )
    road_style: Optional[RoadStyleSettings] = Field(
        default=None,
        description="Enhanced road styling settings",
    )
    atmosphere: Optional[AtmosphereSettings] = Field(
        default=None,
        description="Atmospheric perspective settings",
    )
    terrain_exaggeration: float = Field(
        default=1.0,
        ge=1.0,
        le=5.0,
        description="DEM vertical exaggeration factor for hillshade",
    )

    @property
    def effective_rotation_degrees(self) -> float:
        """Get effective rotation in degrees (orientation_degrees takes precedence)."""
        if self.orientation_degrees is not None:
            return self.orientation_degrees
        return float(self.orientation.rotation_degrees)


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


class CoordinateMapping(BaseModel):
    """Non-linear coordinate mapping using piecewise-linear control points.

    Control points map geographic coordinates to normalized pixel-space (0-1).
    Points between control points are interpolated linearly.
    """

    lat_control_points: list[tuple[float, float]] = Field(
        default_factory=list,
        description="(geographic_lat, normalized_y 0-1) pairs, sorted by lat",
    )
    lon_control_points: list[tuple[float, float]] = Field(
        default_factory=list,
        description="(geographic_lon, normalized_x 0-1) pairs, sorted by lon",
    )


class FocusRegion(BaseModel):
    """A high-detail region (city/metro area) for sectional generation."""

    name: str
    bbox: BoundingBox
    detail_level: DetailLevel = DetailLevel.SIMPLIFIED
    scale_factor: float = Field(
        default=1.0,
        description="Pixel space multiplier vs geographic proportion",
    )
    landmark_names: list[str] = Field(default_factory=list)
    prompt_override: Optional[str] = None


class FillRegion(BaseModel):
    """A low-detail region (desert/corridor) connecting focus regions."""

    name: str
    bbox: BoundingBox
    detail_level: DetailLevel = DetailLevel.COUNTRY
    include_highways: bool = True
    include_rivers: bool = True
    include_terrain_shading: bool = True
    use_ai_illustration: bool = Field(
        default=False,
        description="Use Gemini for hand-painted feel (slower, costs money)",
    )
    prompt_override: Optional[str] = None


class SectionalLayout(BaseModel):
    """Layout configuration for sectional (multi-region) generation."""

    focus_regions: list[FocusRegion] = Field(default_factory=list)
    fill_regions: list[FillRegion] = Field(default_factory=list)
    coordinate_mapping: Optional[CoordinateMapping] = None


class Project(BaseModel):
    """Main project configuration."""

    name: str = Field(..., min_length=1, description="Project name")
    title: Optional[str] = Field(default=None, description="Map title for cartouche")
    subtitle: Optional[str] = Field(default=None, description="Map subtitle")
    region: BoundingBox
    output: OutputSettings = Field(default_factory=OutputSettings)
    style: StyleSettings = Field(default_factory=StyleSettings)
    tiles: TileSettings = Field(default_factory=TileSettings)
    landmarks: list[Landmark] = Field(default_factory=list, description="Landmarks to illustrate")
    border: Optional[BorderSettings] = Field(
        default=None,
        description="Decorative border settings",
    )
    narrative: Optional[NarrativeSettings] = Field(
        default=None,
        description="Landmark discovery and narrative settings",
    )
    sectional_layout: Optional[SectionalLayout] = Field(
        default=None,
        description="Sectional layout for large-region generation",
    )

    # Paths (set when loading from file)
    project_dir: Optional[Path] = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "Project":
        """Load project from YAML file.

        Handles backward-compatible migration:
        - Old `orientation` cardinal direction is preserved but
          `orientation_degrees` takes precedence when present.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        project = cls(**data)
        project.project_dir = path.parent
        return project

    def to_yaml(self, path: Path) -> None:
        """Save project to YAML file."""
        data = self.model_dump(exclude={"project_dir"}, mode="json")
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
