"""Typography and labeling models for illustrated maps."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FontTier(str, Enum):
    """Hierarchy of text tiers from largest to smallest."""

    TITLE = "title"
    SUBTITLE = "subtitle"
    DISTRICT = "district"
    ROAD_MAJOR = "road_major"
    ROAD_MINOR = "road_minor"
    WATER = "water"
    PARK = "park"


# Default font sizes per tier (in points at 300 DPI)
FONT_SIZES = {
    FontTier.TITLE: (72, 120),
    FontTier.SUBTITLE: (48, 60),
    FontTier.DISTRICT: (36, 48),
    FontTier.ROAD_MAJOR: (14, 20),
    FontTier.ROAD_MINOR: (10, 14),
    FontTier.WATER: (18, 24),
    FontTier.PARK: (14, 18),
}

# Default colors per tier
FONT_COLORS = {
    FontTier.TITLE: "#2C1810",
    FontTier.SUBTITLE: "#4A3728",
    FontTier.DISTRICT: "#3D2B1F",
    FontTier.ROAD_MAJOR: "#4A4A4A",
    FontTier.ROAD_MINOR: "#6A6A6A",
    FontTier.WATER: "#1A5276",
    FontTier.PARK: "#1B5E20",
}


class Label(BaseModel):
    """A single text label to be placed on the map."""

    text: str
    tier: FontTier
    latitude: float
    longitude: float
    rotation: float = 0.0  # Degrees, for curved road text
    font_size: Optional[int] = None  # Override default for tier
    color: Optional[str] = None  # Override default for tier


class TextPath(BaseModel):
    """A path along which text should be rendered (for road names)."""

    text: str
    tier: FontTier
    points: list[tuple[float, float]]  # List of (lat, lon) points along the path
    font_size: Optional[int] = None
    color: Optional[str] = None


class TypographySettings(BaseModel):
    """Configuration for the typography system."""

    enabled: bool = Field(default=False, description="Enable typography labels")
    road_labels: bool = Field(default=True, description="Show road name labels")
    district_labels: bool = Field(default=True, description="Show district/neighborhood labels")
    water_labels: bool = Field(default=True, description="Show water body labels")
    park_labels: bool = Field(default=True, description="Show park labels")
    title_text: Optional[str] = Field(default=None, description="Map title text")
    subtitle_text: Optional[str] = Field(default=None, description="Map subtitle text")
    font_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Global font size multiplier",
    )
    halo_width: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Width of text halo/outline for legibility",
    )
    max_labels: int = Field(
        default=200,
        ge=0,
        description="Maximum number of labels to render (to avoid clutter)",
    )
    min_road_length_px: int = Field(
        default=100,
        ge=20,
        description="Minimum road length in pixels to receive a label",
    )
