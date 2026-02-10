"""Landmark model for map features."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    """Type of geographic/architectural feature for a landmark."""

    BUILDING = "building"
    MOUNTAIN = "mountain"
    RIVER = "river"
    PARK = "park"
    MONUMENT = "monument"
    STADIUM = "stadium"
    CAMPUS = "campus"
    AIRPORT = "airport"
    NATURAL = "natural"


class Landmark(BaseModel):
    """A landmark to be illustrated and placed on the map."""

    name: str = Field(..., min_length=1, description="Landmark name")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

    # Input files (relative to project directory)
    photo: Optional[str] = Field(default=None, description="Path to landmark photo")
    logo: Optional[str] = Field(default=None, description="Path to logo PNG")

    # Feature metadata
    feature_type: FeatureType = Field(
        default=FeatureType.BUILDING,
        description="Type of geographic/architectural feature",
    )
    path_coordinates: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Coordinate path for linear features like rivers (lat, lon pairs)",
    )
    elevation_meters: Optional[float] = Field(
        default=None,
        description="Elevation in meters (for mountains/peaks)",
    )
    horizon_feature: bool = Field(
        default=False,
        description="Render on horizon line as silhouette",
    )

    # Display settings
    scale: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Scale factor for landmark (1.0 = actual, >1 = exaggerated)",
    )
    z_index: int = Field(default=0, description="Z-index for layering (higher = on top)")
    rotation: float = Field(default=0, ge=-180, le=180, description="Rotation in degrees")

    # Generated outputs (set during processing)
    illustrated_path: Optional[str] = Field(
        default=None,
        description="Path to generated illustration",
    )
    pixel_position: Optional[tuple[int, int]] = Field(
        default=None,
        description="Calculated pixel position on map",
    )

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return (self.latitude, self.longitude)

    def resolve_photo_path(self, project_dir: Path) -> Optional[Path]:
        """Resolve photo path relative to project directory."""
        if self.photo is None:
            return None
        return project_dir / self.photo

    def resolve_logo_path(self, project_dir: Path) -> Optional[Path]:
        """Resolve logo path relative to project directory."""
        if self.logo is None:
            return None
        return project_dir / self.logo
