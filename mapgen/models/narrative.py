"""Narrative and landmark intelligence models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LandmarkTier(str, Enum):
    """Importance tier for auto-discovered landmarks."""

    MAJOR = "major"  # Scale 2.5x - major attractions
    NOTABLE = "notable"  # Scale 1.8x - notable points of interest
    MINOR = "minor"  # Scale 1.2x - minor POIs


# Scale factors per tier
TIER_SCALES = {
    LandmarkTier.MAJOR: 2.5,
    LandmarkTier.NOTABLE: 1.8,
    LandmarkTier.MINOR: 1.2,
}


class ActivityCategory(str, Enum):
    """Categories for activity markers."""

    DINING = "dining"
    SHOPPING = "shopping"
    SWIMMING = "swimming"
    HIKING = "hiking"
    MUSEUM = "museum"
    THEATER = "theater"
    PARK = "park"
    BEACH = "beach"
    SPORTS = "sports"
    NIGHTLIFE = "nightlife"
    ACCOMMODATION = "accommodation"
    TRANSPORT = "transport"
    VIEWPOINT = "viewpoint"
    HISTORIC = "historic"
    WORSHIP = "worship"


# Mapping from OSM amenity/tourism tags to activity categories
OSM_TAG_TO_CATEGORY = {
    # Amenity tags
    "restaurant": ActivityCategory.DINING,
    "cafe": ActivityCategory.DINING,
    "fast_food": ActivityCategory.DINING,
    "bar": ActivityCategory.NIGHTLIFE,
    "pub": ActivityCategory.NIGHTLIFE,
    "nightclub": ActivityCategory.NIGHTLIFE,
    "theatre": ActivityCategory.THEATER,
    "cinema": ActivityCategory.THEATER,
    "swimming_pool": ActivityCategory.SWIMMING,
    "place_of_worship": ActivityCategory.WORSHIP,
    # Tourism tags
    "museum": ActivityCategory.MUSEUM,
    "hotel": ActivityCategory.ACCOMMODATION,
    "hostel": ActivityCategory.ACCOMMODATION,
    "viewpoint": ActivityCategory.VIEWPOINT,
    "beach_resort": ActivityCategory.BEACH,
    "camp_site": ActivityCategory.HIKING,
    # Historic tags
    "castle": ActivityCategory.HISTORIC,
    "monument": ActivityCategory.HISTORIC,
    "memorial": ActivityCategory.HISTORIC,
    "ruins": ActivityCategory.HISTORIC,
    # Leisure tags
    "sports_centre": ActivityCategory.SPORTS,
    "stadium": ActivityCategory.SPORTS,
    "park": ActivityCategory.PARK,
    "nature_reserve": ActivityCategory.HIKING,
}


class ActivityMarker(BaseModel):
    """An activity icon placed at a POI."""

    category: ActivityCategory
    latitude: float
    longitude: float
    name: Optional[str] = None


class HistoricalMarker(BaseModel):
    """A historical information marker."""

    latitude: float
    longitude: float
    name: str
    description: Optional[str] = None
    year: Optional[int] = None


class NarrativeSettings(BaseModel):
    """Configuration for landmark discovery and narrative content."""

    auto_discover: bool = Field(
        default=False,
        description="Automatically discover landmarks from OSM",
    )
    max_landmarks: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Maximum number of auto-discovered landmarks",
    )
    show_activities: bool = Field(
        default=False,
        description="Show activity icons at POIs",
    )
    max_activity_markers: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Maximum number of activity markers",
    )
    discovery_categories: list[str] = Field(
        default_factory=lambda: [
            "tourism",
            "historic",
            "cathedral",
            "stadium",
            "university",
        ],
        description="OSM categories to search for landmark discovery",
    )
    min_importance_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for inclusion",
    )
