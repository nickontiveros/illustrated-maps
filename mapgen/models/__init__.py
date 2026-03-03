"""Data models for map generation."""

from .project import (
    Project,
    BoundingBox,
    OrientedRegion,
    OutputSettings,
    StyleSettings,
    TileSettings,
    DetailLevel,
    GenerationMode,
    CoordinateMapping,
    FocusRegion,
    FillRegion,
    SectionalLayout,
    get_recommended_detail_level,
    DETAIL_LEVEL_THRESHOLDS,
)
from .landmark import Landmark, FeatureType
from .seam import SeamInfo
from .outpaint_region import OutpaintRegion, HorizonTile, TerrainContext, RegionType

__all__ = [
    "Project",
    "BoundingBox",
    "OrientedRegion",
    "OutputSettings",
    "StyleSettings",
    "TileSettings",
    "DetailLevel",
    "GenerationMode",
    "CoordinateMapping",
    "FocusRegion",
    "FillRegion",
    "SectionalLayout",
    "get_recommended_detail_level",
    "DETAIL_LEVEL_THRESHOLDS",
    "Landmark",
    "FeatureType",
    "SeamInfo",
    "OutpaintRegion",
    "HorizonTile",
    "TerrainContext",
    "RegionType",
]
