"""Data models for map generation."""

from .project import (
    Project,
    BoundingBox,
    OutputSettings,
    StyleSettings,
    TileSettings,
    DetailLevel,
    get_recommended_detail_level,
    DETAIL_LEVEL_THRESHOLDS,
)
from .landmark import Landmark
from .seam import SeamInfo
from .outpaint_region import OutpaintRegion, HorizonTile, TerrainContext, RegionType

__all__ = [
    "Project",
    "BoundingBox",
    "OutputSettings",
    "StyleSettings",
    "TileSettings",
    "DetailLevel",
    "get_recommended_detail_level",
    "DETAIL_LEVEL_THRESHOLDS",
    "Landmark",
    "SeamInfo",
    "OutpaintRegion",
    "HorizonTile",
    "TerrainContext",
    "RegionType",
]
