"""API request/response models."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..models.landmark import FeatureType, Landmark
from ..models.project import (
    BoundingBox,
    CardinalDirection,
    CoordinateMapping,
    DetailLevel,
    FillRegion,
    FocusRegion,
    OutputSettings,
    Project,
    SectionalLayout,
    StyleSettings,
    TileSettings,
)


# =============================================================================
# Project Schemas
# =============================================================================


class ProjectSummary(BaseModel):
    """Summary of a project for list views."""

    name: str
    region: BoundingBox
    area_km2: float
    tile_count: int
    landmark_count: int
    has_generated_tiles: bool = False
    last_modified: Optional[datetime] = None


class ProjectDetail(BaseModel):
    """Full project details including computed fields."""

    name: str
    region: BoundingBox
    output: OutputSettings
    style: StyleSettings
    tiles: TileSettings
    landmarks: list[Landmark]
    sectional_layout: Optional[SectionalLayout] = None

    # Computed fields
    area_km2: float
    detail_level: DetailLevel
    grid_cols: int
    grid_rows: int
    tile_count: int
    estimated_cost: Optional[float] = None

    @classmethod
    def from_project(cls, project: Project) -> "ProjectDetail":
        """Create from a Project model."""
        cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
        area = project.region.calculate_area_km2()

        return cls(
            name=project.name,
            region=project.region,
            output=project.output,
            style=project.style,
            tiles=project.tiles,
            landmarks=project.landmarks,
            sectional_layout=project.sectional_layout,
            area_km2=area,
            detail_level=project.region.get_recommended_detail_level(),
            grid_cols=cols,
            grid_rows=rows,
            tile_count=cols * rows,
        )


class ProjectCreate(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., min_length=1, max_length=100)
    region: BoundingBox
    output: Optional[OutputSettings] = None
    style: Optional[StyleSettings] = None
    tiles: Optional[TileSettings] = None


class ProjectUpdate(BaseModel):
    """Request to update project settings."""

    output: Optional[OutputSettings] = None
    style: Optional[StyleSettings] = None
    tiles: Optional[TileSettings] = None


# =============================================================================
# Tile Schemas
# =============================================================================


class TileStatus(str, Enum):
    """Status of a tile in the generation pipeline."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class TileSpec(BaseModel):
    """Specification for a single tile."""

    col: int
    row: int
    x_offset: int
    y_offset: int
    bbox: BoundingBox
    position_desc: str
    status: TileStatus = TileStatus.PENDING
    has_reference: bool = False
    has_generated: bool = False
    generation_time: Optional[float] = None
    error: Optional[str] = None
    offset_dx: int = 0
    offset_dy: int = 0


class TileGridResponse(BaseModel):
    """Response containing the tile grid for a project."""

    project_name: str
    cols: int
    rows: int
    tile_size: int
    overlap: int
    effective_size: int
    tiles: list[TileSpec]


class TileOffsetRequest(BaseModel):
    """Request to set a tile's position offset."""

    dx: int = Field(..., ge=-50, le=50, description="Horizontal offset in pixels")
    dy: int = Field(..., ge=-50, le=50, description="Vertical offset in pixels")


class TileOffsetResponse(BaseModel):
    """Response containing a tile's offset."""

    col: int
    row: int
    dx: int = 0
    dy: int = 0


class AllTileOffsetsResponse(BaseModel):
    """Response containing all tile offsets."""

    project_name: str
    offsets: list[TileOffsetResponse]


class TileRegenerateRequest(BaseModel):
    """Request to regenerate a specific tile."""

    force: bool = Field(default=False, description="Force regeneration even if cached")


# =============================================================================
# Seam Schemas
# =============================================================================


class SeamInfo(BaseModel):
    """Information about a seam between two tiles."""

    id: str
    orientation: Literal["horizontal", "vertical"]
    tile_a: tuple[int, int]
    tile_b: tuple[int, int]
    x: int
    y: int
    width: int
    height: int
    description: str
    is_repaired: bool = False


class SeamListResponse(BaseModel):
    """Response containing all seams for a project."""

    project_name: str
    total_seams: int
    repaired_seams: int
    seams: list[SeamInfo]


class SeamRepairRequest(BaseModel):
    """Request to repair a seam."""

    pass  # No additional parameters needed


class SeamBatchRepairRequest(BaseModel):
    """Request to repair multiple seams."""

    seam_ids: list[str]


# =============================================================================
# Landmark Schemas
# =============================================================================


class LandmarkCreate(BaseModel):
    """Request to create a new landmark."""

    name: str = Field(..., min_length=1, max_length=200)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    photo: Optional[str] = None
    logo: Optional[str] = None
    scale: float = Field(default=1.5, ge=0.5, le=5.0)
    z_index: int = Field(default=0)
    rotation: float = Field(default=0, ge=-180, le=180)
    feature_type: FeatureType = FeatureType.BUILDING
    path_coordinates: Optional[list[tuple[float, float]]] = None
    elevation_meters: Optional[float] = None
    horizon_feature: bool = False


class LandmarkUpdate(BaseModel):
    """Request to update a landmark."""

    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    scale: Optional[float] = Field(None, ge=0.5, le=5.0)
    z_index: Optional[int] = None
    rotation: Optional[float] = Field(None, ge=-180, le=180)


class LandmarkDetail(BaseModel):
    """Full landmark details including status."""

    name: str
    latitude: float
    longitude: float
    photo: Optional[str] = None
    logo: Optional[str] = None
    scale: float
    z_index: int
    rotation: float
    feature_type: FeatureType = FeatureType.BUILDING
    path_coordinates: Optional[list[tuple[float, float]]] = None
    elevation_meters: Optional[float] = None
    horizon_feature: bool = False
    illustrated_path: Optional[str] = None
    pixel_position: Optional[tuple[int, int]] = None
    has_photo: bool = False
    has_illustration: bool = False

    @classmethod
    def from_landmark(cls, landmark: Landmark, project_dir: Optional[Path] = None) -> "LandmarkDetail":
        """Create from a Landmark model."""
        has_photo = False
        has_illustration = False

        if project_dir:
            if landmark.photo:
                photo_path = project_dir / landmark.photo
                has_photo = photo_path.exists()
            if landmark.illustrated_path:
                illust_path = project_dir / landmark.illustrated_path
                has_illustration = illust_path.exists()

        return cls(
            name=landmark.name,
            latitude=landmark.latitude,
            longitude=landmark.longitude,
            photo=landmark.photo,
            logo=landmark.logo,
            scale=landmark.scale,
            z_index=landmark.z_index,
            rotation=landmark.rotation,
            feature_type=landmark.feature_type,
            path_coordinates=landmark.path_coordinates,
            elevation_meters=landmark.elevation_meters,
            horizon_feature=landmark.horizon_feature,
            illustrated_path=landmark.illustrated_path,
            pixel_position=landmark.pixel_position,
            has_photo=has_photo,
            has_illustration=has_illustration,
        )


# =============================================================================
# Generation Schemas
# =============================================================================


class GenerationStatus(str, Enum):
    """Status of a generation job."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationProgress(BaseModel):
    """Progress of a generation job."""

    status: GenerationStatus
    total_tiles: int
    completed_tiles: int
    failed_tiles: int
    current_tile: Optional[tuple[int, int]] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    error: Optional[str] = None


class GenerationStartRequest(BaseModel):
    """Request to start generation."""

    skip_existing: bool = Field(default=True, description="Skip tiles that already exist")
    tile_filter: Optional[list[tuple[int, int]]] = Field(
        default=None, description="Only generate these specific tiles"
    )


class GenerationStartResponse(BaseModel):
    """Response when starting generation."""

    task_id: str
    status: GenerationStatus
    total_tiles: int
    websocket_url: str


# =============================================================================
# Sectional Generation Schemas
# =============================================================================


class SectionalGenerationStartRequest(BaseModel):
    """Request to start sectional generation."""

    skip_existing: bool = Field(default=True, description="Skip sections that already exist")
    region_filter: Optional[list[str]] = Field(
        default=None, description="Only generate these specific regions by name"
    )


class SectionProgress(BaseModel):
    """Progress of a single section in sectional generation."""

    region_name: str
    region_type: str  # "focus" or "fill"
    status: GenerationStatus
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class SectionalGenerationProgress(BaseModel):
    """Progress of sectional generation job."""

    status: GenerationStatus
    total_sections: int
    completed_sections: int
    failed_sections: int
    current_section: Optional[str] = None
    sections: list[SectionProgress] = []
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


class SectionalGenerationStartResponse(BaseModel):
    """Response when starting sectional generation."""

    task_id: str
    status: GenerationStatus
    total_sections: int
    websocket_url: str


# =============================================================================
# Common Response Schemas
# =============================================================================


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: str
    detail: Optional[str] = None


class CostEstimate(BaseModel):
    """Cost estimate for an operation."""

    operation: str
    estimated_tokens: int
    estimated_cost_usd: float
    tile_count: Optional[int] = None
    seam_count: Optional[int] = None
    landmark_count: Optional[int] = None
