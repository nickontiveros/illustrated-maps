"""Tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from mapgen.api.schemas import (
    CostEstimate,
    ErrorResponse,
    GenerationStartRequest,
    LandmarkCreate,
    LandmarkDetail,
    LandmarkUpdate,
    ProjectCreate,
    ProjectDetail,
    ProjectUpdate,
    SeamInfo,
    SuccessResponse,
    TileOffsetRequest,
    TileSpec,
)
from mapgen.models.landmark import FeatureType, Landmark
from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings


class TestProjectCreate:
    """Test project creation schema."""

    def test_valid_creation(self):
        pc = ProjectCreate(
            name="test",
            region=BoundingBox(north=41, south=40, east=-73, west=-74),
        )
        assert pc.name == "test"

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            ProjectCreate(
                name="",
                region=BoundingBox(north=41, south=40, east=-73, west=-74),
            )

    def test_rejects_long_name(self):
        with pytest.raises(ValidationError):
            ProjectCreate(
                name="x" * 101,
                region=BoundingBox(north=41, south=40, east=-73, west=-74),
            )

    def test_optional_settings(self):
        pc = ProjectCreate(
            name="test",
            region=BoundingBox(north=41, south=40, east=-73, west=-74),
        )
        assert pc.output is None
        assert pc.style is None
        assert pc.tiles is None


class TestProjectUpdate:
    """Test project update schema."""

    def test_all_optional(self):
        pu = ProjectUpdate()
        assert pu.output is None
        assert pu.style is None
        assert pu.tiles is None

    def test_partial_update(self):
        pu = ProjectUpdate(output=OutputSettings(width=2000, height=2000))
        assert pu.output.width == 2000
        assert pu.tiles is None


class TestProjectDetail:
    """Test project detail schema."""

    def test_from_project(self, sample_project):
        detail = ProjectDetail.from_project(sample_project)
        assert detail.name == "test-project"
        assert detail.area_km2 > 0
        assert detail.grid_cols > 0
        assert detail.grid_rows > 0
        assert detail.tile_count == detail.grid_cols * detail.grid_rows

    def test_computed_fields(self, sample_project):
        detail = ProjectDetail.from_project(sample_project)
        assert detail.detail_level is not None
        assert detail.area_km2 > 0


class TestTileSpec:
    """Test tile spec schema."""

    def test_construction(self):
        ts = TileSpec(
            col=0, row=0, x_offset=0, y_offset=0,
            bbox=BoundingBox(north=41, south=40, east=-73, west=-74),
            position_desc="top-left corner",
        )
        assert ts.status.value == "pending"
        assert ts.has_reference is False
        assert ts.has_generated is False


class TestTileOffsetRequest:
    """Test tile offset schema."""

    def test_valid_offset(self):
        req = TileOffsetRequest(dx=10, dy=-5)
        assert req.dx == 10
        assert req.dy == -5

    def test_rejects_too_large(self):
        with pytest.raises(ValidationError):
            TileOffsetRequest(dx=100, dy=0)


class TestLandmarkCreate:
    """Test landmark creation schema."""

    def test_valid(self):
        lc = LandmarkCreate(name="Test", latitude=40.7, longitude=-74.0)
        assert lc.name == "Test"
        assert lc.scale == 1.5  # default
        assert lc.feature_type == FeatureType.BUILDING

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            LandmarkCreate(name="", latitude=40.7, longitude=-74.0)

    def test_rejects_invalid_coords(self):
        with pytest.raises(ValidationError):
            LandmarkCreate(name="Test", latitude=95, longitude=-74.0)


class TestLandmarkUpdate:
    """Test landmark update schema."""

    def test_all_optional(self):
        lu = LandmarkUpdate()
        assert lu.latitude is None
        assert lu.scale is None

    def test_partial_update(self):
        lu = LandmarkUpdate(scale=3.0)
        assert lu.scale == 3.0
        assert lu.latitude is None


class TestLandmarkDetail:
    """Test landmark detail schema."""

    def test_from_landmark(self):
        lm = Landmark(
            name="Test Building",
            latitude=40.77,
            longitude=-73.97,
            scale=2.0,
            z_index=5,
        )
        detail = LandmarkDetail.from_landmark(lm)
        assert detail.name == "Test Building"
        assert detail.latitude == 40.77
        assert detail.scale == 2.0
        assert detail.has_photo is False
        assert detail.has_illustration is False

    def test_from_landmark_with_photo(self, tmp_path):
        photo_path = tmp_path / "landmarks" / "photo.jpg"
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        photo_path.touch()

        lm = Landmark(
            name="Test",
            latitude=40.77,
            longitude=-73.97,
            photo="landmarks/photo.jpg",
        )
        detail = LandmarkDetail.from_landmark(lm, project_dir=tmp_path)
        assert detail.has_photo is True


class TestSeamInfo:
    """Test seam info schema."""

    def test_construction(self):
        si = SeamInfo(
            id="seam_0_1_h",
            orientation="horizontal",
            tile_a=(0, 0),
            tile_b=(1, 0),
            x=100, y=0, width=200, height=50,
            description="Seam between (0,0) and (1,0)",
        )
        assert si.is_repaired is False


class TestCommonSchemas:
    """Test common response schemas."""

    def test_success_response(self):
        sr = SuccessResponse(message="Done")
        assert sr.success is True

    def test_error_response(self):
        er = ErrorResponse(error="Something failed")
        assert er.success is False

    def test_cost_estimate(self):
        ce = CostEstimate(
            operation="generate",
            estimated_tokens=1000,
            estimated_cost_usd=0.50,
            tile_count=10,
        )
        assert ce.estimated_cost_usd == 0.50

    def test_generation_start_request(self):
        req = GenerationStartRequest()
        assert req.skip_existing is True
        assert req.tile_filter is None
