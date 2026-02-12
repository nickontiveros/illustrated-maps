"""Tests for Project, OutputSettings, TileSettings, and related models."""

import pytest
from pydantic import ValidationError

from mapgen.models.project import (
    CardinalDirection,
    DetailLevel,
    OutputSettings,
    Project,
    StyleSettings,
    TileSettings,
    calculate_adjusted_dimensions,
    get_recommended_detail_level,
)


class TestOutputSettings:
    """Test output settings model."""

    def test_defaults(self):
        s = OutputSettings()
        assert s.width == 7016
        assert s.height == 9933
        assert s.dpi == 300

    def test_aspect_ratio(self):
        s = OutputSettings(width=1000, height=500)
        assert s.aspect_ratio == pytest.approx(2.0)

    def test_rejects_too_small(self):
        with pytest.raises(ValidationError):
            OutputSettings(width=50, height=50)


class TestTileSettings:
    """Test tile settings model."""

    def test_defaults(self):
        s = TileSettings()
        assert s.size == 2048
        assert s.overlap == 256

    def test_effective_size(self):
        s = TileSettings(size=1024, overlap=128)
        assert s.effective_size == 896

    def test_calculate_grid(self):
        s = TileSettings(size=512, overlap=64)
        cols, rows = s.calculate_grid(1024, 1024)
        # effective_size = 448, ceil(1024/448) = 3
        assert cols == 3
        assert rows == 3

    def test_calculate_grid_single_tile(self):
        s = TileSettings(size=2048, overlap=256)
        cols, rows = s.calculate_grid(1000, 1000)
        assert cols == 1
        assert rows == 1


class TestCardinalDirection:
    """Test cardinal direction enum."""

    def test_north_rotation(self):
        assert CardinalDirection.NORTH.rotation_degrees == 0

    def test_east_rotation(self):
        assert CardinalDirection.EAST.rotation_degrees == 90

    def test_south_rotation(self):
        assert CardinalDirection.SOUTH.rotation_degrees == 180

    def test_west_rotation(self):
        assert CardinalDirection.WEST.rotation_degrees == 270


class TestDetailLevel:
    """Test detail level enum."""

    def test_values(self):
        assert DetailLevel.FULL.value == "full"
        assert DetailLevel.SIMPLIFIED.value == "simplified"
        assert DetailLevel.REGIONAL.value == "regional"
        assert DetailLevel.COUNTRY.value == "country"


class TestGetRecommendedDetailLevel:
    """Test detail level recommendation function."""

    def test_small_area(self):
        assert get_recommended_detail_level(50) == DetailLevel.FULL

    def test_medium_area(self):
        assert get_recommended_detail_level(500) == DetailLevel.SIMPLIFIED

    def test_large_area(self):
        assert get_recommended_detail_level(10_000) == DetailLevel.REGIONAL

    def test_very_large_area(self):
        assert get_recommended_detail_level(100_000) == DetailLevel.COUNTRY

    def test_boundary_100(self):
        assert get_recommended_detail_level(99) == DetailLevel.FULL
        assert get_recommended_detail_level(100) == DetailLevel.SIMPLIFIED


class TestCalculateAdjustedDimensions:
    """Test dimension calculation function."""

    def test_square_aspect(self):
        w, h = calculate_adjusted_dimensions(1.0)
        assert w == h

    def test_wide_aspect(self):
        w, h = calculate_adjusted_dimensions(2.0)
        assert w > h

    def test_tall_aspect(self):
        w, h = calculate_adjusted_dimensions(0.5)
        assert h > w

    def test_respects_max_dimension(self):
        w, h = calculate_adjusted_dimensions(10.0, max_dimension=5000)
        assert w <= 5000
        assert h <= 5000

    def test_preserves_area(self):
        target_area = 7016 * 9933
        w, h = calculate_adjusted_dimensions(1.5, target_area_pixels=target_area)
        actual_area = w * h
        # Should be within 1% of target area
        assert abs(actual_area - target_area) / target_area < 0.01


class TestStyleSettings:
    """Test style settings model."""

    def test_effective_rotation_default(self):
        s = StyleSettings()
        assert s.effective_rotation_degrees == 0.0

    def test_effective_rotation_cardinal(self):
        s = StyleSettings(orientation=CardinalDirection.EAST)
        assert s.effective_rotation_degrees == 90.0

    def test_orientation_degrees_overrides(self):
        s = StyleSettings(orientation=CardinalDirection.EAST, orientation_degrees=45.0)
        assert s.effective_rotation_degrees == 45.0


class TestProject:
    """Test Project model."""

    def test_construction(self, sample_project):
        assert sample_project.name == "test-project"
        assert sample_project.output.width == 1024

    def test_requires_name(self, sample_bbox):
        with pytest.raises(ValidationError):
            Project(name="", region=sample_bbox)

    def test_landmarks_dir(self, sample_project):
        assert sample_project.landmarks_dir == sample_project.project_dir / "landmarks"

    def test_logos_dir(self, sample_project):
        assert sample_project.logos_dir == sample_project.project_dir / "logos"

    def test_output_dir(self, sample_project):
        assert sample_project.output_dir == sample_project.project_dir / "output"

    def test_dir_properties_raise_without_project_dir(self, sample_bbox):
        project = Project(name="test", region=sample_bbox)
        with pytest.raises(ValueError, match="not loaded from file"):
            _ = project.landmarks_dir

    def test_ensure_directories(self, sample_project):
        sample_project.ensure_directories()
        assert sample_project.landmarks_dir.exists()
        assert sample_project.logos_dir.exists()
        assert sample_project.output_dir.exists()
