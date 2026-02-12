"""Tests for GenerationService pure methods (tile specs, cost, water detection)."""

import numpy as np
import pytest
from PIL import Image

from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings
from mapgen.services.generation_service import GenerationService


@pytest.fixture
def gen_service(tmp_path):
    """GenerationService with a simple project (no API keys needed)."""
    bbox = BoundingBox(north=40.78, south=40.76, east=-73.96, west=-73.98)
    project = Project(
        name="test",
        region=bbox,
        output=OutputSettings(width=1024, height=1024, dpi=72),
        tiles=TileSettings(size=512, overlap=64),
    )
    project.project_dir = tmp_path
    return GenerationService(project, cache_dir=tmp_path / "cache")


class TestCalculateTileSpecs:
    """Test tile grid calculation."""

    def test_tile_count(self, gen_service):
        specs = gen_service.calculate_tile_specs()
        tiles = gen_service.project.tiles
        output = gen_service.project.output
        cols, rows = tiles.calculate_grid(output.width, output.height)
        assert len(specs) == cols * rows

    def test_tiles_cover_all_columns_and_rows(self, gen_service):
        specs = gen_service.calculate_tile_specs()
        cols = {s.col for s in specs}
        rows = {s.row for s in specs}
        assert cols == set(range(max(cols) + 1))
        assert rows == set(range(max(rows) + 1))

    def test_tile_offsets_non_negative(self, gen_service):
        specs = gen_service.calculate_tile_specs()
        for spec in specs:
            assert spec.x_offset >= 0
            assert spec.y_offset >= 0

    def test_tile_bboxes_within_region(self, gen_service):
        specs = gen_service.calculate_tile_specs()
        region = gen_service.project.region
        for spec in specs:
            assert spec.bbox.north <= region.north + 0.001
            assert spec.bbox.south >= region.south - 0.001
            assert spec.bbox.east <= region.east + 0.001
            assert spec.bbox.west >= region.west - 0.001

    def test_position_descriptions(self, gen_service):
        specs = gen_service.calculate_tile_specs()
        descs = {s.position_desc for s in specs}
        # Should have at least corner descriptions
        assert any("corner" in d or "center" in d or "edge" in d for d in descs)


class TestGetPositionDescription:
    """Test human-readable position descriptions."""

    def test_top_left_corner(self, gen_service):
        desc = gen_service._get_position_description(0, 0, 3, 3)
        assert desc == "top-left corner"

    def test_top_right_corner(self, gen_service):
        desc = gen_service._get_position_description(2, 0, 3, 3)
        assert desc == "top-right corner"

    def test_bottom_left_corner(self, gen_service):
        desc = gen_service._get_position_description(0, 2, 3, 3)
        assert desc == "bottom-left corner"

    def test_bottom_right_corner(self, gen_service):
        desc = gen_service._get_position_description(2, 2, 3, 3)
        assert desc == "bottom-right corner"

    def test_center(self, gen_service):
        desc = gen_service._get_position_description(1, 1, 3, 3)
        assert desc == "center"

    def test_top_edge(self, gen_service):
        desc = gen_service._get_position_description(1, 0, 3, 3)
        assert desc == "top edge"

    def test_left_edge(self, gen_service):
        desc = gen_service._get_position_description(0, 1, 3, 3)
        assert desc == "left edge"


class TestIsWaterTile:
    """Test water tile detection."""

    def test_uniform_blue_is_water(self, gen_service):
        # Uniform deep blue (R==G, B dominant) triggers water detection
        arr = np.full((64, 64, 3), [30, 30, 180], dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        assert gen_service._is_water_tile(img) == True

    def test_pure_blue_is_water(self, gen_service):
        arr = np.full((64, 64, 3), [0, 0, 200], dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        assert gen_service._is_water_tile(img) == True

    def test_green_image_is_not_water(self, gen_service):
        arr = np.full((64, 64, 3), [50, 180, 50], dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        assert gen_service._is_water_tile(img) == False

    def test_red_image_is_not_water(self, gen_service):
        arr = np.full((64, 64, 3), [200, 50, 50], dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        assert gen_service._is_water_tile(img) == False

    def test_mixed_image_not_water(self, gen_service, small_test_image):
        assert gen_service._is_water_tile(small_test_image) == False


class TestEstimateCost:
    """Test cost estimation."""

    def test_returns_required_keys(self, gen_service):
        cost = gen_service.estimate_cost()
        assert "tile_cost" in cost
        assert "seam_cost" in cost
        assert "total_cost" in cost

    def test_total_is_sum(self, gen_service):
        cost = gen_service.estimate_cost()
        expected = cost["tile_cost"] + cost["seam_cost"] + cost.get("landmark_cost", 0)
        assert cost["total_cost"] == pytest.approx(expected)

    def test_cost_positive(self, gen_service):
        cost = gen_service.estimate_cost()
        assert cost["total_cost"] > 0


class TestTileRenderSize:
    """Test tile render size calculation."""

    def test_square_bbox_produces_square(self, gen_service):
        bbox = BoundingBox(north=1.005, south=0.995, east=0.005, west=-0.005)
        w, h = gen_service._tile_render_size(bbox)
        # Should be close to square (latitude correction at equator is ~1)
        assert abs(w - h) < 50

    def test_fits_within_tile_size(self, gen_service):
        bbox = BoundingBox(north=41.0, south=40.0, east=-73.0, west=-74.0)
        w, h = gen_service._tile_render_size(bbox)
        tile_size = gen_service.project.tiles.size
        assert w <= tile_size
        assert h <= tile_size

    def test_minimum_dimension(self, gen_service):
        bbox = BoundingBox(north=40.001, south=40.0, east=-73.999, west=-74.0)
        w, h = gen_service._tile_render_size(bbox)
        assert w >= 1
        assert h >= 1


class TestApplyOrientationRotation:
    """Test orientation rotation."""

    def test_no_rotation(self, gen_service):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = gen_service._apply_orientation_rotation(img)
        assert result.size == (100, 100)

    def test_90_degree_rotation(self, tmp_path):
        from mapgen.models.project import CardinalDirection, StyleSettings

        bbox = BoundingBox(north=40.78, south=40.76, east=-73.96, west=-73.98)
        project = Project(
            name="test",
            region=bbox,
            output=OutputSettings(width=1024, height=1024, dpi=72),
            tiles=TileSettings(size=512, overlap=64),
            style=StyleSettings(orientation=CardinalDirection.EAST),
        )
        project.project_dir = tmp_path
        svc = GenerationService(project, cache_dir=tmp_path / "cache")

        img = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        result = svc._apply_orientation_rotation(img)
        # 90-degree rotation swaps dimensions
        assert result.size == (50, 100)
