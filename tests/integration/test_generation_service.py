"""Integration tests for GenerationService with mocked external services."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings
from mapgen.services.generation_service import GenerationService, TileResult
from mapgen.services.gemini_service import GenerationResult


def _make_test_image(w=512, h=512, color=(128, 128, 128, 255)):
    return Image.new("RGBA", (w, h), color)


@pytest.fixture
def mock_gemini():
    svc = MagicMock()
    svc.generate_base_tile.return_value = GenerationResult(
        image=_make_test_image(),
        prompt_used="test prompt",
        model="test-model",
        generation_time=1.0,
    )
    return svc


@pytest.fixture
def mock_satellite():
    svc = MagicMock()
    svc.fetch_satellite_imagery.return_value = _make_test_image()
    return svc


@pytest.fixture
def mock_osm():
    svc = MagicMock()
    osm_data = MagicMock()
    osm_data.has_data.return_value = True
    svc.fetch_region_data.return_value = osm_data
    return svc


@pytest.fixture
def gen_service_with_mocks(tmp_path, mock_gemini, mock_satellite, mock_osm):
    bbox = BoundingBox(north=40.78, south=40.76, east=-73.96, west=-73.98)
    project = Project(
        name="test",
        region=bbox,
        output=OutputSettings(width=1024, height=1024, dpi=72),
        tiles=TileSettings(size=512, overlap=64),
    )
    project.project_dir = tmp_path

    return GenerationService(
        project,
        gemini_service=mock_gemini,
        satellite_service=mock_satellite,
        osm_service=mock_osm,
        cache_dir=tmp_path / "cache",
    )


class TestGenerateTile:
    """Test single tile generation."""

    def test_calls_gemini(self, gen_service_with_mocks, mock_gemini):
        specs = gen_service_with_mocks.calculate_tile_specs()
        with patch.object(gen_service_with_mocks, "generate_tile_reference", return_value=_make_test_image()):
            result = gen_service_with_mocks.generate_tile(specs[0])

        assert result.generated_image is not None
        assert result.error is None
        mock_gemini.generate_base_tile.assert_called_once()

    def test_returns_error_on_failure(self, gen_service_with_mocks, mock_gemini):
        mock_gemini.generate_base_tile.side_effect = RuntimeError("API error")
        specs = gen_service_with_mocks.calculate_tile_specs()

        with patch.object(gen_service_with_mocks, "generate_tile_reference", return_value=_make_test_image()):
            result = gen_service_with_mocks.generate_tile(specs[0], max_retries=1)

        assert result.error is not None
        assert "API error" in result.error

    def test_retries_on_failure(self, gen_service_with_mocks, mock_gemini):
        mock_gemini.generate_base_tile.side_effect = [
            RuntimeError("fail 1"),
            GenerationResult(
                image=_make_test_image(),
                prompt_used="test",
                model="test",
                generation_time=1.0,
            ),
        ]
        specs = gen_service_with_mocks.calculate_tile_specs()

        with patch.object(gen_service_with_mocks, "generate_tile_reference", return_value=_make_test_image()):
            with patch("time.sleep"):  # Skip exponential backoff
                result = gen_service_with_mocks.generate_tile(specs[0], max_retries=3)

        assert result.generated_image is not None
        assert result.retries == 1

    def test_loads_from_cache(self, gen_service_with_mocks, mock_gemini):
        specs = gen_service_with_mocks.calculate_tile_specs()
        spec = specs[0]

        # Pre-populate cache
        cache_path = gen_service_with_mocks.cache_dir / "generated" / f"tile_{spec.col}_{spec.row}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _make_test_image(color=(255, 0, 0, 255)).save(cache_path)

        result = gen_service_with_mocks.generate_tile(spec)

        assert result.generated_image is not None
        # Should NOT have called Gemini (loaded from cache)
        mock_gemini.generate_base_tile.assert_not_called()


class TestAssembleTiles:
    """Test tile assembly."""

    def test_assembles_into_correct_size(self, gen_service_with_mocks):
        specs = gen_service_with_mocks.calculate_tile_specs()
        results = []
        for spec in specs:
            w, h = gen_service_with_mocks._tile_render_size(spec.bbox)
            results.append(TileResult(
                spec=spec,
                generated_image=_make_test_image(w, h),
            ))

        assembled = gen_service_with_mocks.assemble_tiles(results, apply_perspective=False)
        assert assembled is not None
        assert assembled.width == gen_service_with_mocks.project.output.width
        assert assembled.height == gen_service_with_mocks.project.output.height

    def test_returns_none_on_too_many_failures(self, gen_service_with_mocks):
        specs = gen_service_with_mocks.calculate_tile_specs()
        # All tiles failed
        results = [TileResult(spec=spec, error="Failed") for spec in specs]

        assembled = gen_service_with_mocks.assemble_tiles(results, apply_perspective=False)
        assert assembled is None

    def test_assembles_with_perspective(self, gen_service_with_mocks):
        specs = gen_service_with_mocks.calculate_tile_specs()
        results = []
        for spec in specs:
            w, h = gen_service_with_mocks._tile_render_size(spec.bbox)
            results.append(TileResult(
                spec=spec,
                generated_image=_make_test_image(w, h),
            ))

        assembled = gen_service_with_mocks.assemble_tiles(results, apply_perspective=True)
        assert assembled is not None
        # With perspective, height should be larger (margin added)
        assert assembled.height > gen_service_with_mocks.project.output.height


class TestProgressTracking:
    """Test generation progress callback."""

    def test_progress_callback_called(self, gen_service_with_mocks, mock_gemini):
        progress_updates = []

        def callback(progress):
            progress_updates.append(progress)

        with patch.object(gen_service_with_mocks, "generate_tile_reference", return_value=_make_test_image()):
            gen_service_with_mocks.generate_all_tiles(progress_callback=callback, max_retries=1)

        assert len(progress_updates) > 0
