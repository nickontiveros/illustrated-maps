"""Tests for mapgen.services.road_style_service."""

import numpy as np
import pytest

from mapgen.models.road_style import RoadStyleSettings
from mapgen.services.road_style_service import RoadStyleService


@pytest.fixture
def default_service():
    return RoadStyleService()


@pytest.fixture
def custom_service():
    settings = RoadStyleSettings(
        enabled=True,
        motorway_exaggeration=30.0,
        wobble_amount=2.0,
    )
    return RoadStyleService(settings=settings)


# ---------------------------------------------------------------------------
# Lookup dicts
# ---------------------------------------------------------------------------

class TestLookupDicts:
    def test_highway_to_class_is_dict(self):
        assert isinstance(RoadStyleService.HIGHWAY_TO_CLASS, dict)
        assert len(RoadStyleService.HIGHWAY_TO_CLASS) > 0

    def test_road_class_map_is_dict(self):
        assert isinstance(RoadStyleService.ROAD_CLASS_MAP, dict)
        assert len(RoadStyleService.ROAD_CLASS_MAP) > 0


# ---------------------------------------------------------------------------
# get_exaggeration
# ---------------------------------------------------------------------------

class TestGetExaggeration:
    def test_motorway_exaggeration(self, default_service):
        val = default_service.get_exaggeration("motorway")
        assert isinstance(val, float)
        assert val > 0

    def test_unknown_class_returns_default(self, default_service):
        val = default_service.get_exaggeration("unknown_road_type")
        assert isinstance(val, float)

    def test_custom_motorway_exaggeration(self, custom_service):
        val = custom_service.get_exaggeration("motorway")
        assert val == 30.0


# ---------------------------------------------------------------------------
# get_fill_color
# ---------------------------------------------------------------------------

class TestGetFillColor:
    def test_returns_hex_string(self, default_service):
        color = default_service.get_fill_color("motorway")
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_unknown_road_class(self, default_service):
        color = default_service.get_fill_color("made_up_class")
        assert isinstance(color, str)


# ---------------------------------------------------------------------------
# _classify_road
# ---------------------------------------------------------------------------

class TestClassifyRoad:
    def test_motorway_classification(self, default_service):
        row = {"highway": "motorway"}
        result = default_service._classify_road(row)
        assert isinstance(result, str)

    def test_primary_classification(self, default_service):
        row = {"highway": "primary"}
        result = default_service._classify_road(row)
        assert isinstance(result, str)

    def test_residential_classification(self, default_service):
        row = {"highway": "residential"}
        result = default_service._classify_road(row)
        assert isinstance(result, str)

    def test_unknown_highway_tag(self, default_service):
        row = {"highway": "some_new_type"}
        result = default_service._classify_road(row)
        assert isinstance(result, str)

    def test_missing_highway_key(self, default_service):
        row = {"road_class": "primary"}
        result = default_service._classify_road(row)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _apply_wobble
# ---------------------------------------------------------------------------

class TestApplyWobble:
    def test_zero_amplitude_preserves_points(self, default_service):
        points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        result = default_service._apply_wobble(points, amplitude=0.0, frequency=0.02, seed=42)
        np.testing.assert_array_almost_equal(points, result)

    def test_nonzero_amplitude_displaces_inner_points(self, default_service):
        points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        result = default_service._apply_wobble(points, amplitude=5.0, frequency=0.02, seed=42)
        # Endpoints should be preserved
        np.testing.assert_array_almost_equal(result[0], points[0])
        np.testing.assert_array_almost_equal(result[-1], points[-1])
        # Inner point should differ
        assert not np.allclose(result[1], points[1])

    def test_endpoints_preserved(self, default_service):
        points = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 3.0], [15.0, 8.0]])
        result = default_service._apply_wobble(points, amplitude=3.0, frequency=0.05, seed=7)
        np.testing.assert_array_almost_equal(result[0], points[0])
        np.testing.assert_array_almost_equal(result[-1], points[-1])

    def test_deterministic_with_same_seed(self, default_service):
        points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
        r1 = default_service._apply_wobble(points, amplitude=2.0, frequency=0.02, seed=99)
        r2 = default_service._apply_wobble(points, amplitude=2.0, frequency=0.02, seed=99)
        np.testing.assert_array_equal(r1, r2)
