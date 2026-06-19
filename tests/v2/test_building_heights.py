"""Real building heights flow from OSM tags into BuildingFootprint.height_px."""

import math

import pytest

from mapgen.services.osm_service import _parse_float, _parse_metric_height
from mapgen.v2.ingest import GeoFrame, SourceBuilding, SourceData
from mapgen.v2.plan.builder import _DEFAULT_BUILDING_HEIGHT_M, PlanBuilder
from mapgen.v2.types import CameraSpec, CanvasSpec, RegionBBox


def test_parse_metric_height_units():
    assert _parse_metric_height("30") == pytest.approx(30.0)
    assert _parse_metric_height("30 m") == pytest.approx(30.0)
    assert _parse_metric_height("98'") == pytest.approx(98 / 3.28084)
    assert _parse_metric_height("100 ft") == pytest.approx(100 / 3.28084)
    assert _parse_metric_height(None) is None
    assert _parse_metric_height(["12", "9"]) == pytest.approx(12.0)


def test_parse_float_tolerates_nan_and_lists():
    assert _parse_float(float("nan")) is None
    assert _parse_float("5") == pytest.approx(5.0)
    assert _parse_float(["3"]) == pytest.approx(3.0)
    assert _parse_float("garbage") is None


def _square(lon, lat, d=0.0005):
    return [(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d), (lon, lat)]


def _plan_with_buildings(buildings):
    region = RegionBBox(north=40.76, south=40.74, east=-73.98, west=-74.00)
    source = SourceData(region=region, buildings=buildings)
    builder = PlanBuilder(
        canvas=CanvasSpec(width_px=2000, height_px=2800),
        camera=CameraSpec(),
        distortion_strength=0.0,
    )
    return builder.build(source, frame=GeoFrame(region, target_aspect=2800 / 2000))


def test_taller_building_extrudes_higher():
    plan = _plan_with_buildings(
        [
            SourceBuilding(exterior=_square(-73.992, 40.748), height_m=20.0),
            SourceBuilding(exterior=_square(-73.990, 40.748), height_m=100.0),
        ]
    )
    assert len(plan.buildings) == 2
    short, tall = plan.buildings
    # Heights are jittered ±5%, so compare with margin -- 100 m must clearly
    # out-rise 20 m.
    assert tall.height_px > short.height_px * 2.0


def test_missing_height_uses_default():
    tagged = _plan_with_buildings(
        [SourceBuilding(exterior=_square(-73.992, 40.748), height_m=_DEFAULT_BUILDING_HEIGHT_M)]
    )
    untagged = _plan_with_buildings(
        [SourceBuilding(exterior=_square(-73.992, 40.748), height_m=None)]
    )
    # Same effective height (modulo jitter) -> within ~12% of each other.
    assert untagged.buildings[0].height_px == pytest.approx(
        tagged.buildings[0].height_px, rel=0.12
    )


def test_outlier_height_clamped():
    plan = _plan_with_buildings(
        [SourceBuilding(exterior=_square(-73.992, 40.748), height_m=5000.0)]
    )
    assert plan.buildings[0].height_px <= 2000 * 0.06 + 1e-6


def test_nan_height_treated_as_missing():
    plan = _plan_with_buildings(
        [SourceBuilding(exterior=_square(-73.992, 40.748), height_m=float("nan"))]
    )
    assert not math.isnan(plan.buildings[0].height_px)
    assert plan.buildings[0].height_px > 0
