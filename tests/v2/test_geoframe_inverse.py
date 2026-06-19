"""GeoFrame.from_normalized must invert to_normalized exactly.

This round-trip is the linchpin of raster (satellite/hillshade) registration:
without it, sampling a raster into the rotated, aspect-extended frame misaligns
by the rotation angle.
"""

import pytest

from mapgen.v2.ingest import GeoFrame
from mapgen.v2.types import RegionBBox


@pytest.fixture
def region() -> RegionBBox:
    # Lower Manhattan-ish box, non-square so the aspect extension kicks in.
    return RegionBBox(north=40.78, south=40.70, east=-73.95, west=-74.02)


@pytest.mark.parametrize("rotation", [0.0, 12.0, 340.0, 45.0])
@pytest.mark.parametrize("aspect", [None, 1.4, 0.8])
def test_normalized_round_trip(region: RegionBBox, rotation: float, aspect):
    frame = GeoFrame(region, rotation_deg=rotation, target_aspect=aspect)
    for lon in (region.west, -73.99, region.east):
        for lat in (region.south, 40.74, region.north):
            uv = frame.to_normalized((lon, lat))
            lon2, lat2 = frame.from_normalized(uv)
            assert lon2 == pytest.approx(lon, abs=1e-6)
            assert lat2 == pytest.approx(lat, abs=1e-6)


def test_from_normalized_corners_match_extent(region: RegionBBox):
    """uv corners (0,0) and (1,1) round-trip back through to_normalized."""
    frame = GeoFrame(region, rotation_deg=20.0, target_aspect=1.4)
    for uv in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]:
        back = frame.to_normalized(frame.from_normalized(uv))
        assert back[0] == pytest.approx(uv[0], abs=1e-9)
        assert back[1] == pytest.approx(uv[1], abs=1e-9)


def test_km_per_flat_unit_positive(region: RegionBBox):
    frame = GeoFrame(region, target_aspect=1.4)
    kx, ky = frame.km_per_flat_unit(1000.0, 1400.0)
    assert kx > 0 and ky > 0
    # Multiplying back by the flat dimensions recovers the frame's km extent.
    assert kx * 1000.0 == pytest.approx(frame._x1 - frame._x0)
    assert ky * 1400.0 == pytest.approx(frame._y1 - frame._y0)
