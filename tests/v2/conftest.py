"""Shared fixtures: a synthetic coastal town as SourceData."""

import pytest

from mapgen.v2.ingest import SourceData, SourcePlace, SourcePoi, SourcePolygon, SourceRoad
from mapgen.v2.types import CanvasSpec, GroundClass, RegionBBox, RoadClass


@pytest.fixture
def region() -> RegionBBox:
    return RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)


@pytest.fixture
def small_canvas() -> CanvasSpec:
    # Small canvas keeps tests fast; geometry logic is scale-free.
    return CanvasSpec(width_px=1000, height_px=1414, dpi=72)


@pytest.fixture
def source(region: RegionBBox) -> SourceData:
    """A fictional coastal town: bay to the east, park, river, roads, POIs."""
    west, east = region.west, region.east
    south, north = region.south, region.north

    def lon(f: float) -> float:
        return west + f * (east - west)

    def lat(f: float) -> float:
        return south + f * (north - south)

    return SourceData(
        region=region,
        roads=[
            SourceRoad(
                cls=RoadClass.MOTORWAY,
                name="Coast Highway",
                coords=[(lon(0.1), lat(0.05)), (lon(0.15), lat(0.4)), (lon(0.1), lat(0.95))],
            ),
            SourceRoad(
                cls=RoadClass.PRIMARY,
                name="Main Street",
                coords=[(lon(0.1), lat(0.5)), (lon(0.45), lat(0.52)), (lon(0.7), lat(0.5))],
            ),
            SourceRoad(
                cls=RoadClass.SECONDARY,
                name="Harbor Road",
                coords=[(lon(0.4), lat(0.2)), (lon(0.5), lat(0.45)), (lon(0.7), lat(0.6))],
            ),
            SourceRoad(
                cls=RoadClass.LOCAL,
                coords=[(lon(0.3), lat(0.6)), (lon(0.35), lat(0.75))],
            ),
            SourceRoad(
                cls=RoadClass.RIVER,
                name="Silver River",
                coords=[(lon(0.0), lat(0.8)), (lon(0.4), lat(0.7)), (lon(0.75), lat(0.55))],
            ),
        ],
        ground=[
            SourcePolygon(
                cls=GroundClass.WATER,
                name="East Bay",
                exterior=[
                    (lon(0.75), lat(0.0)),
                    (lon(1.0), lat(0.0)),
                    (lon(1.0), lat(1.0)),
                    (lon(0.75), lat(1.0)),
                    (lon(0.7), lat(0.5)),
                ],
            ),
            SourcePolygon(
                cls=GroundClass.PARK,
                name="Town Park",
                exterior=[
                    (lon(0.2), lat(0.55)),
                    (lon(0.4), lat(0.55)),
                    (lon(0.4), lat(0.72)),
                    (lon(0.2), lat(0.72)),
                ],
            ),
            SourcePolygon(
                cls=GroundClass.URBAN,
                exterior=[
                    (lon(0.1), lat(0.3)),
                    (lon(0.6), lat(0.3)),
                    (lon(0.6), lat(0.5)),
                    (lon(0.1), lat(0.5)),
                ],
            ),
        ],
        buildings=[
            [(lon(0.30), lat(0.40)), (lon(0.33), lat(0.40)), (lon(0.33), lat(0.43)), (lon(0.30), lat(0.43))],
            [(lon(0.45), lat(0.35)), (lon(0.49), lat(0.35)), (lon(0.49), lat(0.38)), (lon(0.45), lat(0.38))],
        ],
        pois=[
            SourcePoi(id="lighthouse", name="Old Lighthouse", latitude=lat(0.25), longitude=lon(0.72), tier=1),
            SourcePoi(id="museum", name="Maritime Museum", latitude=lat(0.45), longitude=lon(0.5), tier=2),
            SourcePoi(id="market", name="Fish Market", latitude=lat(0.47), longitude=lon(0.52), tier=3),
        ],
        places=[
            SourcePlace(name="Old Town", latitude=lat(0.4), longitude=lon(0.35), kind="district"),
            SourcePlace(name="East Bay", latitude=lat(0.5), longitude=lon(0.87), kind="water"),
        ],
    )
