"""from_osm_data: the GeoDataFrame -> SourceData conversion seam.

This is the exact boundary a live `mapgen v2 plan` run crosses, exercised
here with synthetic GeoDataFrames shaped like OSMService output.
"""

import pytest

gpd = pytest.importorskip("geopandas")

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from mapgen.services.osm_service import OSMData
from mapgen.v2.ingest import from_osm_data
from mapgen.v2.plan import PlanBuilder
from mapgen.v2.types import GroundClass, RoadClass


@pytest.fixture
def osm_data(region):
    lon0, lat0 = region.west, region.south
    roads = gpd.GeoDataFrame(
        {
            "highway": ["primary", "residential", ["motorway", "trunk"], "footway"],
            "name": ["Main Street", None, ["Coast Highway"], "Trail"],
            "geometry": [
                LineString([(lon0 + 0.01, lat0 + 0.01), (lon0 + 0.05, lat0 + 0.05)]),
                LineString([(lon0 + 0.02, lat0 + 0.02), (lon0 + 0.03, lat0 + 0.02)]),
                MultiLineString(
                    [
                        [(lon0 + 0.01, lat0 + 0.01), (lon0 + 0.01, lat0 + 0.08)],
                        [(lon0 + 0.01, lat0 + 0.08), (lon0 + 0.02, lat0 + 0.09)],
                    ]
                ),
                LineString([(lon0 + 0.04, lat0 + 0.04), (lon0 + 0.05, lat0 + 0.04)]),
            ],
        }
    )
    water = gpd.GeoDataFrame(
        {
            "name": ["East Bay", None],
            "geometry": [
                Polygon(
                    [(lon0 + 0.07, lat0), (lon0 + 0.1, lat0), (lon0 + 0.1, lat0 + 0.1), (lon0 + 0.07, lat0 + 0.1)]
                ),
                MultiPolygon(
                    [
                        Polygon(
                            [(lon0 + 0.01, lat0 + 0.06), (lon0 + 0.02, lat0 + 0.06), (lon0 + 0.02, lat0 + 0.07), (lon0 + 0.01, lat0 + 0.07)]
                        )
                    ]
                ),
            ],
        }
    )
    parks = gpd.GeoDataFrame(
        {
            "name": ["Town Park"],
            "geometry": [
                Polygon(
                    [(lon0 + 0.03, lat0 + 0.05), (lon0 + 0.05, lat0 + 0.05), (lon0 + 0.05, lat0 + 0.07), (lon0 + 0.03, lat0 + 0.07)]
                )
            ],
        }
    )
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [(lon0 + 0.031, lat0 + 0.031), (lon0 + 0.033, lat0 + 0.031), (lon0 + 0.033, lat0 + 0.033), (lon0 + 0.031, lat0 + 0.033)]
                )
            ],
        }
    )
    return OSMData(roads=roads, water=water, parks=parks, buildings=buildings)


def test_road_classes_and_names(osm_data, region):
    source = from_osm_data(osm_data, region)
    by_name = {r.name: r for r in source.roads if r.name}
    assert by_name["Main Street"].cls == RoadClass.PRIMARY
    # List-valued OSM tags are unwrapped to their first entry.
    assert by_name["Coast Highway"].cls == RoadClass.MOTORWAY
    classes = {r.cls for r in source.roads}
    assert RoadClass.LOCAL in classes  # residential
    assert RoadClass.PATH in classes  # footway


def test_multilinestring_explodes_to_parts(osm_data, region):
    source = from_osm_data(osm_data, region)
    coast = [r for r in source.roads if r.name == "Coast Highway"]
    assert len(coast) == 2


def test_polygons_with_classes_and_names(osm_data, region):
    source = from_osm_data(osm_data, region)
    water = [g for g in source.ground if g.cls == GroundClass.WATER]
    parks = [g for g in source.ground if g.cls == GroundClass.PARK]
    assert len(water) == 2  # Polygon + exploded MultiPolygon
    assert len(parks) == 1
    assert any(g.name == "East Bay" for g in water)
    assert len(source.buildings) == 1


def test_converted_data_builds_a_plan(osm_data, region, small_canvas):
    source = from_osm_data(osm_data, region)
    plan = PlanBuilder(canvas=small_canvas).build(source, title="OSM Town")
    assert plan.roads
    assert any(g.cls == GroundClass.WATER for g in plan.ground)
    assert plan.buildings
