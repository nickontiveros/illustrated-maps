"""Tests for scale-aware source generalization (mapgen/v2/generalize.py)."""

from mapgen.v2.generalize import generalize_source
from mapgen.v2.ingest import SourceData, SourcePolygon, SourceRoad
from mapgen.v2.types import GroundClass, RegionBBox, RoadClass


def _square(lon, lat, side_deg, name=None):
    return SourcePolygon(
        cls=GroundClass.WATER,
        exterior=[
            (lon, lat),
            (lon + side_deg, lat),
            (lon + side_deg, lat + side_deg),
            (lon, lat + side_deg),
        ],
        holes=[],
        name=name,
    )


# A clearly large region (> the city-scale generalization floor).
BIG = RegionBBox(north=36.0, south=32.0, east=-109.0, west=-113.0)


def test_tiny_water_dropped_big_kept():
    src = SourceData(
        region=BIG,
        ground=[
            _square(-112.0, 33.0, 0.05, name="Big Lake"),  # ~25 km2
            _square(-112.5, 33.5, 0.001, name="Boulin Tank"),  # ~0.009 km2
            _square(-112.2, 33.2, 0.001),  # tiny unnamed
        ],
    )
    generalize_source(src, BIG)
    water = [g for g in src.ground if g.cls is GroundClass.WATER]
    assert len(water) == 1 and water[0].name == "Big Lake"


def test_tank_names_never_labeled():
    src = SourceData(
        region=BIG,
        ground=[
            _square(-112.0, 33.0, 0.06, name="Apache Lake"),  # big, real
            _square(-111.0, 34.0, 0.06, name="Replacement Tank"),  # big but junk name
        ],
    )
    generalize_source(src, BIG)
    labels = [p.name for p in src.places if p.kind == "water"]
    assert "Apache Lake" in labels
    assert not any("Tank" in n for n in labels)


def test_fragmented_roads_merge_and_keep_ref():
    # Three connected stubs of one route -> one continuous polyline, ref kept.
    src = SourceData(
        region=BIG,
        roads=[
            SourceRoad(cls=RoadClass.MOTORWAY, coords=[(0, 0), (1, 1)], name="I-10", ref="I 10"),
            SourceRoad(cls=RoadClass.MOTORWAY, coords=[(1, 1), (2, 2)], name="I-10", ref="I 10"),
            SourceRoad(cls=RoadClass.MOTORWAY, coords=[(2, 2), (3, 3)], name="I-10", ref="I 10"),
        ],
    )
    generalize_source(src, BIG)
    motorways = [r for r in src.roads if r.cls is RoadClass.MOTORWAY]
    assert len(motorways) == 1
    assert len(motorways[0].coords) == 4  # 0,0 -> 1,1 -> 2,2 -> 3,3
    assert motorways[0].ref == "I 10"


def test_minor_and_unnamed_waterways_dropped():
    src = SourceData(
        region=BIG,
        roads=[
            # named major river, long -> kept
            SourceRoad(cls=RoadClass.RIVER, coords=[(-112.0, 33.0), (-112.0, 33.3)], name="Salt River"),
            # irrigation lateral -> dropped by name
            SourceRoad(cls=RoadClass.RIVER, coords=[(-112.0, 33.0), (-112.0, 33.3)], name="Highline Lateral"),
            # unnamed wash -> dropped (no name)
            SourceRoad(cls=RoadClass.RIVER, coords=[(-111.5, 33.0), (-111.5, 33.05)], name=None),
        ],
    )
    generalize_source(src, BIG)
    rivers = [r for r in src.roads if r.cls is RoadClass.RIVER]
    assert [r.name for r in rivers] == ["Salt River"]


def test_small_region_keeps_detail():
    # A city-scale region must not have its small ponds stripped.
    small = RegionBBox(north=40.78, south=40.70, east=-73.95, west=-74.02)
    src = SourceData(region=small, ground=[_square(-74.0, 40.75, 0.001, name="Pond")])
    generalize_source(src, small)
    assert sum(1 for g in src.ground if g.cls is GroundClass.WATER) == 1
