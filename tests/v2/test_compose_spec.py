"""Tests for the CompositionSpec layout document (mapgen/v2/compose_spec.py).

The load-bearing guarantee: an absent or all-auto spec must reproduce the
pre-spec plan exactly, so adopting the spec is risk-free.
"""

import pytest

from mapgen.v2.compose_spec import (
    COMPOSITION_FILENAME,
    CompositionSpec,
    FeatureSpec,
    LayerSelect,
    PoiOverride,
    RoadOverride,
    WarpRegion,
    WarpSpec,
)
from mapgen.v2.ingest import SourceData, SourcePoi, SourceRoad, assign_feature_ids
from mapgen.v2.plan import PlanBuilder
from mapgen.v2.plan.builder import PlanBuilder as _PB
from mapgen.v2.types import CanvasSpec, RegionBBox, RoadClass


def test_spec_round_trips(tmp_path):
    spec = CompositionSpec()
    path = tmp_path / COMPOSITION_FILENAME
    spec.save(path)
    assert CompositionSpec.load(path) == spec
    assert CompositionSpec.load_or_default(tmp_path) == spec


def test_load_or_default_when_absent(tmp_path):
    # No composition.json on disk -> an all-auto default, not an error.
    assert CompositionSpec.load_or_default(tmp_path) == CompositionSpec()


def test_layer_select_visibility():
    sel = LayerSelect(exclude=["a"], include=["b"])
    assert sel.visible("a", auto_visible=True) is False  # excluded wins
    assert sel.visible("b", auto_visible=False) is True  # included wins
    assert sel.visible("c", auto_visible=True) is True  # falls back to heuristic
    assert sel.visible("c", auto_visible=False) is False
    assert LayerSelect(default="none").visible("x", auto_visible=True) is False
    assert LayerSelect(default="all").visible("x", auto_visible=False) is True


def test_all_auto_spec_reproduces_plan_exactly(source):
    canvas = CanvasSpec(width_px=1000, height_px=1414, dpi=72)
    baseline = PlanBuilder(canvas=canvas).build(source)
    with_spec = PlanBuilder(canvas=canvas, spec=CompositionSpec()).build(source)
    assert with_spec.model_dump_json() == baseline.model_dump_json()


def test_magnify_weight_round_trips():
    # A band of width w with weight that targets magnify m should stretch its
    # interior by ~m: (1+weight)/(1+weight*w) == m.
    for w, m in [(0.2, 2.0), (0.3, 1.5), (0.1, 3.0)]:
        weight = _PB._magnify_weight(w, m)
        got = (1 + weight) / (1 + weight * w)
        assert got == pytest.approx(m, rel=1e-6)
    assert _PB._magnify_weight(0.2, 1.0) == 0.0  # no-op
    assert _PB._magnify_weight(0.9, 2.0) == 1000.0  # too wide -> saturate


def _two_poi_region():
    region = RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)
    lon = lambda f: region.west + f * (region.east - region.west)  # noqa: E731
    lat = lambda f: region.south + f * (region.north - region.south)  # noqa: E731
    pois = [
        SourcePoi(id="a", name="A", latitude=lat(0.42), longitude=lon(0.5), tier=3),
        SourcePoi(id="b", name="B", latitude=lat(0.58), longitude=lon(0.5), tier=3),
    ]
    return SourceData(region=region, pois=pois)


def _sep(plan):
    a = next(s for s in plan.pois if s.id == "a").anchor
    b = next(s for s in plan.pois if s.id == "b").anchor
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def test_warp_manual_region_magnifies_interior():
    src = _two_poi_region()
    canvas = CanvasSpec(width_px=1000, height_px=1414, dpi=72)
    off = PlanBuilder(
        canvas=canvas, spec=CompositionSpec(warp=WarpSpec(mode="off"))
    ).build(src)
    region = WarpRegion(id="r", bounds=(0.3, 0.3, 0.7, 0.7), magnify=2.5)
    manual = PlanBuilder(
        canvas=canvas,
        spec=CompositionSpec(warp=WarpSpec(mode="manual", regions=[region])),
    ).build(src)
    # Both POIs sit inside the magnified region, so they spread apart vs. the
    # unwarped layout.
    assert _sep(manual) > _sep(off) * 1.25


def test_warp_off_is_identity():
    src = _two_poi_region()
    canvas = CanvasSpec(width_px=1000, height_px=1414, dpi=72)
    off = PlanBuilder(
        canvas=canvas, spec=CompositionSpec(warp=WarpSpec(mode="off"))
    ).build(src)
    fit = off.provenance["warp_fit"]
    assert fit["coincident_count"] == 0  # well-separated POIs, no leaders


# --- POI overrides ---------------------------------------------------------------

CANVAS = CanvasSpec(width_px=1000, height_px=1414, dpi=72)


def _one_poi():
    region = RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)
    poi = SourcePoi(id="p", name="P", latitude=40.75, longitude=-74.0, tier=3)
    return SourceData(region=region, pois=[poi])


def _poi(plan, pid="p"):
    return next(s for s in plan.pois if s.id == pid)


def test_poi_size_override_scales_sprite():
    src = _one_poi()
    base = PlanBuilder(canvas=CANVAS).build(src)
    spec = CompositionSpec(pois={"p": PoiOverride(size=1.5)})
    bigger = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    assert _poi(bigger).width_px == pytest.approx(_poi(base).width_px * 1.5, rel=1e-6)


def test_poi_tier_override_changes_size_and_tier():
    src = _one_poi()
    spec = CompositionSpec(pois={"p": PoiOverride(tier=1)})
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    base = PlanBuilder(canvas=CANVAS).build(src)
    assert _poi(plan).tier == 1
    assert _poi(plan).width_px > _poi(base).width_px  # tier 1 wider than tier 3


def test_poi_offset_uv_nudges_anchor():
    src = _one_poi()
    base = PlanBuilder(canvas=CANVAS).build(src)
    spec = CompositionSpec(pois={"p": PoiOverride(offset_uv=(0.1, 0.0))})
    moved = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    assert _poi(moved).anchor[0] > _poi(base).anchor[0] + 50  # shifted right


def test_poi_leader_suppress_clears_connector():
    # Two ~3m-apart POIs normally both get leaders; suppress one.
    region = RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)
    pois = [
        SourcePoi(id="a", name="A", latitude=40.76000, longitude=-74.0, tier=2),
        SourcePoi(id="b", name="B", latitude=40.76003, longitude=-74.00001, tier=2),
    ]
    src = SourceData(region=region, pois=pois)
    spec = CompositionSpec(pois={"a": PoiOverride(leader="suppress")})
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    assert _poi(plan, "a").offset is False and _poi(plan, "a").leader_anchor is None


def test_poi_leader_force_adds_connector():
    src = _one_poi()  # isolated POI -> no leader by default
    base = PlanBuilder(canvas=CANVAS).build(src)
    assert _poi(base).offset is False
    spec = CompositionSpec(pois={"p": PoiOverride(leader="force")})
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    assert _poi(plan).offset is True and _poi(plan).leader_anchor is not None


# --- Feature ids + selection -----------------------------------------------------

REGION = RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)


def test_assign_feature_ids_deterministic_and_unique():
    def make():
        return SourceData(
            region=REGION,
            roads=[
                SourceRoad(cls=RoadClass.PRIMARY, ref="I 10", coords=[(-74.0, 40.75), (-73.96, 40.75)]),
                SourceRoad(cls=RoadClass.PRIMARY, ref="I 10", coords=[(-74.0, 40.75), (-73.96, 40.75)]),
            ],
        )

    s1, s2 = make(), make()
    assign_feature_ids(s1)
    assign_feature_ids(s2)
    assert [r.id for r in s1.roads] == [r.id for r in s2.roads]  # stable
    assert s1.roads[0].id != s1.roads[1].id  # identical geometry de-duped
    assert all(r.id.startswith("road:") for r in s1.roads)


def test_assign_feature_ids_idempotent():
    src = SourceData(region=REGION, roads=[SourceRoad(cls=RoadClass.PRIMARY, ref="I 10", coords=[(-74.0, 40.75), (-73.96, 40.75)])])
    assign_feature_ids(src)
    first = src.roads[0].id
    assign_feature_ids(src)  # second pass must not change a set id
    assert src.roads[0].id == first


def test_plan_roads_carry_ids(source):
    plan = PlanBuilder(canvas=CANVAS).build(source)
    assert plan.roads and all(r.id for r in plan.roads)
    assert plan.ground and all(g.id for g in plan.ground)


def test_feature_exclude_hides_road(source):
    base = PlanBuilder(canvas=CANVAS).build(source)
    rid = next(r.id for r in base.roads if r.cls is not RoadClass.RIVER)
    spec = CompositionSpec(features=FeatureSpec(roads=LayerSelect(exclude=[rid])))
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(source)
    assert all(r.id != rid for r in plan.roads)


def test_rivers_are_a_separate_selection_layer(source):
    # Rivers are governed by features.rivers, not features.roads.
    base = PlanBuilder(canvas=CANVAS).build(source)
    river_id = next(r.id for r in base.roads if r.cls is RoadClass.RIVER)
    # Excluding under `roads` leaves the river alone...
    via_roads = PlanBuilder(
        canvas=CANVAS,
        spec=CompositionSpec(features=FeatureSpec(roads=LayerSelect(exclude=[river_id]))),
    ).build(source)
    assert any(r.id == river_id for r in via_roads.roads)
    # ...but excluding under `rivers` drops it.
    via_rivers = PlanBuilder(
        canvas=CANVAS,
        spec=CompositionSpec(features=FeatureSpec(rivers=LayerSelect(exclude=[river_id]))),
    ).build(source)
    assert all(r.id != river_id for r in via_rivers.roads)


def test_feature_exclude_hides_poi():
    src = _one_poi()
    base = PlanBuilder(canvas=CANVAS).build(src)
    assert any(s.id == "p" for s in base.pois)
    spec = CompositionSpec(features=FeatureSpec(pois=LayerSelect(exclude=["p"])))
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    assert all(s.id != "p" for s in plan.pois)


# --- Road routing ----------------------------------------------------------------

def test_road_routing_hidden_drops_road(source):
    base = PlanBuilder(canvas=CANVAS).build(source)
    rid = base.roads[0].id
    spec = CompositionSpec(roads={rid: RoadOverride(treatment="hidden")})
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(source)
    assert all(r.id != rid for r in plan.roads)


def test_road_reshape_follows_straight_polyline():
    road = SourceRoad(
        cls=RoadClass.PRIMARY, ref="I 10",
        coords=[(-74.0, 40.72), (-73.97, 40.78)], id="road:test",
    )
    src = SourceData(region=REGION, roads=[road])
    spec = CompositionSpec(
        roads={"road:test": RoadOverride(treatment="straight", reshape=[(0.2, 0.5), (0.8, 0.5)])}
    )
    plan = PlanBuilder(canvas=CANVAS, spec=spec).build(src)
    pts = next(r.points for r in plan.roads if r.id == "road:test")
    ys = [p[1] for p in pts]
    assert max(ys) - min(ys) < 2.0  # horizontal reshape -> horizontal road
