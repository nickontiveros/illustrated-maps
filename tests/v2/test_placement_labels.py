from mapgen.v2.plan.labels import plan_labels
from mapgen.v2.plan.placement import (
    assign_leader_lines,
    has_overlaps,
    resolve_poi_overlaps,
    sized_slot,
    tier_demand,
)
from mapgen.v2.types import CanvasSpec, LabelKind, PoiSlot, RoadClass, RoadPath


def _slot(id_: str, x: float, y: float, tier: int = 2) -> PoiSlot:
    base = PoiSlot(id=id_, name=id_, anchor=(x, y), width_px=0, height_px=0, tier=tier)
    return sized_slot(base, canvas_width_px=1000, depth_scale=1.0)


def test_sized_slot_tiers():
    hero = _slot("a", 0, 0, tier=1)
    minor = _slot("b", 0, 0, tier=3)
    assert hero.width_px > minor.width_px
    assert hero.height_px > hero.width_px  # sprite aspect > 1


def test_overlapping_pois_get_separated():
    slots = [_slot("a", 500, 700, 1), _slot("b", 505, 705, 2), _slot("c", 510, 695, 3)]
    assert has_overlaps(slots)
    resolved = resolve_poi_overlaps(slots, 1000, 1400)
    assert not has_overlaps(resolved)


def test_resolution_keeps_slots_on_canvas():
    slots = [_slot("a", 10, 20, 1), _slot("b", 15, 25, 1)]
    resolved = resolve_poi_overlaps(slots, 1000, 1400)
    for s in resolved:
        assert s.anchor[0] - s.width_px / 2 >= -1
        assert s.anchor[0] + s.width_px / 2 <= 1001
        assert s.anchor[1] - s.height_px >= -1
        assert s.anchor[1] <= 1401


def test_higher_tier_moves_less():
    a, b = _slot("hero", 500, 700, 1), _slot("minor", 504, 700, 3)
    resolved = resolve_poi_overlaps([a, b], 1000, 1400)
    hero_shift = abs(resolved[0].anchor[0] - 500)
    minor_shift = abs(resolved[1].anchor[0] - 504)
    assert hero_shift < minor_shift


def test_tier_demand_orders_by_tier():
    assert tier_demand(1) > tier_demand(2) > tier_demand(3)
    assert tier_demand(2, 0.5) == tier_demand(2) * 0.5


def test_assign_leader_lines_flags_displaced_only():
    # a and b were displaced far from their true ground point by the overlap
    # solver (the warp could not place them honestly); c sits on its anchor.
    a, b, c = _slot("a", 500, 700, 2), _slot("b", 500, 700, 2), _slot("c", 5000, 700, 2)
    a.leader_anchor, a.anchor = (500, 700), (200, 700)  # displaced 300px
    b.leader_anchor, b.anchor = (500, 700), (800, 700)  # displaced 300px
    c.leader_anchor, c.anchor = (5000, 700), (5000, 700)  # not displaced

    out = {s.id: s for s in assign_leader_lines([a, b, c], leader_threshold_px=100)}
    assert out["a"].offset and out["b"].offset
    assert out["a"].leader_anchor == (500, 700)
    assert not out["c"].offset and out["c"].leader_anchor is None


def test_uncross_never_shoves_sprite_onto_a_nonoffset_neighbour():
    # Two offset sprites with crossing connectors; swapping them to uncross
    # would drop the larger (A, tier 1) onto a non-offset neighbour C. The
    # uncross guard must check ALL slots, not just the offset ones, so the
    # swap is reverted and no sprite overlap is introduced.
    a = _slot("a", 150, 600, 1)
    a.leader_anchor = (620, 300)
    b = _slot("b", 650, 600, 3)
    b.leader_anchor = (180, 300)
    c = _slot("c", 650, 490, 2)  # non-offset, sits above B's position
    c.leader_anchor = (650, 490)

    assert not has_overlaps([a, b, c])  # clean before
    out = assign_leader_lines([a, b, c], leader_threshold_px=100)
    assert out[0].offset and out[1].offset and not out[2].offset
    assert not has_overlaps(out)  # the uncross must not introduce an overlap


def test_shield_labels_generated_and_cleaned():
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    roads = [
        RoadPath(cls=RoadClass.MOTORWAY, points=[(100, 200), (900, 200)], width_px=8, ref="I 10;AZ 51"),
        RoadPath(cls=RoadClass.PRIMARY, points=[(100, 500), (900, 500)], width_px=5, ref="Future US 60"),
        RoadPath(cls=RoadClass.LOCAL, points=[(100, 800), (900, 800)], width_px=3, ref="AZ 99"),
    ]
    labels = plan_labels(canvas, roads, [], [], [], title="T")
    shields = {l.text for l in labels if l.kind is LabelKind.SHIELD}
    assert "I 10" in shields  # ";"-joined ref reduced to the primary route
    assert "US 60" in shields  # "Future " qualifier stripped
    assert not any(s == "AZ 99" for s in shields)  # LOCAL roads carry no shield


def test_plan_labels_includes_title_and_pois():
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    pois = [_slot("Lighthouse", 800, 400, 1)]
    labels = plan_labels(canvas, [], pois, [], [], title="Test Town")
    kinds = {l.kind for l in labels}
    assert LabelKind.TITLE in kinds
    assert LabelKind.POI in kinds


def test_street_label_requires_named_long_road():
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    long_named = RoadPath(
        cls=RoadClass.PRIMARY,
        name="Main Street",
        points=[(100.0, 1000.0), (900.0, 1000.0)],
        width_px=10,
    )
    short_named = RoadPath(
        cls=RoadClass.PRIMARY, name="Tiny", points=[(0.0, 0.0), (10.0, 0.0)], width_px=10
    )
    unnamed = RoadPath(cls=RoadClass.PRIMARY, points=[(0.0, 500.0), (999.0, 500.0)], width_px=10)
    labels = plan_labels(canvas, [long_named, short_named, unnamed], [], [], [], title="T")
    street_labels = [l for l in labels if l.kind == LabelKind.STREET]
    assert [l.text for l in street_labels] == ["Main Street"]
    # Baseline rides the road.
    assert all(abs(p[1] - 1000.0) < 1 for p in street_labels[0].baseline)


def test_greedy_layout_drops_colliding_labels():
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    districts = [("Alpha", (500.0, 500.0), None), ("Beta", (505.0, 502.0), None)]
    labels = plan_labels(canvas, [], [], districts, [], title="T")
    district_labels = [l for l in labels if l.kind == LabelKind.DISTRICT]
    assert len(district_labels) == 1
