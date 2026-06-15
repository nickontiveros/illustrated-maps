import pytest

from mapgen.v2.plan import PlanBuilder, plan_to_svg
from mapgen.v2.plan.placement import has_overlaps
from mapgen.v2.types import AssetKind, CameraSpec, GroundClass, LabelKind, PlanDocument


@pytest.fixture
def plan(source, small_canvas) -> PlanDocument:
    builder = PlanBuilder(canvas=small_canvas, distortion_strength=0.4, seed=3)
    return builder.build(source, title="Test Town")


def test_plan_has_all_layer_types(plan: PlanDocument):
    assert plan.ground and plan.roads and plan.pois and plan.labels and plan.manifest
    assert plan.buildings
    assert plan.scatter  # park trees / bay boats / urban houses


def test_geometry_is_inside_canvas(plan: PlanDocument):
    w, h = plan.canvas.width_px, plan.canvas.height_px
    for road in plan.roads:
        for x, y in road.points:
            assert -w * 0.2 <= x <= w * 1.2
            assert -h * 0.2 <= y <= h * 1.2


def test_geometry_is_below_horizon(plan: PlanDocument):
    horizon = plan.camera.horizon_margin * plan.canvas.height_px
    for road in plan.roads:
        for _, y in road.points:
            assert y >= horizon - 1


def test_pois_do_not_overlap(plan: PlanDocument):
    assert not has_overlaps(plan.pois)


def test_pois_have_assets_in_manifest(plan: PlanDocument):
    asset_ids = {s.id for s in plan.manifest}
    for slot in plan.pois:
        assert slot.asset_id in asset_ids


def test_depth_increases_northward(plan: PlanDocument):
    """Roads near the top of the flat map should carry larger depth."""
    by_depth = sorted(plan.roads, key=lambda r: r.depth)
    near, far = by_depth[0], by_depth[-1]
    near_y = sum(p[1] for p in near.points) / len(near.points)
    far_y = sum(p[1] for p in far.points) / len(far.points)
    assert far_y < near_y  # far = closer to horizon = smaller y


def test_manifest_covers_ground_classes(plan: PlanDocument):
    texture_subjects = {s.subject for s in plan.manifest if s.kind == AssetKind.TEXTURE}
    ground_classes = {g.cls.value for g in plan.ground}
    assert ground_classes <= texture_subjects
    assert GroundClass.LAND.value in texture_subjects  # base plate texture


def test_labels_include_streets_and_title(plan: PlanDocument):
    kinds = {l.kind for l in plan.labels}
    assert LabelKind.TITLE in kinds
    assert LabelKind.POI in kinds
    street_texts = {l.text for l in plan.labels if l.kind == LabelKind.STREET}
    assert street_texts  # at least one named street labeled


def test_plan_roundtrip(plan: PlanDocument, tmp_path):
    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = PlanDocument.load(path)
    assert loaded == plan


def test_svg_preview(plan: PlanDocument):
    svg = plan_to_svg(plan)
    assert svg.startswith("<svg")
    assert "Test Town" in svg
    assert svg.count("<path") >= len(plan.roads)


def test_leader_lines_render_and_roundtrip(source, small_canvas, tmp_path):
    from mapgen.v2.ingest import SourceData, SourcePoi

    region = source.region
    midlat = (region.north + region.south) / 2
    midlon = (region.east + region.west) / 2
    pois = [
        SourcePoi(id="x", name="X", latitude=midlat, longitude=midlon, tier=2),
        # ~3 m away: no warp can separate them, must be leadered.
        SourcePoi(id="y", name="Y", latitude=midlat + 0.00003, longitude=midlon + 0.00001, tier=2),
        SourcePoi(
            id="z",
            name="Z",
            latitude=region.south + 0.25 * (region.north - region.south),
            longitude=region.west + 0.25 * (region.east - region.west),
            tier=2,
        ),
    ]
    plan = PlanBuilder(canvas=small_canvas).build(SourceData(region=region, pois=pois), title="T")
    offset = [s for s in plan.pois if s.offset]
    assert offset and all(s.leader_anchor is not None for s in offset)
    # Non-coincident slots carry no leader.
    assert all(s.leader_anchor is None for s in plan.pois if not s.offset)
    svg = plan_to_svg(plan)
    assert "<line" in svg and "<circle" in svg
    # New PoiSlot fields survive a save/load round-trip.
    path = tmp_path / "plan.json"
    plan.save(path)
    assert PlanDocument.load(path) == plan


def test_zero_distortion_keeps_relative_positions(source, small_canvas):
    flat = PlanBuilder(canvas=small_canvas, distortion_strength=0.0,
                       camera=CameraSpec(convergence=1.0, vertical_scale=1.0, horizon_margin=0.0))
    plan = flat.build(source, title="T")
    # With identity camera and no distortion, the lighthouse POI should sit
    # at its raw geographic position.
    lighthouse = next(p for p in plan.pois if p.id == "lighthouse")
    expected_x = (lighthouse.longitude - source.region.west) / source.region.width_deg * small_canvas.width_px
    # Anchor may have been nudged by collision resolution; allow slack.
    assert lighthouse.anchor[0] == pytest.approx(expected_x, abs=small_canvas.width_px * 0.15)
