"""Tests for the generalization fixes from the Arizona case study
(V2_GENERALIZATION.md): geo frame, feature types, ink budget, adaptive
warp, label guarantees, scene vocabulary, retitle, provenance, repaint
tone/invention guards, and palette-shifted-key matting."""

import math

import numpy as np
import pytest
from PIL import Image

from mapgen.v2.ingest import (
    GeoFrame,
    SourceData,
    SourcePoi,
    SourceRoad,
    auto_rotation,
)
from mapgen.v2.plan import PlanBuilder
from mapgen.v2.plan.stylize import prune_roads, road_width_px
from mapgen.v2.types import (
    CanvasSpec,
    GroundClass,
    LabelKind,
    RegionBBox,
    RoadClass,
    ScatterKind,
    StyleSpec,
)


# --- GeoFrame: metric mapping, rotation, aspect extension --------------------


class TestGeoFrame:
    def test_cos_latitude_correction(self):
        # A square-in-km region at 60N is half as wide in degrees as at the
        # equator; the frame must map both to the same metric proportions.
        region = RegionBBox(north=60.5, south=59.5, east=10.0, west=8.0)
        frame = GeoFrame(region)
        u0, v0 = frame.to_normalized((8.0, 60.5))
        u1, v1 = frame.to_normalized((10.0, 59.5))
        assert (u0, v0) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert (u1, v1) == pytest.approx((1.0, 1.0), abs=1e-9)
        # Metric extents: 2 deg lon at 60N ~ 111.2 km; 1 deg lat ~ 110.6 km.
        assert frame._x1 - frame._x0 == pytest.approx(2 * 111.32 * math.cos(math.radians(60.0)), rel=0.01)
        assert frame._y1 - frame._y0 == pytest.approx(110.57, rel=0.01)

    def test_aspect_extension_grows_not_stretches(self):
        region = RegionBBox(north=33.5, south=33.0, east=-111.0, west=-112.0)
        frame = GeoFrame(region, target_aspect=2.0)
        width = frame._x1 - frame._x0
        height = frame._y1 - frame._y0
        assert height / width == pytest.approx(2.0, rel=1e-6)
        # The region's own corners sit strictly inside the extended frame.
        for lon, lat in [(-112.0, 33.0), (-111.0, 33.5)]:
            u, v = frame.to_normalized((lon, lat))
            assert 0.0 <= u <= 1.0 and 0.0 < v < 1.0

    def test_rotation_round_trip_envelope(self):
        region = RegionBBox(north=35.0, south=32.0, east=-109.5, west=-112.5)
        frame = GeoFrame(region, rotation_deg=340.0)
        envelope = frame.fetch_region
        # Every region corner must fall inside the fetch envelope.
        assert envelope.north >= region.north and envelope.south <= region.south
        assert envelope.east >= region.east and envelope.west <= region.west
        # And the envelope of a rotated frame is strictly larger.
        assert envelope.area_km2 > region.area_km2

    def test_north_up_is_identity_orientation(self):
        region = RegionBBox(north=41.0, south=40.0, east=-73.0, west=-74.0)
        frame = GeoFrame(region)
        u_west, _ = frame.to_normalized((-74.0, 40.5))
        u_east, _ = frame.to_normalized((-73.0, 40.5))
        _, v_north = frame.to_normalized((-73.5, 41.0))
        _, v_south = frame.to_normalized((-73.5, 40.0))
        assert u_west < u_east and v_north < v_south

    def test_auto_rotation_prefers_fitting_bearing(self):
        region = RegionBBox(north=36.0, south=32.0, east=-109.0, west=-113.0)
        # Diagonal NE-SW chain of points: a tilted bearing must beat north-up
        # for a tall target aspect.
        points = [(-112.5 + 0.5 * i, 32.2 + 0.45 * i) for i in range(8)]
        bearing = auto_rotation(points, region, target_aspect=3.0)
        assert bearing != 0.0

    def test_frame_dict_round_trip(self):
        region = RegionBBox(north=35.0, south=32.0, east=-109.5, west=-112.5)
        frame = GeoFrame(region, rotation_deg=15.0, target_aspect=1.7)
        clone = GeoFrame.from_dict(frame.to_dict())
        for coord in [(-110.0, 33.0), (-112.0, 34.5)]:
            assert clone.to_normalized(coord) == pytest.approx(frame.to_normalized(coord))


# --- Road ink budget ----------------------------------------------------------


class TestInkBudget:
    def _roads(self, n_secondary: int, length: float = 1000.0):
        motorway = (RoadClass.MOTORWAY, [(0.0, 0.0), (length, length)])
        secondaries = [
            (RoadClass.SECONDARY, [(0.0, float(i)), (length, float(i))])
            for i in range(n_secondary)
        ]
        return [motorway] + secondaries

    def test_city_scale_untouched(self):
        roads = self._roads(20)
        kept = prune_roads(roads, canvas_width_px=1000, area_px=1000 * 1400)
        assert len(kept) == 21

    def test_state_scale_capped_but_majors_kept(self):
        roads = self._roads(3000)
        kept = prune_roads(roads, canvas_width_px=1000, area_px=1000 * 1400)
        assert len(kept) < 3001
        assert any(cls is RoadClass.MOTORWAY for cls, _ in kept)
        # Budget respected (within one road's worth of slack).
        ink = sum(
            road_width_px(cls, 1000) * 1000.0 * (2**0.5 if cls is RoadClass.MOTORWAY else 1)
            for cls, _ in kept
        )
        assert ink <= 0.24 * 1000 * 1400 + road_width_px(RoadClass.SECONDARY, 1000) * 1000

    def test_rivers_never_pruned(self):
        roads = self._roads(3000) + [
            (RoadClass.RIVER, [(0.0, 700.0), (1000.0, 700.0)])
        ]
        kept = prune_roads(roads, canvas_width_px=1000, area_px=1000 * 1400)
        assert any(cls is RoadClass.RIVER for cls, _ in kept)


# --- Feature types in the plan -------------------------------------------------


class TestFeatureTypes:
    def _build(self, pois, region):
        source = SourceData(region=region, pois=pois)
        return PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)

    def test_river_poi_becomes_waterway_not_sprite(self, region):
        poi = SourcePoi(
            id="salt_river",
            name="Salt River",
            latitude=40.75,
            longitude=-74.0,
            feature_type="river",
            path=[(40.78, -74.04), (40.75, -74.00), (40.72, -73.96)],
        )
        plan = self._build([poi], region)
        rivers = [r for r in plan.roads if r.cls is RoadClass.RIVER]
        assert len(rivers) == 1 and rivers[0].name == "Salt River"
        assert not plan.pois  # no sprite slot
        assert not any(m.id == "poi_salt_river" for m in plan.manifest)
        water_labels = [l for l in plan.labels if l.kind is LabelKind.WATER]
        assert any(l.text == "Salt River" for l in water_labels)

    def test_mountain_poi_gets_landform_hints(self, region):
        poi = SourcePoi(
            id="camelback",
            name="Camelback Mountain",
            latitude=40.75,
            longitude=-74.0,
            feature_type="mountain",
        )
        plan = self._build([poi], region)
        spec = next(m for m in plan.manifest if m.id == "poi_camelback")
        assert "landform" in spec.prompt_hints
        from mapgen.v2.assets.gemini_client import build_prompt

        prompt = build_prompt(spec, plan.style)
        assert "NEVER draw an animal" in prompt
        assert "single most iconic building" not in prompt

    def test_plain_building_prompt_unchanged_hash(self, region):
        poi = SourcePoi(id="museum", name="Heard Museum", latitude=40.75, longitude=-74.0)
        plan = self._build([poi], region)
        spec = next(m for m in plan.manifest if m.id == "poi_museum")
        assert spec.prompt_hints == ""  # cache-stable for legacy projects


# --- Adaptive warp + sprite fitting -------------------------------------------


class TestAdaptiveWarp:
    def test_dense_cluster_gets_room_or_leaders(self, region):
        # 12 POIs inside ~2% of the map, like metro Phoenix on a state poster.
        pois = [
            SourcePoi(
                id=f"poi{i}",
                name=f"Landmark {i}",
                latitude=40.748 + 0.0008 * (i % 4),
                longitude=-74.001 + 0.0008 * (i // 4),
                tier=2,
            )
            for i in range(12)
        ]
        source = SourceData(region=region, pois=pois)
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)
        fit = plan.provenance["warp_fit"]
        # The cluster is handled cleanly: every sprite is either honestly
        # placed by the warp or offset with a leader line -- never left
        # overlapping or silently far from its anchor.
        assert fit["residual_ok"]
        from mapgen.v2.plan.placement import has_overlaps

        assert not has_overlaps(plan.pois)
        # offset and leader_anchor are consistent for every slot.
        assert all((s.leader_anchor is not None) == s.offset for s in plan.pois)

    def test_subkm_pair_gets_leader_not_blamed_on_warp(self, region):
        # Two POIs ~3m apart: no smooth warp can separate them, so they must
        # be leadered -- not counted as a warp failure.
        pois = [
            SourcePoi(id="a", name="Heard", latitude=40.76000, longitude=-74.00000, tier=2),
            SourcePoi(id="b", name="Phoenix Art", latitude=40.76003, longitude=-74.00001, tier=2),
            SourcePoi(id="far", name="Far", latitude=40.72, longitude=-73.95, tier=2),
        ]
        source = SourceData(region=region, pois=pois)
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)
        fit = plan.provenance["warp_fit"]
        assert fit["coincident_count"] >= 1
        assert fit["residual_ok"]  # the warp is not blamed for the coincident pair
        offset = [s for s in plan.pois if s.offset]
        assert offset and all(s.leader_anchor is not None for s in offset)
        # The far POI is separable -> no leader.
        assert not next(s for s in plan.pois if s.id == "far").offset

    def test_sparse_pois_keep_full_size(self, region):
        pois = [
            SourcePoi(id="a", name="A", latitude=40.78, longitude=-74.04, tier=2),
            SourcePoi(id="b", name="B", latitude=40.72, longitude=-73.96, tier=2),
        ]
        source = SourceData(region=region, pois=pois)
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)
        fit = plan.provenance["warp_fit"]
        assert fit["sprite_scale"] == 1.0 and fit["residual_ok"]
        assert fit["coincident_count"] == 0
        assert all(s.leader_anchor is None and not s.offset for s in plan.pois)


class TestMinimalRoads:
    """The 'minimal' road treatment: mainline I/US routes + major rivers,
    drawn UNWARPED so they read as a clean straight orientation skeleton."""

    def test_is_mainline_keeps_routes_drops_variants(self):
        f = PlanBuilder._is_mainline
        assert f("I 10") and f("US 60")
        assert f("US 180;AZ 64")  # concurrency that includes a mainline
        assert not f("US 80 Hist")
        assert not f("I 10 BUS")
        assert not f("I 40 BUS;US 66 Hist")
        assert not f("AZ 87")
        assert not f(None)

    def _lonlat(self, region):
        lon = lambda f: region.west + f * (region.east - region.west)
        lat = lambda f: region.south + f * (region.north - region.south)
        return lon, lat

    def test_keeps_only_mainline_and_rivers(self, region):
        lon, lat = self._lonlat(region)
        roads = [
            SourceRoad(cls=RoadClass.PRIMARY, ref="I 10",
                       coords=[(lon(0.1), lat(0.5)), (lon(0.9), lat(0.5))]),
            SourceRoad(cls=RoadClass.PRIMARY, ref="US 80 Hist",
                       coords=[(lon(0.1), lat(0.3)), (lon(0.9), lat(0.3))]),
            SourceRoad(cls=RoadClass.PRIMARY, ref="AZ 87",
                       coords=[(lon(0.1), lat(0.7)), (lon(0.9), lat(0.7))]),
            SourceRoad(cls=RoadClass.RIVER, name="Big River",
                       coords=[(lon(0.0), lat(0.8)), (lon(1.0), lat(0.6))]),
        ]
        pois = [SourcePoi(id="p", name="P", latitude=lat(0.5), longitude=lon(0.5), tier=2)]
        src = SourceData(region=region, roads=roads, pois=pois)
        plan = PlanBuilder(
            canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72),
            road_treatment="minimal",
        ).build(src)
        refs = {r.ref for r in plan.roads if r.cls is not RoadClass.RIVER}
        assert refs == {"I 10"}  # historic variant + state route dropped
        assert any(r.cls is RoadClass.RIVER for r in plan.roads)

    def test_minimal_roads_drawn_straight_despite_warp(self, region):
        # A straight interstate stays collinear even when a dense POI cluster
        # warps the map -- a projective/affine camera preserves straight lines,
        # and the minimal treatment bypasses the (non-linear) warp entirely.
        lon, lat = self._lonlat(region)
        road = SourceRoad(
            cls=RoadClass.PRIMARY, ref="I 10",
            coords=[(lon(x / 10.0), lat(0.5)) for x in range(1, 10)],
        )
        pois = [
            SourcePoi(id=f"c{i}", name=f"C{i}",
                      latitude=lat(0.2) + 0.001 * (i % 3),
                      longitude=lon(0.8) + 0.001 * (i // 3), tier=2)
            for i in range(9)
        ]
        src = SourceData(region=region, roads=[road], pois=pois)
        plan = PlanBuilder(
            canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72),
            road_treatment="minimal",
        ).build(src)
        pts = next(r.points for r in plan.roads if r.cls is not RoadClass.RIVER)
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        # max perpendicular distance of any interior point from the end-to-end
        # line: ~0 for a straight road, large if the warp had bent it.
        max_perp = max(
            abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0) / seg_len
            for px, py in pts
        )
        assert max_perp < 2.0


class TestLabels:
    def test_poi_labels_never_dropped(self, region):
        pois = [
            SourcePoi(
                id=f"poi{i}",
                name=f"Crowded Landmark Number {i}",
                latitude=40.748 + 0.0005 * (i % 3),
                longitude=-74.001 + 0.0005 * (i // 3),
                tier=2,
            )
            for i in range(9)
        ]
        source = SourceData(region=region, pois=pois)
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)
        poi_labels = [l for l in plan.labels if l.kind is LabelKind.POI]
        assert len(poi_labels) == 9

    def test_street_labels_prefer_class_over_length(self, source):
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(source)
        street = [l for l in plan.labels if l.kind is LabelKind.STREET]
        if street:  # placement is greedy; the top street pick must be major
            assert street[0].text in ("Coast Highway", "Main Street")


# --- Scene vocabulary -----------------------------------------------------------


class TestSceneVocabulary:
    def test_preset_resolves_full_kit(self):
        from mapgen.v2.styles import resolve_style

        style = resolve_style("southwest_desert")
        assert style.land_scatter and style.water_scatter is None
        assert "coastline" not in style.scene

    def test_explicit_keys_override_preset(self):
        from mapgen.v2.styles import resolve_style

        style = resolve_style({"preset": "southwest_desert", "wobble_px": 3.0})
        assert style.wobble_px == 3.0
        assert style.land_scatter  # preset kit still applied

    def test_style_bible_prompt_uses_scene(self):
        from mapgen.v2.assets.gemini_client import build_prompt
        from mapgen.v2.types import AssetKind, AssetSpec

        spec = AssetSpec(id="style_bible", kind=AssetKind.STYLE_BIBLE, subject="swatch")
        desert = StyleSpec(scene="saguaro cacti and red rock mesas")
        assert "saguaro" in build_prompt(spec, desert)
        assert "coastline" not in build_prompt(spec, desert)

    def test_land_scatter_spawns_on_bare_land(self, region):
        source = SourceData(region=region)
        style = StyleSpec(land_scatter=["cactus", "rock"], water_scatter=None)
        plan = PlanBuilder(
            canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72), style=style
        ).build(source)
        kinds = {s.kind for s in plan.scatter}
        assert ScatterKind.CACTUS in kinds and ScatterKind.ROCK in kinds
        assert any(m.id == "sprites_cactus" for m in plan.manifest)

    def test_no_boats_when_water_scatter_disabled(self, source):
        style = StyleSpec(water_scatter=None)
        plan = PlanBuilder(
            canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72), style=style
        ).build(source)
        assert not any(s.kind is ScatterKind.BOAT for s in plan.scatter)


# --- Retitle --------------------------------------------------------------------


class TestRetitle:
    def test_retitle_patches_plan_in_place(self, tmp_path, region):
        from mapgen.v2 import pipeline

        project = pipeline.V2Project(name="Testland", region=region)
        project.save(tmp_path / "project.yaml")
        plan = PlanBuilder(canvas=CanvasSpec(width_px=1000, height_px=1414, dpi=72)).build(
            SourceData(region=region), title="Testland"
        )
        pipeline.write_plan(plan, tmp_path)

        assert pipeline.retitle_project(tmp_path, "New Grand Title") is True
        patched = pipeline.PlanDocument.load(tmp_path / "plan.json")
        titles = [l.text for l in patched.labels if l.kind is LabelKind.TITLE]
        assert titles == ["New Grand Title"]
        assert pipeline.V2Project.load(tmp_path / "project.yaml").display_title == "New Grand Title"

    def test_retitle_without_plan(self, tmp_path, region):
        from mapgen.v2 import pipeline

        project = pipeline.V2Project(name="Testland", region=region)
        project.save(tmp_path / "project.yaml")
        assert pipeline.retitle_project(tmp_path, "Renamed") is False

    def test_empty_title_rejected(self, tmp_path, region):
        from mapgen.v2 import pipeline

        pipeline.V2Project(name="Testland", region=region).save(tmp_path / "project.yaml")
        with pytest.raises(ValueError):
            pipeline.retitle_project(tmp_path, "   ")


# --- Provenance / warnings ------------------------------------------------------


class TestProvenance:
    def test_failed_layer_produces_warning(self, region):
        from mapgen.v2 import pipeline

        source = SourceData(region=region)
        source.provenance = {"detail_level": "regional", "layers": {"water": "failed"}}
        project = pipeline.V2Project(name="X", region=region)
        plan = pipeline.build_plan(project, source)
        assert any("water fetch FAILED" in w for w in plan.warnings)
        assert any("no ground polygons" in w for w in plan.warnings)
        assert plan.provenance["layers"]["water"] == "failed"

    def test_detail_tiers_by_area(self):
        from mapgen.v2.pipeline import detail_level_for

        nyc = RegionBBox(north=40.735, south=40.695, east=-73.985, west=-74.025)
        az = RegionBBox(north=35.75, south=32.05, east=-109.55, west=-112.45)
        assert detail_level_for(nyc) == "full"
        assert detail_level_for(az) == "regional"


# --- Matting: palette-shifted keys ----------------------------------------------


class TestKeyFamilyMatting:
    def test_white_subject_survives_rose_key(self):
        from mapgen.v2.assets.matting import key_to_alpha

        rose = (181, 78, 120)
        img = Image.new("RGB", (200, 200), rose)
        # White building with a cream roof patch.
        white = Image.new("RGB", (80, 80), (250, 248, 240))
        img.paste(white, (60, 60))
        out = key_to_alpha(img)
        alpha = np.asarray(out.getchannel("A"))
        assert alpha[100, 100] > 250  # subject opaque
        assert alpha[10, 10] == 0  # key transparent

    def test_enclosed_key_pocket_removed(self):
        from mapgen.v2.assets.matting import key_to_alpha

        img = Image.new("RGB", (240, 240), (253, 34, 250))
        # A ring-shaped subject with a magenta center pocket.
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 200, 200], fill=(180, 120, 60))
        draw.ellipse([95, 95, 145, 145], fill=(253, 34, 250))
        out = key_to_alpha(img)
        alpha = np.asarray(out.getchannel("A"))
        assert alpha[120, 70] > 250  # ring body opaque
        assert alpha[120, 120] < 10  # enclosed key pocket transparent

    def test_key_shift_flags(self):
        from mapgen.v2.assets.matting import KEY_SHIFT_THRESHOLD, key_shift

        assert key_shift((255, 0, 255)) == 0.0
        assert key_shift((181, 78, 120)) > KEY_SHIFT_THRESHOLD


# --- Repaint guards --------------------------------------------------------------


class TestRepaintGuards:
    def test_match_low_frequency_removes_tone_patch(self):
        from mapgen.v2.repaint.color_norm import match_low_frequency

        guide = Image.new("RGB", (1024, 1024), (200, 160, 120))
        drifted = np.full((1024, 1024, 3), (200, 160, 120), dtype=np.uint8)
        drifted[:512, :512] = (215, 175, 135)  # one quadrant drifted brighter
        out = match_low_frequency(Image.fromarray(drifted), guide)
        arr = np.asarray(out, dtype=np.float32)
        delta = abs(arr[:400, :400].mean() - arr[600:, 600:].mean())
        assert delta < 4.0  # patchwork flattened

    def test_invention_guard_keeps_guide_on_flat_window(self, tmp_path):
        from mapgen.v2.repaint import RepaintStore
        from mapgen.v2.repaint.engine import RepaintEngine
        from mapgen.v2.repaint.grid import QUAD

        class InventingPainter:
            def paint(self, template, style, style_bible=None):
                out = template.copy()
                from PIL import ImageDraw

                draw = ImageDraw.Draw(out)
                draw.ellipse([200, 200, 800, 800], fill=(60, 40, 30))  # a "lake"
                return out

        # Flat-but-textured guide: noise keeps it from being skipped outright?
        # No -- the blur-skip would drop it, so add faint real structure that
        # survives blurring (a soft gradient) while staying "flat" to the
        # invention guard's std threshold.
        base = np.full((2 * QUAD, 2 * QUAD, 3), (210, 180, 150), dtype=np.float32)
        gradient = np.linspace(0, 8, 2 * QUAD, dtype=np.float32)[None, :, None]
        guide = Image.fromarray(np.clip(base + gradient, 0, 255).astype(np.uint8))
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(InventingPainter(), store, StyleSpec())
        result = engine.run(guide)
        out = np.asarray(result.image, dtype=np.float32)
        # The invented dark lake must not survive.
        assert out[512, 512].mean() > 150

    def test_textured_flat_cells_skipped(self, tmp_path):
        from mapgen.v2.repaint import QuadStatus, RepaintStore
        from mapgen.v2.repaint.engine import RepaintEngine
        from mapgen.v2.repaint.grid import QUAD
        from tests.v2.test_repaint import IdentityPainter

        rng = np.random.default_rng(3)
        # Brush-noise texture, no structure: must be skipped, not painted.
        arr = np.clip(
            rng.normal(0, 6, (2 * QUAD, 2 * QUAD, 3)) + (210, 180, 150), 0, 255
        ).astype(np.uint8)
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(IdentityPainter(), store, StyleSpec())
        result = engine.run(Image.fromarray(arr))
        assert result.calls_made == 0
        assert len(store.cells_with_status(QuadStatus.SKIPPED)) == 4
