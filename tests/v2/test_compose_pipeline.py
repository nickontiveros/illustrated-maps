"""End-to-end: synthetic source -> plan -> stub assets -> composed poster."""

import numpy as np
import pytest
from PIL import Image

from mapgen.v2 import pipeline
from mapgen.v2.assets.stub import StubAssetGenerator
from mapgen.v2.compose import Compositor
from mapgen.v2.plan import PlanBuilder


@pytest.fixture
def plan(source, small_canvas):
    return PlanBuilder(canvas=small_canvas, distortion_strength=0.4).build(source, title="Test Town")


@pytest.fixture
def assets_dir(plan, tmp_path):
    pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
    return tmp_path


def test_compose_full_scale(plan, assets_dir, artifacts):
    img = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME).render(scale=1.0)
    artifacts.save("poster", img)
    assert img.size == (plan.canvas.width_px, plan.canvas.height_px)
    arr = np.asarray(img)
    # The render must contain meaningful variation (not a blank sheet):
    assert arr.std() > 10
    # Water region (east side, mid height) should differ from land (west).
    water = arr[700:720, 900:950].mean(axis=(0, 1))
    land = arr[700:720, 100:150].mean(axis=(0, 1))
    assert np.abs(water - land).sum() > 30


def test_compose_scaled_preview_is_consistent(plan, assets_dir):
    img = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME).render(scale=0.25)
    assert img.width == round(plan.canvas.width_px * 0.25)


def test_compose_without_assets_uses_placeholders(plan, tmp_path):
    """The compositor must render reviewable output before any AI spend."""
    img = Compositor(plan, tmp_path / "missing").render(scale=0.5)
    assert np.asarray(img).std() > 5


def test_leader_lines_render_in_poster(source, small_canvas, tmp_path):
    """An offset POI draws a connector; clearing the offset removes it."""
    from mapgen.v2.ingest import SourceData, SourcePoi

    region = source.region
    midlat = (region.north + region.south) / 2
    midlon = (region.east + region.west) / 2
    pois = [
        SourcePoi(id="x", name="X", latitude=midlat, longitude=midlon, tier=2),
        # ~3 m away: no warp can separate them, so one is leadered.
        SourcePoi(id="y", name="Y", latitude=midlat + 0.00003, longitude=midlon + 0.00001, tier=2),
    ]
    p = PlanBuilder(canvas=small_canvas).build(SourceData(region=region, pois=pois), title="T")
    assert any(s.offset for s in p.pois)
    pipeline.generate_assets(p, tmp_path, StubAssetGenerator())
    adir = tmp_path / pipeline.ASSETS_DIRNAME

    with_leaders = np.asarray(Compositor(p, adir).render_base(scale=0.5))
    p2 = p.model_copy(deep=True)
    for s in p2.pois:
        s.offset, s.leader_anchor = False, None
    without = np.asarray(Compositor(p2, adir).render_base(scale=0.5))
    assert not np.array_equal(with_leaders, without)


def test_render_equals_base_plus_finish(plan, assets_dir):
    """render() must be exactly the split pipeline: base then finish."""
    c = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME)
    whole = np.asarray(c.render(scale=0.3))
    split = np.asarray(c.apply_finish(c.render_base(scale=0.3), scale=0.3))
    assert np.array_equal(whole, split)


def test_water_class_mask_aligns_with_render(plan, assets_dir, artifacts):
    """The deterministic water mask must agree with where the render actually
    paints water (same wobble seeds), and exclude land."""
    from mapgen.v2.types import GroundClass

    c = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME)
    scale = 0.3
    mask = np.asarray(c.render_class_mask({GroundClass.WATER}, scale=scale))
    base = c.render_base(scale=scale)
    artifacts.save("water_mask", mask)
    artifacts.save("base", base)
    assert mask.shape == (base.height, base.width)
    # The synthetic town has a bay on the east side and land on the west.
    h, w = mask.shape
    assert mask[int(h * 0.55), int(w * 0.95)] == 255  # inside East Bay
    assert mask[int(h * 0.55), int(w * 0.1)] == 0  # land
    # Mask is substantial but not the whole canvas.
    frac = (mask > 0).mean()
    assert 0.05 < frac < 0.6


def test_apply_finish_draws_labels_and_frame(plan, assets_dir):
    c = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME)
    base = c.render_base(scale=0.3)
    finished = c.apply_finish(base.copy(), scale=0.3)
    # Finish must change pixels (labels/frame/grain on top of the base).
    assert not np.array_equal(np.asarray(base.convert("RGB")), np.asarray(finished))


def test_render_is_deterministic(plan, assets_dir):
    c = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME)
    a = np.asarray(c.render(scale=0.3))
    b = np.asarray(c.render(scale=0.3))
    assert np.array_equal(a, b)


def test_full_pipeline_writes_outputs(source, small_canvas, tmp_path):
    project = pipeline.V2Project(
        name="Test Town",
        region=source.region,
        output=small_canvas,
        pois=[
            pipeline.PoiConfig(name="Old Lighthouse", lat=40.725, lon=-73.978, tier=1),
        ],
    )
    plan = pipeline.build_plan(project, source)
    plan_path, preview_path = pipeline.write_plan(plan, tmp_path)
    assert plan_path.exists() and preview_path.exists()
    pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
    poster = pipeline.compose_poster(plan, tmp_path, scale=0.5)
    assert poster.exists()
    img = Image.open(poster)
    assert img.width == round(small_canvas.width_px * 0.5)


def test_project_yaml_roundtrip(tmp_path, region):
    project = pipeline.V2Project(
        name="Town",
        region=region,
        pois=[pipeline.PoiConfig(name="A", lat=40.75, lon=-74.0)],
    )
    path = tmp_path / "project.yaml"
    project.save(path)
    loaded = pipeline.V2Project.load(path)
    assert loaded == project


def test_project_yaml_accepts_style_preset_string(tmp_path, region):
    (tmp_path / "project.yaml").write_text(
        "name: T\n"
        "style: vintage_tourist\n"
        f"region: {{north: {region.north}, south: {region.south}, east: {region.east}, west: {region.west}}}\n"
    )
    project = pipeline.V2Project.load(tmp_path / "project.yaml")
    assert project.style.preset == "vintage_tourist"
