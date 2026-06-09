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


def test_compose_full_scale(plan, assets_dir):
    img = Compositor(plan, assets_dir / pipeline.ASSETS_DIRNAME).render(scale=1.0)
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
