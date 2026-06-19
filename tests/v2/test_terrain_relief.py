"""DEM terrain relief: opt-in, self-disabling, and applied only in
illustrated mode."""

from PIL import Image

from mapgen.v2.compose.compositor import Compositor
from mapgen.v2.plan import PlanBuilder


def _plan(source, small_canvas):
    return PlanBuilder(canvas=small_canvas, distortion_strength=0.0, seed=3).build(source)


def test_relief_off_by_default(source, small_canvas, tmp_path):
    plan = _plan(source, small_canvas)
    comp = Compositor(plan, assets_dir=tmp_path)
    assert comp._terrain_relief(0.25) is None  # disabled => no elevation fetch


def test_relief_applied_when_enabled(source, small_canvas, tmp_path, monkeypatch):
    plan = _plan(source, small_canvas)
    plan.style.terrain_relief = True
    plan.style.hillshade_strength = 1.0
    comp = Compositor(plan, assets_dir=tmp_path)

    w = int(round(plan.canvas.width_px * 0.25))
    h = int(round(plan.canvas.height_px * 0.25))
    # Stub a half-dark hillshade covering the whole canvas.
    relief = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    monkeypatch.setattr(comp, "_terrain_relief", lambda scale: relief)

    canvas = Image.new("RGBA", (w, h), (200, 200, 200, 255))
    comp._apply_terrain_relief(canvas, 0.25)
    # strength=1.0, multiply by black => land goes black inside the mask.
    assert canvas.getpixel((w // 2, h // 2))[:3] == (0, 0, 0)


def test_relief_skips_in_satellite_mode(source, small_canvas, tmp_path, monkeypatch):
    """Satellite imagery already carries relief; the hillshade pass is not run
    in the satellite branch of render_base."""
    plan = _plan(source, small_canvas)
    plan.style.terrain_relief = True
    plan.style.base_mode = "satellite"
    comp = Compositor(plan, assets_dir=tmp_path)

    calls = []
    monkeypatch.setattr(comp, "_apply_terrain_relief", lambda c, s: calls.append(s))
    # Force the satellite base to be present so we take the satellite branch.
    w = int(round(plan.canvas.width_px * 0.25))
    h = int(round(plan.canvas.height_px * 0.25))
    monkeypatch.setattr(comp, "_satellite_base", lambda scale: Image.new("RGBA", (w, h)))

    comp.render_base(scale=0.25)
    assert calls == []  # relief never applied on the satellite path
