"""Satellite base mode: registration, warp serialization, and graceful
fallback when no imagery is available."""

import numpy as np
import pytest
from PIL import Image

from mapgen.v2.compose.compositor import Compositor
from mapgen.v2.compose.satellite_base import SatelliteBaseBuilder
from mapgen.v2.ingest import GeoFrame
from mapgen.v2.plan import PlanBuilder
from mapgen.v2.plan.camera import ObliqueCamera
from mapgen.v2.plan.distortion import ImportanceWarp, warp_from_dict
from mapgen.v2.types import CameraSpec, CanvasSpec


def test_warp_round_trips_through_dict():
    warp = ImportanceWarp(centers=[(0.3, 0.6), (0.7, 0.2)], strength=0.8)
    restored = warp_from_dict(warp.to_dict())
    pts = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.4)]
    assert restored.warp_points(pts) == pytest.approx(warp.warp_points(pts))
    # And the inverse undoes the forward warp.
    back = restored.unwarp_points(warp.warp_points(pts))
    assert back == pytest.approx(pts, abs=1e-3)


def test_empty_warp_dict_is_identity():
    warp = warp_from_dict({})
    pts = [(0.2, 0.8), (0.6, 0.1)]
    assert warp.warp_points(pts) == pts
    assert warp.unwarp_points(pts) == pts


def _builder(plan, ortho):
    b = SatelliteBaseBuilder.from_plan(plan)
    b._ortho = ortho  # bypass the network fetch
    return b


def test_poster_inverse_lands_at_frame_center(source, small_canvas):
    """The map-frame center must sample the ortho center after the full
    inverse (camera + warp + frame)."""
    plan = PlanBuilder(canvas=small_canvas, distortion_strength=0.0, seed=3).build(source)
    ortho = Image.new("RGB", (256, 256))
    builder = _builder(plan, ortho)

    cam = ObliqueCamera(plan.camera, plan.canvas)
    # Forward-project the frame-center (normalized 0.5,0.5 -> flat -> poster).
    fx = 0.5 * cam.flat_width
    fy = 0.5 * cam.flat_height
    px, py = cam.project_point((fx, fy))

    ex, ey = builder._poster_to_ortho_px(np.array([px]), np.array([py]), ortho.size)
    # Frame center maps near the ortho center (envelope is larger than the
    # frame, but symmetric about the same center).
    assert ex[0] == pytest.approx(128, abs=4)
    assert ey[0] == pytest.approx(128, abs=4)


def test_warped_base_is_masked_to_trapezoid(source, small_canvas):
    plan = PlanBuilder(canvas=small_canvas, distortion_strength=0.0, seed=3).build(source)
    # A vivid ortho so we can see where it lands.
    ortho = Image.new("RGB", (256, 256), (10, 200, 60))
    builder = _builder(plan, ortho)
    warped = builder._warp_to_poster(ortho, scale=0.25)

    assert warped.mode == "RGBA"
    alpha = warped.getchannel("A")
    # Sky (top-left corner) is outside the converging trapezoid -> transparent.
    assert alpha.getpixel((0, 0)) == 0
    # The near (bottom) edge center is inside the map -> opaque imagery.
    w, h = warped.size
    assert alpha.getpixel((w // 2, h - 2)) == 255


def test_satellite_mode_falls_back_without_imagery(source, small_canvas, tmp_path, monkeypatch):
    """base_mode=satellite with no Mapbox token still renders (illustrated)."""
    monkeypatch.delenv("MAPBOX_ACCESS_TOKEN", raising=False)
    plan = PlanBuilder(canvas=small_canvas, distortion_strength=0.0, seed=3).build(source)
    plan.style.base_mode = "satellite"
    comp = Compositor(plan, assets_dir=tmp_path)
    img = comp.render_base(scale=0.25)
    assert img.size == (
        int(round(plan.canvas.width_px * 0.25)),
        int(round(plan.canvas.height_px * 0.25)),
    )
    # Fallback path memoizes None, so no imagery was used.
    assert comp._sat_base_cache[0.25] is None


def test_satellite_base_composited_when_available(source, small_canvas, tmp_path, monkeypatch):
    """A stubbed builder's base shows through, and scatter is suppressed."""
    plan = PlanBuilder(canvas=small_canvas, distortion_strength=0.0, seed=3).build(source)
    plan.style.base_mode = "satellite"
    comp = Compositor(plan, assets_dir=tmp_path)

    w = int(round(plan.canvas.width_px * 0.25))
    h = int(round(plan.canvas.height_px * 0.25))
    stub = Image.new("RGBA", (w, h), (255, 0, 0, 255))
    monkeypatch.setattr(comp, "_satellite_base", lambda scale: stub)

    img = comp.render_base(scale=0.25).convert("RGB")
    # Red base shows through somewhere on the map surface (bottom-center).
    r, g, b = img.getpixel((w // 2, h - 3))
    assert r > 150 and g < 100 and b < 100
