import math

import pytest

from mapgen.v2.plan.camera import ObliqueCamera, densify
from mapgen.v2.types import CameraSpec, CanvasSpec


@pytest.fixture
def camera() -> ObliqueCamera:
    return ObliqueCamera(
        CameraSpec(convergence=0.7, vertical_scale=0.5, horizon_margin=0.1),
        CanvasSpec(width_px=1000, height_px=1400),
    )


def test_near_edge_is_unchanged(camera: ObliqueCamera):
    """The bottom (near) edge keeps full width and lands at the canvas bottom."""
    left = camera.project_point((0.0, camera.flat_height))
    right = camera.project_point((camera.flat_width, camera.flat_height))
    assert left[0] == pytest.approx(0.0)
    assert right[0] == pytest.approx(1000.0)
    assert left[1] == pytest.approx(1400.0)


def test_far_edge_converges_and_sits_on_horizon(camera: ObliqueCamera):
    left = camera.project_point((0.0, 0.0))
    right = camera.project_point((camera.flat_width, 0.0))
    # Far edge narrows to `convergence` of full width, centered.
    assert right[0] - left[0] == pytest.approx(700.0)
    assert (left[0] + right[0]) / 2 == pytest.approx(500.0)
    # And sits exactly at the horizon margin.
    assert left[1] == pytest.approx(0.1 * 1400)


def test_vertical_compression_increases_with_distance(camera: ObliqueCamera):
    """Equal flat-space steps span fewer pixels near the horizon."""
    h = camera.flat_height
    near = camera.project_point((500, h))[1] - camera.project_point((500, 0.9 * h))[1]
    far = camera.project_point((500, 0.1 * h))[1] - camera.project_point((500, 0.0))[1]
    assert far < near
    assert far / near == pytest.approx(0.5 / 0.95, rel=0.05)


def test_projection_is_monotonic(camera: ObliqueCamera):
    ys = [camera.project_point((500, t * camera.flat_height))[1] for t in [i / 20 for i in range(21)]]
    assert all(a < b for a, b in zip(ys, ys[1:]))


def test_depth_attribute(camera: ObliqueCamera):
    assert camera.depth(0.0) == pytest.approx(1.0)
    assert camera.depth(camera.flat_height) == pytest.approx(0.0)


def test_straight_vertical_line_stays_straight_in_x_center(camera: ObliqueCamera):
    pts = camera.project_points([(500.0, 0.0), (500.0, camera.flat_height)])
    assert all(p[0] == pytest.approx(500.0) for p in pts)


def test_densify_limits_segment_length():
    pts = densify([(0.0, 0.0), (100.0, 0.0)], max_seg_px=8.0)
    for a, b in zip(pts, pts[1:]):
        assert math.hypot(b[0] - a[0], b[1] - a[1]) <= 8.0 + 1e-9
    assert pts[0] == (0.0, 0.0)
    assert pts[-1] == (100.0, 0.0)
