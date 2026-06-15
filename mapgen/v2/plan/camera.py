"""Oblique bird's-eye camera applied to vectors.

The camera maps "flat space" (an orthographic top-down canvas of the
distorted region, y=0 at the far/top edge) into poster space:

- the far edge narrows toward the center (convergence),
- vertical distances compress progressively toward the far edge
  (foreshortening), with the compression factor interpolating linearly
  from ``vertical_scale`` at the far edge to 1.0 at the near edge,
- a horizon margin is reserved at the top of the poster.

Because this is a vector transform applied before any rendering, the
perspective is exact everywhere: there is no raster warp, no resampling
blur, and no seam misalignment by construction.
"""

from __future__ import annotations

import math

from ..types import CameraSpec, CanvasSpec, Point


class ObliqueCamera:
    def __init__(self, camera: CameraSpec, canvas: CanvasSpec):
        self.spec = camera
        self.canvas = canvas
        self.horizon_px = camera.horizon_margin * canvas.height_px
        self.map_height_px = canvas.height_px - self.horizon_px
        # Flat space shares the poster's width; its height is chosen so the
        # *average* vertical compression maps it exactly onto map_height_px.
        self._mean_scale = (camera.vertical_scale + 1.0) / 2.0
        self.flat_width = float(canvas.width_px)
        self.flat_height = self.map_height_px / self._mean_scale

    def _t(self, y: float) -> float:
        """Normalized far-near parameter: 0 at far (top), 1 at near (bottom)."""
        return min(1.0, max(0.0, y / self.flat_height))

    def depth(self, y: float) -> float:
        """Depth attribute: 0 = near (bottom), 1 = far (horizon)."""
        return 1.0 - self._t(y)

    def width_scale(self, y: float) -> float:
        t = self._t(y)
        return self.spec.convergence + (1.0 - self.spec.convergence) * t

    def project_point(self, p: Point) -> Point:
        x, y = p
        t = self._t(y)
        cx = self.flat_width / 2.0
        px = cx + (x - cx) * self.width_scale(y)
        # Vertical position is the integral of the linear scale profile
        # s(t) = vs + (1 - vs) * t, normalized so t=1 lands on map_height_px.
        vs = self.spec.vertical_scale
        integral = vs * t + (1.0 - vs) * t * t / 2.0
        py = self.horizon_px + self.map_height_px * (integral / self._mean_scale)
        return (px, py)

    def project_points(self, points: list[Point], densify_px: float = 8.0) -> list[Point]:
        """Project a polyline/ring, densifying first so curves stay smooth."""
        return [self.project_point(p) for p in densify(points, densify_px)]

    def scale_at(self, y_flat: float) -> float:
        """Local isotropic scale factor (used to size sprites by depth)."""
        return self.width_scale(y_flat)

    def t_at_poster_y(self, poster_y: float) -> float:
        """Invert the vertical projection: poster y -> far-near parameter t.

        project_point maps t through the integral of the linear scale
        profile; this solves that quadratic back, so poster-space samplers
        (e.g. bare-land scatter) can recover depth and local width."""
        vs = self.spec.vertical_scale
        c = max(0.0, (poster_y - self.horizon_px) / self.map_height_px) * self._mean_scale
        a = (1.0 - vs) / 2.0
        if a < 1e-9:
            t = c / max(1e-9, vs)
        else:
            t = (-vs + math.sqrt(vs * vs + 4.0 * a * c)) / (2.0 * a)
        return min(1.0, max(0.0, t))


def densify(points: list[Point], max_seg_px: float) -> list[Point]:
    """Insert vertices so no segment exceeds max_seg_px (in flat space)."""
    if len(points) < 2 or max_seg_px <= 0:
        return list(points)
    out: list[Point] = [points[0]]
    for a, b in zip(points, points[1:]):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(math.ceil(dist / max_seg_px)))
        for i in range(1, steps + 1):
            f = i / steps
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out
