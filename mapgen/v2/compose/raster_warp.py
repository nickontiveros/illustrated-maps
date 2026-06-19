"""Warp an envelope-aligned raster into oblique poster space.

Both the satellite base and the DEM hillshade are north-up rasters covering
the frame's geographic envelope. Getting either onto the poster is the same
operation: invert the plan's forward transform (camera -> warp -> frame) at a
grid of poster nodes, then let PIL's inverse-sampled ``Image.transform(MESH)``
pull the source quads. This module owns that one operation so the two callers
can't drift apart.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..ingest import GeoFrame
from ..plan.camera import ObliqueCamera
from ..types import CanvasSpec

# Mesh grid over the poster. The oblique+warp deformation is smooth and
# low-frequency, so a coarse grid is exact at nodes and near-exact between.
_MESH_COLS = 80
_MESH_ROWS = 120


class ObliqueRasterWarper:
    def __init__(self, frame: GeoFrame, camera: ObliqueCamera, canvas: CanvasSpec, warp):
        self.frame = frame
        self.camera = camera
        self.canvas = canvas
        self.warp = warp
        self.env = frame.fetch_region

    def warp_to_poster(self, raster: Image.Image, scale: float, trapezoid: bool = True) -> Image.Image:
        """Warp ``raster`` (covering the frame envelope, north-up) to poster
        space at ``scale``. When ``trapezoid`` the result is RGBA masked to the
        camera's converging map area; otherwise the raw warp is returned."""
        out_w = max(1, int(round(self.canvas.width_px * scale)))
        out_h = max(1, int(round(self.canvas.height_px * scale)))

        xs = np.linspace(0.0, self.canvas.width_px, _MESH_COLS + 1)
        ys = np.linspace(0.0, self.canvas.height_px, _MESH_ROWS + 1)
        gx, gy = np.meshgrid(xs, ys)
        sx, sy = self.poster_to_source_px(gx, gy, raster.size)

        mesh = []
        for r in range(_MESH_ROWS):
            for c in range(_MESH_COLS):
                dx0, dx1 = int(round(xs[c] * scale)), int(round(xs[c + 1] * scale))
                dy0, dy1 = int(round(ys[r] * scale)), int(round(ys[r + 1] * scale))
                quad = (
                    sx[r, c], sy[r, c],
                    sx[r + 1, c], sy[r + 1, c],
                    sx[r + 1, c + 1], sy[r + 1, c + 1],
                    sx[r, c + 1], sy[r, c + 1],
                )
                mesh.append(((dx0, dy0, dx1, dy1), quad))

        warped = raster.transform(
            (out_w, out_h), Image.Transform.MESH, mesh, Image.Resampling.BILINEAR
        )
        if not trapezoid:
            return warped
        warped = warped.convert("RGBA")
        warped.putalpha(self.trapezoid_mask(out_w, out_h, scale))
        return warped

    def poster_to_source_px(self, px, py, source_size):
        """Vectorized inverse: poster px -> source (envelope-aligned) px."""
        cam = self.camera
        spec = cam.spec
        vs = spec.vertical_scale
        # poster y -> far-near parameter t (cam.t_at_poster_y, vectorized).
        c = np.clip((py - cam.horizon_px) / cam.map_height_px, 0.0, None) * cam._mean_scale
        a = (1.0 - vs) / 2.0
        if a < 1e-9:
            t = c / max(1e-9, vs)
        else:
            t = (-vs + np.sqrt(vs * vs + 4.0 * a * c)) / (2.0 * a)
        t = np.clip(t, 0.0, 1.0)
        flat_y = t * cam.flat_height
        width_scale = spec.convergence + (1.0 - spec.convergence) * t
        cx = cam.flat_width / 2.0
        flat_x = cx + (px - cx) / np.where(width_scale == 0, 1e-9, width_scale)
        wu = np.clip(flat_x / cam.flat_width, 0.0, 1.0)
        wv = np.clip(flat_y / cam.flat_height, 0.0, 1.0)
        u, v = self._unwarp(wu, wv)
        lon, lat = self._frame_from_normalized(u, v)
        sw, sh = source_size
        ex = (lon - self.env.west) / max(1e-9, self.env.width_deg) * sw
        ey = (self.env.north - lat) / max(1e-9, self.env.height_deg) * sh
        return ex, ey

    def _unwarp(self, wu, wv):
        warp = self.warp
        if not hasattr(warp, "_fx"):  # IdentityWarp
            return wu, wv
        u = np.interp(wu, warp._fx, warp._grid)
        v = np.interp(wv, warp._fy, warp._grid)
        return u, v

    def _frame_from_normalized(self, u, v):
        f = self.frame
        x = f._x0 + u * (f._x1 - f._x0)
        y = f._y0 + v * (f._y1 - f._y0)
        e = x * f._cos_b - y * f._sin_b
        n = -(x * f._sin_b + y * f._cos_b)
        return (f._lon_c + e / f._kx, f._lat_c + n / f._ky)

    def trapezoid_mask(self, w: int, h: int, scale: float) -> Image.Image:
        """Alpha mask matching the camera's converging far edge (the same
        trapezoid _draw_base_land paints), so the sky stays empty."""
        cam = self.camera
        horizon = cam.spec.horizon_margin * self.canvas.height_px * scale
        far_half = cam.spec.convergence * w / 2.0
        poly = [
            (w / 2 - far_half, horizon),
            (w / 2 + far_half, horizon),
            (w, h),
            (0, h),
        ]
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        return mask
