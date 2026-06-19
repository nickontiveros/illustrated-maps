"""Photoreal satellite base layer.

In ``base_mode: satellite`` the painted ground textures are replaced by real
satellite imagery, warped to the same oblique camera the vector geometry uses
so roads, labels and 3D buildings register pixel-for-pixel on top.

Registration chain (the inverse of the plan's forward transform):

    poster px --(camera inverse)--> flat px --(/flat dims)--> warped-normalized
              --(warp inverse)--> normalized --(GeoFrame.from_normalized)-->
              lon/lat --(linear bbox map)--> envelope-ortho px

Because PIL's ``Image.transform(MESH)`` is inverse-sampled (it asks, for each
destination cell, which source quad to pull), this inverse map is exactly what
it needs: we lay a grid over the *poster* and resolve each node back to a pixel
in the fetched ortho.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ...models.project import BoundingBox
from ..ingest import GeoFrame
from ..plan.camera import ObliqueCamera
from ..plan.distortion import warp_from_dict
from ..types import CanvasSpec

logger = logging.getLogger(__name__)

# Cap on the fetched ortho's long edge: Mapbox z18 already out-resolves a
# 300-DPI poster once warped, and a larger source just slows the mesh fetch.
_MAX_ORTHO_PX = 4096
# Mesh grid over the poster. The oblique+warp deformation is smooth and
# low-frequency, so a coarse grid is exact at nodes and near-exact between.
_MESH_COLS = 80
_MESH_ROWS = 120


class SatelliteBaseBuilder:
    """Builds the warped, poster-space satellite base for one plan."""

    def __init__(
        self,
        frame: GeoFrame,
        camera: ObliqueCamera,
        canvas: CanvasSpec,
        warp,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
        zoom: Optional[int] = None,
    ):
        self.frame = frame
        self.camera = camera
        self.canvas = canvas
        self.warp = warp
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.token = token
        self.zoom = zoom
        self._env = frame.fetch_region
        self._ortho: Optional[Image.Image] = None

    # -- public ----------------------------------------------------------

    @classmethod
    def from_plan(cls, plan, cache_dir: Optional[Path] = None, token: Optional[str] = None):
        """Reconstruct the builder from a PlanDocument's stored frame/warp."""
        frame = GeoFrame.from_dict(plan.frame)
        camera = ObliqueCamera(plan.camera, plan.canvas)
        warp = warp_from_dict(plan.warp)
        return cls(
            frame,
            camera,
            plan.canvas,
            warp,
            cache_dir=cache_dir,
            token=token,
            zoom=plan.style.satellite_zoom,
        )

    def poster_base(self, scale: float = 1.0) -> Optional[Image.Image]:
        """The satellite base in poster space at ``scale``, masked to the map
        trapezoid (RGBA, transparent sky). Returns None on any fetch failure so
        the compositor can fall back to illustrated land."""
        cached = self._cached_poster(scale)
        if cached is not None:
            return cached
        ortho = self._fetch_ortho()
        if ortho is None:
            return None
        warped = self._warp_to_poster(ortho, scale)
        self._write_cache(warped, scale)
        return warped

    # -- fetch -----------------------------------------------------------

    def _fetch_ortho(self) -> Optional[Image.Image]:
        if self._ortho is not None:
            return self._ortho
        try:
            from ...services.satellite_service import SatelliteService
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("Satellite base unavailable (import failed): %s", exc)
            return None
        try:
            sat_cache = str(self.cache_dir) if self.cache_dir else None
            service = SatelliteService(access_token=self.token, cache_dir=sat_cache)
        except ValueError as exc:
            # No Mapbox token -> graceful fallback to illustrated land.
            logger.warning("Satellite base disabled: %s", exc)
            return None
        bbox = BoundingBox(
            north=self._env.north,
            south=self._env.south,
            east=self._env.east,
            west=self._env.west,
        )
        size = self._ortho_size()
        try:
            img = service.fetch_satellite_imagery(bbox, zoom=self.zoom, output_size=size)
        except Exception as exc:
            logger.warning("Satellite imagery fetch failed: %s", exc)
            return None
        finally:
            service.close()
        self._ortho = img.convert("RGB")
        return self._ortho

    def _ortho_size(self) -> tuple[int, int]:
        """Ortho pixel size matching the envelope aspect, capped at _MAX_ORTHO_PX."""
        w_deg = max(1e-9, self._env.width_deg)
        h_deg = max(1e-9, self._env.height_deg)
        if w_deg >= h_deg:
            w = _MAX_ORTHO_PX
            h = max(1, int(round(_MAX_ORTHO_PX * h_deg / w_deg)))
        else:
            h = _MAX_ORTHO_PX
            w = max(1, int(round(_MAX_ORTHO_PX * w_deg / h_deg)))
        return (w, h)

    # -- warp ------------------------------------------------------------

    def _warp_to_poster(self, ortho: Image.Image, scale: float) -> Image.Image:
        out_w = max(1, int(round(self.canvas.width_px * scale)))
        out_h = max(1, int(round(self.canvas.height_px * scale)))

        # Grid of destination (poster) node coordinates, in *unscaled* poster
        # space so the camera math matches the plan.
        xs = np.linspace(0.0, self.canvas.width_px, _MESH_COLS + 1)
        ys = np.linspace(0.0, self.canvas.height_px, _MESH_ROWS + 1)
        gx, gy = np.meshgrid(xs, ys)  # (rows+1, cols+1)
        sx, sy = self._poster_to_ortho_px(gx, gy, ortho.size)

        mesh = []
        for r in range(_MESH_ROWS):
            for c in range(_MESH_COLS):
                # Destination rectangle (scaled to the output canvas). PIL needs
                # integer destination boxes; the source quad stays float.
                dx0, dx1 = int(round(xs[c] * scale)), int(round(xs[c + 1] * scale))
                dy0, dy1 = int(round(ys[r] * scale)), int(round(ys[r + 1] * scale))
                # Source quad in PIL order: UL, LL, LR, UR.
                quad = (
                    sx[r, c], sy[r, c],
                    sx[r + 1, c], sy[r + 1, c],
                    sx[r + 1, c + 1], sy[r + 1, c + 1],
                    sx[r, c + 1], sy[r, c + 1],
                )
                mesh.append(((dx0, dy0, dx1, dy1), quad))

        warped = ortho.transform(
            (out_w, out_h), Image.Transform.MESH, mesh, Image.Resampling.BILINEAR
        ).convert("RGBA")
        warped.putalpha(self._trapezoid_mask(out_w, out_h, scale))
        return warped

    def _poster_to_ortho_px(self, px, py, ortho_size):
        """Vectorized inverse map: poster px -> ortho px (NaN-safe)."""
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
        # flat-space coords, then warped-normalized.
        flat_y = t * cam.flat_height
        width_scale = spec.convergence + (1.0 - spec.convergence) * t
        cx = cam.flat_width / 2.0
        flat_x = cx + (px - cx) / np.where(width_scale == 0, 1e-9, width_scale)
        wu = np.clip(flat_x / cam.flat_width, 0.0, 1.0)
        wv = np.clip(flat_y / cam.flat_height, 0.0, 1.0)
        # warp inverse: warped-normalized -> normalized.
        u, v = self._unwarp(wu, wv)
        lon, lat = self._frame_from_normalized(u, v)
        ow, oh = ortho_size
        ex = (lon - self._env.west) / max(1e-9, self._env.width_deg) * ow
        ey = (self._env.north - lat) / max(1e-9, self._env.height_deg) * oh
        return ex, ey

    def _unwarp(self, wu, wv):
        warp = self.warp
        if not hasattr(warp, "_fx"):  # IdentityWarp
            return wu, wv
        u = np.interp(wu, warp._fx, warp._grid)
        v = np.interp(wv, warp._fy, warp._grid)
        return u, v

    def _frame_from_normalized(self, u, v):
        """Vectorized GeoFrame.from_normalized over numpy arrays."""
        f = self.frame
        x = f._x0 + u * (f._x1 - f._x0)
        y = f._y0 + v * (f._y1 - f._y0)
        e = x * f._cos_b - y * f._sin_b
        n = -(x * f._sin_b + y * f._cos_b)
        return (f._lon_c + e / f._kx, f._lat_c + n / f._ky)

    def _trapezoid_mask(self, w: int, h: int, scale: float) -> Image.Image:
        """Alpha mask matching the camera's converging far edge (the same
        trapezoid _draw_base_land paints), so the sky stays empty."""
        from PIL import ImageDraw

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

    # -- cache -----------------------------------------------------------

    def _cache_key(self, scale: float) -> str:
        import hashlib

        payload = {
            "frame": self.frame.to_dict(),
            "camera": self.camera.spec.model_dump(),
            "canvas": self.canvas.model_dump(),
            "warp": getattr(self.warp, "to_dict", lambda: {})(),
            "zoom": self.zoom,
            "scale": round(scale, 4),
        }
        import json

        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _cache_path(self, scale: float) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"poster_{self._cache_key(scale)}.png"

    def _cached_poster(self, scale: float) -> Optional[Image.Image]:
        path = self._cache_path(scale)
        if path and path.exists():
            try:
                return Image.open(path).convert("RGBA")
            except Exception:  # pragma: no cover - corrupt cache
                return None
        return None

    def _write_cache(self, image: Image.Image, scale: float) -> None:
        path = self._cache_path(scale)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
