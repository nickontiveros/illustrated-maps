"""Selective geographic distortion.

Tourist maps give important areas more room than they deserve and
compress empty stretches. V2 implements this as a smooth, separable,
monotonic warp of normalized flat coordinates: an importance density is
accumulated per axis from Gaussian bumps centered on the POIs, and each
axis is remapped by the normalized cumulative integral of its density.

Separable axis warps cannot fold or self-intersect, so road topology is
always preserved.
"""

from __future__ import annotations

import numpy as np

from ..types import Point


class ImportanceWarp:
    def __init__(
        self,
        centers: list[Point],  # normalized (0..1, 0..1) importance centers
        strength: float = 0.6,  # 0 = identity; ~1 = strong magnification
        radius: float = 0.18,  # gaussian sigma in normalized units
        samples: int = 1024,
        weights: list[float] | None = None,  # per-center demand (default 1.0)
        radii: list[float] | None = None,  # per-center sigma (default `radius`)
        bands: tuple[list, list] | None = None,  # per-axis plateaus (lo,hi,weight)
    ):
        self.strength = max(0.0, strength)
        grid = np.linspace(0.0, 1.0, samples)
        self._grid = grid
        if bands is not None:
            # Flat-topped magnification over each cluster's range: density is
            # constant inside, so the CDF is linear and the warp is *affine*
            # there -- straight roads stay straight, with bending confined to
            # the soft plateau edges.
            bx, by = bands
            self._fx = self._cdf(self._plateau_density(grid, bx))
            self._fy = self._cdf(self._plateau_density(grid, by))
        else:
            n = len(centers)
            weights = [1.0] * n if weights is None else weights
            radii = [radius] * n if radii is None else radii
            self._fx = self._cdf(self._gauss_density(grid, [c[0] for c in centers], weights, radii))
            self._fy = self._cdf(self._gauss_density(grid, [c[1] for c in centers], weights, radii))

    @staticmethod
    def _cdf(density: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2.0)])
        return cumulative / cumulative[-1]

    def _gauss_density(self, grid, centers, weights, radii) -> np.ndarray:
        density = np.ones_like(grid)
        for c, w, r in zip(centers, weights, radii):
            density += self.strength * w * np.exp(-((grid - c) ** 2) / (2.0 * r**2))
        return density

    def _plateau_density(self, grid, bands) -> np.ndarray:
        density = np.ones_like(grid)
        for lo, hi, w in bands:
            soft = max(0.04, (hi - lo) * 0.5)  # smooth transition width
            up = self._smoothstep((grid - (lo - soft)) / soft)
            down = 1.0 - self._smoothstep((grid - hi) / soft)
            density += self.strength * w * np.clip(np.minimum(up, down), 0.0, 1.0)
        return density

    @staticmethod
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def warp_point(self, p: Point) -> Point:
        u = float(np.interp(p[0], self._grid, self._fx))
        v = float(np.interp(p[1], self._grid, self._fy))
        return (u, v)

    def warp_points(self, points: list[Point]) -> list[Point]:
        if not points:
            return []
        arr = np.asarray(points, dtype=float)
        u = np.interp(arr[:, 0], self._grid, self._fx)
        v = np.interp(arr[:, 1], self._grid, self._fy)
        return list(zip(u.tolist(), v.tolist()))


class IdentityWarp:
    """No-op warp with the same interface."""

    def warp_point(self, p: Point) -> Point:
        return p

    def warp_points(self, points: list[Point]) -> list[Point]:
        return list(points)
