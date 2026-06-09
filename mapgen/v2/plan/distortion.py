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
    ):
        self.strength = max(0.0, strength)
        grid = np.linspace(0.0, 1.0, samples)
        self._grid = grid
        self._fx = self._axis_map(grid, [c[0] for c in centers], radius)
        self._fy = self._axis_map(grid, [c[1] for c in centers], radius)

    def _axis_map(self, grid: np.ndarray, centers: list[float], radius: float) -> np.ndarray:
        density = np.ones_like(grid)
        for c in centers:
            density += self.strength * np.exp(-((grid - c) ** 2) / (2.0 * radius**2))
        cumulative = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2.0)])
        return cumulative / cumulative[-1]

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
