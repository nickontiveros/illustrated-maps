"""Hand-drawn line treatment: smooth, seeded jitter applied to polylines.

The wobble is a low-frequency sinusoidal displacement along each point's
normal, seeded per element, so renders are reproducible and the same
element always wobbles the same way (preview and final agree).
"""

from __future__ import annotations

import math

from ..types import Point


def wobble_polyline(
    points: list[Point],
    amplitude: float,
    wavelength_px: float = 60.0,
    seed: float = 0.0,
) -> list[Point]:
    if amplitude <= 0 or len(points) < 3:
        return list(points)
    out: list[Point] = [points[0]]
    walked = 0.0
    for i in range(1, len(points) - 1):
        prev_pt, p, nxt = points[i - 1], points[i], points[i + 1]
        walked += math.hypot(p[0] - prev_pt[0], p[1] - prev_pt[1])
        # Normal from the local direction.
        dx, dy = nxt[0] - prev_pt[0], nxt[1] - prev_pt[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        phase = walked / wavelength_px * 2 * math.pi + seed
        offset = amplitude * (math.sin(phase) + 0.5 * math.sin(2.3 * phase + 1.7))
        out.append((p[0] + nx * offset, p[1] + ny * offset))
    out.append(points[-1])
    return out


def wobble_ring(points: list[Point], amplitude: float, wavelength_px: float = 60.0, seed: float = 0.0) -> list[Point]:
    if amplitude <= 0 or len(points) < 4:
        return list(points)
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    wobbled = wobble_polyline(closed, amplitude, wavelength_px, seed)
    return wobbled[:-1]
