"""Small pure-Python polygon helpers used by the plan engine."""

from __future__ import annotations

from ..types import Point


def polygon_area(ring: list[Point]) -> float:
    """Unsigned area via the shoelace formula."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def polygon_bounds(ring: list[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_polygon(p: Point, ring: list[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = p
    inside = False
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if x < x_cross:
                inside = not inside
    return inside


def polygon_centroid(ring: list[Point]) -> Point:
    if not ring:
        return (0.0, 0.0)
    if len(ring) < 3:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    area_acc = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        area_acc += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area_acc) < 1e-9:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return (cx / (3.0 * area_acc), cy / (3.0 * area_acc))
