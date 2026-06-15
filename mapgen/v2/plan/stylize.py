"""Vector stylization: simplify, smooth, exaggerate.

Illustrated-map roads are confident, smooth, wildly over-wide ribbons.
This module turns raw OSM polylines into that: Douglas-Peucker
simplification removes survey noise, Chaikin corner cutting rounds the
result into hand-drawable curves, and per-class width tables exaggerate
widths far beyond true scale (see RESEARCH.md, Factor 4).
"""

from __future__ import annotations

import math

from ..types import Point, RoadClass

# Width as a fraction of poster width. A 7016px-wide A1 poster gives
# motorways ~39px ribbons -- roughly 10-30x true scale, per the genre.
ROAD_WIDTH_FRACTIONS: dict[RoadClass, float] = {
    RoadClass.MOTORWAY: 0.0078,
    RoadClass.PRIMARY: 0.0058,
    RoadClass.SECONDARY: 0.0042,
    RoadClass.LOCAL: 0.0024,
    RoadClass.PATH: 0.0012,
    RoadClass.RAIL: 0.0016,
    RoadClass.RIVER: 0.0090,
    RoadClass.STREAM: 0.0032,
}

# Drawing/pruning priority (higher = more important).
ROAD_PRIORITY: dict[RoadClass, int] = {
    RoadClass.RIVER: 7,
    RoadClass.MOTORWAY: 6,
    RoadClass.PRIMARY: 5,
    RoadClass.SECONDARY: 4,
    RoadClass.RAIL: 3,
    RoadClass.STREAM: 3,
    RoadClass.LOCAL: 2,
    RoadClass.PATH: 1,
}


def road_width_px(cls: RoadClass, canvas_width_px: int) -> float:
    return max(2.0, ROAD_WIDTH_FRACTIONS[cls] * canvas_width_px)


def simplify_polyline(points: list[Point], tolerance: float) -> list[Point]:
    """Douglas-Peucker simplification."""
    if len(points) <= 2:
        return list(points)

    def _dp(pts: list[Point]) -> list[Point]:
        if len(pts) <= 2:
            return pts
        a, b = pts[0], pts[-1]
        max_d, idx = -1.0, 0
        for i in range(1, len(pts) - 1):
            d = _point_segment_distance(pts[i], a, b)
            if d > max_d:
                max_d, idx = d, i
        if max_d <= tolerance:
            return [a, b]
        left = _dp(pts[: idx + 1])
        right = _dp(pts[idx:])
        return left[:-1] + right

    return _dp(list(points))


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def chaikin_smooth(points: list[Point], iterations: int = 2, closed: bool = False) -> list[Point]:
    """Chaikin corner cutting: rounds polylines into smooth curves."""
    pts = list(points)
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        out: list[Point] = []
        if not closed:
            out.append(pts[0])
        pairs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
        for a, b in pairs:
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        if not closed:
            out.append(pts[-1])
        pts = out
    return pts


def stylize_polyline(
    points: list[Point],
    simplify_tolerance: float,
    smooth_iterations: int = 2,
    closed: bool = False,
    densify_px: float = 0.0,
) -> list[Point]:
    """Simplify -> densify -> smooth.

    Densifying between simplification and smoothing is load-bearing: Chaikin
    cuts a quarter off each segment at every corner, so on sparse rings
    (e.g. a 5-point bay polygon) it collapses the shape toward an ellipse.
    With short segments, corners are rounded only locally and the overall
    geometry is preserved.
    """
    from .camera import densify

    pts = simplify_polyline(points, simplify_tolerance)
    if densify_px > 0:
        if closed and len(pts) >= 3 and pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        pts = densify(pts, densify_px)
        if closed and len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
    return chaikin_smooth(pts, iterations=smooth_iterations, closed=closed)


def polyline_length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


# Fraction of the map plate the drawn road ribbons may cover. The Manhattan
# reference plan uses ~0.13, so city plans pass untouched; the unpruned
# Arizona plan measured 0.50 (an unreadable mat) and gets cut to budget.
ROAD_INK_BUDGET = 0.24

# Rivers and motorways are the map's skeleton; they are never dropped.
EXEMPT_PRIORITY = ROAD_PRIORITY[RoadClass.MOTORWAY]


def prune_roads(
    roads: list[tuple[RoadClass, list[Point]]],
    canvas_width_px: int,
    area_px: float | None = None,
    ink_budget: float = ROAD_INK_BUDGET,
    max_minor_roads: int = 400,
) -> list[tuple[RoadClass, list[Point]]]:
    """Keep the map breathing: cap total road ink, not just minor-road count.

    Ink is approximated as centerline length x class ribbon width. Roads are
    admitted by (class priority, length) until the budget is spent; rivers
    and motorways are always kept (their ink still counts), LOCAL/PATH keep
    the historical count cap and minimum length on top of the budget.
    """
    min_len = 0.015 * canvas_width_px
    if area_px is None:
        area_px = float(canvas_width_px) * canvas_width_px * 1.4

    candidates: list[tuple[int, float, RoadClass, list[Point]]] = []
    for cls, pts in roads:
        length = polyline_length(pts)
        if ROAD_PRIORITY[cls] < 3 and length < min_len:
            continue
        candidates.append((ROAD_PRIORITY[cls], length, cls, pts))
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    kept: list[tuple[RoadClass, list[Point]]] = []
    budget = ink_budget * area_px
    ink = 0.0
    minor_count = 0
    for priority, length, cls, pts in candidates:
        cost = length * road_width_px(cls, canvas_width_px)
        if priority >= EXEMPT_PRIORITY:
            kept.append((cls, pts))
            ink += cost
            continue
        if ink + cost > budget:
            continue
        if priority < 3:
            if minor_count >= max_minor_roads:
                continue
            minor_count += 1
        kept.append((cls, pts))
        ink += cost
    return kept
