"""POI placement: tier sizing and overlap resolution in poster space.

POIs keep their ground-contact anchor as close to the geographic truth
as possible, but their illustrated sprites must never overlap each other.
A simple iterative pairwise-repulsion solver (same approach as V1's
collision_service) nudges sprite boxes apart, preferring to move
lower-tier POIs.
"""

from __future__ import annotations

from ..types import PoiSlot

# Sprite footprint as fraction of poster width, per importance tier.
TIER_WIDTH_FRACTIONS = {1: 0.13, 2: 0.095, 3: 0.065}
SPRITE_ASPECT = 1.15  # height / width; oblique buildings are a bit taller

# Some landmark TYPES dominate the landscape far more than tier alone implies
# (an international airport vs. a single museum building). Scale the sprite by
# the feature's real-world footprint so the map reads as accurate.
SIZE_BY_FEATURE = {
    "airport": 1.7,
    "stadium": 1.35,
    "campus": 1.25,
    "forest": 1.35,
    "mountain": 1.25,
    "park": 1.15,
    "zoo": 1.1,
}


def tier_demand(tier: int, size_scale: float = 1.0) -> float:
    """Normalized sprite half-width 'room demand' for a tier.

    Shared by the warp-fit cartogram (which weights its magnification by how
    much room each POI's sprite needs) and the leader-line coincidence test.
    Expressed as a fraction of poster width, so it is canvas-size agnostic.
    """
    return TIER_WIDTH_FRACTIONS[tier] * size_scale / 2.0


def sized_slot(
    slot: PoiSlot, canvas_width_px: int, depth_scale: float, size_scale: float = 1.0
) -> PoiSlot:
    """Assign sprite dimensions from tier, shrunk with distance.

    ``size_scale`` is a global multiplier the plan builder uses to shrink
    all sprites together when a dense POI cluster cannot fit at full size.
    """
    fmul = SIZE_BY_FEATURE.get(slot.feature_type, 1.0)
    w = TIER_WIDTH_FRACTIONS[slot.tier] * canvas_width_px * depth_scale * size_scale * fmul
    return slot.model_copy(update={"width_px": w, "height_px": w * SPRITE_ASPECT})


def _boxes_overlap(a: PoiSlot, b: PoiSlot, padding: float) -> tuple[float, float] | None:
    """Return (dx, dy) overlap extents if the sprite boxes overlap, else None.

    A slot's box is centered on its anchor horizontally and sits above the
    anchor (ground contact at the bottom edge).
    """
    ax0, ax1 = a.anchor[0] - a.width_px / 2, a.anchor[0] + a.width_px / 2
    ay0, ay1 = a.anchor[1] - a.height_px, a.anchor[1]
    bx0, bx1 = b.anchor[0] - b.width_px / 2, b.anchor[0] + b.width_px / 2
    by0, by1 = b.anchor[1] - b.height_px, b.anchor[1]
    dx = min(ax1, bx1) - max(ax0, bx0) + padding
    dy = min(ay1, by1) - max(ay0, by0) + padding
    if dx > 0 and dy > 0:
        return (dx, dy)
    return None


def resolve_poi_overlaps(
    slots: list[PoiSlot],
    canvas_width_px: int,
    canvas_height_px: int,
    padding: float = 12.0,
    max_iterations: int = 80,
    reserved: list[tuple[float, float, float, float]] | None = None,
) -> list[PoiSlot]:
    """Iteratively push overlapping sprite boxes apart along the smaller axis.

    Higher tiers (lower tier number) are heavier and move less. Anchors are
    clamped so sprites stay fully on the canvas. ``reserved`` rectangles
    (x0, y0, x1, y1) are top-anchored keep-out zones (e.g. the title
    cartouche): a sprite landing inside one is dropped just below it.
    """
    reserved = reserved or []
    slots = [s.model_copy() for s in slots]
    for _ in range(max_iterations):
        moved = False
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i], slots[j]
                overlap = _boxes_overlap(a, b, padding)
                if overlap is None:
                    continue
                moved = True
                dx, dy = overlap
                # Move along the axis of least separation work.
                wa = 1.0 / (4 - a.tier)  # tier 1 -> 1/3, tier 3 -> 1/1
                wb = 1.0 / (4 - b.tier)
                total = wa + wb
                if dx <= dy:
                    direction = 1.0 if a.anchor[0] >= b.anchor[0] else -1.0
                    a.anchor = (a.anchor[0] + direction * dx * wa / total, a.anchor[1])
                    b.anchor = (b.anchor[0] - direction * dx * wb / total, b.anchor[1])
                else:
                    direction = 1.0 if a.anchor[1] >= b.anchor[1] else -1.0
                    a.anchor = (a.anchor[0], a.anchor[1] + direction * dy * wa / total)
                    b.anchor = (b.anchor[0], b.anchor[1] - direction * dy * wb / total)
        for s in slots:
            # Drop sprites out of top-anchored reserved zones (e.g. cartouche).
            for rx0, ry0, rx1, ry1 in reserved:
                bx0, bx1 = s.anchor[0] - s.width_px / 2, s.anchor[0] + s.width_px / 2
                by0, by1 = s.anchor[1] - s.height_px, s.anchor[1]
                if bx0 < rx1 and rx0 < bx1 and by0 < ry1 and ry0 < by1:
                    s.anchor = (s.anchor[0], ry1 + s.height_px)
                    moved = True
            x = min(canvas_width_px - s.width_px / 2, max(s.width_px / 2, s.anchor[0]))
            y = min(float(canvas_height_px), max(s.height_px, s.anchor[1]))
            s.anchor = (x, y)
        if not moved:
            break
    return slots


def has_overlaps(slots: list[PoiSlot], padding: float = 0.0) -> bool:
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            if _boxes_overlap(slots[i], slots[j], padding):
                return True
    return False


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 properly crosses p3-p4 (collinear cases ignored)."""

    def ccw(a, b, c) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def assign_leader_lines(
    slots: list[PoiSlot],
    leader_threshold_px: float,
    max_uncross_iterations: int = 40,
    canvas_width_px: int | None = None,
    canvas_height_px: int | None = None,
) -> list[PoiSlot]:
    """Tag dishonestly-placed POIs for leader lines and tidy the connectors.

    Expects every slot's ``leader_anchor`` to already hold its *true* (warped)
    ground point, set before ``resolve_poi_overlaps`` moved ``anchor`` (the
    sprite base) into open space. A POI whose sprite ended up more than
    ``leader_threshold_px`` from its true point could not be placed honestly by
    the warp, so it keeps ``leader_anchor`` and is flagged ``offset`` -- a
    connector is drawn from the displaced sprite back to the true point. POIs
    only nudged within the threshold drop their ``leader_anchor`` and render
    exactly as before, with no connector.

    The overlap solver already separated the sprites, so it doubles as the
    offset placement; this pass only decides who gets a leader and then swaps
    crossing pairs' sprite positions (2-opt) so connectors don't tangle.
    """
    slots = [s.model_copy() for s in slots]
    for s in slots:
        true_pt = s.leader_anchor if s.leader_anchor is not None else s.anchor
        # Measure displacement against the on-canvas-*clamped* true point. A
        # sprite shifted only because its true point sits past the canvas edge
        # was clamped to stay visible (an honest, necessary move), not pushed by
        # a real coincidence -- so it must not earn a spurious leader.
        ref = true_pt
        if canvas_width_px is not None and canvas_height_px is not None:
            ref = (
                min(canvas_width_px - s.width_px / 2, max(s.width_px / 2, true_pt[0])),
                min(float(canvas_height_px), max(s.height_px, true_pt[1])),
            )
        dx, dy = s.anchor[0] - ref[0], s.anchor[1] - ref[1]
        if (dx * dx + dy * dy) ** 0.5 > leader_threshold_px:
            s.leader_anchor = true_pt
            s.offset = True
        else:
            s.leader_anchor = None
            s.offset = False

    offset = sorted(
        (s for s in slots if s.offset), key=lambda s: (s.tier, s.id)
    )
    for _ in range(max_uncross_iterations):
        swapped = False
        for i in range(len(offset)):
            for j in range(i + 1, len(offset)):
                a, b = offset[i], offset[j]
                if not _segments_cross(a.anchor, a.leader_anchor, b.anchor, b.leader_anchor):
                    continue
                # Swapping the two sprites' positions uncrosses the pair while
                # each keeps its own true-point connector -- but only if it does
                # not push either sprite into another. Check against ALL slots:
                # the non-offset sprites stay put and are drawn too, so a swap
                # must not land on one of them either.
                a.anchor, b.anchor = b.anchor, a.anchor
                if any(
                    _boxes_overlap(s, other, 0.0)
                    for s in (a, b)
                    for other in slots
                    if other is not s
                ):
                    a.anchor, b.anchor = b.anchor, a.anchor  # revert
                    continue
                swapped = True
        if not swapped:
            break
    return slots
