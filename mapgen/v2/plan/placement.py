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


def sized_slot(slot: PoiSlot, canvas_width_px: int, depth_scale: float) -> PoiSlot:
    """Assign sprite dimensions from tier, shrunk with distance."""
    w = TIER_WIDTH_FRACTIONS[slot.tier] * canvas_width_px * depth_scale
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
) -> list[PoiSlot]:
    """Iteratively push overlapping sprite boxes apart along the smaller axis.

    Higher tiers (lower tier number) are heavier and move less. Anchors are
    clamped so sprites stay fully on the canvas.
    """
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
