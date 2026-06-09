"""Label planning: what text goes where, decided in poster space.

V2.0 uses greedy placement: labels are sorted by priority and placed
unless their estimated box collides with an already-placed label or a POI
sprite box. Street labels ride a sub-path of their (already projected,
already smoothed) road centerline; POI labels sit under their sprite.

All text is rendered by the compositor with real fonts -- the image
model never draws text.
"""

from __future__ import annotations

import math

from ..types import CanvasSpec, LabelKind, LabelSpec, PoiSlot, Point, RoadClass, RoadPath
from .stylize import polyline_length

# Font sizes as fraction of poster width.
FONT_FRACTIONS: dict[LabelKind, float] = {
    LabelKind.TITLE: 0.045,
    LabelKind.DISTRICT: 0.016,
    LabelKind.WATER: 0.014,
    LabelKind.POI: 0.011,
    LabelKind.STREET: 0.0085,
}

STREET_LABEL_CLASSES = {RoadClass.MOTORWAY, RoadClass.PRIMARY, RoadClass.SECONDARY}
CHAR_WIDTH_FACTOR = 0.62  # rough average glyph width / font size


def estimate_label_box(label: LabelSpec) -> tuple[float, float, float, float]:
    """Axis-aligned (x0, y0, x1, y1) box around the baseline midpoint."""
    width = len(label.text) * label.font_size_px * CHAR_WIDTH_FACTOR
    mid = _baseline_midpoint(label.baseline)
    return (
        mid[0] - width / 2,
        mid[1] - label.font_size_px,
        mid[0] + width / 2,
        mid[1] + label.font_size_px,
    )


def _baseline_midpoint(baseline: list[Point]) -> Point:
    if len(baseline) == 1:
        return baseline[0]
    target = polyline_length(baseline) / 2
    walked = 0.0
    for a, b in zip(baseline, baseline[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if walked + seg >= target and seg > 0:
            f = (target - walked) / seg
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        walked += seg
    return baseline[-1]


def _boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _street_baseline(road: RoadPath, text_len_px: float) -> list[Point] | None:
    """Centered sub-path of the road long enough to carry the text."""
    total = polyline_length(road.points)
    if total < text_len_px * 1.15:
        return None
    start = (total - text_len_px) / 2
    end = start + text_len_px
    out: list[Point] = []
    walked = 0.0
    for a, b in zip(road.points, road.points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg == 0:
            continue
        if walked + seg >= start and not out:
            f = (start - walked) / seg
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        if out:
            if walked + seg >= end:
                f = (end - walked) / seg
                out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                break
            out.append(b)
        walked += seg
    if len(out) >= 2 and out[0][0] > out[-1][0]:
        out.reverse()  # keep text left-to-right
    return out if len(out) >= 2 else None


def plan_labels(
    canvas: CanvasSpec,
    roads: list[RoadPath],
    pois: list[PoiSlot],
    districts: list[tuple[str, Point]],
    water_names: list[tuple[str, Point]],
    title: str,
) -> list[LabelSpec]:
    candidates: list[LabelSpec] = []
    w = canvas.width_px

    candidates.append(
        LabelSpec(
            kind=LabelKind.TITLE,
            text=title,
            baseline=[(w * 0.5, canvas.height_px * 0.045)],
            font_size_px=FONT_FRACTIONS[LabelKind.TITLE] * w,
            priority=1.0,
        )
    )

    for slot in pois:
        size = FONT_FRACTIONS[LabelKind.POI] * w * (1.25 if slot.tier == 1 else 1.0)
        candidates.append(
            LabelSpec(
                kind=LabelKind.POI,
                text=slot.name,
                baseline=[(slot.anchor[0], slot.anchor[1] + size * 1.4)],
                font_size_px=size,
                priority=0.9 - 0.05 * slot.tier,
            )
        )

    for name, pos in districts:
        candidates.append(
            LabelSpec(
                kind=LabelKind.DISTRICT,
                text=name.upper(),
                baseline=[pos],
                font_size_px=FONT_FRACTIONS[LabelKind.DISTRICT] * w,
                priority=0.6,
            )
        )

    for name, pos in water_names:
        candidates.append(
            LabelSpec(
                kind=LabelKind.WATER,
                text=name,
                baseline=[pos],
                font_size_px=FONT_FRACTIONS[LabelKind.WATER] * w,
                priority=0.55,
            )
        )

    size = FONT_FRACTIONS[LabelKind.STREET] * w
    seen_streets: set[str] = set()
    for road in roads:
        if road.cls not in STREET_LABEL_CLASSES or not road.name or road.name in seen_streets:
            continue
        baseline = _street_baseline(road, len(road.name) * size * CHAR_WIDTH_FACTOR)
        if baseline is None:
            continue
        seen_streets.add(road.name)
        candidates.append(
            LabelSpec(
                kind=LabelKind.STREET,
                text=road.name,
                baseline=baseline,
                font_size_px=size,
                priority=0.3 + 0.02 * len(road.points),
            )
        )

    # Greedy placement: high priority first, drop anything that collides
    # with placed labels or with POI sprite boxes.
    placed: list[LabelSpec] = []
    blocked = [
        (s.anchor[0] - s.width_px / 2, s.anchor[1] - s.height_px, s.anchor[0] + s.width_px / 2, s.anchor[1])
        for s in pois
    ]
    for label in sorted(candidates, key=lambda l: l.priority, reverse=True):
        box = estimate_label_box(label)
        obstacles = blocked if label.kind not in (LabelKind.POI, LabelKind.TITLE) else []
        if any(_boxes_intersect(box, estimate_label_box(p)) for p in placed):
            continue
        if any(_boxes_intersect(box, ob) for ob in obstacles):
            continue
        placed.append(label)
    return placed
