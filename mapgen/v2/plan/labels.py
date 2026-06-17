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

from ..types import CanvasSpec, LabelKind, LabelSpec, PoiSlot, Point, RoadClass, RoadPath, Typography
from .stylize import polyline_length

# Font sizes as fraction of poster width.
FONT_FRACTIONS: dict[LabelKind, float] = {
    LabelKind.TITLE: 0.055,
    LabelKind.DISTRICT: 0.018,
    LabelKind.WATER: 0.015,
    LabelKind.POI: 0.015,
    LabelKind.STREET: 0.0085,
    LabelKind.SHIELD: 0.011,
}

STREET_LABEL_CLASSES = {RoadClass.MOTORWAY, RoadClass.PRIMARY, RoadClass.SECONDARY}
# Street labels are picked by class first (a motorway name beats any long
# back road), with length only as a tiebreaker inside a class.
STREET_PRIORITY = {
    RoadClass.MOTORWAY: 0.45,
    RoadClass.PRIMARY: 0.40,
    RoadClass.SECONDARY: 0.34,
}
CHAR_WIDTH_FACTOR = 0.62  # rough average glyph width / font size
MAJOR_CITY_POPULATION = 300_000  # cities this big are never dropped


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


def _oob_area(box: tuple[float, float, float, float], w: float, h: float) -> float:
    """Area of the label box lying outside the canvas [0,w] x [0,h]."""
    inside = max(0.0, min(box[2], w) - max(box[0], 0.0)) * max(0.0, min(box[3], h) - max(box[1], 0.0))
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]) - inside)


def _shift_into_bounds(baseline: list[Point], box: tuple[float, float, float, float], w: float, h: float) -> list[Point]:
    dx = -box[0] if box[0] < 0 else (w - box[2] if box[2] > w else 0.0)
    dy = -box[1] if box[1] < 0 else (h - box[3] if box[3] > h else 0.0)
    if dx == 0.0 and dy == 0.0:
        return baseline
    return [(p[0] + dx, p[1] + dy) for p in baseline]


def _centered_subpath(points: list[Point], text_len_px: float) -> list[Point] | None:
    """Centered sub-path of a polyline long enough to carry the text."""
    total = polyline_length(points)
    if total < text_len_px * 1.15:
        return None
    start = (total - text_len_px) / 2
    end = start + text_len_px
    out: list[Point] = []
    walked = 0.0
    for a, b in zip(points, points[1:]):
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
    districts: list[tuple[str, Point, float | None]],
    water_names: list[tuple[str, list[Point]]],
    title: str,
    typography: Typography | None = None,
) -> list[LabelSpec]:
    candidates: list[LabelSpec] = []
    w = canvas.width_px
    h = canvas.height_px
    typo = typography or Typography()

    def font_px(kind: LabelKind) -> float:
        """Base font size for a kind in pixels, honouring typography overrides:
        a per-kind size fraction overrides the FONT_FRACTIONS default, and the
        global scale multiplies everything."""
        fraction = typo.for_kind(kind).size or FONT_FRACTIONS[kind]
        return fraction * typo.scale * w

    candidates.append(
        LabelSpec(
            kind=LabelKind.TITLE,
            text=title,
            baseline=[(w * 0.5, canvas.height_px * 0.045)],
            font_size_px=font_px(LabelKind.TITLE),
            priority=1.0,
        )
    )

    # POI labels are the product: they are never dropped. Each gets a list
    # of fallback positions (below, above, right, left of the sprite); the
    # placement loop tries them in order and, if all collide, keeps the
    # "below" position anyway.
    # One consistent, readable POI label size (no per-tier scaling) so a dense
    # cluster reads as a tidy set rather than a jumble of mismatched sizes.
    poi_size = font_px(LabelKind.POI)
    poi_alternates: dict[int, list[list[Point]]] = {}
    poi_own_box: dict[int, tuple[float, float, float, float]] = {}
    never_drop: set[int] = set()  # labels placed even if every position collides
    for slot in pois:
        size = poi_size
        text_w = len(slot.name) * size * CHAR_WIDTH_FACTOR
        ax, ay = slot.anchor
        hw, hh = slot.width_px / 2, slot.height_px
        gap = size * 0.6
        rx = hw + text_w / 2 + gap  # horizontal reach to clear the sprite
        # Candidate baselines, tried in order: directly below, above, the four
        # sides, then the same pushed out further. The sprite is already placed
        # (and, for clusters, offset by a leader), so more options means a label
        # almost always finds a clear berth.
        positions = [
            [(ax, ay + size * 1.4)],
            [(ax, ay - hh - size * 0.8)],
            [(ax + rx, ay - hh * 0.4)],
            [(ax - rx, ay - hh * 0.4)],
            [(ax + rx, ay + size * 1.4)],
            [(ax - rx, ay + size * 1.4)],
            [(ax, ay + size * 2.8)],
            [(ax, ay - hh - size * 2.0)],
            [(ax + rx * 1.4, ay - hh * 0.4)],
            [(ax - rx * 1.4, ay - hh * 0.4)],
        ]
        label = LabelSpec(
            kind=LabelKind.POI,
            text=slot.name,
            baseline=positions[0],
            font_size_px=size,
            priority=0.9 - 0.05 * slot.tier,
        )
        poi_alternates[id(label)] = positions
        poi_own_box[id(label)] = (
            ax - hw, ay - slot.height_px, ax + hw, ay,
        )
        never_drop.add(id(label))
        candidates.append(label)

    # City/town names, sized and prioritised by population. The major cities
    # (Phoenix, Tucson, ...) are guaranteed a place near their centre like POI
    # labels; smaller towns are dropped if they collide.
    district_base = font_px(LabelKind.DISTRICT)
    for name, pos, population in districts:
        pop = population or 0.0
        factor = max(0.8, min(1.9, 0.6 + 0.0009 * (pop**0.5))) if pop else 0.8
        size = district_base * factor
        label = LabelSpec(
            kind=LabelKind.DISTRICT,
            text=name.upper(),
            baseline=[pos],
            font_size_px=size,
            priority=0.6 + min(0.28, pop / 6.0e6),
        )
        if pop >= MAJOR_CITY_POPULATION:
            ax, ay = pos
            poi_alternates[id(label)] = [
                [pos],
                [(ax, ay - size * 1.6)],
                [(ax, ay + size * 1.6)],
                [(ax + size * 4.0, ay)],
                [(ax - size * 4.0, ay)],
                [(ax, ay - size * 3.4)],
                [(ax, ay + size * 3.4)],
            ]
            poi_own_box[id(label)] = None
            never_drop.add(id(label))
        candidates.append(label)

    for name, pts in water_names:
        size = font_px(LabelKind.WATER)
        baseline = [pts[0]] if len(pts) < 2 else (
            _centered_subpath(pts, len(name) * size * CHAR_WIDTH_FACTOR)
            or [pts[len(pts) // 2]]
        )
        candidates.append(
            LabelSpec(
                kind=LabelKind.WATER,
                text=name,
                baseline=baseline,
                font_size_px=size,
                priority=0.55,
            )
        )

    size = font_px(LabelKind.STREET)
    seen_streets: set[str] = set()
    for road in roads:
        if road.cls not in STREET_LABEL_CLASSES or not road.name or road.name in seen_streets:
            continue
        baseline = _centered_subpath(road.points, len(road.name) * size * CHAR_WIDTH_FACTOR)
        if baseline is None:
            continue
        seen_streets.add(road.name)
        candidates.append(
            LabelSpec(
                kind=LabelKind.STREET,
                text=road.name,
                baseline=baseline,
                font_size_px=size,
                priority=STREET_PRIORITY.get(road.cls, 0.3)
                + min(0.03, 0.0001 * polyline_length(road.points)),
            )
        )

    # Highway shields: one badge per route designation, on the longest road
    # carrying it across the major network, placed at that road's midpoint.
    shield_size = font_px(LabelKind.SHIELD)
    best_by_ref: dict[str, RoadPath] = {}
    for road in roads:
        if road.cls not in (RoadClass.MOTORWAY, RoadClass.PRIMARY) or not road.ref:
            continue
        # OSM refs can be ";"-joined ("US 60;AZ 77") or carry qualifiers
        # ("AZ 202 Loop", "Future AZ 24"). Keep the primary route, two tokens.
        raw = str(road.ref).split(";")[0].replace("Future ", "").strip()
        ref = " ".join(raw.split()[:2])  # "AZ 202 Loop" -> "AZ 202"
        if not ref:
            continue
        cur = best_by_ref.get(ref)
        if cur is None or polyline_length(road.points) > polyline_length(cur.points):
            best_by_ref[ref] = road
    for ref, road in best_by_ref.items():
        candidates.append(
            LabelSpec(
                kind=LabelKind.SHIELD,
                text=ref,
                baseline=[_baseline_midpoint(road.points)],
                font_size_px=shield_size,
                priority=0.5,
                network=road.network,
            )
        )

    # Greedy placement: high priority first. POI labels try their fallback
    # positions and are placed even if every position collides; everything
    # else is dropped on collision with placed labels or POI sprite boxes.
    placed: list[LabelSpec] = []
    blocked = [
        (s.anchor[0] - s.width_px / 2, s.anchor[1] - s.height_px, s.anchor[0] + s.width_px / 2, s.anchor[1])
        for s in pois
    ]
    def _overlap_area(a, b):
        ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ox * oy

    for label in sorted(candidates, key=lambda l: l.priority, reverse=True):
        if id(label) in never_drop:
            own = poi_own_box.get(id(label))
            # Obstacles: every other sprite box (never our own) plus placed labels.
            sprite_obstacles = [b for b in blocked if b != own]
            chosen = None
            best_cost = None
            for baseline in poi_alternates.get(id(label), [label.baseline]):
                attempt = label.model_copy(update={"baseline": baseline})
                box = estimate_label_box(attempt)
                cost = sum(_overlap_area(box, estimate_label_box(p)) for p in placed)
                cost += sum(_overlap_area(box, ob) for ob in sprite_obstacles)
                cost += _oob_area(box, w, h) * 6.0  # strongly prefer on-canvas
                if cost == 0:
                    chosen = attempt
                    break
                if best_cost is None or cost < best_cost:
                    best_cost, chosen = cost, attempt  # least-bad fallback
            if chosen is not None:  # a never-dropped label must still fit on the map
                cbox = estimate_label_box(chosen)
                if _oob_area(cbox, w, h) > 0:
                    chosen = chosen.model_copy(
                        update={"baseline": _shift_into_bounds(chosen.baseline, cbox, w, h)}
                    )
            placed.append(chosen or label)
            continue
        box = estimate_label_box(label)
        if _oob_area(box, w, h) > 0:  # nudge edge labels in; drop if they then collide
            label = label.model_copy(update={"baseline": _shift_into_bounds(label.baseline, box, w, h)})
            box = estimate_label_box(label)
        obstacles = blocked if label.kind != LabelKind.TITLE else []
        if any(_boxes_intersect(box, estimate_label_box(p)) for p in placed):
            continue
        if any(_boxes_intersect(box, ob) for ob in obstacles):
            continue
        placed.append(label)
    return placed
