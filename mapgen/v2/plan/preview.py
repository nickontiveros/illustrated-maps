"""Free, instant SVG preview of a PlanDocument.

Lets the user review and adjust geometry, distortion, POI placement, and
labels before any AI spend. Rendered colors come from the plan's style
palette so the preview approximates the final composition.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from ..types import GroundClass, LabelKind, PlanDocument, Point, RoadClass

ROAD_DRAW_ORDER = [
    RoadClass.PATH,
    RoadClass.LOCAL,
    RoadClass.RAIL,
    RoadClass.SECONDARY,
    RoadClass.PRIMARY,
    RoadClass.MOTORWAY,
    RoadClass.STREAM,
    RoadClass.RIVER,
]


def _path_d(points: list[Point], close: bool = False) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    parts += [f"L {x:.1f} {y:.1f}" for x, y in points[1:]]
    if close:
        parts.append("Z")
    return " ".join(parts)


def plan_to_svg(plan: PlanDocument, scale: float = 0.12) -> str:
    palette = plan.style.palette
    w = plan.canvas.width_px * scale
    h = plan.canvas.height_px * scale
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {plan.canvas.width_px} {plan.canvas.height_px}">',
        f'<rect width="100%" height="100%" fill="{palette["paper"]}"/>',
    ]

    horizon = plan.camera.horizon_margin * plan.canvas.height_px
    out.append(
        f'<rect y="{horizon:.0f}" width="100%" height="{plan.canvas.height_px - horizon:.0f}" '
        f'fill="{palette["land"]}"/>'
    )

    for poly in sorted(plan.ground, key=lambda g: g.cls != GroundClass.WATER):
        fill = palette.get(poly.cls.value, palette["land"])
        d = _path_d(poly.exterior, close=True)
        for hole in poly.holes:
            d += " " + _path_d(hole, close=True)
        out.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="{palette["outline"]}" stroke-width="2" stroke-opacity="0.35"/>')

    order = {cls: i for i, cls in enumerate(ROAD_DRAW_ORDER)}
    for road in sorted(plan.roads, key=lambda r: order.get(r.cls, 0)):
        if road.cls in (RoadClass.RIVER, RoadClass.STREAM):
            color = palette["water"]
        elif road.cls == RoadClass.MOTORWAY:
            color = palette["motorway_fill"]
        elif road.cls == RoadClass.RAIL:
            color = palette["rail"]
        else:
            color = palette["road_fill"]
        out.append(
            f'<path d="{_path_d(road.points)}" fill="none" stroke="{palette["road_casing"]}" '
            f'stroke-width="{road.width_px + 3:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        out.append(
            f'<path d="{_path_d(road.points)}" fill="none" stroke="{color}" '
            f'stroke-width="{road.width_px:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    for b in plan.buildings:
        out.append(
            f'<path d="{_path_d(b.polygon, close=True)}" fill="{palette["building_roof"]}" '
            f'stroke="{palette["outline"]}" stroke-width="1.5"/>'
        )

    for slot in plan.pois:
        x0 = slot.anchor[0] - slot.width_px / 2
        y0 = slot.anchor[1] - slot.height_px
        out.append(
            f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{slot.width_px:.0f}" height="{slot.height_px:.0f}" '
            f'fill="{palette["building_wall"]}" fill-opacity="0.55" stroke="{palette["outline"]}" '
            f'stroke-width="3" stroke-dasharray="14 8" rx="12"/>'
        )

    for label in plan.labels:
        size = label.font_size_px
        mid = label.baseline[len(label.baseline) // 2]
        angle = 0.0
        if len(label.baseline) >= 2:
            a, b = label.baseline[0], label.baseline[-1]
            angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        weight = "bold" if label.kind in (LabelKind.TITLE, LabelKind.DISTRICT) else "normal"
        out.append(
            f'<text x="{mid[0]:.0f}" y="{mid[1]:.0f}" font-size="{size:.0f}" '
            f'font-family="Georgia, serif" font-weight="{weight}" fill="{palette["label"]}" '
            f'text-anchor="middle" transform="rotate({angle:.1f} {mid[0]:.0f} {mid[1]:.0f})">'
            f"{escape(label.text)}</text>"
        )

    out.append("</svg>")
    return "\n".join(out)
