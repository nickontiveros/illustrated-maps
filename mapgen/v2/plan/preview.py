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

MAX_PREVIEW_ROADS = 6000
MAX_PREVIEW_BUILDINGS = 4000


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

    # The preview is a layout-review tool, not an archive: cap the feature
    # count so the SVG stays loadable in a browser whatever the region size.
    # Roads are kept by (class priority, length); the cap is generous enough
    # that city-scale plans are never trimmed.
    roads = plan.roads
    if len(roads) > MAX_PREVIEW_ROADS:
        from .stylize import ROAD_PRIORITY, polyline_length

        roads = sorted(
            roads,
            key=lambda r: (ROAD_PRIORITY.get(r.cls, 0), polyline_length(r.points)),
            reverse=True,
        )[:MAX_PREVIEW_ROADS]
        out.append(f"<!-- preview truncated to {MAX_PREVIEW_ROADS} of {len(plan.roads)} roads -->")
    buildings = plan.buildings
    if len(buildings) > MAX_PREVIEW_BUILDINGS:
        buildings = buildings[:MAX_PREVIEW_BUILDINGS]
        out.append(
            f"<!-- preview truncated to {MAX_PREVIEW_BUILDINGS} of {len(plan.buildings)} buildings -->"
        )

    order = {cls: i for i, cls in enumerate(ROAD_DRAW_ORDER)}
    for road in sorted(roads, key=lambda r: order.get(r.cls, 0)):
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

    for b in buildings:
        out.append(
            f'<path d="{_path_d(b.polygon, close=True)}" fill="{palette["building_roof"]}" '
            f'stroke="{palette["outline"]}" stroke-width="1.5"/>'
        )

    for slot in plan.pois:
        # Coincident POIs are offset into open space with a connector back to
        # their true ground point; draw it (and the dot) under the sprite rect.
        if slot.offset and slot.leader_anchor is not None:
            tx, ty = slot.leader_anchor
            out.append(
                f'<line x1="{slot.anchor[0]:.0f}" y1="{slot.anchor[1]:.0f}" '
                f'x2="{tx:.0f}" y2="{ty:.0f}" stroke="{palette["outline"]}" '
                f'stroke-width="3" stroke-opacity="0.7"/>'
            )
            out.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="8" fill="{palette["outline"]}"/>'
            )
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
