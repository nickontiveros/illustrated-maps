"""Font loading and text-on-path rendering for the compositor.

All map text is rendered with real fonts -- the image model never draws
text. Curved street labels are rendered glyph-by-glyph along their
baseline polyline; straight labels take a fast whole-string path.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..types import Point

# Bundled hand-lettering font (SIL OFL, see fonts/OFL.txt), then sturdy
# fallbacks that exist on most systems.
BUNDLED_FONT = Path(__file__).parent / "fonts" / "Caveat-Variable.ttf"
FONT_CANDIDATES = [
    str(BUNDLED_FONT),
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]


def load_font(
    size: int,
    font_path: str | None = None,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ([font_path] if font_path else []) + FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                font = ImageFont.truetype(candidate, size)
            except OSError:
                continue
            if bold:
                try:
                    font.set_variation_by_axes([700])
                except OSError:
                    pass
            return font
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _position_at(baseline: list[Point], distance: float) -> tuple[Point, float]:
    """Point and tangent angle at a distance along the polyline."""
    walked = 0.0
    for a, b in zip(baseline, baseline[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg == 0:
            continue
        if walked + seg >= distance:
            f = (distance - walked) / seg
            pt = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            return pt, math.atan2(b[1] - a[1], b[0] - a[0])
        walked += seg
    a, b = baseline[-2], baseline[-1]
    return baseline[-1], math.atan2(b[1] - a[1], b[0] - a[0])


def draw_text_on_path(
    canvas: Image.Image,
    text: str,
    baseline: list[Point],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    halo: tuple[int, int, int] | None = None,
    anchor_middle: bool = True,
) -> None:
    """Render text along a polyline (or centered at a single point)."""
    if len(baseline) < 2:
        _draw_straight(canvas, text, baseline[0], 0.0, font, fill, halo)
        return

    total_angle_change = _total_turn(baseline)
    if total_angle_change < math.radians(8):
        # Straight enough: render the whole string rotated at the midpoint.
        mid, angle = _position_at(baseline, _length(baseline) / 2)
        _draw_straight(canvas, text, mid, angle, font, fill, halo)
        return

    # Curved: glyph by glyph.
    widths = [_text_width(font, ch) for ch in text]
    text_w = sum(widths)
    start = max(0.0, (_length(baseline) - text_w) / 2) if anchor_middle else 0.0
    walked = start
    for ch, w in zip(text, widths):
        pt, angle = _position_at(baseline, walked + w / 2)
        _draw_straight(canvas, ch, pt, angle, font, fill, halo)
        walked += w


def _length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _total_turn(points: list[Point]) -> float:
    total = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        a1 = math.atan2(b[1] - a[1], b[0] - a[0])
        a2 = math.atan2(c[1] - b[1], c[0] - b[0])
        diff = abs(a2 - a1)
        total += min(diff, 2 * math.pi - diff)
    return total


def _text_width(font, text: str) -> float:
    try:
        return font.getlength(text)
    except AttributeError:
        return font.getbbox(text)[2]


def _draw_straight(
    canvas: Image.Image,
    text: str,
    center: Point,
    angle_rad: float,
    font,
    fill: tuple[int, int, int],
    halo: tuple[int, int, int] | None,
) -> None:
    """Draw rotated text centered at a point onto an RGBA/RGB canvas."""
    pad = 8
    width = int(_text_width(font, text)) + pad * 2
    try:
        ascent, descent = font.getmetrics()
        height = ascent + descent + pad * 2
    except AttributeError:
        height = int(getattr(font, "size", 12) * 1.6) + pad * 2
    tile = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    if halo:
        for ox in (-2, 0, 2):
            for oy in (-2, 0, 2):
                if ox or oy:
                    draw.text((pad + ox, pad + oy), text, font=font, fill=halo + (170,))
    draw.text((pad, pad), text, font=font, fill=fill + (255,))
    if abs(angle_rad) > 1e-3:
        tile = tile.rotate(-math.degrees(angle_rad), expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(
        tile,
        (int(center[0] - tile.width / 2), int(center[1] - tile.height / 2)),
    )
