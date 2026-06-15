"""Build the 1024px generation window for one repaint call.

The window is cropped from the working canvas (painted quadrants where
they exist, guide pixels elsewhere), so neighbor context comes in
automatically. The selection region -- the only pixels the model's output
is kept for -- is marked with a 2px red boundary drawn just OUTSIDE the
selection (Isometric NYC's convention, moved outside so the extracted
pixels never contain boundary ink and an identity painter round-trips
byte-exact). Sides where the selection touches the window edge carry no
line; the prompt covers that case.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from .grid import WINDOW, Cell, QuadrantGrid, Selection, window_origin

BOUNDARY_RGB = (255, 32, 32)
BOUNDARY_PX = 2

Box = tuple[int, int, int, int]


def selection_px_box(sel: Selection, grid: QuadrantGrid) -> Box:
    """Selection bounds in padded-canvas pixels (left, top, right, bottom)."""
    return (
        sel.x * grid.quad,
        sel.y * grid.quad,
        (sel.x + sel.w) * grid.quad,
        (sel.y + sel.h) * grid.quad,
    )


def build_template(
    working: Image.Image,
    sel: Selection,
    painted: set[Cell],
    grid: QuadrantGrid,
) -> tuple[Image.Image, Box, Box]:
    """Returns (template image, window box in canvas px, selection box in
    window px). `working` must be the padded canvas."""
    ox, oy = window_origin(sel, painted, grid)
    window_box = (ox, oy, ox + WINDOW, oy + WINDOW)
    template = working.crop(window_box).convert("RGB")

    sl, st, sr, sb = selection_px_box(sel, grid)
    sel_box = (sl - ox, st - oy, sr - ox, sb - oy)

    # Stroke just outside the selection; PIL clips parts beyond the window.
    draw = ImageDraw.Draw(template)
    draw.rectangle(
        [
            sel_box[0] - BOUNDARY_PX,
            sel_box[1] - BOUNDARY_PX,
            sel_box[2] + BOUNDARY_PX - 1,
            sel_box[3] + BOUNDARY_PX - 1,
        ],
        outline=BOUNDARY_RGB,
        width=BOUNDARY_PX,
    )
    return template, window_box, sel_box


def extract_selection(result: Image.Image, sel_box: Box) -> Image.Image:
    """Crop the painted selection out of the model's window-sized output."""
    if result.size != (WINDOW, WINDOW):
        result = result.resize((WINDOW, WINDOW), Image.Resampling.LANCZOS)
    return result.convert("RGB").crop(sel_box)
