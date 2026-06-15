"""Quadrant grid and seam-legality rules for tiled repaint.

The canvas is padded to a grid of 512px quadrants. Each generation call
repaints a rectangular selection of 1-2 x 1-2 quadrants inside a 1024px
window; all other window pixels come from already-painted neighbors (or
the guide where nothing is painted yet). A selection is *legal* when a
window placement exists that covers every painted neighbor it touches, so
each painted edge gets blending context and no seam can form.

The rules derive from the Isometric NYC generation rules (their task docs
012/015/019), restated geometrically because their write-up's prose
examples and rule summary disagree in one case (the "middle band"). A
1024px window leaves 512px of slack along an axis where the selection
spans one quadrant, and none where it spans two:

- Axis spanning 2 quadrants: NO painted neighbor may touch either side
  along that axis (no slack to include context). The 2x2-isolated rule
  falls out of applying this to both axes.
- Axis spanning 1 quadrant: painted neighbors on one side -> align the
  window to include that whole neighbor; on both sides -> center the
  window (a 256px band of context each side, their "middle band" case).
- 1x1 with all four sides painted is rejected (their "<= 3 generated
  neighbors" rule): centering both axes leaves only thin bands all around.
  Planners never need it -- spot-redo uses it with <= 3 painted sides.

This module reproduces every legal/illegal example in their docs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

QUAD = 512  # quadrant size in repaint-scale pixels
WINDOW = 2 * QUAD  # generation window is always 2x2 quadrants

Cell = tuple[int, int]


class QuadStatus(str, Enum):
    PENDING = "pending"
    GENERATED = "generated"
    SKIPPED = "skipped"  # blank paper/haze; never painted, never context
    FLAGGED = "flagged"  # generated but marked for redo


@dataclass(frozen=True)
class Selection:
    """A rectangular block of quadrants to paint in one call."""

    x: int  # leftmost quadrant column
    y: int  # topmost quadrant row
    w: int = 1  # 1 or 2 columns
    h: int = 1  # 1 or 2 rows

    def cells(self) -> list[Cell]:
        return [(self.x + dx, self.y + dy) for dy in range(self.h) for dx in range(self.w)]

    def __str__(self) -> str:  # used in store/journal records
        return f"({self.x},{self.y})x({self.w},{self.h})"


class QuadrantGrid:
    """Maps a canvas to its (padded) quadrant grid."""

    def __init__(self, width_px: int, height_px: int, quad: int = QUAD):
        if width_px < quad * 2 or height_px < quad * 2:
            raise ValueError(
                f"canvas {width_px}x{height_px} too small for {quad}px quadrant repaint "
                "(needs at least a 2x2 window)"
            )
        self.quad = quad
        self.width_px = width_px
        self.height_px = height_px
        self.cols = math.ceil(width_px / quad)
        self.rows = math.ceil(height_px / quad)

    @property
    def padded_size(self) -> tuple[int, int]:
        return (self.cols * self.quad, self.rows * self.quad)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def cell_box(self, cell: Cell) -> tuple[int, int, int, int]:
        """Pixel box (left, top, right, bottom) of a quadrant in the padded canvas."""
        x, y = cell
        return (x * self.quad, y * self.quad, (x + 1) * self.quad, (y + 1) * self.quad)

    def all_cells(self) -> list[Cell]:
        return [(x, y) for y in range(self.rows) for x in range(self.cols)]


def side_neighbors(sel: Selection) -> dict[str, list[Cell]]:
    """Side-adjacent cells of a selection (diagonals share no edge and are
    irrelevant to seams). May include out-of-grid cells; callers filter."""
    return {
        "left": [(sel.x - 1, sel.y + dy) for dy in range(sel.h)],
        "right": [(sel.x + sel.w, sel.y + dy) for dy in range(sel.h)],
        "top": [(sel.x + dx, sel.y - 1) for dx in range(sel.w)],
        "bottom": [(sel.x + dx, sel.y + sel.h) for dx in range(sel.w)],
    }


def is_legal(
    sel: Selection,
    painted: set[Cell],
    cols: int,
    rows: int,
    allow_boxed: bool = False,
) -> bool:
    """True if `sel` can be painted without forming a seam against any
    already-painted cell. Pure function; see module docstring for the rules.

    `allow_boxed` permits the 1x1-with-four-painted-sides case (centered
    inner-square window, 256px context bands all around) -- used for spot
    redos of interior quadrants, never during normal planning.
    """
    if sel.w not in (1, 2) or sel.h not in (1, 2):
        return False
    if sel.x < 0 or sel.y < 0 or sel.x + sel.w > cols or sel.y + sel.h > rows:
        return False
    if any(c in painted for c in sel.cells()):
        return False

    sides = side_neighbors(sel)
    touched = {
        side: any(n in painted for n in cells) for side, cells in sides.items()
    }

    # Axis spanning 2 quadrants: no slack, painted context impossible.
    if sel.w == 2 and (touched["left"] or touched["right"]):
        return False
    if sel.h == 2 and (touched["top"] or touched["bottom"]):
        return False
    # 1x1 boxed in on all four sides: context bands too thin everywhere.
    if sel.w == 1 and sel.h == 1 and all(touched.values()) and not allow_boxed:
        return False
    return True


def window_origin(sel: Selection, painted: set[Cell], grid: QuadrantGrid) -> tuple[int, int]:
    """Top-left pixel of the 1024px generation window for a legal selection.

    Per axis: span 2 -> window equals the selection extent; span 1 -> align
    toward a painted side (full-quadrant context), center when both sides
    are painted (256px bands), and otherwise pick the in-bounds placement.
    """
    pad_w, pad_h = grid.padded_size
    sides = side_neighbors(sel)
    touched = {side: any(n in painted for n in cells) for side, cells in sides.items()}

    def axis_origin(pos: int, span: int, lo_side: str, hi_side: str, limit_px: int) -> int:
        px = pos * grid.quad
        if span == 2:
            return px
        if touched[lo_side] and touched[hi_side]:
            return px - grid.quad // 2  # centered: half-quadrant context each side
        if touched[lo_side]:
            return px - grid.quad  # selection at the high side of the window
        if touched[hi_side]:
            return px  # selection at the low side
        # No painted context on this axis: any in-bounds placement works.
        return min(px, limit_px - WINDOW)

    ox = axis_origin(sel.x, sel.w, "left", "right", pad_w)
    oy = axis_origin(sel.y, sel.h, "top", "bottom", pad_h)
    # Legality guarantees alignment placements exist in-bounds (a painted
    # neighbor on a side implies the grid extends to that side).
    assert 0 <= ox <= pad_w - WINDOW and 0 <= oy <= pad_h - WINDOW, (sel, ox, oy)
    return ox, oy
