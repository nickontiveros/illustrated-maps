"""Deterministic generation order for tiled repaint.

Simpler than Isometric NYC's spiral (parallel batching is irrelevant at
~40-call scale): a 2-row band of 2x2 blocks joined by vertical bridges,
then row-by-row depth-1 strips (their strip-plan formula), then a 1x1
cleanup sweep for anything left over (resume holes, flagged redos).

Every emitted selection is legality-checked against the simulated painted
set as the plan unfolds, so the engine can trust the order blindly; cells
whose static-pattern selection turned out illegal (only happens on resume
with unusual pre-painted shapes) fall through to the cleanup sweep.
"""

from __future__ import annotations

from .grid import Cell, QuadrantGrid, Selection, is_legal


def plan_order(
    grid: QuadrantGrid,
    painted: set[Cell] | None = None,
    skipped: set[Cell] | None = None,
) -> list[Selection]:
    """Ordered selections that paint every non-skipped, non-painted cell."""
    painted = set(painted or ())
    skipped = set(skipped or ())
    plan: list[Selection] = []
    sim = set(painted)

    def emit(x: int, y: int, w: int, h: int, allow_boxed: bool = False) -> None:
        w, h = min(w, grid.cols - x), min(h, grid.rows - y)
        if w < 1 or h < 1:
            return
        sel = Selection(x, y, w, h)
        cells = sel.cells()
        if all(c in sim or c in skipped for c in cells):
            return  # nothing left to paint here
        if not is_legal(sel, sim, grid.cols, grid.rows, allow_boxed=allow_boxed):
            return  # cells fall through to the cleanup sweep
        plan.append(sel)
        sim.update(cells)

    # Pass 1: top band (rows 0-1) -- isolated 2x2 blocks every 3 columns...
    for x in range(0, grid.cols, 3):
        emit(x, 0, 2, 2)
    # ...then the 1-wide vertical bridges between them (painted both sides).
    for x in range(2, grid.cols, 3):
        emit(x, 0, 1, 2)

    # Pass 2: each following row as a depth-1 strip against the painted edge
    # above: 2x1 blocks with 1-column gaps, then the 1x1 gap fills.
    for y in range(2, grid.rows):
        for x in range(0, grid.cols, 3):
            emit(x, y, 2, 1)
        for x in range(2, grid.cols, 3):
            emit(x, y, 1, 1)

    # Pass 3: cleanup sweep. Remaining cells (resume holes, redo targets) as
    # 1x1 in raster order; boxed-in cells use the centered inner-square
    # window, which exists precisely for spot redos.
    for cell in grid.all_cells():
        if cell not in sim and cell not in skipped:
            emit(cell[0], cell[1], 1, 1, allow_boxed=True)

    return plan
