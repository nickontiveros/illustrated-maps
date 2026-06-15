"""SQLite bookkeeping for the repaint stage.

The DB tracks quadrant status and the call journal (resume = re-plan over
whatever isn't generated); the painted pixels themselves live as PNGs in
`quads/` next to the DB, where they can be inspected and where ~300 small
files beat blobs for debuggability. The store is also how a crashed or
cost-capped run picks up where it left off.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .grid import Cell, QuadStatus, Selection

SCHEMA = """
CREATE TABLE IF NOT EXISTS quadrants (
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (x, y)
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    selection TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepaintStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.quads_dir = self.directory / "quads"
        self.quads_dir.mkdir(exist_ok=True)
        self.db_path = self.directory / "repaint.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- quadrant status -----------------------------------------------------

    def set_status(self, cell: Cell, status: QuadStatus, notes: str = "") -> None:
        self._conn.execute(
            "INSERT INTO quadrants (x, y, status, notes, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(x, y) DO UPDATE SET status=excluded.status, "
            "notes=excluded.notes, updated_at=excluded.updated_at",
            (cell[0], cell[1], status.value, notes, _now()),
        )
        self._conn.commit()

    def status_map(self) -> dict[Cell, QuadStatus]:
        rows = self._conn.execute("SELECT x, y, status FROM quadrants").fetchall()
        return {(x, y): QuadStatus(status) for x, y, status in rows}

    def cells_with_status(self, *statuses: QuadStatus) -> set[Cell]:
        wanted = {s.value for s in statuses}
        return {
            cell for cell, status in self.status_map().items() if status.value in wanted
        }

    # -- painted pixels --------------------------------------------------------

    def quad_path(self, cell: Cell) -> Path:
        return self.quads_dir / f"{cell[0]:03d}_{cell[1]:03d}.png"

    def save_quadrant(self, cell: Cell, img: Image.Image) -> None:
        img.save(self.quad_path(cell))
        self.set_status(cell, QuadStatus.GENERATED)

    def load_quadrant(self, cell: Cell) -> Image.Image | None:
        path = self.quad_path(cell)
        return Image.open(path).convert("RGB") if path.exists() else None

    # -- call journal ----------------------------------------------------------

    def record_call(self, selection: Selection) -> None:
        self._conn.execute(
            "INSERT INTO calls (selection, created_at) VALUES (?, ?)",
            (str(selection), _now()),
        )
        self._conn.commit()

    def call_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
