"""AI repaint of the composed poster base (below labels).

Default mode is the single-call texture pass (texture_pass.py): one
whole-base img2img generation + frequency-split blend -- no joints, no
cross-window drift, validated live. The tiled window-by-window engine
(grid/planner/engine, recipe from Andy Coenen's Isometric NYC project)
is kept for a future fine-tuned exact-infill painter: live spikes showed
zero-shot models mutate the "keep identical" context pixels, so tiled
butt-joints cannot line up without a model trained for exact continuation.
"""

from .engine import GeminiPainter, IdentityPainter, RepaintEngine, RepaintPainter
from .grid import QUAD, QuadrantGrid, QuadStatus, Selection, is_legal
from .planner import plan_order
from .store import RepaintStore
from .texture_pass import (
    GeminiTexturePass,
    IdentityTexturePass,
    StructureRejection,
    TexturePainter,
    texture_repaint,
)

__all__ = [
    "QUAD",
    "QuadrantGrid",
    "QuadStatus",
    "Selection",
    "is_legal",
    "plan_order",
    "RepaintStore",
    "RepaintEngine",
    "RepaintPainter",
    "IdentityPainter",
    "GeminiPainter",
    "TexturePainter",
    "IdentityTexturePass",
    "GeminiTexturePass",
    "StructureRejection",
    "texture_repaint",
]
