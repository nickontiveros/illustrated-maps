"""Structure guard: catch hallucinated repaints.

The B0 spike showed the failure mode plainly: given a window that is mostly
featureless (padding, blank plate), Gemini invents whole scenes -- an
island village where flat brown fill should be. Image models cannot QA
their own output (Isometric NYC's hard lesson), but THIS failure is
measurable without a model: a faithful repaint keeps the guide's structure,
so the gradient fields correlate; an invention does not. Measured on live
spike data: faithful windows score 0.5-0.98, the hallucinated one -0.02.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Below this gradient correlation the repaint did not follow the guide's
# content and is rejected (guide pixels kept, cells flagged for review).
STRUCTURE_CORR_THRESHOLD = 0.25


def _gradient_magnitude(img: Image.Image, size: int = 128) -> np.ndarray:
    gray = np.asarray(
        img.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
    )
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def structure_correlation(guide: Image.Image, painted: Image.Image) -> float:
    """Pearson correlation of downsampled gradient magnitudes (-1..1).

    Insensitive to palette/texture changes (the point of repainting) but
    collapses when shapes move or new content is invented.
    """
    a = _gradient_magnitude(guide).ravel()
    b = _gradient_magnitude(painted).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if denom < 1e-6:  # both featureless: nothing to contradict
        return 1.0
    return float((a * b).sum() / denom)
