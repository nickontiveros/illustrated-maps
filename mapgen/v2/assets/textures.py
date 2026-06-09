"""Seamless-texture verification and repair for ground textures.

Generated textures are *requested* tileable, but the model is not
reliable about it, so tileability is verified programmatically (opposite
edge difference) and enforced with a cross-fade wrap when needed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def edge_seam_error(img: Image.Image) -> float:
    """Mean absolute difference between opposite edges (0 = perfectly tileable)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    horizontal = np.abs(arr[:, 0, :] - arr[:, -1, :]).mean()
    vertical = np.abs(arr[0, :, :] - arr[-1, :, :]).mean()
    return float((horizontal + vertical) / 2.0)


def make_tileable(img: Image.Image, blend_fraction: float = 0.18) -> Image.Image:
    """Cross-fade wrap: blend the image with a half-period-rolled copy near
    the edges so opposite edges become identical by construction."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    rolled = np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1)

    bw = max(2, int(w * blend_fraction))
    bh = max(2, int(h * blend_fraction))

    # Weight ramps: 1 at the edges (use rolled copy, whose seam is at the
    # center), 0 in the middle (use original).
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    wx = np.clip(1.0 - np.minimum(x, w - 1 - x) / bw, 0.0, 1.0)
    wy = np.clip(1.0 - np.minimum(y, h - 1 - y) / bh, 0.0, 1.0)
    weight = np.maximum(wx[None, :], wy[:, None])[..., None]

    out = arr * (1.0 - weight) + rolled * weight
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def ensure_tileable(img: Image.Image, max_seam_error: float = 8.0) -> Image.Image:
    if edge_seam_error(img) <= max_seam_error:
        return img.convert("RGB")
    return make_tileable(img)
