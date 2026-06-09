"""Flat-key matting: turn sprites generated on a solid key color into RGBA.

All sprite/POI prompts request a flat magenta background (#FF00FF) so
matting is a deterministic chroma-distance operation rather than an ML
segmentation problem: alpha ramps with distance from the key color,
despill removes the magenta cast from edge pixels, and a slight feather
softens the cut.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

KEY_COLOR = (255, 0, 255)


def key_to_alpha(
    img: Image.Image,
    key: tuple[int, int, int] = KEY_COLOR,
    tolerance: float = 90.0,
    softness: float = 60.0,
) -> Image.Image:
    """Convert a flat-key image to RGBA.

    Pixels within `tolerance` of the key are fully transparent; alpha ramps
    to opaque over `softness` additional distance.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    dist = np.sqrt(((rgb - np.array(key, dtype=np.float32)) ** 2).sum(axis=-1))
    alpha = np.clip((dist - tolerance) / max(1.0, softness), 0.0, 1.0)

    # Despill: in semi-transparent edge pixels, pull the channels toward
    # their non-key balance by suppressing the key color's signature
    # (high R+B, low G for magenta).
    spill_zone = (alpha > 0.0) & (alpha < 1.0)
    if spill_zone.any():
        g = rgb[..., 1]
        spill = np.clip((rgb[..., 0] + rgb[..., 2]) / 2.0 - g, 0, 255)
        for ch in (0, 2):
            rgb[..., ch] = np.where(spill_zone, rgb[..., ch] - spill * 0.5, rgb[..., ch])
    rgb = np.clip(rgb, 0, 255)

    out = np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    result = Image.fromarray(out, "RGBA")

    # Feather the alpha edge slightly.
    a = result.getchannel("A").filter(ImageFilter.GaussianBlur(0.8))
    result.putalpha(a)
    return result


def trim_to_content(img: Image.Image, padding: int = 4) -> Image.Image:
    """Crop an RGBA sprite to its non-transparent bounding box."""
    alpha = np.asarray(img.getchannel("A"))
    rows = np.any(alpha > 8, axis=1)
    cols = np.any(alpha > 8, axis=0)
    if not rows.any() or not cols.any():
        return img
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width, x1 + padding)
    y1 = min(img.height, y1 + padding)
    return img.crop((x0, y0, x1, y1))
