"""Color drift control for tiled repaint.

Isometric NYC's #1 scale problem was color drift between generations, fixed
by normalizing anchors -- always excluding water from the statistics. Our
guide is already in the target palette, so we can do better than their
global anchors: each painted selection is matched to ITS OWN guide region
(Lab mean/std), which also keeps the atmosphere gradient intact.

Water gets a dedicated post-stitch repair: image-edit models drift flat
water most (their hard-won lesson). Where they replaced the water color
outright (flat pixel-art water), our water carries brushwork texture, so we
SHIFT the water region by the median error instead -- same correction,
texture preserved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from ..color import normalize_to_reference


def normalize_selection(
    painted: Image.Image,
    guide_region: Image.Image,
    water_mask_region: Optional[Image.Image] = None,
    strength: float = 1.0,
) -> Image.Image:
    """Match a painted selection to its guide region, water excluded."""
    exclude = None
    if water_mask_region is not None:
        exclude = np.asarray(water_mask_region.convert("L")) > 127
    return normalize_to_reference(painted, guide_region, exclude_mask=exclude, strength=strength)


def match_low_frequency(
    img: Image.Image,
    guide: Image.Image,
    radius_px: float = 110.0,
    strength: float = 0.85,
    max_ratio: float = 0.30,
) -> Image.Image:
    """Pull the image's low-frequency tone field back to the guide's.

    Per-selection normalization matches window-level statistics, but on
    large flat regions (desert, open water, plains) the eye reads any
    remaining quadrant-scale tone differences as a checkerboard. Matching
    the blurred (below-quadrant-frequency) field to the guide removes that
    patchwork everywhere while leaving brushwork detail untouched. The
    gain is clamped so genuinely darker painted content (shadows, washes)
    is softened, not erased.
    """
    # The correction field is low-frequency by construction, so compute it
    # at 1/8 resolution and upsample -- full-resolution float32 copies of a
    # print-size poster run to several GB and can OOM a WSL VM.
    ds = 8
    small_size = (max(1, img.width // ds), max(1, img.height // ds))
    blur = ImageFilter.GaussianBlur(radius_px / ds)
    la = np.asarray(
        img.convert("RGB").resize(small_size, Image.Resampling.BILINEAR).filter(blur),
        dtype=np.float32,
    )
    lg = np.asarray(
        guide.convert("RGB").resize(small_size, Image.Resampling.BILINEAR).filter(blur),
        dtype=np.float32,
    )
    ratio = np.clip((lg + 1.0) / (la + 1.0), 1.0 - max_ratio, 1.0 + max_ratio)
    gain_small = 1.0 + strength * (ratio - 1.0)
    # Upsample the gain field as an image (uint16 fixed-point, x4096).
    gain_imgs = [
        Image.fromarray((gain_small[..., c] * 4096.0).astype(np.uint16), "I;16").resize(
            (img.width, img.height), Image.Resampling.BILINEAR
        )
        for c in range(3)
    ]

    out = np.asarray(img.convert("RGB")).copy()
    stripe = 1024
    for y in range(0, out.shape[0], stripe):
        sl = slice(y, min(y + stripe, out.shape[0]))
        gain = np.dstack(
            [
                np.asarray(g.crop((0, sl.start, img.width, sl.stop)), dtype=np.float32)
                for g in gain_imgs
            ]
        ) / 4096.0
        block = out[sl].astype(np.float32) * gain
        out[sl] = np.clip(block, 0, 255).astype(np.uint8)
    result = Image.fromarray(out, "RGB")
    if img.mode == "RGBA":
        result.putalpha(img.getchannel("A"))
    return result


def unify_water(
    img: Image.Image,
    water_mask: Image.Image,
    target_rgb: tuple[int, int, int],
    feather_px: float = 3.0,
    min_pixels: int = 200,
) -> Image.Image:
    """Pull the water region's median color back to the palette target.

    Applies the median error as a uniform shift inside a feathered water
    mask: drift is corrected while brushwork texture and anti-aliased
    shorelines survive (a flat replacement would erase both).
    """
    # Striped uint8 processing: full-size float32 copies of a print poster
    # are ~840 MB each and can OOM a memory-capped WSL VM.
    rgb = img.convert("RGB")
    mask_l = water_mask.convert("L")
    out = np.asarray(rgb).copy()
    soft = mask_l.filter(ImageFilter.GaussianBlur(feather_px))

    # Median water color from a bounded sample of hard-mask pixels.
    hard_total = 0
    samples: list[np.ndarray] = []
    stripe = 1024
    mask_arr_full = np.asarray(mask_l)
    for y in range(0, out.shape[0], stripe):
        sl = slice(y, min(y + stripe, out.shape[0]))
        hard = mask_arr_full[sl] > 127
        count = int(hard.sum())
        if count:
            hard_total += count
            samples.append(out[sl][hard][:: max(1, count // 5000)])
    if hard_total < min_pixels:
        return img
    observed = np.median(np.concatenate(samples).astype(np.float32), axis=0)
    delta = np.array(target_rgb, dtype=np.float32) - observed

    soft_arr = np.asarray(soft)
    for y in range(0, out.shape[0], stripe):
        sl = slice(y, min(y + stripe, out.shape[0]))
        weight = (soft_arr[sl].astype(np.float32) / 255.0)[..., None]
        block = out[sl].astype(np.float32) + delta * weight
        out[sl] = np.clip(block, 0, 255).astype(np.uint8)
    result = Image.fromarray(out, "RGB")
    if img.mode == "RGBA":
        result.putalpha(img.getchannel("A"))
    return result
