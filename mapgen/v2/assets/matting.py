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

# Detected keys further than this (RGB distance) from the nominal magenta are
# palette-shifted generations -- mattable, but worth flagging for review.
KEY_SHIFT_THRESHOLD = 110.0


def key_shift(key: tuple[int, int, int], nominal: tuple[int, int, int] = KEY_COLOR) -> float:
    """RGB distance between the detected key and the requested key color."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(key, nominal))))


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hue/saturation (PIL HSV convention, 0..255) for HxWx3 float RGB."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    c = mx - mn
    sat = np.where(mx > 0, c / np.maximum(mx, 1e-6) * 255.0, 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(mx)
    safe_c = np.maximum(c, 1e-6)
    hue = np.where(mx == r, ((g - b) / safe_c) % 6.0, hue)
    hue = np.where(mx == g, (b - r) / safe_c + 2.0, hue)
    hue = np.where(mx == b, (r - g) / safe_c + 4.0, hue)
    hue = hue / 6.0 * 255.0
    return hue, sat


def detect_key_color(
    img: Image.Image,
    fallback: tuple[int, int, int] = KEY_COLOR,
    border_px: int = 6,
    max_std: float = 18.0,
) -> tuple[int, int, int]:
    """Measure the actual background key color from the image border.

    Image models reproduce the requested flat key only approximately -- a
    style reference can pull "#FF00FF" toward a muted rose. The background
    is whatever uniform color rings the border: if the border is uniform
    (per-channel std below `max_std`), return its median; otherwise (the
    subject touches the border) fall back to the nominal key.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    b = border_px
    ring = np.concatenate(
        [
            rgb[:b].reshape(-1, 3),
            rgb[-b:].reshape(-1, 3),
            rgb[:, :b].reshape(-1, 3),
            rgb[:, -b:].reshape(-1, 3),
        ]
    )
    if ring.std(axis=0).max() > max_std:
        return fallback
    return tuple(int(c) for c in np.median(ring, axis=0))


def key_to_alpha(
    img: Image.Image,
    key: tuple[int, int, int] | None = None,
    tolerance: float = 90.0,
    softness: float = 60.0,
    choke_px: int = 2,
    despill_reach_px: int = 3,
) -> Image.Image:
    """Convert a flat-key image to RGBA.

    `key` defaults to the border-detected background color (see
    `detect_key_color`), so palette-shifted keys still matte cleanly.
    Pixels within `tolerance` of the key are fully transparent; alpha ramps
    to opaque over `softness` additional distance. The matte is then choked
    (eroded) by `choke_px` so the key-contaminated rim falls outside it, and
    a full-strength despill is applied in a band `despill_reach_px` wide
    around the matte edge so no magenta cast survives on edge pixels.
    """
    if key is None:
        key = detect_key_color(img)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    dist = np.sqrt(((rgb - np.array(key, dtype=np.float32)) ** 2).sum(axis=-1))
    alpha = np.clip((dist - tolerance) / max(1.0, softness), 0.0, 1.0)

    # Key-family gate: only pixels that share the key's color family (similar
    # hue, meaningfully saturated) may ever be transparent. When the style
    # reference pulls the key from magenta toward e.g. dusty rose, the raw
    # chroma distance from that rose to white roofs or cream sandstone can sit
    # inside the tolerance band -- but those pixels are neutral or off-hue, so
    # the gate keeps them opaque. Skipped for near-neutral keys (hue would be
    # meaningless); those generations are flagged upstream instead.
    key_hue, key_sat = _hue_sat(np.array(key, dtype=np.float32).reshape(1, 1, 3))
    key_hue, key_sat = float(key_hue[0, 0]), float(key_sat[0, 0])
    in_family = None
    if key_sat >= 40.0:
        hue, sat = _hue_sat(rgb)
        hue_diff = np.abs(hue - key_hue)
        hue_diff = np.minimum(hue_diff, 255.0 - hue_diff)
        in_family = (hue_diff < 30.0) & (sat > max(30.0, 0.35 * key_sat))
        alpha = np.where(in_family, alpha, 1.0)

    # Connectivity guard: when the model shifts the key toward the subject's
    # own palette (e.g. muted rose vs. brick-red roofs), a pure distance matte
    # eats the subject's interior. Protect keyish components that are NOT
    # connected to the border -- but only those whose color measurably differs
    # from the key: enclosed pockets of the actual flat key paint (a frame's
    # empty center, gaps between sprites) have distance ~0 and must still be
    # keyed out.
    from scipy import ndimage

    keyish = dist < (tolerance + softness)
    labels, n_labels = ndimage.label(keyish)
    border_ids = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    interior_ids = np.setdiff1d(np.arange(1, n_labels + 1), border_ids)
    if interior_ids.size:
        # Color-distinct: median is robust to anti-aliased rims, so flat key
        # pockets stay near 0 even when their edges are blended.
        median_dist = np.asarray(
            ndimage.labeled_comprehension(dist, labels, interior_ids, np.median, float, 0.0)
        )
        # Thick: real subject regions are blobs; key showing through thin
        # gaps (between bridge cables, inside flourishes) erodes away.
        thick_ids = np.unique(labels[ndimage.binary_erosion(keyish, iterations=3)])
        protectable = (median_dist > 25.0) & np.isin(interior_ids, thick_ids)
        if in_family is not None:
            # Pockets that are mostly key-family colored are enclosed key
            # paint (a frame's empty center, the gap under an arch), not
            # subject -- never protect those.
            family_frac = np.asarray(
                ndimage.labeled_comprehension(
                    in_family.astype(np.float32), labels, interior_ids, np.mean, float, 0.0
                )
            )
            protectable &= family_frac < 0.5
        protect_ids = interior_ids[protectable]
        if protect_ids.size:
            alpha = np.where(np.isin(labels, protect_ids), 1.0, alpha)

    # Choke: erode the matte inward so the half-keyed rim is cut off rather
    # than composited semi-transparently (the source of visible fringes).
    if choke_px > 0:
        a_img = Image.fromarray((alpha * 255).astype(np.uint8), "L")
        a_img = a_img.filter(ImageFilter.MinFilter(2 * choke_px + 1))
        alpha = np.asarray(a_img, dtype=np.float32) / 255.0

    # Despill: remove the key color's signature (high R+B, low G for magenta)
    # at FULL strength in a band around the matte edge -- semi-transparent
    # pixels plus a few opaque pixels just inside, which also carry cast.
    edge = (alpha > 0.0) & (alpha < 1.0)
    if edge.any():
        band_img = Image.fromarray((edge * 255).astype(np.uint8), "L")
        band_img = band_img.filter(ImageFilter.MaxFilter(2 * despill_reach_px + 1))
        band = (np.asarray(band_img) > 0) & (alpha > 0.0)
        g = rgb[..., 1]
        spill = np.clip((rgb[..., 0] + rgb[..., 2]) / 2.0 - g, 0, 255)
        for ch in (0, 2):
            rgb[..., ch] = np.where(band, rgb[..., ch] - spill, rgb[..., ch])
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
