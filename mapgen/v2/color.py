"""Shared color tools: soft color replacement, palette conformance, and
reference-matched normalization.

These exist because AI-generated imagery drifts: per-asset palette outliers
(scored here, flagged in the asset studio) and per-window color drift during
tiled repaint (corrected by `normalize_to_reference`). `soft_color_replace`
pulls a drifted color back to a palette target through a per-pixel alpha
matte built from color distance, so anti-aliased edges keep their blend
instead of acquiring a hard posterized rim.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .types import hex_to_rgb


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB (0..255 float) -> CIE Lab (D65). Shape (..., 3)."""
    c = rgb / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ m.T
    # Normalize by D65 white point.
    xyz /= np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of `_srgb_to_lab`; returns float RGB clipped to 0..255."""
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.stack([fx, fy, fz], axis=-1)
    f3 = f**3
    xyz = np.where(f3 > eps, f3, (116 * f - 16) / kappa)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    linear = np.clip(xyz @ m_inv.T, 0.0, None)
    c = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return np.clip(c * 255.0, 0.0, 255.0)


def soft_color_replace(
    img: Image.Image,
    target_rgb: tuple[int, int, int],
    replacement_rgb: tuple[int, int, int],
    tolerance: float = 20.0,
    softness: float = 60.0,
    mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Replace `target_rgb` with `replacement_rgb` through a soft alpha matte.

    Pixels within `tolerance` of the target (RGB Euclidean distance) are
    fully replaced; replacement strength ramps to zero over `softness`
    additional distance, so anti-aliased edge pixels receive a partial blend
    that preserves their transition. `mask` (bool or 0..1 float, HxW)
    restricts the replacement region -- e.g. the deterministic water mask --
    and is itself blended so mask edges stay soft.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    dist = np.sqrt(((rgb - np.array(target_rgb, dtype=np.float32)) ** 2).sum(axis=-1))
    alpha = 1.0 - np.clip((dist - tolerance) / max(1.0, softness), 0.0, 1.0)
    if mask is not None:
        alpha = alpha * np.asarray(mask, dtype=np.float32)
    alpha = alpha[..., None]
    solid = np.full_like(rgb, np.array(replacement_rgb, dtype=np.float32))
    out = solid * alpha + rgb * (1.0 - alpha)
    result = Image.fromarray(out.astype(np.uint8), "RGB")
    if img.mode == "RGBA":
        result.putalpha(img.getchannel("A"))
    return result


def palette_distance(
    img: Image.Image,
    palette: dict[str, str],
    alpha_threshold: int = 32,
    worst_fraction: float = 0.1,
) -> float:
    """Mean Lab distance (ΔE76) from each pixel to its nearest palette color,
    averaged over the worst `worst_fraction` of pixels.

    The worst-decile average (rather than the overall mean) is what separates
    an asset with a small off-palette accent from one that drifted wholesale:
    in-palette assets score low even with shading variation, while a wrong-hue
    generation scores high. Transparent pixels (alpha < `alpha_threshold`)
    are ignored so matted sprites are judged on their content only.
    """
    # A summary score doesn't need full resolution, and the N x P distance
    # matrix below would be ~1 GB for a 2048px asset.
    if max(img.size) > 512:
        ratio = 512 / max(img.size)
        img = img.resize(
            (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
            Image.Resampling.BILINEAR,
        )
    rgba = np.asarray(img.convert("RGBA"), dtype=np.float32)
    visible = rgba[..., 3] >= alpha_threshold
    if not visible.any():
        return 0.0
    lab = _srgb_to_lab(rgba[..., :3][visible])
    palette_lab = _srgb_to_lab(
        np.array([hex_to_rgb(c) for c in palette.values()], dtype=np.float32)
    )
    # (N, P) distance matrix; N can be ~4M for a full asset, P ~20 -- fine.
    diff = lab[:, None, :] - palette_lab[None, :, :]
    nearest = np.sqrt((diff**2).sum(axis=-1)).min(axis=1)
    k = max(1, int(round(nearest.size * worst_fraction)))
    worst = np.partition(nearest, nearest.size - k)[-k:]
    return float(worst.mean())


def normalize_to_reference(
    img: Image.Image,
    reference: Image.Image,
    exclude_mask: Optional[np.ndarray] = None,
    strength: float = 1.0,
) -> Image.Image:
    """Match `img`'s per-channel Lab mean/std to `reference`'s.

    `exclude_mask` (bool, HxW, True = excluded) removes pixels from BOTH
    images' statistics -- e.g. water, whose flat color would otherwise skew
    the match -- but the correction still applies to every pixel of `img`.
    `strength` scales the correction (1.0 = full match).
    """
    if img.size != reference.size:
        raise ValueError(
            f"size mismatch: img {img.size} vs reference {reference.size}"
        )
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    src_lab = _srgb_to_lab(src)
    ref_lab = _srgb_to_lab(ref)

    if exclude_mask is not None:
        include = ~np.asarray(exclude_mask, dtype=bool)
        if include.sum() < 16:  # window is ~all excluded; nothing to match
            return img.copy()
        src_stats, ref_stats = src_lab[include], ref_lab[include]
    else:
        src_stats = src_lab.reshape(-1, 3)
        ref_stats = ref_lab.reshape(-1, 3)

    src_mean, src_std = src_stats.mean(axis=0), src_stats.std(axis=0)
    ref_mean, ref_std = ref_stats.mean(axis=0), ref_stats.std(axis=0)
    scale = ref_std / np.maximum(src_std, 1e-4)
    matched = (src_lab - src_mean) * scale + ref_mean
    corrected = src_lab + strength * (matched - src_lab)

    out = _lab_to_srgb(corrected).astype(np.uint8)
    result = Image.fromarray(out, "RGB")
    if img.mode == "RGBA":
        result.putalpha(img.getchannel("A"))
    return result
