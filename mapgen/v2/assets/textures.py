"""Seamless-texture verification and repair for ground textures.

Generated textures are *requested* tileable, but the model is not
reliable about it, so tileability is verified programmatically (opposite
edge difference) and enforced when needed.

Repair uses Moisan's periodic+smooth decomposition (J. Math. Imaging
Vis., 2011): the image is split into a periodic component and a smooth
component that carries the cross-border discontinuity, and the smooth
part is discarded. Unlike cross-fade methods, this removes the seam
globally without leaving blurred bands along the tile borders.
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


def periodic_component(arr: np.ndarray) -> np.ndarray:
    """Periodic part of Moisan's periodic+smooth decomposition.

    The returned array has (near-)identical opposite edges while interior
    content is preserved; the discarded smooth component absorbs the
    cross-border jump.
    """
    h, w = arr.shape[:2]
    arr = arr.astype(np.float64)
    boundary = np.zeros_like(arr)
    row_jump = arr[-1, :] - arr[0, :]
    boundary[0, :] += row_jump
    boundary[-1, :] -= row_jump
    col_jump = arr[:, -1] - arr[:, 0]
    boundary[:, 0] += col_jump
    boundary[:, -1] -= col_jump

    fx = 2.0 * np.cos(2.0 * np.pi * np.arange(w) / w)
    fy = 2.0 * np.cos(2.0 * np.pi * np.arange(h) / h)
    denom = fx[None, :] + fy[:, None] - 4.0
    denom[0, 0] = 1.0  # avoid /0; the DC term is zeroed below

    smooth_hat = np.fft.fft2(boundary, axes=(0, 1)) / denom[:, :, None]
    smooth_hat[0, 0, :] = 0.0
    smooth = np.real(np.fft.ifft2(smooth_hat, axes=(0, 1)))
    return arr - smooth


def make_tileable(img: Image.Image) -> Image.Image:
    """Make opposite edges wrap seamlessly via periodic decomposition."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    periodic = periodic_component(arr)
    return Image.fromarray(np.clip(periodic, 0, 255).astype(np.uint8), "RGB")


def ensure_tileable(img: Image.Image, max_seam_error: float = 0.5) -> Image.Image:
    """Always repair unless the texture is already (near-)exactly periodic.

    The decomposition is an exact identity for periodic input and cheap to
    compute, so even faint seams (which read as a coherent grid once
    tiled) are worth removing.
    """
    if edge_seam_error(img) <= max_seam_error:
        return img.convert("RGB")
    return make_tileable(img)
