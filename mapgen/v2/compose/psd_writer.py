"""A small, dependency-free Photoshop (.psd) writer.

The repo's V1 path used ``pytoshop`` (see ``mapgen/services/psd_service.py``),
but its legacy ``setup.py`` no longer builds against modern setuptools, so the
V2 layered export ships its own writer instead. It covers exactly what the
layered poster needs: an 8-bit RGB document with any number of named, alpha-
masked layers placed at arbitrary positions, plus a flattened composite for
the merged preview that non-layer-aware readers fall back to.

The output is a standard PSD (version 1, ``8BPS``); it round-trips through
``psd-tools`` and opens in Photoshop / GIMP / Krita / Affinity. Layers carry a
PackBits-RLE (Photoshop's default) or raw payload, an ASCII Pascal name and a
Unicode (``luni``) name so non-Latin labels survive.

Reference: Adobe Photoshop File Format Specification (layer & mask section,
image data section).
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

# (channel id, PIL band index) in the order PSD expects them written.
_RGBA_CHANNELS = ((0, 0), (1, 1), (2, 2), (-1, 3))  # R, G, B, alpha


def _u16(v: int) -> bytes:
    return struct.pack(">H", v)


def _i16(v: int) -> bytes:
    return struct.pack(">h", v)


def _u32(v: int) -> bytes:
    return struct.pack(">I", v)


def _i32(v: int) -> bytes:
    return struct.pack(">i", v)


def _pad2(data: bytes) -> bytes:
    return data + b"\x00" if len(data) % 2 else data


def _pack_bits(data: bytes) -> bytes:
    """Standard PackBits run-length encoding (as used by TIFF and PSD)."""
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        if i + 1 < n and data[i] == data[i + 1]:
            run = 2
            while i + run < n and data[i + run] == data[i] and run < 128:
                run += 1
            out.append(257 - run)
            out.append(data[i])
            i += run
        else:
            start = i
            i += 1
            while i < n and not (i + 1 < n and data[i] == data[i + 1]) and (i - start) < 128:
                i += 1
            literal = data[start:i]
            out.append(len(literal) - 1)
            out.extend(literal)
    return bytes(out)


def _encode_plane_rle(plane: np.ndarray) -> tuple[list[int], bytes]:
    """RLE-encode one 8-bit channel row by row; return (per-row byte counts,
    concatenated packed bytes). An empty plane yields no rows."""
    counts: list[int] = []
    chunks: list[bytes] = []
    for row in plane:
        packed = _pack_bits(row.tobytes())
        counts.append(len(packed))
        chunks.append(packed)
    return counts, b"".join(chunks)


class _LayerData:
    """Pre-encoded channels for one layer, plus its rectangle and names."""

    def __init__(self, name: str, image: Image.Image, offset: tuple[int, int], rle: bool):
        self.name = name
        left, top = offset
        img = image.convert("RGBA")
        # A layer must enclose at least one pixel; Photoshop tolerates 1x1.
        w = max(1, img.width)
        h = max(1, img.height)
        if (w, h) != img.size:
            img = img.crop((0, 0, w, h))
        self.left, self.top = int(left), int(top)
        self.right, self.bottom = self.left + w, self.top + h
        self.width, self.height = w, h
        self.rle = rle
        arr = np.asarray(img)  # (h, w, 4)
        # Per channel: (channel_id, encoded-block-bytes).
        self.channels: list[tuple[int, bytes]] = []
        for channel_id, band in _RGBA_CHANNELS:
            plane = np.ascontiguousarray(arr[:, :, band])
            if rle:
                counts, packed = _encode_plane_rle(plane)
                block = _u16(1) + b"".join(_u16(c) for c in counts) + packed
            else:
                block = _u16(0) + plane.tobytes()
            self.channels.append((channel_id, block))

    def record(self) -> bytes:
        """The layer *record* (geometry, channel index, blend info, name)."""
        out = bytearray()
        out += _i32(self.top) + _i32(self.left) + _i32(self.bottom) + _i32(self.right)
        out += _u16(len(self.channels))
        for channel_id, block in self.channels:
            out += _i16(channel_id) + _u32(len(block))
        out += b"8BIM" + b"norm"  # blend mode signature + "normal"
        out += bytes([255, 0, 0, 0])  # opacity, clipping, flags (visible), filler

        # Extra data: mask (empty), blending ranges (empty), name, then luni.
        extra = bytearray()
        extra += _u32(0)  # layer mask data
        extra += _u32(0)  # layer blending ranges
        extra += _pascal_name(self.name)
        extra += _luni_block(self.name)
        out += _u32(len(extra)) + bytes(extra)
        return bytes(out)

    def channel_data(self) -> bytes:
        return b"".join(block for _, block in self.channels)


def _pascal_name(name: str) -> bytes:
    """Pascal string (1 length byte + bytes), padded so the whole field is a
    multiple of 4 bytes (per the layer-record spec)."""
    raw = name.encode("ascii", "replace")[:255]
    field = bytes([len(raw)]) + raw
    pad = (-len(field)) % 4
    return field + b"\x00" * pad


def _luni_block(name: str) -> bytes:
    """Unicode layer name additional-info block (``8BIM`` / ``luni``) so names
    with non-ASCII characters render correctly."""
    chars = name.encode("utf-16-be")
    data = _u32(len(name)) + chars
    data = _pad2(data)
    return b"8BIM" + b"luni" + _u32(len(data)) + data


def write_psd(
    path: Union[str, Path],
    size: tuple[int, int],
    layers: Sequence,
    composite: Optional[Image.Image] = None,
    compression: str = "rle",
) -> Path:
    """Write a layered PSD.

    Args:
        path: output ``.psd`` path.
        size: (width, height) of the document canvas in pixels.
        layers: bottom-to-top sequence of objects exposing ``.name``,
            ``.image`` (a PIL image) and ``.offset`` ((left, top)) -- e.g.
            ``compose.Layer``.
        composite: the flattened RGB(A) preview; if omitted the layers are
            composited onto the canvas to produce one.
        compression: ``"rle"`` (PackBits, compact) or ``"raw"`` (uncompressed).
    """
    width, height = size
    rle = compression == "rle"

    if composite is None:
        merged = Image.new("RGBA", size, (0, 0, 0, 0))
        for layer in layers:
            merged.alpha_composite(layer.image.convert("RGBA"), tuple(layer.offset))
        composite = merged
    composite = composite.convert("RGB")
    if composite.size != size:
        composite = composite.resize(size, Image.Resampling.LANCZOS)

    encoded = [_LayerData(lyr.name, lyr.image, tuple(lyr.offset), rle) for lyr in layers]

    # --- Layer info: count, records, then channel data in the same order. ---
    layer_info = bytearray()
    layer_info += _i16(len(encoded))
    for layer in encoded:
        layer_info += layer.record()
    for layer in encoded:
        layer_info += layer.channel_data()
    layer_info = _pad2(bytes(layer_info))

    layer_and_mask = bytearray()
    layer_and_mask += _u32(len(layer_info)) + layer_info  # layer info section
    layer_and_mask += _u32(0)  # global layer mask info (none)
    layer_and_mask = bytes(layer_and_mask)

    buf = BytesIO()
    # File header: signature, version 1, reserved, channels, dims, depth, mode.
    buf.write(b"8BPS")
    buf.write(_u16(1))
    buf.write(b"\x00" * 6)
    buf.write(_u16(3))  # composite channel count (RGB)
    buf.write(_u32(height))
    buf.write(_u32(width))
    buf.write(_u16(8))  # bit depth
    buf.write(_u16(3))  # color mode: RGB
    buf.write(_u32(0))  # color mode data
    buf.write(_u32(0))  # image resources
    buf.write(_u32(len(layer_and_mask)))
    buf.write(layer_and_mask)
    buf.write(_composite_image_data(composite, rle))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


def _composite_image_data(composite: Image.Image, rle: bool) -> bytes:
    """The trailing merged-image section: one compression flag, then (for RLE)
    every channel's row-count table followed by every channel's packed rows."""
    arr = np.asarray(composite)  # (h, w, 3)
    planes = [np.ascontiguousarray(arr[:, :, c]) for c in range(3)]
    if not rle:
        return _u16(0) + b"".join(p.tobytes() for p in planes)
    tables = bytearray()
    payload = bytearray()
    for plane in planes:
        counts, packed = _encode_plane_rle(plane)
        tables += b"".join(_u16(c) for c in counts)
        payload += packed
    return _u16(1) + bytes(tables) + bytes(payload)
