"""Deterministic stub image generators for E2E tests.

Creates visually identifiable, labeled images that flow through the entire
pipeline (generation, assembly, DZI, postprocess) without any real API calls.
"""

import hashlib

from PIL import Image, ImageDraw, ImageFont


def _color_from_label(label: str) -> tuple[int, int, int]:
    """Deterministic background color derived from label hash."""
    h = hashlib.md5(label.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # Keep colors muted so white text is readable
    r = 40 + (r % 160)
    g = 40 + (g % 160)
    b = 40 + (b % 160)
    return (r, g, b)


def create_numbered_tile(
    width: int,
    height: int,
    label: str,
    bg_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Create a labeled tile image with grid lines.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        label: Text label drawn at center (e.g. "FLAT(2,1)", "OVERVIEW").
        bg_color: Optional explicit background color; derived from label hash if None.

    Returns:
        RGBA PIL Image with grid and centered label.
    """
    if bg_color is None:
        bg_color = _color_from_label(label)

    img = Image.new("RGBA", (width, height), (*bg_color, 255))
    draw = ImageDraw.Draw(img)

    # 4x4 grid lines
    grid_color = (255, 255, 255, 80)
    for i in range(1, 4):
        x = width * i // 4
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for i in range(1, 4):
        y = height * i // 4
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Centered label text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(16, min(width, height) // 6))
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = (height - th) // 2

    # Drop shadow for readability
    draw.text((tx + 2, ty + 2), label, fill=(0, 0, 0, 180), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    return img


def create_satellite_stub(width: int, height: int) -> Image.Image:
    """Create a greenish earth-toned image labeled 'SAT'."""
    return create_numbered_tile(width, height, "SAT", bg_color=(60, 100, 60))
