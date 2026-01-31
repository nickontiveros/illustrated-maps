"""Image processing utilities."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from file."""
    return Image.open(path).convert("RGBA")


def save_image(image: Image.Image, path: Union[str, Path], quality: int = 95) -> None:
    """Save an image to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        # Convert to RGB for JPEG
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        image.save(path, quality=quality)
    else:
        image.save(path)


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize image to specified size."""
    return image.resize(size, resample=resample)


def create_alpha_mask(
    image: Image.Image,
    threshold: int = 10,
    feather: int = 2,
) -> Image.Image:
    """Create alpha mask from image, detecting transparent/near-white backgrounds."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get alpha channel
    r, g, b, a = image.split()

    # Create mask from existing alpha
    mask = np.array(a)

    # Also mask near-white pixels (common background)
    rgb = np.array(image.convert("RGB"))
    brightness = rgb.mean(axis=2)
    white_mask = brightness > (255 - threshold)
    mask[white_mask] = 0

    # Apply feathering
    mask_img = Image.fromarray(mask)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

    return mask_img


def apply_drop_shadow(
    image: Image.Image,
    offset: tuple[int, int] = (5, 5),
    blur_radius: int = 10,
    shadow_color: tuple[int, int, int, int] = (0, 0, 0, 128),
) -> Image.Image:
    """Apply drop shadow effect to image with transparency."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create shadow layer
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Use alpha channel to create shadow shape
    alpha = image.split()[3]
    shadow_shape = Image.new("RGBA", image.size, shadow_color)
    shadow_shape.putalpha(alpha)

    # Create larger canvas to accommodate shadow offset
    new_size = (
        image.size[0] + abs(offset[0]) + blur_radius * 2,
        image.size[1] + abs(offset[1]) + blur_radius * 2,
    )
    result = Image.new("RGBA", new_size, (0, 0, 0, 0))

    # Position shadow
    shadow_pos = (
        blur_radius + max(0, offset[0]),
        blur_radius + max(0, offset[1]),
    )
    result.paste(shadow_shape, shadow_pos)

    # Blur shadow
    result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Paste original image on top
    image_pos = (
        blur_radius + max(0, -offset[0]),
        blur_radius + max(0, -offset[1]),
    )
    result.paste(image, image_pos, image)

    return result


def blend_images(
    base: Image.Image,
    overlay: Image.Image,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
) -> Image.Image:
    """Blend overlay image onto base at specified position."""
    if base.mode != "RGBA":
        base = base.convert("RGBA")
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    # Apply opacity
    if opacity < 1.0:
        r, g, b, a = overlay.split()
        a = a.point(lambda x: int(x * opacity))
        overlay = Image.merge("RGBA", (r, g, b, a))

    # Create a copy of base to modify
    result = base.copy()

    # Paste overlay at position
    result.paste(overlay, position, overlay)

    return result


def crop_to_content(image: Image.Image, padding: int = 0) -> Image.Image:
    """Crop image to non-transparent content."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get bounding box of non-transparent pixels
    bbox = image.getbbox()
    if bbox is None:
        return image

    # Add padding
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)

    return image.crop((left, top, right, bottom))
