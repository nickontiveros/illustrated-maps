"""Tests for image processing utilities."""

import numpy as np
import pytest
from PIL import Image

from mapgen.utils.image_utils import (
    apply_drop_shadow,
    blend_images,
    create_alpha_mask,
    crop_to_content,
    resize_image,
)


class TestResizeImage:
    """Test image resizing."""

    def test_resize_dimensions(self, solid_red_image):
        result = resize_image(solid_red_image, (128, 128))
        assert result.size == (128, 128)

    def test_resize_smaller(self, solid_red_image):
        result = resize_image(solid_red_image, (32, 32))
        assert result.size == (32, 32)

    def test_resize_non_square(self, solid_red_image):
        result = resize_image(solid_red_image, (100, 50))
        assert result.size == (100, 50)


class TestCreateAlphaMask:
    """Test alpha mask creation."""

    def test_opaque_image(self, solid_red_image):
        mask = create_alpha_mask(solid_red_image, threshold=10)
        arr = np.array(mask)
        # Red image is not near-white, so most of mask should be non-zero
        assert arr.mean() > 100

    def test_transparent_image(self, transparent_image):
        mask = create_alpha_mask(transparent_image)
        arr = np.array(mask)
        # Fully transparent -> mask should be all zeros (after feathering may be near-zero)
        assert arr.mean() < 5

    def test_white_image_masked(self):
        white = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        mask = create_alpha_mask(white, threshold=10, feather=0)
        arr = np.array(mask)
        # Pure white should be masked out
        assert arr.mean() == 0

    def test_rgb_input_converted(self):
        rgb = Image.new("RGB", (64, 64), (128, 0, 0))
        mask = create_alpha_mask(rgb)
        assert mask.mode == "L"


class TestApplyDropShadow:
    """Test drop shadow effect."""

    def test_output_larger_than_input(self, solid_red_image):
        result = apply_drop_shadow(solid_red_image, offset=(5, 5), blur_radius=10)
        assert result.width > solid_red_image.width
        assert result.height > solid_red_image.height

    def test_shadow_offset_respected(self, solid_red_image):
        result = apply_drop_shadow(solid_red_image, offset=(10, 10), blur_radius=5)
        expected_width = solid_red_image.width + 10 + 5 * 2
        expected_height = solid_red_image.height + 10 + 5 * 2
        assert result.size == (expected_width, expected_height)

    def test_rgba_output(self, solid_red_image):
        result = apply_drop_shadow(solid_red_image)
        assert result.mode == "RGBA"


class TestBlendImages:
    """Test image blending."""

    def test_full_opacity_overlay(self, solid_red_image, solid_blue_image):
        result = blend_images(solid_red_image, solid_blue_image, opacity=1.0)
        # Blue overlay on red -> all blue
        pixel = result.getpixel((32, 32))
        assert pixel == (0, 0, 255, 255)

    def test_zero_opacity_preserves_base(self, solid_red_image, solid_blue_image):
        result = blend_images(solid_red_image, solid_blue_image, opacity=0.0)
        # Zero opacity overlay is fully transparent -> base shows through
        pixel = result.getpixel((32, 32))
        assert pixel[0] == 255  # Red channel from base

    def test_half_opacity_blends(self, solid_red_image, solid_blue_image):
        result = blend_images(solid_red_image, solid_blue_image, opacity=0.5)
        pixel = result.getpixel((32, 32))
        # Should be a mix of red and blue
        assert pixel[0] > 50  # Some red
        assert pixel[2] > 50  # Some blue

    def test_position_offset(self, solid_red_image, solid_blue_image):
        small_blue = solid_blue_image.resize((32, 32))
        result = blend_images(solid_red_image, small_blue, position=(32, 32))
        # Top-left should still be red
        assert result.getpixel((0, 0))[0] == 255
        # Bottom-right should be blue
        assert result.getpixel((48, 48))[2] == 255

    def test_rgba_output(self, solid_red_image, solid_blue_image):
        result = blend_images(solid_red_image, solid_blue_image)
        assert result.mode == "RGBA"


class TestCropToContent:
    """Test cropping to non-transparent content."""

    def test_crops_transparent_border(self):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        # Draw a 20x20 red square in center
        for x in range(40, 60):
            for y in range(40, 60):
                img.putpixel((x, y), (255, 0, 0, 255))
        cropped = crop_to_content(img)
        assert cropped.width == 20
        assert cropped.height == 20

    def test_padding_respected(self):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        for x in range(40, 60):
            for y in range(40, 60):
                img.putpixel((x, y), (255, 0, 0, 255))
        cropped = crop_to_content(img, padding=5)
        assert cropped.width == 30
        assert cropped.height == 30

    def test_fully_transparent_returns_original(self, transparent_image):
        cropped = crop_to_content(transparent_image)
        assert cropped.size == transparent_image.size

    def test_fully_opaque_unchanged(self, solid_red_image):
        cropped = crop_to_content(solid_red_image)
        assert cropped.size == solid_red_image.size
