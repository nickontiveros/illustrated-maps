"""Tests for mapgen.services.color_consistency_service."""

import numpy as np
import pytest
from PIL import Image

from mapgen.services.color_consistency_service import ColorConsistencyService


@pytest.fixture
def service():
    return ColorConsistencyService(strength=0.5)


@pytest.fixture
def full_strength_service():
    return ColorConsistencyService(strength=1.0)


@pytest.fixture
def zero_strength_service():
    return ColorConsistencyService(strength=0.0)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_strength(self):
        svc = ColorConsistencyService()
        assert hasattr(svc, "strength")

    def test_custom_strength(self):
        svc = ColorConsistencyService(strength=0.8)
        assert svc.strength == 0.8


# ---------------------------------------------------------------------------
# histogram_match
# ---------------------------------------------------------------------------

class TestHistogramMatch:
    def test_returns_same_size_image(self, service, solid_red_image, solid_blue_image):
        result = service.histogram_match(solid_red_image, solid_blue_image)
        assert result.size == solid_red_image.size

    def test_returns_rgba_image(self, service, solid_red_image, solid_blue_image):
        result = service.histogram_match(solid_red_image, solid_blue_image)
        assert result.mode == "RGBA"

    def test_preserves_alpha(self, service, transparent_image, solid_red_image):
        result = service.histogram_match(transparent_image, solid_red_image)
        arr = np.array(result)
        np.testing.assert_array_equal(arr[:, :, 3], np.array(transparent_image)[:, :, 3])

    def test_zero_strength_preserves_source(self, zero_strength_service, gradient_image, solid_red_image):
        result = zero_strength_service.histogram_match(gradient_image, solid_red_image)
        src_arr = np.array(gradient_image)[:, :, :3]
        res_arr = np.array(result)[:, :, :3]
        np.testing.assert_array_almost_equal(src_arr, res_arr, decimal=0)

    def test_full_strength_shifts_toward_reference(self, full_strength_service, gradient_image):
        """Histogram matching a gradient to a blue-tinted reference should shift colors."""
        blue_ref = Image.new("RGBA", (64, 64), (30, 60, 220, 255))
        result = full_strength_service.histogram_match(gradient_image, blue_ref)
        arr = np.array(result)
        # Blue channel should have higher mean than red since reference is blue
        assert arr[:, :, 2].mean() > arr[:, :, 0].mean()


# ---------------------------------------------------------------------------
# build_color_lut
# ---------------------------------------------------------------------------

class TestBuildColorLUT:
    def test_returns_correct_shape(self, service, solid_red_image):
        lut = service.build_color_lut(solid_red_image, lut_size=32)
        assert isinstance(lut, np.ndarray)
        assert lut.shape == (32, 32, 32, 3)

    def test_lut_values_in_valid_range(self, service, gradient_image):
        lut = service.build_color_lut(gradient_image, lut_size=16)
        assert lut.min() >= 0
        assert lut.max() <= 255


# ---------------------------------------------------------------------------
# rgb_to_lab / lab_to_rgb
# ---------------------------------------------------------------------------

class TestColorSpaceConversion:
    def test_rgb_to_lab_returns_array(self):
        rgb = np.array([[[128, 64, 200]]], dtype=np.uint8)
        lab = ColorConsistencyService.rgb_to_lab(rgb)
        assert lab.shape == rgb.shape

    def test_lab_to_rgb_returns_array(self):
        rgb_in = np.array([[[100, 150, 50]]], dtype=np.uint8)
        lab = ColorConsistencyService.rgb_to_lab(rgb_in)
        rgb_out = ColorConsistencyService.lab_to_rgb(lab)
        assert rgb_out.shape == rgb_in.shape

    def test_roundtrip_close(self):
        rgb = np.array([[[100, 150, 50]]], dtype=np.uint8)
        lab = ColorConsistencyService.rgb_to_lab(rgb)
        rgb2 = ColorConsistencyService.lab_to_rgb(lab)
        np.testing.assert_array_almost_equal(rgb, rgb2, decimal=0)

    def test_black_roundtrip(self):
        rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)
        lab = ColorConsistencyService.rgb_to_lab(rgb)
        rgb2 = ColorConsistencyService.lab_to_rgb(lab)
        np.testing.assert_array_almost_equal(rgb, rgb2, decimal=0)

    def test_white_roundtrip(self):
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lab = ColorConsistencyService.rgb_to_lab(rgb)
        rgb2 = ColorConsistencyService.lab_to_rgb(lab)
        np.testing.assert_array_almost_equal(rgb, rgb2, decimal=0)


# ---------------------------------------------------------------------------
# extract_palette
# ---------------------------------------------------------------------------

class TestExtractPalette:
    def test_returns_expected_count(self, service, gradient_image):
        palette = service.extract_palette(gradient_image, n_colors=4)
        assert len(palette) == 4

    def test_returns_rgb_tuples(self, service, solid_red_image):
        palette = service.extract_palette(solid_red_image, n_colors=2)
        for color in palette:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_solid_image_palette_is_dominant_color(self, service, solid_red_image):
        palette = service.extract_palette(solid_red_image, n_colors=1)
        r, g, b = palette[0]
        assert r > 200  # mostly red
        assert g < 50
        assert b < 50
