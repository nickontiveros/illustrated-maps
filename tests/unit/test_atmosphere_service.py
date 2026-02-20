"""Tests for mapgen.services.atmosphere_service."""

import numpy as np
import pytest
from PIL import Image

from mapgen.models.atmosphere import AtmosphereSettings
from mapgen.services.atmosphere_service import AtmosphereService


@pytest.fixture
def enabled_service():
    settings = AtmosphereSettings(enabled=True, haze_strength=0.5)
    return AtmosphereService(settings=settings)


@pytest.fixture
def disabled_service():
    settings = AtmosphereSettings(enabled=False)
    return AtmosphereService(settings=settings)


@pytest.fixture
def default_service():
    return AtmosphereService()


@pytest.fixture
def perspective_params():
    """Minimal perspective params for testing."""
    return {"horizon_y": 0.3, "vanishing_y": 0.0}


# ---------------------------------------------------------------------------
# apply_atmosphere
# ---------------------------------------------------------------------------

class TestApplyAtmosphere:
    def test_disabled_returns_same_image(self, disabled_service, large_test_image, perspective_params):
        result = disabled_service.apply_atmosphere(large_test_image, perspective_params)
        src = np.array(large_test_image)
        res = np.array(result)
        np.testing.assert_array_equal(src, res)

    def test_enabled_returns_same_size(self, enabled_service, large_test_image, perspective_params):
        result = enabled_service.apply_atmosphere(large_test_image, perspective_params)
        assert result.size == large_test_image.size

    def test_enabled_returns_rgba(self, enabled_service, large_test_image, perspective_params):
        result = enabled_service.apply_atmosphere(large_test_image, perspective_params)
        assert result.mode == "RGBA"

    def test_top_rows_hazier_than_bottom(self, enabled_service, large_test_image, perspective_params):
        """Top of image (far away) should have more haze effect than bottom (close)."""
        result = enabled_service.apply_atmosphere(large_test_image, perspective_params)
        arr_orig = np.array(large_test_image).astype(float)
        arr_result = np.array(result).astype(float)
        # Compute per-row mean absolute difference from original
        diff = np.abs(arr_result[:, :, :3] - arr_orig[:, :, :3])
        top_diff = diff[:64].mean()
        bottom_diff = diff[-64:].mean()
        # Top should be more different from original (more haze applied)
        assert top_diff >= bottom_diff


# ---------------------------------------------------------------------------
# _build_gradient
# ---------------------------------------------------------------------------

class TestBuildGradient:
    def test_returns_correct_shape(self, enabled_service, perspective_params):
        gradient = enabled_service._build_gradient(256, perspective_params)
        assert gradient.shape == (256,)

    def test_values_in_range(self, enabled_service, perspective_params):
        gradient = enabled_service._build_gradient(256, perspective_params)
        assert gradient.min() >= 0.0
        assert gradient.max() <= 1.0

    def test_gradient_decreasing_top_to_bottom(self, enabled_service, perspective_params):
        """Gradient should generally decrease from top to bottom (more haze at top)."""
        gradient = enabled_service._build_gradient(256, perspective_params)
        # Top value should be >= bottom value
        assert gradient[0] >= gradient[-1]


# ---------------------------------------------------------------------------
# generate_fog_layer
# ---------------------------------------------------------------------------

class TestGenerateFogLayer:
    def test_returns_correct_size(self, enabled_service, perspective_params):
        result = enabled_service.generate_fog_layer((256, 256), perspective_params)
        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)

    def test_returns_rgba(self, enabled_service, perspective_params):
        result = enabled_service.generate_fog_layer((128, 128), perspective_params)
        assert result.mode == "RGBA"

    def test_fog_has_some_transparency(self, enabled_service, perspective_params):
        """Fog layer should not be fully opaque everywhere."""
        result = enabled_service.generate_fog_layer((128, 128), perspective_params)
        arr = np.array(result)
        # Alpha channel should have variation (not all 255)
        assert arr[:, :, 3].min() < 255 or arr[:, :, 3].max() < 255
