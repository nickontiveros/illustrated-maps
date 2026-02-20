"""Tests for mapgen.models.atmosphere."""

import pytest
from pydantic import ValidationError

from mapgen.models.atmosphere import AtmosphereSettings


class TestAtmosphereSettings:
    def test_defaults(self):
        a = AtmosphereSettings()
        assert a.enabled is False
        assert a.haze_color == "#C8D8E8"
        assert a.haze_strength == 0.3
        assert a.contrast_reduction == 0.2
        assert a.saturation_reduction == 0.15
        assert a.gradient_curve == 1.5

    def test_enabled(self):
        a = AtmosphereSettings(enabled=True)
        assert a.enabled is True

    # --- haze_strength bounds ---
    def test_haze_strength_lower_bound(self):
        a = AtmosphereSettings(haze_strength=0.0)
        assert a.haze_strength == 0.0

    def test_haze_strength_upper_bound(self):
        a = AtmosphereSettings(haze_strength=1.0)
        assert a.haze_strength == 1.0

    def test_haze_strength_below_min_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(haze_strength=-0.01)

    def test_haze_strength_above_max_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(haze_strength=1.01)

    # --- contrast_reduction bounds ---
    def test_contrast_reduction_lower_bound(self):
        a = AtmosphereSettings(contrast_reduction=0.0)
        assert a.contrast_reduction == 0.0

    def test_contrast_reduction_upper_bound(self):
        a = AtmosphereSettings(contrast_reduction=0.5)
        assert a.contrast_reduction == 0.5

    def test_contrast_reduction_above_max_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(contrast_reduction=0.51)

    # --- saturation_reduction bounds ---
    def test_saturation_reduction_lower_bound(self):
        a = AtmosphereSettings(saturation_reduction=0.0)
        assert a.saturation_reduction == 0.0

    def test_saturation_reduction_upper_bound(self):
        a = AtmosphereSettings(saturation_reduction=0.5)
        assert a.saturation_reduction == 0.5

    def test_saturation_reduction_above_max_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(saturation_reduction=0.51)

    # --- gradient_curve bounds ---
    def test_gradient_curve_lower_bound(self):
        a = AtmosphereSettings(gradient_curve=0.5)
        assert a.gradient_curve == 0.5

    def test_gradient_curve_upper_bound(self):
        a = AtmosphereSettings(gradient_curve=3.0)
        assert a.gradient_curve == 3.0

    def test_gradient_curve_below_min_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(gradient_curve=0.4)

    def test_gradient_curve_above_max_raises(self):
        with pytest.raises(ValidationError):
            AtmosphereSettings(gradient_curve=3.1)

    def test_custom_haze_color(self):
        a = AtmosphereSettings(haze_color="#FFFFFF")
        assert a.haze_color == "#FFFFFF"
