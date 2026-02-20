"""Tests for mapgen.services.palette_service."""

import numpy as np
import pytest
from PIL import Image

from mapgen.services.palette_service import PaletteService


# ---------------------------------------------------------------------------
# Constructor / from_preset
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_no_palette(self):
        svc = PaletteService()
        assert svc.palette is None or svc.palette == []

    def test_custom_palette(self):
        palette = ["#FF0000", "#00FF00", "#0000FF"]
        svc = PaletteService(palette=palette, enforcement_strength=0.7)
        assert svc.enforcement_strength == 0.7


class TestFromPreset:
    def test_vintage_tourist(self):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        assert isinstance(svc, PaletteService)

    def test_modern_pop(self):
        svc = PaletteService.from_preset("modern_pop", strength=0.5)
        assert isinstance(svc, PaletteService)

    def test_ink_wash(self):
        svc = PaletteService.from_preset("ink_wash", strength=0.5)
        assert isinstance(svc, PaletteService)

    def test_unknown_preset_raises(self):
        with pytest.raises((KeyError, ValueError)):
            PaletteService.from_preset("nonexistent_preset", strength=0.5)


# ---------------------------------------------------------------------------
# PRESETS
# ---------------------------------------------------------------------------

class TestPresets:
    def test_has_three_presets(self):
        from mapgen.services.palette_service import PRESETS
        assert len(PRESETS) == 3

    def test_preset_keys(self):
        from mapgen.services.palette_service import PRESETS
        assert set(PRESETS.keys()) == {"vintage_tourist", "modern_pop", "ink_wash"}

    def test_presets_are_lists_of_hex(self):
        from mapgen.services.palette_service import PRESETS
        for name, colors in PRESETS.items():
            assert isinstance(colors, list), f"{name} is not a list"
            for c in colors:
                assert c.startswith("#"), f"{name} has non-hex color {c}"


# ---------------------------------------------------------------------------
# build_prompt_instruction
# ---------------------------------------------------------------------------

class TestBuildPromptInstruction:
    def test_returns_string(self):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.build_prompt_instruction()
        assert isinstance(result, str)

    def test_contains_hex_colors(self):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.build_prompt_instruction()
        assert "#" in result


# ---------------------------------------------------------------------------
# clamp_to_palette
# ---------------------------------------------------------------------------

class TestClampToPalette:
    def test_returns_same_size(self, solid_red_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.clamp_to_palette(solid_red_image)
        assert result.size == solid_red_image.size

    def test_returns_rgba(self, solid_red_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.clamp_to_palette(solid_red_image)
        assert result.mode == "RGBA"

    def test_preserves_alpha(self, transparent_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=1.0)
        result = svc.clamp_to_palette(transparent_image)
        arr = np.array(result)
        np.testing.assert_array_equal(arr[:, :, 3], 0)

    def test_zero_strength_preserves_image(self, gradient_image):
        svc = PaletteService(palette=["#FF0000", "#00FF00"], enforcement_strength=0.0)
        result = svc.clamp_to_palette(gradient_image)
        src = np.array(gradient_image)[:, :, :3]
        res = np.array(result)[:, :, :3]
        np.testing.assert_array_almost_equal(src, res, decimal=0)


# ---------------------------------------------------------------------------
# analyze_compliance
# ---------------------------------------------------------------------------

class TestAnalyzeCompliance:
    def test_returns_dict(self, solid_red_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.analyze_compliance(solid_red_image)
        assert isinstance(result, dict)

    def test_dict_has_expected_keys(self, gradient_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.analyze_compliance(gradient_image)
        assert "mean_distance" in result
        assert "compliance_score" in result
        assert "per_color_coverage" in result
        assert "outlier_percentage" in result

    def test_compliance_score_range(self, solid_red_image):
        svc = PaletteService.from_preset("vintage_tourist", strength=0.5)
        result = svc.analyze_compliance(solid_red_image)
        assert 0.0 <= result["compliance_score"] <= 1.0


# ---------------------------------------------------------------------------
# hex_to_rgb / rgb_to_hex
# ---------------------------------------------------------------------------

class TestHexConversions:
    def test_hex_to_rgb(self):
        r, g, b = PaletteService.hex_to_rgb("#FF8000")
        assert r == 255
        assert g == 128
        assert b == 0

    def test_hex_to_rgb_lowercase(self):
        r, g, b = PaletteService.hex_to_rgb("#ff8000")
        assert r == 255
        assert g == 128
        assert b == 0

    def test_rgb_to_hex(self):
        result = PaletteService.rgb_to_hex(255, 128, 0)
        assert result.upper() == "#FF8000"

    def test_roundtrip(self):
        original = "#AABB33"
        r, g, b = PaletteService.hex_to_rgb(original)
        reconstructed = PaletteService.rgb_to_hex(r, g, b)
        assert reconstructed.upper() == original


# ---------------------------------------------------------------------------
# extract_palette_from_image
# ---------------------------------------------------------------------------

class TestExtractPaletteFromImage:
    def test_returns_list_of_hex(self, gradient_image):
        svc = PaletteService()
        result = svc.extract_palette_from_image(gradient_image, n_colors=4)
        assert isinstance(result, list)
        assert len(result) == 4
        for c in result:
            assert c.startswith("#")

    def test_solid_image_returns_similar_colors(self, solid_red_image):
        svc = PaletteService()
        result = svc.extract_palette_from_image(solid_red_image, n_colors=1)
        r, g, b = PaletteService.hex_to_rgb(result[0])
        assert r > 200
