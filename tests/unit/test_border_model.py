"""Tests for mapgen.models.border."""

import pytest
from pydantic import ValidationError

from mapgen.models.border import BorderSettings, BorderStyle, LegendItem


# ---------------------------------------------------------------------------
# BorderStyle enum
# ---------------------------------------------------------------------------

class TestBorderStyle:
    def test_all_styles_exist(self):
        expected = {"VINTAGE_SCROLL", "ART_DECO", "MODERN_MINIMAL", "ORNATE_VICTORIAN"}
        assert {s.name for s in BorderStyle} == expected


# ---------------------------------------------------------------------------
# LegendItem model
# ---------------------------------------------------------------------------

class TestLegendItem:
    def test_minimal_legend_item(self):
        item = LegendItem(label="Forest", color="#228B22")
        assert item.label == "Forest"
        assert item.color == "#228B22"
        assert item.symbol == "rect"

    def test_legend_item_custom_symbol(self):
        item = LegendItem(label="Peak", color="#000000", symbol="triangle")
        assert item.symbol == "triangle"


# ---------------------------------------------------------------------------
# BorderSettings model
# ---------------------------------------------------------------------------

class TestBorderSettings:
    def test_defaults(self):
        bs = BorderSettings()
        assert bs.enabled is False
        assert bs.style == BorderStyle.VINTAGE_SCROLL
        assert bs.margin == 200
        assert bs.show_compass is True
        assert bs.show_legend is True
        assert bs.ornament_opacity == 0.8

    def test_enabled(self):
        bs = BorderSettings(enabled=True)
        assert bs.enabled is True

    def test_style_art_deco(self):
        bs = BorderSettings(style=BorderStyle.ART_DECO)
        assert bs.style == BorderStyle.ART_DECO

    def test_margin_lower_bound(self):
        bs = BorderSettings(margin=50)
        assert bs.margin == 50

    def test_margin_below_min_raises(self):
        with pytest.raises(ValidationError):
            BorderSettings(margin=49)

    def test_margin_upper_bound(self):
        bs = BorderSettings(margin=500)
        assert bs.margin == 500

    def test_margin_above_max_raises(self):
        with pytest.raises(ValidationError):
            BorderSettings(margin=501)

    def test_ornament_opacity_lower_bound(self):
        bs = BorderSettings(ornament_opacity=0.0)
        assert bs.ornament_opacity == 0.0

    def test_ornament_opacity_upper_bound(self):
        bs = BorderSettings(ornament_opacity=1.0)
        assert bs.ornament_opacity == 1.0

    def test_ornament_opacity_below_min_raises(self):
        with pytest.raises(ValidationError):
            BorderSettings(ornament_opacity=-0.1)

    def test_ornament_opacity_above_max_raises(self):
        with pytest.raises(ValidationError):
            BorderSettings(ornament_opacity=1.1)
