"""Tests for mapgen.models.typography."""

import pytest
from pydantic import ValidationError

from mapgen.models.typography import (
    FONT_COLORS,
    FONT_SIZES,
    FontTier,
    Label,
    TextPath,
    TypographySettings,
)


# ---------------------------------------------------------------------------
# FontTier enum
# ---------------------------------------------------------------------------

class TestFontTier:
    def test_all_tiers_exist(self):
        expected = {"TITLE", "SUBTITLE", "DISTRICT", "ROAD_MAJOR", "ROAD_MINOR", "WATER", "PARK"}
        assert {t.name for t in FontTier} == expected

    def test_tier_values_are_strings(self):
        for tier in FontTier:
            assert isinstance(tier.value, str)


# ---------------------------------------------------------------------------
# FONT_SIZES / FONT_COLORS lookups
# ---------------------------------------------------------------------------

class TestFontMappings:
    def test_font_sizes_has_all_tiers(self):
        for tier in FontTier:
            assert tier in FONT_SIZES

    def test_font_sizes_are_min_max_tuples(self):
        for tier, (lo, hi) in FONT_SIZES.items():
            assert lo <= hi, f"{tier}: min {lo} > max {hi}"

    def test_font_colors_has_all_tiers(self):
        for tier in FontTier:
            assert tier in FONT_COLORS

    def test_font_colors_are_hex_strings(self):
        for tier, color in FONT_COLORS.items():
            assert color.startswith("#"), f"{tier} color {color} is not hex"


# ---------------------------------------------------------------------------
# Label model
# ---------------------------------------------------------------------------

class TestLabel:
    def test_minimal_label(self):
        label = Label(text="Hello", tier=FontTier.TITLE, latitude=40.0, longitude=-73.0)
        assert label.text == "Hello"
        assert label.rotation == 0.0
        assert label.font_size is None
        assert label.color is None

    def test_label_with_all_fields(self):
        label = Label(
            text="Park",
            tier=FontTier.PARK,
            latitude=40.5,
            longitude=-73.5,
            rotation=45.0,
            font_size=24,
            color="#FF0000",
        )
        assert label.rotation == 45.0
        assert label.font_size == 24
        assert label.color == "#FF0000"


# ---------------------------------------------------------------------------
# TextPath model
# ---------------------------------------------------------------------------

class TestTextPath:
    def test_minimal_text_path(self):
        tp = TextPath(text="Broadway", tier=FontTier.ROAD_MAJOR, points=[(0, 0), (100, 100)])
        assert tp.text == "Broadway"
        assert len(tp.points) == 2
        assert tp.font_size is None
        assert tp.color is None

    def test_text_path_with_overrides(self):
        tp = TextPath(
            text="5th Ave",
            tier=FontTier.ROAD_MINOR,
            points=[(0, 0), (50, 50), (100, 0)],
            font_size=12,
            color="#333333",
        )
        assert tp.font_size == 12
        assert tp.color == "#333333"


# ---------------------------------------------------------------------------
# TypographySettings model
# ---------------------------------------------------------------------------

class TestTypographySettings:
    def test_defaults(self):
        ts = TypographySettings()
        assert ts.enabled is False
        assert ts.road_labels is True
        assert ts.district_labels is True
        assert ts.water_labels is True
        assert ts.park_labels is True
        assert ts.title_text is None
        assert ts.subtitle_text is None
        assert ts.font_scale == 1.0
        assert ts.halo_width == 2
        assert ts.max_labels == 200
        assert ts.min_road_length_px == 100

    def test_enabled_setting(self):
        ts = TypographySettings(enabled=True)
        assert ts.enabled is True

    def test_font_scale_lower_bound(self):
        ts = TypographySettings(font_scale=0.5)
        assert ts.font_scale == 0.5

    def test_font_scale_below_min_raises(self):
        with pytest.raises(ValidationError):
            TypographySettings(font_scale=0.4)

    def test_font_scale_upper_bound(self):
        ts = TypographySettings(font_scale=3.0)
        assert ts.font_scale == 3.0

    def test_font_scale_above_max_raises(self):
        with pytest.raises(ValidationError):
            TypographySettings(font_scale=3.1)

    def test_halo_width_bounds(self):
        assert TypographySettings(halo_width=0).halo_width == 0
        assert TypographySettings(halo_width=5).halo_width == 5

    def test_halo_width_out_of_bounds(self):
        with pytest.raises(ValidationError):
            TypographySettings(halo_width=-1)
        with pytest.raises(ValidationError):
            TypographySettings(halo_width=6)

    def test_title_and_subtitle(self):
        ts = TypographySettings(title_text="My Map", subtitle_text="A Tour")
        assert ts.title_text == "My Map"
        assert ts.subtitle_text == "A Tour"
