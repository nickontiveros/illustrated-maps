"""Tests for mapgen.models.road_style."""

import pytest
from pydantic import ValidationError

from mapgen.models.road_style import ROAD_STYLE_PRESETS, RoadStyleSettings


class TestRoadStyleSettings:
    def test_defaults(self):
        rs = RoadStyleSettings()
        assert rs.enabled is False
        assert rs.motorway_exaggeration == 20.0
        assert rs.primary_exaggeration == 15.0
        assert rs.secondary_exaggeration == 10.0
        assert rs.residential_exaggeration == 5.0
        assert rs.wobble_amount == 1.5
        assert rs.wobble_frequency == 0.02
        assert rs.overlay_on_output is False
        assert rs.preset is None

    def test_enabled(self):
        rs = RoadStyleSettings(enabled=True)
        assert rs.enabled is True

    # --- motorway_exaggeration bounds ---
    def test_motorway_exaggeration_lower_bound(self):
        rs = RoadStyleSettings(motorway_exaggeration=1.0)
        assert rs.motorway_exaggeration == 1.0

    def test_motorway_exaggeration_below_min_raises(self):
        with pytest.raises(ValidationError):
            RoadStyleSettings(motorway_exaggeration=0.5)

    def test_motorway_exaggeration_upper_bound(self):
        rs = RoadStyleSettings(motorway_exaggeration=50.0)
        assert rs.motorway_exaggeration == 50.0

    def test_motorway_exaggeration_above_max_raises(self):
        with pytest.raises(ValidationError):
            RoadStyleSettings(motorway_exaggeration=51.0)

    # --- wobble_amount bounds ---
    def test_wobble_amount_lower_bound(self):
        rs = RoadStyleSettings(wobble_amount=0.0)
        assert rs.wobble_amount == 0.0

    def test_wobble_amount_above_max_raises(self):
        with pytest.raises(ValidationError):
            RoadStyleSettings(wobble_amount=5.1)

    def test_wobble_amount_upper_bound(self):
        rs = RoadStyleSettings(wobble_amount=5.0)
        assert rs.wobble_amount == 5.0

    def test_preset_assignment(self):
        rs = RoadStyleSettings(preset="vintage_tourist")
        assert rs.preset == "vintage_tourist"


class TestRoadStylePresets:
    def test_presets_has_expected_keys(self):
        expected = {"vintage_tourist", "modern_clean", "ink_sketch"}
        assert set(ROAD_STYLE_PRESETS.keys()) == expected

    def test_presets_are_road_style_settings(self):
        for name, preset in ROAD_STYLE_PRESETS.items():
            assert isinstance(preset, RoadStyleSettings), f"Preset {name} should be a RoadStyleSettings"
