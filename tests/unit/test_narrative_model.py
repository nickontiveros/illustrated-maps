"""Tests for mapgen.models.narrative."""

import pytest
from pydantic import ValidationError

from mapgen.models.narrative import (
    TIER_SCALES,
    OSM_TAG_TO_CATEGORY,
    ActivityCategory,
    ActivityMarker,
    HistoricalMarker,
    LandmarkTier,
    NarrativeSettings,
)


# ---------------------------------------------------------------------------
# LandmarkTier enum + TIER_SCALES
# ---------------------------------------------------------------------------

class TestLandmarkTier:
    def test_all_tiers_exist(self):
        expected = {"MAJOR", "NOTABLE", "MINOR"}
        assert {t.name for t in LandmarkTier} == expected

    def test_tier_scales_mapping(self):
        assert TIER_SCALES[LandmarkTier.MAJOR] == 2.5
        assert TIER_SCALES[LandmarkTier.NOTABLE] == 1.8
        assert TIER_SCALES[LandmarkTier.MINOR] == 1.2


# ---------------------------------------------------------------------------
# ActivityCategory enum
# ---------------------------------------------------------------------------

class TestActivityCategory:
    def test_has_15_values(self):
        assert len(ActivityCategory) == 15

    def test_expected_categories_exist(self):
        names = {c.name for c in ActivityCategory}
        for expected in [
            "DINING", "SHOPPING", "SWIMMING", "HIKING", "MUSEUM",
            "THEATER", "PARK", "BEACH", "SPORTS", "NIGHTLIFE",
            "ACCOMMODATION", "TRANSPORT", "VIEWPOINT", "HISTORIC", "WORSHIP",
        ]:
            assert expected in names


# ---------------------------------------------------------------------------
# OSM_TAG_TO_CATEGORY
# ---------------------------------------------------------------------------

class TestOSMTagToCategory:
    def test_is_non_empty_dict(self):
        assert isinstance(OSM_TAG_TO_CATEGORY, dict)
        assert len(OSM_TAG_TO_CATEGORY) > 0

    def test_values_are_activity_categories(self):
        for tag, cat in OSM_TAG_TO_CATEGORY.items():
            assert isinstance(cat, ActivityCategory), f"Tag '{tag}' maps to non-ActivityCategory"


# ---------------------------------------------------------------------------
# ActivityMarker model
# ---------------------------------------------------------------------------

class TestActivityMarker:
    def test_create_marker(self):
        m = ActivityMarker(
            name="Central Cafe",
            category=ActivityCategory.DINING,
            latitude=40.77,
            longitude=-73.97,
        )
        assert m.name == "Central Cafe"
        assert m.category == ActivityCategory.DINING


# ---------------------------------------------------------------------------
# HistoricalMarker model
# ---------------------------------------------------------------------------

class TestHistoricalMarker:
    def test_create_marker(self):
        m = HistoricalMarker(
            name="Old Fort",
            latitude=40.77,
            longitude=-73.97,
        )
        assert m.name == "Old Fort"


# ---------------------------------------------------------------------------
# NarrativeSettings model
# ---------------------------------------------------------------------------

class TestNarrativeSettings:
    def test_defaults(self):
        ns = NarrativeSettings()
        assert ns.auto_discover is False
        assert ns.max_landmarks == 50
        assert ns.show_activities is False
        assert ns.max_activity_markers == 100
        assert ns.min_importance_score == 0.3

    def test_max_landmarks_lower_bound(self):
        ns = NarrativeSettings(max_landmarks=0)
        assert ns.max_landmarks == 0

    def test_max_landmarks_upper_bound(self):
        ns = NarrativeSettings(max_landmarks=200)
        assert ns.max_landmarks == 200

    def test_max_landmarks_below_min_raises(self):
        with pytest.raises(ValidationError):
            NarrativeSettings(max_landmarks=-1)

    def test_max_landmarks_above_max_raises(self):
        with pytest.raises(ValidationError):
            NarrativeSettings(max_landmarks=201)

    def test_min_importance_score_lower_bound(self):
        ns = NarrativeSettings(min_importance_score=0.0)
        assert ns.min_importance_score == 0.0

    def test_min_importance_score_upper_bound(self):
        ns = NarrativeSettings(min_importance_score=1.0)
        assert ns.min_importance_score == 1.0

    def test_min_importance_score_below_min_raises(self):
        with pytest.raises(ValidationError):
            NarrativeSettings(min_importance_score=-0.01)

    def test_min_importance_score_above_max_raises(self):
        with pytest.raises(ValidationError):
            NarrativeSettings(min_importance_score=1.01)
