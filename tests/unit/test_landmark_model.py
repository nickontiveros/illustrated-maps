"""Tests for Landmark model."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mapgen.models.landmark import FeatureType, Landmark


class TestLandmarkConstruction:
    """Test landmark creation and validation."""

    def test_valid_construction(self, sample_landmark):
        assert sample_landmark.name == "Test Building"
        assert sample_landmark.latitude == 40.770

    def test_defaults(self):
        lm = Landmark(name="Test", latitude=0, longitude=0)
        assert lm.feature_type == FeatureType.BUILDING
        assert lm.scale == 1.5
        assert lm.z_index == 0
        assert lm.rotation == 0
        assert lm.photo is None
        assert lm.logo is None

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            Landmark(name="", latitude=0, longitude=0)

    def test_rejects_invalid_latitude(self):
        with pytest.raises(ValidationError):
            Landmark(name="Test", latitude=91, longitude=0)

    def test_rejects_invalid_scale(self):
        with pytest.raises(ValidationError):
            Landmark(name="Test", latitude=0, longitude=0, scale=10.0)


class TestLandmarkProperties:
    """Test computed properties."""

    def test_coordinates(self, sample_landmark):
        lat, lon = sample_landmark.coordinates
        assert lat == 40.770
        assert lon == -73.973

    def test_resolve_photo_path(self):
        lm = Landmark(name="Test", latitude=0, longitude=0, photo="landmarks/photo.jpg")
        path = lm.resolve_photo_path(Path("/project"))
        assert path == Path("/project/landmarks/photo.jpg")

    def test_resolve_photo_path_none(self):
        lm = Landmark(name="Test", latitude=0, longitude=0)
        assert lm.resolve_photo_path(Path("/project")) is None

    def test_resolve_logo_path(self):
        lm = Landmark(name="Test", latitude=0, longitude=0, logo="logos/logo.png")
        path = lm.resolve_logo_path(Path("/project"))
        assert path == Path("/project/logos/logo.png")

    def test_resolve_logo_path_none(self):
        lm = Landmark(name="Test", latitude=0, longitude=0)
        assert lm.resolve_logo_path(Path("/project")) is None


class TestFeatureType:
    """Test feature type enum."""

    def test_all_values(self):
        expected = {
            "building", "mountain", "river", "park", "monument",
            "stadium", "campus", "airport", "natural",
        }
        actual = {ft.value for ft in FeatureType}
        assert actual == expected
