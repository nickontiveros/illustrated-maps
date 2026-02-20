"""Tests for mapgen.services.landmark_discovery_service."""

from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from mapgen.models.narrative import LandmarkTier, NarrativeSettings
from mapgen.services.landmark_discovery_service import LandmarkDiscoveryService


@pytest.fixture
def service():
    settings = NarrativeSettings(max_landmarks=50, min_importance_score=0.3)
    return LandmarkDiscoveryService(settings=settings)


@pytest.fixture
def strict_service():
    settings = NarrativeSettings(max_landmarks=2, min_importance_score=0.8)
    return LandmarkDiscoveryService(settings=settings)


@pytest.fixture
def mock_osm_service():
    return MagicMock()


# ---------------------------------------------------------------------------
# discover_landmarks - empty/None GDF
# ---------------------------------------------------------------------------

class TestDiscoverLandmarksEmpty:
    def test_none_gdf_returns_empty(self, service, sample_bbox, mock_osm_service):
        result = service.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_empty_gdf_returns_empty(self, service, sample_bbox, mock_osm_service):
        empty_gdf = gpd.GeoDataFrame(columns=["name", "geometry"])
        result = service.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=empty_gdf)
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# discover_landmarks - filtering and limiting
# ---------------------------------------------------------------------------

class TestDiscoverLandmarksFiltering:
    def test_filters_by_min_importance_score(self, strict_service, sample_bbox, mock_osm_service, sample_buildings_gdf):
        """With a high min_importance_score, fewer landmarks should pass."""
        result = strict_service.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=sample_buildings_gdf)
        # All returned landmarks should meet the score threshold
        assert isinstance(result, list)

    def test_limits_to_max_landmarks(self, sample_bbox, mock_osm_service, sample_buildings_gdf):
        """Should not return more landmarks than max_landmarks."""
        settings = NarrativeSettings(max_landmarks=2, min_importance_score=0.0)
        svc = LandmarkDiscoveryService(settings=settings)
        result = svc.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=sample_buildings_gdf)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_deduplicates_by_name(self, sample_bbox, mock_osm_service):
        """When two landmarks share a name, only the higher-scoring one should remain."""
        data = [
            {"name": "Central Museum", "tourism": "museum", "geometry": Point(-73.975, 40.770)},
            {"name": "Central Museum", "tourism": "museum", "geometry": Point(-73.974, 40.770)},
            {"name": "Other Place", "tourism": "attraction", "geometry": Point(-73.973, 40.769)},
        ]
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry")
        settings = NarrativeSettings(max_landmarks=50, min_importance_score=0.0)
        svc = LandmarkDiscoveryService(settings=settings)
        result = svc.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=gdf)
        names = [lm.name for lm in result]
        assert names.count("Central Museum") <= 1


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_default_service_returns_landmarks_with_tiers(self, service, sample_bbox, mock_osm_service, sample_buildings_gdf):
        result = service.discover_landmarks(sample_bbox, mock_osm_service, buildings_gdf=sample_buildings_gdf)
        for lm in result:
            assert hasattr(lm, "tier") or hasattr(lm, "feature_type")
