"""Tests for mapgen.services.narrative_service."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from mapgen.models.narrative import ActivityCategory, NarrativeSettings
from mapgen.services.narrative_service import NarrativeService


@pytest.fixture
def service():
    settings = NarrativeSettings(show_activities=True, max_activity_markers=100)
    return NarrativeService(settings=settings)


@pytest.fixture
def limited_service():
    settings = NarrativeSettings(show_activities=True, max_activity_markers=2)
    return NarrativeService(settings=settings)


@pytest.fixture
def default_service():
    return NarrativeService()


# ---------------------------------------------------------------------------
# extract_activity_markers
# ---------------------------------------------------------------------------

class TestExtractActivityMarkers:
    def test_empty_gdf_returns_empty(self, service, sample_bbox):
        empty_gdf = gpd.GeoDataFrame(columns=["name", "amenity", "geometry"])
        result = service.extract_activity_markers(sample_bbox, empty_gdf)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_none_gdf_returns_empty(self, service, sample_bbox):
        result = service.extract_activity_markers(sample_bbox, None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_restaurant_maps_to_dining(self, service, sample_bbox):
        data = [
            {"name": "Pizzeria", "amenity": "restaurant", "geometry": Point(-73.974, 40.770)},
        ]
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry")
        result = service.extract_activity_markers(sample_bbox, gdf)
        assert len(result) >= 1
        assert result[0].category == ActivityCategory.DINING

    def test_limits_to_max_activity_markers(self, limited_service, sample_bbox):
        data = [
            {"name": f"Place {i}", "amenity": "restaurant", "geometry": Point(-73.974 + i * 0.001, 40.770)}
            for i in range(10)
        ]
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry")
        result = limited_service.extract_activity_markers(sample_bbox, gdf)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# extract_historical_markers
# ---------------------------------------------------------------------------

class TestExtractHistoricalMarkers:
    def test_empty_gdf_returns_empty(self, service):
        empty_gdf = gpd.GeoDataFrame(columns=["name", "historic", "geometry"])
        result = service.extract_historical_markers(empty_gdf)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_markers_from_historic_tag(self, service):
        data = [
            {"name": "Old Fort", "historic": "castle", "geometry": Point(-73.975, 40.770)},
            {"name": "Memorial", "historic": "memorial", "geometry": Point(-73.974, 40.769)},
        ]
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry")
        result = service.extract_historical_markers(gdf)
        assert len(result) >= 1
        names = [m.name for m in result]
        assert "Old Fort" in names

    def test_ignores_rows_without_historic(self, service):
        data = [
            {"name": "Modern Shop", "amenity": "shop", "geometry": Point(-73.975, 40.770)},
        ]
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry")
        result = service.extract_historical_markers(gdf)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# cluster_nearby_markers
# ---------------------------------------------------------------------------

class TestClusterNearbyMarkers:
    def test_empty_returns_empty(self, service):
        result = service.cluster_nearby_markers([], cluster_distance_degrees=0.001)
        assert result == []

    def test_no_duplicates_returns_all(self, service):
        from mapgen.models.narrative import ActivityMarker

        markers = [
            ActivityMarker(name="A", category=ActivityCategory.DINING, latitude=40.770, longitude=-73.974),
            ActivityMarker(name="B", category=ActivityCategory.SHOPPING, latitude=40.780, longitude=-73.960),
        ]
        result = service.cluster_nearby_markers(markers, cluster_distance_degrees=0.001)
        assert len(result) == 2

    def test_deduplicates_same_category_nearby(self, service):
        from mapgen.models.narrative import ActivityMarker

        markers = [
            ActivityMarker(name="Cafe 1", category=ActivityCategory.DINING, latitude=40.770, longitude=-73.974),
            ActivityMarker(name="Cafe 2", category=ActivityCategory.DINING, latitude=40.7701, longitude=-73.9741),
        ]
        result = service.cluster_nearby_markers(markers, cluster_distance_degrees=0.01)
        assert len(result) == 1

    def test_different_categories_not_clustered(self, service):
        from mapgen.models.narrative import ActivityMarker

        markers = [
            ActivityMarker(name="Cafe", category=ActivityCategory.DINING, latitude=40.770, longitude=-73.974),
            ActivityMarker(name="Shop", category=ActivityCategory.SHOPPING, latitude=40.7701, longitude=-73.9741),
        ]
        result = service.cluster_nearby_markers(markers, cluster_distance_degrees=0.01)
        assert len(result) == 2
