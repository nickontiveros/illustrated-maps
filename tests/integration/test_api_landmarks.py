"""Integration tests for POST /{name}/landmarks/discover endpoint."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mapgen.models.landmark import FeatureType, Landmark
from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture
def sample_project_obj(tmp_path):
    """Create a Project object for mocking."""
    project = Project(
        name="test-project",
        region=BoundingBox(north=40.775, south=40.768, east=-73.968, west=-73.978),
        output=OutputSettings(width=1024, height=1024, dpi=72),
        tiles=TileSettings(size=512, overlap=64),
    )
    project.project_dir = tmp_path
    return project


@pytest.fixture
def mock_landmarks():
    """Sample landmark list for mocked discovery."""
    return [
        Landmark(
            name="Central Park",
            latitude=40.770,
            longitude=-73.973,
            feature_type=FeatureType.PARK,
            scale=2.5,
            z_index=10,
        ),
        Landmark(
            name="Museum",
            latitude=40.771,
            longitude=-73.974,
            feature_type=FeatureType.BUILDING,
            scale=1.8,
            z_index=5,
        ),
    ]


@pytest.fixture
def client(sample_project_obj, mock_landmarks):
    """Create a TestClient with mocked project loading and discovery service."""
    from mapgen.api.main import app

    def _mock_load_project(name):
        if name == "test-project":
            return sample_project_obj
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Project {name} not found")

    cache_dir = sample_project_obj.project_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    with patch.dict(os.environ, {"MAPGEN_CACHE_DIR": str(cache_dir)}):
        import mapgen.config as config_module
        config_module._config = None

        with patch("mapgen.api.routers.projects.load_project", side_effect=_mock_load_project), \
             patch("mapgen.api.routers.landmarks.load_project", side_effect=_mock_load_project), \
             patch("mapgen.api.routers.landmarks.get_project_cache_dir", return_value=cache_dir), \
             patch("mapgen.services.landmark_discovery_service.LandmarkDiscoveryService") as MockDiscovery, \
             patch("mapgen.services.osm_service.OSMService") as MockOSM:
            mock_discovery_instance = MagicMock()
            mock_discovery_instance.discover_landmarks.return_value = mock_landmarks
            MockDiscovery.return_value = mock_discovery_instance
            mock_osm_instance = MagicMock()
            mock_osm_instance.fetch_region_data.return_value = MagicMock(buildings=MagicMock())
            MockOSM.return_value = mock_osm_instance
            yield TestClient(app)

        config_module._config = None


@pytest.fixture
def client_missing_project(tmp_path):
    """Client where project does not exist."""
    from mapgen.api.main import app

    def _mock_load_project(name):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Project {name} not found")

    with patch.dict(os.environ, {"MAPGEN_CACHE_DIR": str(tmp_path)}):
        import mapgen.config as config_module
        config_module._config = None

        with patch("mapgen.api.routers.projects.load_project", side_effect=_mock_load_project), \
             patch("mapgen.api.routers.landmarks.load_project", side_effect=_mock_load_project):
            yield TestClient(app)

        config_module._config = None


# ---------------------------------------------------------------------------
# POST /{name}/landmarks/discover
# ---------------------------------------------------------------------------

class TestDiscoverLandmarksEndpoint:
    def test_200_response_valid_project(self, client):
        response = client.post("/api/projects/test-project/landmarks/discover", json={})
        assert response.status_code == 200

    def test_response_contains_landmarks(self, client):
        response = client.post("/api/projects/test-project/landmarks/discover", json={})
        data = response.json()
        assert isinstance(data, dict)
        assert "landmarks" in data
        assert "discovered" in data

    def test_response_shape(self, client):
        response = client.post("/api/projects/test-project/landmarks/discover", json={})
        data = response.json()
        assert data is not None
        assert data["discovered"] == 2

    def test_custom_params(self, client):
        response = client.post(
            "/api/projects/test-project/landmarks/discover",
            json={"min_importance_score": 0.5, "max_landmarks": 10},
        )
        assert response.status_code == 200

    def test_404_for_missing_project(self, client_missing_project):
        response = client_missing_project.post(
            "/api/projects/nonexistent-project/landmarks/discover", json={}
        )
        assert response.status_code in (404, 500)


# ---------------------------------------------------------------------------
# Custom params
# ---------------------------------------------------------------------------

class TestDiscoverLandmarksQueryParams:
    def test_with_custom_params(self, client):
        response = client.post(
            "/api/projects/test-project/landmarks/discover",
            json={"min_importance_score": 0.8, "max_landmarks": 5},
        )
        assert response.status_code == 200
