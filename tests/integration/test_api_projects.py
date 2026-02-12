"""Integration tests for project API endpoints using FastAPI TestClient."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mapgen.models.project import BoundingBox, Project

# FastAPI/httpx may not be installed; skip these tests if not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture
def projects_dir(tmp_path):
    """Create a temporary projects directory."""
    pdir = tmp_path / "projects"
    pdir.mkdir()
    return pdir


@pytest.fixture
def sample_project_on_disk(projects_dir):
    """Create a sample project on disk."""
    project_dir = projects_dir / "test-project"
    project_dir.mkdir()

    bbox = BoundingBox(north=40.78, south=40.76, east=-73.96, west=-73.98)
    project = Project(name="test-project", region=bbox)
    project.project_dir = project_dir
    project.ensure_directories()
    project.to_yaml(project_dir / "project.yaml")
    return project


@pytest.fixture
def client(projects_dir, tmp_path):
    """FastAPI test client with patched directories."""
    # Must patch before importing the app to avoid mounting non-existent dirs
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch.dict(os.environ, {
        "MAPGEN_CACHE_DIR": str(cache_dir),
        "MAPGEN_OUTPUT_DIR": str(tmp_path / "output"),
    }):
        # Reset the global config so it picks up our env vars
        import mapgen.config as config_module
        config_module._config = None

        with patch("mapgen.api.routers.projects.get_projects_dir", return_value=projects_dir):
            from mapgen.api.main import app
            with TestClient(app) as c:
                yield c

        config_module._config = None


class TestHealthCheck:
    """Test health endpoint."""

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestListProjects:
    """Test project listing."""

    def test_empty_list(self, client):
        response = client.get("/api/projects")
        assert response.status_code == 200
        assert response.json() == []

    def test_lists_existing_project(self, client, sample_project_on_disk):
        response = client.get("/api/projects")
        assert response.status_code == 200
        projects = response.json()
        assert len(projects) == 1
        assert projects[0]["name"] == "test-project"


class TestGetProject:
    """Test getting a specific project."""

    def test_existing_project(self, client, sample_project_on_disk):
        response = client.get("/api/projects/test-project")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-project"
        assert "area_km2" in data
        assert "grid_cols" in data

    def test_missing_project_404(self, client):
        response = client.get("/api/projects/nonexistent")
        assert response.status_code == 404


class TestCreateProject:
    """Test project creation."""

    def test_create_project(self, client):
        response = client.post("/api/projects", json={
            "name": "new-project",
            "region": {
                "north": 41.0, "south": 40.0,
                "east": -73.0, "west": -74.0,
            },
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-project"

    def test_create_duplicate_409(self, client, sample_project_on_disk):
        response = client.post("/api/projects", json={
            "name": "test-project",
            "region": {
                "north": 41.0, "south": 40.0,
                "east": -73.0, "west": -74.0,
            },
        })
        assert response.status_code == 409


class TestUpdateProject:
    """Test project updates."""

    def test_update_output_settings(self, client, sample_project_on_disk):
        response = client.put("/api/projects/test-project", json={
            "output": {"width": 2000, "height": 2000, "dpi": 150},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["output"]["width"] == 2000

    def test_update_nonexistent_404(self, client):
        response = client.put("/api/projects/nonexistent", json={
            "output": {"width": 2000, "height": 2000, "dpi": 150},
        })
        assert response.status_code == 404


class TestDeleteProject:
    """Test project deletion."""

    def test_delete_project(self, client, sample_project_on_disk, projects_dir):
        response = client.delete("/api/projects/test-project")
        assert response.status_code == 200
        assert not (projects_dir / "test-project").exists()

    def test_delete_nonexistent_404(self, client):
        response = client.delete("/api/projects/nonexistent")
        assert response.status_code == 404
