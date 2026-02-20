"""Integration tests for PUT /{name} with new fields (typography, road_style, atmosphere, border, narrative)."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


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

    bbox = BoundingBox(north=40.775, south=40.768, east=-73.968, west=-73.978)
    project = Project(name="test-project", region=bbox)
    project.project_dir = project_dir
    project.ensure_directories()
    project.to_yaml(project_dir / "project.yaml")
    return project


@pytest.fixture
def client(projects_dir, sample_project_on_disk, tmp_path):
    """FastAPI test client with patched directories."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch.dict(os.environ, {
        "MAPGEN_CACHE_DIR": str(cache_dir),
        "MAPGEN_OUTPUT_DIR": str(tmp_path / "output"),
    }):
        import mapgen.config as config_module
        config_module._config = None

        with patch("mapgen.api.routers.projects.get_projects_dir", return_value=projects_dir):
            from mapgen.api.main import app
            with TestClient(app) as c:
                yield c

        config_module._config = None


# ---------------------------------------------------------------------------
# PUT /{name} - update title, subtitle
# ---------------------------------------------------------------------------

class TestUpdateProjectBasicFields:
    def test_update_title(self, client):
        response = client.put("/api/projects/test-project", json={"title": "My Beautiful Map"})
        assert response.status_code == 200

    def test_update_subtitle(self, client):
        response = client.put("/api/projects/test-project", json={"subtitle": "A walking tour"})
        assert response.status_code == 200

    def test_update_title_and_subtitle(self, client):
        response = client.put("/api/projects/test-project", json={"title": "My Map", "subtitle": "2024 Edition"})
        assert response.status_code == 200

    def test_404_for_missing_project(self, client):
        response = client.put("/api/projects/nonexistent-project", json={"title": "Nope"})
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# PUT /{name} - update border
# ---------------------------------------------------------------------------

class TestUpdateProjectBorder:
    def test_update_border(self, client):
        response = client.put("/api/projects/test-project", json={
            "border": {
                "enabled": True,
                "style": "vintage_scroll",
                "margin": 150,
            }
        })
        assert response.status_code == 200

    def test_update_border_disabled(self, client):
        response = client.put("/api/projects/test-project", json={
            "border": {"enabled": False}
        })
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# PUT /{name} - update narrative
# ---------------------------------------------------------------------------

class TestUpdateProjectNarrative:
    def test_update_narrative(self, client):
        response = client.put("/api/projects/test-project", json={
            "narrative": {
                "auto_discover": True,
                "max_landmarks": 30,
            }
        })
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# PUT /{name} - update style.typography
# ---------------------------------------------------------------------------

class TestUpdateProjectTypography:
    def test_update_typography(self, client):
        response = client.put("/api/projects/test-project", json={
            "style": {
                "typography": {
                    "enabled": True,
                    "font_scale": 1.5,
                    "halo_width": 3,
                }
            }
        })
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# PUT /{name} - update style.road_style
# ---------------------------------------------------------------------------

class TestUpdateProjectRoadStyle:
    def test_update_road_style(self, client):
        response = client.put("/api/projects/test-project", json={
            "style": {
                "road_style": {
                    "enabled": True,
                    "motorway_exaggeration": 25.0,
                    "wobble_amount": 2.0,
                }
            }
        })
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# PUT /{name} - update style.atmosphere
# ---------------------------------------------------------------------------

class TestUpdateProjectAtmosphere:
    def test_update_atmosphere(self, client):
        response = client.put("/api/projects/test-project", json={
            "style": {
                "atmosphere": {
                    "enabled": True,
                    "haze_strength": 0.6,
                    "contrast_reduction": 0.3,
                }
            }
        })
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Round-trip test: update then GET
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_update_then_get(self, client):
        # Update the project
        update_resp = client.put("/api/projects/test-project", json={"title": "Round Trip Map"})
        assert update_resp.status_code == 200
        # Then retrieve it
        get_resp = client.get("/api/projects/test-project")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert isinstance(data, dict)
        assert data.get("name") == "test-project"
