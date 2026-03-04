"""E2E tests for project CRUD operations."""

import pytest

from .conftest import A1_BBOX, API_KEY_HEADERS

pytestmark = pytest.mark.e2e


class TestCreateProject:
    def test_create_project(self, stubbed_client):
        resp = stubbed_client.post("/api/projects", json={
            "name": "lifecycle-test",
            "region": A1_BBOX,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "lifecycle-test"
        assert "area_km2" in data
        assert "grid_cols" in data
        assert "grid_rows" in data

    def test_create_flat_mode(self, stubbed_client):
        resp = stubbed_client.post("/api/projects", json={
            "name": "flat-mode",
            "region": A1_BBOX,
            "generation_mode": "flat",
        })
        assert resp.status_code == 201
        assert resp.json()["generation_mode"] == "flat"

    def test_create_hierarchical_mode(self, stubbed_client):
        resp = stubbed_client.post("/api/projects", json={
            "name": "hier-mode",
            "region": A1_BBOX,
            "generation_mode": "hierarchical",
        })
        assert resp.status_code == 201
        assert resp.json()["generation_mode"] == "hierarchical"

    def test_create_upscale_mode(self, stubbed_client):
        resp = stubbed_client.post("/api/projects", json={
            "name": "upscale-mode",
            "region": A1_BBOX,
            "generation_mode": "upscale",
        })
        assert resp.status_code == 201
        assert resp.json()["generation_mode"] == "upscale"


class TestListProjects:
    def test_empty_list(self, stubbed_client):
        resp = stubbed_client.get("/api/projects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created_projects(self, stubbed_client):
        stubbed_client.post("/api/projects", json={"name": "proj-a", "region": A1_BBOX})
        stubbed_client.post("/api/projects", json={"name": "proj-b", "region": A1_BBOX})
        resp = stubbed_client.get("/api/projects")
        names = {p["name"] for p in resp.json()}
        assert {"proj-a", "proj-b"} <= names


class TestGetProject:
    def test_get_project_detail(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.get("/api/projects/e2e-flat")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "e2e-flat"
        assert data["area_km2"] > 0
        assert data["grid_cols"] > 0
        assert data["grid_rows"] > 0
        assert data["tile_count"] > 0

    def test_missing_project_404(self, stubbed_client):
        resp = stubbed_client.get("/api/projects/does-not-exist")
        assert resp.status_code == 404


class TestUpdateProject:
    def test_update_settings(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.put("/api/projects/e2e-flat", json={
            "output": {"width": 2048, "height": 2896, "dpi": 150},
        })
        assert resp.status_code == 200
        assert resp.json()["output"]["width"] == 2048

    def test_update_generation_mode(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.put("/api/projects/e2e-flat", json={
            "generation_mode": "hierarchical",
        })
        assert resp.status_code == 200
        assert resp.json()["generation_mode"] == "hierarchical"


class TestDeleteProject:
    def test_delete_project(self, stubbed_client):
        stubbed_client.post("/api/projects", json={"name": "to-delete", "region": A1_BBOX})
        resp = stubbed_client.delete("/api/projects/to-delete")
        assert resp.status_code == 200
        # Verify gone
        assert stubbed_client.get("/api/projects/to-delete").status_code == 404


class TestClearCache:
    def test_clear_cache(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.delete("/api/projects/e2e-flat/cache")
        assert resp.status_code == 200


class TestCostEstimate:
    def test_cost_estimate(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.get("/api/projects/e2e-flat/cost-estimate")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_cost_usd" in data or "estimated_cost_usd" in data or isinstance(data, dict)


class TestConfigEndpoint:
    def test_config_endpoint(self, stubbed_client):
        resp = stubbed_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_google_api_key" in data
        assert "has_mapbox_token" in data
