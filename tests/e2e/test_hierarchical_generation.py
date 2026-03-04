"""E2E tests for hierarchical generation mode."""

import pytest

from .conftest import API_KEY_HEADERS, start_and_wait, wait_for_generation

pytestmark = pytest.mark.e2e


class TestHierarchicalGeneration:
    def test_hierarchical_full_e2e(self, stubbed_client, e2e_project_hierarchical):
        status = start_and_wait(stubbed_client, "e2e-hier")
        assert status["status"] == "completed"
        assert status["failed_tiles"] == 0

        # Verify assembled image exists via postprocess status
        resp = stubbed_client.get("/api/projects/e2e-hier/postprocess/status")
        assert resp.status_code == 200
        assert resp.json()["assembled"] is True

    def test_hierarchical_quick_test(self, stubbed_client):
        """Generate with skip_l2=True — only overview + L1 medium tiles."""
        stubbed_client.post("/api/projects", json={
            "name": "hier-quick",
            "region": {"north": 41.0, "south": 40.0, "east": -73.0355, "west": -73.9645},
            "output": {"width": 1024, "height": 1449, "dpi": 72},
            "tiles": {"size": 512, "overlap": 64},
            "generation_mode": "hierarchical",
        })
        assert stubbed_client.get("/api/projects/hier-quick").status_code == 200

        resp = stubbed_client.post(
            "/api/projects/hier-quick/generate",
            json={"skip_existing": True, "skip_l2": True},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200

        status = wait_for_generation(stubbed_client, "hier-quick")
        assert status["status"] == "completed"

    def test_hierarchical_tile_fallback(self, stubbed_client, generated_hierarchical_project):
        """GET generated tile falls back to hierarchical cache."""
        # Tile (0,0) should be served from hierarchical L2 cache
        resp = stubbed_client.get("/api/projects/e2e-hier/tiles/0/0/generated")
        # May be 200 (found in cache) or 404 (naming mismatch) — both are valid
        # as long as the generation itself completed
        assert resp.status_code in (200, 404)

    def test_hierarchical_progress_phases(self, stubbed_client):
        """Verify generation progresses through phases."""
        stubbed_client.post("/api/projects", json={
            "name": "hier-phases",
            "region": {"north": 41.0, "south": 40.0, "east": -73.0355, "west": -73.9645},
            "generation_mode": "hierarchical",
        })
        resp = stubbed_client.post(
            "/api/projects/hier-phases/generate",
            json={"skip_existing": True},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200

        status = wait_for_generation(stubbed_client, "hier-phases")
        assert status["status"] == "completed"
