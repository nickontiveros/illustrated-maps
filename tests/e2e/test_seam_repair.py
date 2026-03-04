"""E2E tests for seam detection and repair."""

import pytest

from .conftest import API_KEY_HEADERS

pytestmark = pytest.mark.e2e


class TestSeamRepair:
    def test_list_seams(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/seams")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_seams"] > 0
        assert len(data["seams"]) == data["total_seams"]

    def test_get_seam_detail(self, stubbed_client, assembled_project):
        # Get first seam
        resp = stubbed_client.get("/api/projects/e2e-flat/seams")
        seam_id = resp.json()["seams"][0]["id"]

        resp = stubbed_client.get(f"/api/projects/e2e-flat/seams/{seam_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orientation"] in ("horizontal", "vertical")
        assert data["width"] > 0
        assert data["height"] > 0

    def test_seam_preview(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/seams")
        seam_id = resp.json()["seams"][0]["id"]

        resp = stubbed_client.get(f"/api/projects/e2e-flat/seams/{seam_id}/preview")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]

    def test_repair_single_seam(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/seams")
        seam_id = resp.json()["seams"][0]["id"]

        resp = stubbed_client.post(
            f"/api/projects/e2e-flat/seams/{seam_id}/repair",
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200

    def test_repair_batch(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/seams")
        seam_ids = [s["id"] for s in resp.json()["seams"][:2]]

        resp = stubbed_client.post(
            "/api/projects/e2e-flat/seams/repair-batch",
            json={"seam_ids": seam_ids},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200

    def test_repair_all(self, stubbed_client, assembled_project):
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/seams/repair-all",
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200
