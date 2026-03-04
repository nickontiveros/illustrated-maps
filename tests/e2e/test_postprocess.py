"""E2E tests for post-processing pipeline."""

import io
import time

import pytest
from PIL import Image

from .conftest import API_KEY_HEADERS

pytestmark = pytest.mark.e2e


class TestPostprocessStatus:
    def test_status_initial(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assembled"] is False
        assert data["composed"] is False

    def test_status_after_assembly(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/status")
        assert resp.status_code == 200
        assert resp.json()["assembled"] is True


class TestPerspective:
    def test_apply_perspective(self, stubbed_client, assembled_project):
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/postprocess/perspective",
            json={"angle": 35.0, "convergence": 0.7, "vertical_scale": 0.4, "horizon_margin": 0.15},
        )
        assert resp.status_code == 200

        # Verify stage image exists
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/perspective/image")
        assert resp.status_code == 200


class TestLabels:
    def test_add_labels_requires_typography(self, stubbed_client, assembled_project):
        """Labels fail if typography is not enabled in settings."""
        resp = stubbed_client.post("/api/projects/e2e-flat/postprocess/labels")
        # Should fail because typography is not enabled
        assert resp.status_code == 400

    def test_add_labels_with_typography(self, stubbed_client, assembled_project):
        """Enable typography, then add labels."""
        # Enable typography in settings
        stubbed_client.put("/api/projects/e2e-flat", json={
            "style": {"typography": {"enabled": True}},
        })
        resp = stubbed_client.post("/api/projects/e2e-flat/postprocess/labels")
        assert resp.status_code == 200

    def test_add_labels_with_shields(self, stubbed_client, assembled_project):
        stubbed_client.put("/api/projects/e2e-flat", json={
            "style": {"typography": {"enabled": True}},
        })
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/postprocess/labels",
            json={"include_shields": True},
        )
        assert resp.status_code == 200


class TestBorder:
    def test_add_border_requires_settings(self, stubbed_client, assembled_project):
        resp = stubbed_client.post("/api/projects/e2e-flat/postprocess/border")
        assert resp.status_code == 400

    def test_add_border(self, stubbed_client, assembled_project):
        stubbed_client.put("/api/projects/e2e-flat", json={
            "border": {"enabled": True, "style": "vintage_scroll", "margin": 50},
        })
        resp = stubbed_client.post("/api/projects/e2e-flat/postprocess/border")
        assert resp.status_code == 200


class TestOutpaint:
    def test_outpaint_starts(self, stubbed_client, assembled_project):
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/postprocess/outpaint",
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data


class TestPipeline:
    def test_run_pipeline(self, stubbed_client, assembled_project):
        # Enable typography for the labels step
        stubbed_client.put("/api/projects/e2e-flat", json={
            "style": {"typography": {"enabled": True}},
        })
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/postprocess/pipeline",
            json={"steps": ["labels", "perspective"]},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200
        assert "task_id" in resp.json()


class TestStageImages:
    def test_get_assembled_image(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/assembled/image")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        assert img.width > 0

    def test_get_assembled_thumbnail(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/assembled/image?size=256")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        assert max(img.size) <= 256


class TestExportPSD:
    def test_export_psd(self, stubbed_client, assembled_project):
        resp = stubbed_client.post("/api/projects/e2e-flat/postprocess/export-psd")
        assert resp.status_code == 200
        assert "application/octet-stream" in resp.headers["content-type"]
        assert len(resp.content) > 0
