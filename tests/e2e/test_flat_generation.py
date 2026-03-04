"""E2E tests for flat-mode tile generation pipeline."""

import io

import pytest
from PIL import Image

from .conftest import API_KEY_HEADERS, start_and_wait

pytestmark = pytest.mark.e2e


class TestFlatGenerationE2E:
    """Full flat pipeline: generate -> poll -> verify tiles -> assemble."""

    def test_flat_generation_completes(self, stubbed_client, e2e_project_flat):
        status = start_and_wait(stubbed_client, "e2e-flat")
        assert status["status"] == "completed"
        assert status["failed_tiles"] == 0

    def test_flat_tile_grid(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cols"] > 0
        assert data["rows"] > 0
        assert data["tile_size"] == 512
        assert data["overlap"] == 64
        assert len(data["tiles"]) == data["cols"] * data["rows"]

    def test_flat_tile_reference_image(self, stubbed_client, generated_flat_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles/0/0/reference")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(resp.content))
        assert img.width > 0 and img.height > 0

    def test_flat_tile_generated_image(self, stubbed_client, generated_flat_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles/0/0/generated")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        img = Image.open(io.BytesIO(resp.content))
        assert img.width > 0

    def test_flat_tile_thumbnail(self, stubbed_client, generated_flat_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles/0/0/thumbnail")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        assert max(img.size) <= 256

    def test_flat_generation_cancel(self, stubbed_client, e2e_project_flat):
        # Start generation
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/generate",
            json={"skip_existing": True},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200
        # Immediately cancel
        resp = stubbed_client.post("/api/projects/e2e-flat/generate/cancel")
        # May be 200 or may already be completed (stubs are fast)
        assert resp.status_code in (200, 404)


class TestFlatTileOffsets:
    def test_tile_offset_crud(self, stubbed_client, e2e_project_flat):
        # Set offset
        resp = stubbed_client.put(
            "/api/projects/e2e-flat/tiles/0/0/offset",
            json={"dx": 5, "dy": -3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dx"] == 5
        assert data["dy"] == -3

        # Get offset
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles/0/0/offset")
        assert resp.status_code == 200
        assert resp.json()["dx"] == 5

        # Get all offsets
        resp = stubbed_client.get("/api/projects/e2e-flat/tiles/offsets")
        assert resp.status_code == 200
        assert len(resp.json()["offsets"]) >= 1


class TestFlatStyleReference:
    def test_style_reference_lifecycle(self, stubbed_client, e2e_project_flat):
        # Upload
        img = Image.new("RGB", (64, 64), (200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/style-reference",
            files={"file": ("style.png", buf, "image/png")},
        )
        assert resp.status_code == 200

        # HEAD check
        resp = stubbed_client.head("/api/projects/e2e-flat/style-reference")
        assert resp.status_code == 200

        # GET
        resp = stubbed_client.get("/api/projects/e2e-flat/style-reference")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]

        # DELETE
        resp = stubbed_client.delete("/api/projects/e2e-flat/style-reference")
        assert resp.status_code == 200

        # Verify gone
        resp = stubbed_client.get("/api/projects/e2e-flat/style-reference")
        assert resp.status_code == 404
