"""E2E tests for Deep Zoom Image (DZI) generation and serving."""

import pytest

pytestmark = pytest.mark.e2e


class TestDZI:
    def test_dzi_info(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/dzi/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["width"] > 0
        assert data["height"] > 0
        assert "tile_size" in data
        assert "max_level" in data

    def test_dzi_generate(self, stubbed_client, assembled_project):
        resp = stubbed_client.post("/api/projects/e2e-flat/dzi/generate")
        assert resp.status_code == 200

    def test_dzi_descriptor(self, stubbed_client, assembled_project):
        # Generate DZI first
        stubbed_client.post("/api/projects/e2e-flat/dzi/generate")
        resp = stubbed_client.get("/api/projects/e2e-flat/dzi/assembled.dzi")
        assert resp.status_code == 200
        # Should be XML content
        content = resp.text
        assert "Image" in content or "<?xml" in content

    def test_dzi_tile_serving(self, stubbed_client, assembled_project):
        # Generate DZI tiles
        stubbed_client.post("/api/projects/e2e-flat/dzi/generate")
        # Fetch info to know what levels exist
        info = stubbed_client.get("/api/projects/e2e-flat/dzi/info").json()
        # Request lowest level tile (level 0 is a single pixel)
        resp = stubbed_client.get("/api/projects/e2e-flat/dzi/tile/0/0/0")
        assert resp.status_code == 200

    def test_dzi_auto_generate(self, stubbed_client, assembled_project):
        """Requesting a tile without pre-generating triggers auto-generation."""
        resp = stubbed_client.get("/api/projects/e2e-flat/dzi/assembled_files/0/0_0.jpg")
        # Should auto-generate and return the tile
        assert resp.status_code == 200
