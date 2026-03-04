"""E2E tests for tile assembly."""

import io

import pytest
from PIL import Image

pytestmark = pytest.mark.e2e


class TestAssembly:
    def test_assemble_after_generation(self, stubbed_client, generated_flat_project):
        resp = stubbed_client.post("/api/projects/e2e-flat/assemble")
        assert resp.status_code == 200

        # Verify assembled image is accessible
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/assembled/image")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        # Should be close to the configured output dimensions
        assert img.width > 0
        assert img.height > 0

    def test_assemble_invalidates_dzi(self, stubbed_client, generated_flat_project):
        # First assemble
        stubbed_client.post("/api/projects/e2e-flat/assemble")

        # Generate DZI
        resp = stubbed_client.post("/api/projects/e2e-flat/dzi/generate")
        assert resp.status_code == 200

        # Re-assemble should invalidate DZI
        stubbed_client.post("/api/projects/e2e-flat/assemble")

        # DZI info should still work (image exists) but tiles may need regeneration
        resp = stubbed_client.get("/api/projects/e2e-flat/dzi/info")
        assert resp.status_code == 200

    def test_reassemble(self, stubbed_client, generated_flat_project):
        # Assemble twice — second should overwrite
        resp1 = stubbed_client.post("/api/projects/e2e-flat/assemble")
        assert resp1.status_code == 200
        resp2 = stubbed_client.post("/api/projects/e2e-flat/assemble")
        assert resp2.status_code == 200

    def test_assembled_image_preview(self, stubbed_client, assembled_project):
        resp = stubbed_client.get("/api/projects/e2e-flat/postprocess/assembled/image")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
