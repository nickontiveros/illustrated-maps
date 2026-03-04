"""E2E tests for upscale generation mode."""

import pytest

from .conftest import API_KEY_HEADERS, start_and_wait

pytestmark = pytest.mark.e2e


class TestUpscaleGeneration:
    def test_upscale_full_e2e(self, stubbed_client, e2e_project_upscale):
        status = start_and_wait(stubbed_client, "e2e-upscale")
        assert status["status"] == "completed"

        # Verify assembled image exists
        resp = stubbed_client.get("/api/projects/e2e-upscale/postprocess/status")
        assert resp.status_code == 200
        assert resp.json()["assembled"] is True

    def test_upscale_progress_phases(self, stubbed_client):
        """Start upscale generation and verify it completes."""
        stubbed_client.post("/api/projects", json={
            "name": "upscale-phases",
            "region": {"north": 41.0, "south": 40.0, "east": -73.0355, "west": -73.9645},
            "output": {"width": 1024, "height": 1449, "dpi": 72},
            "generation_mode": "upscale",
        })
        status = start_and_wait(stubbed_client, "upscale-phases")
        assert status["status"] == "completed"

    def test_upscale_lanczos_fallback(self, stubbed_client, e2e_project_upscale):
        """Without Real-ESRGAN, upscale falls back to Lanczos resize."""
        # Our stubs don't install Real-ESRGAN, so the service should use Lanczos
        status = start_and_wait(stubbed_client, "e2e-upscale")
        assert status["status"] == "completed"
