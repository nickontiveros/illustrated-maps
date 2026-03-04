"""E2E tests for API key handling and system endpoints."""

import os
from unittest.mock import patch

import pytest

from .conftest import API_KEY_HEADERS

pytestmark = pytest.mark.e2e


class TestConfigShowsKeyPresence:
    def test_config_with_keys(self, stubbed_client):
        resp = stubbed_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        # Env vars set by fixture → server-side keys present
        assert data["has_google_api_key"] is True
        assert data["has_mapbox_token"] is True

    def test_config_with_header_keys(self, stubbed_client):
        resp = stubbed_client.get("/api/config", headers={
            "X-Google-API-Key": "from-header",
            "X-Mapbox-Access-Token": "from-header",
        })
        data = resp.json()
        assert data["has_google_api_key"] is True
        assert data["google_api_key_source"] == "client"


class TestGenerationRequiresKeys:
    def test_generation_without_keys_fails(self, stubbed_client, e2e_project_flat):
        """When no API keys are provided via headers and env vars are absent, generation should fail."""
        # Override env to remove keys
        import mapgen.config as config_module
        saved = config_module._config
        try:
            config_module._config = None
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "MAPBOX_ACCESS_TOKEN": ""}, clear=False):
                config_module._config = None
                resp = stubbed_client.post(
                    "/api/projects/e2e-flat/generate",
                    json={"skip_existing": True},
                    # No API key headers
                )
                # Should fail with 400 (missing keys)
                assert resp.status_code == 400
        finally:
            config_module._config = saved


class TestKeysFromHeaders:
    def test_keys_from_headers(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/generate",
            json={"skip_existing": True},
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200


class TestActiveTasks:
    def test_active_tasks_endpoint(self, stubbed_client):
        resp = stubbed_client.get("/api/tasks/active")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestHealthEndpoint:
    def test_health(self, stubbed_client):
        resp = stubbed_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
