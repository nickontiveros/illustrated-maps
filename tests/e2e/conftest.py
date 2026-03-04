"""E2E test fixtures — stubbed FastAPI client and project lifecycle helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from .stub_services import (
    StubGeminiService,
    StubLandmarkService,
    StubOSMService,
    StubOutpaintingService,
    StubSatelliteService,
    StubTerrainService,
    StubWikipediaImageService,
)

# A1-proportional bbox near Central Park
A1_BBOX = {
    "north": 41.0,
    "south": 40.0,
    "east": -73.0355,
    "west": -73.9645,
}


def _all_service_patches():
    """Return a list of (target, replacement) tuples for every import site."""
    return [
        # --- Source modules: catches ALL local imports (from X import Y) ---
        ("mapgen.services.gemini_service.GeminiService", StubGeminiService),
        ("mapgen.services.satellite_service.SatelliteService", StubSatelliteService),
        ("mapgen.services.osm_service.OSMService", StubOSMService),
        ("mapgen.services.terrain_service.TerrainService", StubTerrainService),
        ("mapgen.services.wikipedia_image_service.WikipediaImageService", StubWikipediaImageService),
        ("mapgen.services.outpainting_service.OutpaintingService", StubOutpaintingService),
        # --- dependencies.py (module-level imports) ---
        ("mapgen.api.dependencies.GeminiService", StubGeminiService),
        ("mapgen.api.dependencies.SatelliteService", StubSatelliteService),
        # --- generation_service.py (module-level imports) ---
        ("mapgen.services.generation_service.GeminiService", StubGeminiService),
        ("mapgen.services.generation_service.SatelliteService", StubSatelliteService),
        ("mapgen.services.generation_service.OSMService", StubOSMService),
        ("mapgen.services.generation_service.TerrainService", StubTerrainService),
        # --- hierarchical_generation_service.py ---
        ("mapgen.services.hierarchical_generation_service.GeminiService", StubGeminiService),
        ("mapgen.services.hierarchical_generation_service.SatelliteService", StubSatelliteService),
        ("mapgen.services.hierarchical_generation_service.OSMService", StubOSMService),
        ("mapgen.services.hierarchical_generation_service.TerrainService", StubTerrainService),
        # --- upscale_generation_service.py ---
        ("mapgen.services.upscale_generation_service.GeminiService", StubGeminiService),
        ("mapgen.services.upscale_generation_service.SatelliteService", StubSatelliteService),
        ("mapgen.services.upscale_generation_service.OSMService", StubOSMService),
        ("mapgen.services.upscale_generation_service.TerrainService", StubTerrainService),
        # --- landmarks.py (module-level import) ---
        ("mapgen.api.routers.landmarks.LandmarkService", StubLandmarkService),
    ]


@pytest.fixture(scope="session")
def _stub_patches():
    """Pre-compute patch list once per session."""
    return _all_service_patches()


@pytest.fixture()
def projects_dir(tmp_path):
    """Temporary projects directory."""
    d = tmp_path / "projects"
    d.mkdir()
    return d


@pytest.fixture()
def stubbed_client(tmp_path, projects_dir, _stub_patches):
    """FastAPI TestClient with all external services stubbed out.

    - Patches every import site for Gemini, Satellite, OSM, Terrain, Wikipedia, etc.
    - Sets fake API keys in the environment so key-required endpoints don't 400.
    - Points project/cache/output dirs at tmp_path.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    env_overrides = {
        "MAPGEN_CACHE_DIR": str(cache_dir),
        "MAPGEN_OUTPUT_DIR": str(output_dir),
        "GOOGLE_API_KEY": "fake-google-key",
        "MAPBOX_ACCESS_TOKEN": "fake-mapbox-token",
    }

    # Build context-manager stack
    import contextlib

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env_overrides))

        # Reset global config so it picks up env vars
        import mapgen.config as config_module
        config_module._config = None

        stack.enter_context(
            patch("mapgen.api.routers.projects.get_projects_dir", return_value=projects_dir)
        )

        # Apply all service patches
        for target, replacement in _stub_patches:
            stack.enter_context(patch(target, replacement))

        from mapgen.api.main import app

        with TestClient(app) as client:
            yield client

        config_module._config = None


# ---------------------------------------------------------------------------
# API key header helpers
# ---------------------------------------------------------------------------

API_KEY_HEADERS = {
    "X-Google-API-Key": "fake-google-key",
    "X-Mapbox-Access-Token": "fake-mapbox-token",
}


# ---------------------------------------------------------------------------
# Project creation helpers
# ---------------------------------------------------------------------------

def _create_project(client: TestClient, name: str, generation_mode: str = "flat") -> dict:
    """Create a project via the API and return the response JSON."""
    resp = client.post("/api/projects", json={
        "name": name,
        "region": A1_BBOX,
        "output": {"width": 1024, "height": 1449, "dpi": 72},
        "tiles": {"size": 512, "overlap": 64},
        "generation_mode": generation_mode,
    })
    assert resp.status_code == 201, f"Create failed: {resp.text}"
    return resp.json()


@pytest.fixture()
def e2e_project_flat(stubbed_client):
    """Create a flat-mode project and return its detail dict."""
    return _create_project(stubbed_client, "e2e-flat", "flat")


@pytest.fixture()
def e2e_project_hierarchical(stubbed_client):
    """Create a hierarchical-mode project."""
    return _create_project(stubbed_client, "e2e-hier", "hierarchical")


@pytest.fixture()
def e2e_project_upscale(stubbed_client):
    """Create an upscale-mode project."""
    return _create_project(stubbed_client, "e2e-upscale", "upscale")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def wait_for_generation(client: TestClient, name: str, timeout: float = 30) -> dict:
    """Poll GET /api/projects/{name}/generate/status until completed or failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/projects/{name}/generate/status")
        assert resp.status_code == 200, f"Status check failed: {resp.text}"
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.1)
    raise TimeoutError(f"Generation for '{name}' did not complete within {timeout}s")


def start_and_wait(client: TestClient, name: str, **gen_kwargs) -> dict:
    """POST /generate then poll to completion. Returns final status."""
    body = {"skip_existing": True, **gen_kwargs}
    resp = client.post(
        f"/api/projects/{name}/generate",
        json=body,
        headers=API_KEY_HEADERS,
    )
    assert resp.status_code == 200, f"Start generation failed: {resp.text}"
    return wait_for_generation(client, name)


@pytest.fixture()
def generated_flat_project(stubbed_client, e2e_project_flat):
    """Flat project with generation completed."""
    status = start_and_wait(stubbed_client, "e2e-flat")
    assert status["status"] == "completed", f"Generation failed: {status}"
    return e2e_project_flat


@pytest.fixture()
def generated_hierarchical_project(stubbed_client, e2e_project_hierarchical):
    """Hierarchical project with generation completed."""
    status = start_and_wait(stubbed_client, "e2e-hier")
    assert status["status"] == "completed", f"Generation failed: {status}"
    return e2e_project_hierarchical


@pytest.fixture()
def generated_upscale_project(stubbed_client, e2e_project_upscale):
    """Upscale project with generation completed."""
    status = start_and_wait(stubbed_client, "e2e-upscale")
    assert status["status"] == "completed", f"Generation failed: {status}"
    return e2e_project_upscale


@pytest.fixture()
def assembled_project(stubbed_client, generated_flat_project):
    """Flat project with tiles assembled into final image."""
    name = "e2e-flat"
    resp = stubbed_client.post(f"/api/projects/{name}/assemble")
    assert resp.status_code == 200, f"Assemble failed: {resp.text}"
    return generated_flat_project
