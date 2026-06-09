"""V2 API tests: project CRUD and the three pipeline stages end-to-end.

Uses a standalone FastAPI app with only the v2 router (V1 routers pull
heavy geo dependencies), the synthetic-town source instead of live OSM,
and stub asset generation. TestClient executes background tasks after
each response, so stage completion is observable on the next request.
"""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mapgen.api.routers import v2 as v2_router
from mapgen.v2 import pipeline


@pytest.fixture
def client(tmp_path, monkeypatch, source):
    monkeypatch.setenv("MAPGEN_PROJECTS_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "fetch_source", lambda project, cache_dir=None: source)
    # Fresh job registry per test.
    monkeypatch.setattr(v2_router, "jobs", v2_router.JobRegistry())
    app = FastAPI()
    app.include_router(v2_router.router, prefix="/api/v2/projects")
    return TestClient(app)


@pytest.fixture
def project_payload(region, small_canvas):
    return {
        "name": "Test Town",
        "region": region.model_dump(),
        "output": small_canvas.model_dump(),
        "pois": [
            {"name": "Old Lighthouse", "lat": 40.725, "lon": -73.978, "tier": 1},
            {"name": "Maritime Museum", "lat": 40.745, "lon": -74.0, "tier": 2},
        ],
    }


def _create(client, payload) -> str:
    response = client.post("/api/v2/projects", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_crud_lifecycle(client, project_payload):
    assert client.get("/api/v2/projects").json() == []
    project_id = _create(client, project_payload)
    assert project_id == "test_town"

    listed = client.get("/api/v2/projects").json()
    assert [p["id"] for p in listed] == ["test_town"]
    assert listed[0]["poi_count"] == 2
    assert not listed[0]["has_plan"]

    detail = client.get(f"/api/v2/projects/{project_id}").json()
    assert detail["config"]["name"] == "Test Town"

    # Update: add a POI.
    config = detail["config"]
    config["pois"].append({"name": "Fish Market", "lat": 40.747, "lon": -73.999, "tier": 3})
    assert client.put(f"/api/v2/projects/{project_id}", json=config).json()["poi_count"] == 3

    assert client.delete(f"/api/v2/projects/{project_id}").status_code == 204
    assert client.get("/api/v2/projects").json() == []


def test_duplicate_create_conflicts(client, project_payload):
    _create(client, project_payload)
    assert client.post("/api/v2/projects", json=project_payload).status_code == 409


def test_missing_project_404(client):
    assert client.get("/api/v2/projects/nope").status_code == 404
    assert client.post("/api/v2/projects/nope/plan").status_code == 404


def test_plan_stage(client, project_payload):
    project_id = _create(client, project_payload)
    assert client.post(f"/api/v2/projects/{project_id}/plan").status_code == 202

    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["plan"]["state"] == "done", status["plan"]

    plan = client.get(f"/api/v2/projects/{project_id}/plan").json()
    assert plan["name"] == "Test Town"
    assert plan["roads"] and plan["pois"]

    preview = client.get(f"/api/v2/projects/{project_id}/preview.svg")
    assert preview.status_code == 200
    assert b"<svg" in preview.content


def test_assets_require_plan(client, project_payload):
    project_id = _create(client, project_payload)
    response = client.post(f"/api/v2/projects/{project_id}/assets", json={"stub": True})
    assert response.status_code == 409


def test_full_pipeline_via_api(client, project_payload):
    project_id = _create(client, project_payload)
    client.post(f"/api/v2/projects/{project_id}/plan")

    # Assets (stub).
    assert client.post(
        f"/api/v2/projects/{project_id}/assets", json={"stub": True}
    ).status_code == 202
    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["assets"]["state"] == "done", status["assets"]
    assert status["assets"]["total"] > 0

    listed = client.get(f"/api/v2/projects/{project_id}/assets").json()
    assert all(a["cached"] for a in listed)
    one = client.get(listed[0]["url"])
    assert one.status_code == 200
    assert one.headers["content-type"] == "image/png"

    # Compose at reduced scale.
    assert client.post(
        f"/api/v2/projects/{project_id}/compose", json={"scale": 0.25}
    ).status_code == 202
    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["compose"]["state"] == "done", status["compose"]

    poster = client.get(f"/api/v2/projects/{project_id}/poster")
    assert poster.status_code == 200
    assert poster.headers["content-type"] == "image/png"

    summary = client.get(f"/api/v2/projects/{project_id}").json()
    assert summary["has_plan"] and summary["has_poster"]


def test_stage_error_is_reported(client, project_payload, monkeypatch):
    project_id = _create(client, project_payload)
    monkeypatch.setattr(
        pipeline, "fetch_source", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OSM down"))
    )
    client.post(f"/api/v2/projects/{project_id}/plan")
    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["plan"]["state"] == "error"
    assert "OSM down" in status["plan"]["detail"]


def test_status_persists_to_disk(client, project_payload, tmp_path):
    project_id = _create(client, project_payload)
    client.post(f"/api/v2/projects/{project_id}/plan")
    assert (tmp_path / project_id / "status.json").exists()


def test_v1_projects_are_not_listed(client, tmp_path, project_payload):
    v1_dir = tmp_path / "old_v1"
    v1_dir.mkdir()
    (v1_dir / "project.yaml").write_text(
        "name: V1\nregion: {north: 1, south: 0, east: 1, west: 0}\ntiles: {size: 2048}\n"
    )
    _create(client, project_payload)
    listed = client.get("/api/v2/projects").json()
    assert [p["id"] for p in listed] == ["test_town"]
