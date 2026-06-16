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


def test_composition_round_trip_and_applied_to_plan(client, project_payload):
    project_id = _create(client, project_payload)
    # No spec saved yet -> an all-auto default.
    default = client.get(f"/api/v2/projects/{project_id}/composition")
    assert default.status_code == 200
    assert default.json()["warp"]["mode"] == "auto"

    # Save a spec that hides one POI.
    spec = {"features": {"pois": {"exclude": ["old_lighthouse"]}}}
    put = client.put(f"/api/v2/projects/{project_id}/composition", json=spec)
    assert put.status_code == 200 and put.json()["saved"] is True

    # It persists...
    saved = client.get(f"/api/v2/projects/{project_id}/composition").json()
    assert saved["features"]["pois"]["exclude"] == ["old_lighthouse"]

    # ...and the plan stage honors it.
    assert client.post(f"/api/v2/projects/{project_id}/plan").status_code == 202
    plan = client.get(f"/api/v2/projects/{project_id}/plan").json()
    ids = [p["id"] for p in plan["pois"]]
    assert "old_lighthouse" not in ids
    assert "maritime_museum" in ids


def test_geocode_endpoint(client, monkeypatch, project_payload):
    monkeypatch.setattr(
        pipeline,
        "geocode_place",
        lambda q: {"query": q, "lat": 33.43, "lon": -112.01, "feature_type": "airport", "display_name": "PHX"},
    )
    r = client.get("/api/v2/projects/geocode", params={"q": "Phoenix Sky Harbor"})
    assert r.status_code == 200 and r.json()["feature_type"] == "airport"
    assert client.get("/api/v2/projects/geocode", params={"q": "  "}).status_code == 400
    # a real project id still resolves (no collision with the /geocode literal)
    pid = _create(client, project_payload)
    assert client.get(f"/api/v2/projects/{pid}").status_code == 200


def test_asset_prompt_override(client, project_payload):
    project_id = _create(client, project_payload)
    client.post(f"/api/v2/projects/{project_id}/plan")
    client.post(f"/api/v2/projects/{project_id}/assets", json={"stub": True})

    assets = client.get(f"/api/v2/projects/{project_id}/assets").json()
    target = assets[0]
    assert "prompt_hints" in target and target["prompt_overridden"] is False

    # Edit one asset's prompt and regenerate just it.
    resp = client.post(
        f"/api/v2/projects/{project_id}/assets",
        json={"stub": True, "only_ids": [target["id"]], "prompt_overrides": {target["id"]: "a bright red barn"}},
    )
    assert resp.status_code == 202

    after = client.get(f"/api/v2/projects/{project_id}/assets").json()
    edited = next(a for a in after if a["id"] == target["id"])
    assert edited["prompt_hints"] == "a bright red barn"
    assert edited["prompt_overridden"] is True
    # The override persists (a fresh list still shows it).
    assert next(a for a in client.get(f"/api/v2/projects/{project_id}/assets").json()
                if a["id"] == target["id"])["prompt_overridden"] is True


def test_source_geojson_and_preview_plan(client, project_payload):
    project_id = _create(client, project_payload)
    # Both editor endpoints need the persisted source from a plan run.
    assert client.get(f"/api/v2/projects/{project_id}/source.geojson").status_code == 404
    assert client.post(f"/api/v2/projects/{project_id}/preview-plan", json={}).status_code == 404

    client.post(f"/api/v2/projects/{project_id}/plan")

    gj = client.get(f"/api/v2/projects/{project_id}/source.geojson")
    assert gj.status_code == 200
    data = gj.json()
    assert data["pois"] and all(p["id"] for p in data["pois"])  # ids present
    assert all(len(p["point"]) == 2 for p in data["pois"])  # normalized (u, v)
    assert "roads" in data and "ground" in data

    # Live preview honors an in-flight spec without persisting it.
    spec = {"features": {"pois": {"exclude": ["old_lighthouse"]}}}
    pv = client.post(f"/api/v2/projects/{project_id}/preview-plan", json=spec)
    assert pv.status_code == 200
    assert "<svg" in pv.json()["svg"]
    # preview-plan must not write composition.json.
    saved = client.get(f"/api/v2/projects/{project_id}/composition").json()
    assert saved["features"]["pois"]["exclude"] == []


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


def test_asset_flagging(client, project_payload):
    project_id = _create(client, project_payload)
    client.post(f"/api/v2/projects/{project_id}/plan")
    client.post(f"/api/v2/projects/{project_id}/assets", json={"stub": True})

    listed = client.get(f"/api/v2/projects/{project_id}/assets").json()
    assert all(a["flagged"] is False for a in listed)
    assert all("palette_score" in a and "palette_outlier" in a for a in listed)
    asset_id = listed[0]["id"]

    assert client.post(
        f"/api/v2/projects/{project_id}/assets/{asset_id}/flag", json={"flagged": True}
    ).json() == {"id": asset_id, "flagged": True}
    listed = client.get(f"/api/v2/projects/{project_id}/assets").json()
    assert [a["id"] for a in listed if a["flagged"]] == [asset_id]

    # Regenerating the flagged asset clears the flag (meta is rewritten).
    client.post(
        f"/api/v2/projects/{project_id}/assets",
        json={"stub": True, "force": True, "only_ids": [asset_id]},
    )
    listed = client.get(f"/api/v2/projects/{project_id}/assets").json()
    assert not any(a["flagged"] for a in listed)

    # Flagging an ungenerated asset 404s.
    assert client.post(
        f"/api/v2/projects/{project_id}/assets/never_generated/flag", json={"flagged": True}
    ).status_code == 404


def test_repaint_stage_via_api(client, region):
    payload = {
        "name": "Repaint Town",
        "region": region.model_dump(),
        # Large enough for the 512px quadrant grid at repaint_scale=1.0.
        "output": {"width_px": 1100, "height_px": 1500, "dpi": 72},
        "pois": [],
    }
    project_id = _create(client, payload)
    client.post(f"/api/v2/projects/{project_id}/plan")
    client.post(f"/api/v2/projects/{project_id}/assets", json={"stub": True})

    # No repaint yet: empty grid state, flagging conflicts.
    assert client.get(f"/api/v2/projects/{project_id}/repaint/quadrants").json() == {
        "grid": None,
        "quadrants": [],
    }
    assert client.post(
        f"/api/v2/projects/{project_id}/repaint/quadrants/0/0/flag", json={"flagged": True}
    ).status_code == 409

    # Default-mode (single) dry run: always exactly 1 call, instant.
    dry = client.post(
        f"/api/v2/projects/{project_id}/repaint", json={"dry_run": True}
    ).json()
    assert dry == {
        "stage": "repaint", "state": "dry_run", "mode": "single",
        "calls_planned": 1, "estimated_cost_usd": dry["estimated_cost_usd"],
    }

    # Default-mode (single) stub run end to end.
    started = client.post(
        f"/api/v2/projects/{project_id}/repaint", json={"stub": True, "scale": 0.5}
    )
    assert started.status_code == 202 and started.json()["mode"] == "single"
    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["repaint"]["state"] == "done", status["repaint"]
    assert client.get(f"/api/v2/projects/{project_id}/poster").status_code == 200
    # Single mode runs no quadrant grid.
    assert client.get(f"/api/v2/projects/{project_id}/repaint/quadrants").json()["grid"] is None

    # Tiled dry run: synchronous plan, no spend.
    dry = client.post(
        f"/api/v2/projects/{project_id}/repaint",
        json={"mode": "tiled", "dry_run": True, "repaint_scale": 1.0},
    ).json()
    assert dry["state"] == "dry_run" and dry["mode"] == "tiled"
    assert dry["calls_planned"] > 0 and dry["estimated_cost_usd"] > 0

    # Tiled stub run (IdentityPainter) end to end.
    assert client.post(
        f"/api/v2/projects/{project_id}/repaint",
        json={"mode": "tiled", "stub": True, "repaint_scale": 1.0},
    ).status_code == 202
    status = client.get(f"/api/v2/projects/{project_id}/status").json()
    assert status["repaint"]["state"] == "done", status["repaint"]

    state = client.get(f"/api/v2/projects/{project_id}/repaint/quadrants").json()
    assert state["grid"]["cols"] >= 2 and state["calls_made"] > 0
    painted = [q for q in state["quadrants"] if q["status"] == "generated"]
    assert painted

    # Flag one painted quadrant, then unflag.
    q = painted[0]
    flagged = client.post(
        f"/api/v2/projects/{project_id}/repaint/quadrants/{q['x']}/{q['y']}/flag",
        json={"flagged": True},
    ).json()
    assert flagged["status"] == "flagged"
    state = client.get(f"/api/v2/projects/{project_id}/repaint/quadrants").json()
    assert {"x": q["x"], "y": q["y"], "status": "flagged"} in state["quadrants"]


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


def test_poi_update_marks_plan_stale(client, project_payload, tmp_path):
    """Editing POIs after planning must flag the plan as stale until re-planned."""
    project_id = _create(client, project_payload)
    client.post(f"/api/v2/projects/{project_id}/plan")
    detail = client.get(f"/api/v2/projects/{project_id}").json()
    assert detail["has_plan"] and not detail["plan_stale"]

    # Updates touch project.yaml after plan.json -> stale. mtime can tie on
    # coarse filesystems, so nudge the plan's mtime backwards first.
    import os

    plan_path = tmp_path / project_id / pipeline.PLAN_FILENAME
    past = plan_path.stat().st_mtime - 10
    os.utime(plan_path, (past, past))

    config = detail["config"]
    config["pois"].append({"name": "New Pier", "lat": 40.73, "lon": -73.99, "tier": 3})
    updated = client.put(f"/api/v2/projects/{project_id}", json=config).json()
    assert updated["poi_count"] == 3
    assert updated["plan_stale"]

    # Re-planning clears the flag and picks up the new POI.
    client.post(f"/api/v2/projects/{project_id}/plan")
    detail = client.get(f"/api/v2/projects/{project_id}").json()
    assert not detail["plan_stale"]
    plan = client.get(f"/api/v2/projects/{project_id}/plan").json()
    assert any(slot["name"] == "New Pier" for slot in plan["pois"])


def test_update_rejected_while_stage_runs(client, project_payload):
    project_id = _create(client, project_payload)
    detail = client.get(f"/api/v2/projects/{project_id}").json()
    v2_router.jobs.update(project_id, "plan", state="running")
    response = client.put(f"/api/v2/projects/{project_id}", json=detail["config"])
    assert response.status_code == 409
