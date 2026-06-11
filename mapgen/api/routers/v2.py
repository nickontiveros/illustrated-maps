"""V2 API: project CRUD + the three pipeline stages (plan/assets/compose).

Stages run as FastAPI background tasks; their state is tracked in a small
in-process job registry and mirrored to ``status.json`` in the project
directory so the UI can poll it and state survives restarts. Stages are
coarse (minutes, not per-tile), so polling beats websockets here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ...v2 import pipeline
from ...v2.assets.manifest import slugify
from ...v2.types import PlanDocument

logger = logging.getLogger(__name__)

router = APIRouter()

STAGES = ("plan", "assets", "compose")


def projects_root() -> Path:
    import os

    return Path(os.environ.get("MAPGEN_PROJECTS_DIR", str(Path.cwd() / "projects")))


# --- job registry -----------------------------------------------------------


class JobState(BaseModel):
    stage: str
    state: str = "idle"  # idle | running | done | error
    detail: str = ""
    current: int = 0
    total: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], JobState] = {}
        self._lock = threading.Lock()

    def get(self, project_id: str, stage: str) -> JobState:
        with self._lock:
            return self._jobs.get((project_id, stage), JobState(stage=stage)).model_copy()

    def is_running(self, project_id: str) -> bool:
        with self._lock:
            return any(
                j.state == "running" for (pid, _), j in self._jobs.items() if pid == project_id
            )

    def update(self, project_id: str, stage: str, **changes) -> JobState:
        with self._lock:
            job = self._jobs.get((project_id, stage), JobState(stage=stage))
            job = job.model_copy(update=changes)
            self._jobs[(project_id, stage)] = job
        self._persist(project_id)
        return job

    def status(self, project_id: str) -> dict[str, JobState]:
        return {stage: self.get(project_id, stage) for stage in STAGES}

    def _persist(self, project_id: str) -> None:
        directory = projects_root() / project_id
        if not directory.is_dir():
            return
        payload = {stage: self.get(project_id, stage).model_dump() for stage in STAGES}
        try:
            (directory / "status.json").write_text(json.dumps(payload, indent=1))
        except OSError:  # status persistence is best-effort
            logger.warning("Could not persist status for %s", project_id)

    def load_persisted(self, project_id: str) -> None:
        path = projects_root() / project_id / "status.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            for stage, data in payload.items():
                if (project_id, stage) not in self._jobs and stage in STAGES:
                    job = JobState.model_validate(data)
                    if job.state == "running":  # stale from a previous process
                        job.state = "error"
                        job.detail = "interrupted"
                    self._jobs[(project_id, stage)] = job


jobs = JobRegistry()


def _run_stage(project_id: str, stage: str, work) -> None:
    jobs.update(project_id, stage, state="running", detail="", current=0, total=0,
                started_at=time.time(), finished_at=None)
    try:
        work()
        jobs.update(project_id, stage, state="done", finished_at=time.time())
    except Exception as exc:  # surface any failure to the UI
        logger.exception("V2 %s stage failed for %s", stage, project_id)
        jobs.update(project_id, stage, state="error", detail=str(exc), finished_at=time.time())


# --- helpers ----------------------------------------------------------------


def _project_dir(project_id: str) -> Path:
    directory = projects_root() / project_id
    if not (directory / "project.yaml").exists():
        raise HTTPException(404, f"V2 project '{project_id}' not found")
    return directory


def _load_project(project_id: str) -> tuple[pipeline.V2Project, Path]:
    directory = _project_dir(project_id)
    try:
        return pipeline.V2Project.load(directory / "project.yaml"), directory
    except Exception as exc:
        raise HTTPException(500, f"Invalid project.yaml: {exc}")


def _load_plan(directory: Path) -> PlanDocument:
    path = directory / pipeline.PLAN_FILENAME
    if not path.exists():
        raise HTTPException(409, "No plan yet; run the plan stage first")
    return PlanDocument.load(path)


def _is_v2_project(directory: Path) -> bool:
    """V2 project.yaml files have a 'region' + 'pois'/'output' shape;
    V1 files have 'tiles'/'style' sections instead."""
    if not (directory / "project.yaml").exists():
        return False
    try:
        import yaml

        data = yaml.safe_load((directory / "project.yaml").read_text())
        return isinstance(data, dict) and "region" in data and "tiles" not in data
    except Exception:
        return False


def _summary(project_id: str) -> dict:
    project, directory = _load_project(project_id)
    jobs.load_persisted(project_id)
    plan_path = directory / pipeline.PLAN_FILENAME
    has_plan = plan_path.exists()
    # The plan bakes in the project config (POIs, region, camera); a config
    # saved after the plan was built means the plan no longer reflects it.
    plan_stale = has_plan and (directory / "project.yaml").stat().st_mtime > plan_path.stat().st_mtime
    return {
        "id": project_id,
        "name": project.name,
        "region": project.region.model_dump(),
        "poi_count": len(project.pois),
        "has_plan": has_plan,
        "plan_stale": plan_stale,
        "has_poster": (directory / pipeline.POSTER_FILENAME).exists(),
        "status": {k: v.model_dump() for k, v in jobs.status(project_id).items()},
    }


# --- project CRUD -----------------------------------------------------------


@router.get("")
def list_projects() -> list[dict]:
    root = projects_root()
    if not root.is_dir():
        return []
    return [
        _summary(d.name)
        for d in sorted(root.iterdir())
        if d.is_dir() and _is_v2_project(d)
    ]


@router.post("", status_code=201)
def create_project(project: pipeline.V2Project) -> dict:
    project_id = slugify(project.name)
    directory = projects_root() / project_id
    if (directory / "project.yaml").exists():
        raise HTTPException(409, f"Project '{project_id}' already exists")
    directory.mkdir(parents=True, exist_ok=True)
    project.save(directory / "project.yaml")
    return _summary(project_id)


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    project, _ = _load_project(project_id)
    summary = _summary(project_id)
    summary["config"] = project.model_dump(mode="json")
    return summary


@router.put("/{project_id}")
def update_project(project_id: str, project: pipeline.V2Project) -> dict:
    _, directory = _load_project(project_id)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is running; try again when it finishes")
    project.save(directory / "project.yaml")
    return _summary(project_id)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    import shutil

    directory = _project_dir(project_id)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is running; try again when it finishes")
    shutil.rmtree(directory)
    return Response(status_code=204)


# --- pipeline stages --------------------------------------------------------


class PlanRequest(BaseModel):
    pass  # reserved for future options (orientation, distortion overrides)


@router.post("/{project_id}/plan", status_code=202)
def start_plan(project_id: str, background: BackgroundTasks) -> dict:
    project, directory = _load_project(project_id)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is already running")

    def work() -> None:
        source = pipeline.fetch_source(project, cache_dir=directory / "cache")
        document = pipeline.build_plan(project, source)
        pipeline.write_plan(document, directory)

    background.add_task(_run_stage, project_id, "plan", work)
    return {"stage": "plan", "state": "started"}


@router.get("/{project_id}/plan")
def get_plan(project_id: str) -> Response:
    _, directory = _load_project(project_id)
    path = directory / pipeline.PLAN_FILENAME
    if not path.exists():
        raise HTTPException(404, "No plan yet")
    return Response(path.read_text(), media_type="application/json")


@router.get("/{project_id}/preview.svg")
def get_preview(project_id: str) -> FileResponse:
    _, directory = _load_project(project_id)
    path = directory / pipeline.PREVIEW_FILENAME
    if not path.exists():
        raise HTTPException(404, "No preview yet; run the plan stage first")
    return FileResponse(path, media_type="image/svg+xml")


class AssetsRequest(BaseModel):
    stub: bool = False
    force: bool = False
    only_ids: Optional[list[str]] = None


@router.post("/{project_id}/assets", status_code=202)
def start_assets(project_id: str, req: AssetsRequest, background: BackgroundTasks, request: Request) -> dict:
    _, directory = _load_project(project_id)
    document = _load_plan(directory)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is already running")

    api_key = request.headers.get("X-Google-API-Key")

    def work() -> None:
        if req.stub:
            from ...v2.assets.stub import StubAssetGenerator

            generator = StubAssetGenerator()
        else:
            from ...v2.assets.gemini_client import GeminiAssetGenerator

            generator = GeminiAssetGenerator(api_key=api_key)
        studio = pipeline.AssetStudio(generator, directory / pipeline.ASSETS_DIRNAME)
        studio.generate_all(
            document,
            force=req.force,
            only_ids=set(req.only_ids) if req.only_ids else None,
            progress=lambda asset_id, i, total: jobs.update(
                project_id, "assets", current=i, total=total, detail=asset_id
            ),
        )

    background.add_task(_run_stage, project_id, "assets", work)
    return {"stage": "assets", "state": "started"}


@router.get("/{project_id}/assets")
def list_assets(project_id: str) -> list[dict]:
    _, directory = _load_project(project_id)
    document = _load_plan(directory)
    studio_dir = directory / pipeline.ASSETS_DIRNAME
    from ...v2.assets.studio import AssetStudio
    from ...v2.assets.stub import StubAssetGenerator

    studio = AssetStudio(StubAssetGenerator(), studio_dir)  # generator unused for cache checks
    return [
        {
            "id": spec.id,
            "kind": spec.kind.value,
            "subject": spec.subject,
            "cached": studio.is_cached(spec),
            "url": f"/api/v2/projects/{project_id}/assets/{spec.id}",
        }
        for spec in document.manifest
    ]


@router.get("/{project_id}/assets/{asset_id}")
def get_asset(project_id: str, asset_id: str) -> FileResponse:
    _, directory = _load_project(project_id)
    path = directory / pipeline.ASSETS_DIRNAME / f"{slugify(asset_id)}.png"
    if not path.exists():
        raise HTTPException(404, f"Asset '{asset_id}' not generated yet")
    return FileResponse(path, media_type="image/png")


class ComposeRequest(BaseModel):
    scale: float = Field(1.0, gt=0.0, le=1.0)
    harmonize: bool = False


@router.post("/{project_id}/compose", status_code=202)
def start_compose(project_id: str, req: ComposeRequest, background: BackgroundTasks, request: Request) -> dict:
    _, directory = _load_project(project_id)
    document = _load_plan(directory)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is already running")

    api_key = request.headers.get("X-Google-API-Key")

    def work() -> None:
        mood_pass = None
        if req.harmonize:
            from ...v2.compose.harmonize import GeminiMoodPass

            mood_pass = GeminiMoodPass(api_key=api_key)
        pipeline.compose_poster(document, directory, scale=req.scale, mood_pass=mood_pass)

    background.add_task(_run_stage, project_id, "compose", work)
    return {"stage": "compose", "state": "started"}


@router.get("/{project_id}/poster")
def get_poster(project_id: str) -> FileResponse:
    _, directory = _load_project(project_id)
    path = directory / pipeline.POSTER_FILENAME
    if not path.exists():
        raise HTTPException(404, "No poster yet; run the compose stage first")
    return FileResponse(path, media_type="image/png")


@router.get("/{project_id}/status")
def get_status(project_id: str) -> dict:
    _project_dir(project_id)
    jobs.load_persisted(project_id)
    return {k: v.model_dump() for k, v in jobs.status(project_id).items()}
