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
from ...v2.compose_spec import COMPOSITION_FILENAME, CompositionSpec
from ...v2.assets.manifest import slugify
from ...v2.types import PlanDocument

logger = logging.getLogger(__name__)

router = APIRouter()

STAGES = ("plan", "assets", "compose", "repaint")


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


# Parsed-plan cache keyed by file mtime: a large plan parsed into pydantic
# models is hundreds of MB, and the UI polls endpoints that need it -- one
# parse per plan version, never one per request.
_plan_cache: dict[str, tuple[float, PlanDocument]] = {}
_plan_cache_lock = threading.Lock()


def _load_plan(directory: Path) -> PlanDocument:
    path = directory / pipeline.PLAN_FILENAME
    if not path.exists():
        raise HTTPException(409, "No plan yet; run the plan stage first")
    key = str(path)
    mtime = path.stat().st_mtime
    with _plan_cache_lock:
        hit = _plan_cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    document = PlanDocument.load(path)
    with _plan_cache_lock:
        _plan_cache[key] = (mtime, document)
        # Bound the cache: keep only the most recent few plans.
        while len(_plan_cache) > 4:
            _plan_cache.pop(next(iter(_plan_cache)))
    return document


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


class RetitleRequest(BaseModel):
    title: str


@router.patch("/{project_id}/title")
def retitle_project(project_id: str, req: RetitleRequest) -> dict:
    """Change the poster title without re-planning: patches project.yaml and
    the existing plan's TITLE label in place. Re-run compose to re-render."""
    _, directory = _load_project(project_id)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is running; try again when it finishes")
    try:
        plan_patched = pipeline.retitle_project(directory, req.title)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"title": req.title.strip(), "plan_patched": plan_patched}


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
        spec = CompositionSpec.load_or_default(directory)
        document = pipeline.build_plan(project, source, spec=spec)
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


@router.get("/{project_id}/composition")
def get_composition(project_id: str) -> Response:
    """The editable layout document; an all-auto default if none is saved yet."""
    _, directory = _load_project(project_id)
    spec = CompositionSpec.load_or_default(directory)
    return Response(spec.model_dump_json(indent=1), media_type="application/json")


@router.put("/{project_id}/composition")
def put_composition(project_id: str, spec: CompositionSpec) -> dict:
    """Persist an edited spec. Re-run the plan stage to apply it."""
    _, directory = _load_project(project_id)
    spec.save(directory / COMPOSITION_FILENAME)
    return {"saved": True, "version": spec.version}


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
    out = []
    for spec in document.manifest:
        meta = studio.read_meta(spec.id)
        out.append(
            {
                "id": spec.id,
                "kind": spec.kind.value,
                "subject": spec.subject,
                "cached": studio.is_cached(spec),
                "url": f"/api/v2/projects/{project_id}/assets/{spec.id}",
                "palette_score": meta.get("palette_score"),
                "palette_outlier": meta.get("palette_outlier", False),
                "flagged": meta.get("flagged", False),
            }
        )
    return out


class FlagRequest(BaseModel):
    flagged: bool = True


@router.post("/{project_id}/assets/{asset_id}/flag")
def flag_asset(project_id: str, asset_id: str, req: FlagRequest) -> dict:
    """Mark an asset for human-reviewed regeneration (or clear the mark).
    Regenerate flagged assets via the assets stage with only_ids + force."""
    _, directory = _load_project(project_id)
    from ...v2.assets.studio import AssetStudio
    from ...v2.assets.stub import StubAssetGenerator

    studio = AssetStudio(StubAssetGenerator(), directory / pipeline.ASSETS_DIRNAME)
    try:
        meta = studio.set_flagged(slugify(asset_id), req.flagged)
    except FileNotFoundError:
        raise HTTPException(404, f"Asset '{asset_id}' not generated yet")
    return {"id": asset_id, "flagged": meta.get("flagged", False)}


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


class RepaintRequest(BaseModel):
    mode: str = Field("single", pattern="^(single|tiled)$")
    scale: float = Field(1.0, gt=0.0, le=1.0)  # final poster scale
    strength: float = Field(1.0, ge=0.0, le=1.0)  # single mode: texture blend dial
    repaint_scale: float = Field(0.5, gt=0.0, le=1.0)  # tiled mode: AI resolution
    max_calls: Optional[int] = Field(None, ge=1)  # tiled mode: hard cost cap
    dry_run: bool = False  # plan only, spend nothing
    stub: bool = False  # identity painter (free, exercises the machinery)


@router.post("/{project_id}/repaint", status_code=202)
def start_repaint(project_id: str, req: RepaintRequest, background: BackgroundTasks, request: Request) -> dict:
    """AI repaint of the poster base. Default mode 'single': one whole-base
    texture call + frequency-split blend (no seams possible). Mode 'tiled'
    is the experimental window-by-window engine (see mapgen.v2.repaint)."""
    _, directory = _load_project(project_id)
    document = _load_plan(directory)
    if jobs.is_running(project_id):
        raise HTTPException(409, "A stage is already running")

    if req.dry_run:
        if req.mode == "single":
            return {"stage": "repaint", "state": "dry_run", "mode": "single",
                    "calls_planned": 1,
                    "estimated_cost_usd": pipeline.REPAINT_COST_PER_CALL}
        return {"stage": "repaint", "state": "dry_run", "mode": "tiled",
                **pipeline.plan_repaint(document, directory, repaint_scale=req.repaint_scale)}

    api_key = request.headers.get("X-Google-API-Key")

    def work() -> None:
        progress = lambda i, total, detail: jobs.update(  # noqa: E731
            project_id, "repaint", current=i, total=total, detail=detail
        )
        if req.mode == "single":
            if req.stub:
                from ...v2.repaint import IdentityTexturePass

                painter = IdentityTexturePass()
            else:
                from ...v2.repaint import GeminiTexturePass

                painter = GeminiTexturePass(api_key=api_key)
            pipeline.texture_poster(
                document, directory, painter,
                scale=req.scale, strength=req.strength, progress=progress,
            )
            return
        if req.stub:
            from ...v2.repaint import IdentityPainter

            painter = IdentityPainter()
        else:
            from ...v2.repaint import GeminiPainter

            painter = GeminiPainter(api_key=api_key)
        _, result = pipeline.repaint_poster(
            document, directory, painter,
            scale=req.scale, repaint_scale=req.repaint_scale,
            max_calls=req.max_calls, progress=progress,
        )
        if not result.completed:
            jobs.update(
                project_id,
                "repaint",
                detail=f"budget reached after {result.calls_made} calls; run again to resume",
            )

    background.add_task(_run_stage, project_id, "repaint", work)
    return {"stage": "repaint", "state": "started", "mode": req.mode}


@router.get("/{project_id}/repaint/quadrants")
def repaint_quadrants(project_id: str) -> dict:
    """Quadrant grid state for the review overlay (empty before first run)."""
    _, directory = _load_project(project_id)
    repaint_dir = directory / pipeline.REPAINT_DIRNAME
    if not (repaint_dir / "repaint.db").exists():
        return {"grid": None, "quadrants": []}
    import math

    from ...v2.repaint import RepaintStore

    document = _load_plan(directory)
    meta = json.loads((repaint_dir / "meta.json").read_text())
    repaint_scale = meta["repaint_scale"]
    cols = math.ceil(round(document.canvas.width_px * repaint_scale) / 512)
    rows = math.ceil(round(document.canvas.height_px * repaint_scale) / 512)
    store = RepaintStore(repaint_dir)
    try:
        cells = [
            {"x": x, "y": y, "status": status.value}
            for (x, y), status in sorted(store.status_map().items())
        ]
        calls = store.call_count()
    finally:
        store.close()
    return {
        "grid": {"cols": cols, "rows": rows, "repaint_scale": repaint_scale},
        "quadrants": cells,
        "calls_made": calls,
    }


@router.post("/{project_id}/repaint/quadrants/{x}/{y}/flag")
def flag_quadrant(project_id: str, x: int, y: int, req: FlagRequest) -> dict:
    """Flag a painted quadrant for redo (next repaint run repaints it with
    centered context from its painted neighbors), or restore it."""
    _, directory = _load_project(project_id)
    repaint_dir = directory / pipeline.REPAINT_DIRNAME
    if not (repaint_dir / "repaint.db").exists():
        raise HTTPException(409, "No repaint run yet")
    from ...v2.repaint import QuadStatus, RepaintStore

    store = RepaintStore(repaint_dir)
    try:
        current = store.status_map().get((x, y))
        if current not in (QuadStatus.GENERATED, QuadStatus.FLAGGED):
            raise HTTPException(409, f"Quadrant ({x},{y}) is not painted (status: {current})")
        status = QuadStatus.FLAGGED if req.flagged else QuadStatus.GENERATED
        store.set_status((x, y), status)
    finally:
        store.close()
    return {"x": x, "y": y, "status": status.value}


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
