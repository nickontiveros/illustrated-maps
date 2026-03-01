"""MapGen API - FastAPI application."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_config
from .routers import dzi, landmarks, postprocess, projects, seams, tiles
from .tasks import task_manager
from .websocket import router as websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config = get_config()
    config.ensure_directories()

    yield

    # Shutdown
    await task_manager.cancel_all()


app = FastAPI(
    title="MapGen API",
    description="API for the Illustrated Map Generator",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware - allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(tiles.router, prefix="/api/projects", tags=["tiles"])
app.include_router(seams.router, prefix="/api/projects", tags=["seams"])
app.include_router(landmarks.router, prefix="/api/projects", tags=["landmarks"])
app.include_router(dzi.router, prefix="/api/projects", tags=["dzi"])
app.include_router(postprocess.router, prefix="/api/projects", tags=["postprocess"])
app.include_router(websocket_router, prefix="/api", tags=["websocket"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/debug/volume")
async def debug_volume():
    """Temporary: inspect volume mount status."""
    import os as _os
    import subprocess
    results: dict = {}

    # Check what's mounted
    try:
        mounts = subprocess.check_output(["mount"], text=True)
        results["mounts"] = [l for l in mounts.splitlines() if "app" in l.lower() or "data" in l.lower()]
    except Exception as e:
        results["mounts_error"] = str(e)

    # Check df for volume info
    try:
        df = subprocess.check_output(["df", "-h"], text=True)
        results["df"] = [l for l in df.splitlines() if "app" in l.lower() or "data" in l.lower() or "Filesystem" in l]
    except Exception as e:
        results["df_error"] = str(e)

    # Walk /app/data up to depth 4
    base = Path("/app/data")
    entries = []
    for root, dirs, files in _os.walk(base):
        depth = root.replace(str(base), "").count(_os.sep)
        if depth > 3:
            dirs.clear()
            continue
        for d in dirs:
            entries.append({"path": str(Path(root) / d), "type": "dir"})
        for f in files:
            fp = Path(root) / f
            try:
                size = fp.stat().st_size
            except Exception:
                size = -1
            entries.append({"path": str(fp), "type": "file", "size": size})
    results["volume_contents"] = entries

    # Env vars
    results["env"] = {
        k: v for k, v in _os.environ.items()
        if "MAPGEN" in k or "DATA" in k or "PROJECT" in k
    }

    return results


@app.get("/api/config")
async def get_api_config(request: Request):
    """Get API configuration (non-sensitive)."""
    config = get_config()
    google_key = request.headers.get("X-Google-API-Key") or config.google_api_key
    mapbox_token = request.headers.get("X-Mapbox-Access-Token") or config.mapbox_access_token
    return {
        "cache_dir": str(config.cache_dir),
        "output_dir": str(config.output_dir),
        "default_tile_size": config.default_tile_size,
        "default_overlap": config.default_overlap,
        "default_dpi": config.default_dpi,
        "gemini_model": config.gemini_model,
        "has_google_api_key": bool(google_key),
        "has_mapbox_token": bool(mapbox_token),
        "google_api_key_source": "client" if request.headers.get("X-Google-API-Key") else ("server" if config.google_api_key else "missing"),
        "mapbox_token_source": "client" if request.headers.get("X-Mapbox-Access-Token") else ("server" if config.mapbox_access_token else "missing"),
    }


@app.get("/api/tasks/active")
async def get_active_tasks():
    """Get all active (running) tasks across all projects."""
    active = []
    for task in task_manager._tasks.values():
        from .schemas import GenerationStatus
        if task.status == GenerationStatus.RUNNING:
            entry = {
                "task_id": task.task_id,
                "project_name": task.project_name,
                "task_type": task.task_type,
                "status": task.status.value,
                "created_at": task.created_at.isoformat() if task.created_at else None,
            }
            if task.progress:
                entry["progress"] = {
                    "total_tiles": task.progress.total_tiles,
                    "completed_tiles": task.progress.completed_tiles,
                    "failed_tiles": task.progress.failed_tiles,
                    "elapsed_seconds": task.progress.elapsed_time,
                    "estimated_remaining_seconds": task.progress.estimated_remaining
                    if task.progress.completed_tiles > 0 else None,
                }
            active.append(entry)
    return active


def mount_static_files(app: FastAPI, projects_dir: Optional[Path] = None):
    """Mount static file directories for serving images.

    Args:
        app: FastAPI application instance
        projects_dir: Directory containing projects (defaults to ./projects)
    """
    if projects_dir is None:
        projects_dir = Path(os.environ.get("MAPGEN_PROJECTS_DIR", str(Path.cwd() / "projects")))

    if projects_dir.exists():
        app.mount(
            "/static/projects",
            StaticFiles(directory=str(projects_dir)),
            name="projects",
        )

    # Mount cache directory for generated tiles
    config = get_config()
    if config.cache_dir.exists():
        app.mount(
            "/static/cache",
            StaticFiles(directory=str(config.cache_dir)),
            name="cache",
        )


# Mount static files if directories exist
mount_static_files(app)

# --- SPA (single-page app) serving for production builds ---
# Serves the built React frontend when /app/static exists (i.e. inside the Docker image).
# This must come AFTER all API routers and /static/* mounts.
_static_dir = Path(os.environ.get("STATIC_DIR", "/app/static"))

if _static_dir.is_dir():
    # Vite puts hashed JS/CSS bundles in assets/
    _assets_dir = _static_dir / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend_assets")

    # SPA catch-all: any non-API, non-static GET request returns index.html
    _index_html = _static_dir / "index.html"

    @app.get("/{full_path:path}")
    async def spa_catch_all(full_path: str):
        """Serve the React SPA for any path not matched by API or static mounts."""
        # Serve actual static files if they exist (e.g. favicon, manifest)
        candidate = _static_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_index_html))
