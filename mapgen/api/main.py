"""MapGen API - FastAPI application."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import get_config
from .routers import dzi, landmarks, projects, seams, tiles
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
app.include_router(websocket_router, prefix="/api", tags=["websocket"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/config")
async def get_api_config():
    """Get API configuration (non-sensitive)."""
    config = get_config()
    return {
        "cache_dir": str(config.cache_dir),
        "output_dir": str(config.output_dir),
        "default_tile_size": config.default_tile_size,
        "default_overlap": config.default_overlap,
        "default_dpi": config.default_dpi,
        "gemini_model": config.gemini_model,
        "has_google_api_key": bool(config.google_api_key),
        "has_mapbox_token": bool(config.mapbox_access_token),
    }


def mount_static_files(app: FastAPI, projects_dir: Optional[Path] = None):
    """Mount static file directories for serving images.

    Args:
        app: FastAPI application instance
        projects_dir: Directory containing projects (defaults to ./projects)
    """
    if projects_dir is None:
        projects_dir = Path.cwd() / "projects"

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
