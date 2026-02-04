"""Project management endpoints."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...config import get_config
from ...models.project import Project
from ..schemas import (
    ErrorResponse,
    ProjectCreate,
    ProjectDetail,
    ProjectSummary,
    ProjectUpdate,
    SuccessResponse,
)

router = APIRouter()


def get_projects_dir() -> Path:
    """Get the projects directory."""
    return Path.cwd() / "projects"


def get_project_path(name: str) -> Path:
    """Get the path to a project's YAML file."""
    return get_projects_dir() / name / "project.yaml"


def load_project(name: str) -> Project:
    """Load a project by name.

    Raises:
        HTTPException: If project not found
    """
    project_path = get_project_path(name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    return Project.from_yaml(project_path)


def get_project_cache_dir(project_name: str) -> Path:
    """Get the cache directory for a project."""
    config = get_config()
    return config.cache_dir / "projects" / project_name


def check_has_generated_tiles(project_name: str) -> bool:
    """Check if a project has any generated tiles."""
    cache_dir = get_project_cache_dir(project_name) / "generation" / "generated"
    if not cache_dir.exists():
        return False
    return any(cache_dir.glob("tile_*.png"))


def get_project_last_modified(project_name: str) -> Optional[datetime]:
    """Get the last modified time of a project."""
    project_path = get_project_path(project_name)
    if project_path.exists():
        stat = project_path.stat()
        return datetime.fromtimestamp(stat.st_mtime)
    return None


@router.get("", response_model=list[ProjectSummary])
async def list_projects():
    """List all available projects."""
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []

    projects = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_yaml = project_dir / "project.yaml"
        if not project_yaml.exists():
            continue

        try:
            project = Project.from_yaml(project_yaml)
            cols, rows = project.tiles.calculate_grid(
                project.output.width, project.output.height
            )

            projects.append(ProjectSummary(
                name=project.name,
                region=project.region,
                area_km2=project.region.calculate_area_km2(),
                tile_count=cols * rows,
                landmark_count=len(project.landmarks),
                has_generated_tiles=check_has_generated_tiles(project.name),
                last_modified=get_project_last_modified(project.name),
            ))
        except Exception as e:
            # Skip invalid projects
            continue

    return sorted(projects, key=lambda p: p.name)


@router.post("", response_model=ProjectDetail, status_code=201)
async def create_project(request: ProjectCreate):
    """Create a new project."""
    projects_dir = get_projects_dir()
    project_dir = projects_dir / request.name

    if project_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Project '{request.name}' already exists"
        )

    # Create project with provided settings or defaults
    project = Project(
        name=request.name,
        region=request.region,
        output=request.output or None,
        style=request.style or None,
        tiles=request.tiles or None,
    )
    project.project_dir = project_dir

    # Create directories
    project_dir.mkdir(parents=True, exist_ok=True)
    project.ensure_directories()

    # Save project
    project.to_yaml(project_dir / "project.yaml")

    return ProjectDetail.from_project(project)


@router.get("/{name}", response_model=ProjectDetail)
async def get_project(name: str):
    """Get project details."""
    project = load_project(name)
    return ProjectDetail.from_project(project)


@router.put("/{name}", response_model=ProjectDetail)
async def update_project(name: str, request: ProjectUpdate):
    """Update project settings."""
    project = load_project(name)

    # Update fields that were provided
    if request.output is not None:
        project.output = request.output
    if request.style is not None:
        project.style = request.style
    if request.tiles is not None:
        project.tiles = request.tiles

    # Save updated project
    project.to_yaml(get_project_path(name))

    return ProjectDetail.from_project(project)


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_project(
    name: str,
    delete_cache: bool = Query(default=True, description="Also delete cached data"),
):
    """Delete a project.

    Args:
        name: Project name
        delete_cache: If True, also delete cached generation data
    """
    project_dir = get_projects_dir() / name
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    import shutil

    # Delete project directory
    shutil.rmtree(project_dir)

    # Optionally delete cache
    if delete_cache:
        cache_dir = get_project_cache_dir(name)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    return SuccessResponse(message=f"Project '{name}' deleted successfully")


@router.get("/{name}/cost-estimate")
async def estimate_project_cost(name: str):
    """Estimate the cost to generate a project."""
    project = load_project(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    tile_count = cols * rows
    landmark_count = len(project.landmarks)

    # Rough cost estimates based on Gemini API pricing
    # These are approximations and may vary
    cost_per_tile = 0.02  # ~$0.02 per tile generation
    cost_per_landmark = 0.02  # ~$0.02 per landmark illustration
    cost_per_seam = 0.01  # ~$0.01 per seam repair

    # Calculate number of seams
    horizontal_seams = rows * (cols - 1)
    vertical_seams = cols * (rows - 1)
    total_seams = horizontal_seams + vertical_seams

    tile_cost = tile_count * cost_per_tile
    landmark_cost = landmark_count * cost_per_landmark
    seam_cost = total_seams * cost_per_seam
    total_cost = tile_cost + landmark_cost + seam_cost

    return {
        "project_name": name,
        "breakdown": {
            "tiles": {
                "count": tile_count,
                "unit_cost": cost_per_tile,
                "total": tile_cost,
            },
            "landmarks": {
                "count": landmark_count,
                "unit_cost": cost_per_landmark,
                "total": landmark_cost,
            },
            "seams": {
                "count": total_seams,
                "unit_cost": cost_per_seam,
                "total": seam_cost,
            },
        },
        "total_estimated_cost_usd": total_cost,
    }
