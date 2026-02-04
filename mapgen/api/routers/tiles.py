"""Tile management and generation endpoints."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ...config import get_config
from ...models.project import BoundingBox, Project
from ...services.generation_service import GenerationProgress, GenerationService, TileSpec as ServiceTileSpec
from ..schemas import (
    GenerationProgress as ProgressSchema,
    GenerationStartRequest,
    GenerationStartResponse,
    GenerationStatus,
    SuccessResponse,
    TileGridResponse,
    TileRegenerateRequest,
    TileSpec,
    TileStatus,
)
from ..tasks import task_manager
from .projects import get_project_cache_dir, load_project

router = APIRouter()


def get_tile_cache_dirs(project_name: str) -> tuple[Path, Path]:
    """Get the cache directories for generated and reference tiles."""
    cache_dir = get_project_cache_dir(project_name)
    generated_dir = cache_dir / "generation" / "generated"
    reference_dir = cache_dir / "generation" / "references"
    return generated_dir, reference_dir


def get_tile_status(project_name: str, col: int, row: int) -> tuple[TileStatus, bool, bool]:
    """Get the status of a tile.

    Returns:
        Tuple of (status, has_reference, has_generated)
    """
    generated_dir, reference_dir = get_tile_cache_dirs(project_name)

    has_reference = (reference_dir / f"tile_{col}_{row}_reference.png").exists()
    has_generated = (generated_dir / f"tile_{col}_{row}.png").exists()

    if has_generated:
        status = TileStatus.COMPLETED
    elif has_reference:
        status = TileStatus.PENDING
    else:
        status = TileStatus.PENDING

    return status, has_reference, has_generated


def service_tile_to_api_tile(spec: ServiceTileSpec, project_name: str) -> TileSpec:
    """Convert a service TileSpec to an API TileSpec."""
    status, has_reference, has_generated = get_tile_status(project_name, spec.col, spec.row)

    return TileSpec(
        col=spec.col,
        row=spec.row,
        x_offset=spec.x_offset,
        y_offset=spec.y_offset,
        bbox=spec.bbox,
        position_desc=spec.position_desc,
        status=status,
        has_reference=has_reference,
        has_generated=has_generated,
    )


@router.get("/{name}/tiles", response_model=TileGridResponse)
async def get_tile_grid(name: str):
    """Get the tile grid for a project."""
    project = load_project(name)

    # Create generation service to calculate tile specs
    service = GenerationService(
        project=project,
        gemini_service=None,  # Don't need Gemini for just calculating specs
    )

    specs = service.calculate_tile_specs()
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)

    tiles = [service_tile_to_api_tile(spec, name) for spec in specs]

    return TileGridResponse(
        project_name=name,
        cols=cols,
        rows=rows,
        tile_size=project.tiles.size,
        overlap=project.tiles.overlap,
        effective_size=project.tiles.effective_size,
        tiles=tiles,
    )


@router.get("/{name}/tiles/{col}/{row}")
async def get_tile_info(name: str, col: int, row: int):
    """Get information about a specific tile."""
    project = load_project(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    if col < 0 or col >= cols or row < 0 or row >= rows:
        raise HTTPException(status_code=404, detail=f"Tile ({col}, {row}) not found")

    service = GenerationService(project=project, gemini_service=None)
    specs = service.calculate_tile_specs()

    # Find the matching tile
    for spec in specs:
        if spec.col == col and spec.row == row:
            return service_tile_to_api_tile(spec, name)

    raise HTTPException(status_code=404, detail=f"Tile ({col}, {row}) not found")


@router.get("/{name}/tiles/{col}/{row}/reference")
async def get_tile_reference(
    name: str,
    col: int,
    row: int,
    size: Optional[int] = Query(None, description="Resize to this size (thumbnail)"),
):
    """Get the reference image for a tile.

    Args:
        name: Project name
        col: Tile column
        row: Tile row
        size: Optional size to resize to (for thumbnails)
    """
    _, reference_dir = get_tile_cache_dirs(name)
    tile_path = reference_dir / f"tile_{col}_{row}_reference.png"

    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="Reference image not found")

    if size:
        # Generate and serve thumbnail
        from PIL import Image
        import io
        from fastapi.responses import StreamingResponse

        img = Image.open(tile_path)
        img.thumbnail((size, size))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    return FileResponse(tile_path, media_type="image/png")


@router.get("/{name}/tiles/{col}/{row}/generated")
async def get_tile_generated(
    name: str,
    col: int,
    row: int,
    size: Optional[int] = Query(None, description="Resize to this size (thumbnail)"),
):
    """Get the generated image for a tile.

    Args:
        name: Project name
        col: Tile column
        row: Tile row
        size: Optional size to resize to (for thumbnails)
    """
    generated_dir, _ = get_tile_cache_dirs(name)
    tile_path = generated_dir / f"tile_{col}_{row}.png"

    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="Generated image not found")

    if size:
        from PIL import Image
        import io
        from fastapi.responses import StreamingResponse

        img = Image.open(tile_path)
        img.thumbnail((size, size))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    return FileResponse(tile_path, media_type="image/png")


@router.get("/{name}/tiles/{col}/{row}/thumbnail")
async def get_tile_thumbnail(name: str, col: int, row: int):
    """Get a 256px thumbnail of the generated tile."""
    return await get_tile_generated(name, col, row, size=256)


@router.get("/{name}/tiles/{col}/{row}/preview")
async def get_tile_preview(name: str, col: int, row: int):
    """Get a 512px preview of the generated tile."""
    return await get_tile_generated(name, col, row, size=512)


@router.post("/{name}/tiles/{col}/{row}/regenerate", response_model=SuccessResponse)
async def regenerate_tile(name: str, col: int, row: int, request: TileRegenerateRequest):
    """Regenerate a specific tile.

    This runs synchronously and may take some time.
    """
    project = load_project(name)
    config = get_config()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    if col < 0 or col >= cols or row < 0 or row >= rows:
        raise HTTPException(status_code=404, detail=f"Tile ({col}, {row}) not found")

    # Check if generation is already running
    active_task = await task_manager.get_active_task(name, "generation")
    if active_task:
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress for this project"
        )

    cache_dir = get_project_cache_dir(name)
    generated_dir, reference_dir = get_tile_cache_dirs(name)
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Delete existing tile if force regenerate
    if request.force:
        tile_path = generated_dir / f"tile_{col}_{row}.png"
        if tile_path.exists():
            tile_path.unlink()

    # Generate the tile
    from ...services.gemini_service import GeminiService

    service = GenerationService(
        project=project,
        cache_dir=cache_dir,
    )

    specs = service.calculate_tile_specs()
    target_spec = None
    for spec in specs:
        if spec.col == col and spec.row == row:
            target_spec = spec
            break

    if not target_spec:
        raise HTTPException(status_code=404, detail=f"Tile ({col}, {row}) not found")

    # Generate reference if needed
    if not (reference_dir / f"tile_{col}_{row}_reference.png").exists():
        await asyncio.to_thread(service.generate_tile_reference, target_spec)

    # Generate the tile
    result = await asyncio.to_thread(service.generate_tile, target_spec)

    if result.error:
        raise HTTPException(status_code=500, detail=f"Tile generation failed: {result.error}")

    return SuccessResponse(message=f"Tile ({col}, {row}) regenerated successfully")


@router.post("/{name}/generate", response_model=GenerationStartResponse)
async def start_generation(name: str, request: GenerationStartRequest):
    """Start generating all tiles for a project.

    Returns immediately with a task ID. Use WebSocket to track progress.
    """
    project = load_project(name)

    # Check if generation is already running
    active_task = await task_manager.get_active_task(name, "generation")
    if active_task:
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress for this project"
        )

    cache_dir = get_project_cache_dir(name)
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    total_tiles = cols * rows

    async def run_generation():
        """Run the generation in background."""
        service = GenerationService(project=project, cache_dir=cache_dir)

        # Generate tiles with progress callback
        def progress_callback(progress: GenerationProgress):
            asyncio.create_task(
                task_manager.update_progress(task_info.task_id, progress)
            )

        results = await asyncio.to_thread(
            service.generate_all_tiles,
            progress_callback=progress_callback,
        )

        return results

    task_info = await task_manager.create_task(
        project_name=name,
        task_type="generation",
        coro=run_generation(),
    )

    return GenerationStartResponse(
        task_id=task_info.task_id,
        status=GenerationStatus.RUNNING,
        total_tiles=total_tiles,
        websocket_url=f"/api/ws/generation/{task_info.task_id}",
    )


@router.get("/{name}/generate/status", response_model=ProgressSchema)
async def get_generation_status(name: str):
    """Get the current generation status for a project."""
    task = await task_manager.get_active_task(name, "generation")

    if task is None:
        # Check if there's a completed task
        tasks = await task_manager.get_project_tasks(name)
        gen_tasks = [t for t in tasks if t.task_type == "generation"]
        if gen_tasks:
            latest = max(gen_tasks, key=lambda t: t.created_at)
            project = load_project(name)
            cols, rows = project.tiles.calculate_grid(
                project.output.width, project.output.height
            )
            return ProgressSchema(
                status=latest.status,
                total_tiles=cols * rows,
                completed_tiles=cols * rows if latest.status == GenerationStatus.COMPLETED else 0,
                failed_tiles=0,
                error=latest.error,
            )

        # No generation has been run
        project = load_project(name)
        cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
        return ProgressSchema(
            status=GenerationStatus.IDLE,
            total_tiles=cols * rows,
            completed_tiles=0,
            failed_tiles=0,
        )

    if task.progress:
        return ProgressSchema(
            status=task.status,
            total_tiles=task.progress.total_tiles,
            completed_tiles=task.progress.completed_tiles,
            failed_tiles=task.progress.failed_tiles,
            current_tile=(task.progress.current_tile.col, task.progress.current_tile.row)
            if task.progress.current_tile
            else None,
            elapsed_seconds=task.progress.elapsed_time,
            estimated_remaining_seconds=task.progress.estimated_remaining,
            error=task.error,
        )

    project = load_project(name)
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    return ProgressSchema(
        status=task.status,
        total_tiles=cols * rows,
        completed_tiles=0,
        failed_tiles=0,
        error=task.error,
    )


@router.post("/{name}/generate/cancel", response_model=SuccessResponse)
async def cancel_generation(name: str):
    """Cancel an ongoing generation."""
    task = await task_manager.get_active_task(name, "generation")

    if task is None:
        raise HTTPException(status_code=404, detail="No active generation found")

    success = await task_manager.cancel_task(task.task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel generation")

    return SuccessResponse(message="Generation cancelled")
