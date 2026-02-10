"""Tile management and generation endpoints."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse

from ...config import get_config
from ...models.project import BoundingBox, Project
from ...services.generation_service import GenerationProgress, GenerationService, TileSpec as ServiceTileSpec
from ..schemas import (
    AllTileOffsetsResponse,
    GenerationProgress as ProgressSchema,
    GenerationStartRequest,
    GenerationStartResponse,
    GenerationStatus,
    SuccessResponse,
    TileGridResponse,
    TileOffsetRequest,
    TileOffsetResponse,
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
    generated_dir = cache_dir / "generated"
    reference_dir = cache_dir / "references"
    return generated_dir, reference_dir


def load_tile_offsets(project_name: str) -> dict[str, dict[str, int]]:
    """Load tile offsets from JSON file."""
    import json
    cache_dir = get_project_cache_dir(project_name)
    offsets_path = cache_dir / "tile_offsets.json"
    if offsets_path.exists():
        return json.loads(offsets_path.read_text())
    return {}


def save_tile_offsets(project_name: str, offsets: dict[str, dict[str, int]]) -> None:
    """Save tile offsets to JSON file."""
    import json
    cache_dir = get_project_cache_dir(project_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets_path = cache_dir / "tile_offsets.json"
    offsets_path.write_text(json.dumps(offsets, indent=2))


def get_tile_status(project_name: str, col: int, row: int) -> tuple[TileStatus, bool, bool]:
    """Get the status of a tile.

    Returns:
        Tuple of (status, has_reference, has_generated)
    """
    generated_dir, reference_dir = get_tile_cache_dirs(project_name)

    has_reference = (reference_dir / f"tile_{col}_{row}_ref.png").exists()
    has_generated = (generated_dir / f"tile_{col}_{row}.png").exists()

    if has_generated:
        status = TileStatus.COMPLETED
    elif has_reference:
        status = TileStatus.PENDING
    else:
        status = TileStatus.PENDING

    return status, has_reference, has_generated


def service_tile_to_api_tile(
    spec: ServiceTileSpec,
    project_name: str,
    offsets: Optional[dict[str, dict[str, int]]] = None,
) -> TileSpec:
    """Convert a service TileSpec to an API TileSpec."""
    status, has_reference, has_generated = get_tile_status(project_name, spec.col, spec.row)

    tile_offset = (offsets or {}).get(f"{spec.col},{spec.row}", {})

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
        offset_dx=tile_offset.get("dx", 0),
        offset_dy=tile_offset.get("dy", 0),
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

    offsets = load_tile_offsets(name)
    tiles = [service_tile_to_api_tile(spec, name, offsets) for spec in specs]

    return TileGridResponse(
        project_name=name,
        cols=cols,
        rows=rows,
        tile_size=project.tiles.size,
        overlap=project.tiles.overlap,
        effective_size=project.tiles.effective_size,
        tiles=tiles,
    )


@router.get("/{name}/tiles/offsets", response_model=AllTileOffsetsResponse)
async def get_all_tile_offsets(name: str):
    """Get all tile offsets for a project."""
    load_project(name)  # Validate project exists
    offsets = load_tile_offsets(name)
    offset_list = []
    for key, val in offsets.items():
        col_str, row_str = key.split(",")
        offset_list.append(TileOffsetResponse(
            col=int(col_str), row=int(row_str),
            dx=val.get("dx", 0), dy=val.get("dy", 0),
        ))
    return AllTileOffsetsResponse(project_name=name, offsets=offset_list)


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
    offsets = load_tile_offsets(name)
    for spec in specs:
        if spec.col == col and spec.row == row:
            return service_tile_to_api_tile(spec, name, offsets)

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
    tile_path = reference_dir / f"tile_{col}_{row}_ref.png"

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
    if not (reference_dir / f"tile_{col}_{row}_ref.png").exists():
        await asyncio.to_thread(service.generate_tile_reference, target_spec)

    # Load style reference if available
    style_ref = load_style_reference(name)

    # Generate the tile
    result = await asyncio.to_thread(
        service.generate_tile, target_spec, style_reference=style_ref
    )

    if result.error:
        raise HTTPException(status_code=500, detail=f"Tile generation failed: {result.error}")

    return SuccessResponse(message=f"Tile ({col}, {row}) regenerated successfully")


@router.get("/{name}/tiles/{col}/{row}/offset", response_model=TileOffsetResponse)
async def get_tile_offset(name: str, col: int, row: int):
    """Get the position offset for a specific tile."""
    offsets = load_tile_offsets(name)
    key = f"{col},{row}"
    offset = offsets.get(key, {})
    return TileOffsetResponse(
        col=col, row=row,
        dx=offset.get("dx", 0), dy=offset.get("dy", 0),
    )


@router.put("/{name}/tiles/{col}/{row}/offset", response_model=TileOffsetResponse)
async def set_tile_offset(name: str, col: int, row: int, request: TileOffsetRequest):
    """Set the position offset for a specific tile."""
    load_project(name)  # Validate project exists
    offsets = load_tile_offsets(name)
    key = f"{col},{row}"

    if request.dx == 0 and request.dy == 0:
        offsets.pop(key, None)
    else:
        offsets[key] = {"dx": request.dx, "dy": request.dy}

    save_tile_offsets(name, offsets)
    return TileOffsetResponse(col=col, row=row, dx=request.dx, dy=request.dy)


def get_style_reference_path(project_name: str) -> Path:
    """Get the path to the style reference image for a project."""
    cache_dir = get_project_cache_dir(project_name)
    return cache_dir / "style_reference.png"


@router.post("/{name}/style-reference", response_model=SuccessResponse)
async def upload_style_reference(name: str, file: UploadFile = File(...)):
    """Upload a style reference image for consistent tile generation."""
    load_project(name)  # Validate project exists

    import io
    from PIL import Image

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    ref_path = get_style_reference_path(name)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(ref_path, format="PNG")

    return SuccessResponse(message="Style reference uploaded")


@router.get("/{name}/style-reference")
async def get_style_reference(name: str):
    """Get the current style reference image."""
    ref_path = get_style_reference_path(name)
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="No style reference set")
    return FileResponse(ref_path, media_type="image/png")


@router.delete("/{name}/style-reference", response_model=SuccessResponse)
async def delete_style_reference(name: str):
    """Delete the style reference image."""
    ref_path = get_style_reference_path(name)
    if ref_path.exists():
        ref_path.unlink()
    return SuccessResponse(message="Style reference removed")


def load_style_reference(project_name: str):
    """Load the style reference image if it exists, returns PIL Image or None."""
    ref_path = get_style_reference_path(project_name)
    if ref_path.exists():
        from PIL import Image
        return Image.open(ref_path).convert("RGBA")
    return None


@router.post("/{name}/generate", response_model=GenerationStartResponse)
async def start_generation(name: str, request: GenerationStartRequest):
    """Start generating all tiles for a project.

    Returns immediately with a task ID. Use WebSocket to track progress.
    """
    import os

    # Validate API keys before starting
    missing_keys = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.environ.get("MAPBOX_ACCESS_TOKEN"):
        missing_keys.append("MAPBOX_ACCESS_TOKEN")
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variable(s): {', '.join(missing_keys)}. "
            "Set them and restart the API server before generating."
        )

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

    # Load style reference if available
    style_ref = load_style_reference(name)

    async def run_generation():
        """Run the generation in background, then assemble tiles."""
        service = GenerationService(project=project, cache_dir=cache_dir)
        loop = asyncio.get_running_loop()

        # Progress callback runs in worker thread, so schedule update on main loop
        def progress_callback(progress: GenerationProgress):
            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                task_manager.update_progress(task_info.task_id, progress),
            )

        results, final_progress = await asyncio.to_thread(
            service.generate_all_tiles,
            progress_callback=progress_callback,
            style_reference=style_ref,
        )

        # Auto-assemble tiles into final image
        gen_offsets = load_tile_offsets(name)

        def assemble():
            assembled = service.assemble_tiles(results, apply_perspective=False, tile_offsets=gen_offsets)
            if assembled is not None:
                output_dir = project.project_dir / "output" if project.project_dir else Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "assembled.png"
                assembled.save(output_path)

        await asyncio.to_thread(assemble)

        return results, final_progress

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


@router.post("/{name}/assemble", response_model=SuccessResponse)
async def assemble_tiles(name: str):
    """Assemble cached tiles into a final image.

    Loads generated tiles from cache and blends them into a single assembled image.
    """
    from ...services.generation_service import TileResult

    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    service = GenerationService(project=project, cache_dir=cache_dir)
    specs = service.calculate_tile_specs()

    offsets = load_tile_offsets(name)

    def do_assemble():
        # Load cached tiles into TileResult objects
        generated_dir = cache_dir / "generated"
        results = []
        for spec in specs:
            tile_path = generated_dir / f"tile_{spec.col}_{spec.row}.png"
            if tile_path.exists():
                from PIL import Image
                img = Image.open(tile_path).convert("RGBA")
                results.append(TileResult(spec=spec, generated_image=img))

        if not results:
            return None

        assembled = service.assemble_tiles(results, apply_perspective=False, tile_offsets=offsets)
        if assembled is not None:
            output_dir = project.project_dir / "output" if project.project_dir else Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "assembled.png"
            assembled.save(output_path)
            return output_path
        return None

    result = await asyncio.to_thread(do_assemble)

    if result is None:
        raise HTTPException(status_code=400, detail="No generated tiles found to assemble")

    return SuccessResponse(message=f"Assembled image saved to {result}")
