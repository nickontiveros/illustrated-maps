"""Seam repair endpoints."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...config import get_config
from ...models.seam import SeamInfo as ServiceSeamInfo
from ...services.seam_repair_service import SeamRepairService
from ..schemas import (
    SeamBatchRepairRequest,
    SeamInfo,
    SeamListResponse,
    SeamRepairRequest,
    SuccessResponse,
)
from .projects import get_project_cache_dir, load_project

router = APIRouter()


def get_seam_service(project_name: str) -> SeamRepairService:
    """Get a seam repair service for a project."""
    project = load_project(project_name)
    return SeamRepairService(
        tile_size=project.tiles.size,
        overlap=project.tiles.overlap,
    )


def get_repaired_seams_dir(project_name: str) -> Path:
    """Get the directory for repaired seam images."""
    return get_project_cache_dir(project_name) / "seams" / "repaired"


def service_seam_to_api_seam(seam: ServiceSeamInfo, project_name: str) -> SeamInfo:
    """Convert a service SeamInfo to an API SeamInfo."""
    repaired_dir = get_repaired_seams_dir(project_name)
    is_repaired = (repaired_dir / f"seam_{seam.id}.png").exists()

    return SeamInfo(
        id=seam.id,
        orientation=seam.orientation,
        tile_a=seam.tile_a,
        tile_b=seam.tile_b,
        x=seam.x,
        y=seam.y,
        width=seam.width,
        height=seam.height,
        description=seam.description,
        is_repaired=is_repaired,
    )


@router.get("/{name}/seams", response_model=SeamListResponse)
async def list_seams(name: str):
    """List all seams in a project."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows)

    api_seams = [service_seam_to_api_seam(seam, name) for seam in seams]
    repaired_count = sum(1 for s in api_seams if s.is_repaired)

    return SeamListResponse(
        project_name=name,
        total_seams=len(api_seams),
        repaired_seams=repaired_count,
        seams=api_seams,
    )


@router.get("/{name}/seams/{seam_id}")
async def get_seam(name: str, seam_id: str):
    """Get details about a specific seam."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows)

    for seam in seams:
        if seam.id == seam_id:
            return service_seam_to_api_seam(seam, name)

    raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")


@router.get("/{name}/seams/{seam_id}/preview")
async def get_seam_preview(name: str, seam_id: str):
    """Get a preview image of the seam region (before repair)."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows)

    seam = None
    for s in seams:
        if s.id == seam_id:
            seam = s
            break

    if seam is None:
        raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")

    # Generate preview from assembled image or tiles
    cache_dir = get_project_cache_dir(name)
    generated_dir = cache_dir / "generation" / "generated"

    # Load the two tiles
    tile_a_path = generated_dir / f"tile_{seam.tile_a[0]}_{seam.tile_a[1]}.png"
    tile_b_path = generated_dir / f"tile_{seam.tile_b[0]}_{seam.tile_b[1]}.png"

    if not tile_a_path.exists() or not tile_b_path.exists():
        raise HTTPException(status_code=404, detail="Tile images not found for seam preview")

    from PIL import Image
    import io
    from fastapi.responses import StreamingResponse

    tile_a = Image.open(tile_a_path)
    tile_b = Image.open(tile_b_path)

    # Create a preview showing the seam region
    context = service.context_margin
    if seam.orientation == "horizontal":
        # Side by side tiles - show the overlap region
        preview_width = service.overlap + 2 * context
        preview_height = min(tile_a.height, 512)

        preview = Image.new("RGB", (preview_width, preview_height))

        # Extract overlap region from tile_a (right edge)
        a_region = tile_a.crop((
            tile_a.width - service.overlap // 2 - context,
            0,
            tile_a.width,
            preview_height,
        ))
        preview.paste(a_region, (0, 0))

        # Extract overlap region from tile_b (left edge)
        b_region = tile_b.crop((
            0,
            0,
            service.overlap // 2 + context,
            preview_height,
        ))
        preview.paste(b_region, (service.overlap // 2 + context, 0))
    else:
        # Stacked tiles - show the overlap region
        preview_width = min(tile_a.width, 512)
        preview_height = service.overlap + 2 * context

        preview = Image.new("RGB", (preview_width, preview_height))

        # Extract overlap region from tile_a (bottom edge)
        a_region = tile_a.crop((
            0,
            tile_a.height - service.overlap // 2 - context,
            preview_width,
            tile_a.height,
        ))
        preview.paste(a_region, (0, 0))

        # Extract overlap region from tile_b (top edge)
        b_region = tile_b.crop((
            0,
            0,
            preview_width,
            service.overlap // 2 + context,
        ))
        preview.paste(b_region, (0, service.overlap // 2 + context))

    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@router.get("/{name}/seams/{seam_id}/repaired")
async def get_seam_repaired(name: str, seam_id: str):
    """Get the repaired seam image."""
    repaired_dir = get_repaired_seams_dir(name)
    seam_path = repaired_dir / f"seam_{seam_id}.png"

    if not seam_path.exists():
        raise HTTPException(status_code=404, detail="Repaired seam image not found")

    return FileResponse(seam_path, media_type="image/png")


@router.post("/{name}/seams/{seam_id}/repair", response_model=SuccessResponse)
async def repair_seam(name: str, seam_id: str, request: SeamRepairRequest):
    """Repair a single seam."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows)

    seam = None
    for s in seams:
        if s.id == seam_id:
            seam = s
            break

    if seam is None:
        raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")

    # Load tiles
    cache_dir = get_project_cache_dir(name)
    generated_dir = cache_dir / "generation" / "generated"

    tile_a = service.load_tile(generated_dir, seam.tile_a[0], seam.tile_a[1])
    tile_b = service.load_tile(generated_dir, seam.tile_b[0], seam.tile_b[1])

    if tile_a is None or tile_b is None:
        raise HTTPException(status_code=404, detail="Tile images not found")

    # Repair the seam
    from ...services.gemini_service import GeminiService

    gemini = GeminiService()
    result = await asyncio.to_thread(
        service.repair_seam,
        seam,
        tile_a,
        tile_b,
        gemini,
    )

    if result.error:
        raise HTTPException(status_code=500, detail=f"Seam repair failed: {result.error}")

    # Save the repaired region
    repaired_dir = get_repaired_seams_dir(name)
    repaired_dir.mkdir(parents=True, exist_ok=True)
    result.repaired_region.save(repaired_dir / f"seam_{seam_id}.png")

    return SuccessResponse(message=f"Seam '{seam_id}' repaired successfully")


@router.post("/{name}/seams/repair-batch", response_model=SuccessResponse)
async def repair_seams_batch(name: str, request: SeamBatchRepairRequest):
    """Repair multiple seams in batch."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    all_seams = service.identify_seams(cols, rows)

    # Filter to requested seams
    seams_to_repair = []
    seam_ids_set = set(request.seam_ids)
    for seam in all_seams:
        if seam.id in seam_ids_set:
            seams_to_repair.append(seam)

    if not seams_to_repair:
        raise HTTPException(status_code=404, detail="No matching seams found")

    cache_dir = get_project_cache_dir(name)
    generated_dir = cache_dir / "generation" / "generated"

    # Load assembled image if it exists
    assembled_path = project.output_dir / "assembled.png" if project.project_dir else None
    assembled = None
    if assembled_path and assembled_path.exists():
        from PIL import Image
        assembled = Image.open(assembled_path)

    from ...services.gemini_service import GeminiService

    gemini = GeminiService()

    # Repair seams
    results = await asyncio.to_thread(
        service.repair_seams_batch,
        seams_to_repair,
        generated_dir,
        assembled,
        gemini,
    )

    # Save repaired regions
    repaired_dir = get_repaired_seams_dir(name)
    repaired_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    for result in results:
        if result.error:
            failed += 1
        else:
            succeeded += 1
            result.repaired_region.save(repaired_dir / f"seam_{result.seam.id}.png")

    return SuccessResponse(
        message=f"Batch repair completed: {succeeded} succeeded, {failed} failed"
    )


@router.post("/{name}/seams/repair-all", response_model=SuccessResponse)
async def repair_all_seams(name: str):
    """Repair all seams in the project."""
    project = load_project(name)
    service = get_seam_service(name)

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows)

    # Filter to only unrepaired seams
    repaired_dir = get_repaired_seams_dir(name)
    unrepaired = [s for s in seams if not (repaired_dir / f"seam_{s.id}.png").exists()]

    if not unrepaired:
        return SuccessResponse(message="All seams already repaired")

    request = SeamBatchRepairRequest(seam_ids=[s.id for s in unrepaired])
    return await repair_seams_batch(name, request)
