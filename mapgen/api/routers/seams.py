"""Seam repair endpoints."""

import asyncio
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from ...models.seam import SeamInfo as ServiceSeamInfo
from ...services.seam_repair_service import SeamRepairService
from ..schemas import (
    SeamBatchRepairRequest,
    SeamInfo,
    SeamListResponse,
    SuccessResponse,
)
from .projects import get_project_cache_dir, load_project

router = APIRouter()


def get_seam_service() -> SeamRepairService:
    """Get a seam repair service."""
    return SeamRepairService()


def get_repaired_seams_dir(project_name: str) -> Path:
    """Get the directory for repaired seam images."""
    return get_project_cache_dir(project_name) / "seams" / "repaired"


def get_assembled_image_path(project_name: str) -> Optional[Path]:
    """Get the path to the assembled image for a project."""
    project = load_project(project_name)
    if not project.project_dir:
        return None

    candidates = [
        project.project_dir / "output" / "assembled.png",
        project.project_dir / "output" / "final.png",
        project.project_dir / "output" / f"{project.name}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_assembled_image(project_name: str) -> Image.Image:
    """Load the assembled image for a project, or raise 404."""
    assembled_path = get_assembled_image_path(project_name)
    if not assembled_path:
        raise HTTPException(
            status_code=404,
            detail="No assembled image found. Run tile generation and assembly first.",
        )
    return Image.open(assembled_path).convert("RGBA")


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
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

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
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

    for seam in seams:
        if seam.id == seam_id:
            return service_seam_to_api_seam(seam, name)

    raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")


@router.get("/{name}/seams/{seam_id}/preview")
async def get_seam_preview(name: str, seam_id: str):
    """Get a preview image of the seam region (before repair)."""
    project = load_project(name)
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

    seam = None
    for s in seams:
        if s.id == seam_id:
            seam = s
            break

    if seam is None:
        raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")

    # Extract the seam strip from the assembled image
    assembled = load_assembled_image(name)
    preview = assembled.crop((seam.x, seam.y, seam.x + seam.width, seam.y + seam.height))

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
async def repair_seam(name: str, seam_id: str):
    """Repair a single seam."""
    project = load_project(name)
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

    seam = None
    for s in seams:
        if s.id == seam_id:
            seam = s
            break

    if seam is None:
        raise HTTPException(status_code=404, detail=f"Seam '{seam_id}' not found")

    assembled = load_assembled_image(name)

    from ...services.gemini_service import GeminiService

    gemini = GeminiService()
    result = await asyncio.to_thread(
        service.repair_seam,
        seam,
        assembled,
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
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    all_seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

    # Filter to requested seams
    seam_ids_set = set(request.seam_ids)
    seams_to_repair = [seam for seam in all_seams if seam.id in seam_ids_set]

    if not seams_to_repair:
        raise HTTPException(status_code=404, detail="No matching seams found")

    assembled = load_assembled_image(name)

    from ...services.gemini_service import GeminiService

    gemini = GeminiService()

    # Repair seams
    _updated_image, results = await asyncio.to_thread(
        service.repair_seams_batch,
        seams_to_repair,
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
    service = get_seam_service()

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    seams = service.identify_seams(cols, rows, project.output.width, project.output.height)

    # Filter to only unrepaired seams
    repaired_dir = get_repaired_seams_dir(name)
    unrepaired = [s for s in seams if not (repaired_dir / f"seam_{s.id}.png").exists()]

    if not unrepaired:
        return SuccessResponse(message="All seams already repaired")

    request = SeamBatchRepairRequest(seam_ids=[s.id for s in unrepaired])
    return await repair_seams_batch(name, request)
