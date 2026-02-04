"""Deep Zoom Image (DZI) endpoints for large image viewing."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from ...services.dzi_service import DZIService, DZIInfo
from ..schemas import SuccessResponse
from .projects import get_project_cache_dir, load_project

router = APIRouter()


def get_dzi_output_dir(project_name: str) -> Path:
    """Get the directory for DZI tiles."""
    return get_project_cache_dir(project_name) / "dzi"


def get_assembled_image_path(project_name: str) -> Optional[Path]:
    """Get the path to the assembled image for a project."""
    project = load_project(project_name)
    if not project.project_dir:
        return None

    # Check multiple possible locations
    candidates = [
        project.project_dir / "output" / "assembled.png",
        project.project_dir / "output" / "final.png",
        project.project_dir / "output" / f"{project.name}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


@router.get("/{name}/dzi/info")
async def get_dzi_info(name: str):
    """Get DZI information for the assembled image.

    Returns dimensions, tile info, and whether tiles are generated.
    """
    assembled_path = get_assembled_image_path(name)
    if not assembled_path:
        raise HTTPException(status_code=404, detail="No assembled image found")

    dzi_dir = get_dzi_output_dir(name)
    service = DZIService()

    # Check if DZI is already generated
    is_generated = service.is_generated(dzi_dir, "assembled")

    # Get info from image
    info = service.get_info(assembled_path)

    return {
        "project_name": name,
        "is_generated": is_generated,
        "width": info.width,
        "height": info.height,
        "tile_size": info.tile_size,
        "overlap": info.overlap,
        "format": info.format,
        "max_level": info.max_level,
        "num_levels": info.num_levels,
    }


@router.post("/{name}/dzi/generate", response_model=SuccessResponse)
async def generate_dzi(name: str, force: bool = False):
    """Generate DZI tiles for the assembled image.

    Args:
        name: Project name
        force: If True, regenerate even if tiles exist
    """
    assembled_path = get_assembled_image_path(name)
    if not assembled_path:
        raise HTTPException(status_code=404, detail="No assembled image found")

    dzi_dir = get_dzi_output_dir(name)
    service = DZIService()

    # Check if already generated
    if not force and service.is_generated(dzi_dir, "assembled"):
        return SuccessResponse(message="DZI tiles already generated")

    # Generate tiles in background thread
    def generate():
        return service.generate_tiles(assembled_path, dzi_dir)

    await asyncio.to_thread(generate)

    return SuccessResponse(message="DZI tiles generated successfully")


@router.get("/{name}/dzi/assembled.dzi")
async def get_dzi_descriptor(name: str):
    """Get the DZI descriptor file."""
    dzi_dir = get_dzi_output_dir(name)
    dzi_path = dzi_dir / "assembled.dzi"

    if not dzi_path.exists():
        # Try to generate on-the-fly
        assembled_path = get_assembled_image_path(name)
        if not assembled_path:
            raise HTTPException(status_code=404, detail="No assembled image found")

        service = DZIService()
        info = service.get_info(assembled_path)

        return Response(
            content=info.to_dzi_xml(),
            media_type="application/xml",
        )

    return FileResponse(dzi_path, media_type="application/xml")


@router.get("/{name}/dzi/assembled_files/{level}/{col}_{row}.jpg")
async def get_dzi_tile_jpg(name: str, level: int, col: int, row: int):
    """Get a DZI tile (JPEG format)."""
    return await _get_dzi_tile(name, level, col, row, "jpg")


@router.get("/{name}/dzi/assembled_files/{level}/{col}_{row}.png")
async def get_dzi_tile_png(name: str, level: int, col: int, row: int):
    """Get a DZI tile (PNG format)."""
    return await _get_dzi_tile(name, level, col, row, "png")


async def _get_dzi_tile(name: str, level: int, col: int, row: int, format: str):
    """Internal function to get a DZI tile."""
    dzi_dir = get_dzi_output_dir(name)
    service = DZIService(format=format)

    tile_path = service.get_tile_path(dzi_dir, "assembled", level, col, row)

    if not tile_path.exists():
        # Check if we need to generate tiles
        if not service.is_generated(dzi_dir, "assembled"):
            assembled_path = get_assembled_image_path(name)
            if assembled_path:
                # Generate tiles on first request
                def generate():
                    return service.generate_tiles(assembled_path, dzi_dir)
                await asyncio.to_thread(generate)

                # Try again
                tile_path = service.get_tile_path(dzi_dir, "assembled", level, col, row)

    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="Tile not found")

    media_type = "image/jpeg" if format == "jpg" else "image/png"
    return FileResponse(tile_path, media_type=media_type)


@router.get("/{name}/dzi/tile/{level}/{col}/{row}")
async def get_dzi_tile_generic(
    name: str,
    level: int,
    col: int,
    row: int,
    format: str = "jpg",
):
    """Get a DZI tile (generic endpoint).

    Args:
        name: Project name
        level: Zoom level (0 = thumbnail, max_level = full resolution)
        col: Column index
        row: Row index
        format: Output format (jpg or png)
    """
    return await _get_dzi_tile(name, level, col, row, format)
