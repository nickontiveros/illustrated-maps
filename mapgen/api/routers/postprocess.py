"""Post-processing pipeline endpoints."""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from ...config import get_config
from ...models.project import Project
from ...services.border_service import BorderService
from ...services.composition_service import CompositionService
from ...services.psd_service import PSDService
from ...services.typography_service import TypographyService
from ..dependencies import APIKeys, get_api_keys, create_gemini_service
from ..schemas import (
    GenerationStartResponse,
    GenerationStatus,
    LabelsRequest,
    PerspectiveRequest,
    PipelineRequest,
    PostProcessStatus,
    SuccessResponse,
)
from ..tasks import task_manager
from .projects import get_project_cache_dir, load_project

router = APIRouter()


# Stage output filenames (in processing order)
STAGE_FILES = {
    "assembled": "assembled.png",
    "composed": "composed.png",
    "labeled": "labeled.png",
    "perspective": "perspective.png",
    "bordered": "bordered.png",
    "outpainted": "outpainted.png",
}

# Stage order for finding the latest available input (reverse pipeline order)
STAGE_ORDER = ["outpainted", "bordered", "perspective", "labeled", "composed", "assembled"]


def get_output_dir(project: Project) -> Path:
    """Get the output directory for a project."""
    output_dir = project.project_dir / "output" if project.project_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_stage_path(project: Project, stage: str) -> Path:
    """Get the file path for a specific stage output."""
    output_dir = get_output_dir(project)
    return output_dir / STAGE_FILES[stage]


def find_latest_input(project: Project, before_stage: Optional[str] = None) -> tuple[Optional[Path], Optional[str]]:
    """Find the latest stage output to use as input.

    Args:
        project: Project model
        before_stage: If set, only consider stages before this one in the pipeline

    Returns:
        (path, stage_name) or (None, None) if nothing found
    """
    order = STAGE_ORDER
    if before_stage and before_stage in order:
        idx = order.index(before_stage)
        order = order[idx + 1:]  # Only stages after (earlier in pipeline)

    for stage in order:
        path = get_stage_path(project, stage)
        if path.exists():
            return path, stage
    return None, None


def load_stage_image(project: Project, before_stage: Optional[str] = None) -> Image.Image:
    """Load the latest stage image, raising 404 if none exists."""
    path, stage = find_latest_input(project, before_stage)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No assembled image found. Generate and assemble tiles first.",
        )
    return Image.open(path).convert("RGBA")


# =============================================================================
# Status endpoint
# =============================================================================


@router.get("/{name}/postprocess/status", response_model=PostProcessStatus)
async def get_postprocess_status(name: str):
    """Get which post-processing stage outputs exist."""
    project = load_project(name)

    stages = {}
    latest = None
    for stage in STAGE_ORDER:
        exists = get_stage_path(project, stage).exists()
        stages[stage] = exists
        if exists and latest is None:
            latest = stage

    return PostProcessStatus(
        assembled=stages["assembled"],
        composed=stages["composed"],
        labeled=stages["labeled"],
        perspective=stages["perspective"],
        bordered=stages["bordered"],
        outpainted=stages["outpainted"],
        latest_stage=latest,
    )


# =============================================================================
# Synchronous post-processing steps
# =============================================================================


@router.post("/{name}/postprocess/compose", response_model=SuccessResponse)
async def compose_landmarks(name: str):
    """Place illustrated landmarks onto the map."""
    project = load_project(name)
    base_image = load_stage_image(project)

    # Place landmarks that have illustrations
    composition = CompositionService(
        perspective_convergence=0.7,
        perspective_vertical_scale=0.4,
        perspective_horizon_margin=0.15,
    )

    placed = []
    for landmark in project.landmarks:
        if not landmark.illustrated_path:
            continue
        illust_path = project.project_dir / landmark.illustrated_path
        if not illust_path.exists():
            continue

        illustration = Image.open(illust_path).convert("RGBA")
        placed_lm = composition.place_landmark(
            landmark=landmark,
            illustration=illustration,
            base_map_size=base_image.size,
            bbox=project.region,
            apply_perspective=False,
        )
        placed.append(placed_lm)

    if not placed:
        raise HTTPException(
            status_code=400,
            detail="No illustrated landmarks found to compose. Illustrate landmarks first.",
        )

    def do_compose():
        return composition.composite_map(base_image, placed)

    result = await asyncio.to_thread(do_compose)

    output_path = get_stage_path(project, "composed")
    result.save(output_path)

    return SuccessResponse(message=f"Composed {len(placed)} landmarks onto map")


@router.post("/{name}/postprocess/labels", response_model=SuccessResponse)
async def add_labels(name: str, request: Optional[LabelsRequest] = None):
    """Render OSM typography labels onto the map."""
    project = load_project(name)
    base_image = load_stage_image(project, before_stage="labeled")

    if not project.style.typography or not project.style.typography.enabled:
        raise HTTPException(
            status_code=400,
            detail="Typography is not enabled. Enable it in project settings first.",
        )

    include_shields = request.include_shields if request else True
    cache_dir = get_project_cache_dir(name)

    def do_labels():
        from ...services.osm_service import OSMService

        osm_service = OSMService(cache_dir=str(cache_dir / "osm"))
        osm_data = osm_service.fetch_region_data(project.region, detail_level="full")

        typo = TypographyService(settings=project.style.typography)
        labeled = typo.extract_and_render(
            base_image,
            osm_data,
            project.region,
            rotation_degrees=project.style.orientation_degrees or 0.0,
        )

        # Add highway shields
        if include_shields and osm_data.roads is not None:
            from ...services.highway_shield_service import HighwayShieldService

            shields = HighwayShieldService.extract_shield_positions(
                osm_data.roads, project.region, labeled.size,
            )
            if shields:
                labeled = HighwayShieldService.render_shields_on_image(labeled, shields)

        return labeled

    result = await asyncio.to_thread(do_labels)

    output_path = get_stage_path(project, "labeled")
    result.save(output_path)

    return SuccessResponse(message="Labels rendered successfully")


@router.post("/{name}/postprocess/border", response_model=SuccessResponse)
async def add_border(name: str):
    """Add a decorative border to the map."""
    project = load_project(name)
    base_image = load_stage_image(project, before_stage="bordered")

    if not project.border or not project.border.enabled:
        raise HTTPException(
            status_code=400,
            detail="Border is not enabled. Enable it in project settings first.",
        )

    def do_border():
        service = BorderService()
        return service.render_border(
            base_image,
            project.border,
            title=project.title,
            subtitle=project.subtitle,
            bbox=project.region,
            rotation_degrees=project.style.orientation_degrees or 0.0,
        )

    result = await asyncio.to_thread(do_border)

    output_path = get_stage_path(project, "bordered")
    result.save(output_path)

    return SuccessResponse(message="Border added successfully")


@router.post("/{name}/postprocess/perspective", response_model=SuccessResponse)
async def apply_perspective(name: str, request: Optional[PerspectiveRequest] = None):
    """Apply bird's eye perspective transformation to the map."""
    project = load_project(name)
    base_image = load_stage_image(project, before_stage="perspective")

    params = request or PerspectiveRequest()

    def do_perspective():
        from ...services.perspective_service import PerspectiveService

        service = PerspectiveService(
            angle=params.angle,
            convergence=params.convergence,
            vertical_scale=params.vertical_scale,
            horizon_margin=params.horizon_margin,
        )
        return service.transform_image(base_image)

    result = await asyncio.to_thread(do_perspective)

    output_path = get_stage_path(project, "perspective")
    result.save(output_path)

    return SuccessResponse(message="Perspective transformation applied")


@router.post("/{name}/postprocess/perspective/preview")
async def preview_perspective(
    name: str,
    request: Optional[PerspectiveRequest] = None,
    size: Optional[int] = Query(800, description="Max dimension for preview thumbnail"),
):
    """Preview perspective transform without saving (returns image directly)."""
    project = load_project(name)
    base_image = load_stage_image(project, before_stage="perspective")

    params = request or PerspectiveRequest()

    def do_perspective():
        from ...services.perspective_service import PerspectiveService

        # Downscale for preview if needed
        img = base_image
        if size and max(img.size) > size:
            img.thumbnail((size, size))

        service = PerspectiveService(
            angle=params.angle,
            convergence=params.convergence,
            vertical_scale=params.vertical_scale,
            horizon_margin=params.horizon_margin,
        )
        return service.transform_image(img)

    result = await asyncio.to_thread(do_perspective)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


# =============================================================================
# Async post-processing steps (long-running)
# =============================================================================


@router.post("/{name}/postprocess/outpaint", response_model=GenerationStartResponse)
async def start_outpaint(name: str, api_keys: APIKeys = Depends(get_api_keys)):
    """Start outpainting to fill empty edges (requires Gemini API key)."""
    project = load_project(name)
    base_image = load_stage_image(project, before_stage="outpainted")

    if not api_keys.google_api_key:
        raise HTTPException(
            status_code=400,
            detail="Google API key is required for outpainting. Set it in API Key Settings.",
        )

    # Check if already running
    active = await task_manager.get_active_task(name, "outpaint")
    if active:
        raise HTTPException(status_code=409, detail="Outpainting already in progress")

    gemini = create_gemini_service(api_keys)

    async def run_outpaint():
        from ...services.outpainting_service import OutpaintingService

        service = OutpaintingService(gemini_service=gemini)
        output_path = get_stage_path(project, "outpainted")

        result = await asyncio.to_thread(
            service.outpaint_image,
            base_image,
            project.region,
            output_path=output_path,
        )
        return result

    task_info = await task_manager.create_task(
        project_name=name,
        task_type="outpaint",
        coro=run_outpaint(),
    )

    return GenerationStartResponse(
        task_id=task_info.task_id,
        status=GenerationStatus.RUNNING,
        total_tiles=1,
        websocket_url=f"/api/ws/generation/{task_info.task_id}",
    )


# =============================================================================
# Pipeline endpoint (run multiple steps sequentially)
# =============================================================================


@router.post("/{name}/postprocess/pipeline", response_model=GenerationStartResponse)
async def start_pipeline(
    name: str,
    request: PipelineRequest,
    api_keys: APIKeys = Depends(get_api_keys),
):
    """Run selected post-processing steps sequentially."""
    project = load_project(name)

    valid_steps = {"compose", "labels", "perspective", "border", "outpaint"}
    invalid = set(request.steps) - valid_steps
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid steps: {invalid}")

    if "outpaint" in request.steps and not api_keys.google_api_key:
        raise HTTPException(
            status_code=400,
            detail="Google API key required for outpainting step.",
        )

    active = await task_manager.get_active_task(name, "pipeline")
    if active:
        raise HTTPException(status_code=409, detail="Pipeline already in progress")

    gemini = create_gemini_service(api_keys) if api_keys.google_api_key else None

    async def run_pipeline():
        nonlocal project
        for step in request.steps:
            # Reload project each step in case settings changed
            project = load_project(name)

            if step == "compose":
                base = load_stage_image(project)
                composition = CompositionService()
                placed = []
                for lm in project.landmarks:
                    if not lm.illustrated_path:
                        continue
                    illust_path = project.project_dir / lm.illustrated_path
                    if not illust_path.exists():
                        continue
                    illustration = Image.open(illust_path).convert("RGBA")
                    placed_lm = composition.place_landmark(
                        landmark=lm,
                        illustration=illustration,
                        base_map_size=base.size,
                        bbox=project.region,
                        apply_perspective=False,
                    )
                    placed.append(placed_lm)
                if placed:
                    result = await asyncio.to_thread(composition.composite_map, base, placed)
                    result.save(get_stage_path(project, "composed"))

            elif step == "labels":
                if project.style.typography and project.style.typography.enabled:
                    base = load_stage_image(project, before_stage="labeled")
                    cache_dir = get_project_cache_dir(name)

                    def do_labels():
                        from ...services.osm_service import OSMService
                        osm_svc = OSMService(cache_dir=str(cache_dir / "osm"))
                        osm_data = osm_svc.fetch_region_data(project.region, detail_level="full")
                        typo = TypographyService(settings=project.style.typography)
                        labeled = typo.extract_and_render(
                            base, osm_data, project.region,
                            rotation_degrees=project.style.orientation_degrees or 0.0,
                        )
                        # Add highway shields
                        if request.include_shields and osm_data.roads is not None:
                            from ...services.highway_shield_service import HighwayShieldService
                            shields = HighwayShieldService.extract_shield_positions(
                                osm_data.roads, project.region, labeled.size,
                            )
                            if shields:
                                labeled = HighwayShieldService.render_shields_on_image(labeled, shields)
                        return labeled

                    result = await asyncio.to_thread(do_labels)
                    result.save(get_stage_path(project, "labeled"))

            elif step == "perspective":
                base = load_stage_image(project, before_stage="perspective")

                def do_perspective():
                    from ...services.perspective_service import PerspectiveService
                    svc = PerspectiveService()
                    return svc.transform_image(base)

                result = await asyncio.to_thread(do_perspective)
                result.save(get_stage_path(project, "perspective"))

            elif step == "border":
                if project.border and project.border.enabled:
                    base = load_stage_image(project, before_stage="bordered")

                    def do_border():
                        svc = BorderService()
                        return svc.render_border(
                            base, project.border,
                            title=project.title, subtitle=project.subtitle,
                            bbox=project.region,
                            rotation_degrees=project.style.orientation_degrees or 0.0,
                        )

                    result = await asyncio.to_thread(do_border)
                    result.save(get_stage_path(project, "bordered"))

            elif step == "outpaint":
                if gemini:
                    base = load_stage_image(project, before_stage="outpainted")
                    from ...services.outpainting_service import OutpaintingService
                    svc = OutpaintingService(gemini_service=gemini)
                    output_path = get_stage_path(project, "outpainted")
                    await asyncio.to_thread(
                        svc.outpaint_image, base, project.region, output_path=output_path,
                    )

    task_info = await task_manager.create_task(
        project_name=name,
        task_type="pipeline",
        coro=run_pipeline(),
    )

    return GenerationStartResponse(
        task_id=task_info.task_id,
        status=GenerationStatus.RUNNING,
        total_tiles=len(request.steps),
        websocket_url=f"/api/ws/generation/{task_info.task_id}",
    )


# =============================================================================
# Image serving
# =============================================================================


@router.get("/{name}/postprocess/{stage}/image")
async def get_stage_image(
    name: str,
    stage: str,
    size: Optional[int] = Query(None, description="Resize to this size (thumbnail)"),
):
    """Serve a post-processing stage's output image."""
    if stage not in STAGE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")

    project = load_project(name)
    path = get_stage_path(project, stage)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Stage '{stage}' output not found")

    if size:
        img = Image.open(path)
        img.thumbnail((size, size))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    return FileResponse(path, media_type="image/png")


# =============================================================================
# PSD Export
# =============================================================================


@router.post("/{name}/postprocess/export-psd")
async def export_psd(name: str):
    """Export the map as a layered PSD file."""
    project = load_project(name)

    # Find the best available base
    path, stage = find_latest_input(project)
    if path is None:
        raise HTTPException(status_code=404, detail="No map output found to export")

    base_image = Image.open(path).convert("RGBA")

    # Build layer stack
    composition = CompositionService()
    placed = []
    for lm in project.landmarks:
        if not lm.illustrated_path:
            continue
        illust_path = project.project_dir / lm.illustrated_path
        if not illust_path.exists():
            continue
        illustration = Image.open(illust_path).convert("RGBA")
        placed_lm = composition.place_landmark(
            landmark=lm,
            illustration=illustration,
            base_map_size=base_image.size,
            bbox=project.region,
            apply_perspective=False,
        )
        placed.append(placed_lm)

    layers = composition.create_layer_stack(base_image, placed)

    def do_export():
        psd_service = PSDService()
        output_dir = get_output_dir(project)
        output_path = output_dir / f"{project.name}.psd"
        psd_service.create_layered_psd(layers, output_path, canvas_size=base_image.size)
        return output_path

    output_path = await asyncio.to_thread(do_export)

    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=f"{project.name}.psd",
    )


# =============================================================================
# Preview endpoints
# =============================================================================


@router.get("/{name}/preview/osm")
async def preview_osm(name: str):
    """Render OSM features as a preview image (no API key needed)."""
    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    def render():
        from ...services.osm_service import OSMService
        from ...services.render_service import RenderService

        osm_service = OSMService(cache_dir=str(cache_dir / "osm"))
        osm_data = osm_service.fetch_region_data(project.region, detail_level="full")

        render_svc = RenderService()
        # Render OSM data onto a blank canvas
        canvas = Image.new("RGBA", (project.output.width, project.output.height), (255, 255, 255, 255))
        result = render_svc.render_composite_reference(
            satellite_image=canvas,
            osm_data=osm_data,
            bbox=project.region,
            output_size=(project.output.width, project.output.height),
            osm_opacity=1.0,
        )
        return result

    result = await asyncio.to_thread(render)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@router.get("/{name}/preview/composite")
async def preview_composite(
    name: str,
    api_keys: APIKeys = Depends(get_api_keys),
):
    """Render satellite + OSM overlay reference (needs Mapbox token)."""
    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    if not api_keys.mapbox_access_token:
        raise HTTPException(
            status_code=400,
            detail="Mapbox access token required for composite preview.",
        )

    def render():
        from ...services.osm_service import OSMService
        from ...services.render_service import RenderService
        from ...services.satellite_service import SatelliteService

        sat_service = SatelliteService(
            access_token=api_keys.mapbox_access_token,
            cache_dir=str(cache_dir / "satellite"),
        )
        satellite = sat_service.fetch_satellite_imagery(
            project.region,
            width=project.output.width,
            height=project.output.height,
        )

        osm_service = OSMService(cache_dir=str(cache_dir / "osm"))
        osm_data = osm_service.fetch_region_data(project.region, detail_level="full")

        render_svc = RenderService()
        result = render_svc.render_composite_reference(
            satellite_image=satellite,
            osm_data=osm_data,
            bbox=project.region,
            output_size=(project.output.width, project.output.height),
            osm_opacity=0.6,
        )
        return result

    result = await asyncio.to_thread(render)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
