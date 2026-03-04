"""Landmark management endpoints."""

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ...config import get_config
from ...models.landmark import Landmark
from ...services.landmark_service import LandmarkService
from ..dependencies import APIKeys, get_api_keys, create_gemini_service
from ..schemas import (
    LandmarkCreate,
    LandmarkDetail,
    LandmarkDiscoverRequest,
    LandmarkDiscoverResponse,
    LandmarkUpdate,
    SuccessResponse,
)
from .projects import get_project_cache_dir, get_project_path, load_project

router = APIRouter()


def get_landmark_output_dir(project_name: str) -> Path:
    """Get the output directory for illustrated landmarks."""
    project = load_project(project_name)
    if project.project_dir:
        return project.project_dir / "output" / "landmarks"
    return get_project_cache_dir(project_name) / "landmarks"


def save_project(project_name: str, project):
    """Save the project configuration."""
    project.to_yaml(get_project_path(project_name))


@router.get("/{name}/landmarks", response_model=list[LandmarkDetail])
async def list_landmarks(name: str):
    """List all landmarks in a project."""
    project = load_project(name)

    return [
        LandmarkDetail.from_landmark(landmark, project.project_dir)
        for landmark in project.landmarks
    ]


@router.post("/{name}/landmarks", response_model=LandmarkDetail, status_code=201)
async def create_landmark(name: str, request: LandmarkCreate):
    """Add a new landmark to the project."""
    project = load_project(name)

    # Check for duplicate name
    for existing in project.landmarks:
        if existing.name.lower() == request.name.lower():
            raise HTTPException(
                status_code=409,
                detail=f"Landmark '{request.name}' already exists"
            )

    # Create landmark
    landmark = Landmark(
        name=request.name,
        latitude=request.latitude,
        longitude=request.longitude,
        photo=request.photo,
        logo=request.logo,
        scale=request.scale,
        z_index=request.z_index,
        rotation=request.rotation,
        wikipedia_url=request.wikipedia_url,
        wikidata_id=request.wikidata_id,
    )

    # Add to project and save
    project.landmarks.append(landmark)
    save_project(name, project)

    return LandmarkDetail.from_landmark(landmark, project.project_dir)


@router.get("/{name}/landmarks/{landmark_name}", response_model=LandmarkDetail)
async def get_landmark(name: str, landmark_name: str):
    """Get details about a specific landmark."""
    project = load_project(name)

    for landmark in project.landmarks:
        if landmark.name.lower() == landmark_name.lower():
            return LandmarkDetail.from_landmark(landmark, project.project_dir)

    raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")


@router.put("/{name}/landmarks/{landmark_name}", response_model=LandmarkDetail)
async def update_landmark(name: str, landmark_name: str, request: LandmarkUpdate):
    """Update a landmark's properties."""
    project = load_project(name)

    landmark_idx = None
    for i, landmark in enumerate(project.landmarks):
        if landmark.name.lower() == landmark_name.lower():
            landmark_idx = i
            break

    if landmark_idx is None:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")

    landmark = project.landmarks[landmark_idx]

    # Update fields
    if request.latitude is not None:
        landmark.latitude = request.latitude
    if request.longitude is not None:
        landmark.longitude = request.longitude
    if request.scale is not None:
        landmark.scale = request.scale
    if request.z_index is not None:
        landmark.z_index = request.z_index
    if request.rotation is not None:
        landmark.rotation = request.rotation
    if request.wikipedia_url is not None:
        landmark.wikipedia_url = request.wikipedia_url
    if request.wikidata_id is not None:
        landmark.wikidata_id = request.wikidata_id

    # Save project
    save_project(name, project)

    return LandmarkDetail.from_landmark(landmark, project.project_dir)


@router.delete("/{name}/landmarks/{landmark_name}", response_model=SuccessResponse)
async def delete_landmark(name: str, landmark_name: str):
    """Remove a landmark from the project."""
    project = load_project(name)

    landmark_idx = None
    for i, landmark in enumerate(project.landmarks):
        if landmark.name.lower() == landmark_name.lower():
            landmark_idx = i
            break

    if landmark_idx is None:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")

    # Remove landmark
    project.landmarks.pop(landmark_idx)
    save_project(name, project)

    return SuccessResponse(message=f"Landmark '{landmark_name}' deleted")


@router.post("/{name}/landmarks/discover", response_model=LandmarkDiscoverResponse)
async def discover_landmarks(name: str, request: LandmarkDiscoverRequest):
    """Discover notable landmarks from OSM data within the project region."""
    from ...services.landmark_discovery_service import LandmarkDiscoveryService
    from ...services.osm_service import OSMService
    from ...models.narrative import NarrativeSettings

    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    # Fetch OSM data
    osm_service = OSMService(cache_dir=str(cache_dir / "osm"))
    osm_data = osm_service.fetch_region_data(project.region, detail_level="full")

    settings = NarrativeSettings(
        min_importance_score=request.min_importance_score,
        max_landmarks=request.max_landmarks,
    )
    discovery = LandmarkDiscoveryService(settings=settings)

    landmarks = discovery.discover_landmarks(
        bbox=project.region,
        buildings_gdf=osm_data.buildings,
    )

    # Convert to LandmarkDetail responses
    landmark_details = [
        LandmarkDetail.from_landmark(lm, project.project_dir)
        for lm in landmarks
    ]

    return LandmarkDiscoverResponse(
        discovered=len(landmarks),
        landmarks=landmark_details,
    )


@router.get("/{name}/landmarks/{landmark_name}/photo")
async def get_landmark_photo(name: str, landmark_name: str):
    """Get the photo image for a landmark."""
    project = load_project(name)

    for landmark in project.landmarks:
        if landmark.name.lower() == landmark_name.lower():
            if not landmark.photo:
                raise HTTPException(status_code=404, detail="Landmark has no photo")

            photo_path = landmark.resolve_photo_path(project.project_dir)
            if not photo_path or not photo_path.exists():
                raise HTTPException(status_code=404, detail="Photo file not found")

            return FileResponse(photo_path)

    raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")


@router.get("/{name}/landmarks/{landmark_name}/illustration")
async def get_landmark_illustration(name: str, landmark_name: str):
    """Get the illustrated image for a landmark."""
    project = load_project(name)

    for landmark in project.landmarks:
        if landmark.name.lower() == landmark_name.lower():
            if not landmark.illustrated_path:
                raise HTTPException(status_code=404, detail="Landmark has no illustration")

            illust_path = project.project_dir / landmark.illustrated_path
            if not illust_path.exists():
                raise HTTPException(status_code=404, detail="Illustration file not found")

            return FileResponse(illust_path)

    raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")


@router.post("/{name}/landmarks/{landmark_name}/illustrate", response_model=SuccessResponse)
async def illustrate_landmark(name: str, landmark_name: str, api_keys: APIKeys = Depends(get_api_keys)):
    """Generate an illustration for a landmark."""
    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    landmark = None
    landmark_idx = None
    for i, lm in enumerate(project.landmarks):
        if lm.name.lower() == landmark_name.lower():
            landmark = lm
            landmark_idx = i
            break

    if landmark is None:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")

    if not landmark.photo:
        raise HTTPException(status_code=400, detail="Landmark has no photo to illustrate")

    # Get style reference from generated tiles
    # Check both flat cache and legacy path
    style_reference = None
    for subdir in ["generated", "generation/generated"]:
        generated_dir = cache_dir / subdir
        if generated_dir.exists():
            from PIL import Image
            for tile_path in generated_dir.glob("tile_*.png"):
                style_reference = Image.open(tile_path)
                break
        if style_reference is not None:
            break

    if style_reference is None:
        raise HTTPException(
            status_code=400,
            detail="No generated tiles found for style reference. Generate tiles first."
        )

    # Create landmark service and illustrate
    gemini = create_gemini_service(api_keys)
    service = LandmarkService(project=project, gemini_service=gemini)

    result = await asyncio.to_thread(
        service.illustrate_landmark,
        landmark,
        style_reference,
    )

    if result.error:
        raise HTTPException(status_code=500, detail=f"Illustration failed: {result.error}")

    # Save the illustration
    output_dir = get_landmark_output_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{landmark.name.replace(' ', '_')}_illustrated.png"
    output_path = output_dir / filename
    result.image.save(output_path)

    # Update landmark with illustration path
    relative_path = f"output/landmarks/{filename}"
    project.landmarks[landmark_idx].illustrated_path = relative_path
    save_project(name, project)

    return SuccessResponse(message=f"Landmark '{landmark_name}' illustrated successfully")


@router.post("/{name}/landmarks/{landmark_name}/upload-photo", response_model=SuccessResponse)
async def upload_landmark_photo(name: str, landmark_name: str, file: UploadFile = File(...)):
    """Upload a photo for a landmark."""
    project = load_project(name)

    landmark_idx = None
    for i, landmark in enumerate(project.landmarks):
        if landmark.name.lower() == landmark_name.lower():
            landmark_idx = i
            break

    if landmark_idx is None:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save the file
    landmarks_dir = project.landmarks_dir
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{landmark_name.replace(' ', '_')}{ext}"
    file_path = landmarks_dir / filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update landmark
    relative_path = f"landmarks/{filename}"
    project.landmarks[landmark_idx].photo = relative_path
    save_project(name, project)

    return SuccessResponse(message=f"Photo uploaded for landmark '{landmark_name}'")


@router.post("/{name}/landmarks/illustrate-all", response_model=SuccessResponse)
async def illustrate_all_landmarks(name: str, api_keys: APIKeys = Depends(get_api_keys)):
    """Generate illustrations for all landmarks that have photos."""
    project = load_project(name)
    cache_dir = get_project_cache_dir(name)

    # Get landmarks with photos but no illustrations
    to_illustrate = [
        (i, lm) for i, lm in enumerate(project.landmarks)
        if lm.photo and not lm.illustrated_path
    ]

    if not to_illustrate:
        return SuccessResponse(message="No landmarks need illustration")

    # Get style reference — check both flat cache and legacy path
    style_reference = None
    for subdir in ["generated", "generation/generated"]:
        generated_dir = cache_dir / subdir
        if generated_dir.exists():
            from PIL import Image
            for tile_path in generated_dir.glob("tile_*.png"):
                style_reference = Image.open(tile_path)
                break
        if style_reference is not None:
            break

    if style_reference is None:
        raise HTTPException(
            status_code=400,
            detail="No generated tiles found for style reference. Generate tiles first."
        )

    # Illustrate all landmarks
    gemini = create_gemini_service(api_keys)
    service = LandmarkService(project=project, gemini_service=gemini)
    output_dir = get_landmark_output_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    for idx, landmark in to_illustrate:
        result = await asyncio.to_thread(
            service.illustrate_landmark,
            landmark,
            style_reference,
        )

        if result.error:
            failed += 1
            continue

        # Save
        filename = f"{landmark.name.replace(' ', '_')}_illustrated.png"
        output_path = output_dir / filename
        result.image.save(output_path)

        project.landmarks[idx].illustrated_path = f"output/landmarks/{filename}"
        succeeded += 1

    save_project(name, project)

    return SuccessResponse(
        message=f"Illustration completed: {succeeded} succeeded, {failed} failed"
    )


@router.post("/{name}/landmarks/{landmark_name}/fetch-photo", response_model=SuccessResponse)
async def fetch_wikipedia_photo(name: str, landmark_name: str):
    """Fetch a photo from Wikipedia/Wikidata for a specific landmark.

    Uses the landmark's wikipedia_url or wikidata_id to find an image.
    Falls back to searching Wikipedia by landmark name.
    """
    project = load_project(name)

    landmark = None
    landmark_idx = None
    for i, lm in enumerate(project.landmarks):
        if lm.name.lower() == landmark_name.lower():
            landmark = lm
            landmark_idx = i
            break

    if landmark is None:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")

    from ...services.wikipedia_image_service import WikipediaImageService

    cache_dir = str(project.project_dir / "cache" / "wikipedia") if project.project_dir else None

    def do_fetch():
        service = WikipediaImageService(cache_dir=cache_dir)
        return service.fetch_image_for_landmark(
            name=landmark.name,
            wikipedia_url=landmark.wikipedia_url,
            wikidata_id=landmark.wikidata_id,
        )

    image = await asyncio.to_thread(do_fetch)

    if image is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Wikipedia/Wikidata photo found for '{landmark_name}'"
        )

    # Save to project
    landmarks_dir = project.project_dir / "landmarks" / "photos"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    safe_name = landmark.name.replace(" ", "_").replace("/", "_")[:50]
    save_path = landmarks_dir / f"{safe_name}_wiki.jpg"
    image.save(save_path, "JPEG", quality=90)

    # Update landmark photo path
    project.landmarks[landmark_idx].photo = str(save_path.relative_to(project.project_dir))
    save_project(name, project)

    return SuccessResponse(message=f"Photo fetched and saved for '{landmark_name}'")
