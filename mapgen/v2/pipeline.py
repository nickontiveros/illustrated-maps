"""End-to-end V2 pipeline: project config -> plan -> assets -> poster.

The V2 project file is intentionally tiny -- the product requirement is
"a region, a list of POIs, and an output size":

    name: "Manhattan"
    region: {north: 40.82, south: 40.70, east: -73.93, west: -74.02}
    output: {width_px: 7016, height_px: 9933, dpi: 300}
    style: vintage_tourist
    distortion_strength: 0.5
    pois:
      - {name: "Empire State Building", lat: 40.7484, lon: -73.9857, tier: 1,
         photo: "photos/esb.jpg"}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

from .assets.manifest import slugify
from .assets.studio import AssetGenerator, AssetStudio
from .compose import Compositor

if TYPE_CHECKING:
    from .compose.harmonize import MoodPass
from .ingest import SourceData, SourcePoi
from .plan import PlanBuilder, plan_to_svg
from .types import CameraSpec, CanvasSpec, PlanDocument, RegionBBox, StyleSpec

logger = logging.getLogger(__name__)

PLAN_FILENAME = "plan.json"
PREVIEW_FILENAME = "preview.svg"
ASSETS_DIRNAME = "assets"
POSTER_FILENAME = "poster.png"


class PoiConfig(BaseModel):
    name: str
    lat: float
    lon: float
    tier: int = Field(2, ge=1, le=3)
    photo: Optional[str] = None


class V2Project(BaseModel):
    name: str
    region: RegionBBox
    output: CanvasSpec = Field(default_factory=CanvasSpec)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    style: StyleSpec = Field(default_factory=StyleSpec)
    distortion_strength: float = 0.5
    seed: int = 7
    pois: list[PoiConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str) -> "V2Project":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("style"), str):
            data["style"] = {"preset": data["style"]}
        return cls.model_validate(data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


def fetch_source(project: V2Project, cache_dir: Optional[Path] = None) -> SourceData:
    """Fetch live OSM data for the project region (requires osmnx)."""
    from mapgen.models.project import BoundingBox  # V1 bbox for OSMService
    from mapgen.services.osm_service import OSMService

    from .ingest import from_osm_data

    bbox = BoundingBox(
        north=project.region.north,
        south=project.region.south,
        east=project.region.east,
        west=project.region.west,
    )
    osm = OSMService(cache_dir=str(cache_dir) if cache_dir else None)
    data = osm.fetch_region_data(bbox)
    source = from_osm_data(data, project.region)
    _attach_pois(source, project)
    return source


def _attach_pois(source: SourceData, project: V2Project) -> None:
    source.pois = [
        SourcePoi(
            id=slugify(p.name),
            name=p.name,
            latitude=p.lat,
            longitude=p.lon,
            tier=p.tier,
            photo=p.photo,
        )
        for p in project.pois
    ]


def build_plan(project: V2Project, source: SourceData) -> PlanDocument:
    _attach_pois(source, project)
    builder = PlanBuilder(
        canvas=project.output,
        camera=project.camera,
        style=project.style,
        distortion_strength=project.distortion_strength,
        seed=project.seed,
    )
    return builder.build(source, title=project.name)


def write_plan(plan: PlanDocument, project_dir: Path) -> tuple[Path, Path]:
    project_dir.mkdir(parents=True, exist_ok=True)
    plan_path = project_dir / PLAN_FILENAME
    preview_path = project_dir / PREVIEW_FILENAME
    plan.save(plan_path)
    preview_path.write_text(plan_to_svg(plan))
    return plan_path, preview_path


def generate_assets(
    plan: PlanDocument,
    project_dir: Path,
    generator: AssetGenerator,
    force: bool = False,
    only_ids: Optional[set[str]] = None,
) -> dict[str, Path]:
    studio = AssetStudio(generator, project_dir / ASSETS_DIRNAME)
    return studio.generate_all(plan, force=force, only_ids=only_ids)


def compose_poster(
    plan: PlanDocument,
    project_dir: Path,
    scale: float = 1.0,
    out_path: Optional[Path] = None,
    mood_pass: Optional["MoodPass"] = None,
) -> Path:
    compositor = Compositor(plan, project_dir / ASSETS_DIRNAME)
    image = compositor.render(scale=scale)
    if mood_pass is not None and plan.style.harmonize_strength > 0:
        from .compose.harmonize import harmonize

        image = harmonize(image, mood_pass, plan.style, plan.style.harmonize_strength)
    out = out_path or (project_dir / POSTER_FILENAME)
    image.save(out, dpi=(plan.canvas.dpi, plan.canvas.dpi))
    logger.info("Poster written to %s (%dx%d)", out, image.width, image.height)
    return out
