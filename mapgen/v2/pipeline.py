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
from typing import Literal, Optional, Union

import yaml
from PIL import Image
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

from .assets.manifest import slugify
from .assets.studio import AssetGenerator, AssetStudio
from .compose import Compositor

if TYPE_CHECKING:
    from .compose.harmonize import MoodPass
    from .compose_spec import CompositionSpec
from .ingest import SourceData, SourcePoi
from .plan import PlanBuilder, plan_to_svg
from .types import CameraSpec, CanvasSpec, PlanDocument, RegionBBox, StyleSpec

logger = logging.getLogger(__name__)

PLAN_FILENAME = "plan.json"
PREVIEW_FILENAME = "preview.svg"
ASSETS_DIRNAME = "assets"
POSTER_FILENAME = "poster.png"
POSTER_BASE_FILENAME = "poster_base.png"
REPAINT_DIRNAME = "repaint"
STYLE_BIBLE_FILENAME = "style_bible.png"

# Rough Gemini image-call price, for dry-run cost estimates only.
REPAINT_COST_PER_CALL = 0.13


class PoiConfig(BaseModel):
    name: str
    lat: float
    lon: float
    tier: int = Field(2, ge=1, le=3)
    photo: Optional[str] = None
    # building | campus | park | garden | forest | river | mountain |
    # monument | airport | stadium | area ... free-form; unknown values fall
    # back to building treatment.
    feature_type: str = "building"
    # (lat, lon) polyline for linear features (feature_type "river").
    path: Optional[list[tuple[float, float]]] = None


class V2Project(BaseModel):
    name: str
    # Poster title; defaults to the project name. Editable after planning via
    # retitle_project (no OSM refetch needed).
    title: Optional[str] = None
    region: RegionBBox
    output: CanvasSpec = Field(default_factory=CanvasSpec)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    style: StyleSpec = Field(default_factory=StyleSpec)
    distortion_strength: float = 0.5
    seed: int = 7
    # Road rendering: "full" warps the whole network; "minimal" keeps only key
    # interstates/US routes + major rivers, drawn straight (good for warped
    # regional posters until the composition editor exists).
    road_treatment: str = "full"
    # Compass bearing of poster-up: 0 = north-up, 340 = tilted toward NW.
    # "auto" picks the bearing that best fits the POI cloud to the canvas.
    rotation_deg: Union[float, Literal["auto"]] = 0.0
    pois: list[PoiConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str) -> "V2Project":
        from .styles import resolve_style

        data = yaml.safe_load(Path(path).read_text())
        if "style" in data:
            data["style"] = resolve_style(data["style"])
        return cls.model_validate(data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @property
    def display_title(self) -> str:
        return (self.title or self.name).strip() or self.name


# Region area (km^2) thresholds for the OSM detail tiers. The "full"
# everything-fetch is only safe for city-scale regions; above that Overpass
# either rate-limits or buries the plan in features (see V2_GENERALIZATION.md).
DETAIL_TIERS = [
    (150.0, "full"),
    (2_500.0, "simplified"),
    (200_000.0, "regional"),
    (float("inf"), "country"),
]

# Layers each tier is expected to populate -- used to tell "tier doesn't
# fetch this" apart from "fetch failed" in provenance, and to retry failed
# layers individually (Overpass rate-limits back-to-back area queries; one
# refused layer must not silently cost its data).
TIER_EXTRACTORS = {
    "full": {
        "roads": "extract_roads",
        "buildings": "extract_buildings",
        "water": "extract_water",
        "parks": "extract_parks",
        "terrain_types": "extract_terrain_types",
        "railways": "extract_railways",
        "washes": "extract_washes",
    },
    "simplified": {
        "roads": "extract_major_roads",
        "buildings": "extract_notable_buildings",
        "water": "extract_water",
        "parks": "extract_parks",
        "washes": "extract_washes",
    },
    "regional": {
        "roads": "extract_primary_roads",
        "buildings": "extract_landmark_buildings",
        "water": "extract_major_water",
        "parks": "extract_major_parks",
        "washes": "extract_washes",
    },
    "country": {
        "roads": "extract_motorways_only",
        "water": "extract_coastline_and_major_rivers",
    },
}
TIER_LAYERS = {tier: list(layers) for tier, layers in TIER_EXTRACTORS.items()}

# Cool-down before retrying layers Overpass refused, and between retries.
OSM_RETRY_COOLDOWN_S = 60.0
OSM_RETRY_SPACING_S = 10.0


def make_frame(project: V2Project) -> "GeoFrame":
    """The single geo->canvas frame shared by the OSM fetch and the plan.

    Target aspect is the camera's flat-space h/w, so the mapped ground keeps
    true proportions and any mismatch with the canvas is absorbed by
    *extending* the covered region (never stretching, never letterboxing).
    """
    from .ingest import GeoFrame, auto_rotation
    from .plan.camera import ObliqueCamera

    cam = ObliqueCamera(project.camera, project.output)
    target_aspect = cam.flat_height / cam.flat_width
    rotation = project.rotation_deg
    if rotation == "auto":
        rotation = auto_rotation(
            [(p.lon, p.lat) for p in project.pois], project.region, target_aspect
        )
        logger.info("Auto rotation chose up-bearing %.0f deg", rotation)
    return GeoFrame(project.region, rotation_deg=float(rotation), target_aspect=target_aspect)


# Road classes worth editing on a poster. Washes/arroyos (STREAM) and, at
# regional+ scale, residential LOCAL roads are clutter -- excluded from the
# editor feed so the canvas stays light (a state extract has ~64k washes).
_EDITABLE_ROADS_BASE = {"motorway", "primary", "secondary", "river", "rail"}
_EDITOR_ROAD_CAP = 3000  # safety cap on roads served (longest kept)


def _decimate(pts: list, max_pts: int = 40) -> list:
    """Thin a polyline for display (endpoints always kept)."""
    if len(pts) <= max_pts:
        return pts
    step = (len(pts) - 1) / (max_pts - 1)
    idx = sorted({round(i * step) for i in range(max_pts)} | {len(pts) - 1})
    return [pts[i] for i in idx]


def source_to_normalized(source: SourceData, frame: "GeoFrame") -> dict:
    """Project the *editable* source features into normalized frame space
    (0..1) for the layout editor: warp-independent coordinates with stable ids,
    simplified and filtered to the features a user actually composes (not the
    full OSM network). ``counts`` reports what was shown vs. total per layer so
    nothing is silently dropped."""

    def nm(coord):
        u, v = frame.to_normalized((coord[0], coord[1]))
        return [round(u, 5), round(v, 5)]

    detail = source.provenance.get("detail_level", "full")
    editable = set(_EDITABLE_ROADS_BASE)
    if detail not in ("regional", "country"):
        editable.add("local")  # city-scale posters do compose on local streets

    candidates = [r for r in source.roads if r.cls.value in editable]
    roads_total = len(source.roads)
    capped = len(candidates) > _EDITOR_ROAD_CAP
    if capped:
        candidates = sorted(candidates, key=lambda r: -len(r.coords))[:_EDITOR_ROAD_CAP]

    roads = [
        {
            "id": r.id,
            "cls": r.cls.value,
            "name": r.name,
            "ref": r.ref,
            "points": [nm(c) for c in _decimate(r.coords)],
        }
        for r in candidates
    ]
    ground = [
        {
            "id": g.id,
            "cls": g.cls.value,
            "name": g.name,
            "exterior": [nm(c) for c in _decimate(g.exterior, 60)],
        }
        for g in source.ground
    ]
    pois = [
        {
            "id": p.id,
            "name": p.name,
            "tier": p.tier,
            "feature_type": p.feature_type,
            "point": nm((p.longitude, p.latitude)),
        }
        for p in source.pois
    ]
    places = [
        {"id": p.id, "name": p.name, "kind": p.kind, "point": nm((p.longitude, p.latitude))}
        for p in source.places
    ]
    return {
        "frame": frame.to_dict(),
        "roads": roads,
        "ground": ground,
        "pois": pois,
        "places": places,
        "counts": {
            "roads_shown": len(roads),
            "roads_total": roads_total,
            "roads_capped": capped,
            "ground": len(ground),
            "pois": len(pois),
            "places": len(places),
        },
    }


# OSM class/type -> our POI feature_type (drives sprite sizing + landform hints).
_GEOCODE_FEATURE_TYPE = {
    ("aeroway", "aerodrome"): "airport",
    ("amenity", "university"): "campus",
    ("amenity", "college"): "campus",
    ("leisure", "stadium"): "stadium",
    ("leisure", "park"): "park",
    ("leisure", "nature_reserve"): "park",
    ("boundary", "national_park"): "park",
    ("natural", "peak"): "mountain",
    ("natural", "wood"): "forest",
    ("landuse", "forest"): "forest",
    ("tourism", "zoo"): "zoo",
}


def geocode_place(query: str) -> dict:
    """Look up a place by name (Nominatim via osmnx, HTTP-cached) and return a
    POI-ready record: coordinates + a best-guess feature_type. Raises on miss."""
    import osmnx as ox

    gdf = ox.geocoder.geocode_to_gdf(query)  # raises if nothing matches
    row = gdf.iloc[0]
    centroid = row.geometry.centroid
    cls, typ = str(row.get("class", "")), str(row.get("type", ""))
    feature_type = _GEOCODE_FEATURE_TYPE.get((cls, typ), "building")
    dn = row.get("display_name")
    return {
        "query": query,
        "display_name": str(dn) if dn is not None else None,  # osmnx returns numpy types
        "lat": float(centroid.y),
        "lon": float(centroid.x),
        "feature_type": feature_type,
        "osm_class": cls,
        "osm_type": typ,
    }


def detail_level_for(region: RegionBBox) -> str:
    area = region.area_km2
    for limit, tier in DETAIL_TIERS:
        if area <= limit:
            return tier
    return "country"


def fetch_source(project: V2Project, cache_dir: Optional[Path] = None) -> SourceData:
    """Fetch live OSM data for the project region (requires osmnx).

    The detail tier is chosen from the region's physical area, and every
    layer's outcome is recorded in ``source.provenance`` so downstream
    stages (and the user) can see when a layer failed or came back empty
    instead of silently rendering a map without it.
    """
    from mapgen.models.project import BoundingBox  # V1 bbox for OSMService
    from mapgen.services.osm_service import OSMService

    from .ingest import from_osm_data

    # Fetch the frame's envelope, not the raw bbox: rotation and aspect
    # extension widen the ground the poster actually shows.
    fetch_region = make_frame(project).fetch_region
    bbox = BoundingBox(
        north=fetch_region.north,
        south=fetch_region.south,
        east=fetch_region.east,
        west=fetch_region.west,
    )
    osm = OSMService(cache_dir=str(cache_dir) if cache_dir else None)
    detail = detail_level_for(fetch_region)
    area = fetch_region.area_km2
    logger.info("Frame envelope is %.0f km^2 -> OSM detail tier %r", area, detail)
    data = osm.fetch_region_data(bbox, detail_level=detail)

    # Inland rivers are waterway *lines*; no tier fetches them. Skip only at
    # country scale, where extract_coastline_and_major_rivers already does.
    if detail != "country":
        data.rivers = osm.extract_rivers(bbox)

    # Overpass rate-limits back-to-back big-area queries (later layers come
    # back connection-refused). Retry failed layers individually after a
    # cool-down instead of shipping a plan without them.
    failed = [
        layer for layer in TIER_EXTRACTORS[detail] if getattr(data, layer, None) is None
    ]
    if detail != "country" and getattr(data, "rivers", None) is None:
        failed.append("rivers")
    if failed:
        import time as _time

        logger.warning(
            "OSM layers failed (%s); retrying after a %.0fs cool-down",
            ", ".join(failed),
            OSM_RETRY_COOLDOWN_S,
        )
        _time.sleep(OSM_RETRY_COOLDOWN_S)
        for layer in failed:
            method = (
                "extract_rivers" if layer == "rivers" else TIER_EXTRACTORS[detail][layer]
            )
            setattr(data, layer, getattr(osm, method)(bbox))
            _time.sleep(OSM_RETRY_SPACING_S)

    source = from_osm_data(data, project.region)

    # City/town names matter once the region spans more than one city.
    cities = None
    if area >= 500.0:
        cities = osm.extract_major_cities(bbox)
        if cities is not None:
            from .ingest import SourcePlace

            for _, row in cities.iterrows():
                name = row.get("display_name") or row.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                c = row.geometry.centroid
                pop = row.get("population")
                try:
                    pop = float(pop) if pop is not None and str(pop) != "nan" else None
                except (TypeError, ValueError):
                    pop = None
                source.places.append(
                    SourcePlace(
                        name=name.strip(), latitude=c.y, longitude=c.x,
                        kind="district", population=pop,
                    )
                )

    def _outcome(gdf) -> object:
        if gdf is None:
            return "failed"
        return int(len(gdf)) or "empty"

    layers = {name: _outcome(getattr(data, name, None)) for name in TIER_LAYERS[detail]}
    if detail != "country":
        layers["rivers"] = _outcome(getattr(data, "rivers", None))
    if area >= 500.0:
        layers["places"] = _outcome(cities)
    source.provenance = {
        "detail_level": detail,
        "region_area_km2": round(area, 1),
        "layers": layers,
    }
    # Drop sub-scale clutter (tiny tanks, wash spaghetti) and stitch the
    # fragmented road ways into continuous polylines before planning; then
    # synthesize urban patches so cities read as built-up areas.
    from .generalize import add_urban_areas, generalize_source
    from .ingest import assign_feature_ids

    generalize_source(source, fetch_region)
    add_urban_areas(source)
    assign_feature_ids(source)  # stable ids for the editor / composition spec
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
            feature_type=p.feature_type,
            path=[(lat, lon) for lat, lon in p.path] if p.path else None,
        )
        for p in project.pois
    ]


def plan_warnings(source: SourceData, plan: PlanDocument) -> list[str]:
    """Human-readable problems with the fetched data, for CLI/UI display."""
    warnings: list[str] = []
    for layer, outcome in source.provenance.get("layers", {}).items():
        if outcome == "failed":
            warnings.append(
                f"OSM {layer} fetch FAILED (likely Overpass rate-limit/timeout); "
                "the poster will render without this layer. Re-run `plan` to retry."
            )
        elif outcome == "empty":
            warnings.append(f"OSM {layer} returned no features for this region.")
    if not plan.ground:
        warnings.append(
            "Plan has no ground polygons (no water/parks/terrain): the map will be "
            "bare base land."
        )
    warp_fit = plan.provenance.get("warp_fit", {})
    coincident = warp_fit.get("coincident_count", 0)
    if coincident:
        # Expected for tight metro clusters -- informational, not a failure.
        warnings.append(
            f"{coincident} POI(s) sit within a sprite's footprint of a neighbour; "
            "they are drawn offset with a leader line to their true location."
        )
    if warp_fit and not warp_fit.get("residual_ok", True):
        warnings.append(
            f"POI cluster too dense even after warp/sprite-size tuning "
            f"(separable sprites sit up to {warp_fit.get('residual_px')}px off their "
            "anchors); consider fewer POIs or a smaller region."
        )
    return warnings


def build_plan(
    project: V2Project,
    source: SourceData,
    spec: "CompositionSpec | None" = None,
) -> PlanDocument:
    _attach_pois(source, project)
    frame = make_frame(project)
    builder = PlanBuilder(
        canvas=project.output,
        camera=project.camera,
        style=project.style,
        distortion_strength=project.distortion_strength,
        seed=project.seed,
        road_treatment=project.road_treatment,
        spec=spec,
    )
    plan = builder.build(source, title=project.display_title, frame=frame)
    plan.frame = frame.to_dict()
    plan.provenance.update(source.provenance)
    plan.warnings = plan_warnings(source, plan)
    for message in plan.warnings:
        logger.warning("%s", message)
    return plan


def retitle_project(project_dir: Path, new_title: str) -> bool:
    """Change the poster title without re-planning.

    Updates project.yaml, then surgically patches the existing plan.json
    (document name + TITLE label text) and regenerates the preview. No OSM
    fetch, no asset spend; only a recompose is needed afterwards. Returns
    True when a plan existed and was patched.
    """
    from .plan import plan_to_svg
    from .types import LabelKind

    new_title = new_title.strip()
    if not new_title:
        raise ValueError("Title must not be empty")
    project_dir = Path(project_dir)
    project = V2Project.load(project_dir / "project.yaml")
    project.title = new_title
    project.save(project_dir / "project.yaml")

    plan_path = project_dir / PLAN_FILENAME
    if not plan_path.exists():
        return False
    plan = PlanDocument.load(plan_path)
    plan.name = new_title
    for label in plan.labels:
        if label.kind == LabelKind.TITLE:
            label.text = new_title
    # Save after project.yaml so the plan stays newer (not flagged stale).
    plan.save(plan_path)
    (project_dir / PREVIEW_FILENAME).write_text(plan_to_svg(plan))
    return True


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


def _repaint_engine(
    plan: PlanDocument,
    project_dir: Path,
    painter,
    compositor: Compositor,
    repaint_scale: float,
    progress=None,
):
    """Shared setup for repaint planning and execution.

    The quadrant store is keyed to one repaint scale; changing the scale
    invalidates it (cells would no longer map to the same pixels), so the
    store is cleared and the run starts over.
    """
    import json
    import shutil

    from .repaint import RepaintEngine, RepaintStore
    from .types import GroundClass

    repaint_dir = project_dir / REPAINT_DIRNAME
    meta_path = repaint_dir / "meta.json"
    meta = {"repaint_scale": repaint_scale}
    if repaint_dir.exists():
        try:
            stale = json.loads(meta_path.read_text()) != meta
        except (OSError, ValueError):
            stale = True
        if stale:
            logger.info("Repaint scale changed; clearing %s", repaint_dir)
            shutil.rmtree(repaint_dir)
    repaint_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))

    water_mask = compositor.render_class_mask({GroundClass.WATER}, scale=repaint_scale)
    style_bible = None
    bible_path = project_dir / ASSETS_DIRNAME / STYLE_BIBLE_FILENAME
    if bible_path.exists():
        style_bible = Image.open(bible_path).convert("RGB")

    store = RepaintStore(repaint_dir)
    engine = RepaintEngine(
        painter=painter,
        store=store,
        style=plan.style,
        style_bible=style_bible,
        water_mask=water_mask,
        raw_dir=repaint_dir / "raw",
        progress=progress,
    )
    return engine, water_mask


def texture_poster(
    plan: PlanDocument,
    project_dir: Path,
    painter,
    scale: float = 1.0,
    strength: float = 1.0,
    out_path: Optional[Path] = None,
    progress=None,
) -> Path:
    """Default repaint: single-call whole-base texture pass + native finish.

    Renders the below-labels base at the output scale, repaints it in ONE
    img2img call (no joints, no cross-window drift by construction), blends
    only the repaint's low/mid-frequency texture over the native render,
    repairs water color, then composites labels/frame/grain. The previous
    poster is kept as poster_base.png for A/B. One call, ~$0.13.

    Raises StructureRejection if the model abandoned the map content
    (hallucination guard) -- nothing is written; re-run to retry.
    """
    from .repaint.color_norm import unify_water
    from .repaint.texture_pass import texture_repaint
    from .types import GroundClass, hex_to_rgb

    compositor = Compositor(plan, project_dir / ASSETS_DIRNAME)
    if progress:
        progress(0, 3, "rendering base")
    base = compositor.render_base(scale=scale).convert("RGB")
    water_mask = compositor.render_class_mask({GroundClass.WATER}, scale=scale)

    style_bible = None
    bible_path = project_dir / ASSETS_DIRNAME / STYLE_BIBLE_FILENAME
    if bible_path.exists():
        style_bible = Image.open(bible_path).convert("RGB")

    raw_dir = project_dir / REPAINT_DIRNAME / "raw"

    def raw_sink(name: str, img: Image.Image) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        img.save(raw_dir / f"{name}.png")

    if progress:
        progress(1, 3, "texture pass (1 call)")
    image = texture_repaint(
        base,
        painter,
        plan.style,
        style_bible=style_bible,
        strength=strength,
        water_mask=water_mask,
        raw_sink=raw_sink,
    )
    image = unify_water(image, water_mask, hex_to_rgb(plan.style.palette["water"]))

    if progress:
        progress(2, 3, "finishing")
    image = compositor.apply_finish(image, scale=scale)

    out = out_path or (project_dir / POSTER_FILENAME)
    if out.exists():
        out.replace(out.with_name(POSTER_BASE_FILENAME))
    image.save(out, dpi=(plan.canvas.dpi, plan.canvas.dpi))
    logger.info("Texture-pass poster written to %s (%dx%d)", out, image.width, image.height)
    return out


def plan_repaint(
    plan: PlanDocument,
    project_dir: Path,
    repaint_scale: float = 0.5,
) -> dict:
    """Dry run for the TILED mode: call count and rough cost. (The default
    single-call mode is always 1 call; no planning needed.)"""
    from .repaint import IdentityPainter

    compositor = Compositor(plan, project_dir / ASSETS_DIRNAME)
    guide = compositor.render_base(scale=repaint_scale).convert("RGB")
    engine, _ = _repaint_engine(plan, project_dir, IdentityPainter(), compositor, repaint_scale)
    grid, selections = engine.plan(guide)
    return {
        "repaint_scale": repaint_scale,
        "grid": {"cols": grid.cols, "rows": grid.rows},
        "calls_planned": len(selections),
        "estimated_cost_usd": round(len(selections) * REPAINT_COST_PER_CALL, 2),
    }


def repaint_poster(
    plan: PlanDocument,
    project_dir: Path,
    painter,
    scale: float = 1.0,
    repaint_scale: float = 0.5,
    max_calls: Optional[int] = None,
    out_path: Optional[Path] = None,
    progress=None,
) -> tuple[Path, "object"]:
    """TILED AI repaint of the poster base (experimental; not the default).

    Window-by-window infill with painted-neighbor context. Live spikes
    showed zero-shot models mutate the "keep identical" context pixels, so
    joints misalign and style drifts between windows -- this path is kept
    for a future fine-tuned exact-infill painter (see texture_poster for
    the default single-call mode). Resumable; `max_calls` is a hard budget.
    """
    from .repaint.color_norm import match_low_frequency, unify_water
    from .types import hex_to_rgb

    compositor = Compositor(plan, project_dir / ASSETS_DIRNAME)
    guide = compositor.render_base(scale=repaint_scale).convert("RGB")
    engine, water_mask = _repaint_engine(
        plan, project_dir, painter, compositor, repaint_scale, progress=progress
    )
    result = engine.run(guide, max_calls=max_calls)

    # Kill quadrant-scale tone patchwork (worst on flat desert/water), then
    # repair water color drift.
    image = match_low_frequency(result.image, guide)
    image = unify_water(image, water_mask, hex_to_rgb(plan.style.palette["water"]))

    target_w = max(1, int(round(plan.canvas.width_px * scale)))
    target_h = max(1, int(round(plan.canvas.height_px * scale)))
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    image = compositor.apply_finish(image, scale=scale)

    out = out_path or (project_dir / POSTER_FILENAME)
    if out.exists():
        out.replace(out.with_name(POSTER_BASE_FILENAME))
    image.save(out, dpi=(plan.canvas.dpi, plan.canvas.dpi))
    logger.info(
        "Repainted poster written to %s (%d calls, completed=%s)",
        out,
        result.calls_made,
        result.completed,
    )
    return out, result


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
