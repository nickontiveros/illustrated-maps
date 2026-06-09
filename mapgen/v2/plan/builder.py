"""PlanBuilder: SourceData -> PlanDocument.

Pipeline (all deterministic, all vector):
  1. geo -> normalized coords
  2. importance warp (expand space around POIs)
  3. normalized -> flat space pixels
  4. stylize (simplify, smooth, exaggerate widths, prune)
  5. POI sizing + overlap resolution (in poster space)
  6. oblique camera projection (vectors -> poster space, depth attached)
  7. scatter placement (trees in parks, boats on water, houses in urban)
  8. label plan
  9. asset manifest
"""

from __future__ import annotations

import random

from ..assets.manifest import build_manifest, poi_asset_id
from ..ingest import SourceData, geo_to_normalized
from ..types import (
    BuildingFootprint,
    CameraSpec,
    CanvasSpec,
    GroundClass,
    GroundPolygon,
    PlanDocument,
    Point,
    PoiSlot,
    RoadClass,
    RoadPath,
    ScatterKind,
    ScatterSlot,
    StyleSpec,
)
from .camera import ObliqueCamera
from .distortion import IdentityWarp, ImportanceWarp
from .labels import plan_labels
from .placement import resolve_poi_overlaps, sized_slot
from .stylize import prune_roads, road_width_px, simplify_polyline, stylize_polyline

SCATTER_FOR_GROUND = {
    GroundClass.PARK: ScatterKind.TREE,
    GroundClass.FOREST: ScatterKind.TREE,
    GroundClass.WATER: ScatterKind.BOAT,
    GroundClass.URBAN: ScatterKind.HOUSE,
}

SCATTER_DENSITY = {  # items per megapixel of polygon area at full scale
    ScatterKind.TREE: 28.0,
    ScatterKind.BOAT: 1.2,
    ScatterKind.HOUSE: 10.0,
}

# Sprite width on the poster as a fraction of canvas width (before depth).
SCATTER_WIDTH_FRACTIONS = {
    ScatterKind.TREE: 0.016,
    ScatterKind.HOUSE: 0.021,
    ScatterKind.BOAT: 0.022,
}


class PlanBuilder:
    def __init__(
        self,
        canvas: CanvasSpec | None = None,
        camera: CameraSpec | None = None,
        style: StyleSpec | None = None,
        distortion_strength: float = 0.5,
        seed: int = 7,
    ):
        self.canvas = canvas or CanvasSpec()
        self.camera_spec = camera or CameraSpec()
        self.style = style or StyleSpec()
        self.distortion_strength = distortion_strength
        self.seed = seed

    def build(self, source: SourceData, title: str | None = None) -> PlanDocument:
        canvas = self.canvas
        cam = ObliqueCamera(self.camera_spec, canvas)
        rng = random.Random(self.seed)

        warp = (
            ImportanceWarp(
                centers=[
                    geo_to_normalized((p.longitude, p.latitude), source.region)
                    for p in source.pois
                ],
                strength=self.distortion_strength,
            )
            if source.pois and self.distortion_strength > 0
            else IdentityWarp()
        )

        def to_flat(coords: list[tuple[float, float]]) -> list[Point]:
            normalized = [geo_to_normalized(c, source.region) for c in coords]
            warped = warp.warp_points(normalized)
            return [(u * cam.flat_width, v * cam.flat_height) for u, v in warped]

        simplify_tol = canvas.width_px * 0.0006
        densify_px = canvas.width_px * 0.02

        # --- Ground polygons ---
        ground: list[GroundPolygon] = []
        for poly in source.ground:
            flat_ext = stylize_polyline(
                to_flat(poly.exterior), simplify_tol, closed=True, densify_px=densify_px
            )
            mid_y = sum(p[1] for p in flat_ext) / max(1, len(flat_ext))
            ground.append(
                GroundPolygon(
                    cls=poly.cls,
                    exterior=cam.project_points(flat_ext),
                    holes=[
                        cam.project_points(
                            stylize_polyline(to_flat(h), simplify_tol, closed=True, densify_px=densify_px)
                        )
                        for h in poly.holes
                    ],
                    depth=cam.depth(mid_y),
                )
            )

        # --- Roads / waterways ---
        flat_roads: list[tuple[RoadClass, list[Point]]] = []
        road_names: list[str | None] = []
        for road in source.roads:
            pts = stylize_polyline(to_flat(road.coords), simplify_tol, densify_px=densify_px)
            if len(pts) >= 2:
                flat_roads.append((road.cls, pts))
                road_names.append(road.name)
        name_of = {id(pts): name for (_, pts), name in zip(flat_roads, road_names)}
        kept = prune_roads(flat_roads, canvas.width_px)
        roads: list[RoadPath] = []
        for cls, pts in kept:
            mid_y = pts[len(pts) // 2][1]
            roads.append(
                RoadPath(
                    cls=cls,
                    points=cam.project_points(pts),
                    width_px=road_width_px(cls, canvas.width_px) * cam.scale_at(mid_y),
                    name=name_of.get(id(pts)),
                    depth=cam.depth(mid_y),
                )
            )

        # --- Buildings (2.5D fabric) ---
        buildings: list[BuildingFootprint] = []
        base_height = canvas.width_px * 0.006
        for footprint in source.buildings:
            # Buildings keep their corners: simplify only, never smooth --
            # a rectangular footprint must stay rectangular.
            flat = simplify_polyline(to_flat(footprint), simplify_tol / 2)
            if len(flat) < 3:
                continue
            mid_y = sum(p[1] for p in flat) / len(flat)
            depth = cam.depth(mid_y)
            buildings.append(
                BuildingFootprint(
                    polygon=cam.project_points(flat),
                    height_px=base_height * cam.scale_at(mid_y) * rng.uniform(0.8, 1.6),
                    depth=depth,
                )
            )

        # --- POIs ---
        slots: list[PoiSlot] = []
        for poi in source.pois:
            normalized = warp.warp_point(geo_to_normalized((poi.longitude, poi.latitude), source.region))
            flat = (normalized[0] * cam.flat_width, normalized[1] * cam.flat_height)
            anchor = cam.project_point(flat)
            slot = PoiSlot(
                id=poi.id,
                name=poi.name,
                anchor=anchor,
                width_px=0,
                height_px=0,
                tier=poi.tier,
                depth=cam.depth(flat[1]),
                asset_id=poi_asset_id(poi.id),
                latitude=poi.latitude,
                longitude=poi.longitude,
            )
            slots.append(sized_slot(slot, canvas.width_px, cam.scale_at(flat[1])))
        slots = resolve_poi_overlaps(slots, canvas.width_px, canvas.height_px)

        # --- Scatter (sprite fabric) ---
        scatter = self._scatter(ground, rng, canvas)

        # --- Labels ---
        districts = []
        water_names = []
        for place in source.places:
            normalized = warp.warp_point(
                geo_to_normalized((place.longitude, place.latitude), source.region)
            )
            flat = (normalized[0] * cam.flat_width, normalized[1] * cam.flat_height)
            pos = cam.project_point(flat)
            (water_names if place.kind == "water" else districts).append((place.name, pos))
        labels = plan_labels(
            canvas,
            roads,
            slots,
            districts,
            water_names,
            title=title or "Untitled Map",
        )

        # --- Manifest ---
        ground_classes = {g.cls for g in ground} | {GroundClass.LAND}
        scatter_kinds = {s.kind for s in scatter}
        manifest = build_manifest(ground_classes, scatter_kinds, slots)
        photo_by_asset = {poi_asset_id(p.id): p.photo for p in source.pois if p.photo}
        for spec in manifest:
            if spec.id in photo_by_asset:
                spec.source_photo = photo_by_asset[spec.id]

        return PlanDocument(
            name=title or "Untitled Map",
            region=source.region,
            canvas=canvas,
            camera=self.camera_spec,
            style=self.style,
            ground=ground,
            roads=roads,
            buildings=buildings,
            pois=slots,
            scatter=scatter,
            labels=labels,
            manifest=manifest,
        )

    def _scatter(
        self,
        ground: list[GroundPolygon],
        rng: random.Random,
        canvas: CanvasSpec,
    ) -> list[ScatterSlot]:
        """Seeded random scatter inside ground polygons, by class."""
        from .geometry import point_in_polygon, polygon_area, polygon_bounds

        slots: list[ScatterSlot] = []
        for poly in ground:
            kind = SCATTER_FOR_GROUND.get(poly.cls)
            if kind is None or len(poly.exterior) < 3:
                continue
            area_mp = polygon_area(poly.exterior) / 1e6
            count = int(SCATTER_DENSITY[kind] * area_mp)
            if count == 0:
                continue
            x0, y0, x1, y1 = polygon_bounds(poly.exterior)
            placed = 0
            attempts = 0
            while placed < count and attempts < count * 20:
                attempts += 1
                p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                if not point_in_polygon(p, poly.exterior):
                    continue
                if any(point_in_polygon(p, hole) for hole in poly.holes):
                    continue
                depth_scale = 1.0 - 0.45 * poly.depth
                slots.append(
                    ScatterSlot(
                        kind=kind,
                        x=p[0],
                        y=p[1],
                        width_px=SCATTER_WIDTH_FRACTIONS[kind]
                        * canvas.width_px
                        * depth_scale
                        * rng.uniform(0.8, 1.25),
                        depth=poly.depth,
                        variant=rng.randrange(6),
                    )
                )
                placed += 1
        return slots
