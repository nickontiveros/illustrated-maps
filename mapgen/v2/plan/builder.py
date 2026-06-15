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

import logging
import random

from ..assets.manifest import build_manifest, poi_asset_id
from ..ingest import GeoFrame, SourceData
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
from .placement import (
    assign_leader_lines,
    resolve_poi_overlaps,
    sized_slot,
    tier_demand,
)
from .stylize import prune_roads, road_width_px, simplify_polyline, stylize_polyline

logger = logging.getLogger(__name__)

SCATTER_FOR_GROUND = {
    GroundClass.PARK: ScatterKind.TREE,
    GroundClass.FOREST: ScatterKind.TREE,
    GroundClass.WATER: ScatterKind.BOAT,  # overridden by style.water_scatter
    GroundClass.URBAN: ScatterKind.HOUSE,
}

SCATTER_DENSITY = {  # items per megapixel of polygon area at full scale
    ScatterKind.TREE: 28.0,
    ScatterKind.BOAT: 1.2,
    ScatterKind.HOUSE: 10.0,
    ScatterKind.CACTUS: 28.0,
    ScatterKind.SHRUB: 28.0,
    ScatterKind.ROCK: 28.0,
}

# Bare-land scatter (style.land_scatter) is sparse set dressing, not fill:
# items per megapixel of *plate* area.
LAND_SCATTER_DENSITY = {
    ScatterKind.CACTUS: 2.2,
    ScatterKind.SHRUB: 2.4,
    ScatterKind.ROCK: 1.0,
    ScatterKind.TREE: 1.6,
    ScatterKind.HOUSE: 0.2,
}

# Sprite width on the poster as a fraction of canvas width (before depth).
# Set-dressing reads as deliberate illustration, not noise, only when the
# pieces are large enough to register against the POI sprites.
SCATTER_WIDTH_FRACTIONS = {
    ScatterKind.TREE: 0.026,
    ScatterKind.HOUSE: 0.032,
    ScatterKind.BOAT: 0.026,
    ScatterKind.CACTUS: 0.024,
    ScatterKind.SHRUB: 0.020,
    ScatterKind.ROCK: 0.028,
}


def _cluster_indices(centers: list[Point], link: float) -> list[list[int]]:
    """Single-linkage grouping: POIs within ``link`` (normalized) join a group.

    Each multi-POI group becomes one magnification plateau, so a metro cluster
    is zoomed as a single affine block (straight roads) rather than by a pile
    of per-POI bumps (wiggly roads)."""
    n = len(centers)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if ((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2) ** 0.5 < link:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _cluster_radius(centers: list[Point]) -> float:
    """Importance-warp sigma adapted to how tightly the POIs cluster.

    The default 0.18 was tuned for a handful of POIs spread over a city
    poster; for a metro cluster occupying a sliver of a state poster it
    smears the magnification over half the map. Use ~3x the median
    nearest-neighbor distance, clamped to a sane band."""
    if len(centers) < 2:
        return 0.18
    nn: list[float] = []
    for i, a in enumerate(centers):
        best = min(
            ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            for j, b in enumerate(centers)
            if i != j
        )
        nn.append(best)
    nn.sort()
    median = nn[len(nn) // 2]
    return min(0.18, max(0.05, 3.0 * median))


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

    def build(
        self,
        source: SourceData,
        title: str | None = None,
        frame: GeoFrame | None = None,
    ) -> PlanDocument:
        canvas = self.canvas
        cam = ObliqueCamera(self.camera_spec, canvas)
        rng = random.Random(self.seed)
        frame = frame or GeoFrame(source.region)

        def geo_norm(coord: tuple[float, float]) -> Point:
            return frame.to_normalized(coord)

        warp, sprite_scale, residual_px, coincident_count = self._fit_warp(
            source, cam, geo_norm
        )

        def to_flat(coords: list[tuple[float, float]]) -> list[Point]:
            normalized = [geo_norm(c) for c in coords]
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
        road_refs: list[str | None] = []
        for road in source.roads:
            pts = stylize_polyline(to_flat(road.coords), simplify_tol, densify_px=densify_px)
            if len(pts) >= 2:
                flat_roads.append((road.cls, pts))
                road_names.append(road.name)
                road_refs.append(road.ref)
        name_of = {id(pts): name for (_, pts), name in zip(flat_roads, road_names)}
        ref_of = {id(pts): ref for (_, pts), ref in zip(flat_roads, road_refs)}
        kept = prune_roads(flat_roads, canvas.width_px, area_px=cam.flat_width * cam.flat_height)
        roads: list[RoadPath] = []
        for cls, pts in kept:
            mid_y = pts[len(pts) // 2][1]
            roads.append(
                RoadPath(
                    cls=cls,
                    points=cam.project_points(pts),
                    width_px=road_width_px(cls, canvas.width_px) * cam.scale_at(mid_y),
                    name=name_of.get(id(pts)),
                    ref=ref_of.get(id(pts)),
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
        # Linear features (rivers) are drawn as waterway ribbons with a water
        # label -- a river must never become a building sprite.
        slots: list[PoiSlot] = []
        river_labels: list[tuple[str, list[Point]]] = []
        for poi in source.pois:
            if poi.feature_type == "river":
                if poi.path and len(poi.path) >= 2:
                    pts = stylize_polyline(
                        to_flat([(lon, lat) for lat, lon in poi.path]),
                        simplify_tol,
                        densify_px=densify_px,
                    )
                    if len(pts) >= 2:
                        mid_y = pts[len(pts) // 2][1]
                        projected = cam.project_points(pts)
                        roads.append(
                            RoadPath(
                                cls=RoadClass.RIVER,
                                points=projected,
                                width_px=road_width_px(RoadClass.RIVER, canvas.width_px)
                                * cam.scale_at(mid_y),
                                name=poi.name,
                                depth=cam.depth(mid_y),
                            )
                        )
                        river_labels.append((poi.name, projected))
                        continue
                logger.warning(
                    "River POI %r has no usable path; labeling its point instead of drawing a ribbon",
                    poi.name,
                )
                normalized = warp.warp_point(
                    geo_norm((poi.longitude, poi.latitude))
                )
                flat = (normalized[0] * cam.flat_width, normalized[1] * cam.flat_height)
                river_labels.append((poi.name, [cam.project_point(flat)]))
                continue
            normalized = warp.warp_point(geo_norm((poi.longitude, poi.latitude)))
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
                feature_type=poi.feature_type,
                # Remember the true (warped) ground point before the overlap
                # solver displaces the sprite; assign_leader_lines decides which
                # slots keep it as a leader-line target.
                leader_anchor=anchor,
            )
            slots.append(sized_slot(slot, canvas.width_px, cam.scale_at(flat[1]), sprite_scale))
        title_box = self._title_box(canvas)
        slots = resolve_poi_overlaps(
            slots, canvas.width_px, canvas.height_px, reserved=[title_box]
        )
        slots = assign_leader_lines(
            slots, leader_threshold_px=self.MAX_RESIDUAL_FRACTION * canvas.width_px
        )
        # A sprite dropped below the title cartouche must not also sprout a
        # leader back up into it (the true dot would hide under the banner) --
        # keep it where it landed, without a connector.
        tx0, ty0, tx1, ty1 = title_box
        for s in slots:
            if s.offset and s.leader_anchor is not None:
                lx, ly = s.leader_anchor
                if tx0 <= lx <= tx1 and ty0 <= ly <= ty1:
                    s.offset, s.leader_anchor = False, None

        # --- Scatter (sprite fabric) ---
        scatter = self._scatter(ground, rng, canvas, cam)

        # --- Labels ---
        districts = []
        water_names: list[tuple[str, list[Point]]] = list(river_labels)
        for place in source.places:
            normalized = warp.warp_point(
                geo_norm((place.longitude, place.latitude))
            )
            flat = (normalized[0] * cam.flat_width, normalized[1] * cam.flat_height)
            pos = cam.project_point(flat)
            if place.kind == "water":
                water_names.append((place.name, [pos]))
            else:
                districts.append((place.name, pos, place.population))
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
        manifest = build_manifest(ground_classes, scatter_kinds, slots, style=self.style)
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
            provenance={
                "warp_fit": {
                    "sprite_scale": round(sprite_scale, 3),
                    "residual_px": round(residual_px, 1),
                    # residual_ok now means "every POI the warp was responsible
                    # for separating fit"; coincident POIs get leader lines and
                    # are reported separately, not as a failure.
                    "residual_ok": residual_px
                    <= self.MAX_RESIDUAL_FRACTION * canvas.width_px,
                    "coincident_count": coincident_count,
                }
            },
        )

    # Overlap-solver residual (max nudge off the geographic anchor) the plan
    # accepts for POIs the warp could separate, as a fraction of poster width.
    MAX_RESIDUAL_FRACTION = 0.015
    MIN_SPRITE_SCALE = 0.65
    # Numeric safety ceiling on the global warp gain.
    MAX_WARP_GAIN = 64.0
    MIN_WARP_RADIUS = 0.012  # narrowest per-POI sigma (resolved at 1024 samples)

    def _title_box(self, canvas: CanvasSpec) -> tuple[float, float, float, float]:
        """Top-center keep-out box matching the compositor's title cartouche."""
        w, h = canvas.width_px, canvas.height_px
        cw = max(0.055 * w * 7, w * 0.40)  # cartouche width (banner, generous)
        return (w / 2 - cw / 2, 0.0, w / 2 + cw / 2, h * 0.12)

    def _poi_radius(self, i: int, centers: list[Point], base_radius: float) -> float:
        """Per-POI Gaussian sigma: tight neighbours -> narrow, local bump.

        Concentrating each POI's magnification near it (rather than the single
        smeared ``base_radius``) lets a dense cluster inflate without dragging
        the empty desert along with it."""
        c = centers[i]
        nn = min(
            (((c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2) ** 0.5 for j, o in enumerate(centers) if j != i),
            default=base_radius,
        )
        return min(base_radius, max(self.MIN_WARP_RADIUS, 2.0 * nn))

    def _fit_warp(self, source: SourceData, cam: ObliqueCamera, geo_norm):
        """Demand-driven separable cartogram fit.

        Each POI contributes a Gaussian density bump weighted by how much room
        its sprite needs (tier) and narrowed to how tightly it clusters, so
        dense areas inflate and empty stretches compress. A global gain is
        raised (then sprites shrunk) to minimise the POIs the warp cannot place
        honestly; any sprite still more than the residual threshold from its
        true point gets a leader line. Returns (warp, sprite_scale, residual_px,
        coincident_count[=leader count]).
        """
        sprite_pois = [p for p in source.pois if p.feature_type != "river"]
        if not sprite_pois or self.distortion_strength <= 0:
            return IdentityWarp(), 1.0, 0.0, 0

        centers = [geo_norm((p.longitude, p.latitude)) for p in sprite_pois]
        base_radius = _cluster_radius(centers)
        base_demand = tier_demand(2)
        weights = [tier_demand(p.tier) / base_demand for p in sprite_pois]
        radii = [self._poi_radius(i, centers, base_radius) for i in range(len(centers))]

        threshold = self.MAX_RESIDUAL_FRACTION * self.canvas.width_px
        gain = self.distortion_strength
        sprite_scale = 1.0
        # Search the warp schedule (strengthen, then shrink sprites) and keep
        # the config needing the FEWEST leader lines, then the gentlest warp. A
        # POI is leadered when, after the warp and overlap solving, its sprite
        # still sits more than `threshold` from its true point: the warp could
        # not place it honestly, so a connector makes the offset explicit.
        best: tuple | None = None
        for _ in range(14):
            warp = ImportanceWarp(
                centers=centers,
                strength=gain,
                radius=base_radius,
                weights=weights,
                radii=radii,
            )
            slots = []
            for poi, center in zip(sprite_pois, centers):
                u, v = warp.warp_point(center)
                flat = (u * cam.flat_width, v * cam.flat_height)
                slot = PoiSlot(
                    id=poi.id,
                    name=poi.name,
                    anchor=cam.project_point(flat),
                    width_px=0,
                    height_px=0,
                    tier=poi.tier,
                    feature_type=poi.feature_type,
                )
                slots.append(
                    sized_slot(slot, self.canvas.width_px, cam.scale_at(flat[1]), sprite_scale)
                )
            resolved = resolve_poi_overlaps(slots, self.canvas.width_px, self.canvas.height_px)
            displaced = [
                ((s.anchor[0] - r.anchor[0]) ** 2 + (s.anchor[1] - r.anchor[1]) ** 2) ** 0.5
                for s, r in zip(slots, resolved)
            ]
            leader_count = sum(1 for d in displaced if d > threshold)
            residual = max((d for d in displaced if d <= threshold), default=0.0)
            key = (leader_count, gain)  # fewest leaders, then gentlest warp
            if best is None or key < best[0]:
                best = (key, warp, sprite_scale, residual, leader_count, gain)
            if leader_count == 0:
                break
            if gain < self.MAX_WARP_GAIN:
                gain = min(self.MAX_WARP_GAIN, gain * 1.6)
            elif sprite_scale > self.MIN_SPRITE_SCALE:
                sprite_scale = max(self.MIN_SPRITE_SCALE, sprite_scale * 0.85)
            else:
                break
        _, warp, sprite_scale, residual, coincident_count, gain = best
        logger.info(
            "Warp fit: gain %.2f, sprite scale %.2f, residual %.0fpx, %d leader POIs",
            gain,
            sprite_scale,
            residual,
            coincident_count,
        )
        return warp, sprite_scale, residual, coincident_count

    def _scatter(
        self,
        ground: list[GroundPolygon],
        rng: random.Random,
        canvas: CanvasSpec,
        cam: ObliqueCamera,
    ) -> list[ScatterSlot]:
        """Seeded random scatter inside ground polygons, by class, plus
        sparse style-driven set dressing on the bare land plate."""
        from .geometry import point_in_polygon, polygon_area, polygon_bounds

        slots: list[ScatterSlot] = []
        for poly in ground:
            kind = SCATTER_FOR_GROUND.get(poly.cls)
            if poly.cls is GroundClass.WATER:
                # Style decides what (if anything) floats on open water.
                kind = (
                    ScatterKind(self.style.water_scatter)
                    if self.style.water_scatter in ScatterKind._value2member_map_
                    else None
                )
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

        slots.extend(self._land_scatter(ground, rng, canvas, cam))
        return slots

    def _land_scatter(
        self,
        ground: list[GroundPolygon],
        rng: random.Random,
        canvas: CanvasSpec,
        cam: ObliqueCamera,
    ) -> list[ScatterSlot]:
        """Sparse style scatter (cacti, shrubs, rocks...) on bare base land:
        anywhere on the plate trapezoid not covered by a ground polygon."""
        from .geometry import point_in_polygon, polygon_bounds

        kinds = [
            ScatterKind(k)
            for k in self.style.land_scatter
            if k in ScatterKind._value2member_map_
        ]
        if not kinds:
            return []

        horizon = cam.horizon_px
        conv = self.camera_spec.convergence
        plate_area_mp = (
            canvas.width_px * (1 + conv) / 2 * (canvas.height_px - horizon)
        ) / 1e6
        obstacles = [
            (polygon_bounds(p.exterior), p)
            for p in ground
            if len(p.exterior) >= 3
        ]

        def on_bare_land(pt: Point) -> bool:
            for (x0, y0, x1, y1), poly in obstacles:
                if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1 and point_in_polygon(pt, poly.exterior):
                    # Inside a hole means this polygon doesn't cover the
                    # point; another polygon still might.
                    if not any(point_in_polygon(pt, hole) for hole in poly.holes):
                        return False
            return True

        slots: list[ScatterSlot] = []
        cx = canvas.width_px / 2
        for kind in kinds:
            # At least one of each requested kind, even on small canvases
            # where density x area rounds to zero.
            count = max(1, int(round(LAND_SCATTER_DENSITY.get(kind, 0.5) * plate_area_mp)))
            placed = 0
            attempts = 0
            while placed < count and attempts < count * 20:
                attempts += 1
                y = rng.uniform(horizon, canvas.height_px)
                t = cam.t_at_poster_y(y)
                half = canvas.width_px / 2 * (conv + (1 - conv) * t)
                pt = (rng.uniform(cx - half, cx + half), y)
                if not on_bare_land(pt):
                    continue
                depth = 1.0 - t
                slots.append(
                    ScatterSlot(
                        kind=kind,
                        x=pt[0],
                        y=pt[1],
                        width_px=SCATTER_WIDTH_FRACTIONS[kind]
                        * canvas.width_px
                        * (1.0 - 0.45 * depth)
                        * rng.uniform(0.8, 1.25),
                        depth=depth,
                        variant=rng.randrange(6),
                    )
                )
                placed += 1
        return slots
