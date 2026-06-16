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
import re

from ..assets.manifest import build_manifest, poi_asset_id
from ..compose_spec import CompositionSpec
from ..ingest import GeoFrame, SourceData, assign_feature_ids
from ..types import (
    BuildingFootprint,
    CameraSpec,
    CanvasSpec,
    GroundClass,
    GroundPolygon,
    LabelKind,
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


def _plateau_weight(band_width: float, target_fraction: float) -> float:
    """Plateau bump weight so a band of width ``band_width`` claims about
    ``target_fraction`` of the axis after the CDF remap (sharp-band estimate;
    soft edges spread it a little more)."""
    bw = min(0.6, max(0.01, band_width))
    f = min(0.7, max(0.0, target_fraction))
    d = f * (1.0 - bw) / (bw * (1.0 - f))  # required interior density ratio
    return max(0.0, d - 1.0)


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
        road_treatment: str = "full",
        spec: "CompositionSpec | None" = None,
    ):
        self.canvas = canvas or CanvasSpec()
        self.camera_spec = camera or CameraSpec()
        self.style = style or StyleSpec()
        self.distortion_strength = distortion_strength
        self.seed = seed
        self.road_treatment = road_treatment
        # The editable layout document. An all-auto spec (the default) reads
        # like no spec at all: every seam below falls back to its heuristic, so
        # the plan is byte-identical to the pre-spec pipeline.
        self.spec = spec or CompositionSpec()

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
        # Stable ids for feature selection / road routing (idempotent: the
        # pipeline already assigned them; hand-built test sources get them now).
        assign_feature_ids(source)
        source = self._select_features(source)

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
                    id=poly.id,
                )
            )

        # --- Roads / waterways ---
        roads = self._build_roads(
            source, cam, geo_norm, canvas, to_flat, simplify_tol, densify_px
        )
        # Bird's-eye posters shouldn't have roads bleeding off the edges into
        # the border ornament; clip every road to an inset of the canvas (a
        # road may split into several in-frame pieces).
        roads = self._clip_roads(roads, canvas)

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
            ov = self.spec.pois.get(poi.id)
            normalized = warp.warp_point(geo_norm((poi.longitude, poi.latitude)))
            if ov and ov.offset_uv:
                # A manual nudge in (warp-independent) normalized space.
                normalized = (normalized[0] + ov.offset_uv[0], normalized[1] + ov.offset_uv[1])
            flat = (normalized[0] * cam.flat_width, normalized[1] * cam.flat_height)
            anchor = cam.project_point(flat)
            slot = PoiSlot(
                id=poi.id,
                name=poi.name,
                anchor=anchor,
                width_px=0,
                height_px=0,
                tier=ov.tier if (ov and ov.tier) else poi.tier,
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
            size_mul = ov.size if (ov and ov.size) else 1.0
            slots.append(
                sized_slot(slot, canvas.width_px, cam.scale_at(flat[1]), sprite_scale * size_mul)
            )
        title_box = self._title_box(canvas)
        # True (pre-overlap) anchors, so a forced leader can point home even if
        # the heuristic would not have drawn one.
        true_anchor = {s.id: s.anchor for s in slots}
        slots = resolve_poi_overlaps(
            slots, canvas.width_px, canvas.height_px, reserved=[title_box]
        )
        slots = assign_leader_lines(
            slots,
            leader_threshold_px=self.MAX_RESIDUAL_FRACTION * canvas.width_px,
            canvas_width_px=canvas.width_px,
            canvas_height_px=canvas.height_px,
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
        # Per-POI leader overrides from the spec.
        for s in slots:
            ov = self.spec.pois.get(s.id)
            if not ov:
                continue
            if ov.leader == "suppress":
                s.offset, s.leader_anchor = False, None
            elif ov.leader == "force":
                s.leader_anchor = true_anchor.get(s.id, s.anchor)
                s.offset = True

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
        # Hand-placed label overrides. Each generated label is resolved back to
        # its *source feature id* (the editor keys overrides by that stable id,
        # not by the post-layout text, which is uppercased/reformatted). A
        # matched label is moved to the user's normalized anchor -- warped +
        # projected like everything else -- translating the whole baseline so a
        # curved road label keeps its shape.
        overrides = self.spec.labels.overrides
        if overrides:
            slot_by_name = {s.name: s.id for s in slots}
            place_by_upper = {p.name.upper(): p.id for p in source.places if p.kind == "district"}
            water_by_name = {p.name: p.id for p in source.places if p.kind == "water"}
            road_by_ref = {r.ref: r.id for r in roads if r.ref}
            road_by_name = {r.name: r.id for r in roads if r.name}

            def _source_id(lab):
                if lab.kind is LabelKind.POI:
                    return slot_by_name.get(lab.text)
                if lab.kind is LabelKind.DISTRICT:
                    return place_by_upper.get(lab.text)
                if lab.kind is LabelKind.WATER:
                    return water_by_name.get(lab.text)
                if lab.kind is LabelKind.SHIELD:
                    return road_by_ref.get(lab.text)
                if lab.kind is LabelKind.STREET:
                    return road_by_name.get(lab.text)
                return None

            for lab in labels:
                sid = _source_id(lab)
                uv = overrides.get(sid) if sid else None
                if not uv or not lab.baseline:
                    continue
                wu, wv = warp.warp_point((uv[0], uv[1]))
                tx, ty = cam.project_point((wu * cam.flat_width, wv * cam.flat_height))
                bx, by = lab.baseline[0]
                dx, dy = tx - bx, ty - by
                lab.baseline = [(p[0] + dx, p[1] + dy) for p in lab.baseline]

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
    MAX_WARP_GAIN = 64.0  # (legacy gaussian path)
    MIN_WARP_RADIUS = 0.012  # (legacy gaussian path)
    # Bounded magnification plateau: each dense cluster zooms uniformly (a
    # constant-density region => linear CDF => affine warp => straight interior
    # roads), capped in width and in the axis fraction it may claim so it can
    # never run away and eat the map.
    WARP_LINK_DIST = 0.07          # normalized distance grouping POIs into a cluster
    MAX_HALF_BAND = 0.09           # cap on a plateau's half-width per axis
    WARP_TARGET_FRACTION = 0.40    # axis fraction a single plateau may claim
    MIN_PLATEAU_EXTENT = 0.015     # clusters narrower than this stay unwarped

    # In the "minimal" treatment keep only *mainline* interstates and US routes
    # (the orientation skeleton) plus major rivers. A ref may carry several
    # concurrent designations ("US 180;AZ 64"); keep the road if ANY is a clean
    # mainline. Historic/business/spur variants ("US 80 Hist", "I 10 BUS") are
    # dropped -- they fragment and clutter without aiding orientation.
    _MAINLINE = re.compile(r"^(I|US)\s+\d+$")

    @classmethod
    def _is_mainline(cls, ref: str | None) -> bool:
        if not ref:
            return False
        return any(cls._MAINLINE.match(part.strip()) for part in str(ref).split(";"))

    def _select_features(self, source: SourceData) -> SourceData:
        """Drop features the spec excludes (or whose layer default hides them),
        before any layout work. An all-auto spec keeps everything."""
        from dataclasses import replace

        f = self.spec.features

        def road_ok(r) -> bool:
            sel = f.rivers if r.cls in (RoadClass.RIVER, RoadClass.STREAM) else f.roads
            return sel.visible(r.id, auto_visible=True)

        return replace(
            source,
            roads=[r for r in source.roads if road_ok(r)],
            pois=[p for p in source.pois if f.pois.visible(p.id, auto_visible=True)],
            places=[p for p in source.places if f.places.visible(p.id, auto_visible=True)],
        )

    def _road_treatment_for(self, road) -> tuple[str, list | None]:
        """Effective routing for one road: per-road spec override, else the
        global ``road_treatment``. Returns (treatment, reshape-polyline)."""
        ov = self.spec.roads.get(road.id)
        if ov is not None:
            return ov.treatment, ov.reshape
        if self.road_treatment == "minimal":
            # Mainline I/US routes + major rivers drawn straight; the rest hidden.
            if road.cls is RoadClass.RIVER or self._is_mainline(road.ref):
                return "straight", None
            return "hidden", None
        return "warped", None

    def _build_roads(
        self, source, cam, geo_norm, canvas, to_flat, simplify_tol, densify_px
    ) -> list["RoadPath"]:
        """Route every road by its effective treatment: warped roads go through
        the collective ink-budget prune; straight roads bypass the warp (and may
        follow a user-supplied reshape polyline); hidden roads are dropped."""
        warped_src = []
        straight: list[RoadPath] = []
        for road in source.roads:
            treat, reshape = self._road_treatment_for(road)
            if treat == "hidden":
                continue
            if treat == "straight":
                rp = self._straight_road(road, cam, geo_norm, canvas, reshape)
                if rp is not None:
                    straight.append(rp)
            else:  # "warped"
                warped_src.append(road)
        warped = self._warped_roads(
            warped_src, cam, canvas, to_flat, simplify_tol, densify_px
        )
        return warped + straight

    def _straight_road(
        self, road, cam: ObliqueCamera, geo_norm, canvas: CanvasSpec, reshape=None
    ) -> "RoadPath | None":
        """One road drawn UNWARPED (straight). With ``reshape`` (a normalized
        polyline) the user's hand-drawn centerline is used instead of the OSM
        geometry."""
        simplify_tol = canvas.width_px * 0.004
        if reshape:
            flat = [(u * cam.flat_width, v * cam.flat_height) for u, v in reshape]
        else:
            flat = [
                (geo_norm(c)[0] * cam.flat_width, geo_norm(c)[1] * cam.flat_height)
                for c in road.coords
            ]
        pts = simplify_polyline(cam.project_points(flat), simplify_tol)
        if len(pts) < 2:
            return None
        mid_y = pts[len(pts) // 2][1]
        return RoadPath(
            cls=road.cls,
            points=pts,
            width_px=road_width_px(road.cls, canvas.width_px) * cam.scale_at(mid_y),
            name=road.name,
            ref=road.ref,
            depth=cam.depth(mid_y),
            id=road.id,
        )

    def _clip_roads(self, roads: list["RoadPath"], canvas: CanvasSpec) -> list["RoadPath"]:
        """Clip each road to an inset of the canvas so nothing bleeds off the
        edge. A road clipped by the frame may yield several in-frame pieces."""
        from shapely.geometry import LineString, box

        m = canvas.width_px * 0.02  # keep clear of the decorative border
        frame = box(m, m, canvas.width_px - m, canvas.height_px - m)
        out: list[RoadPath] = []
        for r in roads:
            if len(r.points) < 2:
                continue
            clipped = LineString(r.points).intersection(frame)
            if clipped.is_empty:
                continue
            geoms = clipped.geoms if clipped.geom_type == "MultiLineString" else [clipped]
            for g in geoms:
                if g.geom_type != "LineString" or g.length <= 0:
                    continue
                pts = [(float(x), float(y)) for x, y in g.coords]
                if len(pts) >= 2:
                    out.append(r.model_copy(update={"points": pts}))
        return out

    def _warped_roads(
        self, roads_src, cam, canvas, to_flat, simplify_tol, densify_px
    ) -> list["RoadPath"]:
        """Warp + stylize + collectively prune (ink budget) a set of roads."""
        flat_roads: list[tuple[RoadClass, list[Point]]] = []
        road_names: list[str | None] = []
        road_refs: list[str | None] = []
        road_ids: list[str | None] = []
        for road in roads_src:
            pts = stylize_polyline(to_flat(road.coords), simplify_tol, densify_px=densify_px)
            if len(pts) >= 2:
                flat_roads.append((road.cls, pts))
                road_names.append(road.name)
                road_refs.append(road.ref)
                road_ids.append(road.id)
        name_of = {id(pts): name for (_, pts), name in zip(flat_roads, road_names)}
        ref_of = {id(pts): ref for (_, pts), ref in zip(flat_roads, road_refs)}
        id_of = {id(pts): rid for (_, pts), rid in zip(flat_roads, road_ids)}
        kept = prune_roads(flat_roads, canvas.width_px, area_px=cam.flat_width * cam.flat_height)
        out: list[RoadPath] = []
        for cls, pts in kept:
            mid_y = pts[len(pts) // 2][1]
            out.append(
                RoadPath(
                    cls=cls,
                    points=cam.project_points(pts),
                    width_px=road_width_px(cls, canvas.width_px) * cam.scale_at(mid_y),
                    name=name_of.get(id(pts)),
                    ref=ref_of.get(id(pts)),
                    depth=cam.depth(mid_y),
                    id=id_of.get(id(pts)),
                )
            )
        return out

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

    @staticmethod
    def _magnify_weight(band_width: float, magnify: float) -> float:
        """Plateau weight (density bump) that magnifies a band by ``magnify``.

        A band of width ``w`` with interior density ``1 + weight`` over a base
        of 1 stretches its contents by ``(1 + weight) / (1 + weight * w)``
        relative to uniform. Solving for the requested magnification gives
        ``weight = (m - 1) / (1 - m * w)``. ``magnify < 1`` yields a negative
        weight (compression); the weight is floored at -0.9 so interior density
        stays positive, and a region too wide to reach ``m`` saturates instead
        of blowing up.
        """
        if magnify == 1.0 or band_width <= 0:
            return 0.0
        denom = 1.0 - magnify * band_width
        if denom <= 1e-3:
            return 1000.0  # region too wide for this magnification; saturate
        return max(-0.9, (magnify - 1.0) / denom)

    def _manual_bands(self, regions):
        """Plateau bands (per axis) from the spec's drawn warp regions."""
        bands_x: list[tuple[float, float, float]] = []
        bands_y: list[tuple[float, float, float]] = []
        for r in regions:
            u0, u1 = sorted((r.bounds[0], r.bounds[2]))
            v0, v1 = sorted((r.bounds[1], r.bounds[3]))
            if u1 - u0 <= 0 or v1 - v0 <= 0:
                continue
            bands_x.append((u0, u1, self._magnify_weight(u1 - u0, r.magnify)))
            bands_y.append((v0, v1, self._magnify_weight(v1 - v0, r.magnify)))
        return bands_x, bands_y

    def _fit_warp(self, source: SourceData, cam: ObliqueCamera, geo_norm):
        """Bounded-plateau cartogram fit.

        POIs are grouped by proximity; each dense cluster becomes a flat-topped
        magnification *plateau* over its (capped) extent, scaled to claim a
        fixed fraction of each axis. The constant-density interior makes the
        warp affine there, so roads through metro Phoenix stay straight; the
        whole metro (incl. the east valley) zooms uniformly rather than the
        empty desert between it and far outliers. POIs the bounded zoom can't
        separate get leader lines. Returns (warp, sprite_scale, residual_px,
        coincident_count[=leader count]).

        ``spec.warp.mode`` selects the source of the plateau bands: ``auto``
        clusters the POIs (below); ``manual`` builds them from the user's drawn
        regions; ``off`` is the identity warp. All three share the residual /
        leader pass at the end.
        """
        mode = self.spec.warp.mode
        sprite_pois = [p for p in source.pois if p.feature_type != "river"]
        centers = [geo_norm((p.longitude, p.latitude)) for p in sprite_pois]
        bands_x: list[tuple[float, float, float]] = []
        bands_y: list[tuple[float, float, float]] = []

        if mode == "off":
            warp: IdentityWarp | ImportanceWarp = IdentityWarp()
        elif mode == "manual":
            bands_x, bands_y = self._manual_bands(self.spec.warp.regions)
            warp = (
                ImportanceWarp(centers=[], strength=1.0, bands=(bands_x, bands_y))
                if bands_x
                else IdentityWarp()
            )
        else:  # "auto"
            if not sprite_pois or self.distortion_strength <= 0:
                return IdentityWarp(), 1.0, 0.0, 0
            for members in _cluster_indices(centers, self.WARP_LINK_DIST):
                if len(members) < 2:
                    continue
                xs = [centers[i][0] for i in members]
                ys = [centers[i][1] for i in members]
                if max(max(xs) - min(xs), max(ys) - min(ys)) < self.MIN_PLATEAU_EXTENT:
                    continue
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                hx = min((max(xs) - min(xs)) / 2, self.MAX_HALF_BAND)
                hy = min((max(ys) - min(ys)) / 2, self.MAX_HALF_BAND)
                bands_x.append((cx - hx, cx + hx, _plateau_weight(2 * hx, self.WARP_TARGET_FRACTION)))
                bands_y.append((cy - hy, cy + hy, _plateau_weight(2 * hy, self.WARP_TARGET_FRACTION)))
            warp = (
                ImportanceWarp(centers=[], strength=1.0, bands=(bands_x, bands_y))
                if bands_x
                else IdentityWarp()
            )

        if not sprite_pois:
            return warp, 1.0, 0.0, 0

        threshold = self.MAX_RESIDUAL_FRACTION * self.canvas.width_px
        sprite_scale = 1.0
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
            slots.append(sized_slot(slot, self.canvas.width_px, cam.scale_at(flat[1]), sprite_scale))
        resolved = resolve_poi_overlaps(slots, self.canvas.width_px, self.canvas.height_px)
        displaced = [
            ((s.anchor[0] - r.anchor[0]) ** 2 + (s.anchor[1] - r.anchor[1]) ** 2) ** 0.5
            for s, r in zip(slots, resolved)
        ]
        coincident_count = sum(1 for d in displaced if d > threshold)
        residual = max((d for d in displaced if d <= threshold), default=0.0)
        logger.info(
            "Warp fit (plateau): %d bands, residual %.0fpx, %d leader POIs",
            len(bands_x),
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
