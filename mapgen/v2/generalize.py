"""Scale-aware generalization of fetched OSM source data.

A complete local extract (see osm_pbf.py) returns *everything* -- on a
state-scale poster that means tens of thousands of stock-tank ponds, dense
irrigation-ditch spaghetti, and road geometry fragmented into thousands of
tiny way-stubs. None of it belongs on an illustrated tourist map. This pass
drops sub-threshold water, keeps only notable named water/rivers, and merges
connected road segments into continuous polylines (so they no longer render
as a chain of round-capped beads).

City-scale regions are left untouched -- a small pond matters there.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .ingest import SourceData, SourcePlace, SourcePolygon, SourceRoad
from .types import GroundClass, RoadClass, RegionBBox

# Names that are never worth a label on a tourist poster.
_JUNK_WATER = re.compile(r"\b(tank|stock pond|dry lake|sewage|settling|detention)\b", re.I)
# Minor man-made waterways -- irrigation plumbing, not scenic rivers.
_MINOR_WATERWAY = re.compile(r"\b(ditch|lateral|wash|drain|sluice)\b", re.I)

# Below this region area, keep full detail (city / small-region posters).
_GENERALIZE_MIN_KM2 = 2_000.0


def _poly_km2(coords: list[tuple[float, float]]) -> float:
    """Approximate polygon area in km^2 (shoelace, cos-latitude corrected)."""
    if len(coords) < 3:
        return 0.0
    lat0 = sum(c[1] for c in coords) / len(coords)
    kx = 111.0 * math.cos(math.radians(lat0))
    ky = 111.0
    a = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i][0] * kx, coords[i][1] * ky
        x2, y2 = coords[(i + 1) % n][0] * kx, coords[(i + 1) % n][1] * ky
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _line_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    lat0 = sum(c[1] for c in coords) / len(coords)
    kx = 111.0 * math.cos(math.radians(lat0))
    ky = 111.0
    return sum(
        math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * ky)
        for a, b in zip(coords, coords[1:])
    )


def _centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    return (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))


def _merge_roads(roads: list[SourceRoad]) -> list[SourceRoad]:
    """Stitch connected segments of the same class into long polylines.

    The extract returns each OSM way separately -- and a single physical
    highway is tagged with many different local names and route refs along its
    length -- so it arrives as dozens of short stubs that render as a chain of
    round-capped dots. Merging by shared endpoints within a class
    (shapely.linemerge, which stops at real junctions of degree > 2) restores
    continuous centerlines; each merged run then inherits the name/ref that
    covers the most of its length, so labels and shields still work.
    """
    from collections import Counter

    from shapely.geometry import LineString
    from shapely.ops import linemerge
    from shapely.strtree import STRtree

    by_cls: dict[object, list[SourceRoad]] = defaultdict(list)
    for r in roads:
        if len(r.coords) >= 2:
            by_cls[r.cls].append(r)

    out: list[SourceRoad] = []
    for cls, group in by_cls.items():
        lines = [LineString(r.coords) for r in group]
        merged = linemerge(lines)
        merged_lines = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
        merged_lines = [m for m in merged_lines if m.length > 0]
        if not merged_lines:
            continue
        # Attribute each merged run by majority constituent length.
        tree = STRtree(merged_lines)
        votes = [Counter() for _ in merged_lines]
        for r, ls in zip(group, lines):
            idx = int(tree.nearest(ls.interpolate(0.5, normalized=True)))
            votes[idx][(r.name, r.ref)] += ls.length
        for ml, vote in zip(merged_lines, votes):
            name, ref = vote.most_common(1)[0][0] if vote else (None, None)
            coords = [(float(x), float(y)) for x, y in ml.coords]
            if len(coords) >= 2:
                out.append(SourceRoad(cls=cls, coords=coords, name=name, ref=ref))
    return out


_CITY_PATCH_MIN_KM = 3.0
_CITY_PATCH_MAX_KM = 22.0


def city_radius_km(population: float | None) -> float:
    """Urban-patch radius from population (sqrt-scaled, clamped)."""
    if not population or population <= 0:
        return _CITY_PATCH_MIN_KM
    r = 2.5 + 0.012 * (population**0.5)
    return max(_CITY_PATCH_MIN_KM, min(_CITY_PATCH_MAX_KM, r))


def add_urban_areas(source: SourceData) -> SourceData:
    """Synthesize an URBAN ground patch per city.

    Built-up areas otherwise read as bare desert at this scale (OSM building
    coverage is patchy and would fight the warp/POI layers anyway). Each city
    place becomes an organic URBAN blob sized by population; the existing
    pipeline then textures it (texture_urban) and scatters houses
    (GroundClass.URBAN -> ScatterKind.HOUSE) -- all warped with the map.
    Overlapping metro patches (Phoenix/Mesa/Tempe) merge into one built area.
    """
    patches: list[SourcePolygon] = []
    for p in source.places:
        if p.kind != "district":
            continue
        r_km = city_radius_km(p.population)
        r_lat = r_km / 111.0
        r_lon = r_km / (111.0 * math.cos(math.radians(p.latitude)))
        ring = []
        for k in range(24):
            a = 2.0 * math.pi * k / 24
            rr = 1.0 + 0.14 * math.sin(3.0 * a) + 0.07 * math.sin(5.0 * a)  # organic edge
            ring.append((p.longitude + r_lon * rr * math.cos(a), p.latitude + r_lat * rr * math.sin(a)))
        patches.append(SourcePolygon(cls=GroundClass.URBAN, exterior=ring, holes=[], name=None))
    source.ground.extend(patches)
    return source


def generalize_source(source: SourceData, region: RegionBBox) -> SourceData:
    """Drop sub-scale clutter and merge fragmented roads, in place."""
    area = region.area_km2

    # Always merge roads -- the fragmentation is a rendering problem at any
    # scale, and merging is lossless.
    source.roads = _merge_roads(source.roads)

    if area < _GENERALIZE_MIN_KM2:
        return source  # keep small-region detail otherwise

    # Area cutoffs scale with the region: a lake worth drawing on a state map
    # is far bigger than one worth drawing on a city map.
    keep_km2 = max(0.05, area * 2.5e-6)  # drop water smaller than this
    label_km2 = max(0.5, area * 1.5e-5)  # only label water at least this big
    river_min_km = max(2.0, (area ** 0.5) * 0.02)  # drop washes shorter than this

    # --- Water polygons: drop tiny tanks; relabel only notable named water ---
    kept_ground: list[SourcePolygon] = []
    water_labels: list[tuple[str, float, tuple[float, float]]] = []
    for g in source.ground:
        if g.cls is GroundClass.WATER:
            a = _poly_km2(g.exterior)
            if a < keep_km2:
                continue
            if g.name and a >= label_km2 and not _JUNK_WATER.search(g.name):
                water_labels.append((g.name, a, _centroid(g.exterior)))
        kept_ground.append(g)
    source.ground = kept_ground

    # Rebuild water place-labels: one per name, largest instance wins, capped.
    source.places = [p for p in source.places if p.kind != "water"]
    best_by_name: dict[str, tuple[float, tuple[float, float]]] = {}
    for name, a, c in water_labels:
        if name not in best_by_name or a > best_by_name[name][0]:
            best_by_name[name] = (a, c)
    for name, (_, c) in sorted(best_by_name.items(), key=lambda kv: kv[1][0], reverse=True)[:25]:
        source.places.append(SourcePlace(name=name, latitude=c[1], longitude=c[0], kind="water"))

    # --- River/waterway lines: keep notable named rivers, drop wash spaghetti ---
    kept_roads: list[SourceRoad] = []
    for r in source.roads:
        if r.cls is RoadClass.RIVER:
            if not r.name or _MINOR_WATERWAY.search(r.name) or _line_km(r.coords) < river_min_km:
                continue
        kept_roads.append(r)
    source.roads = kept_roads

    return source
