"""Ingestion boundary: geographic source data in plain Python.

The plan engine consumes SourceData -- plain lists of lon/lat coordinate
sequences -- so it has no dependency on geopandas/osmnx. The conversion
from V1's OSMData (GeoDataFrames) lives here behind a lazy import; tests
and offline development use synthetic SourceData directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import GroundClass, Point, RegionBBox, RoadClass

LonLat = tuple[float, float]


@dataclass
class SourceRoad:
    cls: RoadClass
    coords: list[LonLat]
    name: Optional[str] = None
    ref: Optional[str] = None  # route designation (e.g. "I 10", "US 60")


@dataclass
class SourcePolygon:
    cls: GroundClass
    exterior: list[LonLat]
    holes: list[list[LonLat]] = field(default_factory=list)
    name: Optional[str] = None


@dataclass
class SourcePoi:
    id: str
    name: str
    latitude: float
    longitude: float
    tier: int = 2
    photo: Optional[str] = None
    feature_type: str = "building"
    # Polyline in (lat, lon) order for linear features (rivers): drawn as a
    # waterway ribbon instead of a sprite.
    path: Optional[list[tuple[float, float]]] = None


@dataclass
class SourcePlace:
    """Named district/neighborhood/water-body label anchor."""

    name: str
    latitude: float
    longitude: float
    kind: str = "district"  # district | water
    population: Optional[float] = None  # drives city patch + label size


@dataclass
class SourceData:
    region: RegionBBox
    roads: list[SourceRoad] = field(default_factory=list)
    ground: list[SourcePolygon] = field(default_factory=list)
    buildings: list[list[LonLat]] = field(default_factory=list)
    pois: list[SourcePoi] = field(default_factory=list)
    places: list[SourcePlace] = field(default_factory=list)
    # Fetch provenance: detail level used and per-layer outcome
    # (feature count, "failed", or "empty"), recorded into the plan so a
    # missing layer is visible instead of silently absent.
    provenance: dict = field(default_factory=dict)


_HIGHWAY_TO_CLASS = {
    "motorway": RoadClass.MOTORWAY,
    "trunk": RoadClass.MOTORWAY,
    "primary": RoadClass.PRIMARY,
    "secondary": RoadClass.SECONDARY,
    "tertiary": RoadClass.SECONDARY,
    "residential": RoadClass.LOCAL,
    "unclassified": RoadClass.LOCAL,
    "service": RoadClass.PATH,
    "footway": RoadClass.PATH,
    "cycleway": RoadClass.PATH,
    "pedestrian": RoadClass.PATH,
}


def geo_to_flat(coord: LonLat, region: RegionBBox, flat_w: float, flat_h: float) -> Point:
    """Lon/lat -> flat-space pixels (y=0 at the north/far edge)."""
    x = (coord[0] - region.west) / max(1e-12, region.width_deg) * flat_w
    y = (region.north - coord[1]) / max(1e-12, region.height_deg) * flat_h
    return (x, y)


def geo_to_normalized(coord: LonLat, region: RegionBBox) -> Point:
    return (
        (coord[0] - region.west) / max(1e-12, region.width_deg),
        (region.north - coord[1]) / max(1e-12, region.height_deg),
    )


class GeoFrame:
    """Geo -> normalized map-frame transform: metric, rotatable, aspect-true.

    Replaces the naive degree-stretch mapping (which distorted east-west by
    1/cos(latitude) and warped any region whose shape didn't match the
    canvas). The frame works in local kilometers around the region center,
    rotates so poster-up points at compass bearing ``rotation_deg`` (0 =
    north-up, 340 = tilted toward NW like V1's orientation_degrees), and --
    rather than stretching or letterboxing -- *extends* the covered ground
    symmetrically on one axis until the metric aspect matches
    ``target_aspect`` (the camera's flat-space h/w). ``fetch_region`` is the
    axis-aligned geographic envelope of the final frame, used for the OSM
    fetch so the extension is filled with real data.
    """

    def __init__(
        self,
        region: RegionBBox,
        rotation_deg: float = 0.0,
        target_aspect: Optional[float] = None,
    ):
        import math

        self.region = region
        self.rotation_deg = float(rotation_deg) % 360.0
        self.target_aspect = target_aspect
        self._lat_c = region.mid_latitude
        self._lon_c = (region.east + region.west) / 2.0
        self._kx = 111.32 * math.cos(math.radians(self._lat_c))
        self._ky = 110.57
        beta = math.radians(self.rotation_deg)
        self._cos_b, self._sin_b = math.cos(beta), math.sin(beta)

        corners = [
            (region.west, region.south),
            (region.west, region.north),
            (region.east, region.south),
            (region.east, region.north),
        ]
        xs, ys = zip(*(self._to_frame_km(c) for c in corners))
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if target_aspect:
            width, height = x1 - x0, y1 - y0
            if height / max(1e-9, width) < target_aspect:
                grow = (target_aspect * width - height) / 2.0
                y0, y1 = y0 - grow, y1 + grow
            else:
                grow = (height / target_aspect - width) / 2.0
                x0, x1 = x0 - grow, x1 + grow
        self._x0, self._x1, self._y0, self._y1 = x0, x1, y0, y1

    def _to_frame_km(self, coord: LonLat) -> Point:
        """Lon/lat -> rotated local km (x right, y down in poster terms)."""
        e = (coord[0] - self._lon_c) * self._kx
        n = (coord[1] - self._lat_c) * self._ky
        x = e * self._cos_b - n * self._sin_b
        y = -(e * self._sin_b + n * self._cos_b)
        return (x, y)

    def to_normalized(self, coord: LonLat) -> Point:
        x, y = self._to_frame_km(coord)
        return (
            (x - self._x0) / max(1e-9, self._x1 - self._x0),
            (y - self._y0) / max(1e-9, self._y1 - self._y0),
        )

    @property
    def fetch_region(self) -> RegionBBox:
        """Axis-aligned geographic envelope of the (rotated, extended) frame."""
        corners_km = [
            (self._x0, self._y0),
            (self._x0, self._y1),
            (self._x1, self._y0),
            (self._x1, self._y1),
        ]
        lons, lats = [], []
        for x, y in corners_km:
            # Invert the frame rotation: (x, y) -> (e, n).
            e = x * self._cos_b - y * self._sin_b
            n = -(x * self._sin_b + y * self._cos_b)
            lons.append(self._lon_c + e / self._kx)
            lats.append(self._lat_c + n / self._ky)
        return RegionBBox(north=max(lats), south=min(lats), east=max(lons), west=min(lons))

    def to_dict(self) -> dict:
        return {
            "rotation_deg": self.rotation_deg,
            "target_aspect": self.target_aspect,
            "region": self.region.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeoFrame":
        return cls(
            RegionBBox.model_validate(data["region"]),
            rotation_deg=data.get("rotation_deg", 0.0),
            target_aspect=data.get("target_aspect"),
        )


def auto_rotation(
    points: list[LonLat], region: RegionBBox, target_aspect: float
) -> float:
    """Pick the up-bearing (in -40..40 deg) whose rotated point-cloud extent
    best matches the canvas aspect -- the 'tilt the map so the landmarks fill
    the poster' move a hand illustrator makes."""
    import math

    if len(points) < 2:
        return 0.0
    best, best_score = 0.0, float("inf")
    for deg in range(-40, 41, 5):
        frame = GeoFrame(region, rotation_deg=deg % 360)
        xs, ys = zip(*(frame._to_frame_km(p) for p in points))
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width < 1e-6 or height < 1e-6:
            continue
        score = abs(math.log((height / width) / target_aspect))
        if score < best_score:
            best, best_score = float(deg % 360), score
    return best


def _clean_name(value) -> Optional[str]:
    """Coerce a GeoDataFrame cell to a clean str or None.

    Unnamed OSM features come back as pandas NaN (a float), and ``nan or None``
    evaluates to ``nan`` because NaN is truthy -- so guard explicitly.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def from_osm_data(osm_data, region: RegionBBox) -> SourceData:  # pragma: no cover - needs geopandas
    """Convert V1 OSMData (GeoDataFrames) into plain SourceData."""
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

    source = SourceData(region=region)

    def _lines(geom):
        if isinstance(geom, LineString):
            yield list(geom.coords)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                yield list(line.coords)

    def _polys(geom):
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, MultiPolygon):
            yield from geom.geoms

    roads_gdf = getattr(osm_data, "roads", None)
    if roads_gdf is not None:
        for _, row in roads_gdf.iterrows():
            highway = row.get("highway")
            if isinstance(highway, list):
                highway = highway[0] if highway else None
            cls = _HIGHWAY_TO_CLASS.get(str(highway), RoadClass.LOCAL)
            name = _clean_name(row.get("name"))
            ref = _clean_name(row.get("ref_normalized") or row.get("ref"))
            for coords in _lines(row.geometry):
                source.roads.append(SourceRoad(cls=cls, coords=coords, name=name, ref=ref))

    for attr, ground_cls in (("water", GroundClass.WATER), ("parks", GroundClass.PARK)):
        gdf = getattr(osm_data, attr, None)
        if gdf is None:
            continue
        for _, row in gdf.iterrows():
            name = _clean_name(row.get("name"))
            for poly in _polys(row.geometry):
                source.ground.append(
                    SourcePolygon(
                        cls=ground_cls,
                        exterior=list(poly.exterior.coords),
                        holes=[list(ring.coords) for ring in poly.interiors],
                        name=name,
                    )
                )
                if name and ground_cls is GroundClass.WATER:
                    c = poly.centroid
                    source.places.append(
                        SourcePlace(name=name, latitude=c.y, longitude=c.x, kind="water")
                    )
                    name = None  # label each named water body once

    # Terrain landuse/natural polygons (full detail tier only).
    _TERRAIN_TO_GROUND = {
        "urban": GroundClass.URBAN,
        "desert": GroundClass.SAND,
        "forest": GroundClass.FOREST,
        "grassland": GroundClass.FARMLAND,
    }
    terrain_gdf = getattr(osm_data, "terrain_types", None)
    if terrain_gdf is not None:
        for _, row in terrain_gdf.iterrows():
            cls = _TERRAIN_TO_GROUND.get(row.get("terrain_class"))
            if cls is None:
                continue
            for poly in _polys(row.geometry):
                source.ground.append(
                    SourcePolygon(
                        cls=cls,
                        exterior=list(poly.exterior.coords),
                        holes=[list(ring.coords) for ring in poly.interiors],
                        name=_clean_name(row.get("name")),
                    )
                )

    # Waterway centerlines: rivers as major ribbons, washes/canals as streams.
    for attr, cls in (("rivers", RoadClass.RIVER), ("washes", RoadClass.STREAM)):
        gdf = getattr(osm_data, attr, None)
        if gdf is None:
            continue
        for _, row in gdf.iterrows():
            name = _clean_name(row.get("name"))
            for coords in _lines(row.geometry):
                source.roads.append(SourceRoad(cls=cls, coords=coords, name=name))

    railways_gdf = getattr(osm_data, "railways", None)
    if railways_gdf is not None:
        for _, row in railways_gdf.iterrows():
            for coords in _lines(row.geometry):
                source.roads.append(
                    SourceRoad(cls=RoadClass.RAIL, coords=coords, name=_clean_name(row.get("name")))
                )

    buildings_gdf = getattr(osm_data, "buildings", None)
    if buildings_gdf is not None:
        for _, row in buildings_gdf.iterrows():
            for poly in _polys(row.geometry):
                source.buildings.append(list(poly.exterior.coords))

    return source
