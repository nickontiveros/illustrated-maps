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


@dataclass
class SourcePlace:
    """Named district/neighborhood/water-body label anchor."""

    name: str
    latitude: float
    longitude: float
    kind: str = "district"  # district | water


@dataclass
class SourceData:
    region: RegionBBox
    roads: list[SourceRoad] = field(default_factory=list)
    ground: list[SourcePolygon] = field(default_factory=list)
    buildings: list[list[LonLat]] = field(default_factory=list)
    pois: list[SourcePoi] = field(default_factory=list)
    places: list[SourcePlace] = field(default_factory=list)


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
            for coords in _lines(row.geometry):
                source.roads.append(SourceRoad(cls=cls, coords=coords, name=name))

    for attr, ground_cls in (("water", GroundClass.WATER), ("parks", GroundClass.PARK)):
        gdf = getattr(osm_data, attr, None)
        if gdf is None:
            continue
        for _, row in gdf.iterrows():
            for poly in _polys(row.geometry):
                source.ground.append(
                    SourcePolygon(
                        cls=ground_cls,
                        exterior=list(poly.exterior.coords),
                        holes=[list(ring.coords) for ring in poly.interiors],
                        name=_clean_name(row.get("name")),
                    )
                )

    buildings_gdf = getattr(osm_data, "buildings", None)
    if buildings_gdf is not None:
        for _, row in buildings_gdf.iterrows():
            for poly in _polys(row.geometry):
                source.buildings.append(list(poly.exterior.coords))

    return source
