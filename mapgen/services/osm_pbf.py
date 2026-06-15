"""Local OSM extract backend (reads a Geofabrik ``.osm.pbf`` via GDAL).

Public Overpass instances refuse state/region-size queries (rate-limits,
IP-bans, whitelists -- see V2_GENERALIZATION.md). For large regions we instead
read a local ``.osm.pbf`` extract through GDAL's OSM driver (exposed by
``pyogrio``, already a geopandas dependency), so there is no network call and
no rate limit.

The backend mimics the two osmnx primitives ``OSMService`` relies on:

* ``features(bbox, tags)``  ~ ``ox.features_from_bbox(bbox, tags)``
* ``road_edges(bbox, types)`` ~ ``ox.graph_from_bbox(...)`` -> edges GDF

``tags`` follows the osmnx convention: ``{key: True}`` matches any value,
``{key: [v1, v2]}`` matches membership, and multiple keys are OR-ed together.

Activated by setting ``MAPGEN_OSM_PBF`` to the extract path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..models.project import BoundingBox

# GDAL's OSM driver splits an extract into these geometry layers. We scan
# points/lines/multipolygons (the multilinestrings/other_relations layers are
# relation roll-ups we don't render). Each layer promotes a fixed set of common
# tags to columns; everything else lands in an hstore-style ``other_tags``
# string we parse on demand.
_FEATURE_LAYERS = ("points", "lines", "multipolygons")

# "key"=>"value","key2"=>"value2"  (values may contain escaped quotes)
_HSTORE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def _parse_other_tags(value) -> dict:
    """Parse GDAL's ``other_tags`` hstore string into a plain dict."""
    if not value or not isinstance(value, str):
        return {}
    out = {}
    for key, val in _HSTORE_RE.findall(value):
        out[key.replace('\\"', '"')] = val.replace('\\"', '"')
    return out


class PbfBackend:
    """Query a local ``.osm.pbf`` extract as if it were Overpass."""

    def __init__(self, pbf_path: str | Path):
        self.pbf_path = str(pbf_path)
        if not Path(self.pbf_path).exists():
            raise FileNotFoundError(f"OSM extract not found: {self.pbf_path}")
        # GDAL's OSM driver has no spatial index, so every read re-parses the
        # whole extract. A single fetch issues many tag queries over the same
        # bbox, so cache each (layer, bbox) read and its parsed other_tags.
        self._layer_cache: dict = {}
        self._parsed_cache: dict = {}

    def _read_layer(self, layer: str, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        # pyogrio pushes the bbox spatial filter down into GDAL (xmin, ymin,
        # xmax, ymax in the layer CRS, which for OSM is EPSG:4326 lon/lat).
        spatial = (bbox.west, bbox.south, bbox.east, bbox.north)
        key = (layer, spatial)
        if key in self._layer_cache:
            return self._layer_cache[key]
        try:
            gdf = gpd.read_file(
                self.pbf_path, layer=layer, bbox=spatial, engine="pyogrio"
            )
        except Exception:
            gdf = None
        if gdf is not None and len(gdf) == 0:
            gdf = None
        self._layer_cache[key] = gdf
        return gdf

    def _parsed_tags(self, layer: str, bbox: BoundingBox, gdf: gpd.GeoDataFrame) -> pd.Series:
        """Parsed other_tags for a layer read, cached per (layer, bbox)."""
        key = (layer, (bbox.west, bbox.south, bbox.east, bbox.north))
        cached = self._parsed_cache.get(key)
        if cached is None:
            if "other_tags" in gdf.columns:
                cached = gdf["other_tags"].map(_parse_other_tags)
            else:
                cached = pd.Series([{}] * len(gdf), index=gdf.index)
            self._parsed_cache[key] = cached
        return cached

    @staticmethod
    def _combined(gdf: gpd.GeoDataFrame, parsed: pd.Series, key: str) -> pd.Series:
        """Value of ``key`` per row: promoted column first, else other_tags."""
        from_tags = parsed.map(lambda d: d.get(key))
        if key in gdf.columns:
            col = gdf[key]
            return col.where(col.notna(), from_tags)
        return from_tags

    def features(self, bbox: BoundingBox, tags: dict) -> Optional[gpd.GeoDataFrame]:
        """Features matching ``tags`` (osmnx semantics) across geometry layers."""
        frames: list[gpd.GeoDataFrame] = []
        for layer in _FEATURE_LAYERS:
            gdf = self._read_layer(layer, bbox)
            if gdf is None:
                continue
            parsed = self._parsed_tags(layer, bbox, gdf)

            mask = pd.Series(False, index=gdf.index)
            for key, want in tags.items():
                vals = self._combined(gdf, parsed, key)
                if want is True:
                    mask |= vals.notna()
                else:
                    wanted = {str(v) for v in want}
                    mask |= vals.astype("object").isin(wanted)

            sub = gdf[mask]
            if len(sub) == 0:
                continue
            sub = self._materialize(sub, parsed[mask], tags)
            frames.append(sub)

        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    @staticmethod
    def _materialize(
        sub: gpd.GeoDataFrame, parsed: pd.Series, tags: dict
    ) -> gpd.GeoDataFrame:
        """Promote tag keys from other_tags into real columns (osmnx-like).

        Downstream code does ``row.get("name")``, ``gdf["waterway"]`` etc., so
        the filtered keys plus a handful of commonly-read tags must exist as
        columns even when GDAL left them inside other_tags.
        """
        sub = sub.copy()
        wanted_keys = set(tags) | {
            "name", "ref", "population", "intermittent", "waterway", "highway",
        }
        # Union of keys actually present in this subset's other_tags.
        present: set[str] = set()
        for d in parsed:
            present.update(d.keys())
        for key in wanted_keys:
            from_tags = parsed.map(lambda d: d.get(key)) if key in present else None
            if key in sub.columns:
                if from_tags is not None:
                    sub[key] = sub[key].where(sub[key].notna(), from_tags)
            else:
                sub[key] = from_tags if from_tags is not None else None
        return sub

    def road_edges(
        self, bbox: BoundingBox, highway_types: Optional[list[str]]
    ) -> Optional[gpd.GeoDataFrame]:
        """Highway lines of the given types, shaped like osmnx edge GDFs.

        Returns LineStrings with ``highway``/``ref``/``name`` columns. We skip
        the osmnx graph build (topology/simplification) -- an illustrated
        poster only needs the geometry and class.
        """
        gdf = self._read_layer("lines", bbox)
        if gdf is None or "highway" not in gdf.columns:
            return None
        if highway_types is None:
            sub = gdf[gdf["highway"].notna()]
        else:
            wanted = set(highway_types)
            sub = gdf[gdf["highway"].astype("object").isin(wanted)]
        sub = sub[sub.geometry.type.isin(["LineString", "MultiLineString"])]
        if len(sub) == 0:
            return None
        parsed = self._parsed_tags("lines", bbox, gdf).loc[sub.index]
        sub = sub.copy()
        ref_from_tags = parsed.map(lambda d: d.get("ref"))
        if "ref" not in sub.columns:
            sub["ref"] = ref_from_tags
        else:
            sub["ref"] = sub["ref"].where(sub["ref"].notna(), ref_from_tags)
        return gpd.GeoDataFrame(
            sub.reset_index(drop=True), geometry="geometry", crs="EPSG:4326"
        )
