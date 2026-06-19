"""OpenStreetMap data extraction service."""

from dataclasses import dataclass
from typing import Callable, Optional

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon

from ..models.project import BoundingBox

_FEET_PER_METER = 3.28084


def _parse_float(value) -> Optional[float]:
    """First numeric token of an OSM tag value, or None.

    OSM cells arrive as strings ("12"), lists (multi-valued ways), or pandas
    NaN floats; tolerate all three.
    """
    import re

    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # NaN check
    if not isinstance(value, str):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(m.group()) if m else None


def _parse_metric_height(value) -> Optional[float]:
    """OSM ``height`` value in meters, honoring a "ft"/"'" unit suffix."""
    num = _parse_float(value)
    if num is None:
        return None
    text = value[0] if isinstance(value, list) and value else value
    if isinstance(text, str) and ("ft" in text or "'" in text or '"' in text):
        return num / _FEET_PER_METER
    return num


@dataclass
class OSMData:
    """Container for extracted OSM data layers."""

    roads: Optional[gpd.GeoDataFrame] = None
    buildings: Optional[gpd.GeoDataFrame] = None
    water: Optional[gpd.GeoDataFrame] = None
    parks: Optional[gpd.GeoDataFrame] = None
    terrain_types: Optional[gpd.GeoDataFrame] = None
    railways: Optional[gpd.GeoDataFrame] = None
    pois: Optional[gpd.GeoDataFrame] = None
    washes: Optional[gpd.GeoDataFrame] = None

    def has_data(self) -> bool:
        """Check if any data was extracted."""
        return any(
            df is not None and len(df) > 0
            for df in [
                self.roads,
                self.buildings,
                self.water,
                self.parks,
                self.terrain_types,
                self.washes,
            ]
        )


class OSMService:
    """Service for extracting OpenStreetMap data."""

    # Road type classifications for rendering
    ROAD_TYPES = {
        "highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"],
        "major": ["motorway", "trunk", "primary"],
        "minor": ["secondary", "tertiary", "residential", "unclassified"],
    }

    # Terrain type tags
    TERRAIN_TAGS = {
        "urban": {
            "landuse": ["residential", "commercial", "industrial", "retail"],
        },
        "desert": {
            "natural": ["desert", "sand", "dune"],
        },
        "forest": {
            "landuse": ["forest"],
            "natural": ["wood"],
        },
        "grassland": {
            "landuse": ["grass", "meadow", "farmland"],
            "natural": ["grassland", "heath"],
        },
        "wetland": {
            "natural": ["wetland"],
        },
        "beach": {
            "natural": ["beach"],
        },
        "cemetery": {
            "landuse": ["cemetery"],
            "amenity": ["grave_yard"],
        },
        "water": {
            "natural": ["water"],
            "waterway": True,
        },
        "mountain": {
            "natural": ["peak", "ridge", "cliff", "rock", "scree"],
        },
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize OSM service.

        Args:
            cache_dir: Optional directory to cache OSM data
        """
        self.cache_dir = cache_dir
        # Configure osmnx
        ox.settings.use_cache = cache_dir is not None
        if cache_dir:
            ox.settings.cache_folder = cache_dir
        # Large regions hammer the default Overpass instance with dozens of
        # sub-queries and get the client IP rate-limited (connection refused).
        # MAPGEN_OVERPASS_URL switches to a mirror (e.g.
        # https://overpass.kumi.systems/api) without code changes.
        import os

        overpass_url = os.environ.get("MAPGEN_OVERPASS_URL")
        if overpass_url:
            ox.settings.overpass_url = overpass_url
        # Bigger sub-query tiles mean far fewer requests for huge regions
        # (default 2,500 km^2 splits a state into ~76 queries per layer).
        max_query_area = os.environ.get("MAPGEN_OSM_MAX_QUERY_AREA_M2")
        if max_query_area:
            ox.settings.max_query_area_size = int(float(max_query_area))
        # Public Overpass refuses state/region-size queries (rate-limit, IP-ban,
        # whitelist). Point MAPGEN_OSM_PBF at a local Geofabrik .osm.pbf extract
        # to read those layers offline via GDAL instead -- no network, no limit.
        self._pbf = None
        pbf_path = os.environ.get("MAPGEN_OSM_PBF")
        if pbf_path:
            from .osm_pbf import PbfBackend

            self._pbf = PbfBackend(pbf_path)

    def _features(self, bbox: BoundingBox, tags: dict) -> Optional[gpd.GeoDataFrame]:
        """Fetch features by tag, from the local extract if one is configured."""
        if self._pbf is not None:
            return self._pbf.features(bbox, tags)
        return ox.features_from_bbox(bbox=bbox.to_osmnx_bbox(), tags=tags)

    def _road_edges(
        self, bbox: BoundingBox, highway_types: Optional[list[str]]
    ) -> Optional[gpd.GeoDataFrame]:
        """Fetch highway lines as an edges GDF (local extract or osmnx graph).

        ``highway_types=None`` means the full drivable network.
        """
        if self._pbf is not None:
            return self._pbf.road_edges(bbox, highway_types)
        # osmnx drops tags not in useful_tags_way; keep "network" so shields can
        # pick accurate per-network artwork (e.g. "US:I", "US:AZ").
        if "network" not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["network"]
        kwargs = dict(bbox=bbox.to_osmnx_bbox(), network_type="drive", simplify=True)
        if highway_types is not None:
            kwargs["custom_filter"] = f'["highway"~"{"|".join(highway_types)}"]'
        G = ox.graph_from_bbox(**kwargs)
        return ox.graph_to_gdfs(G, nodes=False, edges=True)

    def fetch_region_data(
        self,
        bbox: BoundingBox,
        detail_level: str = "full",
        progress_callback: Optional[Callable] = None,
    ) -> OSMData:
        """
        Fetch OSM data for a region.

        Args:
            bbox: Bounding box defining the region
            detail_level: Level of detail - "full", "simplified", "regional", or "country"
                - full: All features (for areas < 100 km²)
                - simplified: Major roads, notable buildings (100-1,000 km²)
                - regional: Primary roads, landmarks only (1,000-50,000 km²)
                - country: Motorways, major cities only (> 50,000 km²)

        Returns:
            OSMData containing extracted layers
        """
        if detail_level == "simplified":
            return self.fetch_simplified_region_data(bbox, progress_callback=progress_callback)
        elif detail_level == "regional":
            return self.fetch_regional_region_data(bbox, progress_callback=progress_callback)
        elif detail_level == "country":
            return self.fetch_country_region_data(bbox, progress_callback=progress_callback)

        # Full detail level with verbose logging
        import sys
        data = OSMData()

        def _report(label: str, step: int, total: int = 8):
            print(f"  [OSM] {label}...", end=" ", flush=True)
            sys.stdout.flush()
            if progress_callback:
                progress_callback(label, step, total)

        _report("Fetching all roads", 1)
        data.roads = self.extract_roads(bbox)
        road_count = len(data.roads) if data.roads is not None else 0
        print(f"({road_count} features)")

        _report("Fetching all buildings", 2)
        data.buildings = self.extract_buildings(bbox)
        building_count = len(data.buildings) if data.buildings is not None else 0
        print(f"({building_count} features)")

        _report("Fetching water", 3)
        data.water = self.extract_water(bbox)
        water_count = len(data.water) if data.water is not None else 0
        print(f"({water_count} features)")

        _report("Fetching parks", 4)
        data.parks = self.extract_parks(bbox)
        park_count = len(data.parks) if data.parks is not None else 0
        print(f"({park_count} features)")

        _report("Fetching terrain types", 5)
        data.terrain_types = self.extract_terrain_types(bbox)
        terrain_count = len(data.terrain_types) if data.terrain_types is not None else 0
        print(f"({terrain_count} features)")

        _report("Fetching railways", 6)
        data.railways = self.extract_railways(bbox)
        railway_count = len(data.railways) if data.railways is not None else 0
        print(f"({railway_count} features)")

        _report("Fetching washes/arroyos", 7)
        data.washes = self.extract_washes(bbox)
        wash_count = len(data.washes) if data.washes is not None else 0
        print(f"({wash_count} features)")

        if progress_callback:
            progress_callback("OSM data complete", 8, 8)

        return data

    def fetch_simplified_region_data(self, bbox: BoundingBox, verbose: bool = True, progress_callback: Optional[Callable] = None) -> OSMData:
        """
        Fetch simplified OSM data - major features only for illustrated maps.

        Only includes:
        - Major roads (highways, primary, secondary)
        - Notable buildings (named, historic, tourism)
        - All water bodies
        - All parks

        Args:
            bbox: Bounding box defining the region
            verbose: Print progress messages

        Returns:
            OSMData with simplified layers
        """
        import sys
        data = OSMData()

        if verbose:
            print("  [OSM] Fetching major roads...", end=" ", flush=True)
            sys.stdout.flush()
        data.roads = self.extract_major_roads(bbox)
        if verbose:
            road_count = len(data.roads) if data.roads is not None else 0
            print(f"({road_count} features)")

        if verbose:
            print("  [OSM] Fetching notable buildings...", end=" ", flush=True)
            sys.stdout.flush()
        data.buildings = self.extract_notable_buildings(bbox)
        if verbose:
            building_count = len(data.buildings) if data.buildings is not None else 0
            print(f"({building_count} features)")

        if verbose:
            print("  [OSM] Fetching water...", end=" ", flush=True)
            sys.stdout.flush()
        data.water = self.extract_water(bbox)
        if verbose:
            water_count = len(data.water) if data.water is not None else 0
            print(f"({water_count} features)")

        if verbose:
            print("  [OSM] Fetching parks...", end=" ", flush=True)
            sys.stdout.flush()
        data.parks = self.extract_parks(bbox)
        if verbose:
            park_count = len(data.parks) if data.parks is not None else 0
            print(f"({park_count} features)")

        # Skip railways and detailed terrain for simplified mode

        if verbose:
            print("  [OSM] Fetching washes/arroyos...", end=" ", flush=True)
            sys.stdout.flush()
        data.washes = self.extract_washes(bbox)
        if verbose:
            wash_count = len(data.washes) if data.washes is not None else 0
            print(f"({wash_count} features)")

        return data

    def fetch_regional_region_data(self, bbox: BoundingBox, progress_callback: Optional[Callable] = None) -> OSMData:
        """
        Fetch regional OSM data - primary roads and landmarks only.

        For medium-large regions (1,000-50,000 km²).

        Only includes:
        - Primary roads (motorways, trunks, primary)
        - Named landmarks only (famous buildings, monuments)
        - Major water bodies
        - Major parks

        Args:
            bbox: Bounding box defining the region

        Returns:
            OSMData with regional layers
        """
        import sys
        data = OSMData()

        print("  [OSM] Fetching primary roads...", end=" ", flush=True)
        sys.stdout.flush()
        data.roads = self.extract_primary_roads(bbox)
        road_count = len(data.roads) if data.roads is not None else 0
        print(f"({road_count} features)")

        print("  [OSM] Fetching landmark buildings...", end=" ", flush=True)
        sys.stdout.flush()
        data.buildings = self.extract_landmark_buildings(bbox)
        building_count = len(data.buildings) if data.buildings is not None else 0
        print(f"({building_count} features)")

        print("  [OSM] Fetching major water...", end=" ", flush=True)
        sys.stdout.flush()
        data.water = self.extract_major_water(bbox)
        water_count = len(data.water) if data.water is not None else 0
        print(f"({water_count} features)")

        print("  [OSM] Fetching major parks...", end=" ", flush=True)
        sys.stdout.flush()
        data.parks = self.extract_major_parks(bbox)
        park_count = len(data.parks) if data.parks is not None else 0
        print(f"({park_count} features)")

        print("  [OSM] Fetching washes/arroyos...", end=" ", flush=True)
        sys.stdout.flush()
        data.washes = self.extract_washes(bbox)
        wash_count = len(data.washes) if data.washes is not None else 0
        print(f"({wash_count} features)")

        return data

    def fetch_country_region_data(self, bbox: BoundingBox, progress_callback: Optional[Callable] = None) -> OSMData:
        """
        Fetch country-level OSM data - motorways and major cities only.

        For very large regions (> 50,000 km²).

        Only includes:
        - Motorways only
        - Capital/major cities (as points or simplified polygons)
        - Coastlines and major rivers
        - No parks (too detailed for country scale)

        Args:
            bbox: Bounding box defining the region

        Returns:
            OSMData with country-level layers
        """
        import sys
        data = OSMData()

        print("  [OSM] Fetching motorways...", end=" ", flush=True)
        sys.stdout.flush()
        data.roads = self.extract_motorways_only(bbox)
        road_count = len(data.roads) if data.roads is not None else 0
        print(f"({road_count} features)")

        print("  [OSM] Fetching major cities...", end=" ", flush=True)
        sys.stdout.flush()
        data.buildings = self.extract_major_cities(bbox)
        city_count = len(data.buildings) if data.buildings is not None else 0
        print(f"({city_count} features)")

        print("  [OSM] Fetching coastline and rivers...", end=" ", flush=True)
        sys.stdout.flush()
        data.water = self.extract_coastline_and_major_rivers(bbox)
        water_count = len(data.water) if data.water is not None else 0
        print(f"({water_count} features)")

        # No parks at country level

        return data

    def extract_roads(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract road network from OSM.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with road geometries and attributes
        """
        try:
            # Get road network (full drivable network)
            edges = self._road_edges(bbox, None)
            if edges is None or len(edges) == 0:
                return edges

            # Add road classification
            edges["road_class"] = edges["highway"].apply(self._classify_road)

            # Normalize ref tags for highway shields
            if "ref" in edges.columns:
                edges["ref_normalized"] = edges["ref"].apply(self._normalize_ref)
            # Normalize network tags (e.g. "US:I", "US:AZ") for shield artwork
            if "network" in edges.columns:
                edges["network_normalized"] = edges["network"].apply(self._normalize_ref)

            return edges

        except Exception as e:
            print(f"Warning: Could not extract roads: {e}")
            return None

    def extract_major_roads(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only major roads (highways, primary, secondary).

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with major road geometries
        """
        try:
            # Define major road types
            major_highway_types = [
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link",
                "primary",
                "primary_link",
                "secondary",
                "secondary_link",
            ]

            edges = self._road_edges(bbox, major_highway_types)
            if edges is None or len(edges) == 0:
                return edges

            # Add road classification
            edges["road_class"] = edges["highway"].apply(self._classify_road)

            # Normalize ref tags for highway shields
            if "ref" in edges.columns:
                edges["ref_normalized"] = edges["ref"].apply(self._normalize_ref)
            # Normalize network tags (e.g. "US:I", "US:AZ") for shield artwork
            if "network" in edges.columns:
                edges["network_normalized"] = edges["network"].apply(self._normalize_ref)

            return edges

        except Exception as e:
            print(f"Warning: Could not extract major roads: {e}")
            return None

    def extract_primary_roads(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only primary roads (motorways, trunks, primary).

        For regional-scale maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with primary road geometries
        """
        try:
            primary_highway_types = [
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link",
                "primary",
                "primary_link",
            ]

            edges = self._road_edges(bbox, primary_highway_types)
            if edges is None or len(edges) == 0:
                return edges

            edges["road_class"] = edges["highway"].apply(self._classify_road)

            # Normalize ref tags for highway shields
            if "ref" in edges.columns:
                edges["ref_normalized"] = edges["ref"].apply(self._normalize_ref)

            return edges

        except Exception as e:
            print(f"Warning: Could not extract primary roads: {e}")
            return None

    def extract_motorways_only(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only motorways for country-scale maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with motorway geometries
        """
        try:
            motorway_types = ["motorway", "motorway_link"]
            edges = self._road_edges(bbox, motorway_types)
            if edges is None or len(edges) == 0:
                return edges

            edges["road_class"] = "major"

            return edges

        except Exception as e:
            print(f"Warning: Could not extract motorways: {e}")
            return None

    @staticmethod
    def _normalize_ref(ref_value) -> Optional[str]:
        """Normalize a road ref tag value.

        OSMnx sometimes returns ref as a list; this extracts the first element
        and returns it as a string.

        Args:
            ref_value: Raw ref value from OSM (str, list, or NaN)

        Returns:
            Normalized ref string, or None
        """
        if ref_value is None:
            return None
        if isinstance(ref_value, list):
            return str(ref_value[0]) if ref_value else None
        if isinstance(ref_value, str):
            return ref_value
        # Handle NaN/float values
        try:
            import math
            if math.isnan(ref_value):
                return None
        except (TypeError, ValueError):
            pass
        return str(ref_value)

    def _classify_road(self, highway_type) -> str:
        """Classify road type for rendering."""
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        if highway_type in self.ROAD_TYPES["major"]:
            return "major"
        elif highway_type in self.ROAD_TYPES["minor"]:
            return "minor"
        else:
            return "other"

    def extract_buildings(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract building footprints from OSM.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with building polygons
        """
        try:
            buildings = self._features(bbox, {"building": True})

            # Filter to polygons only
            buildings = buildings[
                buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
            ]

            # Extract building type if available
            if "building" in buildings.columns:
                buildings["building_type"] = buildings["building"].fillna("yes")
            else:
                buildings["building_type"] = "yes"

            # Real-world height in meters, from an explicit ``height`` tag or
            # derived from ``building:levels``; NaN where neither is tagged.
            buildings = buildings.copy()
            buildings["height_m"] = buildings.apply(self._building_height_m, axis=1)

            return buildings

        except Exception as e:
            print(f"Warning: Could not extract buildings: {e}")
            return None

    def _building_height_m(self, row) -> float:
        """Meters of building height from OSM tags, or NaN when untagged.

        ``height`` is metric (may carry a "m"/"ft" unit suffix);
        ``building:levels`` is a storey count converted at ~3.2 m/level.
        """
        import math

        h = _parse_metric_height(row.get("height"))
        if h is None:
            levels = _parse_float(row.get("building:levels"))
            if levels is not None and levels > 0:
                h = levels * 3.2
        if h is None or h <= 0:
            return math.nan
        return h

    def extract_notable_buildings(
        self,
        bbox: BoundingBox,
        min_area_sqm: float = 5000,
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only notable buildings (named, historic, large, or tourist attractions).

        Args:
            bbox: Bounding box
            min_area_sqm: Minimum building area in square meters to include

        Returns:
            GeoDataFrame with notable building polygons
        """
        try:
            # Tags that indicate a notable building
            notable_tags = {
                "building": True,
            }

            buildings = self._features(bbox, notable_tags)

            # Filter to polygons only
            buildings = buildings[
                buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
            ]

            if len(buildings) == 0:
                return buildings

            # Calculate area (approximate, in square degrees)
            # We'll filter by multiple criteria
            buildings = buildings.copy()

            # Check for notable attributes
            has_name = buildings.get("name", "").notna() if "name" in buildings.columns else False
            has_historic = buildings.get("historic", "").notna() if "historic" in buildings.columns else False
            has_tourism = buildings.get("tourism", "").notna() if "tourism" in buildings.columns else False
            has_amenity = buildings.get("amenity", "").notna() if "amenity" in buildings.columns else False

            # Check for large buildings by geometry area
            # Convert to UTM for accurate area calculation
            try:
                buildings_projected = buildings.to_crs(epsg=3857)  # Web Mercator
                area_sqm = buildings_projected.geometry.area
                is_large = area_sqm > min_area_sqm
            except Exception:
                # Fallback: use degree-based area (very approximate)
                area_deg = buildings.geometry.area
                is_large = area_deg > (min_area_sqm / 1e10)  # Very rough conversion

            # Combine criteria: notable if named OR historic OR tourism OR large
            is_notable = has_name | has_historic | has_tourism | has_amenity | is_large

            # Filter to notable buildings
            notable_buildings = buildings[is_notable].copy()

            # Extract building type
            if "building" in notable_buildings.columns:
                notable_buildings["building_type"] = notable_buildings["building"].fillna("yes")
            else:
                notable_buildings["building_type"] = "yes"

            # Add name for reference
            if "name" in notable_buildings.columns:
                notable_buildings["display_name"] = notable_buildings["name"]
            else:
                notable_buildings["display_name"] = None

            print(f"Found {len(notable_buildings)} notable buildings out of {len(buildings)} total")

            return notable_buildings

        except Exception as e:
            print(f"Warning: Could not extract notable buildings: {e}")
            return None

    def extract_landmark_buildings(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only famous landmark buildings for regional-scale maps.

        Only includes buildings with tourism, historic, or famous names.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with landmark building polygons
        """
        try:
            # Tags that indicate a landmark
            landmark_tags = {
                "tourism": ["attraction", "museum", "monument", "artwork"],
                "historic": ["monument", "memorial", "castle", "ruins", "fort"],
                "building": ["cathedral", "church", "mosque", "temple", "stadium"],
            }

            landmarks = self._features(bbox, landmark_tags)

            # Filter to polygons and points
            landmarks = landmarks[
                landmarks.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])
            ]

            if len(landmarks) > 0 and "name" in landmarks.columns:
                # Prioritize named landmarks
                landmarks = landmarks[landmarks["name"].notna()]
                landmarks["display_name"] = landmarks["name"]

            print(f"Found {len(landmarks)} landmark buildings")
            return landmarks

        except Exception as e:
            print(f"Warning: Could not extract landmark buildings: {e}")
            return None

    def extract_major_cities(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract major cities/towns for country-scale maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with city points
        """
        try:
            city_tags = {
                "place": ["city", "town"],
            }

            cities = self._features(bbox, city_tags)

            if len(cities) > 0:
                # Filter to only named places
                if "name" in cities.columns:
                    cities = cities[cities["name"].notna()].copy()
                    cities["display_name"] = cities["name"]

                # Get population if available for filtering
                if "population" in cities.columns:
                    # Convert to numeric, drop non-numeric
                    cities["population"] = gpd.pd.to_numeric(
                        cities["population"], errors="coerce"
                    )
                    # Keep larger cities (> 50,000) or all if no population data
                    has_pop = cities["population"].notna()
                    if has_pop.any():
                        cities = cities[
                            ~has_pop | (cities["population"] > 50000)
                        ]

            print(f"Found {len(cities)} major cities")
            return cities

        except Exception as e:
            print(f"Warning: Could not extract major cities: {e}")
            return None

    def extract_water(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract water bodies and waterways from OSM.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with water features
        """
        try:
            # Get water bodies
            water_tags = {
                "natural": ["water", "bay", "strait"],
                "waterway": ["river", "stream", "canal"],
                "landuse": ["reservoir"],
            }

            water = self._features(bbox, water_tags)

            # Classify water type
            water["water_type"] = "water"
            if "waterway" in water.columns:
                water.loc[water["waterway"].notna(), "water_type"] = "waterway"

            return water

        except Exception as e:
            print(f"Warning: Could not extract water: {e}")
            return None

    def extract_major_water(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only major water bodies for regional-scale maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with major water features
        """
        try:
            water_tags = {
                "natural": ["water", "bay", "strait"],
                "waterway": ["river"],  # Only rivers, not streams/canals
            }

            water = self._features(bbox, water_tags)

            water["water_type"] = "water"
            if "waterway" in water.columns:
                water.loc[water["waterway"].notna(), "water_type"] = "waterway"

            return water

        except Exception as e:
            print(f"Warning: Could not extract major water: {e}")
            return None

    def extract_coastline_and_major_rivers(
        self, bbox: BoundingBox
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Extract coastlines and major rivers for country-scale maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with coastline and major river features
        """
        try:
            water_tags = {
                "natural": ["coastline", "water", "bay"],
                "waterway": ["river"],
            }

            water = self._features(bbox, water_tags)

            # Filter to only named rivers (major ones have names)
            if "waterway" in water.columns and "name" in water.columns:
                is_river = water["waterway"] == "river"
                has_name = water["name"].notna()
                # Keep all non-rivers OR rivers with names
                water = water[~is_river | has_name]

            water["water_type"] = "water"
            if "waterway" in water.columns:
                water.loc[water["waterway"].notna(), "water_type"] = "river"
            if "natural" in water.columns:
                water.loc[water["natural"] == "coastline", "water_type"] = "coastline"

            return water

        except Exception as e:
            print(f"Warning: Could not extract coastline and rivers: {e}")
            return None

    def extract_parks(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract parks and green spaces from OSM.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with park polygons
        """
        try:
            park_tags = {
                "leisure": ["park", "garden", "nature_reserve", "golf_course"],
                "landuse": ["recreation_ground", "village_green"],
            }

            parks = self._features(bbox, park_tags)

            # Filter to polygons
            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]

            return parks

        except Exception as e:
            print(f"Warning: Could not extract parks: {e}")
            return None

    def extract_major_parks(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract only major parks (national parks, nature reserves) for regional maps.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with major park polygons
        """
        try:
            park_tags = {
                "leisure": ["nature_reserve"],
                "boundary": ["national_park", "protected_area"],
            }

            parks = self._features(bbox, park_tags)

            # Filter to polygons
            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]

            # Filter to named parks (major ones have names)
            if "name" in parks.columns:
                parks = parks[parks["name"].notna()]

            print(f"Found {len(parks)} major parks")
            return parks

        except Exception as e:
            print(f"Warning: Could not extract major parks: {e}")
            return None

    def extract_terrain_types(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract terrain type information from OSM landuse/natural tags.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with terrain type polygons
        """
        try:
            terrain_tags = {
                "landuse": [
                    "residential",
                    "commercial",
                    "industrial",
                    "forest",
                    "grass",
                    "meadow",
                    "farmland",
                    "orchard",
                    "vineyard",
                    "cemetery",
                ],
                "natural": [
                    "wood",
                    "grassland",
                    "scrub",
                    "heath",
                    "sand",
                    "beach",
                    "wetland",
                    "rock",
                    "bare_rock",
                ],
                "amenity": ["grave_yard"],
            }

            terrain = self._features(bbox, terrain_tags)

            # Filter to polygons
            terrain = terrain[terrain.geometry.type.isin(["Polygon", "MultiPolygon"])]

            # Classify terrain
            terrain["terrain_class"] = terrain.apply(self._classify_terrain, axis=1)

            return terrain

        except Exception as e:
            print(f"Warning: Could not extract terrain types: {e}")
            return None

    def _classify_terrain(self, row) -> str:
        """Classify terrain type from OSM tags."""
        landuse = row.get("landuse")
        natural = row.get("natural")
        amenity = row.get("amenity")

        # Check against terrain categories
        for terrain_type, tags in self.TERRAIN_TAGS.items():
            if "landuse" in tags and landuse in tags["landuse"]:
                return terrain_type
            if "natural" in tags and natural in tags["natural"]:
                return terrain_type
            if "amenity" in tags and amenity in tags["amenity"]:
                return terrain_type

        return "other"

    def extract_railways(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract railway lines from OSM.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with railway geometries
        """
        try:
            railways = self._features(bbox, {"railway": ["rail", "subway", "light_rail", "tram"]})

            return railways

        except Exception as e:
            print(f"Warning: Could not extract railways: {e}")
            return None

    def extract_washes(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract dry washes, arroyos, canals, and intermittent waterways from OSM.

        These are critical features in desert landscapes (e.g., Arizona) that
        are not captured by the standard water extraction.

        Args:
            bbox: Bounding box

        Returns:
            GeoDataFrame with wash/arroyo geometries and a 'wash_type' column
        """
        try:
            # Query for desert waterway types
            wash_tags = {
                "waterway": ["ditch", "drain", "wadi", "canal"],
            }
            washes = self._features(bbox, wash_tags)

            # Also query for intermittent waterways
            try:
                intermittent = self._features(bbox, {"intermittent": ["yes"]})
                if intermittent is not None and len(intermittent) > 0:
                    import pandas as pd
                    washes = pd.concat([washes, intermittent]).drop_duplicates(
                        subset="geometry"
                    )
            except Exception:
                pass  # Intermittent query may fail; that's fine

            if washes is None or len(washes) == 0:
                return None

            washes = washes.copy()

            # Classify wash subtypes
            def _classify_wash(row):
                ww = row.get("waterway", "")
                if isinstance(ww, list):
                    ww = ww[0] if ww else ""
                if ww == "wadi":
                    return "arroyo"
                elif ww == "canal":
                    return "canal"
                elif ww in ("ditch", "drain"):
                    return "ditch"
                elif row.get("intermittent") == "yes":
                    return "intermittent"
                return "wash"

            washes["wash_type"] = washes.apply(_classify_wash, axis=1)

            return washes

        except Exception as e:
            print(f"Warning: Could not extract washes: {e}")
            return None

    def extract_rivers(self, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """
        Extract named river/canal centerlines (waterway lines).

        Complements extract_water (polygons) and extract_washes (minor desert
        waterways): inland rivers are usually mapped as waterway=river lines,
        which neither of those queries returns.
        """
        try:
            rivers = self._features(bbox, {"waterway": ["river", "canal"]})
            if rivers is None or len(rivers) == 0:
                return None
            rivers = rivers[rivers.geometry.type.isin(["LineString", "MultiLineString"])]
            return rivers if len(rivers) > 0 else None
        except Exception as e:
            print(f"Warning: Could not extract rivers: {e}")
            return None

    def get_city_boundary(self, city_name: str) -> Optional[Polygon]:
        """
        Get boundary polygon for a city by name.

        Args:
            city_name: City name (e.g., "Manhattan, New York, USA")

        Returns:
            Polygon of city boundary
        """
        try:
            gdf = ox.geocode_to_gdf(city_name)
            if len(gdf) > 0:
                geom = gdf.iloc[0].geometry
                if isinstance(geom, (Polygon, MultiPolygon)):
                    return geom
            return None

        except Exception as e:
            print(f"Warning: Could not get city boundary: {e}")
            return None

    def geocode_location(self, query: str) -> Optional[tuple[float, float]]:
        """
        Geocode a location query to coordinates.

        Args:
            query: Location query (e.g., "Empire State Building, NYC")

        Returns:
            (latitude, longitude) tuple
        """
        try:
            location = ox.geocode(query)
            return location  # Returns (lat, lon)

        except Exception as e:
            print(f"Warning: Could not geocode '{query}': {e}")
            return None
