"""OpenStreetMap data extraction service."""

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon

from ..models.project import BoundingBox


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
        "water": {
            "natural": ["water", "wetland"],
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

    def fetch_region_data(
        self,
        bbox: BoundingBox,
        detail_level: str = "full",
    ) -> OSMData:
        """
        Fetch OSM data for a region.

        Args:
            bbox: Bounding box defining the region
            detail_level: "full" (all data), "simplified" (major features only)

        Returns:
            OSMData containing extracted layers
        """
        if detail_level == "simplified":
            return self.fetch_simplified_region_data(bbox)

        data = OSMData()

        # Fetch each data type
        data.roads = self.extract_roads(bbox)
        data.buildings = self.extract_buildings(bbox)
        data.water = self.extract_water(bbox)
        data.parks = self.extract_parks(bbox)
        data.terrain_types = self.extract_terrain_types(bbox)
        data.railways = self.extract_railways(bbox)

        return data

    def fetch_simplified_region_data(self, bbox: BoundingBox) -> OSMData:
        """
        Fetch simplified OSM data - major features only for illustrated maps.

        Only includes:
        - Major roads (highways, primary, secondary)
        - Notable buildings (named, historic, tourism)
        - All water bodies
        - All parks

        Args:
            bbox: Bounding box defining the region

        Returns:
            OSMData with simplified layers
        """
        data = OSMData()

        data.roads = self.extract_major_roads(bbox)
        data.buildings = self.extract_notable_buildings(bbox)
        data.water = self.extract_water(bbox)
        data.parks = self.extract_parks(bbox)
        # Skip railways and detailed terrain for simplified mode

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
            # Get road network graph
            G = ox.graph_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                network_type="drive",
                simplify=True,
            )

            # Convert to GeoDataFrame
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

            # Add road classification
            edges["road_class"] = edges["highway"].apply(self._classify_road)

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

            # Get road network with custom filter
            custom_filter = (
                f'["highway"~"{'|'.join(major_highway_types)}"]'
            )

            G = ox.graph_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                network_type="drive",
                simplify=True,
                custom_filter=custom_filter,
            )

            # Convert to GeoDataFrame
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

            # Add road classification
            edges["road_class"] = edges["highway"].apply(self._classify_road)

            return edges

        except Exception as e:
            print(f"Warning: Could not extract major roads: {e}")
            return None

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
            buildings = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags={"building": True},
            )

            # Filter to polygons only
            buildings = buildings[
                buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
            ]

            # Extract building type if available
            if "building" in buildings.columns:
                buildings["building_type"] = buildings["building"].fillna("yes")
            else:
                buildings["building_type"] = "yes"

            return buildings

        except Exception as e:
            print(f"Warning: Could not extract buildings: {e}")
            return None

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

            buildings = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags=notable_tags,
            )

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

            water = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags=water_tags,
            )

            # Classify water type
            water["water_type"] = "water"
            if "waterway" in water.columns:
                water.loc[water["waterway"].notna(), "water_type"] = "waterway"

            return water

        except Exception as e:
            print(f"Warning: Could not extract water: {e}")
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
                "leisure": ["park", "garden", "nature_reserve"],
                "landuse": ["recreation_ground", "village_green"],
            }

            parks = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags=park_tags,
            )

            # Filter to polygons
            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]

            return parks

        except Exception as e:
            print(f"Warning: Could not extract parks: {e}")
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
                ],
                "natural": [
                    "wood",
                    "grassland",
                    "scrub",
                    "heath",
                    "sand",
                    "beach",
                    "rock",
                    "bare_rock",
                ],
            }

            terrain = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags=terrain_tags,
            )

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

        # Check against terrain categories
        for terrain_type, tags in self.TERRAIN_TAGS.items():
            if "landuse" in tags and landuse in tags["landuse"]:
                return terrain_type
            if "natural" in tags and natural in tags["natural"]:
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
            railways = ox.features_from_bbox(
                bbox=bbox.to_osmnx_bbox(),
                tags={"railway": ["rail", "subway", "light_rail", "tram"]},
            )

            return railways

        except Exception as e:
            print(f"Warning: Could not extract railways: {e}")
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
