"""Drop-in stub classes for all external services.

Each stub matches the real service constructor + method signatures exactly,
returning deterministic labeled images and data structures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import LineString

from mapgen.models.project import BoundingBox
from mapgen.services.gemini_service import GenerationResult
from mapgen.services.osm_service import OSMData
from mapgen.services.terrain_service import ElevationData

from .stub_images import create_numbered_tile, create_satellite_stub


# ---------------------------------------------------------------------------
# GeminiService stub
# ---------------------------------------------------------------------------

class StubGeminiService:
    """Stub for GeminiService — returns labeled numbered tiles."""

    def __init__(self, api_key: Optional[str] = None, model: str = "stub"):
        self.api_key = api_key
        self.model = model

    def _result(self, img: Image.Image, prompt: str = "stub") -> GenerationResult:
        return GenerationResult(
            image=img,
            prompt_used=prompt,
            model=self.model,
            generation_time=0.01,
            tokens_used=10,
        )

    def generate_base_tile(
        self,
        reference_image: Image.Image,
        style_prompt: Optional[str] = None,
        terrain_description: Optional[str] = None,
        tile_position: Optional[str] = None,
        style_reference: Optional[Image.Image] = None,
    ) -> GenerationResult:
        # Parse tile_position like "top-left (col 0, row 0)" to get col/row
        label = "FLAT"
        if tile_position:
            import re
            m = re.search(r"col\s*(\d+).*row\s*(\d+)", tile_position)
            if m:
                label = f"FLAT({m.group(1)},{m.group(2)})"
        w, h = reference_image.size
        return self._result(create_numbered_tile(w, h, label))

    def generate_overview(
        self,
        reference_image: Image.Image,
        style_prompt: Optional[str] = None,
        terrain_description: Optional[str] = None,
    ) -> GenerationResult:
        w, h = reference_image.size
        return self._result(create_numbered_tile(w, h, "OVERVIEW"))

    def generate_enhanced_tile(
        self,
        illustration_crop: Image.Image,
        reference_image: Image.Image,
        level: str = "medium",
        terrain_description: Optional[str] = None,
        tile_position: Optional[str] = None,
    ) -> GenerationResult:
        import re
        w, h = reference_image.size
        lvl = "L1" if level == "medium" else "L2"
        col, row = 0, 0
        if tile_position:
            m = re.search(r"col\s*(\d+).*row\s*(\d+)", tile_position)
            if m:
                col, row = int(m.group(1)), int(m.group(2))
        label = f"{lvl}({col},{row})"
        return self._result(create_numbered_tile(w, h, label))

    def inpaint_seam(
        self,
        seam_region: Image.Image,
        orientation: str = "horizontal",
    ) -> GenerationResult:
        w, h = seam_region.size
        label = "SEAM-H" if orientation == "horizontal" else "SEAM-V"
        return self._result(create_numbered_tile(w, h, label))

    def stylize_landmark(
        self,
        photo: Image.Image,
        style_reference: Optional[Image.Image] = None,
        landmark_name: Optional[str] = None,
        style_prompt: Optional[str] = None,
    ) -> GenerationResult:
        w, h = photo.size
        name_part = (landmark_name or "UNK")[:12]
        return self._result(create_numbered_tile(w, h, f"LM-{name_part}"))

    def generate_text_to_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> GenerationResult:
        return self._result(create_numbered_tile(width, height, "OUTPAINT"))

    def estimate_cost(self, num_tiles: int, num_landmarks: int) -> dict:
        return {
            "total_cost_usd": 0.0,
            "tile_cost": 0.0,
            "landmark_cost": 0.0,
            "num_tiles": num_tiles,
            "num_landmarks": num_landmarks,
        }

    @staticmethod
    def detect_terrain_modifier(terrain_description: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# SatelliteService stub
# ---------------------------------------------------------------------------

class StubSatelliteService:
    """Stub for SatelliteService — returns labeled satellite images."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.access_token = access_token
        self.cache_dir = cache_dir

    def fetch_satellite_imagery(
        self,
        bbox: BoundingBox,
        zoom: Optional[int] = None,
        output_size: Optional[tuple[int, int]] = None,
        progress_callback: Optional[Callable] = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image:
        if output_size:
            w, h = output_size
        elif width and height:
            w, h = width, height
        else:
            w, h = 512, 512
        return create_satellite_stub(w, h)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# OSMService stub
# ---------------------------------------------------------------------------

class StubOSMService:
    """Stub for OSMService — returns near-empty but valid OSM data."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def _make_roads(self, bbox: BoundingBox) -> gpd.GeoDataFrame:
        """Two road lines within the bbox for label/shield tests."""
        mid_lat = (bbox.north + bbox.south) / 2
        mid_lon = (bbox.east + bbox.west) / 2
        roads = gpd.GeoDataFrame(
            {
                "name": ["Main Street", "Broadway"],
                "highway": ["primary", "secondary"],
                "ref": ["US 1", "SR 9"],
                "geometry": [
                    LineString([(bbox.west, mid_lat), (bbox.east, mid_lat)]),
                    LineString([(mid_lon, bbox.south), (mid_lon, bbox.north)]),
                ],
            },
            crs="EPSG:4326",
        )
        return roads

    def _make_buildings(self, bbox: BoundingBox) -> gpd.GeoDataFrame:
        """A couple of notable buildings for landmark discovery."""
        from shapely.geometry import box

        mid_lat = (bbox.north + bbox.south) / 2
        mid_lon = (bbox.east + bbox.west) / 2
        d = 0.001
        return gpd.GeoDataFrame(
            {
                "name": ["Test Museum", "Old Castle"],
                "tourism": ["museum", None],
                "historic": [None, "castle"],
                "geometry": [
                    box(mid_lon - d, mid_lat - d, mid_lon + d, mid_lat + d),
                    box(mid_lon + d, mid_lat + d, mid_lon + 2 * d, mid_lat + 2 * d),
                ],
            },
            crs="EPSG:4326",
        )

    def _osm_data(self, bbox: BoundingBox) -> OSMData:
        return OSMData(
            roads=self._make_roads(bbox),
            buildings=self._make_buildings(bbox),
            water=None,
            parks=None,
            terrain_types=None,
            railways=None,
            pois=None,
            washes=None,
        )

    def fetch_region_data(
        self,
        bbox: BoundingBox,
        detail_level: str = "full",
        progress_callback: Optional[Callable] = None,
    ) -> OSMData:
        return self._osm_data(bbox)

    def fetch_simplified_region_data(self, bbox: BoundingBox, verbose: bool = True, progress_callback=None) -> OSMData:
        return self._osm_data(bbox)

    def fetch_regional_region_data(self, bbox: BoundingBox, progress_callback=None) -> OSMData:
        return self._osm_data(bbox)

    def fetch_country_region_data(self, bbox: BoundingBox, progress_callback=None) -> OSMData:
        return self._osm_data(bbox)

    def extract_roads(self, bbox: BoundingBox):
        return self._make_roads(bbox)

    def extract_major_roads(self, bbox: BoundingBox):
        return self._make_roads(bbox)

    def extract_primary_roads(self, bbox: BoundingBox):
        return self._make_roads(bbox)

    def extract_motorways_only(self, bbox: BoundingBox):
        return self._make_roads(bbox)

    def extract_buildings(self, bbox: BoundingBox):
        return self._make_buildings(bbox)

    def extract_notable_buildings(self, bbox: BoundingBox, min_area_sqm: float = 5000):
        return self._make_buildings(bbox)

    def extract_landmark_buildings(self, bbox: BoundingBox):
        return self._make_buildings(bbox)

    def extract_major_cities(self, bbox: BoundingBox):
        return None

    def extract_water(self, bbox: BoundingBox):
        return None

    def extract_major_water(self, bbox: BoundingBox):
        return None

    def extract_coastline_and_major_rivers(self, bbox: BoundingBox):
        return None

    def extract_parks(self, bbox: BoundingBox):
        return None

    def extract_major_parks(self, bbox: BoundingBox):
        return None

    def extract_terrain_types(self, bbox: BoundingBox):
        return None

    def extract_railways(self, bbox: BoundingBox):
        return None

    def extract_washes(self, bbox: BoundingBox):
        return None

    def get_city_boundary(self, city_name: str):
        return None

    def geocode_location(self, query: str):
        return None


# ---------------------------------------------------------------------------
# TerrainService stub
# ---------------------------------------------------------------------------

class StubTerrainService:
    """Stub for TerrainService — flat DEM at 500m."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def fetch_elevation_data(
        self,
        bbox: BoundingBox,
        resolution: int = 30,
    ) -> ElevationData:
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        return ElevationData(
            dem=dem,
            bbox=bbox,
            resolution=(30.0, 30.0),
        )

    def get_terrain_description(self, elevation_data: ElevationData) -> str:
        return "Flat terrain, average elevation 500m"

    def compute_hillshade(self, elevation_data, azimuth=315, altitude=45, vertical_exaggeration=1.0):
        return np.full(elevation_data.dem.shape, 180, dtype=np.uint8)

    def compute_slope(self, elevation_data):
        return np.zeros(elevation_data.dem.shape, dtype=np.float32)

    def detect_terrain_features(self, elevation_data, min_prominence=50):
        return []

    def elevation_to_image(self, elevation_data, colormap="terrain"):
        h, w = elevation_data.dem.shape
        return Image.new("RGBA", (w, h), (128, 128, 128, 255))

    def hillshade_to_image(self, elevation_data, azimuth=315, altitude=45):
        h, w = elevation_data.dem.shape
        return Image.new("RGBA", (w, h), (180, 180, 180, 255))


# ---------------------------------------------------------------------------
# WikipediaImageService stub
# ---------------------------------------------------------------------------

class StubWikipediaImageService:
    """Stub for WikipediaImageService — returns labeled images."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def fetch_image_for_landmark(
        self,
        name: str,
        wikipedia_url: Optional[str] = None,
        wikidata_id: Optional[str] = None,
        target_size: int = 800,
    ) -> Optional[Image.Image]:
        short = name[:12].replace(" ", "")
        img = create_numbered_tile(target_size, target_size, f"WIKI-{short}")
        return img.convert("RGB")  # Endpoint saves as JPEG


# ---------------------------------------------------------------------------
# LandmarkService stub (used in illustrate endpoints)
# ---------------------------------------------------------------------------

@dataclass
class StubLandmarkResult:
    image: Optional[Image.Image] = None
    error: Optional[str] = None


class StubLandmarkService:
    """Stub for LandmarkService — returns labeled illustrations."""

    def __init__(self, project=None, gemini_service=None):
        self.project = project
        self.gemini_service = gemini_service

    def illustrate_landmark(self, landmark, style_reference):
        name_part = (landmark.name or "UNK")[:12]
        img = create_numbered_tile(512, 512, f"LM-{name_part}")
        return StubLandmarkResult(image=img)


# ---------------------------------------------------------------------------
# OutpaintingService stub
# ---------------------------------------------------------------------------

class StubOutpaintingService:
    """Stub for OutpaintingService — returns slightly expanded image."""

    def __init__(self, convergence=0.7, vertical_scale=0.4, horizon_margin=0.15,
                 max_gemini_size=2048, fill_color=(128, 128, 128), gemini_service=None):
        self.gemini_service = gemini_service

    def outpaint_image(self, image, bbox, output_path=None, progress_callback=None):
        result = create_numbered_tile(image.width, image.height, "OUTPAINT")
        if output_path:
            result.save(str(output_path))
        return result

    def estimate_cost(self):
        return {"total_cost_usd": 0.0, "calls": 1}
