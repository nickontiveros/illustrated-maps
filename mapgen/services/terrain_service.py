"""Terrain and elevation data service using SRTM/DEM data."""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..models.project import BoundingBox


@dataclass
class ElevationData:
    """Container for elevation data."""

    dem: np.ndarray  # Digital Elevation Model (heights in meters)
    bbox: BoundingBox
    resolution: tuple[float, float]  # meters per pixel (x, y)
    nodata_value: float = -9999

    @property
    def min_elevation(self) -> float:
        """Minimum elevation excluding nodata."""
        valid = self.dem[self.dem != self.nodata_value]
        return float(np.min(valid)) if len(valid) > 0 else 0

    @property
    def max_elevation(self) -> float:
        """Maximum elevation excluding nodata."""
        valid = self.dem[self.dem != self.nodata_value]
        return float(np.max(valid)) if len(valid) > 0 else 0

    @property
    def elevation_range(self) -> float:
        """Range of elevation values."""
        return self.max_elevation - self.min_elevation


@dataclass
class TerrainFeature:
    """Detected terrain feature (peak, valley, ridge)."""

    feature_type: str  # "peak", "valley", "ridge", "slope"
    latitude: float
    longitude: float
    elevation: float
    prominence: Optional[float] = None  # For peaks
    name: Optional[str] = None


class TerrainService:
    """Service for fetching and processing terrain/elevation data."""

    # SRTM data has 30m resolution
    SRTM_RESOLUTION = 30  # meters

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize terrain service.

        Args:
            cache_dir: Directory to cache downloaded elevation data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "mapgen_dem"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_elevation_data(
        self,
        bbox: BoundingBox,
        resolution: int = 30,
    ) -> ElevationData:
        """
        Fetch elevation data for a region.

        Uses the `elevation` library to download SRTM data.

        Args:
            bbox: Bounding box for the region
            resolution: Desired resolution in meters (default 30m SRTM)

        Returns:
            ElevationData with DEM array
        """
        try:
            import elevation
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            # Create bounds string for elevation library
            bounds = (bbox.west, bbox.south, bbox.east, bbox.north)

            # Output file path
            dem_path = self.cache_dir / f"dem_{bbox.west}_{bbox.south}_{bbox.east}_{bbox.north}.tif"

            if not dem_path.exists():
                # Download SRTM data
                elevation.clip(bounds=bounds, output=str(dem_path))

            # Read the DEM
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
                nodata = src.nodata or -9999

                # Calculate resolution in meters
                transform = src.transform
                pixel_width = abs(transform.a)  # degrees
                pixel_height = abs(transform.e)

                # Convert to meters (approximate at center latitude)
                center_lat = (bbox.north + bbox.south) / 2
                meters_per_deg = 111320 * math.cos(math.radians(center_lat))
                res_x = pixel_width * meters_per_deg
                res_y = pixel_height * 111320  # latitude is constant

            return ElevationData(
                dem=dem.astype(np.float32),
                bbox=bbox,
                resolution=(res_x, res_y),
                nodata_value=nodata,
            )

        except ImportError:
            print("Warning: 'elevation' or 'rasterio' not installed. Using flat terrain.")
            return self._create_flat_terrain(bbox)

        except Exception as e:
            print(f"Warning: Could not fetch elevation data: {e}. Using flat terrain.")
            return self._create_flat_terrain(bbox)

    def _create_flat_terrain(self, bbox: BoundingBox, elevation: float = 0) -> ElevationData:
        """Create flat terrain as fallback."""
        # Create a small array representing flat terrain
        dem = np.full((100, 100), elevation, dtype=np.float32)
        return ElevationData(
            dem=dem,
            bbox=bbox,
            resolution=(100, 100),
            nodata_value=-9999,
        )

    def compute_hillshade(
        self,
        elevation_data: ElevationData,
        azimuth: float = 315,
        altitude: float = 45,
    ) -> np.ndarray:
        """
        Compute hillshade from elevation data.

        Args:
            elevation_data: Elevation data
            azimuth: Light source azimuth in degrees (0=North, 90=East, etc.)
            altitude: Light source altitude in degrees above horizon

        Returns:
            Hillshade array (0-255)
        """
        dem = elevation_data.dem
        res_x, res_y = elevation_data.resolution

        # Replace nodata with mean elevation
        dem_filled = dem.copy()
        valid_mask = dem != elevation_data.nodata_value
        if valid_mask.any():
            dem_filled[~valid_mask] = np.mean(dem[valid_mask])

        # Calculate gradients
        dx = np.gradient(dem_filled, res_x, axis=1)
        dy = np.gradient(dem_filled, res_y, axis=0)

        # Calculate slope and aspect
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)

        # Convert azimuth and altitude to radians
        azimuth_rad = math.radians(azimuth)
        altitude_rad = math.radians(altitude)

        # Calculate hillshade
        hillshade = (
            np.sin(altitude_rad) * np.cos(slope)
            + np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
        )

        # Normalize to 0-255
        hillshade = np.clip(hillshade, 0, 1)
        hillshade = (hillshade * 255).astype(np.uint8)

        return hillshade

    def compute_slope(self, elevation_data: ElevationData) -> np.ndarray:
        """
        Compute slope in degrees.

        Args:
            elevation_data: Elevation data

        Returns:
            Slope array in degrees (0-90)
        """
        dem = elevation_data.dem
        res_x, res_y = elevation_data.resolution

        # Replace nodata
        dem_filled = dem.copy()
        valid_mask = dem != elevation_data.nodata_value
        if valid_mask.any():
            dem_filled[~valid_mask] = np.mean(dem[valid_mask])

        # Calculate gradients
        dx = np.gradient(dem_filled, res_x, axis=1)
        dy = np.gradient(dem_filled, res_y, axis=0)

        # Calculate slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        return slope

    def detect_terrain_features(
        self,
        elevation_data: ElevationData,
        min_prominence: float = 50,
    ) -> list[TerrainFeature]:
        """
        Detect terrain features (peaks, ridges, valleys).

        Args:
            elevation_data: Elevation data
            min_prominence: Minimum prominence in meters for peaks

        Returns:
            List of detected terrain features
        """
        features = []
        dem = elevation_data.dem
        bbox = elevation_data.bbox

        # Replace nodata
        dem_filled = dem.copy()
        valid_mask = dem != elevation_data.nodata_value
        if not valid_mask.any():
            return features

        dem_filled[~valid_mask] = np.mean(dem[valid_mask])

        # Find local maxima (peaks)
        from scipy.ndimage import maximum_filter, minimum_filter

        # Use a neighborhood size proportional to resolution
        neighborhood_size = max(5, int(1000 / min(elevation_data.resolution)))

        # Find peaks
        local_max = maximum_filter(dem_filled, size=neighborhood_size) == dem_filled
        # Require minimum elevation difference from surroundings
        local_min_nearby = minimum_filter(dem_filled, size=neighborhood_size)
        prominence = dem_filled - local_min_nearby

        peak_mask = local_max & (prominence >= min_prominence) & valid_mask

        # Convert peak positions to coordinates
        peak_positions = np.where(peak_mask)
        for i, j in zip(peak_positions[0], peak_positions[1]):
            # Convert array indices to lat/lon
            lat = bbox.north - (i / dem.shape[0]) * (bbox.north - bbox.south)
            lon = bbox.west + (j / dem.shape[1]) * (bbox.east - bbox.west)

            features.append(
                TerrainFeature(
                    feature_type="peak",
                    latitude=lat,
                    longitude=lon,
                    elevation=float(dem_filled[i, j]),
                    prominence=float(prominence[i, j]),
                )
            )

        return features

    def get_terrain_description(self, elevation_data: ElevationData) -> str:
        """
        Generate a text description of the terrain for use in image prompts.

        Args:
            elevation_data: Elevation data

        Returns:
            Text description of terrain characteristics
        """
        dem = elevation_data.dem
        valid_mask = dem != elevation_data.nodata_value

        if not valid_mask.any():
            return "Flat terrain with no significant elevation changes."

        dem_valid = dem[valid_mask]

        min_elev = np.min(dem_valid)
        max_elev = np.max(dem_valid)
        mean_elev = np.mean(dem_valid)
        elev_range = max_elev - min_elev

        # Calculate slope statistics
        slope = self.compute_slope(elevation_data)
        mean_slope = np.mean(slope[valid_mask])
        max_slope = np.max(slope[valid_mask])

        # Detect terrain features
        features = self.detect_terrain_features(elevation_data)
        num_peaks = len([f for f in features if f.feature_type == "peak"])

        # Build description
        parts = []

        # Elevation range description
        if elev_range < 50:
            parts.append(f"Relatively flat terrain at approximately {mean_elev:.0f}m elevation")
        elif elev_range < 200:
            parts.append(
                f"Gently rolling terrain with elevations from {min_elev:.0f}m to {max_elev:.0f}m"
            )
        elif elev_range < 500:
            parts.append(f"Hilly terrain with elevations ranging from {min_elev:.0f}m to {max_elev:.0f}m")
        else:
            parts.append(
                f"Mountainous terrain with significant elevation changes from {min_elev:.0f}m to {max_elev:.0f}m"
            )

        # Slope description
        if mean_slope < 5:
            parts.append("mostly level surfaces")
        elif mean_slope < 15:
            parts.append("moderate slopes")
        else:
            parts.append("steep slopes")

        # Peak description
        if num_peaks > 0:
            parts.append(f"{num_peaks} notable peaks or hills")

        # Directional characteristics (where is high ground?)
        dem_filled = dem.copy()
        dem_filled[~valid_mask] = mean_elev

        # Check quadrants for elevation bias
        h, w = dem_filled.shape
        north_avg = np.mean(dem_filled[: h // 3, :])
        south_avg = np.mean(dem_filled[2 * h // 3 :, :])
        west_avg = np.mean(dem_filled[:, : w // 3])
        east_avg = np.mean(dem_filled[:, 2 * w // 3 :])

        max_dir = max(
            [("north", north_avg), ("south", south_avg), ("west", west_avg), ("east", east_avg)],
            key=lambda x: x[1],
        )

        if max_dir[1] > mean_elev + elev_range * 0.1:
            parts.append(f"higher ground to the {max_dir[0]}")

        return "; ".join(parts) + "."

    def elevation_to_image(
        self,
        elevation_data: ElevationData,
        colormap: str = "terrain",
    ) -> Image.Image:
        """
        Convert elevation data to a colored image.

        Args:
            elevation_data: Elevation data
            colormap: Matplotlib colormap name

        Returns:
            PIL Image with elevation visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        dem = elevation_data.dem.copy()
        valid_mask = dem != elevation_data.nodata_value

        # Normalize elevation values
        if valid_mask.any():
            vmin = np.min(dem[valid_mask])
            vmax = np.max(dem[valid_mask])
        else:
            vmin, vmax = 0, 1

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(colormap)

        # Apply colormap
        colored = cmap(norm(dem))

        # Set nodata areas to transparent
        colored[~valid_mask] = [0, 0, 0, 0]

        # Convert to PIL Image
        colored_uint8 = (colored * 255).astype(np.uint8)
        image = Image.fromarray(colored_uint8, mode="RGBA")

        return image

    def hillshade_to_image(
        self,
        elevation_data: ElevationData,
        azimuth: float = 315,
        altitude: float = 45,
    ) -> Image.Image:
        """
        Create hillshade image from elevation data.

        Args:
            elevation_data: Elevation data
            azimuth: Light source azimuth
            altitude: Light source altitude

        Returns:
            Grayscale PIL Image with hillshade
        """
        hillshade = self.compute_hillshade(elevation_data, azimuth, altitude)
        return Image.fromarray(hillshade, mode="L")
