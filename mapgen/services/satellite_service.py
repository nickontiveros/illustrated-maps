"""Satellite imagery service using Mapbox."""

import math
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

from ..models.project import BoundingBox


class SatelliteService:
    """Service for fetching satellite/aerial imagery from Mapbox."""

    # Mapbox tile size is always 512x512 for raster tiles
    TILE_SIZE = 512

    # Mapbox satellite tileset
    TILESET = "mapbox.satellite"

    def __init__(
        self,
        access_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize satellite service.

        Args:
            access_token: Mapbox access token (or set MAPBOX_ACCESS_TOKEN env var)
            cache_dir: Directory to cache downloaded tiles
        """
        self.access_token = access_token or os.environ.get("MAPBOX_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError(
                "Mapbox access token required. Set MAPBOX_ACCESS_TOKEN environment "
                "variable or pass access_token parameter."
            )

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = httpx.Client(timeout=30.0)

    def fetch_satellite_imagery(
        self,
        bbox: BoundingBox,
        zoom: Optional[int] = None,
        output_size: Optional[tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Fetch satellite imagery for a bounding box.

        Args:
            bbox: Geographic bounding box
            zoom: Tile zoom level (auto-calculated if not provided)
            output_size: Optional (width, height) to resize result

        Returns:
            PIL Image with satellite imagery
        """
        # Calculate appropriate zoom level if not provided
        if zoom is None:
            zoom = self._calculate_zoom(bbox, output_size or (2048, 2048))

        # Get tile coordinates that cover the bbox
        tiles = self._get_tiles_for_bbox(bbox, zoom)

        # Download and stitch tiles
        stitched = self._download_and_stitch_tiles(tiles, zoom)

        # Crop to exact bbox
        cropped = self._crop_to_bbox(stitched, bbox, zoom)

        # Resize if output size specified
        if output_size:
            cropped = cropped.resize(output_size, Image.Resampling.LANCZOS)

        return cropped

    def _calculate_zoom(
        self,
        bbox: BoundingBox,
        target_size: tuple[int, int],
    ) -> int:
        """
        Calculate appropriate zoom level for desired resolution.

        Args:
            bbox: Bounding box
            target_size: Target (width, height) in pixels

        Returns:
            Zoom level (0-22)
        """
        target_width, target_height = target_size

        # Calculate degrees per pixel we want
        degrees_per_pixel_x = bbox.width_degrees / target_width
        degrees_per_pixel_y = bbox.height_degrees / target_height
        degrees_per_pixel = min(degrees_per_pixel_x, degrees_per_pixel_y)

        # At zoom 0, the world is 256 pixels (for standard tiles) or 512 (for @2x)
        # 360 degrees / 512 pixels = 0.703125 degrees per pixel
        # Each zoom level doubles the resolution

        world_degrees_per_pixel_z0 = 360 / self.TILE_SIZE

        # Find zoom level where tile resolution matches target
        zoom = math.log2(world_degrees_per_pixel_z0 / degrees_per_pixel)

        # Clamp to valid range and round
        zoom = max(0, min(18, int(zoom)))

        return zoom

    def _get_tiles_for_bbox(
        self,
        bbox: BoundingBox,
        zoom: int,
    ) -> list[tuple[int, int]]:
        """
        Get list of tile coordinates (x, y) that cover the bbox.

        Args:
            bbox: Bounding box
            zoom: Zoom level

        Returns:
            List of (x, y) tile coordinates
        """
        # Convert lat/lon to tile coordinates
        min_x = self._lon_to_tile_x(bbox.west, zoom)
        max_x = self._lon_to_tile_x(bbox.east, zoom)
        min_y = self._lat_to_tile_y(bbox.north, zoom)  # Note: y is inverted
        max_y = self._lat_to_tile_y(bbox.south, zoom)

        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append((x, y))

        return tiles

    def _lon_to_tile_x(self, lon: float, zoom: int) -> int:
        """Convert longitude to tile X coordinate."""
        n = 2 ** zoom
        return int((lon + 180) / 360 * n)

    def _lat_to_tile_y(self, lat: float, zoom: int) -> int:
        """Convert latitude to tile Y coordinate."""
        n = 2 ** zoom
        lat_rad = math.radians(lat)
        return int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)

    def _tile_to_lon(self, x: int, zoom: int) -> float:
        """Convert tile X coordinate to longitude."""
        n = 2 ** zoom
        return x / n * 360 - 180

    def _tile_to_lat(self, y: int, zoom: int) -> float:
        """Convert tile Y coordinate to latitude."""
        n = 2 ** zoom
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad)

    def _download_tile(self, x: int, y: int, zoom: int) -> Image.Image:
        """
        Download a single tile.

        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            zoom: Zoom level

        Returns:
            PIL Image of tile
        """
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{zoom}_{x}_{y}.png"
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")

        # Build Mapbox tile URL
        # Using @2x for higher resolution
        url = (
            f"https://api.mapbox.com/v4/{self.TILESET}/{zoom}/{x}/{y}@2x.png"
            f"?access_token={self.access_token}"
        )

        response = self._client.get(url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Cache the tile
        if self.cache_dir:
            image.save(cache_path)

        return image

    def _download_and_stitch_tiles(
        self,
        tiles: list[tuple[int, int]],
        zoom: int,
    ) -> Image.Image:
        """
        Download multiple tiles and stitch them together.

        Args:
            tiles: List of (x, y) tile coordinates
            zoom: Zoom level

        Returns:
            Stitched PIL Image
        """
        if not tiles:
            raise ValueError("No tiles to download")

        # Find bounds
        min_x = min(t[0] for t in tiles)
        max_x = max(t[0] for t in tiles)
        min_y = min(t[1] for t in tiles)
        max_y = max(t[1] for t in tiles)

        # Download first tile to get actual pixel size
        first_tile = self._download_tile(tiles[0][0], tiles[0][1], zoom)
        tile_pixel_size = first_tile.width  # Get actual tile size (512 for @2x)

        # Calculate output dimensions
        width = (max_x - min_x + 1) * tile_pixel_size
        height = (max_y - min_y + 1) * tile_pixel_size

        # Create output image
        stitched = Image.new("RGB", (width, height))

        # Paste first tile
        px = (tiles[0][0] - min_x) * tile_pixel_size
        py = (tiles[0][1] - min_y) * tile_pixel_size
        stitched.paste(first_tile, (px, py))

        # Download and paste remaining tiles
        for x, y in tiles[1:]:
            tile = self._download_tile(x, y, zoom)

            # Calculate position in output
            px = (x - min_x) * tile_pixel_size
            py = (y - min_y) * tile_pixel_size

            stitched.paste(tile, (px, py))

        # Store tile bounds for cropping
        self._last_tile_bounds = (min_x, min_y, max_x, max_y)
        self._last_zoom = zoom
        self._last_tile_pixel_size = tile_pixel_size

        return stitched

    def _crop_to_bbox(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        zoom: int,
    ) -> Image.Image:
        """
        Crop stitched image to exact bounding box.

        Args:
            image: Stitched tile image
            bbox: Target bounding box
            zoom: Zoom level used

        Returns:
            Cropped PIL Image
        """
        min_x, min_y, max_x, max_y = self._last_tile_bounds

        # Calculate the geographic bounds of the tile grid
        tile_west = self._tile_to_lon(min_x, zoom)
        tile_east = self._tile_to_lon(max_x + 1, zoom)
        tile_north = self._tile_to_lat(min_y, zoom)
        tile_south = self._tile_to_lat(max_y + 1, zoom)

        # Calculate pixel coordinates for bbox within tile grid
        tile_width_deg = tile_east - tile_west
        tile_height_deg = tile_north - tile_south

        left_pct = (bbox.west - tile_west) / tile_width_deg
        right_pct = (bbox.east - tile_west) / tile_width_deg
        top_pct = (tile_north - bbox.north) / tile_height_deg
        bottom_pct = (tile_north - bbox.south) / tile_height_deg

        # Convert to pixels
        left = int(left_pct * image.width)
        right = int(right_pct * image.width)
        top = int(top_pct * image.height)
        bottom = int(bottom_pct * image.height)

        # Clamp to image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)

        return image.crop((left, top, right, bottom))

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
