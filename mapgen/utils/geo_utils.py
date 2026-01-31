"""Geographic and coordinate utilities."""

import math
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, box

from ..models.project import BoundingBox


def bbox_to_polygon(bbox: BoundingBox) -> Polygon:
    """Convert bounding box to Shapely polygon."""
    return box(bbox.west, bbox.south, bbox.east, bbox.north)


def calculate_aspect_ratio(bbox: BoundingBox) -> float:
    """Calculate aspect ratio (width/height) of bounding box in meters."""
    # Approximate conversion at center latitude
    center_lat = (bbox.north + bbox.south) / 2
    lat_rad = math.radians(center_lat)

    # Meters per degree at this latitude
    meters_per_deg_lat = 111320  # Approximately constant
    meters_per_deg_lon = 111320 * math.cos(lat_rad)

    width_m = bbox.width_degrees * meters_per_deg_lon
    height_m = bbox.height_degrees * meters_per_deg_lat

    return width_m / height_m if height_m > 0 else 1.0


def meters_per_pixel(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[float, float]:
    """Calculate meters per pixel for x and y dimensions."""
    center_lat = (bbox.north + bbox.south) / 2
    lat_rad = math.radians(center_lat)

    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * math.cos(lat_rad)

    width_m = bbox.width_degrees * meters_per_deg_lon
    height_m = bbox.height_degrees * meters_per_deg_lat

    return (width_m / image_width, height_m / image_height)


def gps_to_pixel(
    lat: float,
    lon: float,
    bbox: BoundingBox,
    image_size: tuple[int, int],
    isometric_matrix: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """
    Convert GPS coordinates to pixel position on map.

    Args:
        lat: Latitude
        lon: Longitude
        bbox: Map bounding box
        image_size: (width, height) in pixels
        isometric_matrix: Optional transformation matrix for isometric projection

    Returns:
        (x, y) pixel coordinates
    """
    width, height = image_size

    # Normalize to 0-1 within bbox
    x_norm = (lon - bbox.west) / (bbox.east - bbox.west)
    y_norm = (lat - bbox.south) / (bbox.north - bbox.south)

    if isometric_matrix is not None:
        # Apply isometric transformation
        point_3d = np.array([x_norm, y_norm, 0, 1])
        transformed = isometric_matrix @ point_3d
        x_norm = transformed[0]
        y_norm = transformed[1]

    # Convert to pixel coordinates
    pixel_x = int(x_norm * width)
    pixel_y = int((1 - y_norm) * height)  # Flip Y axis (image origin is top-left)

    return (pixel_x, pixel_y)


def pixel_to_gps(
    x: int,
    y: int,
    bbox: BoundingBox,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """
    Convert pixel position to GPS coordinates.

    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        bbox: Map bounding box
        image_size: (width, height) in pixels

    Returns:
        (latitude, longitude)
    """
    width, height = image_size

    # Normalize pixel to 0-1
    x_norm = x / width
    y_norm = 1 - (y / height)  # Flip Y axis

    # Convert to GPS
    lon = bbox.west + x_norm * (bbox.east - bbox.west)
    lat = bbox.south + y_norm * (bbox.north - bbox.south)

    return (lat, lon)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_map_scale(bbox: BoundingBox, image_width: int) -> float:
    """
    Calculate map scale (1:N) for the map.

    Returns the denominator N (e.g., 10000 means 1:10000 scale).
    """
    # Width of bbox in meters
    center_lat = (bbox.north + bbox.south) / 2
    lat_rad = math.radians(center_lat)
    meters_per_deg_lon = 111320 * math.cos(lat_rad)
    width_m = bbox.width_degrees * meters_per_deg_lon

    # Assume 96 DPI screen, so ~0.0254m per pixel at 100% zoom
    # For print at 300 DPI, 1 pixel = 0.0254/3 meters = 0.00847m
    meters_per_pixel = 0.0254 / 3  # 300 DPI

    # Scale = real_distance / map_distance
    real_meters_per_pixel = width_m / image_width
    scale = real_meters_per_pixel / meters_per_pixel

    return scale
