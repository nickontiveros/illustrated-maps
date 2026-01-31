"""Perspective transformation service for isometric map projection."""

import math
from typing import Optional, Union

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from ..models.project import BoundingBox


class PerspectiveService:
    """Service for applying isometric perspective transformations to map data."""

    # Standard isometric angle (arctan(1/sqrt(2)) ≈ 35.264 degrees)
    ISOMETRIC_ANGLE = 35.264

    def __init__(self, angle: float = ISOMETRIC_ANGLE):
        """
        Initialize perspective service.

        Args:
            angle: Viewing angle in degrees from horizontal
        """
        self.angle = angle
        self._matrix: Optional[np.ndarray] = None
        self._inverse_matrix: Optional[np.ndarray] = None

    def create_isometric_matrix(self, angle: Optional[float] = None) -> np.ndarray:
        """
        Create isometric projection matrix.

        The isometric projection simulates looking at the map from an angle,
        creating the "tilted aerial view" effect seen in Disneyland maps.

        Args:
            angle: Optional override for viewing angle

        Returns:
            4x4 transformation matrix
        """
        if angle is None:
            angle = self.angle

        angle_rad = math.radians(angle)

        # Rotation around X axis (tilt forward)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Isometric projection matrix
        # This rotates around the X axis to create the tilted view
        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, cos_a, -sin_a, 0],
                [0, sin_a, cos_a, 0],
                [0, 0, 0, 1],
            ]
        )

        # Optional: slight rotation around Z for more interesting angle
        # Using 45 degrees gives classic isometric look
        z_angle = math.radians(0)  # Set to 45 for true isometric
        cos_z = math.cos(z_angle)
        sin_z = math.sin(z_angle)

        Rz = np.array(
            [
                [cos_z, -sin_z, 0, 0],
                [sin_z, cos_z, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Combined transformation
        self._matrix = Rx @ Rz
        return self._matrix

    @property
    def matrix(self) -> np.ndarray:
        """Get or create the transformation matrix."""
        if self._matrix is None:
            self.create_isometric_matrix()
        return self._matrix

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Get inverse transformation matrix."""
        if self._inverse_matrix is None:
            self._inverse_matrix = np.linalg.inv(self.matrix)
        return self._inverse_matrix

    def transform_point(
        self,
        x: float,
        y: float,
        z: float = 0,
        elevation_scale: float = 0.0001,
    ) -> tuple[float, float]:
        """
        Transform a single point using isometric projection.

        Args:
            x: X coordinate (normalized 0-1 or pixel)
            y: Y coordinate (normalized 0-1 or pixel)
            z: Z coordinate (elevation in meters)
            elevation_scale: Scale factor for elevation

        Returns:
            Transformed (x, y) coordinates
        """
        # Scale elevation
        z_scaled = z * elevation_scale

        # Create homogeneous coordinate
        point = np.array([x, y, z_scaled, 1])

        # Apply transformation
        transformed = self.matrix @ point

        # Return x, y (ignore z after projection)
        return (transformed[0], transformed[1])

    def transform_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray] = None,
        elevation_scale: float = 0.00001,
    ) -> gpd.GeoDataFrame:
        """
        Transform a GeoDataFrame to isometric projection.

        Args:
            gdf: GeoDataFrame with geometries
            bbox: Bounding box for normalization
            elevation_data: Optional DEM array for 3D effect
            elevation_scale: Scale factor for elevation effect

        Returns:
            Transformed GeoDataFrame
        """
        if gdf is None or len(gdf) == 0:
            return gdf

        gdf = gdf.copy()

        def transform_geometry(geom):
            if geom is None or geom.is_empty:
                return geom
            return self._transform_geometry(geom, bbox, elevation_data, elevation_scale)

        gdf["geometry"] = gdf.geometry.apply(transform_geometry)
        return gdf

    def _transform_geometry(
        self,
        geom,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray],
        elevation_scale: float,
    ):
        """Transform a single geometry."""
        if isinstance(geom, Point):
            return self._transform_point_geom(geom, bbox, elevation_data, elevation_scale)
        elif isinstance(geom, LineString):
            return self._transform_linestring(geom, bbox, elevation_data, elevation_scale)
        elif isinstance(geom, Polygon):
            return self._transform_polygon(geom, bbox, elevation_data, elevation_scale)
        elif isinstance(geom, MultiPolygon):
            return MultiPolygon(
                [
                    self._transform_polygon(p, bbox, elevation_data, elevation_scale)
                    for p in geom.geoms
                ]
            )
        elif isinstance(geom, MultiLineString):
            return MultiLineString(
                [
                    self._transform_linestring(ls, bbox, elevation_data, elevation_scale)
                    for ls in geom.geoms
                ]
            )
        elif isinstance(geom, GeometryCollection):
            return GeometryCollection(
                [
                    self._transform_geometry(g, bbox, elevation_data, elevation_scale)
                    for g in geom.geoms
                ]
            )
        else:
            # Unknown geometry type, return as-is
            return geom

    def _transform_point_geom(
        self,
        point: Point,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray],
        elevation_scale: float,
    ) -> Point:
        """Transform a Point geometry."""
        x, y = point.x, point.y

        # Normalize to 0-1
        x_norm = (x - bbox.west) / (bbox.east - bbox.west)
        y_norm = (y - bbox.south) / (bbox.north - bbox.south)

        # Get elevation if available
        z = self._get_elevation_at(x, y, bbox, elevation_data)

        # Transform
        tx, ty = self.transform_point(x_norm, y_norm, z, elevation_scale)

        # Denormalize
        new_x = bbox.west + tx * (bbox.east - bbox.west)
        new_y = bbox.south + ty * (bbox.north - bbox.south)

        return Point(new_x, new_y)

    def _transform_linestring(
        self,
        line: LineString,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray],
        elevation_scale: float,
    ) -> LineString:
        """Transform a LineString geometry."""
        coords = list(line.coords)
        new_coords = []

        for x, y in coords:
            x_norm = (x - bbox.west) / (bbox.east - bbox.west)
            y_norm = (y - bbox.south) / (bbox.north - bbox.south)
            z = self._get_elevation_at(x, y, bbox, elevation_data)

            tx, ty = self.transform_point(x_norm, y_norm, z, elevation_scale)

            new_x = bbox.west + tx * (bbox.east - bbox.west)
            new_y = bbox.south + ty * (bbox.north - bbox.south)
            new_coords.append((new_x, new_y))

        return LineString(new_coords)

    def _transform_polygon(
        self,
        polygon: Polygon,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray],
        elevation_scale: float,
    ) -> Polygon:
        """Transform a Polygon geometry."""
        # Transform exterior ring
        exterior_coords = list(polygon.exterior.coords)
        new_exterior = []

        for x, y in exterior_coords:
            x_norm = (x - bbox.west) / (bbox.east - bbox.west)
            y_norm = (y - bbox.south) / (bbox.north - bbox.south)
            z = self._get_elevation_at(x, y, bbox, elevation_data)

            tx, ty = self.transform_point(x_norm, y_norm, z, elevation_scale)

            new_x = bbox.west + tx * (bbox.east - bbox.west)
            new_y = bbox.south + ty * (bbox.north - bbox.south)
            new_exterior.append((new_x, new_y))

        # Transform interior rings (holes)
        new_interiors = []
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            new_interior = []

            for x, y in interior_coords:
                x_norm = (x - bbox.west) / (bbox.east - bbox.west)
                y_norm = (y - bbox.south) / (bbox.north - bbox.south)
                z = self._get_elevation_at(x, y, bbox, elevation_data)

                tx, ty = self.transform_point(x_norm, y_norm, z, elevation_scale)

                new_x = bbox.west + tx * (bbox.east - bbox.west)
                new_y = bbox.south + ty * (bbox.north - bbox.south)
                new_interior.append((new_x, new_y))

            new_interiors.append(new_interior)

        return Polygon(new_exterior, new_interiors)

    def _get_elevation_at(
        self,
        lon: float,
        lat: float,
        bbox: BoundingBox,
        elevation_data: Optional[np.ndarray],
    ) -> float:
        """Get elevation value at a coordinate."""
        if elevation_data is None:
            return 0

        # Convert lon/lat to array indices
        x_idx = int((lon - bbox.west) / (bbox.east - bbox.west) * elevation_data.shape[1])
        y_idx = int((bbox.north - lat) / (bbox.north - bbox.south) * elevation_data.shape[0])

        # Clamp to valid range
        x_idx = max(0, min(elevation_data.shape[1] - 1, x_idx))
        y_idx = max(0, min(elevation_data.shape[0] - 1, y_idx))

        return float(elevation_data[y_idx, x_idx])

    def calculate_y_offset(self, elevation: float, elevation_scale: float = 0.00001) -> float:
        """
        Calculate Y offset for a given elevation.

        In isometric projection, higher elevations appear higher on screen.

        Args:
            elevation: Elevation in meters
            elevation_scale: Scale factor

        Returns:
            Y offset to apply
        """
        angle_rad = math.radians(self.angle)
        return elevation * elevation_scale * math.sin(angle_rad)

    def transform_image_coordinates(
        self,
        x: int,
        y: int,
        image_size: tuple[int, int],
        elevation: float = 0,
        elevation_scale: float = 0.001,
    ) -> tuple[int, int]:
        """
        Transform pixel coordinates with isometric projection.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            image_size: (width, height) of image
            elevation: Elevation at this point
            elevation_scale: Scale factor for elevation

        Returns:
            Transformed (x, y) pixel coordinates
        """
        width, height = image_size

        # Normalize to 0-1
        x_norm = x / width
        y_norm = y / height

        # Transform
        tx, ty = self.transform_point(x_norm, y_norm, elevation, elevation_scale)

        # Convert back to pixels
        new_x = int(tx * width)
        new_y = int(ty * height)

        return (new_x, new_y)
