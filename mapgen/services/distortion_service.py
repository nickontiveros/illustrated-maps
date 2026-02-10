"""Non-linear coordinate mapping for sectional map generation.

Provides piecewise-linear distortion that enlarges dense landmark areas
(cities) and compresses sparse areas (desert) in pixel space.
"""

from typing import Optional

import numpy as np

from ..models.project import BoundingBox, CoordinateMapping


class DistortionService:
    """Maps geographic coordinates to pixel coordinates using non-linear control points."""

    def __init__(
        self,
        mapping: CoordinateMapping,
        region: BoundingBox,
        output_size: tuple[int, int],
    ):
        """
        Args:
            mapping: Control point mapping (geographic -> normalized 0-1).
            region: Geographic bounding box of the full map.
            output_size: (width, height) in pixels.
        """
        self.mapping = mapping
        self.region = region
        self.output_size = output_size

        # Pre-sort control points for interpolation
        self._lat_geo = np.array([p[0] for p in mapping.lat_control_points])
        self._lat_norm = np.array([p[1] for p in mapping.lat_control_points])
        self._lon_geo = np.array([p[0] for p in mapping.lon_control_points])
        self._lon_norm = np.array([p[1] for p in mapping.lon_control_points])

        # Ensure sorted order for np.interp
        lat_order = np.argsort(self._lat_geo)
        self._lat_geo = self._lat_geo[lat_order]
        self._lat_norm = self._lat_norm[lat_order]

        lon_order = np.argsort(self._lon_geo)
        self._lon_geo = self._lon_geo[lon_order]
        self._lon_norm = self._lon_norm[lon_order]

    def geo_to_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        """Map geographic coordinates to pixel position using non-linear interpolation.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            (x, y) pixel coordinates.
        """
        width, height = self.output_size

        # Interpolate normalized position from control points
        x_norm = float(np.interp(lon, self._lon_geo, self._lon_norm))
        y_norm = float(np.interp(lat, self._lat_geo, self._lat_norm))

        pixel_x = int(x_norm * width)
        pixel_y = int((1 - y_norm) * height)  # Flip Y (image origin top-left)

        return (pixel_x, pixel_y)

    def pixel_to_geo(self, x: int, y: int) -> tuple[float, float]:
        """Map pixel position back to geographic coordinates (inverse mapping).

        Args:
            x: X pixel coordinate.
            y: Y pixel coordinate.

        Returns:
            (latitude, longitude).
        """
        width, height = self.output_size

        x_norm = x / width
        y_norm = 1 - (y / height)  # Flip Y

        # Inverse interpolation: normalized -> geographic
        lon = float(np.interp(x_norm, self._lon_norm, self._lon_geo))
        lat = float(np.interp(y_norm, self._lat_norm, self._lat_geo))

        return (lat, lon)

    def get_geographic_rect_for_pixel_rect(
        self, x: int, y: int, w: int, h: int
    ) -> BoundingBox:
        """Map a pixel-space rectangle back to geographic bounds.

        Useful for determining what satellite/OSM data to fetch for a
        pixel-space tile or section.

        Args:
            x: Left edge in pixels.
            y: Top edge in pixels.
            w: Width in pixels.
            h: Height in pixels.

        Returns:
            Geographic BoundingBox covering the pixel rectangle.
        """
        # Sample corners and edges for accuracy
        corners = [
            self.pixel_to_geo(x, y),
            self.pixel_to_geo(x + w, y),
            self.pixel_to_geo(x, y + h),
            self.pixel_to_geo(x + w, y + h),
        ]

        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]

        return BoundingBox(
            north=max(lats),
            south=min(lats),
            east=max(lons),
            west=min(lons),
        )

    def get_pixel_density_at(self, lat: float, lon: float) -> float:
        """Get pixels-per-degree at a geographic point.

        Higher values mean more detail at this location (city areas).
        Can be used to select appropriate Mapbox zoom levels.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Approximate pixels per degree at this point (average of lat/lon).
        """
        width, height = self.output_size
        delta = 0.01  # Small geographic step for finite difference

        # Pixel density in X (longitude) direction
        x1, _ = self.geo_to_pixel(lat, lon - delta / 2)
        x2, _ = self.geo_to_pixel(lat, lon + delta / 2)
        px_per_deg_lon = abs(x2 - x1) / delta

        # Pixel density in Y (latitude) direction
        _, y1 = self.geo_to_pixel(lat - delta / 2, lon)
        _, y2 = self.geo_to_pixel(lat + delta / 2, lon)
        px_per_deg_lat = abs(y2 - y1) / delta

        return (px_per_deg_lon + px_per_deg_lat) / 2


def compute_control_points(
    region: BoundingBox,
    landmarks: list,
    min_focus_ratio: float = 0.3,
    num_bins: int = 10,
) -> CoordinateMapping:
    """Auto-compute coordinate mapping control points from landmark density.

    Clusters landmarks by proximity and allocates more pixel space to
    denser clusters while ensuring sparse areas don't vanish.

    Args:
        region: Full geographic bounding box.
        landmarks: List of Landmark objects (need .latitude, .longitude).
        min_focus_ratio: Minimum pixel-space fraction for landmark-sparse areas.
        num_bins: Number of bins for latitude and longitude.

    Returns:
        CoordinateMapping with computed control points.
    """
    if not landmarks:
        # No landmarks: linear mapping
        lat_pts = [
            (region.south, 0.0),
            (region.north, 1.0),
        ]
        lon_pts = [
            (region.west, 0.0),
            (region.east, 1.0),
        ]
        return CoordinateMapping(lat_control_points=lat_pts, lon_control_points=lon_pts)

    # Build density histograms
    lats = np.array([lm.latitude for lm in landmarks])
    lons = np.array([lm.longitude for lm in landmarks])

    lat_control_points = _compute_axis_control_points(
        lats, region.south, region.north, num_bins, min_focus_ratio
    )
    lon_control_points = _compute_axis_control_points(
        lons, region.west, region.east, num_bins, min_focus_ratio
    )

    return CoordinateMapping(
        lat_control_points=lat_control_points,
        lon_control_points=lon_control_points,
    )


def _compute_axis_control_points(
    values: np.ndarray,
    geo_min: float,
    geo_max: float,
    num_bins: int,
    min_focus_ratio: float,
) -> list[tuple[float, float]]:
    """Compute control points for one axis based on value density.

    Bins the axis into `num_bins` segments, counts values in each,
    then allocates pixel space proportional to (count + baseline).

    Args:
        values: Geographic coordinates of landmarks along this axis.
        geo_min: Minimum geographic value (south or west).
        geo_max: Maximum geographic value (north or east).
        num_bins: Number of bins.
        min_focus_ratio: Minimum allocation for empty bins.

    Returns:
        List of (geographic_value, normalized_position) tuples.
    """
    bin_edges = np.linspace(geo_min, geo_max, num_bins + 1)

    # Count landmarks per bin
    counts = np.zeros(num_bins)
    for v in values:
        bin_idx = int((v - geo_min) / (geo_max - geo_min) * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)
        counts[bin_idx] += 1

    # Allocate pixel space: baseline + density
    # Baseline ensures empty bins get at least min_focus_ratio / num_bins of space
    baseline = min_focus_ratio / num_bins
    weights = baseline + counts
    weights = weights / weights.sum()

    # Build cumulative control points
    points = [(geo_min, 0.0)]
    cumulative = 0.0
    for i in range(num_bins):
        cumulative += weights[i]
        geo_val = bin_edges[i + 1]
        points.append((geo_val, cumulative))

    # Ensure exact endpoints
    points[-1] = (geo_max, 1.0)

    return points
