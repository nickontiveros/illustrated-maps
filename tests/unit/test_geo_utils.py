"""Tests for geographic utility functions."""

import math

import pytest
from shapely.geometry import Polygon

from mapgen.models.project import BoundingBox
from mapgen.utils.geo_utils import (
    bbox_to_polygon,
    calculate_aspect_ratio,
    calculate_map_scale,
    gps_to_pixel,
    haversine_distance,
    meters_per_pixel,
    pixel_to_gps,
)


class TestBboxToPolygon:
    """Test bounding box to polygon conversion."""

    def test_returns_polygon(self, sample_bbox):
        poly = bbox_to_polygon(sample_bbox)
        assert isinstance(poly, Polygon)

    def test_polygon_bounds(self, sample_bbox):
        poly = bbox_to_polygon(sample_bbox)
        minx, miny, maxx, maxy = poly.bounds
        assert minx == pytest.approx(sample_bbox.west)
        assert miny == pytest.approx(sample_bbox.south)
        assert maxx == pytest.approx(sample_bbox.east)
        assert maxy == pytest.approx(sample_bbox.north)


class TestCalculateAspectRatio:
    """Test aspect ratio calculation."""

    def test_square_at_equator(self):
        bbox = BoundingBox(north=1, south=-1, east=1, west=-1)
        ratio = calculate_aspect_ratio(bbox)
        # At equator, should be ~1.0
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_wide_region(self):
        bbox = BoundingBox(north=40.01, south=39.99, east=-73.0, west=-74.0)
        ratio = calculate_aspect_ratio(bbox)
        assert ratio > 1.0  # Wider than tall

    def test_tall_region(self, sample_bbox):
        ratio = calculate_aspect_ratio(sample_bbox)
        # width=0.01 deg, height=0.007 deg -> ratio > 1 with lat correction at ~40N
        assert ratio > 0.5


class TestMetersPerPixel:
    """Test meters-per-pixel calculation."""

    def test_returns_tuple(self, sample_bbox):
        mx, my = meters_per_pixel(sample_bbox, 1000, 1000)
        assert mx > 0
        assert my > 0

    def test_higher_resolution_smaller_scale(self, sample_bbox):
        mx1, my1 = meters_per_pixel(sample_bbox, 500, 500)
        mx2, my2 = meters_per_pixel(sample_bbox, 1000, 1000)
        assert mx2 < mx1
        assert my2 < my1


class TestGpsToPixel:
    """Test GPS to pixel coordinate conversion."""

    def test_center_maps_to_center(self, sample_bbox):
        center_lat, center_lon = sample_bbox.center
        x, y = gps_to_pixel(center_lat, center_lon, sample_bbox, (1000, 1000))
        assert x == pytest.approx(500, abs=1)
        assert y == pytest.approx(500, abs=1)

    def test_southwest_corner_maps_to_bottom_left(self, sample_bbox):
        x, y = gps_to_pixel(sample_bbox.south, sample_bbox.west, sample_bbox, (1000, 1000))
        assert x == pytest.approx(0, abs=1)
        assert y == pytest.approx(1000, abs=1)

    def test_northeast_corner_maps_to_top_right(self, sample_bbox):
        x, y = gps_to_pixel(sample_bbox.north, sample_bbox.east, sample_bbox, (1000, 1000))
        assert x == pytest.approx(1000, abs=1)
        assert y == pytest.approx(0, abs=1)


class TestPixelToGps:
    """Test pixel to GPS coordinate conversion."""

    def test_center_pixel(self, sample_bbox):
        lat, lon = pixel_to_gps(500, 500, sample_bbox, (1000, 1000))
        center_lat, center_lon = sample_bbox.center
        assert lat == pytest.approx(center_lat, abs=0.001)
        assert lon == pytest.approx(center_lon, abs=0.001)

    def test_roundtrip(self, sample_bbox):
        """gps_to_pixel followed by pixel_to_gps should return original coords."""
        orig_lat, orig_lon = 40.770, -73.973
        size = (2000, 2000)
        px, py = gps_to_pixel(orig_lat, orig_lon, sample_bbox, size)
        lat, lon = pixel_to_gps(px, py, sample_bbox, size)
        assert lat == pytest.approx(orig_lat, abs=0.001)
        assert lon == pytest.approx(orig_lon, abs=0.001)


class TestHaversineDistance:
    """Test great-circle distance calculation."""

    def test_same_point(self):
        d = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert d == pytest.approx(0.0)

    def test_known_distance(self):
        # NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) ~ 3944 km
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert d == pytest.approx(3_944_000, rel=0.02)

    def test_one_degree_latitude(self):
        # 1 degree of latitude ~ 111.32 km
        d = haversine_distance(0, 0, 1, 0)
        assert d == pytest.approx(111_320, rel=0.01)

    def test_symmetry(self):
        d1 = haversine_distance(40.0, -74.0, 41.0, -73.0)
        d2 = haversine_distance(41.0, -73.0, 40.0, -74.0)
        assert d1 == pytest.approx(d2)


class TestCalculateMapScale:
    """Test map scale calculation."""

    def test_positive_scale(self, sample_bbox):
        scale = calculate_map_scale(sample_bbox, 1000)
        assert scale > 0

    def test_higher_resolution_smaller_scale(self, sample_bbox):
        scale1 = calculate_map_scale(sample_bbox, 500)
        scale2 = calculate_map_scale(sample_bbox, 1000)
        assert scale2 < scale1
