"""Tests for BoundingBox model."""

import math

import pytest
from pydantic import ValidationError

from mapgen.models.project import BoundingBox, DetailLevel, OrientedRegion, OutputSettings, validate_aspect_ratio


class TestBoundingBoxConstruction:
    """Test BoundingBox creation and validation."""

    def test_valid_construction(self, sample_bbox):
        assert sample_bbox.north == 40.775
        assert sample_bbox.south == 40.768
        assert sample_bbox.east == -73.968
        assert sample_bbox.west == -73.978

    def test_rejects_latitude_out_of_range(self):
        with pytest.raises(ValidationError):
            BoundingBox(north=91, south=40.0, east=-73.0, west=-74.0)

    def test_rejects_longitude_out_of_range(self):
        with pytest.raises(ValidationError):
            BoundingBox(north=41.0, south=40.0, east=181, west=-74.0)

    def test_accepts_boundary_values(self):
        bbox = BoundingBox(north=90, south=-90, east=180, west=-180)
        assert bbox.north == 90
        assert bbox.west == -180


class TestBoundingBoxProperties:
    """Test computed properties."""

    def test_center(self, sample_bbox):
        lat, lon = sample_bbox.center
        assert lat == pytest.approx((40.775 + 40.768) / 2)
        assert lon == pytest.approx((-73.968 + -73.978) / 2)

    def test_width_degrees(self, sample_bbox):
        assert sample_bbox.width_degrees == pytest.approx(0.01)

    def test_height_degrees(self, sample_bbox):
        assert sample_bbox.height_degrees == pytest.approx(0.007)

    def test_geographic_aspect_ratio_accounts_for_latitude(self, sample_bbox):
        ratio = sample_bbox.geographic_aspect_ratio
        # At ~40.77N, cosine correction makes width shorter
        center_lat = (40.775 + 40.768) / 2
        correction = math.cos(math.radians(center_lat))
        expected = (0.01 * correction) / 0.007
        assert ratio == pytest.approx(expected, rel=1e-6)

    def test_geographic_aspect_ratio_equator(self):
        """At the equator, correction factor is 1."""
        bbox = BoundingBox(north=1.0, south=-1.0, east=1.0, west=-1.0)
        # width_degrees == height_degrees == 2, correction ~= cos(0) = 1
        assert bbox.geographic_aspect_ratio == pytest.approx(1.0, rel=1e-3)


class TestBoundingBoxArea:
    """Test area calculations."""

    def test_calculate_area_km2_small_region(self, sample_bbox):
        area = sample_bbox.calculate_area_km2()
        # Central Park area bbox: roughly 1km x 0.8km = ~0.8 km²
        assert 0.1 < area < 5.0

    def test_calculate_area_km2_large_region(self, wide_bbox):
        area = wide_bbox.calculate_area_km2()
        # Manhattan-sized bbox: roughly 7.5km x 11km = ~80 km²
        assert 50 < area < 150

    def test_get_recommended_detail_level_small(self, sample_bbox):
        level = sample_bbox.get_recommended_detail_level()
        assert level == DetailLevel.FULL

    def test_get_recommended_detail_level_large(self):
        bbox = BoundingBox(north=45, south=35, east=-70, west=-80)
        level = bbox.get_recommended_detail_level()
        assert level == DetailLevel.COUNTRY


class TestBoundingBoxTransforms:
    """Test geometric transforms."""

    def test_expanded_for_rotation_zero(self, sample_bbox):
        expanded = sample_bbox.expanded_for_rotation(0)
        assert expanded.north == sample_bbox.north
        assert expanded.south == sample_bbox.south

    def test_expanded_for_rotation_360(self, sample_bbox):
        expanded = sample_bbox.expanded_for_rotation(360)
        assert expanded.north == sample_bbox.north

    def test_expanded_for_rotation_45_grows(self, sample_bbox):
        expanded = sample_bbox.expanded_for_rotation(45)
        # Rotated bbox should be larger
        assert expanded.width_degrees >= sample_bbox.width_degrees
        assert expanded.height_degrees >= sample_bbox.height_degrees

    def test_expanded_for_rotation_90_swaps(self):
        bbox = BoundingBox(north=1, south=-1, east=2, west=-2)
        expanded = bbox.expanded_for_rotation(90)
        # 90-degree rotation swaps width and height
        assert expanded.width_degrees >= bbox.height_degrees - 0.01
        assert expanded.height_degrees >= bbox.width_degrees - 0.01


class TestBoundingBoxSerialization:
    """Test format conversions."""

    def test_to_tuple(self, sample_bbox):
        t = sample_bbox.to_tuple()
        assert t == (40.775, 40.768, -73.968, -73.978)

    def test_to_osmnx_bbox(self, sample_bbox):
        t = sample_bbox.to_osmnx_bbox()
        assert t == (-73.978, 40.768, -73.968, 40.775)


class TestValidateAspectRatio:
    """Test the validate_aspect_ratio helper."""

    def test_matching_ratios(self):
        assert validate_aspect_ratio(0.7063, 0.7063) is True

    def test_within_tolerance(self):
        # 1% off should pass with 2% tolerance
        assert validate_aspect_ratio(0.7063 * 1.01, 0.7063) is True

    def test_outside_tolerance(self):
        # 5% off should fail
        assert validate_aspect_ratio(0.7063 * 1.05, 0.7063) is False

    def test_exact_match(self):
        assert validate_aspect_ratio(1.0, 1.0) is True

    def test_custom_tolerance(self):
        assert validate_aspect_ratio(0.7063 * 1.04, 0.7063, tolerance=0.05) is True
        assert validate_aspect_ratio(0.7063 * 1.04, 0.7063, tolerance=0.02) is False


class TestConstrainToAspectRatio:
    """Test BoundingBox.constrain_to_aspect_ratio()."""

    def test_lock_height_adjusts_ew(self):
        bbox = BoundingBox(north=1.0, south=-1.0, east=1.0, west=-1.0)
        target = 0.5  # narrower
        result = bbox.constrain_to_aspect_ratio(target, lock="height")
        assert result.north == bbox.north
        assert result.south == bbox.south
        # New width should be narrower
        assert result.geographic_aspect_ratio == pytest.approx(target, rel=0.01)

    def test_lock_width_adjusts_ns(self):
        bbox = BoundingBox(north=1.0, south=-1.0, east=1.0, west=-1.0)
        target = 0.5  # narrower → need taller height
        result = bbox.constrain_to_aspect_ratio(target, lock="width")
        assert result.east == bbox.east
        assert result.west == bbox.west
        assert result.geographic_aspect_ratio == pytest.approx(target, rel=0.01)

    def test_a1_proportion_at_nyc_latitude(self):
        """Constrain a square-ish bbox at NYC latitude to A1 proportions."""
        bbox = BoundingBox(north=41.0, south=40.0, east=-73.0, west=-74.0)
        a1_aspect = OutputSettings().aspect_ratio
        result = bbox.constrain_to_aspect_ratio(a1_aspect)
        assert validate_aspect_ratio(result.geographic_aspect_ratio, a1_aspect) is True


class TestOrientedRegionAspectRatio:
    """Test OrientedRegion.aspect_ratio property."""

    def test_basic(self):
        region = OrientedRegion(
            center_lat=40.0, center_lon=-74.0,
            width_km=7.0, height_km=10.0,
        )
        assert region.aspect_ratio == pytest.approx(0.7)

    def test_square(self):
        region = OrientedRegion(
            center_lat=0.0, center_lon=0.0,
            width_km=5.0, height_km=5.0,
        )
        assert region.aspect_ratio == pytest.approx(1.0)

    def test_a1_proportions(self):
        a1_aspect = OutputSettings().aspect_ratio
        region = OrientedRegion(
            center_lat=40.0, center_lon=-74.0,
            width_km=10.0 * a1_aspect, height_km=10.0,
        )
        assert validate_aspect_ratio(region.aspect_ratio, a1_aspect) is True
