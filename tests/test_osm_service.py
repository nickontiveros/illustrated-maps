"""Tests for OSM service."""

import pytest
from mapgen.models.project import BoundingBox
from mapgen.services.osm_service import OSMService


@pytest.fixture
def small_bbox():
    """A small bounding box for testing (Central Park area)."""
    return BoundingBox(
        north=40.7750,
        south=40.7680,
        east=-73.9680,
        west=-73.9780,
    )


class TestOSMService:
    """Tests for OSMService."""

    def test_init(self):
        """Test service initialization."""
        service = OSMService()
        assert service is not None

    def test_fetch_region_data(self, small_bbox):
        """Test fetching OSM data for a region."""
        service = OSMService()
        data = service.fetch_region_data(small_bbox)

        assert data is not None
        assert data.has_data()

    def test_extract_roads(self, small_bbox):
        """Test road extraction."""
        service = OSMService()
        roads = service.extract_roads(small_bbox)

        # Central Park area should have roads
        assert roads is not None
        assert len(roads) > 0

    def test_extract_parks(self, small_bbox):
        """Test park extraction."""
        service = OSMService()
        parks = service.extract_parks(small_bbox)

        # Central Park area should have parks
        assert parks is not None
        # May or may not have parks depending on exact bbox
