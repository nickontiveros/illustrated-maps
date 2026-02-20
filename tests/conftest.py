"""Shared test fixtures."""

import numpy as np
import pytest
from PIL import Image

from mapgen.models.landmark import FeatureType, Landmark
from mapgen.models.project import BoundingBox, OutputSettings, Project, TileSettings


@pytest.fixture
def sample_bbox():
    """Small bounding box around Central Park, NYC."""
    return BoundingBox(north=40.775, south=40.768, east=-73.968, west=-73.978)


@pytest.fixture
def wide_bbox():
    """Wider bounding box spanning Manhattan."""
    return BoundingBox(north=40.80, south=40.70, east=-73.93, west=-74.02)


@pytest.fixture
def sample_project(tmp_path, sample_bbox):
    """Minimal project config for testing."""
    project = Project(
        name="test-project",
        region=sample_bbox,
        output=OutputSettings(width=1024, height=1024, dpi=72),
        tiles=TileSettings(size=512, overlap=64),
    )
    project.project_dir = tmp_path
    return project


@pytest.fixture
def sample_landmark():
    """A test landmark."""
    return Landmark(
        name="Test Building",
        latitude=40.770,
        longitude=-73.973,
        feature_type=FeatureType.BUILDING,
        scale=2.0,
        z_index=5,
    )


@pytest.fixture
def solid_red_image():
    """64x64 solid red RGBA image."""
    return Image.new("RGBA", (64, 64), (255, 0, 0, 255))


@pytest.fixture
def solid_blue_image():
    """64x64 solid blue RGBA image."""
    return Image.new("RGBA", (64, 64), (0, 0, 255, 255))


@pytest.fixture
def transparent_image():
    """64x64 fully transparent image."""
    return Image.new("RGBA", (64, 64), (0, 0, 0, 0))


@pytest.fixture
def gradient_image():
    """64x64 horizontal gradient from black to white."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    arr[:, :, 1] = arr[:, :, 0]
    arr[:, :, 2] = arr[:, :, 0]
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def small_test_image():
    """32x32 image with some non-trivial content for testing."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    # Red top-left quadrant
    arr[:16, :16] = [255, 0, 0, 255]
    # Green top-right quadrant
    arr[:16, 16:] = [0, 255, 0, 255]
    # Blue bottom-left quadrant
    arr[16:, :16] = [0, 0, 255, 255]
    # White bottom-right quadrant
    arr[16:, 16:] = [255, 255, 255, 255]
    return Image.fromarray(arr, "RGBA")


from mapgen.models.typography import TypographySettings
from mapgen.models.road_style import RoadStyleSettings
from mapgen.models.border import BorderSettings, BorderStyle
from mapgen.models.atmosphere import AtmosphereSettings
from mapgen.models.narrative import NarrativeSettings


@pytest.fixture
def sample_typography_settings():
    """Typography settings with labels enabled."""
    return TypographySettings(enabled=True)


@pytest.fixture
def sample_road_style_settings():
    """Road style settings enabled with defaults."""
    return RoadStyleSettings(enabled=True)


@pytest.fixture
def sample_border_settings():
    """Border settings with vintage_scroll style."""
    return BorderSettings(enabled=True, style=BorderStyle.VINTAGE_SCROLL, margin=100)


@pytest.fixture
def sample_atmosphere_settings():
    """Atmosphere settings enabled with mild haze."""
    return AtmosphereSettings(enabled=True, haze_strength=0.3)


@pytest.fixture
def sample_narrative_settings():
    """Narrative settings for landmark discovery."""
    return NarrativeSettings(max_landmarks=20)


@pytest.fixture
def large_test_image():
    """256x256 RGBA image for border/atmosphere tests."""
    arr = np.zeros((256, 256, 4), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.linspace(50, 200, 256, dtype=np.uint8), (256, 1))
    arr[:, :, 1] = np.tile(np.linspace(100, 180, 256, dtype=np.uint8).reshape(256, 1), (1, 256))
    arr[:, :, 2] = 150
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def sample_buildings_gdf():
    """Minimal GeoDataFrame with OSM-like columns for testing discovery."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box

    data = [
        {"name": "Big Museum", "tourism": "museum", "geometry": box(-73.975, 40.769, -73.974, 40.770)},
        {"name": "Old Castle", "historic": "castle", "geometry": box(-73.976, 40.770, -73.975, 40.771)},
        {"name": "City Park", "leisure": "park", "geometry": box(-73.977, 40.771, -73.976, 40.772)},
        {"name": "Local Cafe", "amenity": "restaurant", "geometry": Point(-73.974, 40.769)},
        {"name": "Train Station", "amenity": "library", "geometry": Point(-73.973, 40.770)},
        {"name": None, "tourism": "attraction", "geometry": Point(-73.972, 40.771)},
    ]
    df = pd.DataFrame(data)
    return gpd.GeoDataFrame(df, geometry="geometry")
