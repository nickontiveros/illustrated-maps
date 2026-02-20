"""Enhanced road styling service with width exaggeration and organic linework."""

import math
from typing import Optional

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString

from ..models.project import BoundingBox
from ..models.road_style import ROAD_STYLE_PRESETS, RoadStyleSettings


class RoadStyleService:
    """Service for rendering roads with enhanced styling.

    Provides width exaggeration per road class, organic wobble via
    Perlin-like noise, and hierarchy-based colors. Produces a road
    layer that can be used as:
      (a) Higher-opacity reference sent to Gemini for guidance
      (b) Transparent overlay composited on final illustrated output
    """

    # Mapping from OSM highway types to internal road classes
    HIGHWAY_TO_CLASS = {
        "motorway": "motorway",
        "motorway_link": "motorway",
        "trunk": "primary",
        "trunk_link": "primary",
        "primary": "primary",
        "primary_link": "primary",
        "secondary": "secondary",
        "secondary_link": "secondary",
        "tertiary": "secondary",
        "tertiary_link": "secondary",
        "residential": "residential",
        "unclassified": "residential",
        "living_street": "residential",
        "service": "residential",
    }

    # Mapping from existing road_class in OSM data to our classes
    ROAD_CLASS_MAP = {
        "major": "primary",
        "minor": "secondary",
        "other": "residential",
    }

    def __init__(self, settings: Optional[RoadStyleSettings] = None):
        """Initialize road style service.

        Args:
            settings: Road style settings. If None, uses defaults.
                      If settings.preset is set, loads that preset.
        """
        if settings is None:
            self.settings = RoadStyleSettings()
        elif settings.preset and settings.preset in ROAD_STYLE_PRESETS:
            self.settings = ROAD_STYLE_PRESETS[settings.preset].model_copy()
        else:
            self.settings = settings

    def get_exaggeration(self, road_class: str) -> float:
        """Get width exaggeration factor for a road class.

        Args:
            road_class: One of 'motorway', 'primary', 'secondary', 'residential'

        Returns:
            Width multiplier
        """
        return {
            "motorway": self.settings.motorway_exaggeration,
            "primary": self.settings.primary_exaggeration,
            "secondary": self.settings.secondary_exaggeration,
            "residential": self.settings.residential_exaggeration,
        }.get(road_class, self.settings.residential_exaggeration)

    def get_fill_color(self, road_class: str) -> str:
        """Get fill color for a road class."""
        return {
            "motorway": self.settings.motorway_color,
            "primary": self.settings.primary_color,
            "secondary": self.settings.secondary_color,
            "residential": self.settings.residential_color,
        }.get(road_class, self.settings.residential_color) or "#FFFFFF"

    def _classify_road(self, row) -> str:
        """Classify a road from OSM data into our road classes.

        Checks for 'highway' column first, then falls back to 'road_class'.
        """
        highway = row.get("highway", None)
        if highway and highway in self.HIGHWAY_TO_CLASS:
            return self.HIGHWAY_TO_CLASS[highway]

        road_class = row.get("road_class", "other")
        return self.ROAD_CLASS_MAP.get(road_class, "residential")

    def _geo_to_pixel(
        self,
        lon: float,
        lat: float,
        bbox: BoundingBox,
        image_size: tuple[int, int],
    ) -> tuple[float, float]:
        """Convert geographic coordinates to pixel coordinates.

        Args:
            lon: Longitude
            lat: Latitude
            bbox: Map bounding box
            image_size: (width, height) in pixels

        Returns:
            (x, y) pixel coordinates
        """
        w, h = image_size
        x = (lon - bbox.west) / (bbox.east - bbox.west) * w
        y = (1.0 - (lat - bbox.south) / (bbox.north - bbox.south)) * h
        return (x, y)

    def _compute_pixel_width(
        self,
        road_class: str,
        bbox: BoundingBox,
        image_size: tuple[int, int],
    ) -> float:
        """Compute pixel width for a road class with exaggeration.

        Uses geographic scale to determine base road width, then
        applies the exaggeration factor.

        Args:
            road_class: Road classification
            bbox: Map bounding box
            image_size: Output image dimensions

        Returns:
            Road width in pixels
        """
        # Approximate real road widths in meters
        real_widths = {
            "motorway": 14.0,
            "primary": 10.0,
            "secondary": 7.0,
            "residential": 5.0,
        }

        real_width = real_widths.get(road_class, 5.0)
        exaggeration = self.get_exaggeration(road_class)

        # Calculate meters per pixel
        center_lat = (bbox.north + bbox.south) / 2
        meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        meters_per_pixel = (bbox.east - bbox.west) * meters_per_deg_lon / image_size[0]

        # Exaggerated width in pixels (clamped to reasonable range)
        pixel_width = (real_width * exaggeration) / meters_per_pixel
        return max(1.0, min(pixel_width, 60.0))

    def _apply_wobble(
        self,
        points: list[tuple[float, float]],
        amplitude: float,
        frequency: float,
        seed: int = 0,
    ) -> list[tuple[float, float]]:
        """Apply organic wobble to a polyline using simplex-like noise.

        Displaces each point perpendicular to the local road direction
        by a noise value, creating a hand-drawn feel while preserving
        connectivity (endpoints are not displaced).

        Args:
            points: List of (x, y) pixel coordinates
            amplitude: Maximum displacement in pixels
            frequency: Spatial frequency of the wobble
            seed: Random seed for reproducibility

        Returns:
            List of displaced (x, y) coordinates
        """
        if amplitude <= 0 or len(points) < 3:
            return points

        rng = np.random.RandomState(seed)
        result = list(points)  # Copy

        # Don't displace first and last points (preserve connectivity)
        for i in range(1, len(points) - 1):
            x, y = points[i]

            # Compute local tangent direction
            px, py = points[i - 1]
            nx, ny = points[i + 1]
            dx = nx - px
            dy = ny - py
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                continue

            # Perpendicular direction (normal)
            normal_x = -dy / length
            normal_y = dx / length

            # Generate noise value using sin-based pseudo-noise
            # This is simpler than true Perlin noise but gives smooth results
            t = i * frequency + seed * 0.1
            noise = (
                math.sin(t * 6.2831) * 0.5
                + math.sin(t * 2.7183 * 6.2831) * 0.3
                + math.sin(t * 1.4142 * 6.2831) * 0.2
            )

            # Displace perpendicular to road direction
            displacement = noise * amplitude
            result[i] = (x + normal_x * displacement, y + normal_y * displacement)

        return result

    def _linestring_to_pixels(
        self,
        geom,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        sample_interval: float = 3.0,
    ) -> list[tuple[float, float]]:
        """Convert a LineString geometry to pixel coordinates.

        Samples points along the line at regular intervals for smooth rendering.

        Args:
            geom: Shapely LineString
            bbox: Map bounding box
            image_size: Output image size
            sample_interval: Distance between samples in pixels (approximate)

        Returns:
            List of (x, y) pixel coordinates
        """
        if not isinstance(geom, LineString) or geom.is_empty:
            return []

        coords = list(geom.coords)
        if len(coords) < 2:
            return []

        # Convert all coordinates to pixels
        pixel_coords = [
            self._geo_to_pixel(lon, lat, bbox, image_size)
            for lon, lat in coords
        ]

        # Calculate total pixel length
        total_length = 0.0
        for i in range(1, len(pixel_coords)):
            dx = pixel_coords[i][0] - pixel_coords[i - 1][0]
            dy = pixel_coords[i][1] - pixel_coords[i - 1][1]
            total_length += math.sqrt(dx * dx + dy * dy)

        if total_length < 2.0:
            return pixel_coords

        # Resample at regular intervals for smooth lines
        n_samples = max(2, int(total_length / sample_interval))
        resampled = []
        for i in range(n_samples + 1):
            frac = i / n_samples
            point = geom.interpolate(frac, normalized=True)
            px, py = self._geo_to_pixel(point.x, point.y, bbox, image_size)
            resampled.append((px, py))

        return resampled

    def render_road_layer(
        self,
        roads: gpd.GeoDataFrame,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        background_alpha: int = 0,
    ) -> Image.Image:
        """Render the complete road layer as a transparent RGBA image.

        Roads are rendered back-to-front by class (residential first,
        motorway last) with outlines drawn before fills.

        Args:
            roads: GeoDataFrame with road geometries
            bbox: Map bounding box
            image_size: (width, height) of output
            background_alpha: Alpha value for background (0=transparent)

        Returns:
            RGBA Image with rendered roads
        """
        w, h = image_size
        image = Image.new("RGBA", (w, h), (0, 0, 0, background_alpha))
        draw = ImageDraw.Draw(image)

        if roads is None or len(roads) == 0:
            return image

        # Classify roads and group by class
        road_groups: dict[str, list] = {
            "residential": [],
            "secondary": [],
            "primary": [],
            "motorway": [],
        }

        for idx, row in roads.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            road_class = self._classify_road(row)

            # Handle MultiLineString
            if isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    pixels = self._linestring_to_pixels(line, bbox, image_size)
                    if len(pixels) >= 2:
                        road_groups.setdefault(road_class, []).append((pixels, idx))
            elif isinstance(geom, LineString):
                pixels = self._linestring_to_pixels(geom, bbox, image_size)
                if len(pixels) >= 2:
                    road_groups.setdefault(road_class, []).append((pixels, idx))

        # Render order: residential -> secondary -> primary -> motorway
        outline_color = self.settings.outline_color or "#B0A090"

        for road_class in ["residential", "secondary", "primary", "motorway"]:
            segments = road_groups.get(road_class, [])
            if not segments:
                continue

            pixel_width = self._compute_pixel_width(road_class, bbox, image_size)
            fill_color = self.get_fill_color(road_class)
            outline_width = pixel_width + max(2, pixel_width * 0.3)

            # Apply wobble to all segments
            wobbled_segments = []
            for pixels, seed_val in segments:
                if self.settings.wobble_amount > 0:
                    wobbled = self._apply_wobble(
                        pixels,
                        self.settings.wobble_amount,
                        self.settings.wobble_frequency,
                        seed=seed_val if isinstance(seed_val, int) else hash(seed_val) % 10000,
                    )
                else:
                    wobbled = pixels
                wobbled_segments.append(wobbled)

            # Draw outlines first
            for wobbled in wobbled_segments:
                if len(wobbled) >= 2:
                    flat_coords = []
                    for x, y in wobbled:
                        flat_coords.extend([x, y])
                    draw.line(
                        flat_coords,
                        fill=outline_color,
                        width=max(1, int(outline_width)),
                        joint="curve",
                    )

            # Draw fills
            for wobbled in wobbled_segments:
                if len(wobbled) >= 2:
                    flat_coords = []
                    for x, y in wobbled:
                        flat_coords.extend([x, y])
                    draw.line(
                        flat_coords,
                        fill=fill_color,
                        width=max(1, int(pixel_width)),
                        joint="curve",
                    )

        return image

    def create_reference_layer(
        self,
        roads: gpd.GeoDataFrame,
        bbox: BoundingBox,
        image_size: tuple[int, int],
    ) -> Image.Image:
        """Create a road reference layer for Gemini with higher opacity.

        This is composited onto the satellite+OSM reference before sending
        to Gemini, providing clear road width guidance.

        Args:
            roads: GeoDataFrame with road geometries
            bbox: Map bounding box
            image_size: Output dimensions

        Returns:
            RGBA Image with road reference at reference_opacity
        """
        road_layer = self.render_road_layer(roads, bbox, image_size)

        # Apply reference opacity
        opacity = self.settings.reference_opacity
        if opacity < 1.0:
            r, g, b, a = road_layer.split()
            a = a.point(lambda x: int(x * opacity))
            road_layer = Image.merge("RGBA", (r, g, b, a))

        return road_layer

    def create_overlay_layer(
        self,
        roads: gpd.GeoDataFrame,
        bbox: BoundingBox,
        image_size: tuple[int, int],
    ) -> Image.Image:
        """Create a road overlay for the final illustrated output.

        This transparent overlay is composited on top of the Gemini-generated
        illustration to guarantee road presence and correct widths.

        Args:
            roads: GeoDataFrame with road geometries
            bbox: Map bounding box
            image_size: Output dimensions

        Returns:
            RGBA Image with road overlay at overlay_opacity
        """
        road_layer = self.render_road_layer(roads, bbox, image_size)

        # Apply overlay opacity
        opacity = self.settings.overlay_opacity
        if opacity < 1.0:
            r, g, b, a = road_layer.split()
            a = a.point(lambda x: int(x * opacity))
            road_layer = Image.merge("RGBA", (r, g, b, a))

        return road_layer
