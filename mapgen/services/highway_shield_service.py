"""Highway shield rendering service.

Generates and composites Interstate, US Highway, and State Route shield
icons onto map images based on road ref tags from OSM data.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, MultiLineString

from ..models.project import BoundingBox


@dataclass
class ShieldSpec:
    """Specification for a single highway shield to render."""

    shield_type: str  # "interstate", "us_highway", "state_route"
    number: str  # Route number (e.g., "10", "60", "87")
    x: int  # X pixel position on image
    y: int  # Y pixel position on image
    ref: str  # Original ref string (e.g., "I-10")


# Minimum pixel distance between shields of the same route
MIN_SHIELD_SPACING = 400


class HighwayShieldService:
    """Service for rendering highway route shields on map images."""

    # Patterns for classifying route references
    _INTERSTATE_PATTERN = re.compile(
        r"^I[- ]?(\d+)$|^Interstate\s+(\d+)$", re.IGNORECASE
    )
    _US_HIGHWAY_PATTERN = re.compile(
        r"^US[- ]?(\d+)$|^U\.?S\.?\s*Route\s+(\d+)$", re.IGNORECASE
    )
    _STATE_ROUTE_PATTERN = re.compile(
        r"^(?:AZ|SR|State Route|Loop|L)[- ]?(\d+)$", re.IGNORECASE
    )

    @classmethod
    def classify_ref(cls, ref: str) -> Optional[tuple[str, str]]:
        """Classify a road ref string into shield type and number.

        Args:
            ref: Road reference string (e.g., "I-10", "US 60", "AZ 87")

        Returns:
            Tuple of (shield_type, number) or None if not recognized.
            shield_type is one of "interstate", "us_highway", "state_route".
        """
        if not ref or not isinstance(ref, str):
            return None

        ref = ref.strip()

        m = cls._INTERSTATE_PATTERN.match(ref)
        if m:
            number = m.group(1) or m.group(2)
            return ("interstate", number)

        m = cls._US_HIGHWAY_PATTERN.match(ref)
        if m:
            number = m.group(1) or m.group(2)
            return ("us_highway", number)

        m = cls._STATE_ROUTE_PATTERN.match(ref)
        if m:
            number = m.group(1)
            return ("state_route", number)

        return None

    @classmethod
    def extract_shield_positions(
        cls,
        roads_gdf: gpd.GeoDataFrame,
        bbox: BoundingBox,
        image_size: tuple[int, int],
    ) -> list[ShieldSpec]:
        """Extract shield positions from road geometries.

        Samples positions along road lines at regular intervals and
        deduplicates overlapping shields.

        Args:
            roads_gdf: GeoDataFrame with road geometries and ref_normalized column.
            bbox: Geographic bounding box of the image.
            image_size: (width, height) of the target image in pixels.

        Returns:
            List of ShieldSpec objects to render.
        """
        if roads_gdf is None or len(roads_gdf) == 0:
            return []

        ref_col = "ref_normalized" if "ref_normalized" in roads_gdf.columns else "ref"
        if ref_col not in roads_gdf.columns:
            return []

        width, height = image_size

        # Geo-to-pixel conversion
        def geo_to_pixel(lon: float, lat: float) -> tuple[int, int]:
            x = int((lon - bbox.west) / (bbox.east - bbox.west) * width)
            y = int((bbox.north - lat) / (bbox.north - bbox.south) * height)
            return (x, y)

        shields: list[ShieldSpec] = []
        # Track placed shields per ref to enforce spacing
        placed_positions: dict[str, list[tuple[int, int]]] = {}

        for _, row in roads_gdf.iterrows():
            ref_val = row.get(ref_col)
            if ref_val is None or (isinstance(ref_val, float) and math.isnan(ref_val)):
                continue

            classification = cls.classify_ref(str(ref_val))
            if classification is None:
                continue

            shield_type, number = classification
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            # Get line geometries
            lines = []
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)

            for line in lines:
                if line.length == 0:
                    continue

                # Sample at midpoint and at regular intervals
                total_length = line.length
                interval = total_length * 0.5  # Place shield at midpoint

                point = line.interpolate(interval)
                px, py = geo_to_pixel(point.x, point.y)

                # Skip if outside image bounds
                if px < 20 or px > width - 20 or py < 20 or py > height - 20:
                    continue

                # Check spacing against already-placed shields of same ref
                ref_key = f"{shield_type}_{number}"
                too_close = False
                for existing_x, existing_y in placed_positions.get(ref_key, []):
                    dist = math.sqrt((px - existing_x) ** 2 + (py - existing_y) ** 2)
                    if dist < MIN_SHIELD_SPACING:
                        too_close = True
                        break

                if too_close:
                    continue

                shields.append(ShieldSpec(
                    shield_type=shield_type,
                    number=number,
                    x=px,
                    y=py,
                    ref=str(ref_val),
                ))
                placed_positions.setdefault(ref_key, []).append((px, py))

        return shields

    @classmethod
    def render_interstate_shield(
        cls, number: str, size: int = 48
    ) -> Image.Image:
        """Render an Interstate highway shield.

        Blue background with red top bar and white number.

        Args:
            number: Route number string (e.g., "10").
            size: Shield size in pixels.

        Returns:
            RGBA PIL Image of the shield.
        """
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Shield shape: blue pointed shield
        margin = size * 0.08
        mid_x = size / 2
        # Main blue body
        blue = (0, 57, 136)
        red = (175, 30, 45)
        white = (255, 255, 255)

        # Draw shield outline (simplified shield shape)
        shield_points = [
            (margin, size * 0.15),  # top-left
            (size - margin, size * 0.15),  # top-right
            (size - margin, size * 0.65),  # right
            (mid_x, size * 0.92),  # bottom point
            (margin, size * 0.65),  # left
        ]
        draw.polygon(shield_points, fill=blue, outline=white)

        # Red top bar
        red_points = [
            (margin, size * 0.15),
            (size - margin, size * 0.15),
            (size - margin, size * 0.32),
            (margin, size * 0.32),
        ]
        draw.polygon(red_points, fill=red)

        # White line between red and blue
        draw.line(
            [(margin, size * 0.32), (size - margin, size * 0.32)],
            fill=white, width=max(1, size // 24),
        )

        # Number text
        font_size = int(size * 0.35)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), number, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = mid_x - text_w / 2
        text_y = size * 0.38
        draw.text((text_x, text_y), number, fill=white, font=font)

        return img

    @classmethod
    def render_us_highway_shield(
        cls, number: str, size: int = 48
    ) -> Image.Image:
        """Render a US Highway shield.

        Black border, white interior with route number.

        Args:
            number: Route number string.
            size: Shield size in pixels.

        Returns:
            RGBA PIL Image of the shield.
        """
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        mid_x = size / 2
        mid_y = size / 2
        radius = size * 0.42
        inner_radius = size * 0.36

        black = (0, 0, 0)
        white = (255, 255, 255)

        # Outer black circle
        draw.ellipse(
            [mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius],
            fill=black,
        )
        # Inner white circle
        draw.ellipse(
            [mid_x - inner_radius, mid_y - inner_radius,
             mid_x + inner_radius, mid_y + inner_radius],
            fill=white,
        )

        # Number text
        font_size = int(size * 0.35)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), number, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = mid_x - text_w / 2
        text_y = mid_y - text_h / 2 - text_bbox[1]
        draw.text((text_x, text_y), number, fill=black, font=font)

        return img

    @classmethod
    def render_state_route_shield(
        cls, number: str, size: int = 48
    ) -> Image.Image:
        """Render a State Route shield.

        Simple circle with number.

        Args:
            number: Route number string.
            size: Shield size in pixels.

        Returns:
            RGBA PIL Image of the shield.
        """
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        mid_x = size / 2
        mid_y = size / 2
        radius = size * 0.42

        white = (255, 255, 255)
        black = (0, 0, 0)

        # White circle with black border
        border_w = max(2, size // 16)
        draw.ellipse(
            [mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius],
            fill=white, outline=black, width=border_w,
        )

        # Number text
        font_size = int(size * 0.35)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), number, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = mid_x - text_w / 2
        text_y = mid_y - text_h / 2 - text_bbox[1]
        draw.text((text_x, text_y), number, fill=black, font=font)

        return img

    @classmethod
    def render_shield(cls, shield_type: str, number: str, size: int = 48) -> Image.Image:
        """Render a shield of the given type.

        Args:
            shield_type: One of "interstate", "us_highway", "state_route".
            number: Route number string.
            size: Shield size in pixels.

        Returns:
            RGBA PIL Image of the shield.
        """
        if shield_type == "interstate":
            return cls.render_interstate_shield(number, size)
        elif shield_type == "us_highway":
            return cls.render_us_highway_shield(number, size)
        elif shield_type == "state_route":
            return cls.render_state_route_shield(number, size)
        else:
            return cls.render_state_route_shield(number, size)

    @classmethod
    def render_shields_on_image(
        cls,
        image: Image.Image,
        shields: list[ShieldSpec],
        scale: float = 1.0,
    ) -> Image.Image:
        """Composite all highway shields onto a map image.

        Args:
            image: Base map image to draw shields on.
            shields: List of ShieldSpec objects with positions and types.
            scale: Scale factor for shield size.

        Returns:
            New PIL Image with shields composited.
        """
        if not shields:
            return image

        result = image.copy().convert("RGBA")
        shield_size = max(24, int(48 * scale))

        # Pre-render shields (cache by type+number)
        shield_cache: dict[str, Image.Image] = {}

        for spec in shields:
            cache_key = f"{spec.shield_type}_{spec.number}"
            if cache_key not in shield_cache:
                shield_cache[cache_key] = cls.render_shield(
                    spec.shield_type, spec.number, shield_size
                )

            shield_img = shield_cache[cache_key]
            # Position shield centered on the point
            paste_x = spec.x - shield_size // 2
            paste_y = spec.y - shield_size // 2

            # Clamp to image bounds
            paste_x = max(0, min(result.width - shield_size, paste_x))
            paste_y = max(0, min(result.height - shield_size, paste_y))

            result.paste(shield_img, (paste_x, paste_y), shield_img)

        return result
