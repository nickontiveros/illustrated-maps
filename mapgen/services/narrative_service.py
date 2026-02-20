"""Narrative content service for activity markers and POI categorization."""

import logging
import math
from typing import Optional

import geopandas as gpd

from ..models.narrative import (
    ActivityCategory,
    ActivityMarker,
    HistoricalMarker,
    NarrativeSettings,
    OSM_TAG_TO_CATEGORY,
)
from ..models.project import BoundingBox

logger = logging.getLogger(__name__)


class NarrativeService:
    """Service for generating narrative content (activity markers, historical info).

    Reads OSM data from GeoDataFrames (typically the ``buildings`` layer
    produced by :class:`OSMService`) and converts tagged features into
    activity markers and historical markers suitable for rendering on the
    illustrated map.
    """

    # Tag columns to inspect when categorising a POI, in priority order.
    _TAG_COLUMNS = ("amenity", "tourism", "leisure", "historic")

    def __init__(self, settings: Optional[NarrativeSettings] = None):
        """Initialize narrative service.

        Args:
            settings: Narrative settings controlling marker limits.
                      Uses defaults if not provided.
        """
        self.settings = settings or NarrativeSettings()

    def extract_activity_markers(
        self,
        bbox: BoundingBox,
        buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> list[ActivityMarker]:
        """Extract activity markers from OSM building/POI data.

        Searches the GeoDataFrame for amenity/tourism/leisure tags and maps
        them to :class:`ActivityCategory` via :data:`OSM_TAG_TO_CATEGORY`.

        Args:
            bbox: Bounding box (currently used for logging context only;
                  the GeoDataFrame is assumed to already be clipped).
            buildings_gdf: GeoDataFrame with columns such as ``amenity``,
                           ``tourism``, ``leisure``, ``historic``, ``name``,
                           ``geometry``.

        Returns:
            Markers sorted by category name, limited to
            ``max_activity_markers``.
        """
        if buildings_gdf is None or len(buildings_gdf) == 0:
            logger.info("No building data provided for activity extraction")
            return []

        markers: list[ActivityMarker] = []

        for _idx, row in buildings_gdf.iterrows():
            category = self._categorize_poi(row)
            if category is None:
                continue

            # Extract position from geometry centroid
            geom = getattr(row, "geometry", None)
            if geom is None:
                continue
            try:
                centroid = geom.centroid
            except Exception:
                continue

            name = self._safe_get_str(row, "name")

            markers.append(
                ActivityMarker(
                    category=category,
                    latitude=centroid.y,
                    longitude=centroid.x,
                    name=name,
                )
            )

        # Sort by category value (alphabetical) for deterministic output
        markers.sort(key=lambda m: m.category.value)

        # Limit count
        max_markers = self.settings.max_activity_markers
        if len(markers) > max_markers:
            logger.info(
                "Trimming activity markers from %d to %d", len(markers), max_markers
            )
            markers = markers[:max_markers]

        logger.info(
            "Extracted %d activity markers within bbox (%.4f,%.4f)-(%.4f,%.4f)",
            len(markers),
            bbox.south,
            bbox.west,
            bbox.north,
            bbox.east,
        )
        return markers

    def _categorize_poi(self, row) -> Optional[ActivityCategory]:
        """Determine the activity category for a POI from its OSM tags.

        Checks the ``amenity``, ``tourism``, ``leisure``, and ``historic``
        columns in that order.  The first matching value found in
        :data:`OSM_TAG_TO_CATEGORY` wins.

        Args:
            row: A GeoDataFrame row (pandas Series-like).

        Returns:
            The matching :class:`ActivityCategory`, or ``None`` if no
            category applies.
        """
        for col in self._TAG_COLUMNS:
            tag_value = self._safe_get_str(row, col)
            if tag_value is None:
                continue

            # The tag value may be a compound like "yes" for building;
            # look it up directly.
            if tag_value in OSM_TAG_TO_CATEGORY:
                return OSM_TAG_TO_CATEGORY[tag_value]

        return None

    def extract_historical_markers(
        self,
        buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> list[HistoricalMarker]:
        """Extract historical markers from OSM data.

        Looks for features with a ``historic`` tag and a name.

        Args:
            buildings_gdf: GeoDataFrame of features to scan.

        Returns:
            A list of :class:`HistoricalMarker` instances, sorted by name.
        """
        if buildings_gdf is None or len(buildings_gdf) == 0:
            logger.info("No building data provided for historical extraction")
            return []

        markers: list[HistoricalMarker] = []

        for _idx, row in buildings_gdf.iterrows():
            historic_value = self._safe_get_str(row, "historic")
            if historic_value is None:
                continue

            name = self._safe_get_str(row, "name")
            if name is None:
                # Historical markers without a name are not useful
                continue

            # Extract position
            geom = getattr(row, "geometry", None)
            if geom is None:
                continue
            try:
                centroid = geom.centroid
            except Exception:
                continue

            # Try to get optional fields
            description = self._safe_get_str(row, "description")
            year = self._safe_get_int(row, "start_date")
            if year is None:
                year = self._safe_get_int(row, "year")

            markers.append(
                HistoricalMarker(
                    latitude=centroid.y,
                    longitude=centroid.x,
                    name=name,
                    description=description,
                    year=year,
                )
            )

        markers.sort(key=lambda m: m.name)

        logger.info("Extracted %d historical markers", len(markers))
        return markers

    def get_activity_icon_name(self, category: ActivityCategory) -> str:
        """Get the icon filename for an activity category.

        Args:
            category: The activity category.

        Returns:
            A string like ``'dining.png'``, ``'museum.png'``, etc.
        """
        return f"{category.value}.png"

    def cluster_nearby_markers(
        self,
        markers: list[ActivityMarker],
        cluster_distance_degrees: float = 0.001,
    ) -> list[ActivityMarker]:
        """Cluster nearby markers of the same category.

        When multiple markers of the same category are within
        *cluster_distance_degrees*, keep only the first one found to
        reduce clutter.  The distance check uses simple Euclidean
        distance in degree-space, which is adequate at the map scales
        this project targets.

        Args:
            markers: Input markers (not modified in place).
            cluster_distance_degrees: Maximum distance in degrees for
                markers of the same category to be considered duplicates.

        Returns:
            A filtered list of markers with nearby same-category
            duplicates removed.
        """
        if not markers:
            return []

        threshold_sq = cluster_distance_degrees ** 2
        kept: list[ActivityMarker] = []

        for marker in markers:
            is_duplicate = False
            for existing in kept:
                if existing.category != marker.category:
                    continue
                dlat = marker.latitude - existing.latitude
                dlon = marker.longitude - existing.longitude
                dist_sq = dlat * dlat + dlon * dlon
                if dist_sq <= threshold_sq:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(marker)

        if len(kept) < len(markers):
            logger.info(
                "Clustered %d activity markers down to %d (distance=%.5f deg)",
                len(markers),
                len(kept),
                cluster_distance_degrees,
            )

        return kept

    # ------------------------------------------------------------------
    # Safe accessor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get_str(row, key: str) -> Optional[str]:
        """Safely get a string value from a row.

        Returns None if the column is missing, the value is NaN, or
        the value is not a non-empty string.
        """
        try:
            val = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
        except Exception:
            return None

        if val is None:
            return None

        # Handle pandas NaN
        try:
            import pandas as pd

            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(val, str) and val.strip():
            return val.strip()

        return None

    @staticmethod
    def _safe_get_int(row, key: str) -> Optional[int]:
        """Safely get an integer value from a row.

        Attempts to parse string values that look like years (4-digit
        integers).  Returns None on any failure.
        """
        try:
            val = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
        except Exception:
            return None

        if val is None:
            return None

        # Handle pandas NaN
        try:
            import pandas as pd

            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass

        # Already an int or float
        if isinstance(val, (int, float)):
            try:
                return int(val)
            except (ValueError, OverflowError):
                return None

        # Try parsing from string (e.g. "1889", "~1900", "15th century")
        if isinstance(val, str):
            val = val.strip()
            # Try direct parse
            try:
                return int(val)
            except ValueError:
                pass

            # Try extracting a 4-digit year
            import re

            match = re.search(r"(\d{4})", val)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        return None
