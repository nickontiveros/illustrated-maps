"""Automated landmark discovery from OpenStreetMap data."""

import logging
import math
from typing import Optional

import geopandas as gpd
from shapely.geometry import Point

from ..models.landmark import Landmark, FeatureType
from ..models.narrative import LandmarkTier, TIER_SCALES, NarrativeSettings
from ..models.project import BoundingBox

logger = logging.getLogger(__name__)


# Mapping from OSM tag values to FeatureType for discovered landmarks
_TAG_TO_FEATURE_TYPE = {
    "castle": FeatureType.MONUMENT,
    "fort": FeatureType.MONUMENT,
    "palace": FeatureType.MONUMENT,
    "monument": FeatureType.MONUMENT,
    "memorial": FeatureType.MONUMENT,
    "ruins": FeatureType.MONUMENT,
    "cathedral": FeatureType.BUILDING,
    "church": FeatureType.BUILDING,
    "mosque": FeatureType.BUILDING,
    "temple": FeatureType.BUILDING,
    "university": FeatureType.CAMPUS,
    "stadium": FeatureType.STADIUM,
    "museum": FeatureType.BUILDING,
    "gallery": FeatureType.BUILDING,
    "theatre": FeatureType.BUILDING,
    "zoo": FeatureType.PARK,
    "aquarium": FeatureType.BUILDING,
    "theme_park": FeatureType.PARK,
    "attraction": FeatureType.BUILDING,
    "viewpoint": FeatureType.NATURAL,
    "park": FeatureType.PARK,
    "nature_reserve": FeatureType.NATURAL,
    "sports_centre": FeatureType.STADIUM,
    "marina": FeatureType.BUILDING,
    "hospital": FeatureType.BUILDING,
    "library": FeatureType.BUILDING,
    "place_of_worship": FeatureType.BUILDING,
}


class LandmarkDiscoveryService:
    """Discovers notable landmarks from OSM data within a region."""

    # OSM tags that indicate notable places, with importance weights
    NOTABLE_TAGS = {
        # Tourism (high importance)
        "tourism": {
            "attraction": 0.9,
            "museum": 0.85,
            "gallery": 0.7,
            "viewpoint": 0.7,
            "zoo": 0.8,
            "aquarium": 0.75,
            "theme_park": 0.9,
        },
        # Historic (high importance)
        "historic": {
            "castle": 0.9,
            "monument": 0.8,
            "memorial": 0.7,
            "ruins": 0.75,
            "fort": 0.8,
            "palace": 0.9,
            "cathedral": 0.95,
        },
        # Amenity (medium importance)
        "amenity": {
            "place_of_worship": 0.6,
            "theatre": 0.65,
            "university": 0.7,
            "hospital": 0.5,
            "library": 0.5,
        },
        # Leisure (medium importance)
        "leisure": {
            "stadium": 0.75,
            "sports_centre": 0.5,
            "marina": 0.6,
        },
        # Building (conditional - only if named and notable)
        "building": {
            "cathedral": 0.9,
            "church": 0.5,
            "stadium": 0.75,
            "university": 0.7,
        },
    }

    def __init__(self, settings: Optional[NarrativeSettings] = None):
        """Initialize landmark discovery service.

        Args:
            settings: Narrative settings controlling discovery behaviour.
                      Uses defaults if not provided.
        """
        self.settings = settings or NarrativeSettings()

    def discover_landmarks(
        self,
        bbox: BoundingBox,
        osm_service=None,  # Optional OSMService for direct queries
        buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> list[Landmark]:
        """Discover notable landmarks within a region.

        If buildings_gdf is provided, searches it for notable buildings.
        If osm_service is provided, queries OSM directly for POIs using Overpass.

        Steps:
        1. Search buildings_gdf for named/notable buildings (check 'name',
           'tourism', 'historic', 'amenity' columns)
        2. Score each building by tag importance and area
        3. Rank and filter by min_importance_score
        4. Assign tiers (MAJOR/NOTABLE/MINOR) based on score
        5. Create Landmark objects with appropriate scale from TIER_SCALES
        6. Limit to max_landmarks

        Args:
            bbox: Geographic bounding box for the search area.
            osm_service: Optional OSMService instance for direct Overpass queries.
            buildings_gdf: Optional GeoDataFrame of buildings already fetched.

        Returns:
            List of Landmark objects sorted by importance (most important first).
        """
        candidates: list[tuple[float, object]] = []

        # --- Source 1: buildings GeoDataFrame ---
        if buildings_gdf is not None and len(buildings_gdf) > 0:
            logger.info(
                "Scanning %d buildings for notable landmarks", len(buildings_gdf)
            )
            candidates.extend(self._scan_geodataframe(buildings_gdf))

        # --- Source 2: direct OSM query via osm_service ---
        if osm_service is not None:
            logger.info("Querying OSM for POIs via osm_service")
            try:
                pois_gdf = self._query_pois(osm_service, bbox)
                if pois_gdf is not None and len(pois_gdf) > 0:
                    candidates.extend(self._scan_geodataframe(pois_gdf))
            except Exception as exc:
                logger.warning("Failed to query POIs from OSM: %s", exc)

        if not candidates:
            logger.info("No landmark candidates found")
            return []

        # Deduplicate by name (keep the higher-scored entry)
        seen_names: dict[str, tuple[float, object]] = {}
        for score, row in candidates:
            name = self._get_name(row)
            if name is None:
                # Unnamed features are kept individually (no dedup key)
                key = id(row)
            else:
                key = name.lower().strip()

            if key not in seen_names or seen_names[key][0] < score:
                seen_names[key] = (score, row)

        unique_candidates = list(seen_names.values())

        # Filter by minimum importance
        min_score = self.settings.min_importance_score
        filtered = [
            (score, row)
            for score, row in unique_candidates
            if score >= min_score
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x[0], reverse=True)

        # Limit to max_landmarks
        filtered = filtered[: self.settings.max_landmarks]

        # Convert to Landmark objects
        landmarks: list[Landmark] = []
        for score, row in filtered:
            try:
                landmark = self._feature_to_landmark(row, score)
                landmarks.append(landmark)
            except Exception as exc:
                name = self._get_name(row)
                logger.warning(
                    "Failed to convert feature '%s' to Landmark: %s", name, exc
                )

        logger.info(
            "Discovered %d landmarks (from %d candidates, min_score=%.2f)",
            len(landmarks),
            len(unique_candidates),
            min_score,
        )
        return landmarks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_geodataframe(
        self, gdf: gpd.GeoDataFrame
    ) -> list[tuple[float, object]]:
        """Score every row in a GeoDataFrame and return (score, row) pairs.

        Only rows that have at least one notable tag are included.
        """
        results: list[tuple[float, object]] = []
        for idx, row in gdf.iterrows():
            score = self._score_feature(row)
            if score > 0:
                results.append((score, row))
        return results

    def _query_pois(self, osm_service, bbox: BoundingBox) -> Optional[gpd.GeoDataFrame]:
        """Query POIs through the osm_service using its existing extraction methods.

        Tries to fetch notable buildings and landmark buildings from the
        osm_service. Falls back gracefully if the methods are unavailable.
        """
        combined_frames: list[gpd.GeoDataFrame] = []

        # Try extract_notable_buildings
        try:
            notable = osm_service.extract_notable_buildings(bbox)
            if notable is not None and len(notable) > 0:
                combined_frames.append(notable)
        except Exception as exc:
            logger.debug("extract_notable_buildings failed: %s", exc)

        # Try extract_landmark_buildings
        try:
            landmarks = osm_service.extract_landmark_buildings(bbox)
            if landmarks is not None and len(landmarks) > 0:
                combined_frames.append(landmarks)
        except Exception as exc:
            logger.debug("extract_landmark_buildings failed: %s", exc)

        if not combined_frames:
            return None

        import pandas as pd

        combined = pd.concat(combined_frames, ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry")

    def _score_feature(self, row) -> float:
        """Calculate importance score for an OSM feature.

        Considers:
        - Tag importance from NOTABLE_TAGS
        - Whether it has a name (bonus +0.1)
        - Whether it has a Wikipedia/Wikidata link (bonus +0.15)
        - Building area (larger = more important, log scale)

        Returns:
            A float between 0.0 and ~1.3. Returns 0.0 if the feature has
            no notable tags at all.
        """
        base_score = 0.0

        # Check each tag category against NOTABLE_TAGS
        for tag_key, value_weights in self.NOTABLE_TAGS.items():
            tag_value = self._safe_get(row, tag_key)
            if tag_value is None:
                continue

            # tag_value may be a string or a list
            if isinstance(tag_value, list):
                for v in tag_value:
                    if v in value_weights:
                        base_score = max(base_score, value_weights[v])
            elif isinstance(tag_value, str):
                if tag_value in value_weights:
                    base_score = max(base_score, value_weights[tag_value])

        if base_score == 0.0:
            return 0.0

        # Name bonus
        name = self._get_name(row)
        if name is not None and len(name.strip()) > 0:
            base_score += 0.1

        # Wikipedia / Wikidata bonus
        wikipedia = self._safe_get(row, "wikipedia")
        wikidata = self._safe_get(row, "wikidata")
        if wikipedia is not None or wikidata is not None:
            base_score += 0.15

        # Area bonus (log scale, capped contribution)
        area = self._get_area(row)
        if area is not None and area > 0:
            # Use log10 of area in square degrees.  A typical large building
            # is ~1e-8 deg^2; a stadium ~1e-6; a park ~1e-4.
            log_area = math.log10(max(area, 1e-12))
            # Map range [-12, -3] to [0, 0.15]
            area_bonus = max(0.0, min(0.15, (log_area + 12) / 60.0))
            base_score += area_bonus

        # Clamp to [0, 1.3] — we allow above 1.0 so tier boundaries
        # still work even with many bonuses stacked.
        return min(base_score, 1.3)

    def _assign_tier(self, score: float) -> LandmarkTier:
        """Assign a tier based on importance score."""
        if score >= 0.8:
            return LandmarkTier.MAJOR
        elif score >= 0.6:
            return LandmarkTier.NOTABLE
        else:
            return LandmarkTier.MINOR

    def _feature_to_landmark(self, row, score: float) -> Landmark:
        """Convert an OSM feature row to a Landmark object.

        Extracts position from the geometry centroid, determines the
        feature type from OSM tags, and assigns tier-based scale.

        Args:
            row: A GeoDataFrame row (or Series-like object).
            score: The pre-computed importance score.

        Returns:
            A Landmark instance.
        """
        # -- Name --
        name = self._get_name(row) or "Unnamed landmark"

        # -- Position (centroid) --
        geom = row.geometry if hasattr(row, "geometry") else row.get("geometry")
        if geom is None:
            raise ValueError("Feature has no geometry")

        centroid = geom.centroid
        latitude = centroid.y
        longitude = centroid.x

        # -- Feature type --
        feature_type = self._determine_feature_type(row)

        # -- Tier and scale --
        tier = self._assign_tier(score)
        scale = TIER_SCALES[tier]

        return Landmark(
            name=name,
            latitude=latitude,
            longitude=longitude,
            feature_type=feature_type,
            scale=scale,
        )

    def _determine_feature_type(self, row) -> FeatureType:
        """Determine the best FeatureType for a feature based on its OSM tags."""
        # Check tag columns in priority order
        for tag_key in ("historic", "tourism", "building", "amenity", "leisure"):
            tag_value = self._safe_get(row, tag_key)
            if tag_value is None:
                continue

            if isinstance(tag_value, list):
                for v in tag_value:
                    if v in _TAG_TO_FEATURE_TYPE:
                        return _TAG_TO_FEATURE_TYPE[v]
            elif isinstance(tag_value, str):
                if tag_value in _TAG_TO_FEATURE_TYPE:
                    return _TAG_TO_FEATURE_TYPE[tag_value]

        # Default to BUILDING
        return FeatureType.BUILDING

    # ------------------------------------------------------------------
    # Safe accessor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(row, key: str):
        """Safely get a value from a row, returning None if missing or NaN.

        Works with both dict-like rows and pandas Series (via .get()).
        Treats pandas NaN/None as missing.
        """
        try:
            val = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
        except Exception:
            return None

        if val is None:
            return None

        # pandas NaN check
        try:
            import pandas as pd

            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass

        return val

    @staticmethod
    def _get_name(row) -> Optional[str]:
        """Extract the best available name from a row."""
        for col in ("name", "display_name", "name:en"):
            try:
                val = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
            except Exception:
                continue
            if val is not None:
                try:
                    import pandas as pd

                    if pd.isna(val):
                        continue
                except (TypeError, ValueError):
                    pass
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    @staticmethod
    def _get_area(row) -> Optional[float]:
        """Get the area of the feature's geometry, or None if unavailable."""
        geom = row.geometry if hasattr(row, "geometry") else row.get("geometry") if hasattr(row, "get") else None
        if geom is None:
            return None
        try:
            return geom.area
        except Exception:
            return None
