"""Tile generation orchestration service.

This service coordinates the full tiled map generation pipeline:
1. Split region into overlapping tiles
2. Generate reference images (satellite + OSM) for each tile
3. Send each tile to Gemini for illustration
4. Blend tiles together into final image
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models.project import BoundingBox, CardinalDirection, DetailLevel, Project, get_recommended_detail_level
from .blending_service import BlendingService, TileInfo
from .color_consistency_service import ColorConsistencyService
from .gemini_service import GeminiService, GenerationResult
from .osm_service import OSMData, OSMService
from .perspective_service import PerspectiveService
from .render_service import RenderService
from .road_style_service import RoadStyleService
from .satellite_service import SatelliteService

# Area threshold for per-tile OSM fetching (in km²)
PER_TILE_OSM_THRESHOLD = 10_000

# Water tile detection thresholds
WATER_BLUE_HUE_RANGE = (180, 240)  # Hue range for ocean blue (in degrees)
WATER_SATURATION_MIN = 30  # Minimum saturation for water detection
WATER_UNIFORMITY_THRESHOLD = 0.85  # % of pixels that must be "water-like"
WATER_VARIANCE_THRESHOLD = 500  # Max variance in pixel values for uniform tile

# Prompt for water-only tiles
WATER_TILE_PROMPT = (
    "Generate illustrated ocean water in a hand illustrated tourist map style. "
    "Create gentle, stylized waves with soft blue-green tones. "
    "The water should have subtle color variation and hand-painted texture. "
    "Use muted, warm colors that match a classic illustrated map aesthetic. "
    "Fill the entire tile with continuous water that tiles seamlessly. "
    "DO NOT include: land, boats, sea creatures, or any other elements."
)


@dataclass
class TileSpec:
    """Specification for a single tile."""

    col: int  # Column index (0-based)
    row: int  # Row index (0-based)
    bbox: BoundingBox  # Geographic bounds
    x_offset: int  # X offset in final image (pixels)
    y_offset: int  # Y offset in final image (pixels)
    position_desc: str  # Human-readable position (e.g., "top-left corner")


@dataclass
class TileResult:
    """Result of generating a single tile."""

    spec: TileSpec
    reference_image: Optional[Image.Image] = None  # Satellite + OSM composite
    generated_image: Optional[Image.Image] = None  # Gemini output
    generation_time: float = 0.0
    error: Optional[str] = None
    retries: int = 0


@dataclass
class GenerationProgress:
    """Progress tracking for tile generation."""

    total_tiles: int
    completed_tiles: int = 0
    failed_tiles: int = 0
    current_tile: Optional[TileSpec] = None
    start_time: float = field(default_factory=time.time)
    tile_times: list[float] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_tile_time(self) -> float:
        if not self.tile_times:
            return 0.0
        return sum(self.tile_times) / len(self.tile_times)

    @property
    def estimated_remaining(self) -> float:
        remaining = self.total_tiles - self.completed_tiles
        return remaining * self.avg_tile_time


class GenerationService:
    """Orchestrates the full tiled map generation pipeline."""

    def __init__(
        self,
        project: Project,
        gemini_service: Optional[GeminiService] = None,
        satellite_service: Optional[SatelliteService] = None,
        osm_service: Optional[OSMService] = None,
        cache_dir: Optional[Path] = None,
        detail_level: Optional[DetailLevel] = None,
    ):
        """
        Initialize generation service.

        Args:
            project: Project configuration
            gemini_service: Optional pre-configured Gemini service
            satellite_service: Optional pre-configured satellite service
            osm_service: Optional pre-configured OSM service
            cache_dir: Directory for caching tiles
            detail_level: Override detail level (auto-selected if None)
        """
        self.project = project
        self.cache_dir = cache_dir or (project.project_dir / "cache" if project.project_dir else None)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Calculate region area and determine detail level
        self._region_area_km2 = project.region.calculate_area_km2()
        self._detail_level = detail_level or project.region.get_recommended_detail_level()
        self._use_per_tile_osm = self._region_area_km2 > PER_TILE_OSM_THRESHOLD

        # Initialize services lazily
        self._gemini = gemini_service
        self._satellite = satellite_service
        self._osm = osm_service
        self._blending = BlendingService()

        # Cache for OSM data (for small regions, fetched once; for large regions, per-tile)
        self._osm_data: Optional[OSMData] = None
        self._osm_tile_cache: dict[tuple[int, int], OSMData] = {}

        # Store effective rotation (orientation_degrees takes precedence over cardinal)
        self._rotation_degrees = project.style.effective_rotation_degrees

        # Color consistency service (initialized if enabled)
        cc_strength = project.style.color_consistency_strength
        self._color_consistency: Optional[ColorConsistencyService] = (
            ColorConsistencyService(strength=cc_strength)
            if cc_strength > 0
            else None
        )

        # Road style service (initialized if enabled)
        road_style = project.style.road_style
        self._road_style: Optional[RoadStyleService] = (
            RoadStyleService(settings=road_style)
            if road_style is not None and road_style.enabled
            else None
        )

    @property
    def region_area_km2(self) -> float:
        """Return the region area in square kilometers."""
        return self._region_area_km2

    @property
    def detail_level(self) -> DetailLevel:
        """Return the detail level being used."""
        return self._detail_level

    @property
    def uses_per_tile_osm(self) -> bool:
        """Return whether per-tile OSM fetching is enabled."""
        return self._use_per_tile_osm

    @property
    def gemini(self) -> GeminiService:
        if self._gemini is None:
            self._gemini = GeminiService()
        return self._gemini

    @property
    def satellite(self) -> SatelliteService:
        if self._satellite is None:
            cache_path = str(self.cache_dir / "satellite") if self.cache_dir else None
            self._satellite = SatelliteService(cache_dir=cache_path)
        return self._satellite

    @property
    def osm(self) -> OSMService:
        if self._osm is None:
            cache_path = str(self.cache_dir / "osm") if self.cache_dir else None
            self._osm = OSMService(cache_dir=cache_path)
        return self._osm

    def _apply_orientation_rotation(self, image: Image.Image) -> Image.Image:
        """Apply rotation based on map orientation setting.

        Supports arbitrary angles via orientation_degrees. For exact
        multiples of 90 degrees, uses lossless transpose operations.

        Args:
            image: Image to rotate

        Returns:
            Rotated image (or original if rotation is 0)
        """
        degrees = self._rotation_degrees
        if degrees == 0:
            return image

        # Use lossless transpose for exact 90° multiples
        if degrees == 90:
            return image.transpose(Image.Transpose.ROTATE_90)
        elif degrees == 180:
            return image.transpose(Image.Transpose.ROTATE_180)
        elif degrees == 270:
            return image.transpose(Image.Transpose.ROTATE_270)

        # Arbitrary angle: use Image.rotate with expand to avoid clipping
        # Negative because PIL rotates counter-clockwise, we want clockwise
        rotated = image.rotate(
            -degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0) if image.mode == "RGBA" else None,
        )
        return rotated

    def calculate_tile_specs(self) -> list[TileSpec]:
        """
        Calculate tile specifications for the entire map.

        Returns:
            List of TileSpec objects describing each tile
        """
        tiles = self.project.tiles
        output = self.project.output
        region = self.project.region

        # Calculate grid dimensions
        cols, rows = tiles.calculate_grid(output.width, output.height)

        # Calculate geographic step per tile
        lon_step = region.width_degrees / cols
        lat_step = region.height_degrees / rows

        # Account for overlap in geographic coordinates
        overlap_lon = (tiles.overlap / tiles.size) * lon_step
        overlap_lat = (tiles.overlap / tiles.size) * lat_step

        specs = []
        for row in range(rows):
            for col in range(cols):
                # Calculate geographic bounds with overlap
                west = region.west + col * lon_step - (overlap_lon if col > 0 else 0)
                east = region.west + (col + 1) * lon_step + (overlap_lon if col < cols - 1 else 0)
                south = region.north - (row + 1) * lat_step - (overlap_lat if row < rows - 1 else 0)
                north = region.north - row * lat_step + (overlap_lat if row > 0 else 0)

                # Clamp to region bounds
                west = max(west, region.west)
                east = min(east, region.east)
                south = max(south, region.south)
                north = min(north, region.north)

                # Calculate pixel offsets
                x_offset = col * tiles.effective_size
                y_offset = row * tiles.effective_size

                # Generate position description
                position_desc = self._get_position_description(col, row, cols, rows)

                specs.append(TileSpec(
                    col=col,
                    row=row,
                    bbox=BoundingBox(north=north, south=south, east=east, west=west),
                    x_offset=x_offset,
                    y_offset=y_offset,
                    position_desc=position_desc,
                ))

        return specs

    def _get_position_description(self, col: int, row: int, cols: int, rows: int) -> str:
        """Get human-readable position description for a tile."""
        # Vertical position
        if row == 0:
            v_pos = "top"
        elif row == rows - 1:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Horizontal position
        if col == 0:
            h_pos = "left"
        elif col == cols - 1:
            h_pos = "right"
        else:
            h_pos = "center"

        # Combine
        if v_pos == "middle" and h_pos == "center":
            return "center"
        elif v_pos == "middle":
            return f"{h_pos} edge"
        elif h_pos == "center":
            return f"{v_pos} edge"
        else:
            return f"{v_pos}-{h_pos} corner"

    def fetch_osm_data(self, tile_bbox: Optional[BoundingBox] = None) -> OSMData:
        """
        Fetch and cache OSM data for the region or tile.

        For small regions (< 10,000 km²), fetches once for entire region.
        For large regions, fetches per-tile to avoid query size limits.

        Args:
            tile_bbox: Optional tile bounding box for per-tile fetching

        Returns:
            OSMData for the region or tile
        """
        if self._use_per_tile_osm and tile_bbox is not None:
            # Per-tile OSM fetching for large regions
            return self._fetch_osm_for_tile(tile_bbox)
        else:
            # Whole-region OSM fetching for small regions
            if self._osm_data is None:
                print(f"Fetching OSM data with detail level: {self._detail_level.value}")
                self._osm_data = self.osm.fetch_region_data(
                    self.project.region,
                    detail_level=self._detail_level.value,
                )
            return self._osm_data

    def _fetch_osm_for_tile(self, tile_bbox: BoundingBox) -> OSMData:
        """
        Fetch OSM data for a specific tile, with caching.

        Args:
            tile_bbox: Tile bounding box

        Returns:
            OSMData for the tile
        """
        # Create cache key from bbox (rounded to avoid float precision issues)
        cache_key = (
            round(tile_bbox.north, 4),
            round(tile_bbox.south, 4),
            round(tile_bbox.east, 4),
            round(tile_bbox.west, 4),
        )

        if cache_key not in self._osm_tile_cache:
            self._osm_tile_cache[cache_key] = self.osm.fetch_region_data(
                tile_bbox,
                detail_level=self._detail_level.value,
            )

        return self._osm_tile_cache[cache_key]

    def _is_water_tile(self, image: Image.Image) -> bool:
        """
        Detect if an image is predominantly water (uniform blue).

        Uses color analysis to detect ocean/water tiles:
        - Checks if most pixels are in the blue hue range
        - Verifies the image is relatively uniform (low variance)

        Args:
            image: PIL Image to analyze

        Returns:
            True if the image appears to be mostly water
        """
        # Convert to numpy array
        img_array = np.array(image.convert("RGB"))

        # Convert to HSV for better color analysis
        # We'll do a simplified HSV conversion
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Calculate value (max of RGB)
        v = np.maximum(np.maximum(r, g), b)

        # Calculate saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = v - min_rgb
        s = np.where(v > 0, (delta / v) * 100, 0)

        # Calculate hue (simplified)
        h = np.zeros_like(r, dtype=float)
        mask = delta > 0

        # Red is max
        red_max = (v == r) & mask
        h[red_max] = 60 * (((g[red_max] - b[red_max]) / delta[red_max]) % 6)

        # Green is max
        green_max = (v == g) & mask
        h[green_max] = 60 * (((b[green_max] - r[green_max]) / delta[green_max]) + 2)

        # Blue is max
        blue_max = (v == b) & mask
        h[blue_max] = 60 * (((r[blue_max] - g[blue_max]) / delta[blue_max]) + 4)

        # Check how many pixels are "water-like" (blue hue with some saturation)
        min_hue, max_hue = WATER_BLUE_HUE_RANGE
        is_blue = (h >= min_hue) & (h <= max_hue)
        has_saturation = s >= WATER_SATURATION_MIN
        is_water_pixel = is_blue & has_saturation

        water_ratio = np.mean(is_water_pixel)

        # Also check overall variance - water tiles are very uniform
        variance = np.var(img_array)

        is_water = (
            water_ratio >= WATER_UNIFORMITY_THRESHOLD
            or (water_ratio >= 0.5 and variance < WATER_VARIANCE_THRESHOLD)
        )

        return is_water

    def _tile_render_size(self, bbox: 'BoundingBox') -> tuple[int, int]:
        """Calculate render dimensions that preserve the geographic aspect ratio.

        Fits within tile_size × tile_size while matching the bbox's true
        geographic aspect ratio (accounting for latitude correction).

        Returns:
            (width, height) in pixels
        """
        import math

        tile_size = self.project.tiles.size
        geo_aspect = bbox.geographic_aspect_ratio  # width / height

        if geo_aspect >= 1.0:
            # Wider than tall
            w = tile_size
            h = max(1, round(tile_size / geo_aspect))
        else:
            # Taller than wide
            h = tile_size
            w = max(1, round(tile_size * geo_aspect))

        return (w, h)

    def generate_tile_reference(
        self,
        spec: TileSpec,
        apply_perspective: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Image.Image:
        """
        Generate reference image for a single tile.

        The render dimensions match the tile's geographic aspect ratio so that
        features are not squashed/stretched in the reference sent to Gemini.

        Args:
            spec: Tile specification
            apply_perspective: Whether to apply perspective transform
            progress_callback: Optional callback for progress updates

        Returns:
            Composite reference image (satellite + OSM)
        """
        render_size = self._tile_render_size(spec.bbox)

        # Fetch satellite imagery for tile
        if progress_callback:
            progress_callback("Fetching satellite imagery...")
        satellite_image = self.satellite.fetch_satellite_imagery(
            bbox=spec.bbox,
            output_size=render_size,
        )
        if progress_callback:
            progress_callback("Satellite imagery fetched")

        # Get OSM data (cached, or per-tile for large regions)
        if self._use_per_tile_osm:
            if progress_callback:
                progress_callback(f"Fetching OSM data ({self._detail_level.value} detail)...")
        osm_data = self.fetch_osm_data(tile_bbox=spec.bbox if self._use_per_tile_osm else None)
        if self._use_per_tile_osm and progress_callback:
            progress_callback("OSM data fetched")

        # Create perspective and render services
        perspective = PerspectiveService(
            angle=self.project.style.perspective_angle,
            convergence=0.7,
            vertical_scale=0.4,
            horizon_margin=0.0,  # No horizon for tiles, applied to final
        )
        render = RenderService(perspective_service=perspective)

        # Render composite
        composite = render.render_composite_reference(
            satellite_image=satellite_image,
            osm_data=osm_data,
            bbox=spec.bbox,
            output_size=render_size,
            osm_opacity=0.5,
            apply_perspective=apply_perspective,
        )

        # Apply enhanced road layer if configured
        if self._road_style is not None and osm_data.roads is not None:
            road_ref = self._road_style.create_reference_layer(
                osm_data.roads, spec.bbox, render_size
            )
            from PIL import Image as PILImage
            composite = PILImage.alpha_composite(composite.convert("RGBA"), road_ref)

        # Apply orientation rotation if not north-up
        composite = self._apply_orientation_rotation(composite)

        return composite

    def generate_tile(
        self,
        spec: TileSpec,
        style_prompt: Optional[str] = None,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None,
        style_reference: Optional[Image.Image] = None,
    ) -> TileResult:
        """
        Generate a single illustrated tile.

        Args:
            spec: Tile specification
            style_prompt: Custom style prompt (uses project default if None)
            max_retries: Maximum retries on failure
            progress_callback: Optional callback for progress updates
            style_reference: Optional style reference image for visual consistency

        Returns:
            TileResult with generated image
        """
        result = TileResult(spec=spec)

        # Check cache first
        cached = self._load_cached_tile(spec)
        if cached is not None:
            if progress_callback:
                progress_callback("Using cached tile")
            result.generated_image = cached
            return result

        # Generate reference image
        try:
            result.reference_image = self.generate_tile_reference(spec, progress_callback=progress_callback)
        except Exception as e:
            result.error = f"Failed to generate reference: {e}"
            return result

        # Save reference for debugging
        if self.cache_dir:
            ref_path = self.cache_dir / "references" / f"tile_{spec.col}_{spec.row}_ref.png"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            result.reference_image.save(ref_path)

        # Detect if this is a water-only tile and select appropriate prompt
        is_water = self._is_water_tile(result.reference_image)
        if is_water:
            if progress_callback:
                progress_callback("Detected water tile, using water prompt")
            prompt = WATER_TILE_PROMPT
        else:
            prompt = style_prompt or self.project.style.prompt
        last_error = None

        for attempt in range(max_retries):
            try:
                if progress_callback:
                    retry_msg = f" (retry {attempt + 1}/{max_retries})" if attempt > 0 else ""
                    progress_callback(f"Calling Gemini API{retry_msg}...")
                start_time = time.time()

                gen_result = self.gemini.generate_base_tile(
                    reference_image=result.reference_image,
                    style_prompt=prompt,
                    tile_position=spec.position_desc,
                    style_reference=style_reference,
                )

                result.generated_image = gen_result.image
                result.generation_time = time.time() - start_time
                result.retries = attempt

                # Cache the result
                self._save_cached_tile(spec, gen_result.image)
                if progress_callback:
                    progress_callback(f"Generated in {result.generation_time:.1f}s")

                return result

            except Exception as e:
                last_error = str(e)
                result.retries = attempt + 1
                if progress_callback:
                    progress_callback(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        result.error = f"Failed after {max_retries} attempts: {last_error}"
        return result

    def generate_all_tiles(
        self,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
        max_retries: int = 3,
        style_reference: Optional[Image.Image] = None,
    ) -> tuple[list[TileResult], GenerationProgress]:
        """
        Generate all tiles for the map.

        Args:
            progress_callback: Optional callback for progress updates
            max_retries: Maximum retries per tile
            style_reference: Optional style reference image; if None, first successful tile is auto-captured

        Returns:
            Tuple of (list of TileResults, final progress)
        """
        specs = self.calculate_tile_specs()
        progress = GenerationProgress(total_tiles=len(specs))
        results = []

        # Log generation settings
        print(f"Region area: {self._region_area_km2:,.0f} km²")
        print(f"Detail level: {self._detail_level.value}")
        print(f"Per-tile OSM: {'enabled' if self._use_per_tile_osm else 'disabled'}")

        # Pre-fetch OSM data for small regions (large regions use per-tile fetching)
        if not self._use_per_tile_osm:
            self.fetch_osm_data()

        # If no style reference provided, generate the central tile first and use it
        current_style_ref = style_reference

        if current_style_ref is None and len(specs) > 1:
            # Find the most central tile
            mid_col = max(s.col for s in specs) / 2.0
            mid_row = max(s.row for s in specs) / 2.0
            center_spec = min(specs, key=lambda s: (s.col - mid_col) ** 2 + (s.row - mid_row) ** 2)

            print(f"Generating central tile ({center_spec.col},{center_spec.row}) first as style reference")
            progress.current_tile = center_spec
            if progress_callback:
                progress_callback(progress)

            tile_start = time.time()
            center_result = self.generate_tile(center_spec, max_retries=max_retries)
            tile_time = time.time() - tile_start
            progress.tile_times.append(tile_time)

            if center_result.error:
                progress.failed_tiles += 1
                print(f"Central tile failed: {center_result.error}, proceeding without style reference")
            else:
                progress.completed_tiles += 1
                current_style_ref = center_result.generated_image
                print("Using central tile as style reference")

            if progress_callback:
                progress_callback(progress)

            # Generate remaining tiles (skip the center tile we already did)
            remaining_specs = [s for s in specs if not (s.col == center_spec.col and s.row == center_spec.row)]
        else:
            center_result = None
            remaining_specs = specs

        # Build results in original spec order
        result_map: dict[tuple[int, int], 'TileResult'] = {}
        if center_result is not None:
            result_map[(center_spec.col, center_spec.row)] = center_result

        for spec in remaining_specs:
            progress.current_tile = spec

            if progress_callback:
                progress_callback(progress)

            tile_start = time.time()
            result = self.generate_tile(
                spec,
                max_retries=max_retries,
                style_reference=current_style_ref,
            )
            tile_time = time.time() - tile_start

            result_map[(spec.col, spec.row)] = result
            progress.tile_times.append(tile_time)

            if result.error:
                progress.failed_tiles += 1
            else:
                progress.completed_tiles += 1

            if progress_callback:
                progress_callback(progress)

        # Return results in original spec order
        results = [result_map[(s.col, s.row)] for s in specs]
        progress.current_tile = None

        # Apply color consistency: histogram-match each tile to the style reference
        if self._color_consistency is not None and current_style_ref is not None:
            print("Applying cross-tile color consistency...")
            for result in results:
                if result.generated_image is not None:
                    result.generated_image = self._color_consistency.histogram_match(
                        result.generated_image, current_style_ref
                    )

        return results, progress

    def assemble_tiles(
        self,
        results: list[TileResult],
        apply_perspective: bool = True,
        tile_offsets: Optional[dict[str, dict[str, int]]] = None,
    ) -> Optional[Image.Image]:
        """
        Assemble generated tiles into final image.

        Each tile is generated at size×size pixels covering a geographic bbox
        that varies per tile (edge tiles have overlap on fewer sides). This means
        tiles have different pixels-per-degree scales, and the geographic overlap
        does NOT map to a fixed number of pixels.

        To assemble correctly, we:
        1. Compute each tile's unique geographic contribution (its grid cell, no overlap)
        2. Convert that to a pixel crop within the tile (based on its actual bbox)
        3. Resize each contribution to a uniform pixel size
        4. Place contributions edge-to-edge

        Args:
            results: List of TileResult objects
            apply_perspective: Whether to apply perspective to final image
            tile_offsets: Optional dict of "col,row" -> {"dx": int, "dy": int} offsets

        Returns:
            Final assembled image, or None if too many failures
        """
        # Filter successful tiles
        successful = [r for r in results if r.generated_image is not None]

        if len(successful) < len(results) * 0.8:
            # Too many failures
            return None

        tiles_config = self.project.tiles
        output_config = self.project.output
        region = self.project.region

        cols, rows = tiles_config.calculate_grid(output_config.width, output_config.height)
        lon_step = region.width_degrees / cols
        lat_step = region.height_degrees / rows

        # Each tile's contribution is one grid cell. Use uniform pixel size per cell.
        cell_w = round(output_config.width / cols)
        cell_h = round(output_config.height / rows)
        out_w = cell_w * cols
        out_h = cell_h * rows

        offsets = tile_offsets or {}
        assembled = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

        for result in successful:
            spec = result.spec
            tile_image = result.generated_image

            # Resize the generated tile to match the reference dimensions for this bbox.
            # Gemini may return a different resolution than the reference we sent it.
            expected_w, expected_h = self._tile_render_size(spec.bbox)
            if tile_image.width != expected_w or tile_image.height != expected_h:
                tile_image = tile_image.resize(
                    (expected_w, expected_h),
                    Image.Resampling.LANCZOS,
                )

            # This tile's geographic bbox (includes overlap)
            tile_geo_w = spec.bbox.east - spec.bbox.west
            tile_geo_h = spec.bbox.north - spec.bbox.south

            # This tile's unique contribution (its grid cell, no overlap)
            contrib_west = region.west + spec.col * lon_step
            contrib_east = region.west + (spec.col + 1) * lon_step
            contrib_north = region.north - spec.row * lat_step
            contrib_south = region.north - (spec.row + 1) * lat_step

            # Convert contribution bounds to pixel crop within the tile
            # (proportional to geographic position within the tile's bbox)
            crop_left = round((contrib_west - spec.bbox.west) / tile_geo_w * tile_image.width)
            crop_top = round((spec.bbox.north - contrib_north) / tile_geo_h * tile_image.height)
            crop_right = round((spec.bbox.east - contrib_east) / tile_geo_w * tile_image.width)
            crop_bottom = round((contrib_south - spec.bbox.south) / tile_geo_h * tile_image.height)

            cropped = tile_image.crop((
                crop_left,
                crop_top,
                tile_image.width - crop_right,
                tile_image.height - crop_bottom,
            ))

            # Resize to uniform contribution size
            cropped = cropped.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

            # Apply per-tile offset adjustment
            key = f"{spec.col},{spec.row}"
            offset = offsets.get(key, {})
            dx = offset.get("dx", 0)
            dy = offset.get("dy", 0)

            paste_x = spec.col * cell_w + dx
            paste_y = spec.row * cell_h + dy

            assembled.paste(cropped, (paste_x, paste_y))

        # Scale to desired output dimensions if different
        if (out_w, out_h) != (output_config.width, output_config.height):
            assembled = assembled.resize(
                (output_config.width, output_config.height),
                Image.Resampling.LANCZOS,
            )

        # Apply global color grading if color consistency is enabled
        if self._color_consistency is not None:
            # Use the first successful tile as the grading reference
            ref_tile = next(
                (r.generated_image for r in results if r.generated_image is not None),
                None,
            )
            if ref_tile is not None:
                print("Applying global color grading...")
                assembled = self._color_consistency.apply_global_grading(assembled, ref_tile)

        # Apply perspective to final image
        if apply_perspective:
            perspective = PerspectiveService(
                angle=self.project.style.perspective_angle,
                convergence=0.7,
                vertical_scale=0.4,
                horizon_margin=0.15,
            )
            assembled = perspective.transform_image(assembled)

        return assembled

    def _load_cached_tile(self, spec: TileSpec) -> Optional[Image.Image]:
        """Load a cached generated tile if it exists."""
        if self.cache_dir is None:
            return None

        cache_path = self.cache_dir / "generated" / f"tile_{spec.col}_{spec.row}.png"
        if cache_path.exists():
            return Image.open(cache_path).convert("RGBA")
        return None

    def _save_cached_tile(self, spec: TileSpec, image: Image.Image) -> None:
        """Save a generated tile to cache."""
        if self.cache_dir is None:
            return

        cache_path = self.cache_dir / "generated" / f"tile_{spec.col}_{spec.row}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_path)

    def generate_for_subregion(
        self,
        sub_bbox: BoundingBox,
        output_size: tuple[int, int],
        detail_level: Optional[DetailLevel] = None,
        style_prompt: Optional[str] = None,
        style_reference: Optional[Image.Image] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Image.Image]:
        """Generate an illustrated map for a specific sub-region at a given output size.

        Reuses the existing tile grid, reference composition, and Gemini pipeline
        but scoped to a smaller geographic area. Used by sectional generation to
        generate focus regions (cities) independently.

        Args:
            sub_bbox: Geographic bounds of the sub-region.
            output_size: (width, height) in pixels for the output.
            detail_level: Override detail level for this sub-region.
            style_prompt: Custom style prompt (uses project default if None).
            style_reference: Optional style reference image.
            progress_callback: Optional callback for progress updates.

        Returns:
            Illustrated image for the sub-region, or None on failure.
        """
        sub_detail = detail_level or get_recommended_detail_level(
            sub_bbox.calculate_area_km2()
        )

        if progress_callback:
            progress_callback(f"Fetching satellite imagery for sub-region...")

        satellite_image = self.satellite.fetch_satellite_imagery(
            bbox=sub_bbox,
            output_size=output_size,
        )

        if progress_callback:
            progress_callback(f"Fetching OSM data ({sub_detail.value} detail)...")

        osm_data = self.osm.fetch_region_data(
            sub_bbox,
            detail_level=sub_detail.value,
        )

        if progress_callback:
            progress_callback("Building reference composite...")

        perspective = PerspectiveService(
            angle=self.project.style.perspective_angle,
            convergence=0.7,
            vertical_scale=0.4,
            horizon_margin=0.0,
        )
        render = RenderService(perspective_service=perspective)

        reference = render.render_composite_reference(
            satellite_image=satellite_image,
            osm_data=osm_data,
            bbox=sub_bbox,
            output_size=output_size,
            osm_opacity=0.5,
            apply_perspective=False,
        )

        # Apply orientation rotation
        reference = self._apply_orientation_rotation(reference)

        prompt = style_prompt or self.project.style.prompt

        if progress_callback:
            progress_callback("Calling Gemini API for sub-region...")

        try:
            gen_result = self.gemini.generate_base_tile(
                reference_image=reference,
                style_prompt=prompt,
                tile_position="full region",
                style_reference=style_reference,
            )
            return gen_result.image
        except Exception as e:
            if progress_callback:
                progress_callback(f"Sub-region generation failed: {e}")
            return None

    def estimate_cost(self) -> dict:
        """Estimate generation costs (without requiring API key)."""
        specs = self.calculate_tile_specs()
        num_tiles = len(specs)

        # Approximate costs (same as GeminiService)
        cost_per_image = 0.13  # USD

        tile_cost = num_tiles * cost_per_image
        # Estimate seam repairs at ~20% of tiles
        seam_cost = int(num_tiles * 0.2) * cost_per_image

        return {
            "tile_cost": tile_cost,
            "landmark_cost": 0,  # Landmarks calculated separately
            "seam_cost": seam_cost,
            "total_cost": tile_cost + seam_cost,
        }

    def generate_single_test_tile(
        self,
        col: int = 0,
        row: int = 0,
    ) -> TileResult:
        """
        Generate a single tile for testing.

        Useful for validating the pipeline before running full generation.

        Args:
            col: Column index
            row: Row index

        Returns:
            TileResult for the specified tile
        """
        specs = self.calculate_tile_specs()

        # Find the specified tile
        for spec in specs:
            if spec.col == col and spec.row == row:
                return self.generate_tile(spec)

        raise ValueError(f"Tile ({col}, {row}) not found in grid")
