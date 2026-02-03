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

from PIL import Image

from ..models.project import BoundingBox, DetailLevel, Project, get_recommended_detail_level
from .blending_service import BlendingService, TileInfo
from .gemini_service import GeminiService, GenerationResult
from .osm_service import OSMData, OSMService
from .perspective_service import PerspectiveService
from .render_service import RenderService
from .satellite_service import SatelliteService

# Area threshold for per-tile OSM fetching (in km²)
PER_TILE_OSM_THRESHOLD = 10_000


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

    def generate_tile_reference(
        self,
        spec: TileSpec,
        apply_perspective: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Image.Image:
        """
        Generate reference image for a single tile.

        Args:
            spec: Tile specification
            apply_perspective: Whether to apply perspective transform
            progress_callback: Optional callback for progress updates

        Returns:
            Composite reference image (satellite + OSM)
        """
        tile_size = self.project.tiles.size

        # Fetch satellite imagery for tile
        if progress_callback:
            progress_callback("Fetching satellite imagery...")
        satellite_image = self.satellite.fetch_satellite_imagery(
            bbox=spec.bbox,
            output_size=(tile_size, tile_size),
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
            output_size=(tile_size, tile_size),
            osm_opacity=0.5,
            apply_perspective=apply_perspective,
        )

        return composite

    def generate_tile(
        self,
        spec: TileSpec,
        style_prompt: Optional[str] = None,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TileResult:
        """
        Generate a single illustrated tile.

        Args:
            spec: Tile specification
            style_prompt: Custom style prompt (uses project default if None)
            max_retries: Maximum retries on failure
            progress_callback: Optional callback for progress updates

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

        # Call Gemini with retries
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
    ) -> tuple[list[TileResult], GenerationProgress]:
        """
        Generate all tiles for the map.

        Args:
            progress_callback: Optional callback for progress updates
            max_retries: Maximum retries per tile

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

        for spec in specs:
            progress.current_tile = spec

            if progress_callback:
                progress_callback(progress)

            tile_start = time.time()
            result = self.generate_tile(spec, max_retries=max_retries)
            tile_time = time.time() - tile_start

            results.append(result)
            progress.tile_times.append(tile_time)

            if result.error:
                progress.failed_tiles += 1
            else:
                progress.completed_tiles += 1

            if progress_callback:
                progress_callback(progress)

        progress.current_tile = None
        return results, progress

    def assemble_tiles(
        self,
        results: list[TileResult],
        apply_perspective: bool = True,
    ) -> Optional[Image.Image]:
        """
        Assemble generated tiles into final image.

        Args:
            results: List of TileResult objects
            apply_perspective: Whether to apply perspective to final image

        Returns:
            Final assembled image, or None if too many failures
        """
        # Filter successful tiles
        successful = [r for r in results if r.generated_image is not None]

        if len(successful) < len(results) * 0.8:
            # Too many failures
            return None

        # Expected tile size
        expected_size = self.project.tiles.size
        tiles_config = self.project.tiles
        output_config = self.project.output

        # Calculate the natural grid dimensions that fit all tiles without clipping
        # This ensures no content is lost from edge tiles
        cols, rows = tiles_config.calculate_grid(output_config.width, output_config.height)
        natural_width = (cols - 1) * tiles_config.effective_size + expected_size
        natural_height = (rows - 1) * tiles_config.effective_size + expected_size

        # Convert to TileInfo for blending, resizing tiles if needed
        tile_infos = []
        for result in successful:
            tile_image = result.generated_image

            # Resize tile to expected size if Gemini returned different dimensions
            if tile_image.width != expected_size or tile_image.height != expected_size:
                tile_image = tile_image.resize(
                    (expected_size, expected_size),
                    Image.Resampling.LANCZOS,
                )

            tile_infos.append(TileInfo(
                image=tile_image,
                col=result.spec.col,
                row=result.spec.row,
                x_offset=result.spec.x_offset,
                y_offset=result.spec.y_offset,
            ))

        # Blend tiles at natural grid size (no content clipping)
        assembled = self._blending.blend_tiles(
            tiles=tile_infos,
            output_size=(natural_width, natural_height),
            overlap=tiles_config.overlap,
        )

        # Scale to desired output dimensions if different from natural size
        if (natural_width, natural_height) != (output_config.width, output_config.height):
            assembled = assembled.resize(
                (output_config.width, output_config.height),
                Image.Resampling.LANCZOS,
            )

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
