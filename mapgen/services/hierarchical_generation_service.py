"""Hierarchical overview-first generation service.

Generates maps via a 3-level hierarchy:
  Level 0 (Overview):  1 Gemini call → full-region illustration (~2048×1449)
  Level 1 (Medium):    6 calls (2×3 grid) → 2048×2048 each, guided by L0 crops
  Level 2 (Fine):     24 calls (2×2 per L1) → 2048×2048 each, guided by L1 crops
  Assembly:            Crop each L2 tile to unique contribution → A1 output

Each level receives a crop of the *same geographic area* at lower resolution as
style guidance, eliminating the geography-vs-style confusion of the flat pipeline.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ..models.project import BoundingBox, DetailLevel, Project, get_recommended_detail_level
from .color_consistency_service import ColorConsistencyService
from .gemini_service import GeminiService, GenerationResult
from .osm_service import OSMData, OSMService
from .perspective_service import PerspectiveService
from .render_service import RenderService
from .satellite_service import SatelliteService
from .terrain_service import TerrainService

logger = logging.getLogger(__name__)

# Overlap fraction between adjacent tiles at each level (12%)
TILE_OVERLAP_FRAC = 0.12

# Target tile size for Gemini calls
TILE_SIZE = 2048


@dataclass
class HierarchicalTileSpec:
    """Specification for a tile at any level of the hierarchy."""

    level: int  # 0=overview, 1=medium, 2=fine
    col: int
    row: int
    bbox: BoundingBox  # Geographic bounds (with overlap)
    parent_col: int = 0  # Parent tile indices (for L1/L2)
    parent_row: int = 0
    position_desc: str = ""


@dataclass
class HierarchicalProgress:
    """Progress tracking compatible with the existing GenerationProgress interface."""

    total_tiles: int  # Total across all levels (1 + 6 + 24 = 31)
    completed_tiles: int = 0
    failed_tiles: int = 0
    current_tile: Optional[HierarchicalTileSpec] = None
    start_time: float = field(default_factory=time.time)
    tile_times: list[float] = field(default_factory=list)
    phase: str = "generating_overview"
    phase_detail: Optional[str] = None
    phase_progress: Optional[tuple[int, int]] = None

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


class HierarchicalGenerationService:
    """Orchestrates hierarchical overview-first map generation."""

    # Grid dimensions for each level
    L1_COLS, L1_ROWS = 2, 3  # 6 medium tiles
    L2_COLS, L2_ROWS = 2, 2  # 4 fine tiles per medium tile → 24 total

    def __init__(
        self,
        project: Project,
        gemini_service: Optional[GeminiService] = None,
        satellite_service: Optional[SatelliteService] = None,
        osm_service: Optional[OSMService] = None,
        cache_dir: Optional[Path] = None,
        detail_level: Optional[DetailLevel] = None,
    ):
        self.project = project
        self.cache_dir = cache_dir or (project.project_dir / "cache" if project.project_dir else None)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._region_area_km2 = project.region.calculate_area_km2()
        self._detail_level = detail_level or project.region.get_recommended_detail_level()

        self._gemini = gemini_service
        self._satellite = satellite_service
        self._osm = osm_service
        self._terrain: Optional[TerrainService] = None

        # Cached data
        self._osm_data: Optional[OSMData] = None

        # Color consistency
        cc_strength = project.style.color_consistency_strength
        self._color_consistency: Optional[ColorConsistencyService] = (
            ColorConsistencyService(strength=cc_strength)
            if cc_strength > 0
            else None
        )

        # Store effective rotation
        self._rotation_degrees = project.style.effective_rotation_degrees

    # -- Lazy service properties --

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

    @property
    def terrain(self) -> TerrainService:
        if self._terrain is None:
            cache_path = str(self.cache_dir / "terrain") if self.cache_dir else None
            self._terrain = TerrainService(cache_dir=cache_path)
        return self._terrain

    # -- Reference image helpers --

    def _ensure_osm_data(self) -> OSMData:
        """Fetch and cache OSM data for the full region (single Overpass call)."""
        if self._osm_data is None:
            region = self.project.generation_bbox
            logger.info("Fetching OSM data for full region (single call)...")
            self._osm_data = self.osm.fetch_region_data(
                region, detail_level=self._detail_level.value,
            )
        return self._osm_data

    def _fetch_reference(
        self,
        bbox: BoundingBox,
        output_size: tuple[int, int],
    ) -> Image.Image:
        """Fetch satellite + OSM composite for a geographic area.

        OSM data is fetched once for the full region and reused for all tiles.
        Satellite imagery is fetched per-tile (individual map tiles are cached).
        At COUNTRY detail level, OSM is skipped (satellite already shows the
        features that matter and the Overpass query is very slow).
        """
        satellite_image = self.satellite.fetch_satellite_imagery(
            bbox=bbox,
            output_size=output_size,
        )

        # Skip OSM overlay for country-scale regions — the satellite already
        # shows highways, rivers, coastlines, and city extents clearly, and
        # the Overpass query for such large areas is extremely slow.
        if self._detail_level == DetailLevel.COUNTRY:
            logger.info("Skipping OSM overlay (COUNTRY detail level)")
            return satellite_image.resize(output_size, Image.Resampling.LANCZOS).convert("RGBA")

        osm_data = self._ensure_osm_data()

        perspective = PerspectiveService(
            angle=self.project.style.perspective_angle,
            convergence=0.7,
            vertical_scale=0.4,
            horizon_margin=0.0,
        )
        render = RenderService(perspective_service=perspective)

        return render.render_composite_reference(
            satellite_image=satellite_image,
            osm_data=osm_data,
            bbox=bbox,
            output_size=output_size,
            osm_opacity=0.5,
            apply_perspective=False,
        )

    def _get_terrain_description(self, bbox: BoundingBox = None) -> Optional[str]:
        """Get terrain description (fetched once for full region and cached)."""
        if not hasattr(self, '_terrain_desc_cached'):
            try:
                region = self.project.generation_bbox
                elev_data = self.terrain.fetch_elevation_data(region)
                self._terrain_desc_cached = self.terrain.get_terrain_description(elev_data)
            except Exception as e:
                logger.debug("Terrain fetch skipped: %s", e)
                self._terrain_desc_cached = None
        return self._terrain_desc_cached

    # -- Grid geometry --

    def _subdivide_bbox(
        self,
        parent_bbox: BoundingBox,
        cols: int,
        rows: int,
        overlap_frac: float = TILE_OVERLAP_FRAC,
    ) -> list[list[BoundingBox]]:
        """Subdivide a bbox into a grid of overlapping sub-bboxes.

        Returns a 2D list indexed as result[row][col].
        """
        lon_step = parent_bbox.width_degrees / cols
        lat_step = parent_bbox.height_degrees / rows
        overlap_lon = lon_step * overlap_frac
        overlap_lat = lat_step * overlap_frac

        grid: list[list[BoundingBox]] = []
        for row in range(rows):
            row_bboxes: list[BoundingBox] = []
            for col in range(cols):
                west = parent_bbox.west + col * lon_step - (overlap_lon if col > 0 else 0)
                east = parent_bbox.west + (col + 1) * lon_step + (overlap_lon if col < cols - 1 else 0)
                north = parent_bbox.north - row * lat_step + (overlap_lat if row > 0 else 0)
                south = parent_bbox.north - (row + 1) * lat_step - (overlap_lat if row < rows - 1 else 0)

                # Clamp to parent bounds
                west = max(west, parent_bbox.west)
                east = min(east, parent_bbox.east)
                north = min(north, parent_bbox.north)
                south = max(south, parent_bbox.south)

                row_bboxes.append(BoundingBox(north=north, south=south, east=east, west=west))
            grid.append(row_bboxes)
        return grid

    def _crop_illustration_for_tile(
        self,
        illustration: Image.Image,
        parent_bbox: BoundingBox,
        tile_bbox: BoundingBox,
    ) -> Image.Image:
        """Crop an illustration image to the geographic area of a tile.

        Maps tile_bbox within parent_bbox to pixel coordinates in the illustration.
        """
        parent_w = parent_bbox.width_degrees
        parent_h = parent_bbox.height_degrees

        # Proportional pixel coordinates
        left = round((tile_bbox.west - parent_bbox.west) / parent_w * illustration.width)
        right = round((tile_bbox.east - parent_bbox.west) / parent_w * illustration.width)
        top = round((parent_bbox.north - tile_bbox.north) / parent_h * illustration.height)
        bottom = round((parent_bbox.north - tile_bbox.south) / parent_h * illustration.height)

        # Clamp to image bounds
        left = max(0, min(left, illustration.width - 1))
        right = max(left + 1, min(right, illustration.width))
        top = max(0, min(top, illustration.height - 1))
        bottom = max(top + 1, min(bottom, illustration.height))

        return illustration.crop((left, top, right, bottom))

    @staticmethod
    def _position_desc(col: int, row: int, cols: int, rows: int) -> str:
        """Human-readable position description."""
        if row == 0:
            v = "top"
        elif row == rows - 1:
            v = "bottom"
        else:
            v = "middle"

        if col == 0:
            h = "left"
        elif col == cols - 1:
            h = "right"
        else:
            h = "center"

        if v == "middle" and h == "center":
            return "center"
        if v == "middle":
            return f"{h} edge"
        if h == "center":
            return f"{v} edge"
        return f"{v}-{h} corner"

    # -- Cache helpers --

    def _cache_path(self, level: int, col: int, row: int, parent_col: int = 0, parent_row: int = 0) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        subdir = self.cache_dir / "hierarchical"
        if level == 0:
            return subdir / "overview.png"
        elif level == 1:
            return subdir / f"L1_{col}_{row}.png"
        else:
            return subdir / f"L2_{parent_col}_{parent_row}_{col}_{row}.png"

    def _load_cached(self, level: int, col: int, row: int, parent_col: int = 0, parent_row: int = 0) -> Optional[Image.Image]:
        path = self._cache_path(level, col, row, parent_col, parent_row)
        if path and path.exists():
            return Image.open(path).convert("RGBA")
        return None

    def _save_cached(self, image: Image.Image, level: int, col: int, row: int, parent_col: int = 0, parent_row: int = 0) -> None:
        path = self._cache_path(level, col, row, parent_col, parent_row)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)

        # For L2 tiles, also save to the flat cache path so the existing
        # tile preview API can serve them during generation.
        if level == 2 and self.cache_dir:
            global_col = parent_col * self.L2_COLS + col
            global_row = parent_row * self.L2_ROWS + row
            flat_dir = self.cache_dir / "generated"
            flat_dir.mkdir(parents=True, exist_ok=True)
            flat_path = flat_dir / f"tile_{global_col}_{global_row}.png"
            image.save(flat_path)
            logger.debug("Also saved to flat cache: %s", flat_path)

    # -- Generation phases --

    def generate_overview(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[Image.Image]:
        """Level 0: Generate full-region overview illustration.

        Returns the overview image or None on failure.
        """
        # Check cache
        cached = self._load_cached(0, 0, 0)
        if cached is not None:
            logger.info("Using cached overview")
            return cached

        region = self.project.generation_bbox
        # Overview size: fit within 2048 preserving aspect ratio
        aspect = region.geographic_aspect_ratio
        if aspect >= 1:
            ov_w = TILE_SIZE
            ov_h = round(TILE_SIZE / aspect)
        else:
            ov_h = TILE_SIZE
            ov_w = round(TILE_SIZE * aspect)

        if progress_callback:
            progress_callback("Fetching overview reference...")

        reference = self._fetch_reference(region, (ov_w, ov_h))

        terrain_desc = self._get_terrain_description(region)

        if progress_callback:
            progress_callback("Calling Gemini for overview...")

        try:
            result = self.gemini.generate_overview(
                reference_image=reference,
                style_prompt=None,
                terrain_description=terrain_desc,
            )
            self._save_cached(result.image, 0, 0, 0)
            return result.image
        except Exception as e:
            logger.error("Overview generation failed: %s", e)
            if progress_callback:
                progress_callback(f"Overview failed: {e}")
            return None

    def generate_medium_tiles(
        self,
        overview: Image.Image,
        progress_callback: Optional[Callable] = None,
    ) -> list[list[Optional[Image.Image]]]:
        """Level 1: Generate 2x3 grid of medium tiles guided by overview crops.

        Returns a 2D list indexed as result[row][col].
        """
        region = self.project.generation_bbox
        l1_grid = self._subdivide_bbox(region, self.L1_COLS, self.L1_ROWS)

        results: list[list[Optional[Image.Image]]] = []

        for row in range(self.L1_ROWS):
            row_results: list[Optional[Image.Image]] = []
            for col in range(self.L1_COLS):
                tile_bbox = l1_grid[row][col]

                # Check cache
                cached = self._load_cached(1, col, row)
                if cached is not None:
                    logger.info("Using cached L1 tile (%d,%d)", col, row)
                    row_results.append(cached)
                    if progress_callback:
                        progress_callback(f"L1 tile ({col},{row}): cached")
                    continue

                pos_desc = self._position_desc(col, row, self.L1_COLS, self.L1_ROWS)

                # Crop overview for this tile's geographic area
                illustration_crop = self._crop_illustration_for_tile(overview, region, tile_bbox)

                if progress_callback:
                    progress_callback(f"Fetching L1 reference ({col},{row})...")

                reference = self._fetch_reference(tile_bbox, (TILE_SIZE, TILE_SIZE))

                terrain_desc = self._get_terrain_description(tile_bbox)

                if progress_callback:
                    progress_callback(f"Generating L1 tile ({col},{row})...")

                try:
                    result = self.gemini.generate_enhanced_tile(
                        illustration_crop=illustration_crop,
                        reference_image=reference,
                        level="medium",
                        terrain_description=terrain_desc,
                        tile_position=pos_desc,
                    )
                    self._save_cached(result.image, 1, col, row)
                    row_results.append(result.image)
                except Exception as e:
                    logger.error("L1 tile (%d,%d) failed: %s", col, row, e)
                    if progress_callback:
                        progress_callback(f"L1 tile ({col},{row}) failed: {e}")
                    row_results.append(None)

            results.append(row_results)
        return results

    def generate_fine_tiles(
        self,
        medium_tiles: list[list[Optional[Image.Image]]],
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """Level 2: Generate 2x2 sub-tiles for each medium tile.

        Returns a flat list of dicts with keys:
            'image': Image or None
            'l1_col', 'l1_row': parent medium tile indices
            'l2_col', 'l2_row': sub-tile indices within parent
            'bbox': geographic bounding box for this fine tile
        """
        region = self.project.generation_bbox
        l1_grid = self._subdivide_bbox(region, self.L1_COLS, self.L1_ROWS)

        fine_tiles: list[dict] = []

        for l1_row in range(self.L1_ROWS):
            for l1_col in range(self.L1_COLS):
                medium_img = medium_tiles[l1_row][l1_col]
                l1_bbox = l1_grid[l1_row][l1_col]

                # Subdivide this medium tile's bbox into 2x2
                l2_grid = self._subdivide_bbox(l1_bbox, self.L2_COLS, self.L2_ROWS)

                for l2_row in range(self.L2_ROWS):
                    for l2_col in range(self.L2_COLS):
                        tile_bbox = l2_grid[l2_row][l2_col]

                        # Check cache
                        cached = self._load_cached(2, l2_col, l2_row, l1_col, l1_row)
                        if cached is not None:
                            logger.info("Using cached L2 tile (%d,%d)->(%d,%d)", l1_col, l1_row, l2_col, l2_row)
                            fine_tiles.append({
                                'image': cached,
                                'l1_col': l1_col, 'l1_row': l1_row,
                                'l2_col': l2_col, 'l2_row': l2_row,
                                'bbox': tile_bbox,
                            })
                            if progress_callback:
                                progress_callback(f"L2 ({l1_col},{l1_row})->({l2_col},{l2_row}): cached")
                            continue

                        # If the parent medium tile failed, skip
                        if medium_img is None:
                            logger.warning("Skipping L2 (%d,%d)->(%d,%d): parent failed", l1_col, l1_row, l2_col, l2_row)
                            fine_tiles.append({
                                'image': None,
                                'l1_col': l1_col, 'l1_row': l1_row,
                                'l2_col': l2_col, 'l2_row': l2_row,
                                'bbox': tile_bbox,
                            })
                            if progress_callback:
                                progress_callback(f"L2 ({l1_col},{l1_row})->({l2_col},{l2_row}): skipped (parent failed)")
                            continue

                        # Crop medium tile for this area
                        illustration_crop = self._crop_illustration_for_tile(medium_img, l1_bbox, tile_bbox)

                        # Global position for position description
                        global_col = l1_col * self.L2_COLS + l2_col
                        global_row = l1_row * self.L2_ROWS + l2_row
                        total_cols = self.L1_COLS * self.L2_COLS
                        total_rows = self.L1_ROWS * self.L2_ROWS
                        pos_desc = self._position_desc(global_col, global_row, total_cols, total_rows)

                        if progress_callback:
                            progress_callback(f"Fetching L2 reference ({l1_col},{l1_row})->({l2_col},{l2_row})...")

                        reference = self._fetch_reference(tile_bbox, (TILE_SIZE, TILE_SIZE))

                        terrain_desc = self._get_terrain_description(tile_bbox)

                        if progress_callback:
                            progress_callback(f"Generating L2 ({l1_col},{l1_row})->({l2_col},{l2_row})...")

                        try:
                            result = self.gemini.generate_enhanced_tile(
                                illustration_crop=illustration_crop,
                                reference_image=reference,
                                level="fine",
                                terrain_description=terrain_desc,
                                tile_position=pos_desc,
                            )
                            self._save_cached(result.image, 2, l2_col, l2_row, l1_col, l1_row)
                            fine_tiles.append({
                                'image': result.image,
                                'l1_col': l1_col, 'l1_row': l1_row,
                                'l2_col': l2_col, 'l2_row': l2_row,
                                'bbox': tile_bbox,
                            })
                        except Exception as e:
                            logger.error("L2 tile (%d,%d)->(%d,%d) failed: %s", l1_col, l1_row, l2_col, l2_row, e)
                            if progress_callback:
                                progress_callback(f"L2 ({l1_col},{l1_row})->({l2_col},{l2_row}) failed: {e}")
                            fine_tiles.append({
                                'image': None,
                                'l1_col': l1_col, 'l1_row': l1_row,
                                'l2_col': l2_col, 'l2_row': l2_row,
                                'bbox': tile_bbox,
                            })

        return fine_tiles

    def assemble(
        self,
        fine_tiles: list[dict],
        apply_color_grading: bool = True,
        color_reference: Optional[Image.Image] = None,
    ) -> Optional[Image.Image]:
        """Assemble L2 fine tiles into the final output image.

        Each fine tile is cropped to its unique geographic contribution (grid cell
        without overlap), resized to uniform cell size, and placed edge-to-edge.

        Args:
            fine_tiles: List of dicts from generate_fine_tiles().
            apply_color_grading: Whether to apply global color grading.
            color_reference: Optional image to use as color grading reference
                (e.g. the overview). Falls back to first successful tile.

        Returns:
            Final assembled image, or None if too many failures.
        """
        successful = [t for t in fine_tiles if t['image'] is not None]
        if len(successful) < len(fine_tiles) * 0.8:
            logger.error("Too many failed tiles (%d/%d)", len(fine_tiles) - len(successful), len(fine_tiles))
            return None

        region = self.project.generation_bbox
        output = self.project.output
        canvas_w, canvas_h = self.project.canvas_size
        rotation = self._rotation_degrees

        # The fine tiles form a 4×6 grid (L1_COLS * L2_COLS × L1_ROWS * L2_ROWS)
        total_cols = self.L1_COLS * self.L2_COLS
        total_rows = self.L1_ROWS * self.L2_ROWS

        # Uniform cell size
        cell_w = round(canvas_w / total_cols)
        cell_h = round(canvas_h / total_rows)
        out_w = cell_w * total_cols
        out_h = cell_h * total_rows

        # Geographic step per cell (without overlap)
        lon_step = region.width_degrees / total_cols
        lat_step = region.height_degrees / total_rows

        assembled = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

        for tile_info in successful:
            tile_image = tile_info['image']
            tile_bbox = tile_info['bbox']

            global_col = tile_info['l1_col'] * self.L2_COLS + tile_info['l2_col']
            global_row = tile_info['l1_row'] * self.L2_ROWS + tile_info['l2_row']

            # Resize tile to TILE_SIZE×TILE_SIZE if Gemini returned different dims
            if tile_image.width != TILE_SIZE or tile_image.height != TILE_SIZE:
                tile_image = tile_image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)

            # This tile's unique contribution (its grid cell, no overlap)
            contrib_west = region.west + global_col * lon_step
            contrib_east = region.west + (global_col + 1) * lon_step
            contrib_north = region.north - global_row * lat_step
            contrib_south = region.north - (global_row + 1) * lat_step

            # Convert contribution to pixel crop within the tile
            tile_geo_w = tile_bbox.east - tile_bbox.west
            tile_geo_h = tile_bbox.north - tile_bbox.south

            if tile_geo_w <= 0 or tile_geo_h <= 0:
                continue

            crop_left = round((contrib_west - tile_bbox.west) / tile_geo_w * tile_image.width)
            crop_top = round((tile_bbox.north - contrib_north) / tile_geo_h * tile_image.height)
            crop_right = round((tile_bbox.east - contrib_east) / tile_geo_w * tile_image.width)
            crop_bottom = round((contrib_south - tile_bbox.south) / tile_geo_h * tile_image.height)

            # Clamp
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = max(0, crop_right)
            crop_bottom = max(0, crop_bottom)

            cropped = tile_image.crop((
                crop_left,
                crop_top,
                tile_image.width - crop_right,
                tile_image.height - crop_bottom,
            ))

            # Resize to uniform cell size
            cropped = cropped.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

            paste_x = global_col * cell_w
            paste_y = global_row * cell_h
            assembled.paste(cropped, (paste_x, paste_y))

        # Apply orientation rotation
        needs_rotation = rotation != 0 and rotation % 360 != 0
        if needs_rotation:
            assembled = assembled.rotate(
                -rotation,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(0, 0, 0, 0),
            )
            cx, cy = assembled.width // 2, assembled.height // 2
            final_w, final_h = output.width, output.height
            left = cx - final_w // 2
            top = cy - final_h // 2
            assembled = assembled.crop((left, top, left + final_w, top + final_h))
        elif (out_w, out_h) != (output.width, output.height):
            assembled = assembled.resize(
                (output.width, output.height),
                Image.Resampling.LANCZOS,
            )

        # Apply global color grading
        if apply_color_grading and self._color_consistency is not None:
            ref_tile = color_reference if color_reference is not None else next(
                (t['image'] for t in fine_tiles if t['image'] is not None),
                None,
            )
            if ref_tile is not None:
                logger.info("Applying global color grading...")
                assembled = self._color_consistency.apply_global_grading(assembled, ref_tile)

        return assembled

    def _assemble_l1(
        self,
        medium_tiles: list[list[Optional[Image.Image]]],
        l1_grid: list[list[BoundingBox]],
    ) -> Optional[Image.Image]:
        """Assemble L1 medium tiles directly (quick-test / skip_l2 mode).

        Uses the same contribution-cell logic as assemble() but for a 2×3 grid.
        """
        region = self.project.generation_bbox
        output = self.project.output
        canvas_w, canvas_h = self.project.canvas_size

        cell_w = round(canvas_w / self.L1_COLS)
        cell_h = round(canvas_h / self.L1_ROWS)
        out_w = cell_w * self.L1_COLS
        out_h = cell_h * self.L1_ROWS

        lon_step = region.width_degrees / self.L1_COLS
        lat_step = region.height_degrees / self.L1_ROWS

        assembled = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

        for row in range(self.L1_ROWS):
            for col in range(self.L1_COLS):
                tile_img = medium_tiles[row][col]
                if tile_img is None:
                    continue

                tile_bbox = l1_grid[row][col]

                if tile_img.width != TILE_SIZE or tile_img.height != TILE_SIZE:
                    tile_img = tile_img.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)

                # Contribution cell
                contrib_west = region.west + col * lon_step
                contrib_east = region.west + (col + 1) * lon_step
                contrib_north = region.north - row * lat_step
                contrib_south = region.north - (row + 1) * lat_step

                tile_geo_w = tile_bbox.east - tile_bbox.west
                tile_geo_h = tile_bbox.north - tile_bbox.south
                if tile_geo_w <= 0 or tile_geo_h <= 0:
                    continue

                crop_left = round((contrib_west - tile_bbox.west) / tile_geo_w * tile_img.width)
                crop_top = round((tile_bbox.north - contrib_north) / tile_geo_h * tile_img.height)
                crop_right = round((tile_bbox.east - contrib_east) / tile_geo_w * tile_img.width)
                crop_bottom = round((contrib_south - tile_bbox.south) / tile_geo_h * tile_img.height)

                crop_left = max(0, crop_left)
                crop_top = max(0, crop_top)
                crop_right = max(0, crop_right)
                crop_bottom = max(0, crop_bottom)

                cropped = tile_img.crop((
                    crop_left, crop_top,
                    tile_img.width - crop_right,
                    tile_img.height - crop_bottom,
                ))
                cropped = cropped.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                assembled.paste(cropped, (col * cell_w, row * cell_h))

        if (out_w, out_h) != (output.width, output.height):
            assembled = assembled.resize(
                (output.width, output.height),
                Image.Resampling.LANCZOS,
            )

        return assembled

    # -- Main orchestrator --

    def generate_all_tiles(
        self,
        progress_callback: Optional[Callable] = None,
        max_retries: int = 3,
        style_reference: Optional[Image.Image] = None,
        skip_l2: bool = False,
    ) -> tuple[list, HierarchicalProgress]:
        """Orchestrate the full hierarchical generation pipeline.

        The signature matches GenerationService.generate_all_tiles() so it can
        be used as a drop-in replacement via the API.

        Args:
            progress_callback: Callback receiving HierarchicalProgress updates.
            max_retries: Max retries per tile (used for retry wrapper).
            style_reference: Ignored (hierarchical mode generates its own reference).
            skip_l2: If True, only generate L0 overview + L1 medium tiles (7
                     calls instead of 31). Useful for quick iteration to validate
                     style and geography before committing to the full run.

        Returns:
            Tuple of (fine_tiles list, progress). The fine_tiles list contains dicts
            but the API layer only uses this for tile counting.
        """
        l1_count = self.L1_COLS * self.L1_ROWS
        l2_count = l1_count * self.L2_COLS * self.L2_ROWS
        total = 1 + l1_count + (0 if skip_l2 else l2_count)
        progress = HierarchicalProgress(total_tiles=total)

        logger.info("Starting hierarchical generation: %d total calls", total)
        logger.info("Region area: %.0f km²", self._region_area_km2)

        def update_progress():
            if progress_callback:
                progress_callback(progress)

        # -- Phase 1: Overview --
        progress.phase = "generating_overview"
        progress.phase_detail = "Generating full-region overview..."
        progress.phase_progress = (0, 1)
        update_progress()

        t0 = time.time()
        overview = self._generate_with_retries(
            lambda cb: self.generate_overview(progress_callback=cb),
            max_retries=max_retries,
        )
        progress.tile_times.append(time.time() - t0)

        if overview is None:
            progress.failed_tiles += 1
            logger.error("Overview generation failed — cannot proceed with hierarchical mode")
            progress.phase = "generating_overview"
            progress.phase_detail = "Overview generation failed"
            update_progress()
            return [], progress

        progress.completed_tiles += 1
        progress.phase_progress = (1, 1)
        update_progress()

        # -- Phase 2: Medium tiles --
        l1_total = self.L1_COLS * self.L1_ROWS
        progress.phase = "generating_medium"
        progress.phase_detail = "Enhancing medium tiles..."
        progress.phase_progress = (0, l1_total)
        update_progress()

        l1_count = 0

        def l1_progress_cb(msg: str):
            nonlocal l1_count
            progress.phase_detail = msg

        medium_tiles = [[None] * self.L1_COLS for _ in range(self.L1_ROWS)]
        for row in range(self.L1_ROWS):
            for col in range(self.L1_COLS):
                t0 = time.time()

                # Generate a single medium tile via retry wrapper
                def gen_l1(cb, r=row, c=col):
                    region = self.project.generation_bbox
                    l1_grid = self._subdivide_bbox(region, self.L1_COLS, self.L1_ROWS)
                    tile_bbox = l1_grid[r][c]

                    cached = self._load_cached(1, c, r)
                    if cached is not None:
                        return cached

                    pos_desc = self._position_desc(c, r, self.L1_COLS, self.L1_ROWS)
                    illustration_crop = self._crop_illustration_for_tile(overview, region, tile_bbox)
                    reference = self._fetch_reference(tile_bbox, (TILE_SIZE, TILE_SIZE))

                    terrain_desc = self._get_terrain_description(tile_bbox)

                    if cb:
                        cb(f"Generating L1 tile ({c},{r})...")

                    result = self.gemini.generate_enhanced_tile(
                        illustration_crop=illustration_crop,
                        reference_image=reference,
                        level="medium",
                        terrain_description=terrain_desc,
                        tile_position=pos_desc,
                    )
                    self._save_cached(result.image, 1, c, r)
                    return result.image

                tile_img = self._generate_with_retries(gen_l1, max_retries=max_retries)
                progress.tile_times.append(time.time() - t0)

                medium_tiles[row][col] = tile_img
                l1_count += 1

                if tile_img is None:
                    progress.failed_tiles += 1
                else:
                    progress.completed_tiles += 1

                progress.phase_progress = (l1_count, l1_total)
                progress.phase_detail = f"Medium tiles ({l1_count}/{l1_total})"
                update_progress()

        if skip_l2:
            # Quick-test mode: assemble from L1 medium tiles directly
            fine_tiles: list[dict] = []
            region = self.project.generation_bbox
            l1_grid = self._subdivide_bbox(region, self.L1_COLS, self.L1_ROWS)
            for l1_row in range(self.L1_ROWS):
                for l1_col in range(self.L1_COLS):
                    fine_tiles.append({
                        'image': medium_tiles[l1_row][l1_col],
                        'l1_col': l1_col, 'l1_row': l1_row,
                        'l2_col': 0, 'l2_row': 0,
                        'bbox': l1_grid[l1_row][l1_col],
                    })

            # Color harmonization for L1 tiles against overview
            if self._color_consistency is not None and overview is not None:
                progress.phase = "color_harmonization"
                progress.phase_detail = "Harmonizing tile colors..."
                update_progress()
                for l1_row in range(self.L1_ROWS):
                    for l1_col in range(self.L1_COLS):
                        if medium_tiles[l1_row][l1_col] is not None:
                            medium_tiles[l1_row][l1_col] = self._color_consistency.histogram_match(
                                medium_tiles[l1_row][l1_col], overview
                            )
                            self._save_cached(medium_tiles[l1_row][l1_col], 1, l1_col, l1_row)

            progress.phase = "assembling"
            progress.phase_detail = "Assembling from L1 tiles (quick-test)..."
            progress.phase_progress = None
            update_progress()

            assembled = self._assemble_l1(medium_tiles, l1_grid)
        else:
            # -- Phase 3: Fine tiles --
            l2_total = self.L1_COLS * self.L1_ROWS * self.L2_COLS * self.L2_ROWS
            progress.phase = "generating_fine"
            progress.phase_detail = "Enhancing fine tiles..."
            progress.phase_progress = (0, l2_total)
            update_progress()

            region = self.project.generation_bbox
            l1_grid = self._subdivide_bbox(region, self.L1_COLS, self.L1_ROWS)
            fine_tiles = []
            l2_count = 0

            for l1_row in range(self.L1_ROWS):
                for l1_col in range(self.L1_COLS):
                    medium_img = medium_tiles[l1_row][l1_col]
                    l1_bbox = l1_grid[l1_row][l1_col]
                    l2_grid = self._subdivide_bbox(l1_bbox, self.L2_COLS, self.L2_ROWS)

                    for l2_row in range(self.L2_ROWS):
                        for l2_col in range(self.L2_COLS):
                            tile_bbox = l2_grid[l2_row][l2_col]
                            t0 = time.time()

                            def gen_l2(cb, mi=medium_img, lb=l1_bbox, tb=tile_bbox,
                                       lc=l1_col, lr=l1_row, c=l2_col, r=l2_row):
                                cached = self._load_cached(2, c, r, lc, lr)
                                if cached is not None:
                                    return cached

                                if mi is None:
                                    return None

                                illustration_crop = self._crop_illustration_for_tile(mi, lb, tb)

                                global_col = lc * self.L2_COLS + c
                                global_row = lr * self.L2_ROWS + r
                                total_cols = self.L1_COLS * self.L2_COLS
                                total_rows = self.L1_ROWS * self.L2_ROWS
                                pos_desc = self._position_desc(global_col, global_row, total_cols, total_rows)

                                reference = self._fetch_reference(tb, (TILE_SIZE, TILE_SIZE))

                                terrain_desc = self._get_terrain_description(tb)

                                if cb:
                                    cb(f"Generating L2 ({lc},{lr})->({c},{r})...")

                                result = self.gemini.generate_enhanced_tile(
                                    illustration_crop=illustration_crop,
                                    reference_image=reference,
                                    level="fine",
                                    terrain_description=terrain_desc,
                                    tile_position=pos_desc,
                                )
                                self._save_cached(result.image, 2, c, r, lc, lr)
                                return result.image

                            tile_img = self._generate_with_retries(gen_l2, max_retries=max_retries)
                            progress.tile_times.append(time.time() - t0)

                            fine_tiles.append({
                                'image': tile_img,
                                'l1_col': l1_col, 'l1_row': l1_row,
                                'l2_col': l2_col, 'l2_row': l2_row,
                                'bbox': tile_bbox,
                            })
                            l2_count += 1

                            if tile_img is None:
                                progress.failed_tiles += 1
                            else:
                                progress.completed_tiles += 1

                            progress.phase_progress = (l2_count, l2_total)
                            progress.phase_detail = f"Fine tiles ({l2_count}/{l2_total})"
                            update_progress()

            # -- Phase 3.5: Color harmonization --
            if self._color_consistency is not None and overview is not None:
                progress.phase = "color_harmonization"
                progress.phase_detail = "Harmonizing tile colors..."
                update_progress()
                for tile_info in fine_tiles:
                    if tile_info['image'] is not None:
                        tile_info['image'] = self._color_consistency.histogram_match(
                            tile_info['image'], overview
                        )
                        self._save_cached(
                            tile_info['image'], 2,
                            tile_info['l2_col'], tile_info['l2_row'],
                            tile_info['l1_col'], tile_info['l1_row'],
                        )

            # -- Phase 4: Assemble --
            progress.phase = "assembling"
            progress.phase_detail = "Assembling final image..."
            progress.phase_progress = None
            update_progress()

            assembled = self.assemble(fine_tiles, color_reference=overview)

        if assembled is not None and self.project.project_dir:
            output_dir = self.project.project_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "assembled.png"
            assembled.save(output_path)
            assembled.close()
            logger.info("Saved assembled image to %s", output_path)

            # Invalidate stale DZI tiles so the viewer shows the new assembly
            if self.cache_dir:
                import shutil
                dzi_dir = self.cache_dir / "dzi"
                if dzi_dir.exists():
                    shutil.rmtree(dzi_dir)
                    logger.info("Cleared stale DZI cache at %s", dzi_dir)

        progress.phase_detail = "Complete"
        update_progress()

        return fine_tiles, progress

    def _generate_with_retries(
        self,
        gen_fn: Callable,
        max_retries: int = 3,
    ) -> Optional[Image.Image]:
        """Call gen_fn with retry logic and exponential backoff.

        gen_fn takes a single progress_callback argument and returns an Image or None.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                result = gen_fn(None)
                return result
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("All %d attempts failed: %s", max_retries, last_error)
        return None

    def estimate_cost(self) -> dict:
        """Estimate generation costs for hierarchical mode."""
        total_calls = 1 + (self.L1_COLS * self.L1_ROWS) + (self.L1_COLS * self.L1_ROWS * self.L2_COLS * self.L2_ROWS)
        cost_per_image = 0.13
        return {
            "tile_cost": total_calls * cost_per_image,
            "landmark_cost": 0,
            "seam_cost": 0,  # Hierarchical mode should rarely need seam repair
            "total_cost": total_calls * cost_per_image,
            "overview_calls": 1,
            "medium_calls": self.L1_COLS * self.L1_ROWS,
            "fine_calls": self.L1_COLS * self.L1_ROWS * self.L2_COLS * self.L2_ROWS,
            "total_calls": total_calls,
        }
