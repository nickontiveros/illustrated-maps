"""Upscale generation service: single overview + Real-ESRGAN super-resolution.

Generates a single overview illustration via Gemini, then upscales it to
the target output dimensions using Real-ESRGAN (4x native) followed by
Lanczos resize if needed. This produces perfectly consistent output since
the entire map comes from a single generation call.
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models.project import BoundingBox, DetailLevel, Project
from .gemini_service import GeminiService
from .osm_service import OSMData, OSMService
from .perspective_service import PerspectiveService
from .render_service import RenderService
from .satellite_service import SatelliteService
from .terrain_service import TerrainService

logger = logging.getLogger(__name__)

# Target overview size (max dimension fed to Gemini)
OVERVIEW_MAX_DIM = 2048


@dataclass
class UpscaleProgress:
    """Progress tracking for upscale generation."""

    total_tiles: int = 1  # Always 1 Gemini call
    completed_tiles: int = 0
    failed_tiles: int = 0
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


class UpscaleGenerationService:
    """Single overview + Real-ESRGAN upscale generation.

    Produces a consistent map by generating one overview illustration
    and upscaling it to the target output dimensions.
    """

    def __init__(
        self,
        project: Project,
        cache_dir: Optional[Path] = None,
        gemini_service: Optional[GeminiService] = None,
        satellite_service: Optional[SatelliteService] = None,
        osm_service: Optional[OSMService] = None,
    ):
        self.project = project
        self.cache_dir = cache_dir or (project.project_dir / "cache" if project.project_dir else None)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._detail_level = project.region.get_recommended_detail_level()
        self._gemini = gemini_service
        self._satellite = satellite_service
        self._osm = osm_service
        self._terrain: Optional[TerrainService] = None
        self._osm_data: Optional[OSMData] = None
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

    # -- Reference image helpers (reused from HierarchicalGenerationService) --

    def _ensure_osm_data(self) -> OSMData:
        if self._osm_data is None:
            region = self.project.generation_bbox
            logger.info("Fetching OSM data for full region...")
            self._osm_data = self.osm.fetch_region_data(
                region, detail_level=self._detail_level.value,
            )
        return self._osm_data

    def _fetch_reference(
        self,
        bbox: BoundingBox,
        output_size: tuple[int, int],
    ) -> Image.Image:
        satellite_image = self.satellite.fetch_satellite_imagery(
            bbox=bbox,
            output_size=output_size,
        )

        # Skip OSM overlay for country-scale regions
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

    def _get_terrain_description(self) -> Optional[str]:
        try:
            region = self.project.generation_bbox
            elev_data = self.terrain.fetch_elevation_data(region)
            return self.terrain.get_terrain_description(elev_data)
        except Exception as e:
            logger.debug("Terrain fetch skipped: %s", e)
            return None

    # -- Real-ESRGAN upscaling (reused from OutpaintingService) --

    @staticmethod
    def _find_model_weights() -> str:
        """Find Real-ESRGAN model weights."""
        possible_paths = [
            Path("weights/RealESRGAN_x4plus.pth"),
            Path.home() / ".cache/realesrgan/RealESRGAN_x4plus.pth",
            Path("/tmp/RealESRGAN_x4plus.pth"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Download if not found
        logger.info("Downloading Real-ESRGAN weights...")
        import urllib.request

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        weights_dir = Path.home() / ".cache/realesrgan"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "RealESRGAN_x4plus.pth"
        urllib.request.urlretrieve(url, weights_path)
        return str(weights_path)

    @staticmethod
    def _init_upsampler():
        """Initialize Real-ESRGAN upsampler."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            weights_path = UpscaleGenerationService._find_model_weights()

            upsampler = RealESRGANer(
                scale=4,
                model_path=weights_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            return upsampler
        except ImportError:
            logger.warning("Real-ESRGAN not installed. Using Lanczos upscaling.")
            return None

    @staticmethod
    def _upscale(
        image: Image.Image,
        target_size: tuple[int, int],
        upsampler,
    ) -> Image.Image:
        """Upscale image using Real-ESRGAN, then resize to exact target.

        Args:
            image: Image to upscale.
            target_size: Target (width, height).
            upsampler: RealESRGANer instance or None for Lanczos fallback.

        Returns:
            Upscaled image at target_size.
        """
        if upsampler is None:
            return image.resize(target_size, Image.Resampling.LANCZOS).convert("RGBA")

        # Convert PIL to numpy BGR (Real-ESRGAN expects BGR)
        img_array = np.array(image.convert("RGB"))
        img_bgr = img_array[:, :, ::-1]

        try:
            output, _ = upsampler.enhance(img_bgr, outscale=4)
            output_rgb = output[:, :, ::-1]
            result = Image.fromarray(output_rgb)
        except Exception as e:
            logger.warning("Real-ESRGAN failed: %s. Falling back to Lanczos.", e)
            result = image.resize(target_size, Image.Resampling.LANCZOS)

        # Ensure exact target size
        if result.size != target_size:
            result = result.resize(target_size, Image.Resampling.LANCZOS)

        return result.convert("RGBA")

    # -- Main generation --

    def generate(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> tuple[list, "UpscaleProgress"]:
        """Run the upscale generation pipeline: 1 Gemini call + 1 upscale.

        Args:
            progress_callback: Callback receiving UpscaleProgress updates.

        Returns:
            Tuple of (empty tile list, progress) for API compatibility.
        """
        progress = UpscaleProgress()

        def update_progress():
            if progress_callback:
                progress_callback(progress)

        region = self.project.generation_bbox
        output = self.project.output

        # -- Phase 1: Generate overview --
        progress.phase = "generating_overview"
        progress.phase_detail = "Fetching reference imagery..."
        progress.phase_progress = (0, 1)
        update_progress()

        aspect = region.geographic_aspect_ratio
        if aspect >= 1:
            ov_w = OVERVIEW_MAX_DIM
            ov_h = round(OVERVIEW_MAX_DIM / aspect)
        else:
            ov_h = OVERVIEW_MAX_DIM
            ov_w = round(OVERVIEW_MAX_DIM * aspect)

        reference = self._fetch_reference(region, (ov_w, ov_h))
        terrain_desc = self._get_terrain_description()

        progress.phase_detail = "Calling Gemini for overview..."
        update_progress()

        t0 = time.time()
        try:
            overview_result = self.gemini.generate_overview(
                reference_image=reference,
                terrain_description=terrain_desc,
            )
            overview = overview_result.image
        except Exception as e:
            logger.error("Overview generation failed: %s", e)
            progress.failed_tiles += 1
            progress.phase_detail = f"Overview failed: {e}"
            update_progress()
            return [], progress

        progress.tile_times.append(time.time() - t0)
        progress.completed_tiles += 1
        progress.phase_progress = (1, 1)
        update_progress()

        # Save overview to cache
        if self.cache_dir:
            overview_dir = self.cache_dir / "hierarchical"
            overview_dir.mkdir(parents=True, exist_ok=True)
            overview.save(overview_dir / "overview.png")

        # -- Phase 2: Upscale with Real-ESRGAN --
        progress.phase = "upscaling"
        progress.phase_detail = "Initializing Real-ESRGAN..."
        update_progress()

        upsampler = self._init_upsampler()

        target_w, target_h = output.width, output.height
        # Account for rotation: use canvas_size which is expanded for rotation
        canvas_w, canvas_h = self.project.canvas_size

        progress.phase_detail = f"Upscaling to {canvas_w}x{canvas_h}..."
        update_progress()

        upscaled = self._upscale(overview, (canvas_w, canvas_h), upsampler)

        # -- Phase 3: Apply rotation if needed --
        rotation = self._rotation_degrees
        if rotation != 0 and rotation % 360 != 0:
            progress.phase_detail = "Applying rotation..."
            update_progress()

            upscaled = upscaled.rotate(
                -rotation,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(0, 0, 0, 0),
            )
            cx, cy = upscaled.width // 2, upscaled.height // 2
            left = cx - target_w // 2
            top = cy - target_h // 2
            upscaled = upscaled.crop((left, top, left + target_w, top + target_h))
        elif (canvas_w, canvas_h) != (target_w, target_h):
            upscaled = upscaled.resize(
                (target_w, target_h),
                Image.Resampling.LANCZOS,
            )

        # -- Phase 4: Save output --
        progress.phase = "assembling"
        progress.phase_detail = "Saving final image..."
        update_progress()

        if self.project.project_dir:
            output_dir = self.project.project_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "assembled.png"
            upscaled.save(output_path)
            upscaled.close()
            logger.info("Saved upscaled image to %s", output_path)

            # Invalidate stale DZI tiles
            if self.cache_dir:
                dzi_dir = self.cache_dir / "dzi"
                if dzi_dir.exists():
                    shutil.rmtree(dzi_dir)
                    logger.info("Cleared stale DZI cache at %s", dzi_dir)

        progress.phase_detail = "Complete"
        update_progress()

        return [], progress
