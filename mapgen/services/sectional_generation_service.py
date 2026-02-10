"""Sectional generation service for large-region maps.

Orchestrates a two-tier generation approach:
- Focus regions (cities) get high-detail AI-illustrated generation
- Fill regions (desert corridors) get rendered base maps with optional AI polish
- All sections are composed into the final output with blended edges
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter

from ..models.project import (
    BoundingBox,
    DetailLevel,
    FillRegion,
    FocusRegion,
    Project,
    SectionalLayout,
)
from .blending_service import BlendingService
from .distortion_service import DistortionService, compute_control_points
from .generation_service import GenerationService
from .osm_service import OSMService
from .perspective_service import PerspectiveService
from .render_service import RenderService


@dataclass
class SectionResult:
    """Result of generating a single section."""

    name: str
    region_type: str  # "focus" or "fill"
    image: Optional[Image.Image] = None
    pixel_rect: Optional[tuple[int, int, int, int]] = None  # x, y, w, h in output
    generation_time: float = 0.0
    error: Optional[str] = None


@dataclass
class SectionalProgress:
    """Progress tracking for sectional generation."""

    total_sections: int
    completed_sections: int = 0
    failed_sections: int = 0
    current_section: Optional[str] = None
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class SectionalGenerationService:
    """Orchestrates two-tier sectional map generation."""

    # Blend margin in pixels for section edges
    BLEND_MARGIN = 500

    def __init__(
        self,
        project: Project,
        generation_service: Optional[GenerationService] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.project = project
        self.cache_dir = cache_dir or (
            project.project_dir / "cache" if project.project_dir else None
        )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._gen_service = generation_service
        self._distortion: Optional[DistortionService] = None

    @property
    def generation_service(self) -> GenerationService:
        if self._gen_service is None:
            self._gen_service = GenerationService(
                self.project, cache_dir=self.cache_dir
            )
        return self._gen_service

    @property
    def distortion(self) -> Optional[DistortionService]:
        """Get or create distortion service from project config."""
        if self._distortion is not None:
            return self._distortion

        layout = self.project.sectional_layout
        if layout is None:
            return None

        mapping = layout.coordinate_mapping
        if mapping is None:
            # Auto-compute from landmarks
            mapping = compute_control_points(
                self.project.region, self.project.landmarks
            )

        self._distortion = DistortionService(
            mapping=mapping,
            region=self.project.region,
            output_size=(self.project.output.width, self.project.output.height),
        )
        return self._distortion

    def generate_all_sections(
        self,
        progress_callback: Optional[Callable[[SectionalProgress], None]] = None,
        skip_existing: bool = True,
        region_filter: Optional[list[str]] = None,
        style_reference: Optional[Image.Image] = None,
    ) -> tuple[list[SectionResult], SectionalProgress]:
        """Generate all sections (focus + fill) and compose them.

        Args:
            progress_callback: Progress update callback.
            skip_existing: Skip sections with cached results.
            region_filter: Only generate these region names (None = all).
            style_reference: Optional style reference image.

        Returns:
            (list of SectionResults, final progress).
        """
        layout = self.project.sectional_layout
        if layout is None:
            raise ValueError("Project has no sectional_layout configured")

        all_sections = []
        for fr in layout.focus_regions:
            all_sections.append(("focus", fr.name, fr))
        for fr in layout.fill_regions:
            all_sections.append(("fill", fr.name, fr))

        if region_filter:
            all_sections = [
                s for s in all_sections if s[1] in region_filter
            ]

        progress = SectionalProgress(total_sections=len(all_sections))
        results: list[SectionResult] = []

        for region_type, name, region_config in all_sections:
            progress.current_section = name
            if progress_callback:
                progress_callback(progress)

            # Check cache
            if skip_existing:
                cached = self._load_cached_section(name)
                if cached is not None:
                    result = SectionResult(
                        name=name,
                        region_type=region_type,
                        image=cached,
                    )
                    results.append(result)
                    progress.completed_sections += 1
                    continue

            start = time.time()
            try:
                if region_type == "focus":
                    result = self._generate_focus_region(
                        region_config, style_reference=style_reference
                    )
                else:
                    result = self._generate_fill_region(region_config)

                result.generation_time = time.time() - start

                if result.image is not None:
                    self._save_cached_section(name, result.image)
                    progress.completed_sections += 1

                    # Use first successful focus region as style reference
                    if (
                        style_reference is None
                        and region_type == "focus"
                        and result.image is not None
                    ):
                        style_reference = result.image

                else:
                    progress.failed_sections += 1

            except Exception as e:
                result = SectionResult(
                    name=name,
                    region_type=region_type,
                    error=str(e),
                    generation_time=time.time() - start,
                )
                progress.failed_sections += 1

            results.append(result)

            if progress_callback:
                progress_callback(progress)

        progress.current_section = None
        return results, progress

    def compose_sections(
        self,
        results: list[SectionResult],
        apply_perspective: bool = True,
    ) -> Optional[Image.Image]:
        """Compose all section results into the final output image.

        Places each section at its distortion-mapped position and blends edges.

        Args:
            results: List of section generation results.
            apply_perspective: Whether to apply perspective transform.

        Returns:
            Final composed image, or None if too many failures.
        """
        successful = [r for r in results if r.image is not None]
        if not successful:
            return None

        output_w = self.project.output.width
        output_h = self.project.output.height
        canvas = Image.new("RGBA", (output_w, output_h), (0, 0, 0, 0))

        layout = self.project.sectional_layout
        if layout is None:
            return None

        distortion = self.distortion

        # Place each section
        for result in successful:
            region_config = self._find_region_config(result.name, layout)
            if region_config is None:
                continue

            # Determine pixel placement from distortion mapping
            if distortion is not None:
                bbox = region_config.bbox
                # Map the region's geographic corners to pixel space
                tl_x, tl_y = distortion.geo_to_pixel(bbox.north, bbox.west)
                br_x, br_y = distortion.geo_to_pixel(bbox.south, bbox.east)

                # Ensure correct ordering
                px = min(tl_x, br_x)
                py = min(tl_y, br_y)
                pw = abs(br_x - tl_x)
                ph = abs(br_y - tl_y)
            else:
                # Linear placement fallback
                region = self.project.region
                bbox = region_config.bbox
                px = int(
                    (bbox.west - region.west) / region.width_degrees * output_w
                )
                py = int(
                    (region.north - bbox.north) / region.height_degrees * output_h
                )
                pw = int(bbox.width_degrees / region.width_degrees * output_w)
                ph = int(bbox.height_degrees / region.height_degrees * output_h)

            # Resize section to fit its allocated pixel space
            section_img = result.image.resize(
                (max(1, pw), max(1, ph)), Image.Resampling.LANCZOS
            )

            result.pixel_rect = (px, py, pw, ph)

            # Blend onto canvas with gradient edges
            self._blend_section_onto_canvas(canvas, section_img, px, py)

        # Apply perspective
        if apply_perspective:
            perspective = PerspectiveService(
                angle=self.project.style.perspective_angle,
                convergence=0.7,
                vertical_scale=0.4,
                horizon_margin=0.15,
            )
            canvas = perspective.transform_image(canvas)

        return canvas

    def _generate_focus_region(
        self,
        config: FocusRegion,
        style_reference: Optional[Image.Image] = None,
    ) -> SectionResult:
        """Generate a high-detail focus region (city)."""
        # Determine output size for this section based on distortion
        distortion = self.distortion
        if distortion is not None:
            bbox = config.bbox
            tl_x, tl_y = distortion.geo_to_pixel(bbox.north, bbox.west)
            br_x, br_y = distortion.geo_to_pixel(bbox.south, bbox.east)
            out_w = max(512, abs(br_x - tl_x))
            out_h = max(512, abs(br_y - tl_y))
        else:
            out_w = 2048
            out_h = 2048

        # Cap at reasonable size for single Gemini call
        max_dim = 4096
        if out_w > max_dim or out_h > max_dim:
            scale = max_dim / max(out_w, out_h)
            out_w = int(out_w * scale)
            out_h = int(out_h * scale)

        output_size = (out_w, out_h)

        image = self.generation_service.generate_for_subregion(
            sub_bbox=config.bbox,
            output_size=output_size,
            detail_level=config.detail_level,
            style_prompt=config.prompt_override,
            style_reference=style_reference,
        )

        return SectionResult(
            name=config.name,
            region_type="focus",
            image=image,
            error=None if image else "Generation returned None",
        )

    def _generate_fill_region(self, config: FillRegion) -> SectionResult:
        """Generate a low-detail fill region (desert/corridor)."""
        distortion = self.distortion
        if distortion is not None:
            bbox = config.bbox
            tl_x, tl_y = distortion.geo_to_pixel(bbox.north, bbox.west)
            br_x, br_y = distortion.geo_to_pixel(bbox.south, bbox.east)
            out_w = max(256, abs(br_x - tl_x))
            out_h = max(256, abs(br_y - tl_y))
        else:
            out_w = 2048
            out_h = 2048

        output_size = (out_w, out_h)

        # For fill regions, use RenderService directly (no Gemini by default)
        try:
            osm_service = self.generation_service.osm
            osm_data = osm_service.fetch_region_data(
                config.bbox,
                detail_level=config.detail_level.value,
            )

            perspective = PerspectiveService(
                angle=self.project.style.perspective_angle,
                convergence=0.7,
                vertical_scale=0.4,
                horizon_margin=0.0,
            )
            render = RenderService(perspective_service=perspective)

            image = render.render_desert_fill(
                osm_data=osm_data,
                bbox=config.bbox,
                output_size=output_size,
                include_highways=config.include_highways,
                include_rivers=config.include_rivers,
                include_terrain_shading=config.include_terrain_shading,
            )

            # Optional AI illustration pass
            if config.use_ai_illustration and image is not None:
                prompt = config.prompt_override or (
                    "Transform this desert map into a hand illustrated style. "
                    "Warm sand tones, stylized highways, subtle terrain texture. "
                    "Hand-painted aesthetic matching a tourist map."
                )
                try:
                    gen_result = self.generation_service.gemini.generate_base_tile(
                        reference_image=image,
                        style_prompt=prompt,
                        tile_position="desert fill region",
                    )
                    image = gen_result.image
                except Exception:
                    pass  # Keep the rendered version if AI fails

            return SectionResult(
                name=config.name,
                region_type="fill",
                image=image,
            )

        except Exception as e:
            return SectionResult(
                name=config.name,
                region_type="fill",
                error=str(e),
            )

    def _blend_section_onto_canvas(
        self,
        canvas: Image.Image,
        section: Image.Image,
        x: int,
        y: int,
    ) -> None:
        """Paste a section onto the canvas with gradient-blended edges.

        Creates a feathered alpha mask that fades out at the edges for
        smooth blending between adjacent sections.
        """
        margin = min(self.BLEND_MARGIN, section.width // 4, section.height // 4)

        if section.mode != "RGBA":
            section = section.convert("RGBA")

        # Create gradient mask for feathered edges
        mask = Image.new("L", section.size, 255)
        mask_array = np.array(mask, dtype=np.float32)

        h, w = mask_array.shape

        # Fade left edge
        for i in range(min(margin, w)):
            mask_array[:, i] = min(mask_array[:, i], 255 * i / margin)
        # Fade right edge
        for i in range(min(margin, w)):
            mask_array[:, w - 1 - i] = min(mask_array[:, w - 1 - i], 255 * i / margin)
        # Fade top edge
        for i in range(min(margin, h)):
            mask_array[i, :] = np.minimum(mask_array[i, :], 255 * i / margin)
        # Fade bottom edge
        for i in range(min(margin, h)):
            mask_array[h - 1 - i, :] = np.minimum(mask_array[h - 1 - i, :], 255 * i / margin)

        blend_mask = Image.fromarray(mask_array.astype(np.uint8))

        # Apply the blend mask to section's alpha
        r, g, b, a = section.split()
        a = Image.fromarray(
            np.minimum(np.array(a), np.array(blend_mask)).astype(np.uint8)
        )
        section = Image.merge("RGBA", (r, g, b, a))

        # Paste onto canvas
        canvas.paste(section, (x, y), section)

    def _find_region_config(
        self, name: str, layout: SectionalLayout
    ) -> Optional[FocusRegion | FillRegion]:
        """Find a region config by name."""
        for fr in layout.focus_regions:
            if fr.name == name:
                return fr
        for fr in layout.fill_regions:
            if fr.name == name:
                return fr
        return None

    def _load_cached_section(self, name: str) -> Optional[Image.Image]:
        """Load a cached section image."""
        if self.cache_dir is None:
            return None
        safe_name = name.replace(" ", "_").lower()
        path = self.cache_dir / "sections" / f"{safe_name}.png"
        if path.exists():
            return Image.open(path).convert("RGBA")
        return None

    def _save_cached_section(self, name: str, image: Image.Image) -> None:
        """Save a section image to cache."""
        if self.cache_dir is None:
            return
        safe_name = name.replace(" ", "_").lower()
        path = self.cache_dir / "sections" / f"{safe_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
