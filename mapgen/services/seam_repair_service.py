"""Seam repair service for fixing tile discontinuities."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models.seam import SeamInfo
from .gemini_service import GeminiService


@dataclass
class RepairResult:
    """Result of a seam repair operation."""

    seam: SeamInfo
    repaired_region: Image.Image
    generation_time: float
    error: Optional[str] = None


class SeamRepairService:
    """Service for identifying and repairing seams between tiles.

    Works on the assembled image directly — extracts a strip around each
    cell boundary, sends it to Gemini for inpainting, and pastes it back.
    """

    def __init__(self, repair_width: int = 256):
        """
        Initialize seam repair service.

        Args:
            repair_width: Width of the strip to repair around each cell boundary (pixels)
        """
        self.repair_width = repair_width

    def identify_seams(
        self,
        cols: int,
        rows: int,
        output_width: int,
        output_height: int,
    ) -> list[SeamInfo]:
        """
        Identify all internal seams in a tile grid.

        Seam positions are computed from cell boundaries in the assembled image.

        Args:
            cols: Number of tile columns
            rows: Number of tile rows
            output_width: Width of the assembled image in pixels
            output_height: Height of the assembled image in pixels

        Returns:
            List of SeamInfo objects for all internal seams
        """
        seams = []
        cell_w = round(output_width / cols)
        cell_h = round(output_height / rows)
        half = self.repair_width // 2

        # Horizontal seams (between horizontally adjacent tiles)
        for row in range(rows):
            for col in range(cols - 1):
                x_boundary = (col + 1) * cell_w
                x = max(0, x_boundary - half)
                width = min(self.repair_width, output_width - x)
                y = row * cell_h
                height = min(cell_h, output_height - y)

                seams.append(SeamInfo(
                    orientation="horizontal",
                    tile_a=(col, row),
                    tile_b=(col + 1, row),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                ))

        # Vertical seams (between vertically adjacent tiles)
        for row in range(rows - 1):
            for col in range(cols):
                y_boundary = (row + 1) * cell_h
                y = max(0, y_boundary - half)
                height = min(self.repair_width, output_height - y)
                x = col * cell_w
                width = min(cell_w, output_width - x)

                seams.append(SeamInfo(
                    orientation="vertical",
                    tile_a=(col, row),
                    tile_b=(col, row + 1),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                ))

        return seams

    def find_seam(
        self,
        seams: list[SeamInfo],
        tile_a: tuple[int, int],
        tile_b: tuple[int, int],
    ) -> Optional[SeamInfo]:
        """
        Find a specific seam by tile coordinates.

        Args:
            seams: List of all seams
            tile_a: (col, row) of first tile
            tile_b: (col, row) of second tile

        Returns:
            SeamInfo if found, None otherwise
        """
        for seam in seams:
            if seam.tile_a == tile_a and seam.tile_b == tile_b:
                return seam
            if seam.tile_a == tile_b and seam.tile_b == tile_a:
                return seam
        return None

    def parse_seam_spec(self, spec: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Parse a seam specification string like '1,2-2,2'.

        Args:
            spec: Seam specification in format 'col,row-col,row'

        Returns:
            Tuple of ((col1, row1), (col2, row2))

        Raises:
            ValueError: If spec is invalid
        """
        try:
            parts = spec.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid seam spec: {spec}")

            a_parts = parts[0].split(",")
            b_parts = parts[1].split(",")

            if len(a_parts) != 2 or len(b_parts) != 2:
                raise ValueError(f"Invalid seam spec: {spec}")

            tile_a = (int(a_parts[0]), int(a_parts[1]))
            tile_b = (int(b_parts[0]), int(b_parts[1]))

            return tile_a, tile_b
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid seam spec '{spec}': expected format 'col,row-col,row'") from e

    def repair_seam(
        self,
        seam: SeamInfo,
        assembled: Image.Image,
        gemini: GeminiService,
    ) -> RepairResult:
        """
        Repair a single seam by extracting a strip from the assembled image.

        Args:
            seam: Seam information
            assembled: Full assembled image
            gemini: Gemini service for inpainting

        Returns:
            RepairResult with repaired region
        """
        try:
            # Crop the seam strip from assembled image
            seam_region = assembled.crop((
                seam.x, seam.y,
                seam.x + seam.width, seam.y + seam.height,
            ))

            result = gemini.inpaint_seam(seam_region, seam.orientation)

            return RepairResult(
                seam=seam,
                repaired_region=result.image,
                generation_time=result.generation_time,
            )

        except Exception as e:
            return RepairResult(
                seam=seam,
                repaired_region=Image.new("RGBA", (seam.width, seam.height)),
                generation_time=0.0,
                error=str(e),
            )

    def apply_repair(
        self,
        assembled: Image.Image,
        seam: SeamInfo,
        repaired_region: Image.Image,
        feather_size: int = 32,
    ) -> Image.Image:
        """
        Apply a repaired seam region back to the assembled image.

        Uses feathered blending to smoothly composite the repair.

        Args:
            assembled: Full assembled image
            seam: Seam that was repaired
            repaired_region: Repaired seam region from Gemini
            feather_size: Size of feathered edge for blending

        Returns:
            Updated assembled image with repair applied
        """
        # Resize repaired region to match seam dimensions if needed
        if repaired_region.size != (seam.width, seam.height):
            repaired_region = repaired_region.resize(
                (seam.width, seam.height),
                Image.Resampling.LANCZOS,
            )

        # Ensure RGBA mode
        if repaired_region.mode != "RGBA":
            repaired_region = repaired_region.convert("RGBA")

        # Create feathered mask for smooth blending
        mask = self._create_feather_mask(seam.width, seam.height, feather_size)

        # Convert to numpy for blending
        assembled_array = np.array(assembled).astype(np.float32)
        repair_array = np.array(repaired_region).astype(np.float32)

        # Extract region from assembled
        y1, y2 = seam.y, seam.y + seam.height
        x1, x2 = seam.x, seam.x + seam.width

        # Clamp to image bounds
        y2 = min(y2, assembled.height)
        x2 = min(x2, assembled.width)
        actual_height = y2 - y1
        actual_width = x2 - x1

        # Resize repair and mask if needed due to boundary
        if actual_width != seam.width or actual_height != seam.height:
            repair_array = repair_array[:actual_height, :actual_width]
            mask = mask[:actual_height, :actual_width]

        # Blend using mask
        for c in range(4):
            assembled_array[y1:y2, x1:x2, c] = (
                assembled_array[y1:y2, x1:x2, c] * (1 - mask) +
                repair_array[:, :, c] * mask
            )

        # Convert back to PIL
        result = Image.fromarray(assembled_array.astype(np.uint8), mode="RGBA")
        return result

    def _create_feather_mask(
        self,
        width: int,
        height: int,
        feather_size: int,
    ) -> np.ndarray:
        """Create a feathered mask for smooth blending."""
        mask = np.ones((height, width), dtype=np.float32)

        # Feather all edges
        if feather_size > 0:
            # Left edge
            for i in range(min(feather_size, width)):
                mask[:, i] *= i / feather_size
            # Right edge
            for i in range(min(feather_size, width)):
                mask[:, -(i + 1)] *= i / feather_size
            # Top edge
            for i in range(min(feather_size, height)):
                mask[i, :] *= i / feather_size
            # Bottom edge
            for i in range(min(feather_size, height)):
                mask[-(i + 1), :] *= i / feather_size

        return mask

    def repair_seams_batch(
        self,
        seams: list[SeamInfo],
        assembled: Image.Image,
        gemini: GeminiService,
        progress_callback: Optional[Callable[[int, int, SeamInfo], None]] = None,
    ) -> tuple[Image.Image, list[RepairResult]]:
        """
        Repair multiple seams in batch.

        Args:
            seams: List of seams to repair
            assembled: Current assembled image
            gemini: Gemini service for inpainting
            progress_callback: Optional callback(current, total, seam)

        Returns:
            Tuple of (updated image, list of repair results)
        """
        results = []
        current_image = assembled.copy()

        for i, seam in enumerate(seams):
            if progress_callback:
                progress_callback(i, len(seams), seam)

            result = self.repair_seam(seam, current_image, gemini)
            results.append(result)

            if result.error is None:
                current_image = self.apply_repair(
                    current_image,
                    seam,
                    result.repaired_region,
                )

        return current_image, results
