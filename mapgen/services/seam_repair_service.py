"""Seam repair service for fixing tile discontinuities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..models.seam import SeamInfo
from .blending_service import TileInfo
from .gemini_service import GeminiService


@dataclass
class RepairResult:
    """Result of a seam repair operation."""

    seam: SeamInfo
    repaired_region: Image.Image
    generation_time: float
    error: Optional[str] = None


class SeamRepairService:
    """Service for identifying and repairing seams between tiles."""

    def __init__(
        self,
        tile_size: int = 2048,
        overlap: int = 256,
        context_margin: int = 128,
    ):
        """
        Initialize seam repair service.

        Args:
            tile_size: Size of each tile in pixels
            overlap: Overlap between adjacent tiles in pixels
            context_margin: Extra margin around seam for context (pixels)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.context_margin = context_margin
        self.effective_size = tile_size - overlap

    def identify_seams(
        self,
        cols: int,
        rows: int,
    ) -> list[SeamInfo]:
        """
        Identify all internal seams in a tile grid.

        Args:
            cols: Number of tile columns
            rows: Number of tile rows

        Returns:
            List of SeamInfo objects for all internal seams
        """
        seams = []

        # Horizontal seams (between horizontally adjacent tiles)
        for row in range(rows):
            for col in range(cols - 1):
                # Seam is at the overlap between tile (col, row) and (col+1, row)
                x = (col + 1) * self.effective_size - self.overlap // 2
                y = row * self.effective_size
                width = self.overlap
                height = self.tile_size

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
                # Seam is at the overlap between tile (col, row) and (col, row+1)
                x = col * self.effective_size
                y = (row + 1) * self.effective_size - self.overlap // 2
                width = self.tile_size
                height = self.overlap

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
            # Also check reverse order
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

    def load_tile(
        self,
        cache_dir: Path,
        col: int,
        row: int,
    ) -> Optional[Image.Image]:
        """
        Load a cached tile image.

        Args:
            cache_dir: Directory containing cached tiles
            col: Tile column
            row: Tile row

        Returns:
            PIL Image if found, None otherwise
        """
        tile_path = cache_dir / "generated" / f"tile_{col}_{row}.png"
        if tile_path.exists():
            return Image.open(tile_path).convert("RGBA")
        return None

    def repair_seam(
        self,
        seam: SeamInfo,
        tile_a: Image.Image,
        tile_b: Image.Image,
        gemini: GeminiService,
    ) -> RepairResult:
        """
        Repair a single seam using Gemini inpainting.

        Args:
            seam: Seam information
            tile_a: First tile image
            tile_b: Second tile image
            gemini: Gemini service for inpainting

        Returns:
            RepairResult with repaired region
        """
        try:
            # Use Gemini's inpaint_seam method
            result = gemini.inpaint_seam(
                tile_a=tile_a,
                tile_b=tile_b,
                overlap_region=(0, 0, seam.width, seam.height),
                orientation=seam.orientation,
            )

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
        cache_dir: Path,
        assembled: Image.Image,
        gemini: GeminiService,
        progress_callback: Optional[Callable[[int, int, SeamInfo], None]] = None,
    ) -> tuple[Image.Image, list[RepairResult]]:
        """
        Repair multiple seams in batch.

        Args:
            seams: List of seams to repair
            cache_dir: Directory containing cached tiles
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

            # Load tiles
            tile_a = self.load_tile(cache_dir, seam.tile_a[0], seam.tile_a[1])
            tile_b = self.load_tile(cache_dir, seam.tile_b[0], seam.tile_b[1])

            if tile_a is None or tile_b is None:
                results.append(RepairResult(
                    seam=seam,
                    repaired_region=Image.new("RGBA", (seam.width, seam.height)),
                    generation_time=0.0,
                    error=f"Missing tiles for seam {seam.id}",
                ))
                continue

            # Repair the seam
            result = self.repair_seam(seam, tile_a, tile_b, gemini)
            results.append(result)

            # Apply repair if successful
            if result.error is None:
                current_image = self.apply_repair(
                    current_image,
                    seam,
                    result.repaired_region,
                )

        return current_image, results
