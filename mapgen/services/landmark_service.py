"""Landmark illustration service."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ..models.landmark import Landmark
from ..models.project import Project
from ..utils.image_utils import crop_to_content, create_alpha_mask
from .gemini_service import GeminiService


@dataclass
class IllustrationResult:
    """Result of illustrating a single landmark."""

    landmark: Landmark
    image: Optional[Image.Image] = None
    generation_time: float = 0.0
    error: Optional[str] = None


class LandmarkService:
    """Service for illustrating landmarks and managing the landmark pipeline."""

    def __init__(
        self,
        project: Project,
        gemini_service: Optional[GeminiService] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize landmark service.

        Args:
            project: Project configuration
            gemini_service: Optional pre-configured Gemini service
            output_dir: Directory for saving illustrated landmarks
        """
        self.project = project
        self._gemini = gemini_service
        self.output_dir = output_dir or (project.output_dir / "landmarks" if project.project_dir else None)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def gemini(self) -> GeminiService:
        """Lazy initialization of Gemini service."""
        if self._gemini is None:
            self._gemini = GeminiService()
        return self._gemini

    def get_landmarks(self) -> list[Landmark]:
        """Get all landmarks from project configuration."""
        return self.project.landmarks

    def get_landmark_by_name(self, name: str) -> Optional[Landmark]:
        """Find a landmark by name (case-insensitive)."""
        name_lower = name.lower()
        for landmark in self.project.landmarks:
            if landmark.name.lower() == name_lower:
                return landmark
        return None

    def load_photo(self, landmark: Landmark) -> Optional[Image.Image]:
        """
        Load landmark photo from disk.

        Args:
            landmark: Landmark to load photo for

        Returns:
            PIL Image or None if no photo path set
        """
        if landmark.photo is None:
            return None

        photo_path = landmark.resolve_photo_path(self.project.project_dir)
        if photo_path is None or not photo_path.exists():
            return None

        return Image.open(photo_path).convert("RGBA")

    def load_logo(self, landmark: Landmark) -> Optional[Image.Image]:
        """
        Load landmark logo from disk.

        Args:
            landmark: Landmark to load logo for

        Returns:
            PIL Image or None if no logo path set
        """
        if landmark.logo is None:
            return None

        logo_path = landmark.resolve_logo_path(self.project.project_dir)
        if logo_path is None or not logo_path.exists():
            return None

        return Image.open(logo_path).convert("RGBA")

    def illustrate_landmark(
        self,
        landmark: Landmark,
        style_reference: Optional[Image.Image] = None,
        max_retries: int = 3,
    ) -> IllustrationResult:
        """
        Illustrate a single landmark using Gemini.

        Args:
            landmark: Landmark to illustrate
            style_reference: Optional reference image for style matching
            max_retries: Maximum retries on failure

        Returns:
            IllustrationResult with generated image
        """
        result = IllustrationResult(landmark=landmark)

        # Load photo
        photo = self.load_photo(landmark)
        if photo is None:
            result.error = f"No photo found for landmark '{landmark.name}'"
            return result

        # Try generation with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                gen_result = self.gemini.stylize_landmark(
                    photo=photo,
                    style_reference=style_reference,
                    landmark_name=landmark.name,
                )

                result.image = gen_result.image
                result.generation_time = time.time() - start_time

                # Post-process: crop to content
                if result.image is not None:
                    result.image = crop_to_content(result.image, padding=10)

                return result

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        result.error = f"Failed after {max_retries} attempts: {last_error}"
        return result

    def illustrate_all(
        self,
        landmarks: Optional[list[Landmark]] = None,
        style_reference: Optional[Image.Image] = None,
        progress_callback: Optional[Callable[[int, int, Landmark], None]] = None,
    ) -> list[IllustrationResult]:
        """
        Illustrate all landmarks.

        Args:
            landmarks: Landmarks to illustrate (defaults to all project landmarks)
            style_reference: Optional reference image for style matching
            progress_callback: Optional callback(current, total, landmark)

        Returns:
            List of IllustrationResult objects
        """
        if landmarks is None:
            landmarks = self.get_landmarks()

        results = []
        for i, landmark in enumerate(landmarks):
            if progress_callback:
                progress_callback(i, len(landmarks), landmark)

            result = self.illustrate_landmark(landmark, style_reference)
            results.append(result)

            # Save successful results
            if result.image is not None and self.output_dir:
                self.save_illustrated(landmark, result.image)

        return results

    def save_illustrated(self, landmark: Landmark, image: Image.Image) -> Path:
        """
        Save illustrated landmark to disk.

        Args:
            landmark: Landmark that was illustrated
            image: Generated illustration

        Returns:
            Path where image was saved
        """
        if self.output_dir is None:
            raise ValueError("No output directory configured")

        # Create safe filename from landmark name
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in landmark.name)
        safe_name = safe_name.replace(" ", "_").lower()

        output_path = self.output_dir / f"{safe_name}_illustrated.png"
        image.save(output_path)

        return output_path

    def load_illustrated(self, landmark: Landmark) -> Optional[Image.Image]:
        """
        Load previously illustrated landmark from disk.

        Args:
            landmark: Landmark to load illustration for

        Returns:
            PIL Image or None if not found
        """
        if self.output_dir is None:
            return None

        # Try to find by landmark name
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in landmark.name)
        safe_name = safe_name.replace(" ", "_").lower()

        illustrated_path = self.output_dir / f"{safe_name}_illustrated.png"
        if illustrated_path.exists():
            return Image.open(illustrated_path).convert("RGBA")

        # Also check illustrated_path field
        if landmark.illustrated_path:
            alt_path = self.project.project_dir / landmark.illustrated_path
            if alt_path.exists():
                return Image.open(alt_path).convert("RGBA")

        return None

    def get_style_reference(self) -> Optional[Image.Image]:
        """
        Get a style reference image from generated tiles.

        Uses the first generated tile as a style reference for consistent
        landmark illustration.

        Returns:
            PIL Image of first tile, or None if no tiles generated
        """
        from ..config import get_config

        config = get_config()
        tile_cache = config.cache_dir / "generation" / "generated"

        # Look for tile_0_0.png first
        ref_path = tile_cache / "tile_0_0.png"
        if ref_path.exists():
            return Image.open(ref_path).convert("RGBA")

        # Fall back to any tile
        for tile_path in tile_cache.glob("tile_*.png"):
            return Image.open(tile_path).convert("RGBA")

        return None

    def estimate_cost(self) -> dict:
        """
        Estimate cost for illustrating all landmarks.

        Returns:
            Cost estimate dictionary
        """
        landmarks = self.get_landmarks()
        landmarks_with_photos = [l for l in landmarks if l.photo is not None]

        cost_per_image = 0.13  # USD

        return {
            "total_landmarks": len(landmarks),
            "landmarks_with_photos": len(landmarks_with_photos),
            "estimated_cost": len(landmarks_with_photos) * cost_per_image,
        }
