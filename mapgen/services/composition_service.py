"""Composition service for assembling landmarks and labels onto the map."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..models.landmark import Landmark
from ..models.project import BoundingBox
from ..utils.geo_utils import gps_to_pixel
from ..utils.image_utils import apply_drop_shadow, blend_images


@dataclass
class PlacedLandmark:
    """A landmark that has been positioned on the map."""

    landmark: Landmark
    illustration: Image.Image
    position: tuple[int, int]  # (x, y) pixel position
    scaled_size: tuple[int, int]  # Actual size after scaling
    logo: Optional[Image.Image] = None
    logo_position: Optional[tuple[int, int]] = None


class CompositionService:
    """Service for compositing landmarks and labels onto the base map."""

    def __init__(self):
        """Initialize composition service."""
        self.placed_landmarks: list[PlacedLandmark] = []

    def place_landmark(
        self,
        landmark: Landmark,
        illustration: Image.Image,
        base_map_size: tuple[int, int],
        bbox: BoundingBox,
        isometric_matrix: Optional[np.ndarray] = None,
    ) -> PlacedLandmark:
        """
        Calculate placement position for a landmark.

        Args:
            landmark: Landmark model
            illustration: Illustrated landmark image
            base_map_size: (width, height) of base map
            bbox: Map bounding box
            isometric_matrix: Optional transformation matrix

        Returns:
            PlacedLandmark with calculated position
        """
        # Convert GPS to pixel coordinates
        pixel_x, pixel_y = gps_to_pixel(
            landmark.latitude,
            landmark.longitude,
            bbox,
            base_map_size,
            isometric_matrix,
        )

        # Apply scale factor
        scale = landmark.scale
        new_width = int(illustration.width * scale)
        new_height = int(illustration.height * scale)

        if scale != 1.0:
            illustration = illustration.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

        # Center the landmark on its position
        position = (
            pixel_x - new_width // 2,
            pixel_y - new_height,  # Bottom of landmark at GPS point
        )

        placed = PlacedLandmark(
            landmark=landmark,
            illustration=illustration,
            position=position,
            scaled_size=(new_width, new_height),
        )

        self.placed_landmarks.append(placed)
        return placed

    def place_logo(
        self,
        placed_landmark: PlacedLandmark,
        logo: Image.Image,
        offset_y: int = 10,
        max_width: Optional[int] = None,
    ) -> PlacedLandmark:
        """
        Position a logo label below a landmark.

        Args:
            placed_landmark: Already placed landmark
            logo: Logo image
            offset_y: Vertical offset below landmark
            max_width: Maximum logo width (will scale down if needed)

        Returns:
            Updated PlacedLandmark with logo position
        """
        # Scale logo if needed
        if max_width and logo.width > max_width:
            ratio = max_width / logo.width
            new_size = (max_width, int(logo.height * ratio))
            logo = logo.resize(new_size, Image.Resampling.LANCZOS)

        # Position logo centered below landmark
        landmark_center_x = placed_landmark.position[0] + placed_landmark.scaled_size[0] // 2
        landmark_bottom_y = placed_landmark.position[1] + placed_landmark.scaled_size[1]

        logo_x = landmark_center_x - logo.width // 2
        logo_y = landmark_bottom_y + offset_y

        placed_landmark.logo = logo
        placed_landmark.logo_position = (logo_x, logo_y)

        return placed_landmark

    def check_collision(
        self,
        new_rect: tuple[int, int, int, int],
        existing_rects: list[tuple[int, int, int, int]],
        padding: int = 5,
    ) -> bool:
        """
        Check if a new rectangle collides with existing ones.

        Args:
            new_rect: (x, y, width, height) of new element
            existing_rects: List of existing rectangles
            padding: Additional padding around rectangles

        Returns:
            True if collision detected
        """
        x1, y1, w1, h1 = new_rect

        for x2, y2, w2, h2 in existing_rects:
            # Check overlap with padding
            if (
                x1 - padding < x2 + w2 + padding
                and x1 + w1 + padding > x2 - padding
                and y1 - padding < y2 + h2 + padding
                and y1 + h1 + padding > y2 - padding
            ):
                return True

        return False

    def avoid_collisions(
        self,
        placed_landmarks: list[PlacedLandmark],
        map_size: tuple[int, int],
    ) -> list[PlacedLandmark]:
        """
        Adjust logo positions to avoid collisions.

        Args:
            placed_landmarks: List of placed landmarks with logos
            map_size: (width, height) of map

        Returns:
            Landmarks with adjusted positions
        """
        occupied_rects = []
        width, height = map_size

        for placed in placed_landmarks:
            # Add landmark rect to occupied
            lm_rect = (
                placed.position[0],
                placed.position[1],
                placed.scaled_size[0],
                placed.scaled_size[1],
            )
            occupied_rects.append(lm_rect)

            if placed.logo and placed.logo_position:
                logo_rect = (
                    placed.logo_position[0],
                    placed.logo_position[1],
                    placed.logo.width,
                    placed.logo.height,
                )

                # Try different positions if collision
                if self.check_collision(logo_rect, occupied_rects):
                    # Try positions: below, above, left, right
                    offsets = [
                        (0, 20),  # Further below
                        (0, -placed.scaled_size[1] - placed.logo.height - 10),  # Above
                        (-placed.scaled_size[0] // 2 - placed.logo.width, 0),  # Left
                        (placed.scaled_size[0] // 2 + 10, 0),  # Right
                    ]

                    for dx, dy in offsets:
                        new_x = placed.logo_position[0] + dx
                        new_y = placed.logo_position[1] + dy

                        # Check bounds
                        if new_x < 0 or new_x + placed.logo.width > width:
                            continue
                        if new_y < 0 or new_y + placed.logo.height > height:
                            continue

                        new_rect = (new_x, new_y, placed.logo.width, placed.logo.height)
                        if not self.check_collision(new_rect, occupied_rects):
                            placed.logo_position = (new_x, new_y)
                            break

                # Add final logo rect to occupied
                occupied_rects.append(
                    (
                        placed.logo_position[0],
                        placed.logo_position[1],
                        placed.logo.width,
                        placed.logo.height,
                    )
                )

        return placed_landmarks

    def composite_map(
        self,
        base_map: Image.Image,
        placed_landmarks: list[PlacedLandmark],
        add_shadows: bool = True,
        shadow_offset: tuple[int, int] = (5, 5),
        shadow_blur: int = 10,
    ) -> Image.Image:
        """
        Composite all landmarks and logos onto the base map.

        Args:
            base_map: Base map image
            placed_landmarks: List of placed landmarks
            add_shadows: Whether to add drop shadows
            shadow_offset: Shadow offset (x, y)
            shadow_blur: Shadow blur radius

        Returns:
            Composited map image
        """
        result = base_map.copy()

        # Sort by z_index and y-position (lower y = further back)
        sorted_landmarks = sorted(
            placed_landmarks,
            key=lambda p: (p.landmark.z_index, p.position[1]),
        )

        for placed in sorted_landmarks:
            illustration = placed.illustration

            # Add shadow if requested
            if add_shadows:
                illustration = apply_drop_shadow(
                    illustration,
                    offset=shadow_offset,
                    blur_radius=shadow_blur,
                )
                # Adjust position for shadow padding
                pos = (
                    placed.position[0] - shadow_blur,
                    placed.position[1] - shadow_blur,
                )
            else:
                pos = placed.position

            # Composite landmark
            result = blend_images(result, illustration, pos)

            # Composite logo if present
            if placed.logo and placed.logo_position:
                # Add subtle shadow to logo
                if add_shadows:
                    logo = apply_drop_shadow(
                        placed.logo,
                        offset=(2, 2),
                        blur_radius=5,
                        shadow_color=(0, 0, 0, 100),
                    )
                    logo_pos = (
                        placed.logo_position[0] - 5,
                        placed.logo_position[1] - 5,
                    )
                else:
                    logo = placed.logo
                    logo_pos = placed.logo_position

                result = blend_images(result, logo, logo_pos)

        return result

    def create_layer_stack(
        self,
        base_map: Image.Image,
        placed_landmarks: list[PlacedLandmark],
    ) -> dict[str, tuple[Image.Image, tuple[int, int]]]:
        """
        Create a stack of layers for PSD export.

        Args:
            base_map: Base map image
            placed_landmarks: List of placed landmarks

        Returns:
            Dictionary mapping layer names to (image, position) tuples
        """
        layers = {}

        # Base map layer
        layers["Base Map"] = (base_map, (0, 0))

        # Individual landmark layers
        for i, placed in enumerate(placed_landmarks, 1):
            name = placed.landmark.name or f"Landmark {i}"

            # Landmark illustration
            layers[f"{name} - Building"] = (placed.illustration, placed.position)

            # Logo layer
            if placed.logo and placed.logo_position:
                layers[f"{name} - Label"] = (placed.logo, placed.logo_position)

        return layers
