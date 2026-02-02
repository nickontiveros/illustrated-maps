"""Composition service for assembling landmarks and labels onto the map."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..models.landmark import Landmark
from ..models.project import BoundingBox
from ..utils.geo_utils import gps_to_pixel
from ..utils.image_utils import apply_drop_shadow, blend_images, create_alpha_mask


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

    def __init__(
        self,
        perspective_convergence: float = 0.7,
        perspective_vertical_scale: float = 0.4,
        perspective_horizon_margin: float = 0.15,
    ):
        """
        Initialize composition service.

        Args:
            perspective_convergence: How much the top narrows (0.0-1.0)
            perspective_vertical_scale: Vertical compression at top (0.0-1.0)
            perspective_horizon_margin: Extra space at top for horizon
        """
        self.placed_landmarks: list[PlacedLandmark] = []
        self.convergence = perspective_convergence
        self.vertical_scale = perspective_vertical_scale
        self.horizon_margin = perspective_horizon_margin

    def place_landmark(
        self,
        landmark: Landmark,
        illustration: Image.Image,
        base_map_size: tuple[int, int],
        bbox: BoundingBox,
        isometric_matrix: Optional[np.ndarray] = None,
        apply_perspective: bool = True,
        remove_background: bool = True,
    ) -> PlacedLandmark:
        """
        Calculate placement position for a landmark.

        Args:
            landmark: Landmark model
            illustration: Illustrated landmark image
            base_map_size: (width, height) of base map
            bbox: Map bounding box
            isometric_matrix: Optional transformation matrix
            apply_perspective: Whether to transform coords for perspective map
            remove_background: Whether to remove white/light backgrounds

        Returns:
            PlacedLandmark with calculated position
        """
        width, height = base_map_size

        # Remove background if requested (fix transparency issues)
        if remove_background:
            illustration = self._remove_background(illustration)

        # First, get flat GPS to pixel coordinates (before perspective)
        # We need to work with the original flat map size
        if apply_perspective:
            # Calculate original flat map height (before horizon margin was added)
            flat_height = int(height / (1 + self.horizon_margin))
            flat_size = (width, flat_height)
        else:
            flat_size = base_map_size

        # Convert GPS to flat pixel coordinates
        flat_x, flat_y = gps_to_pixel(
            landmark.latitude,
            landmark.longitude,
            bbox,
            flat_size,
            isometric_matrix,
        )

        # Now apply perspective transformation if the map has perspective
        if apply_perspective:
            pixel_x, pixel_y = self._apply_perspective_to_coords(
                flat_x, flat_y, flat_size
            )
            # Calculate perspective scale factor (smaller at top, larger at bottom)
            perspective_scale = self._get_perspective_scale(flat_y, flat_size[1])
        else:
            pixel_x, pixel_y = flat_x, flat_y
            perspective_scale = 1.0

        # Apply user scale factor AND perspective scale
        total_scale = landmark.scale * perspective_scale
        new_width = int(illustration.width * total_scale)
        new_height = int(illustration.height * total_scale)

        if total_scale != 1.0:
            illustration = illustration.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

        # Center the landmark horizontally, anchor bottom at GPS point
        position = (
            int(pixel_x - new_width // 2),
            int(pixel_y - new_height),  # Bottom of landmark at GPS point
        )

        placed = PlacedLandmark(
            landmark=landmark,
            illustration=illustration,
            position=position,
            scaled_size=(new_width, new_height),
        )

        self.placed_landmarks.append(placed)
        return placed

    def _remove_background(
        self,
        image: Image.Image,
        threshold: int = 240,
        feather: int = 2,
    ) -> Image.Image:
        """
        Remove white/light backgrounds from an image.

        Also detects and removes checkerboard transparency patterns that
        some AI generators produce instead of true transparency.

        Args:
            image: Input image
            threshold: Brightness threshold for background detection
            feather: Feather radius for smooth edges

        Returns:
            Image with transparent background
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Convert to numpy for processing
        data = np.array(image)
        rgb = data[:, :, :3].astype(float)

        # Detect checkerboard pattern (alternating grey ~204 and white ~255)
        # This happens when AI generators render "transparency" visually
        has_checkerboard = self._detect_checkerboard(rgb)

        if has_checkerboard:
            # For checkerboard, remove both the grey (~190-210) and white (>240) pixels
            grey_mask = (
                (rgb[:, :, 0] > 185) & (rgb[:, :, 0] < 215) &
                (rgb[:, :, 1] > 185) & (rgb[:, :, 1] < 215) &
                (rgb[:, :, 2] > 185) & (rgb[:, :, 2] < 215)
            )
            white_mask = rgb.mean(axis=2) > threshold
            background_mask = grey_mask | white_mask
        else:
            # Standard white background removal
            brightness = rgb.mean(axis=2)
            background_mask = brightness > threshold

        # Create mask: 255 for foreground, 0 for background
        mask = np.where(background_mask, 0, 255).astype(np.uint8)

        # Also preserve existing alpha
        existing_alpha = data[:, :, 3]
        mask = np.minimum(mask, existing_alpha)

        # Apply feathering for smooth edges
        mask_img = Image.fromarray(mask)
        if feather > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

        # Apply the mask
        result = image.copy()
        result.putalpha(mask_img)

        return result

    def _detect_checkerboard(
        self,
        rgb: np.ndarray,
        grey_threshold: float = 0.02,
    ) -> bool:
        """
        Detect if an image has a checkerboard transparency pattern.

        Checkerboard patterns have alternating grey (~204,204,204) and
        white (~255,255,255) pixels in a grid pattern.

        Args:
            rgb: RGB array of the image
            grey_threshold: Minimum fraction of grey pixels to consider checkerboard

        Returns:
            True if checkerboard pattern detected
        """
        # Count grey pixels (~190-210 range, typical checkerboard grey)
        grey_mask = (
            (rgb[:, :, 0] > 185) & (rgb[:, :, 0] < 215) &
            (rgb[:, :, 1] > 185) & (rgb[:, :, 1] < 215) &
            (rgb[:, :, 2] > 185) & (rgb[:, :, 2] < 215)
        )

        grey_fraction = grey_mask.sum() / (rgb.shape[0] * rgb.shape[1])

        # If more than 2% of pixels are in the grey range, likely checkerboard
        # (Normal images rarely have this specific grey in large amounts)
        return grey_fraction > grey_threshold

    def _apply_perspective_to_coords(
        self,
        x: float,
        y: float,
        flat_size: tuple[int, int],
    ) -> tuple[float, float]:
        """
        Transform coordinates from flat map space to perspective map space.

        Args:
            x: X coordinate in flat map
            y: Y coordinate in flat map
            flat_size: (width, height) of flat map

        Returns:
            (x, y) in perspective map coordinates
        """
        width, height = flat_size

        # Calculate output dimensions (with horizon margin)
        margin_pixels = int(height * self.horizon_margin)

        # Calculate perspective parameters
        top_inset = (width * (1 - self.convergence)) / 2

        # Normalize y position (0 = top, 1 = bottom)
        y_norm = y / height

        # Interpolate x inset based on y position
        # At y=0 (top), full inset; at y=height (bottom), no inset
        x_inset = top_inset * (1 - y_norm)

        # Calculate new x position
        scale_at_y = 1 - (1 - self.convergence) * (1 - y_norm)
        new_x = x_inset + x * scale_at_y

        # Calculate new y position (compressed at top)
        y_scale = self.vertical_scale + (1 - self.vertical_scale) * y_norm
        new_y = margin_pixels + y * y_scale

        return (new_x, new_y)

    def _get_perspective_scale(self, flat_y: float, flat_height: int) -> float:
        """
        Calculate scale factor based on position in perspective.

        Objects at the top (far away) should be smaller.
        Objects at the bottom (close) should be larger.

        Args:
            flat_y: Y position in flat map coordinates
            flat_height: Height of flat map

        Returns:
            Scale factor (0.0-1.0 range, typically convergence to 1.0)
        """
        # Normalize y: 0 = top (far), 1 = bottom (near)
        y_norm = flat_y / flat_height

        # Scale ranges from convergence (at top) to 1.0 (at bottom)
        # This matches how the map itself is scaled
        scale = self.convergence + (1.0 - self.convergence) * y_norm

        return scale

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
