"""Collision detection and resolution for map elements."""

import math
from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class MapElement:
    """A rectangular element on the map with position and size."""

    x: float  # Center x
    y: float  # Center y
    width: float
    height: float
    importance: float = 1.0  # Higher = less likely to be moved
    label: str = ""

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2

    def overlaps(self, other: "MapElement", padding: float = 5.0) -> bool:
        """Check if this element overlaps another with padding.

        Two axis-aligned rectangles overlap when they overlap on both axes.

        Args:
            other: The other element to test against.
            padding: Extra spacing (in pixels) added around each element.

        Returns:
            True if the elements overlap (including padding).
        """
        return (
            self.left - padding < other.right + padding
            and self.right + padding > other.left - padding
            and self.top - padding < other.bottom + padding
            and self.bottom + padding > other.top - padding
        )

    def overlap_area(self, other: "MapElement") -> float:
        """Calculate overlap area between two elements.

        Returns:
            The area (in square pixels) of the intersection rectangle,
            or 0.0 if there is no overlap.
        """
        x_overlap = max(
            0.0, min(self.right, other.right) - max(self.left, other.left)
        )
        y_overlap = max(
            0.0, min(self.bottom, other.bottom) - max(self.top, other.top)
        )
        return x_overlap * y_overlap


class CollisionService:
    """Service for detecting and resolving collisions between map elements.

    Uses a simple force-based approach: elements that overlap are pushed
    apart, with the less-important element doing the moving.
    """

    def __init__(self, map_size: tuple[int, int], padding: float = 10.0):
        """Initialize collision service.

        Args:
            map_size: (width, height) of the map canvas in pixels.
            padding: Minimum spacing between elements in pixels.
        """
        self.map_width, self.map_height = map_size
        self.padding = padding

    def detect_collisions(
        self, elements: list[MapElement]
    ) -> list[tuple[int, int]]:
        """Detect all pairs of overlapping elements.

        Uses brute-force O(n^2) pairwise checks, which is fine for
        typical landmark counts (< 200).

        Args:
            elements: List of map elements to check.

        Returns:
            List of (index_a, index_b) pairs where index_a < index_b.
        """
        collisions: list[tuple[int, int]] = []
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                if elements[i].overlaps(elements[j], self.padding):
                    collisions.append((i, j))
        return collisions

    def resolve_collisions(
        self,
        elements: list[MapElement],
        max_iterations: int = 50,
    ) -> list[MapElement]:
        """Resolve overlaps using force-based resolution.

        For each overlapping pair:
        - The element with lower importance moves.
        - It moves away from the higher-importance element.
        - Movement is along the vector connecting the two centres
          (or perpendicular if centres coincide).
        - Elements are kept within map bounds.

        Iterates until no collisions remain or max_iterations reached.

        Args:
            elements: List of map elements (not modified in place).
            max_iterations: Safety limit on iteration count.

        Returns:
            A new list of MapElement instances with adjusted positions.
        """
        # Work on copies so the caller's list is not mutated
        result = [replace(e) for e in elements]

        for iteration in range(max_iterations):
            collisions = self.detect_collisions(result)
            if not collisions:
                break

            for idx_a, idx_b in collisions:
                a = result[idx_a]
                b = result[idx_b]

                # Decide who moves: lower importance moves.
                # If tied, the one with the higher index moves.
                if a.importance <= b.importance:
                    mover_idx, anchor_idx = idx_a, idx_b
                else:
                    mover_idx, anchor_idx = idx_b, idx_a

                # Repulsion strength decays over iterations to help
                # convergence. Start strong (1.0) and taper.
                strength = max(0.3, 1.0 - iteration * 0.02)

                result[mover_idx] = self._apply_repulsion(
                    result[mover_idx], result[anchor_idx], strength
                )
                result[mover_idx] = self._clamp_to_bounds(result[mover_idx])

        return result

    def _apply_repulsion(
        self, mover: MapElement, anchor: MapElement, strength: float = 1.0
    ) -> MapElement:
        """Push *mover* away from *anchor*.

        The displacement direction is along the vector from anchor's
        centre to mover's centre.  If the centres are coincident, a
        small rightward nudge is used to break the tie.

        The displacement magnitude is proportional to the overlap depth
        plus the padding, scaled by *strength*.

        Args:
            mover: The element that will be displaced.
            anchor: The element that stays in place.
            strength: Multiplier for displacement distance.

        Returns:
            A new MapElement with updated (x, y).
        """
        dx = mover.x - anchor.x
        dy = mover.y - anchor.y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            # Centres coincide — nudge to the right
            dx, dy, dist = 1.0, 0.0, 1.0

        # Unit direction vector from anchor to mover
        ux = dx / dist
        uy = dy / dist

        # Required separation along that direction.
        # Half-widths projected onto the direction vector give the
        # combined extent.  For axis-aligned rectangles a simpler
        # calculation per axis is more accurate, so we compute the
        # minimum translation vector (MTV) per axis and pick the
        # smaller one.
        overlap_x = (
            (mover.width / 2 + anchor.width / 2 + self.padding)
            - abs(dx)
        )
        overlap_y = (
            (mover.height / 2 + anchor.height / 2 + self.padding)
            - abs(dy)
        )

        # If there is no actual overlap on one axis the elements are not
        # colliding; but we were called because overlaps() said True, so
        # at least one of these should be positive.  Move along the axis
        # with the smaller positive overlap (the MTV axis).
        if overlap_x <= 0 and overlap_y <= 0:
            return mover

        if overlap_x <= 0:
            # Only overlapping on Y axis
            move_x = 0.0
            move_y = math.copysign(overlap_y * strength, dy) if dy != 0 else overlap_y * strength
        elif overlap_y <= 0:
            # Only overlapping on X axis
            move_x = math.copysign(overlap_x * strength, dx) if dx != 0 else overlap_x * strength
            move_y = 0.0
        elif overlap_x < overlap_y:
            # Cheaper to separate along X
            move_x = math.copysign(overlap_x * strength, dx) if dx != 0 else overlap_x * strength
            move_y = 0.0
        else:
            # Cheaper to separate along Y
            move_x = 0.0
            move_y = math.copysign(overlap_y * strength, dy) if dy != 0 else overlap_y * strength

        return replace(mover, x=mover.x + move_x, y=mover.y + move_y)

    def _clamp_to_bounds(self, element: MapElement) -> MapElement:
        """Ensure element stays within map bounds.

        The element's centre is clamped so the entire rectangle
        (including half-width/height) stays inside the canvas.

        Args:
            element: Element to clamp.

        Returns:
            A new MapElement with clamped position.
        """
        min_x = element.width / 2
        max_x = self.map_width - element.width / 2
        min_y = element.height / 2
        max_y = self.map_height - element.height / 2

        # Handle the case where the element is wider/taller than the map
        if min_x > max_x:
            clamped_x = self.map_width / 2
        else:
            clamped_x = max(min_x, min(element.x, max_x))

        if min_y > max_y:
            clamped_y = self.map_height / 2
        else:
            clamped_y = max(min_y, min(element.y, max_y))

        if clamped_x == element.x and clamped_y == element.y:
            return element

        return replace(element, x=clamped_x, y=clamped_y)

    def remove_least_important(
        self,
        elements: list[MapElement],
        max_elements: int,
    ) -> list[MapElement]:
        """If too many elements remain after resolution, remove least important ones.

        Keeps the *max_elements* most important elements, preserving
        their relative order.

        Args:
            elements: List of map elements.
            max_elements: Maximum number to keep.

        Returns:
            A (possibly shorter) list, sorted by their original order.
        """
        if len(elements) <= max_elements:
            return list(elements)

        # Pair each element with its original index
        indexed = list(enumerate(elements))

        # Sort by importance descending, then by original index ascending
        # (so ties keep earlier elements)
        indexed.sort(key=lambda pair: (-pair[1].importance, pair[0]))

        # Take top max_elements
        kept = indexed[:max_elements]

        # Restore original order
        kept.sort(key=lambda pair: pair[0])

        return [elem for _idx, elem in kept]
