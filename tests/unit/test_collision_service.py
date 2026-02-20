"""Tests for mapgen.services.collision_service."""

import pytest

from mapgen.services.collision_service import CollisionService, MapElement


# ---------------------------------------------------------------------------
# MapElement dataclass
# ---------------------------------------------------------------------------

class TestMapElement:
    def test_creation(self):
        el = MapElement(x=10, y=20, width=50, height=30, importance=0.8, label="Test")
        assert el.x == 10
        assert el.y == 20
        assert el.width == 50
        assert el.height == 30
        assert el.importance == 0.8
        assert el.label == "Test"

    def test_left_right_top_bottom(self):
        el = MapElement(x=10, y=20, width=50, height=30, importance=0.5, label="A")
        # x,y are center coords: left = x - width/2, right = x + width/2
        assert el.left == 10 - 25  # -15
        assert el.right == 10 + 25  # 35
        assert el.top == 20 - 15  # 5
        assert el.bottom == 20 + 15  # 35

    def test_overlaps_true(self):
        a = MapElement(x=0, y=0, width=20, height=20, importance=1.0, label="A")
        b = MapElement(x=10, y=10, width=20, height=20, importance=0.5, label="B")
        assert a.overlaps(b, padding=0) is True

    def test_overlaps_false(self):
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=20, y=20, width=10, height=10, importance=0.5, label="B")
        assert a.overlaps(b, padding=0) is False

    def test_overlaps_adjacent_no_overlap(self):
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=10, y=0, width=10, height=10, importance=0.5, label="B")
        assert a.overlaps(b, padding=0) is False

    def test_overlaps_with_padding(self):
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=12, y=0, width=10, height=10, importance=0.5, label="B")
        # Without padding, they don't overlap
        assert a.overlaps(b, padding=0) is False
        # With padding=3, they do overlap
        assert a.overlaps(b, padding=3) is True

    def test_overlap_area_no_overlap(self):
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=20, y=20, width=10, height=10, importance=0.5, label="B")
        assert a.overlap_area(b) == 0.0

    def test_overlap_area_partial(self):
        a = MapElement(x=0, y=0, width=20, height=20, importance=1.0, label="A")
        b = MapElement(x=10, y=10, width=20, height=20, importance=0.5, label="B")
        area = a.overlap_area(b)
        assert area == 100.0  # 10x10 overlap

    def test_overlap_area_full_containment(self):
        a = MapElement(x=0, y=0, width=30, height=30, importance=1.0, label="A")
        b = MapElement(x=5, y=5, width=10, height=10, importance=0.5, label="B")
        area = a.overlap_area(b)
        assert area == 100.0  # b is fully inside a


# ---------------------------------------------------------------------------
# CollisionService
# ---------------------------------------------------------------------------

class TestCollisionService:
    def test_creation(self):
        svc = CollisionService(map_size=(512, 512), padding=5)
        assert svc.map_width == 512
        assert svc.map_height == 512
        assert svc.padding == 5


class TestDetectCollisions:
    def test_empty_list(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        assert svc.detect_collisions([]) == []

    def test_single_element_no_collisions(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        elements = [MapElement(x=10, y=10, width=20, height=20, importance=1.0, label="A")]
        assert svc.detect_collisions(elements) == []

    def test_two_overlapping_elements(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        a = MapElement(x=0, y=0, width=20, height=20, importance=1.0, label="A")
        b = MapElement(x=10, y=10, width=20, height=20, importance=0.5, label="B")
        collisions = svc.detect_collisions([a, b])
        assert len(collisions) == 1
        assert collisions[0] == (0, 1)

    def test_two_non_overlapping_elements(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=50, y=50, width=10, height=10, importance=0.5, label="B")
        collisions = svc.detect_collisions([a, b])
        assert len(collisions) == 0

    def test_three_all_overlapping(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        a = MapElement(x=0, y=0, width=30, height=30, importance=1.0, label="A")
        b = MapElement(x=10, y=10, width=30, height=30, importance=0.8, label="B")
        c = MapElement(x=20, y=20, width=30, height=30, importance=0.6, label="C")
        collisions = svc.detect_collisions([a, b, c])
        assert len(collisions) == 3  # (0,1), (0,2), (1,2)


class TestResolveCollisions:
    def test_no_collisions_returns_same(self):
        svc = CollisionService(map_size=(200, 200), padding=0)
        a = MapElement(x=0, y=0, width=10, height=10, importance=1.0, label="A")
        b = MapElement(x=50, y=50, width=10, height=10, importance=0.5, label="B")
        resolved = svc.resolve_collisions([a, b], max_iterations=10)
        # Should return list of same length
        assert len(resolved) == 2

    def test_does_not_mutate_original(self):
        svc = CollisionService(map_size=(200, 200), padding=0)
        a = MapElement(x=0, y=0, width=20, height=20, importance=1.0, label="A")
        b = MapElement(x=5, y=5, width=20, height=20, importance=0.5, label="B")
        original_bx = b.x
        original_by = b.y
        resolved = svc.resolve_collisions([a, b], max_iterations=10)
        # Original elements should not be mutated
        assert b.x == original_bx
        assert b.y == original_by

    def test_moves_lower_importance_element(self):
        svc = CollisionService(map_size=(200, 200), padding=0)
        a = MapElement(x=10, y=10, width=20, height=20, importance=1.0, label="A")
        b = MapElement(x=15, y=15, width=20, height=20, importance=0.3, label="B")
        resolved = svc.resolve_collisions([a, b], max_iterations=20)
        # After resolution, a should stay roughly in place, b should move
        assert resolved[0].x == a.x
        assert resolved[0].y == a.y

    def test_keeps_elements_in_bounds(self):
        svc = CollisionService(map_size=(100, 100), padding=0)
        a = MapElement(x=0, y=0, width=50, height=50, importance=1.0, label="A")
        b = MapElement(x=10, y=10, width=50, height=50, importance=0.5, label="B")
        resolved = svc.resolve_collisions([a, b], max_iterations=20)
        for el in resolved:
            assert el.x >= 0
            assert el.y >= 0
            assert el.right <= 100
            assert el.bottom <= 100
