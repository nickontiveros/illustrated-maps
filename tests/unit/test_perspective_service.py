"""Tests for PerspectiveService."""

import math

import numpy as np
import pytest
from PIL import Image

from mapgen.services.perspective_service import PerspectiveService


class TestPerspectiveServiceInit:
    """Test service initialization."""

    def test_default_parameters(self):
        svc = PerspectiveService()
        assert svc.angle == 35.0
        assert svc.convergence == 0.7
        assert svc.vertical_scale == 0.4
        assert svc.horizon_margin == 0.15

    def test_custom_parameters(self):
        svc = PerspectiveService(angle=45, convergence=0.5, vertical_scale=0.6, horizon_margin=0.2)
        assert svc.angle == 45
        assert svc.convergence == 0.5


class TestTransformCoordinates:
    """Test coordinate transformation."""

    def test_bottom_center_stays_near_center_x(self):
        svc = PerspectiveService()
        x, y = svc.transform_coordinates(500, 1000, (1000, 1000))
        # Bottom of image has no convergence -> x stays ~500
        assert x == pytest.approx(500, abs=5)

    def test_top_row_converges_inward(self):
        svc = PerspectiveService(convergence=0.7)
        lx, _ = svc.transform_coordinates(0, 0, (1000, 1000))
        rx, _ = svc.transform_coordinates(1000, 0, (1000, 1000))
        # Top row should be narrower than 1000px
        assert (rx - lx) < 1000

    def test_y_offset_includes_margin(self):
        svc = PerspectiveService(horizon_margin=0.15)
        _, y = svc.transform_coordinates(0, 0, (1000, 1000))
        # Top of image should have a y offset >= margin
        assert y >= 150  # 15% of 1000


class TestGetOutputSize:
    """Test output size calculation."""

    def test_adds_margin(self):
        svc = PerspectiveService(horizon_margin=0.15)
        w, h = svc.get_output_size((1000, 1000))
        assert w == 1000
        assert h == 1150  # 1000 + 15%

    def test_zero_margin(self):
        svc = PerspectiveService(horizon_margin=0.0)
        w, h = svc.get_output_size((800, 600))
        assert (w, h) == (800, 600)


class TestGetRotatedOutputSize:
    """Test rotated output size calculation."""

    def test_zero_rotation(self):
        svc = PerspectiveService(horizon_margin=0.0)
        w, h = svc.get_rotated_output_size((1000, 500), 0)
        assert (w, h) == (1000, 500)

    def test_90_degree_rotation(self):
        svc = PerspectiveService(horizon_margin=0.0)
        w, h = svc.get_rotated_output_size((1000, 500), 90)
        # After 90-degree rotation, width/height swap (approximately)
        assert w != 1000 or h != 500


class TestIsometricMatrix:
    """Test isometric matrix creation."""

    def test_matrix_shape(self):
        svc = PerspectiveService()
        matrix = svc.create_isometric_matrix()
        assert matrix.shape == (4, 4)

    def test_matrix_invertible(self):
        svc = PerspectiveService()
        matrix = svc.create_isometric_matrix()
        det = np.linalg.det(matrix)
        assert abs(det) > 1e-10

    def test_matrix_property_caches(self):
        svc = PerspectiveService()
        m1 = svc.matrix
        m2 = svc.matrix
        assert np.array_equal(m1, m2)

    def test_inverse_matrix(self):
        svc = PerspectiveService()
        m = svc.matrix
        inv = svc.inverse_matrix
        identity = m @ inv
        assert np.allclose(identity, np.eye(4), atol=1e-10)


class TestTransformPoint:
    """Test single point transformation."""

    def test_origin(self):
        svc = PerspectiveService()
        tx, ty = svc.transform_point(0, 0, 0)
        assert tx == pytest.approx(0, abs=1e-10)
        assert ty == pytest.approx(0, abs=1e-10)

    def test_elevation_shifts_y(self):
        svc = PerspectiveService()
        _, y0 = svc.transform_point(0.5, 0.5, 0)
        _, y1 = svc.transform_point(0.5, 0.5, 1000, elevation_scale=0.001)
        # Non-zero elevation should shift y
        assert y0 != y1


class TestTransformImage:
    """Test full image transformation."""

    def test_produces_output(self):
        svc = PerspectiveService()
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = svc.transform_image(img)
        assert result is not None
        assert result.mode == "RGBA"

    def test_output_dimensions(self):
        svc = PerspectiveService(horizon_margin=0.15)
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = svc.transform_image(img)
        expected_w, expected_h = svc.get_output_size((100, 100))
        assert result.size == (expected_w, expected_h)

    def test_background_color(self):
        svc = PerspectiveService(horizon_margin=0.5)
        bg = (100, 200, 50, 255)
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        result = svc.transform_image(img, background_color=bg)
        # Top margin area should have background color
        pixel = result.getpixel((50, 0))
        assert pixel == bg


class TestCalculateYOffset:
    """Test elevation Y offset."""

    def test_zero_elevation(self):
        svc = PerspectiveService()
        assert svc.calculate_y_offset(0) == 0.0

    def test_positive_elevation(self):
        svc = PerspectiveService(angle=45)
        offset = svc.calculate_y_offset(1000, elevation_scale=0.001)
        expected = 1000 * 0.001 * math.sin(math.radians(45))
        assert offset == pytest.approx(expected)


class TestTransformImageCoordinates:
    """Test pixel coordinate transformation."""

    def test_origin(self):
        svc = PerspectiveService()
        x, y = svc.transform_image_coordinates(0, 0, (100, 100))
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_no_elevation(self):
        svc = PerspectiveService()
        x1, y1 = svc.transform_image_coordinates(50, 50, (100, 100), elevation=0)
        x2, y2 = svc.transform_image_coordinates(50, 50, (100, 100), elevation=1000)
        # With elevation, coordinates should differ
        assert (x1, y1) != (x2, y2)
