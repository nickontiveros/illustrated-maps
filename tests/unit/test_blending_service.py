"""Tests for BlendingService."""

import numpy as np
import pytest
from PIL import Image

from mapgen.services.blending_service import BlendingService, TileInfo


class TestBlendingServiceInit:
    """Test service initialization."""

    def test_default_levels(self):
        svc = BlendingService()
        assert svc.num_pyramid_levels == 5

    def test_custom_levels(self):
        svc = BlendingService(num_pyramid_levels=3)
        assert svc.num_pyramid_levels == 3


class TestCreateGradientMask:
    """Test gradient mask creation."""

    def test_horizontal_shape(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(100, 50, direction="horizontal")
        assert mask.shape == (50, 100)

    def test_vertical_shape(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(100, 50, direction="vertical")
        assert mask.shape == (50, 100)

    def test_horizontal_values_range(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(100, 50, direction="horizontal")
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_horizontal_gradient_direction(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(100, 50, direction="horizontal", center=0.5, falloff=0.3)
        # Left side should have lower values than right side
        assert mask[25, 10] < mask[25, 90]

    def test_vertical_gradient_direction(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(50, 100, direction="vertical", center=0.5, falloff=0.3)
        # Top should have lower values than bottom
        assert mask[10, 25] < mask[90, 25]

    def test_horizontal_rows_uniform(self):
        svc = BlendingService()
        mask = svc.create_gradient_mask(100, 50, direction="horizontal")
        # All rows should be identical for horizontal gradient
        assert np.allclose(mask[0], mask[25])
        assert np.allclose(mask[0], mask[49])


class TestMultibandBlend:
    """Test Laplacian pyramid blending."""

    def test_identity_blend(self):
        svc = BlendingService(num_pyramid_levels=3)
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        mask = np.zeros((64, 64), dtype=np.float32)  # All img1
        result = svc.multiband_blend(img, img, mask, levels=3)
        arr = np.array(result)
        # Blending identical images should produce approximately the same image
        assert np.allclose(arr[:, :, :3], 128, atol=10)

    def test_full_mask_selects_img2(self):
        svc = BlendingService(num_pyramid_levels=3)
        img1 = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
        mask = np.ones((64, 64), dtype=np.float32)  # All img2
        result = svc.multiband_blend(img1, img2, mask, levels=3)
        arr = np.array(result)
        # Should be close to blue
        center = arr[32, 32]
        assert center[2] > center[0]  # More blue than red

    def test_output_is_rgba(self):
        svc = BlendingService(num_pyramid_levels=3)
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        mask = np.zeros((64, 64), dtype=np.float32)
        result = svc.multiband_blend(img, img, mask, levels=3)
        assert result.mode == "RGBA"


class TestGaussianPyramid:
    """Test Gaussian pyramid building."""

    def test_pyramid_levels(self):
        svc = BlendingService()
        arr = np.random.rand(64, 64, 4).astype(np.float32)
        pyramid = svc._build_gaussian_pyramid(arr, 4)
        assert len(pyramid) == 4

    def test_pyramid_sizes_decrease(self):
        svc = BlendingService()
        arr = np.random.rand(64, 64, 4).astype(np.float32)
        pyramid = svc._build_gaussian_pyramid(arr, 4)
        for i in range(len(pyramid) - 1):
            assert pyramid[i].shape[0] >= pyramid[i + 1].shape[0]
            assert pyramid[i].shape[1] >= pyramid[i + 1].shape[1]


class TestLaplacianPyramid:
    """Test Laplacian pyramid."""

    def test_pyramid_levels(self):
        svc = BlendingService()
        arr = np.random.rand(64, 64, 4).astype(np.float32)
        pyramid = svc._build_laplacian_pyramid(arr, 4)
        assert len(pyramid) == 4

    def test_reconstruction_preserves_content(self):
        svc = BlendingService()
        arr = np.random.rand(64, 64, 4).astype(np.float32)
        pyramid = svc._build_laplacian_pyramid(arr, 3)
        reconstructed = svc._reconstruct_from_pyramid(pyramid)
        # Should be approximately the same as original
        assert np.allclose(arr, reconstructed, atol=0.1)


class TestBlendTiles:
    """Test tile assembly."""

    def _make_tile(self, color, col, row, x_offset, y_offset, max_col=0, max_row=0):
        img = Image.new("RGBA", (64, 64), color)
        return TileInfo(
            image=img,
            col=col,
            row=row,
            x_offset=x_offset,
            y_offset=y_offset,
            max_col=max_col,
            max_row=max_row,
        )

    def test_single_tile(self):
        svc = BlendingService()
        tile = self._make_tile((255, 0, 0, 255), 0, 0, 0, 0)
        result = svc.blend_tiles([tile], (64, 64), overlap=0)
        assert result.size == (64, 64)
        assert result.getpixel((32, 32)) == (255, 0, 0, 255)

    def test_two_tiles_horizontal(self):
        svc = BlendingService()
        tile1 = self._make_tile((255, 0, 0, 255), 0, 0, 0, 0)
        tile2 = self._make_tile((0, 0, 255, 255), 1, 0, 48, 0)
        result = svc.blend_tiles([tile1, tile2], (112, 64), overlap=16)
        assert result.size == (112, 64)
        # Left should be red
        assert result.getpixel((5, 32))[0] == 255
        # Right should be blue
        assert result.getpixel((100, 32))[2] == 255

    def test_output_size_respected(self):
        svc = BlendingService()
        tile = self._make_tile((255, 0, 0, 255), 0, 0, 0, 0)
        result = svc.blend_tiles([tile], (200, 200), overlap=0)
        assert result.size == (200, 200)


class TestWeightMask:
    """Test weight mask creation."""

    def test_center_is_one(self):
        svc = BlendingService()
        mask = svc._create_weight_mask(100, 100, 20)
        assert mask[50, 50] == pytest.approx(1.0, abs=0.05)

    def test_shape(self):
        svc = BlendingService()
        mask = svc._create_weight_mask(80, 60, 10)
        assert mask.shape == (60, 80)

    def test_selective_feathering(self):
        svc = BlendingService()
        mask_full = svc._create_weight_mask_selective(100, 100, 20, True, True, True, True)
        mask_left_only = svc._create_weight_mask_selective(100, 100, 20, True, False, False, False)
        # Left-only feathered should have higher weight on the right edge
        assert mask_left_only[50, 95] > mask_full[50, 95]
