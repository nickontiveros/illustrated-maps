"""Tests for mapgen.services.border_service."""

import numpy as np
import pytest
from PIL import Image

from mapgen.models.border import BorderSettings, BorderStyle, LegendItem
from mapgen.models.project import BoundingBox
from mapgen.services.border_service import BorderService


@pytest.fixture
def service():
    return BorderService()


# ---------------------------------------------------------------------------
# _expand_canvas
# ---------------------------------------------------------------------------

class TestExpandCanvas:
    def test_expand_canvas_size(self, service, solid_red_image):
        margin = 20
        result = BorderService._expand_canvas(solid_red_image, margin=margin, bg_color="#FFFFFF")
        expected_w = solid_red_image.width + 2 * margin
        expected_h = solid_red_image.height + 2 * margin
        assert result.size == (expected_w, expected_h)

    def test_expand_canvas_preserves_original(self, service, small_test_image):
        margin = 10
        result = BorderService._expand_canvas(small_test_image, margin=margin, bg_color="#000000")
        # The center area should match the original
        arr = np.array(result)
        orig = np.array(small_test_image)
        center = arr[margin:margin + small_test_image.height, margin:margin + small_test_image.width]
        np.testing.assert_array_equal(center, orig)

    def test_expand_canvas_bg_color(self, service):
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
        margin = 5
        result = BorderService._expand_canvas(img, margin=margin, bg_color="#FF0000")
        arr = np.array(result)
        # Top-left corner should be red
        assert arr[0, 0, 0] == 255  # R
        assert arr[0, 0, 1] == 0    # G
        assert arr[0, 0, 2] == 0    # B

    def test_zero_margin(self, service, solid_red_image):
        result = BorderService._expand_canvas(solid_red_image, margin=0, bg_color="#000000")
        assert result.size == solid_red_image.size


# ---------------------------------------------------------------------------
# render_border - all 4 styles
# ---------------------------------------------------------------------------

class TestRenderBorder:
    @pytest.mark.parametrize("style", list(BorderStyle))
    def test_render_all_styles_no_error(self, service, large_test_image, sample_bbox, style):
        settings = BorderSettings(enabled=True, style=style, margin=50)
        result = service.render_border(
            map_image=large_test_image,
            settings=settings,
            title="Test Map",
            subtitle="A subtitle",
            bbox=sample_bbox,
            rotation_degrees=0,
        )
        assert isinstance(result, Image.Image)

    def test_render_border_returns_larger_image(self, service, large_test_image, sample_bbox):
        settings = BorderSettings(enabled=True, margin=50)
        result = service.render_border(
            map_image=large_test_image,
            settings=settings,
            title="Test",
            subtitle=None,
            bbox=sample_bbox,
            rotation_degrees=0,
        )
        assert result.width >= large_test_image.width
        assert result.height >= large_test_image.height

    def test_render_border_with_title(self, service, large_test_image, sample_bbox):
        settings = BorderSettings(enabled=True, margin=80)
        result = service.render_border(
            map_image=large_test_image,
            settings=settings,
            title="My Beautiful Map",
            subtitle="Subtitle Here",
            bbox=sample_bbox,
            rotation_degrees=0,
        )
        assert isinstance(result, Image.Image)

    def test_render_border_with_rotation(self, service, large_test_image, sample_bbox):
        settings = BorderSettings(enabled=True, margin=50)
        result = service.render_border(
            map_image=large_test_image,
            settings=settings,
            title="Rotated Map",
            subtitle=None,
            bbox=sample_bbox,
            rotation_degrees=15,
        )
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# _auto_generate_legend_items
# ---------------------------------------------------------------------------

class TestAutoGenerateLegendItems:
    def test_returns_six_items(self):
        items = BorderService._auto_generate_legend_items()
        assert len(items) == 6

    def test_items_are_legend_items(self):
        items = BorderService._auto_generate_legend_items()
        for item in items:
            assert isinstance(item, LegendItem)

    def test_items_have_labels_and_colors(self):
        items = BorderService._auto_generate_legend_items()
        for item in items:
            assert item.label
            assert item.color.startswith("#")
