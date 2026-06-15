"""Tests for the shared color toolkit (soft replace, palette ΔE, Lab match)."""

import numpy as np
import pytest
from PIL import Image

from mapgen.v2.color import (
    _lab_to_srgb,
    _srgb_to_lab,
    normalize_to_reference,
    palette_distance,
    soft_color_replace,
)

PALETTE = {"paper": "#f3e9d4", "water": "#8fb8b2", "park": "#a8b87f"}


def make_image(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, color)


class TestLabRoundTrip:
    def test_round_trip_preserves_colors(self):
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, size=(50, 3)).astype(np.float32)
        back = _lab_to_srgb(_srgb_to_lab(rgb))
        assert np.abs(back - rgb).max() < 1.0

    def test_white_and_black(self):
        lab = _srgb_to_lab(np.array([[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]]))
        assert lab[0, 0] == pytest.approx(100.0, abs=0.1)
        assert lab[1, 0] == pytest.approx(0.0, abs=0.1)


class TestSoftColorReplace:
    def test_exact_target_fully_replaced(self):
        img = make_image((100, 150, 200))
        out = soft_color_replace(img, (100, 150, 200), (10, 20, 30))
        assert np.allclose(np.asarray(out), [10, 20, 30], atol=1)

    def test_distant_color_untouched(self):
        img = make_image((250, 30, 30))
        out = soft_color_replace(img, (100, 150, 200), (10, 20, 30))
        assert np.array_equal(np.asarray(out), np.asarray(img))

    def test_edge_pixels_get_partial_blend(self):
        # A pixel partway between target and far color receives a partial
        # replacement, preserving anti-aliased transitions.
        target, far = np.array([100.0, 150.0, 200.0]), np.array([250.0, 30.0, 30.0])
        edge = tuple(int(v) for v in (0.7 * target + 0.3 * far))
        img = make_image(edge)
        out = np.asarray(soft_color_replace(img, tuple(target.astype(int)), (0, 0, 0),
                                            tolerance=20.0, softness=80.0)).astype(float)
        original = np.asarray(img).astype(float)
        assert (out < original - 5).any()  # darkened toward replacement...
        assert (out > 5).all()  # ...but not fully replaced

    def test_mask_restricts_replacement(self):
        img = make_image((100, 150, 200), size=(10, 10))
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5] = True
        out = np.asarray(soft_color_replace(img, (100, 150, 200), (0, 0, 0), mask=mask))
        assert np.allclose(out[:5], 0, atol=1)
        assert np.allclose(out[5:], [100, 150, 200], atol=1)

    def test_alpha_channel_preserved(self):
        img = make_image((100, 150, 200)).convert("RGBA")
        alpha = Image.new("L", img.size, 128)
        img.putalpha(alpha)
        out = soft_color_replace(img, (100, 150, 200), (0, 0, 0))
        assert out.mode == "RGBA"
        assert np.asarray(out.getchannel("A")).max() == 128


class TestPaletteDistance:
    def test_in_palette_scores_low(self):
        img = make_image((143, 184, 178))  # palette water #8fb8b2
        assert palette_distance(img, PALETTE) < 2.0

    def test_off_palette_scores_high(self):
        img = make_image((200, 40, 160))  # magenta: nothing close in palette
        assert palette_distance(img, PALETTE) > 35.0

    def test_small_accent_does_not_dominate(self):
        # 5% off-palette accent: worst-decile mean includes it, but a half
        # off-palette image must score much worse.
        base = np.tile(np.array([143, 184, 178], dtype=np.uint8), (100, 100, 1))
        accent = base.copy()
        accent[:5] = [200, 40, 160]
        half = base.copy()
        half[:50] = [200, 40, 160]
        accent_score = palette_distance(Image.fromarray(accent), PALETTE)
        half_score = palette_distance(Image.fromarray(half), PALETTE)
        assert accent_score < half_score

    def test_transparent_pixels_ignored(self):
        img = make_image((200, 40, 160)).convert("RGBA")
        img.putalpha(Image.new("L", img.size, 0))
        assert palette_distance(img, PALETTE) == 0.0

    def test_large_image_downsampled_not_oom(self):
        img = make_image((143, 184, 178), size=(2048, 2048))
        assert palette_distance(img, PALETTE) < 2.0


class TestNormalizeToReference:
    def test_corrects_brightness_drift(self):
        rng = np.random.default_rng(3)
        ref_arr = rng.integers(60, 200, size=(64, 64, 3)).astype(np.uint8)
        drifted = np.clip(ref_arr.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        out = normalize_to_reference(Image.fromarray(drifted), Image.fromarray(ref_arr))
        out_mean = np.asarray(out, dtype=float).mean()
        ref_mean = ref_arr.astype(float).mean()
        assert abs(out_mean - ref_mean) < 4.0

    def test_identity_when_already_matched(self):
        rng = np.random.default_rng(4)
        arr = rng.integers(40, 220, size=(64, 64, 3)).astype(np.uint8)
        img = Image.fromarray(arr)
        out = np.asarray(normalize_to_reference(img, img), dtype=float)
        assert np.abs(out - arr.astype(float)).max() < 3.0

    def test_excluded_pixels_do_not_skew_stats(self):
        # Same content in both images, but the (excluded) top half of the
        # "generated" image is a flat wrong color (think: water). With the
        # mask, the bottom half must come back essentially unchanged.
        rng = np.random.default_rng(5)
        content = rng.integers(60, 200, size=(32, 64, 3)).astype(np.uint8)
        ref = np.vstack([np.full((32, 64, 3), 120, dtype=np.uint8), content])
        gen = np.vstack([np.full((32, 64, 3), 30, dtype=np.uint8), content])
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32] = True
        out = np.asarray(
            normalize_to_reference(Image.fromarray(gen), Image.fromarray(ref), exclude_mask=mask),
            dtype=float,
        )
        assert np.abs(out[32:] - content.astype(float)).max() < 3.0

    def test_strength_scales_correction(self):
        ref = make_image((100, 100, 100))
        drifted = make_image((160, 160, 160))
        half = np.asarray(normalize_to_reference(drifted, ref, strength=0.5), dtype=float).mean()
        full = np.asarray(normalize_to_reference(drifted, ref, strength=1.0), dtype=float).mean()
        assert full < half < 160

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            normalize_to_reference(make_image((0, 0, 0), (10, 10)), make_image((0, 0, 0), (20, 20)))

    def test_fully_excluded_window_returned_unchanged(self):
        img = make_image((50, 60, 70))
        mask = np.ones((64, 64), dtype=bool)
        out = normalize_to_reference(img, make_image((200, 200, 200)), exclude_mask=mask)
        assert np.array_equal(np.asarray(out), np.asarray(img))
