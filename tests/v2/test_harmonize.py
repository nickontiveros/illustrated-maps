import numpy as np
from PIL import Image

from mapgen.v2.compose.harmonize import IdentityMoodPass, harmonize, low_frequency
from mapgen.v2.types import StyleSpec


class WarmTintPass:
    """Mood pass that shifts everything warm (low-frequency change only)."""

    def repaint(self, composite, style):
        arr = np.asarray(composite.convert("RGB"), dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] + 30, 0, 255)  # +red
        arr[..., 2] = np.clip(arr[..., 2] - 20, 0, 255)  # -blue
        return Image.fromarray(arr.astype(np.uint8), "RGB")


class VandalPass:
    """Mood pass that draws high-frequency garbage (must be filtered out)."""

    def repaint(self, composite, style):
        arr = np.asarray(composite.convert("RGB")).copy()
        arr[::2, ::2] = (255, 0, 255)  # checkerboard noise
        return Image.fromarray(arr, "RGB")


def _poster() -> Image.Image:
    """A small synthetic poster with sharp detail (a black grid on cream)."""
    img = Image.new("RGB", (400, 560), (240, 230, 205))
    arr = np.asarray(img).copy()
    arr[::40, :] = (40, 30, 20)
    arr[:, ::40] = (40, 30, 20)
    return Image.fromarray(arr, "RGB")


def test_strength_zero_is_identity():
    poster = _poster()
    out = harmonize(poster, WarmTintPass(), StyleSpec(), strength=0.0)
    assert out is poster


def test_mood_shifts_color_globally(artifacts):
    poster = _poster()
    out = harmonize(poster, WarmTintPass(), StyleSpec(), strength=1.0)
    artifacts.save("before", poster)
    artifacts.save("after", out)
    before = np.asarray(poster, dtype=np.float32)
    after = np.asarray(out, dtype=np.float32)
    # Red goes up, blue goes down, on average.
    assert after[..., 0].mean() > before[..., 0].mean() + 15
    assert after[..., 2].mean() < before[..., 2].mean() - 8


def test_detail_survives_harmonization():
    """The sharp grid lines must remain sharp: high frequencies untouched."""
    poster = _poster()
    out = harmonize(poster, WarmTintPass(), StyleSpec(), strength=1.0)
    before = np.asarray(poster.convert("L"), dtype=np.float32)
    after = np.asarray(out.convert("L"), dtype=np.float32)
    # Edge contrast across a grid line, sampled away from image borders.
    edge_before = abs(before[200, 39] - before[200, 20])
    edge_after = abs(after[200, 39] - after[200, 20])
    assert edge_after > edge_before * 0.8


def test_high_frequency_vandalism_is_filtered_out(artifacts):
    poster = _poster()
    out = harmonize(poster, VandalPass(), StyleSpec(), strength=1.0)
    artifacts.save("vandal_filtered", out)
    after = np.asarray(out, dtype=np.float32)
    before = np.asarray(poster, dtype=np.float32)
    # The checkerboard would flip half the pixels to magenta; after the
    # low-pass blend the per-pixel difference must stay small and smooth.
    diff = np.abs(after - before)
    assert diff.max() < 90  # no raw magenta pixels survive
    # Neighboring pixels changed by nearly identical amounts (no checkering).
    checker = np.abs(np.diff(after[..., 0], axis=1))[100:120, 100:120].max()
    original_checker = np.abs(np.diff(before[..., 0], axis=1))[100:120, 100:120].max()
    assert checker <= original_checker + 6


def test_identity_pass_changes_nothing():
    poster = _poster()
    out = harmonize(poster, IdentityMoodPass(), StyleSpec(), strength=1.0)
    before = np.asarray(poster, dtype=np.float32)
    after = np.asarray(out, dtype=np.float32)
    # Down/up-scale round trip introduces only low-frequency error.
    assert np.abs(after - before).mean() < 3.0


def test_low_frequency_is_blurry():
    poster = _poster()
    lf = low_frequency(poster, radius=12)
    # Grid lines are averaged away.
    assert np.abs(np.diff(lf[..., 0], axis=1)).max() < 30
