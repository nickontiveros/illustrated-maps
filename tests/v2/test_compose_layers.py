"""Layered (PSD) export: the compositor peels sprites and labels into their
own layers, and the self-contained PSD writer round-trips through psd-tools."""

import numpy as np
import pytest
from PIL import Image

from mapgen.v2 import pipeline
from mapgen.v2.assets.stub import StubAssetGenerator
from mapgen.v2.compose import Compositor, Layer, LayerStack
from mapgen.v2.compose.psd_writer import _pack_bits, write_psd
from mapgen.v2.plan import PlanBuilder


@pytest.fixture
def plan(source, small_canvas):
    builder = PlanBuilder(canvas=small_canvas, distortion_strength=0.4)
    return builder.build(source, title="Test Town")


@pytest.fixture
def assets_dir(plan, tmp_path):
    pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
    return tmp_path / pipeline.ASSETS_DIRNAME


# --- PackBits ---------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"A",
        b"AAAAAAA",
        b"ABCDEFG",
        b"AAAABCDAAAA",
        bytes(range(256)) + b"\x00" * 300,
    ],
)
def test_packbits_roundtrips(data):
    # Decode PackBits the standard way and compare to the source.
    packed = _pack_bits(data)
    out = bytearray()
    i = 0
    while i < len(packed):
        n = packed[i]
        i += 1
        if n < 128:
            out += packed[i : i + n + 1]
            i += n + 1
        elif n > 128:
            out += bytes([packed[i]]) * (257 - n)
            i += 1
    assert bytes(out) == data


# --- render_layers ----------------------------------------------------------


def test_render_layers_structure(plan, assets_dir):
    stack = Compositor(plan, assets_dir).render_layers(scale=0.3)
    assert isinstance(stack, LayerStack)
    names = [layer.name for layer in stack.layers]
    assert names[0] == "Base"  # flattened plate at the bottom
    assert names[-1] == "Frame & ornaments"  # frame on top
    # Each planned POI becomes its own layer.
    for poi in plan.pois:
        assert any(n == f"POI - {poi.name}" for n in names), poi.name
    # Labels become their own kind-prefixed layers.
    label_prefixes = ("Poi -", "Street -", "District -", "Water -", "Title -")
    assert any(n.startswith(label_prefixes) for n in names)
    # Every layer carries real content (cropped, non-empty) and a valid offset.
    w, h = stack.size
    for layer in stack.layers:
        assert layer.image.getbbox() is not None
        x, y = layer.offset
        assert 0 <= x <= w and 0 <= y <= h


def test_render_layers_flatten_resembles_render(plan, assets_dir, artifacts):
    """Flattening the layer stack must reassemble into a real poster (not a
    blank sheet), with land/water variation preserved."""
    c = Compositor(plan, assets_dir)
    flat = c.render_layers(scale=0.3).flatten().convert("RGB")
    artifacts.save("flattened_layers", flat)
    arr = np.asarray(flat)
    assert flat.size == (round(plan.canvas.width_px * 0.3), round(plan.canvas.height_px * 0.3))
    assert arr.std() > 10
    # The Base layer is opaque, so the flattened image has no transparent gaps.
    assert np.asarray(c.render_layers(scale=0.3).flatten())[:, :, 3].min() == 255


def test_render_unchanged_by_layer_refactor(plan, assets_dir):
    """The flat render path must be byte-identical across repeated calls and
    equal to base+finish (guards the _draw_* refactor)."""
    c = Compositor(plan, assets_dir)
    a = np.asarray(c.render(scale=0.3))
    b = np.asarray(c.apply_finish(c.render_base(scale=0.3), scale=0.3))
    assert np.array_equal(a, b)


# --- PSD writer -------------------------------------------------------------


@pytest.mark.parametrize("compression", ["rle", "raw"])
def test_write_psd_roundtrips_with_psd_tools(tmp_path, compression):
    psd_tools = pytest.importorskip("psd_tools")
    size = (120, 90)
    base = Image.new("RGBA", size, (200, 180, 140, 255))
    red = Image.new("RGBA", (30, 20), (255, 0, 0, 255))
    layers = [Layer("Base", base, (0, 0)), Layer("Spot - café", red, (40, 30))]

    out = tmp_path / "test.psd"
    write_psd(out, size, layers, compression=compression)
    assert out.exists()

    psd = psd_tools.PSDImage.open(out)
    assert psd.width == 120 and psd.height == 90
    assert [layer.name for layer in psd] == ["Base", "Spot - café"]  # bottom..top
    spot = psd[1]
    assert (spot.left, spot.top) == (40, 30)
    assert (spot.width, spot.height) == (30, 20)
    # The composite preview reassembles the layers: a red patch over tan base.
    composite = np.asarray(psd.composite().convert("RGB"))
    assert tuple(composite[40, 55]) == (255, 0, 0)  # inside the red patch
    assert tuple(composite[5, 5]) == (200, 180, 140)  # base elsewhere


def test_write_psd_layer_pixels_preserved(tmp_path):
    psd_tools = pytest.importorskip("psd_tools")
    size = (64, 48)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (48, 64, 4), dtype=np.uint8)
    noise[:, :, 3] = 255  # opaque so psd_tools returns it intact
    base = Image.fromarray(noise, "RGBA")
    out = write_psd(tmp_path / "noise.psd", size, [Layer("Base", base, (0, 0))], compression="rle")

    psd = psd_tools.PSDImage.open(out)
    back = np.asarray(psd[0].numpy("color") * 255).round().astype(np.uint8)
    assert np.array_equal(back, noise[:, :, :3])


def test_compose_layered_pipeline(plan, tmp_path):
    pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
    out = pipeline.compose_layered(plan, tmp_path, scale=0.3)
    assert out.exists() and out.suffix == ".psd"
    psd_tools = pytest.importorskip("psd_tools")
    psd = psd_tools.PSDImage.open(out)
    assert psd.width == round(plan.canvas.width_px * 0.3)
    assert len(list(psd)) >= 3  # at least base, a sprite/label, frame
