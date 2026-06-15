import numpy as np
import pytest
from PIL import Image

from mapgen.v2.assets.gemini_client import build_prompt
from mapgen.v2.assets.manifest import build_manifest, poi_asset_id, slugify
from mapgen.v2.assets.matting import KEY_COLOR, key_to_alpha, trim_to_content
from mapgen.v2.assets.studio import AssetStudio, load_sprites_from_sheet
from mapgen.v2.assets.stub import StubAssetGenerator
from mapgen.v2.assets.textures import edge_seam_error, ensure_tileable, make_tileable
from mapgen.v2.types import AssetKind, AssetSpec, GroundClass, PoiSlot, ScatterKind, StyleSpec


def test_slugify():
    assert slugify("Empire State Building!") == "empire_state_building"
    assert slugify("---") == "x"


def test_manifest_contents():
    pois = [PoiSlot(id="esb", name="Empire State", anchor=(0, 0), width_px=400, height_px=460)]
    manifest = build_manifest({GroundClass.WATER, GroundClass.LAND}, {ScatterKind.TREE}, pois)
    ids = {s.id for s in manifest}
    assert "style_bible" in ids
    assert "texture_water" in ids and "texture_land" in ids
    assert "sprites_tree" in ids
    assert poi_asset_id("esb") in ids
    assert "ornament_compass" in ids
    # Style bible must come first when sorted the way the studio sorts.
    specs = sorted(manifest, key=lambda s: s.kind != AssetKind.STYLE_BIBLE)
    assert specs[0].id == "style_bible"


def test_poi_sprite_oversamples_footprint():
    pois = [PoiSlot(id="big", name="Big", anchor=(0, 0), width_px=900, height_px=1000)]
    manifest = build_manifest(set(), set(), pois)
    spec = next(s for s in manifest if s.kind == AssetKind.POI_SPRITE)
    assert spec.width_px == 1800


def test_matting_removes_key_and_keeps_content(artifacts):
    img = Image.new("RGB", (50, 50), KEY_COLOR)
    img.paste((120, 80, 40), (10, 10, 40, 40))
    rgba = key_to_alpha(img)
    artifacts.save("keyed_input", img)
    artifacts.save("matted", rgba)
    arr = np.asarray(rgba)
    assert arr[0, 0, 3] == 0  # key corner transparent
    assert arr[25, 25, 3] == 255  # content opaque


def test_trim_to_content():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (40, 40, 60, 60))
    trimmed = trim_to_content(img, padding=2)
    assert trimmed.width <= 24 and trimmed.height <= 24


def test_make_tileable_reduces_seam_error():
    rng = np.random.default_rng(0)
    # A gradient image: maximally non-tileable.
    grad = np.linspace(40, 215, 128)
    arr = np.clip(
        np.tile(grad[None, :, None], (128, 1, 3)) + rng.normal(0, 3, (128, 128, 3)), 0, 255
    ).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    before = edge_seam_error(img)
    after = edge_seam_error(make_tileable(img))
    assert after < before * 0.1


def _blotchy_texture(size: int = 128, seed: int = 3) -> Image.Image:
    """Smooth blotchy noise, shaped like a real painted-wash texture."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=8)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    arr = (120 + 80 * noise).astype(np.uint8)
    return Image.fromarray(np.dstack([arr, arr, arr]), "RGB")


def test_tiled_mosaic_join_is_invisible(artifacts):
    """Tiling two copies side by side must not produce a visible seam."""
    tileable = make_tileable(_blotchy_texture())
    mosaic = Image.new("RGB", (tileable.width * 2, tileable.height))
    mosaic.paste(tileable, (0, 0))
    mosaic.paste(tileable, (tileable.width, 0))
    artifacts.save("mosaic_2x1", mosaic)
    arr = np.asarray(tileable, dtype=np.float32)
    w = arr.shape[1]
    join_diff = np.abs(arr[:, -1] - arr[:, 0]).mean()  # column W-1 abuts column 0
    typical_diff = np.abs(np.diff(arr, axis=1)).mean()
    assert join_diff <= typical_diff * 3


def test_make_tileable_leaves_no_edge_bands():
    """The repair must not blur bands along the borders (the 'ghost grid').

    Local contrast in the border zone should stay comparable to the
    interior; a cross-fade repair fails this by averaging the edges.
    """
    tileable = make_tileable(_blotchy_texture())
    grad = np.abs(np.diff(np.asarray(tileable, dtype=np.float32).mean(axis=2), axis=1))
    w = grad.shape[1]
    border = np.concatenate([grad[:, : w // 8], grad[:, -w // 8 :]], axis=1).mean()
    interior = grad[:, 3 * w // 8 : 5 * w // 8].mean()
    assert border > interior * 0.6


def test_ensure_tileable_passthrough_for_good_textures():
    img = Image.new("RGB", (64, 64), (100, 120, 90))
    assert edge_seam_error(img) == 0
    out = ensure_tileable(img)
    assert np.array_equal(np.asarray(out), np.asarray(img.convert("RGB")))


def test_stub_studio_end_to_end(tmp_path):
    pois = [PoiSlot(id="esb", name="Empire State", anchor=(0, 0), width_px=400, height_px=460)]
    manifest = build_manifest({GroundClass.WATER}, {ScatterKind.TREE}, pois)
    from mapgen.v2.types import PlanDocument, RegionBBox

    plan = PlanDocument(
        region=RegionBBox(north=1, south=0, east=1, west=0),
        manifest=manifest,
        pois=pois,
    )
    studio = AssetStudio(StubAssetGenerator(), tmp_path)
    paths = studio.generate_all(plan)
    assert set(paths) == {s.id for s in manifest}
    # Sprites are RGBA after matting; textures RGB.
    tree = Image.open(paths["sprites_tree"])
    assert tree.mode == "RGBA"
    water = Image.open(paths["texture_water"])
    assert water.mode == "RGB"
    assert edge_seam_error(water) < 12.0

    # Caching: second run does not regenerate (generator that explodes).
    class Exploder:
        def generate(self, *a, **k):
            raise AssertionError("should not be called")

    cached = AssetStudio(Exploder(), tmp_path).generate_all(plan)
    assert cached.keys() == paths.keys()


def test_sheet_cutting(tmp_path):
    spec = AssetSpec(
        id="sprites_tree", kind=AssetKind.SPRITE_SHEET, subject="tree",
        width_px=300, height_px=200, sheet_grid=(3, 2),
    )
    img = StubAssetGenerator().generate(spec, StyleSpec())
    rgba = key_to_alpha(img)
    path = tmp_path / "sheet.png"
    rgba.save(path)
    sprites = load_sprites_from_sheet(path, (3, 2))
    assert len(sprites) == 6
    assert all(s.mode == "RGBA" for s in sprites)


def test_prompts_enforce_conventions():
    style = StyleSpec()
    sprite_spec = AssetSpec(id="sprites_tree", kind=AssetKind.SPRITE_SHEET, subject="tree", sheet_grid=(3, 2))
    poi_spec = AssetSpec(id="poi_x", kind=AssetKind.POI_SPRITE, subject="Old Lighthouse")
    texture_spec = AssetSpec(id="texture_water", kind=AssetKind.TEXTURE, subject="water")
    for spec in (sprite_spec, poi_spec):
        prompt = build_prompt(spec, style)
        assert "magenta" in prompt
        assert "no text" in prompt.lower()
        assert "oblique" in prompt
    assert "tileable" in build_prompt(texture_spec, style)


def test_content_hash_changes_with_spec():
    a = AssetSpec(id="x", kind=AssetKind.TEXTURE, subject="water")
    b = AssetSpec(id="x", kind=AssetKind.TEXTURE, subject="park")
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() == AssetSpec(id="x", kind=AssetKind.TEXTURE, subject="water").content_hash()


def test_force_regenerates(tmp_path):
    spec = AssetSpec(id="texture_water", kind=AssetKind.TEXTURE, subject="water")
    from mapgen.v2.types import PlanDocument, RegionBBox

    plan = PlanDocument(region=RegionBBox(north=1, south=0, east=1, west=0), manifest=[spec])
    calls = []

    class Counter:
        def generate(self, spec, style, style_reference=None):
            calls.append(spec.id)
            return StubAssetGenerator().generate(spec, style, style_reference)

    studio = AssetStudio(Counter(), tmp_path)
    studio.generate_all(plan)
    studio.generate_all(plan)
    assert len(calls) == 1
    studio.generate_all(plan, force=True)
    assert len(calls) == 2
