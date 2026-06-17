from PIL import Image

from mapgen.v2.compose.shields import (
    cached_shield_reference,
    classify_shield,
    compose_ai_shield,
    reference_title,
    render_shield,
    shield_asset_id,
    shield_network_slug,
)
from mapgen.v2.plan.labels import plan_labels
from mapgen.v2.types import CanvasSpec, LabelKind, RoadClass, RoadPath


def test_classify_prefers_network_tag():
    assert classify_shield("I 10", "US:I").kind == "interstate"
    assert classify_shield("US 60", "US:US").kind == "us"
    az = classify_shield("AZ 87", "US:AZ")
    assert az.kind == "state" and az.state == "AZ" and az.number == "87"
    loop = classify_shield("Loop 101", "US:AZ:Loop")
    assert loop.kind == "state" and loop.state == "AZ"


def test_classify_falls_back_to_ref_prefix():
    assert classify_shield("I-10", None).kind == "interstate"
    assert classify_shield("US 60", None).kind == "us"
    assert classify_shield("AZ 87", None).kind == "state"


def test_classify_cleans_compound_and_qualified_refs():
    # ";"-joined keeps the primary route; "Future" qualifier is dropped.
    assert classify_shield("US 60;AZ 77", None).number == "60"
    az = classify_shield("Future AZ 24", None)
    assert az.kind == "state" and az.number == "24"


def test_classify_unknown_is_generic_badge():
    c = classify_shield("weird thing", None)
    assert c.kind == "generic" and c.text


def test_render_shield_returns_rgba_sprite_for_every_tier():
    for ref, net in [("I 10", "US:I"), ("US 60", "US:US"), ("AZ 87", "US:AZ"), ("weird", None)]:
        sprite = render_shield(ref, net, size=24)
        assert isinstance(sprite, Image.Image)
        assert sprite.mode == "RGBA"
        assert sprite.width > 0 and sprite.height > 0


def test_render_shield_never_raises_on_empty_input():
    assert render_shield("", None, size=24).mode == "RGBA"


def test_reference_title_maps_networks_to_commons_files():
    assert reference_title("US:I") == "File:I-blank.svg"
    assert reference_title("US:US") == "File:US blank.svg"
    assert reference_title("US:AZ") == "File:Arizona blank.svg"
    assert reference_title("US:CA") == "File:California blank.svg"
    # Sub-network qualifiers (Loop, Business) still resolve to the state blank.
    assert reference_title("US:AZ:Loop") == "File:Arizona blank.svg"
    # Unknown / non-US networks have no rule -> procedural fallback.
    assert reference_title("US:ZZ") is None
    assert reference_title("GB:M-road") is None
    assert reference_title(None) is None


def test_shield_asset_id_and_slug():
    assert shield_network_slug("US:AZ") == "us-az"
    assert shield_asset_id("US:I") == "shield_us-i"


def test_cached_reference_offline_for_uncommitted_network():
    # No committed ref for Wyoming and we must not hit the network here.
    assert cached_shield_reference("US:WY") is None


def test_committed_example_references_exist():
    # The desert example's references are committed so offline runs work.
    for net in ("US:I", "US:US", "US:AZ"):
        path = cached_shield_reference(net)
        assert path is not None and path.exists()


def test_compose_ai_shield_overlays_number_on_sprite():
    import numpy as np

    blank = Image.new("RGBA", (120, 120), (255, 255, 255, 255))
    out = compose_ai_shield(blank, "AZ 87", "US:AZ", size=48)
    assert out.mode == "RGBA" and out.height == int(48 * 1.5)
    # The number was drawn: dark pixels now exist on the white sprite.
    assert np.asarray(out)[..., :3].min() < 80


def test_shield_prompt_demands_blank_keyed_sign():
    from mapgen.v2.assets.gemini_client import build_prompt
    from mapgen.v2.types import AssetKind, AssetSpec, StyleSpec

    spec = AssetSpec(id="shield_us-az", kind=AssetKind.SHIELD, subject="US:AZ",
                     source_photo="ref.png", width_px=512, height_px=512)
    prompt = build_prompt(spec, StyleSpec()).lower()
    assert "blank" in prompt
    assert "no route number" in prompt or "no" in prompt and "number" in prompt
    assert "magenta" in prompt  # flat key for matting
    assert "reference" in prompt  # uses the attached blank-shield image


def test_plan_labels_propagates_network_to_shield():
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    roads = [
        RoadPath(
            cls=RoadClass.MOTORWAY,
            points=[(100, 200), (900, 200)],
            width_px=8,
            ref="I 10",
            network="US:I",
        ),
    ]
    labels = plan_labels(canvas, roads, [], [], [], title="T")
    shields = [l for l in labels if l.kind is LabelKind.SHIELD]
    assert len(shields) == 1
    assert shields[0].network == "US:I"
