from pathlib import Path

from mapgen.v2.compose.text import resolve_font_path
from mapgen.v2.plan.labels import FONT_FRACTIONS, plan_labels
from mapgen.v2.styles import resolve_style
from mapgen.v2.types import (
    CanvasSpec,
    LabelKind,
    RoadClass,
    RoadPath,
    StyleSpec,
    Typography,
)


def _street_size(typography: Typography | None) -> float:
    canvas = CanvasSpec(width_px=1000, height_px=1400)
    road = RoadPath(
        cls=RoadClass.PRIMARY,
        name="Main Street",
        points=[(100.0, 1000.0), (900.0, 1000.0)],
        width_px=10,
    )
    labels = plan_labels(canvas, [road], [], [], [], title="T", typography=typography)
    street = next(l for l in labels if l.kind == LabelKind.STREET)
    return street.font_size_px


def test_default_typography_reproduces_builtin_sizes():
    # The regression guard: no typography == the old FONT_FRACTIONS sizing.
    expected = FONT_FRACTIONS[LabelKind.STREET] * 1000
    assert _street_size(None) == expected
    assert _street_size(Typography()) == expected


def test_global_scale_multiplies_every_size():
    base = _street_size(None)
    assert _street_size(Typography(scale=2.0)) == base * 2.0


def test_per_kind_size_override():
    sized = _street_size(Typography(kinds={"street": {"size": 0.02}}))
    assert sized == 0.02 * 1000


def test_per_kind_size_override_composes_with_scale():
    sized = _street_size(Typography(scale=1.5, kinds={"street": {"size": 0.02}}))
    assert sized == 0.02 * 1.5 * 1000


def test_stylespec_roundtrips_nested_typography_dict():
    style = resolve_style(
        {
            "preset": "vintage_tourist",
            "typography": {
                "font": "DejaVu Serif",
                "scale": 1.3,
                "kinds": {"title": {"color": "#123456", "halo": "none"}},
            },
        }
    )
    assert isinstance(style, StyleSpec)
    assert style.typography.font == "DejaVu Serif"
    assert style.typography.scale == 1.3
    title = style.typography.for_kind(LabelKind.TITLE)
    assert title.color == "#123456" and title.halo == "none"


def test_preset_carries_typography():
    style = resolve_style("southwest_desert")
    assert style.typography.scale == 1.05
    assert style.typography.for_kind(LabelKind.DISTRICT).color == "#7a3b22"


def test_resolve_font_path_finds_bundled_font_by_name():
    path = resolve_font_path("Caveat")
    assert path is not None and Path(path).exists()
    assert "Caveat" in Path(path).name


def test_resolve_font_path_passes_through_existing_path():
    from mapgen.v2.compose.text import BUNDLED_FONT

    assert resolve_font_path(str(BUNDLED_FONT)) == str(BUNDLED_FONT)


def test_resolve_font_path_none_for_missing_and_empty():
    assert resolve_font_path(None) is None
    assert resolve_font_path("ThisFontDoesNotExist12345") is None
