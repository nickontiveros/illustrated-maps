"""Named style presets: palette + scene vocabulary per region character.

A project can say ``style: southwest_desert`` and get the full kit --
palette, style-bible scene, scatter vocabulary, texture hints -- instead
of inheriting coastal-city defaults everywhere. Unknown preset names fall
back to StyleSpec defaults (vintage_tourist), so presets are additive.
"""

from __future__ import annotations

from .types import StyleSpec

PRESETS: dict[str, dict] = {
    "vintage_tourist": {},  # the StyleSpec defaults
    "southwest_desert": {
        "description": (
            "vintage hand-painted southwestern desert tourist map illustration, "
            "warm terracotta, adobe and sage gouache palette, saguaro cactus "
            "country, red rock mesas, dusty sun-bleached tones, 1950s Arizona "
            "highway travel poster feel"
        ),
        "scene": (
            "a desert highway with a few adobe buildings, saguaro cacti, "
            "red rock mesas on the horizon and a dry wash"
        ),
        "palette": {
            "paper": "#f5ead2",
            "land": "#eed9ae",
            "urban": "#e6cda1",
            "water": "#7fb2a8",
            "water_edge": "#55867c",
            "park": "#b3b079",
            "forest": "#7e8f5a",
            "sand": "#eccf95",
            "farmland": "#d6c184",
            "road_fill": "#fdf3da",
            "road_casing": "#955f38",
            "motorway_fill": "#d98e54",
            "rail": "#8a7a66",
            "building_roof": "#c1704e",
            "building_wall": "#b07850",
            "outline": "#6e4f33",
            "label": "#4d3826",
            "haze": "#e7d3b8",
            "shadow": "#3f2e1f",
        },
        "land_scatter": ["cactus", "shrub", "rock"],
        "water_scatter": None,  # no sailboats on desert reservoirs
        "typography": {
            # Warm, slightly larger hand-lettering for the travel-poster feel;
            # district names in deep terracotta ink, streets a touch smaller.
            "scale": 1.05,
            "kinds": {
                "district": {"color": "#7a3b22"},
                "street": {"size": 0.008},
            },
        },
        "texture_hints": {
            "water": "calm desert river water, soft teal brush strokes",
            "land": "dry desert ground wash, warm sand tones, faint scrub stippling",
            "sand": "desert sand with sparse creosote stippling",
            "park": "dry scrubland green, patchy brush strokes",
        },
        "sprite_hints": {
            # Desert flora, not lush park trees; low adobe, not pitched roofs.
            # Phrased like the cactus hint (terse single object) so the model
            # lays out a grid of icons rather than one detailed illustration.
            "tree": "small palo verde or mesquite desert tree, sparse green-gold airy canopy, thin trunk, small shadow",
            "house": "single small adobe house with a flat roof, small shadow",
        },
    },
}


def resolve_style(value) -> StyleSpec:
    """Build a StyleSpec from a preset name or a config mapping.

    A bare string is a preset lookup; a mapping may carry a ``preset`` key
    whose kit provides defaults that explicit keys then override.
    """
    if isinstance(value, StyleSpec):
        return value
    if isinstance(value, str):
        return StyleSpec(preset=value, **PRESETS.get(value, {}))
    if isinstance(value, dict):
        preset = value.get("preset", "vintage_tourist")
        merged = {**PRESETS.get(preset, {}), **{k: v for k, v in value.items() if k != "preset"}}
        return StyleSpec(preset=preset, **merged)
    raise TypeError(f"Unsupported style config: {type(value)!r}")
