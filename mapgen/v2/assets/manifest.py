"""Derive the asset manifest (what the AI must produce) from plan content."""

from __future__ import annotations

import re

from ..types import AssetKind, AssetSpec, GroundClass, PoiSlot, ScatterKind, StyleSpec

SPRITE_SHEET_GRID = (4, 4)  # 16 variants per sheet (less obvious tiling)

TEXTURE_HINTS: dict[GroundClass, str] = {
    GroundClass.WATER: "calm sea water, soft wave strokes",
    GroundClass.PARK: "grassy parkland, soft meadow strokes",
    GroundClass.FOREST: "dense leafy forest canopy from above",
    GroundClass.SAND: "dry sandy ground, light stippling",
    GroundClass.FARMLAND: "patchwork farm fields from above",
    GroundClass.URBAN: "subtle urban block fill, faint paving tone",
    GroundClass.LAND: "plain paper-toned ground wash",
}

SPRITE_HINTS: dict[ScatterKind, str] = {
    ScatterKind.TREE: "single park tree, round leafy canopy with small shadow",
    ScatterKind.HOUSE: "small generic town house with pitched roof",
    ScatterKind.BOAT: "small sailing boat seen from above-oblique",
    ScatterKind.CACTUS: "single saguaro cactus with arms, small shadow",
    ScatterKind.SHRUB: "small desert shrub or creosote bush, low and scrubby",
    ScatterKind.ROCK: "small red rock outcrop or boulder cluster",
}

# Per-feature-type guidance for POI sprites. "building" stays empty so plain
# building POIs keep their pre-existing prompts (and cached assets). Types
# not listed here fall back to building treatment. "river" never reaches the
# manifest (drawn as a waterway ribbon by the plan builder).
POI_TYPE_HINTS: dict[str, str] = {
    "building": "",
    "mountain": (
        "this is a natural mountain or rock formation: paint the actual landform "
        "-- ridges, cliffs, talus slopes and foothill vegetation seen from afar. "
        "Even if the name suggests an animal or object, NEVER draw an animal or "
        "object; draw the real geological landform that carries that name"
    ),
    "park": (
        "this is a park or natural area: paint a small landscape vignette of its "
        "signature scenery -- terrain and native vegetation, at most one tiny "
        "iconic structure -- wider than tall, not a building"
    ),
    "forest": (
        "this is a forest or natural preserve: paint a small stand of its "
        "characteristic trees or fossil landscape as a landscape vignette, "
        "not a building"
    ),
    "garden": (
        "this is a botanical garden: paint a lush cluster of its signature "
        "plants around one small pavilion or path, a landscape vignette rather "
        "than a plain building"
    ),
    "campus": (
        "this is a university campus: paint a tight cluster of its two or three "
        "most iconic buildings rendered as one compact group"
    ),
    "airport": (
        "this is an airport: paint the terminal with its control tower and one "
        "tiny aircraft beside it as a single compact group"
    ),
    "stadium": "this is a stadium: paint the full stadium bowl seen from outside",
    "monument": (
        "this is a monument or historic ruin: paint the actual ruin/monument "
        "structure in its terrain, not a modern building"
    ),
    "zoo": (
        "this is a zoo: paint its iconic entrance with one or two tiny animal "
        "figures, as a single compact group"
    ),
    "area": (
        "this is a district or area: paint its single most iconic structure "
        "or scene as one compact vignette"
    ),
}


def poi_asset_id(poi_id: str) -> str:
    return f"poi_{slugify(poi_id)}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


def build_manifest(
    ground_classes: set[GroundClass],
    scatter_kinds: set[ScatterKind],
    pois: list[PoiSlot],
    style: "StyleSpec | None" = None,
) -> list[AssetSpec]:
    texture_overrides = style.texture_hints if style is not None else {}
    sprite_overrides = style.sprite_hints if style is not None else {}
    manifest: list[AssetSpec] = [
        AssetSpec(
            id="style_bible",
            kind=AssetKind.STYLE_BIBLE,
            subject="style reference swatch",
            width_px=1024,
            height_px=1024,
        )
    ]

    for cls in sorted(ground_classes, key=lambda c: c.value):
        manifest.append(
            AssetSpec(
                id=f"texture_{cls.value}",
                kind=AssetKind.TEXTURE,
                subject=cls.value,
                prompt_hints=texture_overrides.get(cls.value, TEXTURE_HINTS[cls]),
                width_px=1024,
                height_px=1024,
            )
        )

    for kind in sorted(scatter_kinds, key=lambda k: k.value):
        manifest.append(
            AssetSpec(
                id=f"sprites_{kind.value}",
                kind=AssetKind.SPRITE_SHEET,
                subject=kind.value,
                prompt_hints=sprite_overrides.get(kind.value, SPRITE_HINTS[kind]),
                width_px=1536,
                height_px=1024,
                sheet_grid=SPRITE_SHEET_GRID,
            )
        )

    for slot in pois:
        # Oversample the printed footprint ~2x, capped at the model ceiling.
        target = int(min(2048, max(768, slot.width_px * 2)))
        manifest.append(
            AssetSpec(
                id=poi_asset_id(slot.id),
                kind=AssetKind.POI_SPRITE,
                subject=slot.name,
                prompt_hints=POI_TYPE_HINTS.get(slot.feature_type, ""),
                source_photo=None,  # filled by the builder when a photo exists
                width_px=target,
                height_px=int(target * 1.15),
            )
        )

    manifest.append(
        AssetSpec(
            id="ornament_compass",
            kind=AssetKind.ORNAMENT,
            subject="compass rose",
            width_px=768,
            height_px=768,
        )
    )
    manifest.append(
        AssetSpec(
            id="ornament_cartouche",
            kind=AssetKind.ORNAMENT,
            subject="decorative title cartouche frame, empty center",
            width_px=1536,
            height_px=768,
        )
    )
    return manifest
