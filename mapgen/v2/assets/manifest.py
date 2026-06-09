"""Derive the asset manifest (what the AI must produce) from plan content."""

from __future__ import annotations

import re

from ..types import AssetKind, AssetSpec, GroundClass, PoiSlot, ScatterKind

SPRITE_SHEET_GRID = (3, 2)  # 6 variants per sheet

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
}


def poi_asset_id(poi_id: str) -> str:
    return f"poi_{slugify(poi_id)}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


def build_manifest(
    ground_classes: set[GroundClass],
    scatter_kinds: set[ScatterKind],
    pois: list[PoiSlot],
) -> list[AssetSpec]:
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
                prompt_hints=TEXTURE_HINTS[cls],
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
                prompt_hints=SPRITE_HINTS[kind],
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
