"""Core V2 types: the plan.json scene-graph contract.

The PlanDocument is the single contract between the plan engine, the
asset studio, and the compositor. All geometry inside a PlanDocument is
expressed in final poster pixel space (after distortion and the oblique
camera), so the compositor never performs coordinate transforms.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

Point = tuple[float, float]

PLAN_VERSION = "2.0"


class RegionBBox(BaseModel):
    """Geographic region in WGS84 degrees."""

    north: float
    south: float
    east: float
    west: float

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south

    @property
    def mid_latitude(self) -> float:
        return (self.north + self.south) / 2.0

    @property
    def width_km(self) -> float:
        import math

        return abs(self.width_deg) * 111.32 * math.cos(math.radians(self.mid_latitude))

    @property
    def height_km(self) -> float:
        return abs(self.height_deg) * 110.57

    @property
    def area_km2(self) -> float:
        return self.width_km * self.height_km


class CanvasSpec(BaseModel):
    """Output poster dimensions."""

    width_px: int = 7016
    height_px: int = 9933
    dpi: int = 300


class CameraSpec(BaseModel):
    """Oblique bird's-eye camera, applied to vectors in the plan engine.

    convergence: width of the top (far) edge relative to the bottom (near)
        edge; 1.0 = no convergence.
    vertical_scale: vertical foreshortening at the far edge relative to the
        near edge; 1.0 = no foreshortening.
    horizon_margin: fraction of the canvas height reserved above the far
        edge for horizon/sky/title.
    """

    convergence: float = Field(0.78, gt=0.0, le=1.0)
    vertical_scale: float = Field(0.55, gt=0.0, le=1.0)
    horizon_margin: float = Field(0.06, ge=0.0, lt=0.5)


class GroundClass(str, Enum):
    WATER = "water"
    PARK = "park"
    FOREST = "forest"
    SAND = "sand"
    FARMLAND = "farmland"
    URBAN = "urban"
    LAND = "land"  # base/default ground


class RoadClass(str, Enum):
    MOTORWAY = "motorway"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    LOCAL = "local"
    PATH = "path"
    RAIL = "rail"
    RIVER = "river"
    STREAM = "stream"


class GroundPolygon(BaseModel):
    cls: GroundClass
    exterior: list[Point]
    holes: list[list[Point]] = Field(default_factory=list)
    depth: float = 0.0  # 0 = near (bottom), 1 = far (horizon)


class RoadPath(BaseModel):
    cls: RoadClass
    points: list[Point]
    width_px: float
    name: Optional[str] = None
    depth: float = 0.0
    ref: Optional[str] = None  # route designation for shields, e.g. "I 10"


class BuildingFootprint(BaseModel):
    polygon: list[Point]
    height_px: float
    depth: float = 0.0


class PoiSlot(BaseModel):
    """A placed point of interest awaiting its illustrated sprite."""

    id: str
    name: str
    anchor: Point  # ground-contact point in poster px
    width_px: float
    height_px: float
    tier: int = Field(2, ge=1, le=3)  # 1 = hero, 3 = minor
    depth: float = 0.0
    asset_id: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # What the landmark physically is (building, campus, park, garden,
    # mountain, monument, airport, stadium, area, ...). Drives the sprite
    # prompt; "river" POIs never become slots at all (see PlanBuilder).
    feature_type: str = "building"
    # When a POI sits within a neighbour's sprite footprint, no smooth warp can
    # separate them, so the sprite is offset into open space and a connector is
    # drawn back to the true ground point. ``anchor`` always stays the sprite
    # base; ``leader_anchor`` (when ``offset``) is that true warped point.
    leader_anchor: Optional[Point] = None
    offset: bool = False


class ScatterKind(str, Enum):
    TREE = "tree"
    HOUSE = "house"
    BOAT = "boat"
    CACTUS = "cactus"
    SHRUB = "shrub"
    ROCK = "rock"


class ScatterSlot(BaseModel):
    kind: ScatterKind
    x: float
    y: float
    width_px: float  # desired on-poster sprite width (depth already applied)
    depth: float = 0.0
    variant: int = 0  # which sprite from the library sheet


class LabelKind(str, Enum):
    SHIELD = "shield"
    STREET = "street"
    DISTRICT = "district"
    POI = "poi"
    WATER = "water"
    TITLE = "title"


class LabelSpec(BaseModel):
    kind: LabelKind
    text: str
    baseline: list[Point]  # >= 2 points; straight or curved
    font_size_px: float
    priority: float = 0.5  # higher places first in greedy layout


class AssetKind(str, Enum):
    STYLE_BIBLE = "style_bible"
    TEXTURE = "texture"
    SPRITE_SHEET = "sprite_sheet"
    POI_SPRITE = "poi_sprite"
    ORNAMENT = "ornament"


class AssetSpec(BaseModel):
    """One unit of AI work for the asset studio."""

    id: str
    kind: AssetKind
    subject: str  # e.g. "water", "tree", "Empire State Building", "compass rose"
    width_px: int = 1024
    height_px: int = 1024
    prompt_hints: str = ""
    source_photo: Optional[str] = None  # path/URL for POI photo reference
    sheet_grid: Optional[tuple[int, int]] = None  # (cols, rows) for sprite sheets

    def content_hash(self) -> str:
        """Stable hash for asset caching.

        Includes the reference photo's *content* when it exists -- swapping
        the image behind an unchanged path must invalidate the cache."""
        payload = self.model_dump_json(exclude_none=True)
        digest = hashlib.sha256(payload.encode())
        if self.source_photo:
            try:
                digest.update(Path(self.source_photo).read_bytes())
            except OSError:
                pass
        return digest.hexdigest()[:16]


class StyleSpec(BaseModel):
    """Palette + finish parameters shared by asset prompts and compositor."""

    preset: str = "vintage_tourist"
    description: str = (
        "vintage hand-painted tourist map illustration, warm muted gouache "
        "palette, soft brushwork, 1950s travel poster feel"
    )
    palette: dict[str, str] = Field(
        default_factory=lambda: {
            "paper": "#f3e9d4",
            "land": "#ecdfc3",
            "urban": "#e7d5b5",
            "water": "#8fb8b2",
            "water_edge": "#5f8d88",
            "park": "#a8b87f",
            "forest": "#84996a",
            "sand": "#e8d49a",
            "farmland": "#d6c992",
            "road_fill": "#fdf7e2",
            "road_casing": "#8f6f45",
            "motorway_fill": "#e8b86d",
            "rail": "#8a7a66",
            "building_roof": "#cf8d62",
            "building_wall": "#b5774f",
            "outline": "#6b5337",
            "label": "#4a3a26",
            "haze": "#dce5e3",
            "shadow": "#3a2f22",
        }
    )
    wobble_px: float = 1.6  # hand-drawn line jitter amplitude at full scale
    paper_grain: float = 0.05  # 0..1 strength of paper texture overlay
    harmonize_strength: float = 0.5  # low-frequency AI mood blend (0 = off)

    # --- scene vocabulary (region character, not just colors) ---
    # What the style-bible swatch should depict. Keep it generic by default;
    # presets override it with regional scenery (no hard-coded coastlines).
    scene: str = "a road, a few local buildings, native trees and the region's typical terrain"
    # Scatter kinds sprinkled on bare base land (deserts get cacti, not
    # nothing). Values must be ScatterKind names.
    land_scatter: list[str] = Field(default_factory=list)
    # Scatter kind for open water; None disables (no sailboats on a desert
    # reservoir).
    water_scatter: Optional[str] = "boat"
    # Per-ground-class texture prompt overrides (e.g. WATER on a desert map
    # is a river, not "calm sea water").
    texture_hints: dict[str, str] = Field(default_factory=dict)
    # Per-scatter-kind sprite prompt overrides (e.g. TREE on a desert map is a
    # palo verde, not a lush park tree). Keyed by ScatterKind value.
    sprite_hints: dict[str, str] = Field(default_factory=dict)


class PlanDocument(BaseModel):
    """Complete scene graph for one poster, in poster pixel space."""

    version: str = PLAN_VERSION
    name: str = "Untitled Map"
    region: RegionBBox
    canvas: CanvasSpec = Field(default_factory=CanvasSpec)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    style: StyleSpec = Field(default_factory=StyleSpec)

    ground: list[GroundPolygon] = Field(default_factory=list)
    roads: list[RoadPath] = Field(default_factory=list)
    buildings: list[BuildingFootprint] = Field(default_factory=list)
    pois: list[PoiSlot] = Field(default_factory=list)
    scatter: list[ScatterSlot] = Field(default_factory=list)
    labels: list[LabelSpec] = Field(default_factory=list)
    manifest: list[AssetSpec] = Field(default_factory=list)

    # Source-data provenance (detail tier, per-layer outcomes) and
    # human-readable warnings -- a layer that failed to fetch must be
    # visible here, never silently absent from the poster.
    provenance: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # GeoFrame parameters (rotation, aspect extension) used to map geo
    # coordinates into this plan -- consumers adding geo features later must
    # reconstruct the same frame (ingest.GeoFrame.from_dict).
    frame: dict = Field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "PlanDocument":
        return cls.model_validate(json.loads(Path(path).read_text()))


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Darken (<1.0) or lighten (>1.0) a color, clamped to valid range."""
    return tuple(max(0, min(255, int(round(c * factor)))) for c in rgb)  # type: ignore[return-value]
