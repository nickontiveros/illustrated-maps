"""The CompositionSpec: a per-poster, editable layout document.

The planner's layout used to be decided entirely by heuristics (plateau
clustering for the warp, generalization for feature selection, a global
``road_treatment`` flag, ``SIZE_BY_FEATURE`` for POIs). Those choices are
editorial and per-poster, so they belong to the user, not to a fixed rule.

A ``CompositionSpec`` (persisted as ``composition.json`` beside
``project.yaml``) captures those choices declaratively. ``PlanBuilder`` reads
it and *applies* it, falling back to the original heuristic for any section
left on ``auto``/empty. Crucially, an **absent or all-auto spec reproduces the
old plan exactly** -- the spec only ever overrides, never silently changes
defaults -- so adopting it is risk-free and a fresh project keeps working.

All geometry is authored in **normalized frame space** ``(u, v) in [0, 1]^2``
(see ``GeoFrame.to_normalized``), which is warp-independent: a warp region or a
POI nudge stays put as the rest of the layout is edited. Feature selection and
road routing key off stable feature ids, which are coordinate-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

COMPOSITION_FILENAME = "composition.json"


class WarpRegion(BaseModel):
    """A user-drawn rectangle to magnify (or, with magnify<1, compress)."""

    id: str
    label: Optional[str] = None
    bounds: tuple[float, float, float, float]  # (u0, v0, u1, v1), normalized
    magnify: float = 1.5  # target relative magnification of the region


class WarpSpec(BaseModel):
    # "auto"   -> the built-in plateau fit (current behavior)
    # "manual" -> build the warp from `regions` below
    # "off"    -> identity (no warp)
    mode: Literal["auto", "manual", "off"] = "auto"
    regions: list[WarpRegion] = Field(default_factory=list)


class LayerSelect(BaseModel):
    """Per-layer feature visibility. `default` is the baseline for features not
    named in either list; `include`/`exclude` are id overrides on top of it."""

    default: Literal["auto", "all", "none"] = "auto"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    def visible(self, feature_id: str, auto_visible: bool) -> bool:
        """Resolve visibility for one feature id given the heuristic's verdict."""
        if feature_id in self.exclude:
            return False
        if feature_id in self.include:
            return True
        if self.default == "all":
            return True
        if self.default == "none":
            return False
        return auto_visible


class FeatureSpec(BaseModel):
    roads: LayerSelect = Field(default_factory=LayerSelect)
    rivers: LayerSelect = Field(default_factory=LayerSelect)
    pois: LayerSelect = Field(default_factory=LayerSelect)
    places: LayerSelect = Field(default_factory=LayerSelect)


class RoadOverride(BaseModel):
    treatment: Literal["warped", "straight", "hidden"] = "warped"
    # Optional manual centerline in normalized space; drawn straight (unwarped).
    reshape: Optional[list[tuple[float, float]]] = None


class PoiOverride(BaseModel):
    size: Optional[float] = None  # multiplier on the tier width
    tier: Optional[int] = None
    offset_uv: Optional[tuple[float, float]] = None  # normalized nudge
    leader: Literal["auto", "force", "suppress"] = "auto"
    label_side: Literal["auto", "left", "right", "above", "below"] = "auto"


class LabelOverrides(BaseModel):
    title_anchor_uv: Optional[tuple[float, float]] = None
    # Hand-placed labels, keyed by "<kind>:<text>" (e.g. "poi:Phoenix Sky
    # Harbor", "district:Phoenix"); value is the normalized anchor the label
    # should move to. Matched against the generated labels after layout.
    overrides: dict[str, tuple[float, float]] = Field(default_factory=dict)


class CompositionSpec(BaseModel):
    version: str = "1.0"
    # "heuristics" until the user edits; lets a re-seed know it may overwrite.
    seeded_from: Literal["heuristics", "manual"] = "heuristics"

    warp: WarpSpec = Field(default_factory=WarpSpec)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    roads: dict[str, RoadOverride] = Field(default_factory=dict)  # keyed by road id
    pois: dict[str, PoiOverride] = Field(default_factory=dict)  # keyed by poi id
    labels: LabelOverrides = Field(default_factory=LabelOverrides)

    @classmethod
    def load(cls, path: str | Path) -> "CompositionSpec":
        return cls.model_validate(json.loads(Path(path).read_text()))

    @classmethod
    def load_or_default(cls, project_dir: str | Path) -> "CompositionSpec":
        """The on-disk spec for a project, or an all-auto default if none."""
        path = Path(project_dir) / COMPOSITION_FILENAME
        return cls.load(path) if path.exists() else cls()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1))
