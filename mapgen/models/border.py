"""Decorative border and marginalia models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BorderStyle(str, Enum):
    """Available border template styles."""

    VINTAGE_SCROLL = "vintage_scroll"
    ART_DECO = "art_deco"
    MODERN_MINIMAL = "modern_minimal"
    ORNATE_VICTORIAN = "ornate_victorian"


class LegendItem(BaseModel):
    """A single item in the map legend."""

    label: str
    color: str  # Hex color
    symbol: str = "rect"  # rect, circle, line, dashed


class BorderSettings(BaseModel):
    """Configuration for decorative borders and marginalia."""

    enabled: bool = Field(default=False, description="Enable decorative border")
    style: BorderStyle = Field(default=BorderStyle.VINTAGE_SCROLL)
    margin: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Border margin in pixels per side",
    )
    show_compass: bool = Field(default=True, description="Show compass rose")
    show_legend: bool = Field(default=True, description="Auto-generate legend")
    show_scale_bar: bool = Field(default=False, description="Show scale bar")
    legend_items: list[LegendItem] = Field(
        default_factory=list,
        description="Custom legend items (auto-generated if empty)",
    )
    border_color: Optional[str] = Field(
        default=None,
        description="Override border color (hex)",
    )
    background_color: Optional[str] = Field(
        default="#FFF8F0",
        description="Border background color",
    )
    ornament_opacity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Opacity of decorative ornaments",
    )
