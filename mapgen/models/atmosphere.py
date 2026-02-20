"""Atmospheric perspective and terrain enhancement models."""

from typing import Optional

from pydantic import BaseModel, Field


class AtmosphereSettings(BaseModel):
    """Configuration for atmospheric perspective effects."""

    enabled: bool = Field(default=False, description="Enable atmospheric perspective")
    haze_color: str = Field(
        default="#C8D8E8",
        description="Color of atmospheric haze (hex)",
    )
    haze_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum haze opacity at the top of the map",
    )
    contrast_reduction: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="How much to reduce contrast in distant areas",
    )
    saturation_reduction: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="How much to desaturate distant areas",
    )
    gradient_curve: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="Curve exponent for the gradient (>1 = more effect near top)",
    )
