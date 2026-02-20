"""Road styling models for enhanced road treatment."""

from typing import Optional

from pydantic import BaseModel, Field


class RoadStyleSettings(BaseModel):
    """Configuration for enhanced road rendering."""

    enabled: bool = Field(default=False, description="Enable enhanced road styling")

    # Width exaggeration factors per road class (multiplier on true geographic width)
    motorway_exaggeration: float = Field(default=20.0, ge=1.0, le=50.0)
    primary_exaggeration: float = Field(default=15.0, ge=1.0, le=50.0)
    secondary_exaggeration: float = Field(default=10.0, ge=1.0, le=30.0)
    residential_exaggeration: float = Field(default=5.0, ge=1.0, le=20.0)

    # Colors per road class (overrides RenderService defaults when set)
    motorway_color: Optional[str] = Field(default="#F5F0E0", description="Motorway fill color")
    primary_color: Optional[str] = Field(default="#FFFDE8", description="Primary road fill color")
    secondary_color: Optional[str] = Field(default="#FFFFFF", description="Secondary road fill color")
    residential_color: Optional[str] = Field(default="#FAFAFA", description="Residential road fill color")
    outline_color: Optional[str] = Field(default="#B0A090", description="Road outline color")

    # Organic wobble for hand-drawn feel
    wobble_amount: float = Field(
        default=1.5,
        ge=0.0,
        le=5.0,
        description="Perlin noise displacement amplitude in pixels for hand-drawn feel",
    )
    wobble_frequency: float = Field(
        default=0.02,
        ge=0.005,
        le=0.1,
        description="Frequency of wobble undulation",
    )

    # Overlay options
    overlay_on_output: bool = Field(
        default=False,
        description="Composite road layer on top of illustrated output",
    )
    overlay_opacity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Opacity of road overlay on final output",
    )
    reference_opacity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Opacity of road layer in reference sent to Gemini",
    )

    # Presets
    preset: Optional[str] = Field(
        default=None,
        description="Named preset: vintage_tourist, modern_clean, ink_sketch",
    )


# Named presets
ROAD_STYLE_PRESETS = {
    "vintage_tourist": RoadStyleSettings(
        enabled=True,
        motorway_exaggeration=20.0,
        primary_exaggeration=15.0,
        secondary_exaggeration=10.0,
        residential_exaggeration=5.0,
        motorway_color="#F5E6C8",
        primary_color="#FAF0D0",
        secondary_color="#FFF8E8",
        residential_color="#FFFEF5",
        outline_color="#8B7355",
        wobble_amount=2.0,
    ),
    "modern_clean": RoadStyleSettings(
        enabled=True,
        motorway_exaggeration=18.0,
        primary_exaggeration=12.0,
        secondary_exaggeration=8.0,
        residential_exaggeration=4.0,
        motorway_color="#FFFFFF",
        primary_color="#F8F8F8",
        secondary_color="#F0F0F0",
        residential_color="#E8E8E8",
        outline_color="#CCCCCC",
        wobble_amount=0.0,
    ),
    "ink_sketch": RoadStyleSettings(
        enabled=True,
        motorway_exaggeration=15.0,
        primary_exaggeration=10.0,
        secondary_exaggeration=7.0,
        residential_exaggeration=3.0,
        motorway_color="#F0F0F0",
        primary_color="#E8E8E8",
        secondary_color="#E0E0E0",
        residential_color="#D8D8D8",
        outline_color="#2A2A2A",
        wobble_amount=3.0,
        wobble_frequency=0.03,
    ),
}
