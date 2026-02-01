"""Configuration management for map generator."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application-level configuration."""

    # API Keys
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google API key for Gemini",
    )
    mapbox_access_token: Optional[str] = Field(
        default=None,
        description="Mapbox access token for satellite imagery",
    )

    # Directories
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "mapgen",
        description="Cache directory for downloaded data",
    )
    output_dir: Path = Field(
        default=Path.cwd() / "output",
        description="Default output directory",
    )

    # Generation defaults
    default_tile_size: int = Field(default=2048, description="Default tile size")
    default_overlap: int = Field(default=256, description="Default tile overlap")
    default_dpi: int = Field(default=300, description="Default output DPI")

    # Model settings
    gemini_model: str = Field(
        default="gemini-3-pro-image-preview",
        description="Gemini model for image generation",
    )

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment and defaults."""
        return cls(
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            mapbox_access_token=os.environ.get("MAPBOX_ACCESS_TOKEN"),
            cache_dir=Path(os.environ.get("MAPGEN_CACHE_DIR", str(cls.model_fields["cache_dir"].default))),
            output_dir=Path(os.environ.get("MAPGEN_OUTPUT_DIR", str(cls.model_fields["output_dir"].default))),
        )

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config
