"""Shared dependencies for API endpoints."""

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from ..config import get_config
from ..services.gemini_service import GeminiService
from ..services.satellite_service import SatelliteService


@dataclass
class APIKeys:
    """API keys extracted from request headers or server config."""

    google_api_key: Optional[str] = None
    mapbox_access_token: Optional[str] = None


def get_api_keys(request: Request) -> APIKeys:
    """Extract API keys from request headers, falling back to env vars / config."""
    config = get_config()
    return APIKeys(
        google_api_key=request.headers.get("X-Google-API-Key") or config.google_api_key,
        mapbox_access_token=request.headers.get("X-Mapbox-Access-Token") or config.mapbox_access_token,
    )


def create_gemini_service(keys: APIKeys) -> GeminiService:
    """Create a GeminiService using the provided API keys."""
    return GeminiService(api_key=keys.google_api_key)


def create_satellite_service(keys: APIKeys, cache_dir=None) -> SatelliteService:
    """Create a SatelliteService using the provided API keys."""
    return SatelliteService(access_token=keys.mapbox_access_token, cache_dir=cache_dir)
