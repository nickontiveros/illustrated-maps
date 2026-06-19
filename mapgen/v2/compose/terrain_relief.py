"""DEM hillshade relief, warped to oblique poster space.

A grayscale hillshade of the region's terrain, registered through the exact
same oblique warp as the satellite base (see raster_warp). Multiplied under
the painted land in illustrated mode it gives mountains real shadow and form
-- the relief the Silicon-Desert-style references show -- without any 3D mesh.

Self-disabling: missing ``elevation`` library or flat terrain yields None and
the compositor simply skips the pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from ...models.project import BoundingBox
from ..ingest import GeoFrame
from ..plan.camera import ObliqueCamera
from ..plan.distortion import warp_from_dict
from .raster_warp import ObliqueRasterWarper

logger = logging.getLogger(__name__)

# Below this elevation range (meters) terrain reads as flat -- skip the pass
# rather than multiply by an all-lit, featureless hillshade.
_MIN_RELIEF_RANGE_M = 30.0


class TerrainReliefBuilder:
    def __init__(self, frame: GeoFrame, camera: ObliqueCamera, canvas, warp,
                 exaggeration: float = 2.0, cache_dir: Optional[Path] = None):
        self.frame = frame
        self.exaggeration = exaggeration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._env = frame.fetch_region
        self._warper = ObliqueRasterWarper(frame, camera, canvas, warp)

    @classmethod
    def from_plan(cls, plan, cache_dir: Optional[Path] = None):
        return cls(
            GeoFrame.from_dict(plan.frame),
            ObliqueCamera(plan.camera, plan.canvas),
            plan.canvas,
            warp_from_dict(plan.warp),
            exaggeration=plan.style.terrain_exaggeration,
            cache_dir=cache_dir,
        )

    def poster_relief(self, scale: float = 1.0) -> Optional[Image.Image]:
        """RGBA hillshade in poster space (gray RGB, trapezoid alpha), or None
        when terrain is flat / elevation data is unavailable."""
        hill = self._hillshade()
        if hill is None:
            return None
        return self._warper.warp_to_poster(hill, scale, trapezoid=True)

    def _hillshade(self) -> Optional[Image.Image]:
        try:
            from ...services.terrain_service import TerrainService
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("Terrain relief unavailable (import failed): %s", exc)
            return None
        bbox = BoundingBox(
            north=self._env.north,
            south=self._env.south,
            east=self._env.east,
            west=self._env.west,
        )
        try:
            service = TerrainService(cache_dir=str(self.cache_dir) if self.cache_dir else None)
            elevation = service.fetch_elevation_data(bbox)
        except Exception as exc:
            logger.warning("Elevation fetch failed; skipping terrain relief: %s", exc)
            return None
        if elevation.elevation_range < _MIN_RELIEF_RANGE_M:
            logger.info("Terrain relief skipped: region is effectively flat")
            return None
        shade = service.compute_hillshade(elevation, vertical_exaggeration=self.exaggeration)
        # compute_hillshade returns a uint8 HxW array, north-up over the bbox.
        return Image.fromarray(shade, mode="L").convert("RGB")
