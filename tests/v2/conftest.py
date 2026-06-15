"""Shared fixtures: a synthetic coastal town as SourceData, plus the
visual-artifact harness (see `artifacts`)."""

import html
import os
from pathlib import Path

import pytest

from mapgen.v2.ingest import SourceData, SourcePlace, SourcePoi, SourcePolygon, SourceRoad
from mapgen.v2.types import CanvasSpec, GroundClass, RegionBBox, RoadClass

# --- visual artifact harness -------------------------------------------------
#
# Image-pipeline logic is hard to judge from asserts alone (seams, mattes,
# tiling order). When MAPGEN_TEST_ARTIFACTS=1, tests can save illustrative
# PNGs via the `artifacts` fixture; an index.html thumbnail gallery is
# regenerated at session end for at-a-glance review. Disabled (the default,
# and in CI) the fixture is a no-op, so tests call it unconditionally.

ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def _artifacts_enabled() -> bool:
    return os.environ.get("MAPGEN_TEST_ARTIFACTS") == "1"


class ArtifactSink:
    def __init__(self, directory: Path | None):
        self.directory = directory

    def save(self, name: str, image) -> None:
        """Save a PIL Image (or HxW / HxWx3 / HxWx4 numpy array) as <name>.png."""
        if self.directory is None:
            return
        import numpy as np
        from PIL import Image

        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype == bool:
                arr = arr.astype("uint8") * 255
            elif arr.dtype != "uint8":
                arr = np.clip(arr, 0, 255).astype("uint8")
            image = Image.fromarray(arr)
        self.directory.mkdir(parents=True, exist_ok=True)
        image.save(self.directory / f"{name}.png")


@pytest.fixture
def artifacts(request) -> ArtifactSink:
    if not _artifacts_enabled():
        return ArtifactSink(None)
    module = request.module.__name__.rsplit(".", 1)[-1]
    test = request.node.name.replace("/", "_")
    return ArtifactSink(ARTIFACTS_ROOT / module / test)


def pytest_sessionfinish(session, exitstatus):
    if not _artifacts_enabled() or not ARTIFACTS_ROOT.is_dir():
        return
    rows = []
    for module_dir in sorted(p for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()):
        cells = []
        for test_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
            for png in sorted(test_dir.glob("*.png")):
                rel = png.relative_to(ARTIFACTS_ROOT)
                label = html.escape(f"{test_dir.name} / {png.stem}")
                cells.append(
                    f'<figure><a href="{rel}"><img src="{rel}" loading="lazy"></a>'
                    f"<figcaption>{label}</figcaption></figure>"
                )
        if cells:
            rows.append(f"<h2>{html.escape(module_dir.name)}</h2>" + "".join(cells))
    (ARTIFACTS_ROOT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>mapgen test artifacts</title>"
        "<style>body{font-family:sans-serif;margin:2rem}figure{display:inline-block;"
        "margin:.5rem;text-align:center}img{max-width:280px;max-height:280px;"
        "border:1px solid #ccc}figcaption{font-size:.75rem;max-width:280px}</style>"
        "<h1>mapgen test artifacts</h1>" + "".join(rows)
    )


@pytest.fixture
def region() -> RegionBBox:
    return RegionBBox(north=40.80, south=40.70, east=-73.95, west=-74.05)


@pytest.fixture
def small_canvas() -> CanvasSpec:
    # Small canvas keeps tests fast; geometry logic is scale-free.
    return CanvasSpec(width_px=1000, height_px=1414, dpi=72)


@pytest.fixture
def source(region: RegionBBox) -> SourceData:
    """A fictional coastal town: bay to the east, park, river, roads, POIs."""
    west, east = region.west, region.east
    south, north = region.south, region.north

    def lon(f: float) -> float:
        return west + f * (east - west)

    def lat(f: float) -> float:
        return south + f * (north - south)

    return SourceData(
        region=region,
        roads=[
            SourceRoad(
                cls=RoadClass.MOTORWAY,
                name="Coast Highway",
                coords=[(lon(0.1), lat(0.05)), (lon(0.15), lat(0.4)), (lon(0.1), lat(0.95))],
            ),
            SourceRoad(
                cls=RoadClass.PRIMARY,
                name="Main Street",
                coords=[(lon(0.1), lat(0.5)), (lon(0.45), lat(0.52)), (lon(0.7), lat(0.5))],
            ),
            SourceRoad(
                cls=RoadClass.SECONDARY,
                name="Harbor Road",
                coords=[(lon(0.4), lat(0.2)), (lon(0.5), lat(0.45)), (lon(0.7), lat(0.6))],
            ),
            SourceRoad(
                cls=RoadClass.LOCAL,
                coords=[(lon(0.3), lat(0.6)), (lon(0.35), lat(0.75))],
            ),
            SourceRoad(
                cls=RoadClass.RIVER,
                name="Silver River",
                coords=[(lon(0.0), lat(0.8)), (lon(0.4), lat(0.7)), (lon(0.75), lat(0.55))],
            ),
        ],
        ground=[
            SourcePolygon(
                cls=GroundClass.WATER,
                name="East Bay",
                exterior=[
                    (lon(0.75), lat(0.0)),
                    (lon(1.0), lat(0.0)),
                    (lon(1.0), lat(1.0)),
                    (lon(0.75), lat(1.0)),
                    (lon(0.7), lat(0.5)),
                ],
            ),
            SourcePolygon(
                cls=GroundClass.PARK,
                name="Town Park",
                exterior=[
                    (lon(0.2), lat(0.55)),
                    (lon(0.4), lat(0.55)),
                    (lon(0.4), lat(0.72)),
                    (lon(0.2), lat(0.72)),
                ],
            ),
            SourcePolygon(
                cls=GroundClass.URBAN,
                exterior=[
                    (lon(0.1), lat(0.3)),
                    (lon(0.6), lat(0.3)),
                    (lon(0.6), lat(0.5)),
                    (lon(0.1), lat(0.5)),
                ],
            ),
        ],
        buildings=[
            [(lon(0.30), lat(0.40)), (lon(0.33), lat(0.40)), (lon(0.33), lat(0.43)), (lon(0.30), lat(0.43))],
            [(lon(0.45), lat(0.35)), (lon(0.49), lat(0.35)), (lon(0.49), lat(0.38)), (lon(0.45), lat(0.38))],
        ],
        pois=[
            SourcePoi(id="lighthouse", name="Old Lighthouse", latitude=lat(0.25), longitude=lon(0.72), tier=1),
            SourcePoi(id="museum", name="Maritime Museum", latitude=lat(0.45), longitude=lon(0.5), tier=2),
            SourcePoi(id="market", name="Fish Market", latitude=lat(0.47), longitude=lon(0.52), tier=3),
        ],
        places=[
            SourcePlace(name="Old Town", latitude=lat(0.4), longitude=lon(0.35), kind="district"),
            SourcePlace(name="East Bay", latitude=lat(0.5), longitude=lon(0.87), kind="water"),
        ],
    )
