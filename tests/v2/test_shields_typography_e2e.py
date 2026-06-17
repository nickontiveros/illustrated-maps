"""Offline end-to-end: a synthetic region with classified highways and a
custom typography style renders all the way to a poster image, exercising the
network -> RoadPath -> LabelSpec -> render_shield path and the typography wiring
without any OSM fetch or AI assets."""

import numpy as np

from mapgen.v2.assets.studio import AssetStudio
from mapgen.v2.assets.stub import StubAssetGenerator
from mapgen.v2.compose.compositor import Compositor
from mapgen.v2.compose.shields import shield_asset_id
from mapgen.v2.ingest import SourceData, SourcePlace, SourcePoi, SourceRoad
from mapgen.v2.plan.builder import PlanBuilder
from mapgen.v2.styles import resolve_style
from mapgen.v2.types import AssetKind, CanvasSpec, LabelKind, RegionBBox, RoadClass


def _source() -> SourceData:
    region = RegionBBox(north=34.0, south=33.0, east=-111.0, west=-112.5)
    w, e, s, n = region.west, region.east, region.south, region.north

    def lon(f):
        return w + f * (e - w)

    def lat(f):
        return s + f * (n - s)

    return SourceData(
        region=region,
        # Three long highways at well-separated latitudes so their (midpoint)
        # shields don't collide and all survive greedy placement.
        roads=[
            SourceRoad(
                cls=RoadClass.MOTORWAY, name="Maricopa Freeway", ref="I 10", network="US:I",
                coords=[(lon(0.05), lat(0.22)), (lon(0.95), lat(0.22))],
            ),
            SourceRoad(
                cls=RoadClass.PRIMARY, name="Grand Avenue", ref="US 60", network="US:US",
                coords=[(lon(0.05), lat(0.5)), (lon(0.95), lat(0.5))],
            ),
            SourceRoad(
                cls=RoadClass.PRIMARY, name="Beeline Highway", ref="AZ 87", network="US:AZ",
                coords=[(lon(0.05), lat(0.78)), (lon(0.95), lat(0.78))],
            ),
        ],
        places=[
            SourcePlace(name="Phoenix", latitude=lat(0.95), longitude=lon(0.15), population=1_600_000),
        ],
        pois=[
            SourcePoi(id="capitol", name="State Capitol", latitude=lat(0.05), longitude=lon(0.85), tier=1),
        ],
    )


def test_full_render_with_shields_and_typography(tmp_path, artifacts):
    canvas = CanvasSpec(width_px=1000, height_px=1400, dpi=72)
    style = resolve_style("southwest_desert")  # carries a typography block
    plan = PlanBuilder(canvas=canvas, style=style).build(_source(), title="Arizona")

    # Shields reached the plan, each carrying its OSM network for artwork lookup.
    shields = [l for l in plan.labels if l.kind == LabelKind.SHIELD]
    networks = {l.network for l in shields}
    assert {"US:I", "US:US", "US:AZ"} <= networks

    # Typography from the preset is on the plan's style.
    assert plan.style.typography.scale == 1.05

    # Each placed network with a committed reference becomes a SHIELD asset in
    # the manifest, primed with its blank-shield reference image.
    shield_assets = {s.id: s for s in plan.manifest if s.kind == AssetKind.SHIELD}
    for net in ("US:I", "US:US", "US:AZ"):
        spec = shield_assets[shield_asset_id(net)]
        assert spec.source_photo and spec.source_photo.endswith(".png")

    # Generate the shield sprites offline (stub studio) and render. The
    # compositor must use the generated sprite tier for these networks.
    studio = AssetStudio(StubAssetGenerator(), tmp_path)
    paths = studio.generate_all(plan)
    for net in ("US:I", "US:US", "US:AZ"):
        assert shield_asset_id(net) in paths

    image = Compositor(plan, assets_dir=tmp_path).render(scale=0.25)
    artifacts.save("arizona_shields", image)
    assert image.mode == "RGB"
    arr = np.asarray(image)
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert arr.std() > 5  # actually drew content, not a flat fill
