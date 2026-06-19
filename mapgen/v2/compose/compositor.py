"""Deterministic full-resolution poster renderer.

Renders a PlanDocument bottom-up: paper -> ground textures with painterly
edges -> roads with wobble and tapered casings -> 2.5D building fabric ->
scattered sprites -> POI sprites with shadows -> atmospheric depth
gradient -> labels -> frame/ornaments -> paper grain.

A ``scale`` parameter renders the same plan at reduced size (previews,
tests); all geometry is scaled numerically, so a 10% render is a faithful
miniature of the full poster.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..assets.studio import load_sprites_from_sheet
from ..types import (
    AssetKind,
    GroundClass,
    LabelKind,
    PlanDocument,
    Point,
    RoadClass,
    ScatterKind,
    hex_to_rgb,
    shade,
)
from .shields import compose_ai_shield, render_shield, shield_asset_id
from .text import draw_text_on_path, load_font, resolve_font_path
from .wobble import wobble_polyline, wobble_ring

logger = logging.getLogger(__name__)

ROAD_DRAW_ORDER = [
    RoadClass.RAIL,
    RoadClass.PATH,
    RoadClass.LOCAL,
    RoadClass.SECONDARY,
    RoadClass.PRIMARY,
    RoadClass.MOTORWAY,
]
WATERWAY_CLASSES = (RoadClass.RIVER, RoadClass.STREAM)


@dataclass
class Layer:
    """One element of a layered export: a cropped RGBA bitmap and the
    canvas position (left, top) of its top-left corner."""

    name: str
    image: Image.Image
    offset: tuple[int, int] = (0, 0)


@dataclass
class LayerStack:
    """An ordered (bottom-to-top) set of `Layer`s plus the canvas size they
    were composed against -- the input to a PSD / layered-PNG writer."""

    size: tuple[int, int]
    layers: list[Layer] = field(default_factory=list)

    def flatten(self) -> Image.Image:
        """Composite the stack into a single RGBA image (sanity check that the
        layers reassemble into the poster, and the merged preview for writers
        that need one)."""
        out = Image.new("RGBA", self.size, (0, 0, 0, 0))
        for layer in self.layers:
            out.alpha_composite(layer.image.convert("RGBA"), layer.offset)
        return out


def _label_layer_name(label) -> str:
    """A readable, kind-prefixed layer name for a text label."""
    text = " ".join((label.text or "").split())
    if len(text) > 40:
        text = text[:39] + "…"
    prefix = label.kind.value.capitalize()
    return f"{prefix} - {text}" if text else prefix


class Compositor:
    def __init__(self, plan: PlanDocument, assets_dir: Path | str, font_path: str | None = None):
        self.plan = plan
        self.assets_dir = Path(assets_dir)
        self.font_path = font_path
        self.palette = {k: hex_to_rgb(v) for k, v in plan.style.palette.items()}
        self._sprite_cache: dict[str, list[Image.Image]] = {}
        self._sat_base_cache: dict[float, Image.Image | None] = {}
        self._relief_cache: dict[float, Image.Image | None] = {}

    @property
    def _satellite_mode(self) -> bool:
        return self.plan.style.base_mode == "satellite"

    def _satellite_base(self, scale: float) -> Image.Image | None:
        """The warped, poster-space satellite base for this scale, or None when
        not in satellite mode / no Mapbox token / fetch failed (the caller then
        falls back to the painted illustrated land)."""
        if not self._satellite_mode:
            return None
        if scale not in self._sat_base_cache:
            from .satellite_base import SatelliteBaseBuilder

            img = None
            try:
                builder = SatelliteBaseBuilder.from_plan(
                    self.plan, cache_dir=self.assets_dir / "satellite"
                )
                img = builder.poster_base(scale)
            except Exception as exc:  # never let the base sink a whole poster
                logger.warning("Satellite base build failed; using illustrated land: %s", exc)
                img = None
            if img is None:
                logger.warning("Satellite base unavailable; falling back to illustrated land")
            self._sat_base_cache[scale] = img
        return self._sat_base_cache[scale]

    def _terrain_relief(self, scale: float) -> Image.Image | None:
        """Warped DEM hillshade for this scale, or None when disabled / flat /
        elevation data unavailable (the relief pass is then skipped)."""
        if not self.plan.style.terrain_relief:
            return None
        if scale not in self._relief_cache:
            from .terrain_relief import TerrainReliefBuilder

            img = None
            try:
                builder = TerrainReliefBuilder.from_plan(
                    self.plan, cache_dir=self.assets_dir / "terrain"
                )
                img = builder.poster_relief(scale)
            except Exception as exc:
                logger.warning("Terrain relief build failed; skipping: %s", exc)
                img = None
            self._relief_cache[scale] = img
        return self._relief_cache[scale]

    def _apply_terrain_relief(self, canvas: Image.Image, scale: float) -> None:
        """Multiply the painted land by the hillshade so slopes gain shadow.
        No-op unless terrain_relief is enabled and the DEM has real range."""
        relief = self._terrain_relief(scale)
        if relief is None:
            return
        from PIL import ImageChops

        strength = self.plan.style.hillshade_strength
        base_rgb = canvas.convert("RGB")
        multiplied = ImageChops.multiply(base_rgb, relief.convert("RGB"))
        blended = Image.blend(base_rgb, multiplied, strength)
        # Confine to the map trapezoid (relief alpha); paper/sky stay untouched.
        canvas.paste(blended, (0, 0), relief.getchannel("A"))

    # -- asset access ------------------------------------------------------

    def _asset(self, asset_id: str) -> Image.Image | None:
        path = self.assets_dir / f"{asset_id}.png"
        if not path.exists():
            return None
        return Image.open(path)

    def _texture(self, cls: GroundClass) -> Image.Image | None:
        img = self._asset(f"texture_{cls.value}")
        return img.convert("RGB") if img else None

    def _sprites(self, kind: ScatterKind) -> list[Image.Image]:
        key = f"sprites_{kind.value}"
        if key not in self._sprite_cache:
            spec = next((s for s in self.plan.manifest if s.id == key), None)
            path = self.assets_dir / f"{key}.png"
            if spec is None or not path.exists():
                self._sprite_cache[key] = []
            else:
                self._sprite_cache[key] = load_sprites_from_sheet(path, spec.sheet_grid or (3, 2))
        return self._sprite_cache[key]

    # -- main entry ----------------------------------------------------------

    def render(self, scale: float = 1.0) -> Image.Image:
        return self.apply_finish(self.render_base(scale), scale)

    def render_base(
        self, scale: float = 1.0, sprites: bool = True, atmosphere: bool = True
    ) -> Image.Image:
        """Everything below the labels: the repaintable map surface.

        Split from `apply_finish` so a tiled AI repaint can transform the
        painted ground/fabric while labels, frame, and grain are composited
        afterwards at native resolution (they must never be repainted).

        ``sprites=False`` omits the scatter and POI passes; ``atmosphere=False``
        omits the distance-haze gradient. The layered export uses both to build
        a clean ground/road/building "plate" while the sprites and the haze are
        peeled into their own editable layers (see `render_layers`).
        """
        plan = self.plan
        width = max(1, int(round(plan.canvas.width_px * scale)))
        height = max(1, int(round(plan.canvas.height_px * scale)))
        wobble = self._wobble_amp(scale)

        canvas = Image.new("RGBA", (width, height), self.palette["paper"] + (255,))
        logger.info("Compositing %dx%d (scale %.2f)", width, height, scale)
        self._set_wavelength(scale)

        sat_base = self._satellite_base(scale)
        if sat_base is not None:
            # Photoreal base replaces the painted land + ground textures; the
            # ground pass still inks water/coast edges so shorelines read.
            self._draw_base_land(canvas, scale)  # paper backdrop under the trapezoid
            canvas.alpha_composite(sat_base)
            self._draw_ground(canvas, scale, wobble, edges_only=True)
        else:
            self._draw_base_land(canvas, scale)
            self._draw_ground(canvas, scale, wobble)
            # DEM hillshade multiplies the painted land (mountains gain form).
            # Satellite imagery already carries real relief, so this is
            # illustrated-mode only and applied before roads/buildings.
            self._apply_terrain_relief(canvas, scale)
        self._draw_waterways(canvas, scale, wobble)
        self._draw_roads(canvas, scale, wobble)
        self._draw_buildings(canvas, scale)
        if sprites:
            # Painted scatter (trees, houses) double up on real imagery, so the
            # land-dressing pass is suppressed when a satellite base is shown.
            if sat_base is None:
                self._draw_scatter(canvas, scale)
            self._draw_pois(canvas, scale)
        if atmosphere:
            self._draw_atmosphere(canvas, scale)
        return canvas

    def apply_finish(self, canvas: Image.Image, scale: float = 1.0) -> Image.Image:
        """Labels, frame/ornaments, and paper grain over a rendered base.

        `scale` must describe the canvas (labels and frame metrics derive
        from it), not the scale the base happened to be rendered at -- the
        repaint path renders the base at 0.5 and finishes at 1.0 after
        upscaling.
        """
        canvas = canvas.convert("RGBA")
        self._set_wavelength(scale)
        self._draw_labels(canvas, scale)
        self._draw_frame(canvas, scale)
        self._apply_grain(canvas)
        return canvas.convert("RGB")

    def render_layers(self, scale: float = 1.0) -> LayerStack:
        """Render the poster as a stack of named, positioned layers for a
        layered-graphics (PSD) export, bottom-to-top:

          * **Base** -- flattened ground/water textures, roads and buildings.
            The hard-to-edit surface is deliberately pre-merged into this one
            plate.
          * **Scatter - <kind>** -- one layer per scatter sprite kind
            (trees, boats, cacti, ...), so a whole class can be hidden,
            recoloured or thinned at once. Each is a flat raster, not separable
            sprites.
          * **POI leaders** -- the connector lines, kept under the sprites.
          * **POI - <name>** -- one layer per landmark sprite (its halo,
            drop-shadow and art), individually movable.
          * **Haze** -- the distance-haze gradient, as a separate overlay above
            the map content (matching the flat render order) so its strength
            can be dialed back or masked off; sits below the labels.
          * **<Kind> - <text>** -- one layer per text label.
          * **Frame & ornaments** -- the border and compass.

        Sprites and haze are peeled out of the base; paper grain is omitted as a
        finish best applied to a flattened copy. The standard `render()` PNG
        remains the canonical print output -- this stack is for hand-editing.
        """
        width = max(1, int(round(self.plan.canvas.width_px * scale)))
        height = max(1, int(round(self.plan.canvas.height_px * scale)))
        size = (width, height)
        self._set_wavelength(scale)
        layers: list[Layer] = []

        # Drop to RGB: the base is the opaque backdrop, and render_base leaves
        # antialiased polygon edges at sub-255 alpha (a paste-with-mask quirk
        # the flat render hides by ending on convert("RGB")); keeping that
        # alpha would punch faint holes in the export's bottom plate.
        base = self.render_base(scale=scale, sprites=False, atmosphere=False).convert("RGB")
        layers.append(Layer("Base", base, (0, 0)))

        # Scatter: one layer per kind, first-seen order kept stable.
        kinds: list[ScatterKind] = []
        for slot in self.plan.scatter:
            if slot.kind not in kinds:
                kinds.append(slot.kind)
        for kind in kinds:
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            self._draw_scatter(layer, scale, kinds={kind})
            self._append_cropped(layers, f"Scatter - {kind.value}", layer)

        # POI leaders sit under every sprite (a pre-pass in the flat render).
        leaders = Image.new("RGBA", size, (0, 0, 0, 0))
        self._draw_poi_leaders(leaders, scale)
        self._append_cropped(layers, "POI leaders", leaders)

        # One layer per POI sprite, back-to-front so nearer sprites stack on top.
        for slot in sorted(self.plan.pois, key=lambda s: s.anchor[1]):
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            self._draw_poi_sprites(layer, scale, [slot])
            self._append_cropped(layers, f"POI - {slot.name}", layer)

        # Distance haze as its own overlay over the map content (the flat render
        # composites it after the sprites). On a transparent canvas the
        # atmosphere pass leaves just the haze gradient.
        haze = Image.new("RGBA", size, (0, 0, 0, 0))
        self._draw_atmosphere(haze, scale)
        self._append_cropped(layers, "Haze", haze)

        # One layer per text label.
        typo = self.plan.style.typography
        global_font = resolve_font_path(typo.font) or self.font_path
        for label in self.plan.labels:
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            self._draw_one_label(layer, label, scale, typo, global_font)
            self._append_cropped(layers, _label_layer_name(label), layer)

        frame = Image.new("RGBA", size, (0, 0, 0, 0))
        self._draw_frame(frame, scale)
        self._append_cropped(layers, "Frame & ornaments", frame)

        return LayerStack(size=size, layers=layers)

    @staticmethod
    def _append_cropped(layers: list[Layer], name: str, image: Image.Image) -> None:
        """Crop a full-canvas layer to its painted content and append it; an
        empty layer (nothing drawn) is skipped so the export has no blanks."""
        bbox = image.getbbox()
        if bbox is None:
            return
        layers.append(Layer(name, image.crop(bbox), (bbox[0], bbox[1])))

    def render_class_mask(self, classes: set[GroundClass], scale: float = 1.0) -> Image.Image:
        """Mode-"L" mask (255 = inside) of where the given ground classes are
        visible, replaying the ground draw order with the same wobble seeds
        as `render_base` so mask and render agree pixel-for-pixel.

        Later polygons erase earlier ones exactly as they occlude them in
        the render. For WATER the waterway strokes are included; roads drawn
        over water (bridges) are not subtracted -- callers using the mask for
        statistics tolerate that sliver.
        """
        width = max(1, int(round(self.plan.canvas.width_px * scale)))
        height = max(1, int(round(self.plan.canvas.height_px * scale)))
        wobble = self._wobble_amp(scale)
        self._set_wavelength(scale)

        mask = Image.new("L", (width, height), 0)
        mdraw = ImageDraw.Draw(mask)
        for poly, ring, holes in self._ground_rings(scale, wobble):
            if len(ring) < 3:
                continue
            mdraw.polygon(ring, fill=255 if poly.cls in classes else 0)
            for hole in holes:
                if len(hole) >= 3:
                    mdraw.polygon(hole, fill=0)
        if GroundClass.WATER in classes:
            for i, road in enumerate(r for r in self.plan.roads if r.cls in WATERWAY_CLASSES):
                pts = wobble_polyline(
                    self._scaled(road.points, scale), wobble * 1.5, self._wavelength * 1.5, seed=i * 7.7
                )
                w = max(2, int(road.width_px * scale)) + max(2, int(4 * scale))
                mdraw.line(pts, fill=255, width=w, joint="curve")
                r_cap = w / 2
                for p in (pts[0], pts[-1]):
                    mdraw.ellipse([p[0] - r_cap, p[1] - r_cap, p[0] + r_cap, p[1] + r_cap], fill=255)
        return mask

    def _wobble_amp(self, scale: float) -> float:
        return self.plan.style.wobble_px * max(0.35, scale)

    def _set_wavelength(self, scale: float) -> None:
        # Wobble wavelength tracks the poster, not absolute pixels, so a
        # preview render and the full-res render crinkle identically.
        self._wavelength = max(20.0, self.plan.canvas.width_px * 0.012 * scale)

    def _ground_rings(self, scale: float, wobble: float):
        """Yield (polygon, wobbled exterior ring, wobbled hole rings) in draw
        order. Single source of the ring geometry so `_draw_ground` and
        `render_class_mask` can never diverge."""
        ordered = sorted(self.plan.ground, key=lambda g: g.cls != GroundClass.WATER)
        for idx, poly in enumerate(ordered):
            ring = wobble_ring(
                self._scaled(poly.exterior, scale), wobble * 2, self._wavelength * 2, seed=idx * 3.1
            )
            holes = [
                wobble_ring(self._scaled(h, scale), wobble * 2, self._wavelength * 2, seed=idx * 3.1 + 1)
                for h in poly.holes
            ]
            yield poly, ring, holes

    # -- layers --------------------------------------------------------------

    def _draw_base_land(self, canvas: Image.Image, scale: float) -> None:
        """The map plate: land fill below the horizon line, trapezoid-shaped
        to match the camera's converging far edge."""
        plan = self.plan
        horizon = plan.camera.horizon_margin * plan.canvas.height_px * scale
        w = canvas.width
        far_half = plan.camera.convergence * w / 2
        cx = w / 2
        plate = [
            (cx - far_half, horizon),
            (cx + far_half, horizon),
            (w, canvas.height),
            (0.0, canvas.height),
        ]
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        texture = self._texture(GroundClass.LAND)
        draw.polygon(plate, fill=self.palette["land"] + (255,))
        if texture is not None:
            mask = Image.new("L", canvas.size, 0)
            ImageDraw.Draw(mask).polygon(plate, fill=255)
            layer.paste(self._tile_texture(texture, canvas.size, scale), (0, 0), mask)
        canvas.alpha_composite(layer)

    def _tile_texture(
        self,
        texture: Image.Image,
        size: tuple[int, int],
        scale: float,
        offset: tuple[int, int] = (0, 0),
        canvas_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Fill a region with a texture without readable repetition.

        Even a perfectly seamless tile repeats its features at the tile
        period, which the eye picks up as a grid. Two countermeasures:
        the texture is laid down at two incommensurate periods (golden
        ratio apart) and blended, and a canvas-wide low-frequency tone
        field modulates the result so no period survives intact.

        ``offset`` is the region's position on the canvas: tiling and the
        tone field are anchored to canvas coordinates, so patches rendered
        separately (per-polygon bounding boxes) line up seamlessly.
        """
        tile_size = max(64, int(texture.width * max(0.25, scale)))
        base = self._tile_plain(texture, size, tile_size, offset)
        overlay = self._tile_plain(texture, size, max(64, int(tile_size * 1.618)), offset)
        out = Image.blend(base, overlay, 0.4)
        return self._tone_variation(out, offset, canvas_size or size)

    def _tile_plain(
        self,
        texture: Image.Image,
        size: tuple[int, int],
        tile_size: int,
        offset: tuple[int, int] = (0, 0),
    ) -> Image.Image:
        tile = texture.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        out = Image.new("RGB", size)
        start_x = -(offset[0] % tile_size)
        start_y = -(offset[1] % tile_size)
        for y in range(start_y, size[1], tile_size):
            for x in range(start_x, size[0], tile_size):
                out.paste(tile, (x, y))
        return out

    def _tone_field(self, canvas_size: tuple[int, int]) -> Image.Image:
        """Canvas-wide seeded luminance field (L mode), cached per size."""
        cached = getattr(self, "_tone_field_cache", None)
        if cached is None or cached[0] != canvas_size:
            rng = np.random.default_rng(11)
            coarse = rng.uniform(0.0, 255.0, (9, 7)).astype(np.uint8)
            field = Image.fromarray(coarse, "L").resize(canvas_size, Image.Resampling.BICUBIC)
            self._tone_field_cache = (canvas_size, field)
        return self._tone_field_cache[1]

    def _tone_variation(
        self,
        img: Image.Image,
        offset: tuple[int, int],
        canvas_size: tuple[int, int],
        amount: float = 0.045,
    ) -> Image.Image:
        """Seeded large-scale luminance variation, like uneven washes.

        Processed in row stripes: a full-poster float32 copy is ~840 MB and
        rapid allocations that size can OOM a WSL VM."""
        field = self._tone_field(canvas_size).crop(
            (offset[0], offset[1], offset[0] + img.width, offset[1] + img.height)
        )
        out = np.asarray(img.convert("RGB")).copy()
        field_arr = np.asarray(field)
        stripe = 1024
        for y in range(0, out.shape[0], stripe):
            sl = slice(y, min(y + stripe, out.shape[0]))
            factor = 1.0 + amount * (field_arr[sl].astype(np.float32) / 127.5 - 1.0)
            block = out[sl].astype(np.float32) * factor[..., None]
            out[sl] = np.clip(block, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    def _scaled(self, points: list[Point], scale: float) -> list[Point]:
        return [(x * scale, y * scale) for x, y in points]

    def _draw_ground(
        self, canvas: Image.Image, scale: float, wobble: float, edges_only: bool = False
    ) -> None:
        # Water first, then vegetation/land classes on top (see _ground_rings).
        # All mask/texture work happens on the polygon's bounding box, not the
        # full canvas: a state-scale plan has hundreds of ground polygons, and
        # full-canvas layers per polygon (~1.7 GB transient each at print
        # size) OOM-kill a WSL VM long before they finish.
        #
        # ``edges_only`` (satellite base mode) skips the texture/colour fill and
        # keeps only the painterly water/coast edge ink, so shorelines stay
        # legible over the photoreal imagery.
        for poly, ring, holes in self._ground_rings(scale, wobble):
            if len(ring) < 3:
                continue
            if not edges_only:
                pad = 4
                x0 = max(0, int(min(p[0] for p in ring)) - pad)
                y0 = max(0, int(min(p[1] for p in ring)) - pad)
                x1 = min(canvas.width, int(max(p[0] for p in ring)) + pad)
                y1 = min(canvas.height, int(max(p[1] for p in ring)) + pad)
                if x1 <= x0 or y1 <= y0:
                    continue
                box_size = (x1 - x0, y1 - y0)
                local_ring = [(x - x0, y - y0) for x, y in ring]

                mask = Image.new("L", box_size, 0)
                mdraw = ImageDraw.Draw(mask)
                mdraw.polygon(local_ring, fill=255)
                for hole_ring in holes:
                    if len(hole_ring) >= 3:
                        mdraw.polygon([(x - x0, y - y0) for x, y in hole_ring], fill=0)

                texture = self._texture(poly.cls)
                fill_color = self.palette.get(poly.cls.value, self.palette["land"])
                if texture is not None:
                    patch = self._tile_texture(
                        texture, box_size, scale, offset=(x0, y0), canvas_size=canvas.size
                    )
                else:
                    patch = Image.new("RGB", box_size, fill_color)
                canvas.paste(patch, (x0, y0), mask)

            # Painterly edge: darkened wobbly outline, heavier for water.
            # Urban/land fills are subtle tone shifts, not washes -- no edge.
            if poly.cls in (GroundClass.URBAN, GroundClass.LAND):
                continue
            water = poly.cls == GroundClass.WATER
            # Over photoreal imagery only shorelines get inked; vegetation
            # outlines would fight the real textures.
            if edges_only and not water:
                continue
            edge_key = "water_edge" if water else "outline"
            edge_width = max(1, int((3.0 if water else 2.0) * scale * 2))
            alpha = 140 if water else 90
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.line(ring + ring[:1], fill=self.palette[edge_key] + (alpha,), width=edge_width, joint="curve")

    def _draw_waterways(self, canvas: Image.Image, scale: float, wobble: float) -> None:
        draw = ImageDraw.Draw(canvas, "RGBA")
        for i, road in enumerate(r for r in self.plan.roads if r.cls in WATERWAY_CLASSES):
            pts = wobble_polyline(self._scaled(road.points, scale), wobble * 1.5, self._wavelength * 1.5, seed=i * 7.7)
            width = max(2, int(road.width_px * scale))
            draw.line(pts, fill=self.palette["water_edge"] + (255,), width=width + max(2, int(4 * scale)), joint="curve")
            draw.line(pts, fill=self.palette["water"] + (255,), width=width, joint="curve")
            self._round_caps(draw, pts, width, self.palette["water"])

    def _draw_roads(self, canvas: Image.Image, scale: float, wobble: float) -> None:
        draw = ImageDraw.Draw(canvas, "RGBA")
        order = {cls: i for i, cls in enumerate(ROAD_DRAW_ORDER)}
        roads = [r for r in self.plan.roads if r.cls not in WATERWAY_CLASSES]
        roads.sort(key=lambda r: order.get(r.cls, 0))
        casing = self.palette["road_casing"]
        for i, road in enumerate(roads):
            pts = wobble_polyline(self._scaled(road.points, scale), wobble, self._wavelength, seed=i * 1.3)
            if len(pts) < 2:
                continue
            width = max(1, int(road.width_px * scale))
            if road.cls == RoadClass.RAIL:
                self._draw_rail(draw, pts, width, scale)
                continue
            fill = self.palette["motorway_fill"] if road.cls == RoadClass.MOTORWAY else self.palette["road_fill"]
            casing_w = width + max(2, int(width * 0.6))
            draw.line(pts, fill=casing + (255,), width=casing_w, joint="curve")
            self._round_caps(draw, pts, casing_w, casing)
            draw.line(pts, fill=fill + (255,), width=width, joint="curve")
            self._round_caps(draw, pts, width, fill)

    def _draw_rail(self, draw: ImageDraw.ImageDraw, pts: list[Point], width: int, scale: float) -> None:
        rail = self.palette["rail"]
        draw.line(pts, fill=rail + (255,), width=max(1, width), joint="curve")
        # Cross ties.
        tie_len = width * 3.0
        spacing = max(8.0, 22 * scale)
        walked = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg == 0:
                continue
            n = ((b[1] - a[1]) / seg, -(b[0] - a[0]) / seg)
            d = walked % spacing
            while d < seg:
                f = d / seg
                cx, cy = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                draw.line(
                    [(cx - n[0] * tie_len / 2, cy - n[1] * tie_len / 2), (cx + n[0] * tie_len / 2, cy + n[1] * tie_len / 2)],
                    fill=rail + (255,),
                    width=max(1, int(width * 0.6)),
                )
                d += spacing
            walked = (walked + seg) % spacing

    def _round_caps(self, draw: ImageDraw.ImageDraw, pts: list[Point], width: int, color: tuple[int, int, int]) -> None:
        r = width / 2
        for p in (pts[0], pts[-1]):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color + (255,))

    def _draw_buildings(self, canvas: Image.Image, scale: float) -> None:
        """2.5D extrusion of footprints: roof shifted up, visible side faces.

        In satellite base mode the roof is the real imagery under the footprint
        (sampled from the warped base) lifted to the roof height, so rooftops
        match the ground; walls stay solid-shaded (top-down imagery has none)."""
        draw = ImageDraw.Draw(canvas, "RGBA")
        wall = self.palette["building_wall"]
        roof = self.palette["building_roof"]
        outline = self.palette["outline"]
        sat_base = self._satellite_base(scale)
        for b in sorted(self.plan.buildings, key=lambda b: max(p[1] for p in b.polygon)):
            base = self._scaled(b.polygon, scale)
            if len(base) < 3:
                continue
            lift = b.height_px * scale
            top = [(x, y - lift) for x, y in base]
            # South-facing edges (normal pointing toward the viewer) get walls.
            n = len(base)
            for i in range(n):
                p1, p2 = base[i], base[(i + 1) % n]
                if p2[0] < p1[0]:  # viewer-facing edge for clockwise rings
                    quad = [p1, p2, (p2[0], p2[1] - lift), (p1[0], p1[1] - lift)]
                    edge_shade = 0.78 + 0.18 * abs(p2[0] - p1[0]) / (math.hypot(p2[0] - p1[0], p2[1] - p1[1]) or 1)
                    draw.polygon(quad, fill=shade(wall, edge_shade) + (255,), outline=outline + (200,))
            if sat_base is not None and self._paste_satellite_roof(canvas, sat_base, base, lift):
                # Faint outline so the roof reads against busy imagery.
                draw.polygon(top, outline=outline + (180,))
            else:
                draw.polygon(top, fill=shade(roof, 1.0 - 0.25 * b.depth) + (255,), outline=outline + (220,))

    def _paste_satellite_roof(
        self, canvas: Image.Image, sat_base: Image.Image, base: list[Point], lift: float
    ) -> bool:
        """Lift the satellite imagery under a footprint up to the roof.

        The base is top-down, so the pixels beneath the footprint *are* the
        roof; copying them up by ``lift`` keeps the roof registered to the
        ground. Returns False (caller draws a palette roof) when the footprint
        falls outside the imagery."""
        x0 = int(min(p[0] for p in base))
        y0 = int(min(p[1] for p in base))
        x1 = int(max(p[0] for p in base)) + 1
        y1 = int(max(p[1] for p in base)) + 1
        x0c, y0c = max(0, x0), max(0, y0)
        x1c = min(sat_base.width, x1)
        y1c = min(sat_base.height, y1)
        if x1c <= x0c or y1c <= y0c:
            return False
        patch = sat_base.crop((x0c, y0c, x1c, y1c))
        mask = Image.new("L", patch.size, 0)
        ImageDraw.Draw(mask).polygon([(x - x0c, y - y0c) for x, y in base], fill=255)
        # Don't carry the base's trapezoid alpha into the roof -- a footprint at
        # the map edge would punch a hole; the footprint mask is the only mask.
        canvas.paste(patch.convert("RGB"), (x0c, int(y0c - lift)), mask)
        return True

    def _draw_scatter(
        self, canvas: Image.Image, scale: float, kinds: set[ScatterKind] | None = None
    ) -> None:
        for slot in sorted(self.plan.scatter, key=lambda s: s.y):
            if kinds is not None and slot.kind not in kinds:
                continue
            sprites = self._sprites(slot.kind)
            if not sprites:
                continue
            sprite = sprites[slot.variant % len(sprites)]
            # Per-instance variation, deterministic from position, so the same
            # handful of sheet variants don't read as obvious tiling: mirror
            # half of them and nudge each one's size.
            h = (int(slot.x * 13.7) ^ int(slot.y * 7.3)) & 0xFFFF
            if h & 1:
                sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
            jitter = 0.78 + (h >> 1 & 0xFF) / 255.0 * 0.5  # 0.78..1.28
            target_w = max(4, int(slot.width_px * scale * jitter))
            target_h = max(4, int(sprite.height * target_w / max(1, sprite.width)))
            resized = sprite.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x = int(slot.x * scale - target_w / 2)
            y = int(slot.y * scale - target_h)  # anchor at the base
            canvas.alpha_composite(resized, (x, y))

    def _draw_pois(self, canvas: Image.Image, scale: float) -> None:
        # Leaders first, as a pre-pass, so every sprite paints on top of every
        # connector (a connector must never cover a neighbour's sprite).
        self._draw_poi_leaders(canvas, scale)
        self._draw_poi_sprites(canvas, scale, self.plan.pois)

    def _draw_poi_leaders(self, canvas: Image.Image, scale: float) -> None:
        outline_rgb = self.palette["outline"]
        leader = ImageDraw.Draw(canvas, "RGBA")
        lw = max(1, int(self.plan.canvas.width_px * scale * 0.0012))
        dot_r = max(2, int(self.plan.canvas.width_px * scale * 0.0035))
        for slot in self.plan.pois:
            if not (slot.offset and slot.leader_anchor is not None):
                continue
            x0, y0 = slot.anchor[0] * scale, slot.anchor[1] * scale
            x1, y1 = slot.leader_anchor[0] * scale, slot.leader_anchor[1] * scale
            leader.line([(x0, y0), (x1, y1)], fill=outline_rgb + (200,), width=lw)
            leader.ellipse(
                [x1 - dot_r, y1 - dot_r, x1 + dot_r, y1 + dot_r], fill=outline_rgb + (255,)
            )

    def _draw_poi_sprites(self, canvas: Image.Image, scale: float, slots) -> None:
        shadow_rgb = self.palette["shadow"]
        for slot in sorted(slots, key=lambda s: s.anchor[1]):
            sprite = self._asset(slot.asset_id)
            w = max(8, int(slot.width_px * scale))
            h = max(8, int(slot.height_px * scale))
            x = int(slot.anchor[0] * scale - w / 2)
            y = int(slot.anchor[1] * scale - h)
            if sprite is None:
                # Plan-preview placeholder so composition is reviewable
                # before POI assets exist.
                draw = ImageDraw.Draw(canvas, "RGBA")
                draw.rounded_rectangle([x, y, x + w, y + h], radius=int(w * 0.08),
                                       fill=self.palette["building_wall"] + (130,),
                                       outline=self.palette["outline"] + (255,), width=max(1, int(3 * scale)))
                continue
            # Fit inside the slot box preserving the sprite's own aspect
            # (anchored bottom-center) -- the slot is a layout budget, not a
            # target shape to squash into.
            fit = min(w / sprite.width, h / sprite.height)
            fw, fh = max(1, round(sprite.width * fit)), max(1, round(sprite.height * fit))
            sprite = sprite.convert("RGBA").resize((fw, fh), Image.Resampling.LANCZOS)
            x += (w - fw) // 2
            y += h - fh
            w, h = fw, fh
            # Lift the landmark off the busy ground: a soft paper-toned halo
            # behind it (separation), then a firm dark drop shadow to the SE.
            pad = max(4, int(w * 0.10))
            big = Image.new("RGBA", (fw + 2 * pad, fh + 2 * pad), (0, 0, 0, 0))
            base_alpha = sprite.getchannel("A")

            halo_a = Image.new("L", big.size, 0)
            halo_a.paste(base_alpha, (pad, pad))
            halo_a = halo_a.filter(ImageFilter.GaussianBlur(max(2, int(pad * 0.7))))
            halo_a = halo_a.point(lambda a: min(255, int(a * 1.7)))
            halo = Image.new("RGBA", big.size, self.palette["paper"] + (0,))
            halo.putalpha(halo_a)
            big.alpha_composite(halo)

            dark = tuple(int(c * 0.55) for c in shadow_rgb)
            shadow_a = Image.new("L", big.size, 0)
            shadow_a.paste(base_alpha.point(lambda a: int(a * 0.62)), (pad, pad))
            shadow_a = shadow_a.filter(ImageFilter.GaussianBlur(max(1, int(w * 0.02))))
            shadow = Image.new("RGBA", big.size, dark + (0,))
            shadow.putalpha(shadow_a)
            offset = max(3, int(w * 0.05))
            big.alpha_composite(shadow, (offset, offset))

            big.alpha_composite(sprite, (pad, pad))
            canvas.alpha_composite(big, (x - pad, y - pad))

    def _draw_atmosphere(self, canvas: Image.Image, scale: float) -> None:
        """Distance haze: lighten and desaturate toward the horizon."""
        plan = self.plan
        horizon = plan.camera.horizon_margin * plan.canvas.height_px * scale
        fade_end = horizon + (canvas.height - horizon) * 0.35
        max_alpha = 90
        gradient = Image.new("L", (1, canvas.height), 0)
        for y in range(canvas.height):
            if y <= horizon:
                a = max_alpha
            elif y < fade_end:
                a = int(max_alpha * (1.0 - (y - horizon) / (fade_end - horizon)) ** 2.0)
            else:
                a = 0
            gradient.putpixel((0, y), a)
        haze = Image.new("RGBA", canvas.size, self.palette["haze"] + (0,))
        haze.putalpha(gradient.resize(canvas.size))
        canvas.alpha_composite(haze)

    def _draw_labels(self, canvas: Image.Image, scale: float) -> None:
        typo = self.plan.style.typography
        global_font = resolve_font_path(typo.font) or self.font_path
        for label in self.plan.labels:
            self._draw_one_label(canvas, label, scale, typo, global_font)

    def _draw_one_label(self, canvas, label, scale, typo, global_font) -> None:
        label_rgb = self.palette["label"]
        halo_default = self.palette["paper"]
        style = typo.for_kind(label.kind)
        size = max(7, int(label.font_size_px * scale))
        default_bold = label.kind in (LabelKind.TITLE, LabelKind.DISTRICT, LabelKind.POI)
        bold = style.bold if style.bold is not None else default_bold
        font_path = resolve_font_path(style.font) or global_font
        font = load_font(size, font_path, bold=bold)
        baseline = self._scaled(label.baseline, scale)

        # Per-kind colour and halo, falling back to palette defaults.
        if style.color:
            fill = hex_to_rgb(style.color)
        elif label.kind == LabelKind.WATER:
            fill = self.palette["water_edge"]
        else:
            fill = label_rgb
        if style.halo is not None:
            halo = None if style.halo.lower() == "none" else hex_to_rgb(style.halo)
        else:
            halo = halo_default

        if label.kind == LabelKind.TITLE:
            self._draw_title(canvas, label.text, baseline[0], size, scale, font_path, fill, halo)
            return
        if label.kind == LabelKind.SHIELD:
            self._draw_shield(canvas, label.text, baseline[0], size, label.network, font_path)
            return
        if label.kind == LabelKind.POI:
            # Landmark names must stay readable on busy painted ground:
            # set them on a small paper plate (mini-cartouche).
            self._draw_label_plate(canvas, label.text, baseline[0], font, size)
        draw_text_on_path(canvas, label.text, baseline, font, fill, halo=halo)

    def _draw_label_plate(
        self,
        canvas: Image.Image,
        text: str,
        center: Point,
        font,
        size: int,
    ) -> None:
        try:
            text_w = font.getlength(text)
        except AttributeError:
            text_w = font.getbbox(text)[2]
        pad_x = size * 0.55
        pad_y = size * 0.30
        try:
            ascent, descent = font.getmetrics()
            text_h = ascent + descent
        except AttributeError:
            text_h = size * 1.3
        x0 = center[0] - text_w / 2 - pad_x
        x1 = center[0] + text_w / 2 + pad_x
        y0 = center[1] - text_h / 2 - pad_y
        y1 = center[1] + text_h / 2 + pad_y
        draw = ImageDraw.Draw(canvas, "RGBA")
        radius = max(3, int((y1 - y0) * 0.38))
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=radius,
            fill=self.palette["paper"] + (225,),
            outline=self.palette["outline"] + (255,),
            width=max(1, int(size * 0.07)),
        )

    def _draw_shield(
        self,
        canvas: Image.Image,
        text: str,
        center: Point,
        size: int,
        network: str | None = None,
        font_path: str | None = None,
    ) -> None:
        """Render a highway shield placard: a studio-generated sprite (the
        blank shield painted in the map style) with the route number drawn on
        top when available, else an accurate procedural shield, else a text
        badge."""
        fp = font_path or self.font_path
        art = self._asset(shield_asset_id(network)) if network else None
        if art is not None:
            sprite = compose_ai_shield(art, text, network, size, fp)
        else:
            sprite = render_shield(text, network, size, fp)
        canvas.alpha_composite(
            sprite,
            (int(center[0] - sprite.width / 2), int(center[1] - sprite.height / 2)),
        )

    def _draw_title(
        self,
        canvas: Image.Image,
        text: str,
        center: Point,
        size: int,
        scale: float,
        font_path: str | None = None,
        fill: tuple[int, int, int] | None = None,
        halo: tuple[int, int, int] | None = None,
    ) -> None:
        cartouche = self._asset("ornament_cartouche")
        font = load_font(size, font_path or self.font_path, bold=True)
        if fill is None:
            fill = self.palette["label"]
        if halo is None:
            halo = self.palette["paper"]
        if cartouche is not None:
            # Size the frame around the lettering, not the other way round:
            # the ornament needs clear margin beyond the text on both sides.
            try:
                text_w = font.getlength(text)
            except AttributeError:
                text_w = len(text) * size * 0.62
            cw = int(max(text_w * 1.7, size * 7))
            ch = int(cw * cartouche.height / max(1, cartouche.width))
            # A title cartouche is a banner, not a centerpiece: cap its
            # height and accept stretching the ornament when the generated
            # asset came back squarer than requested.
            max_ch = int(canvas.height * 0.11)
            ch = min(ch, max_ch)
            cartouche = cartouche.convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
            # Clamp fully on-canvas (the title anchor can sit near the top
            # edge, which would clip the ornament), then center the
            # lettering in the frame wherever it actually landed.
            margin = max(8, int(canvas.width * 0.015))
            x0 = min(max(margin, int(center[0] - cw / 2)), max(margin, canvas.width - cw - margin))
            y0 = min(max(margin, int(center[1] - ch / 2)), max(margin, canvas.height - ch - margin))
            canvas.alpha_composite(cartouche, (x0, y0))
            center = (x0 + cw / 2, y0 + ch / 2)
        draw_text_on_path(canvas, text, [center], font, fill, halo=halo)

    def _draw_frame(self, canvas: Image.Image, scale: float) -> None:
        draw = ImageDraw.Draw(canvas, "RGBA")
        outline = self.palette["outline"]
        margin = max(6, int(canvas.width * 0.012))
        inner = max(3, int(canvas.width * 0.004))
        draw.rectangle([margin, margin, canvas.width - margin, canvas.height - margin],
                       outline=outline + (255,), width=max(2, int(canvas.width * 0.0035)))
        draw.rectangle([margin + inner * 2, margin + inner * 2,
                        canvas.width - margin - inner * 2, canvas.height - margin - inner * 2],
                       outline=outline + (180,), width=max(1, int(canvas.width * 0.001)))
        compass = self._asset("ornament_compass")
        if compass is not None:
            size = int(canvas.width * 0.09)
            compass = compass.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            canvas.alpha_composite(
                compass,
                (canvas.width - size - margin * 2, canvas.height - size - margin * 2),
            )

    def _apply_grain(self, canvas: Image.Image) -> None:
        strength = self.plan.style.paper_grain
        if strength <= 0:
            return
        # Striped: full-poster float32 noise + image copies peak at multiple
        # GB and can OOM a memory-capped WSL VM.
        rng = np.random.default_rng(42)
        out = np.asarray(canvas.convert("RGB")).copy()
        stripe = 1024
        for y in range(0, out.shape[0], stripe):
            sl = slice(y, min(y + stripe, out.shape[0]))
            noise = rng.normal(0.0, 1.0, (out[sl].shape[0], out.shape[1], 1)).astype(np.float32)
            block = out[sl].astype(np.float32) * (1.0 + strength * 0.35 * noise)
            out[sl] = np.clip(block, 0, 255).astype(np.uint8)
        grained = Image.fromarray(out, "RGB").convert("RGBA")
        canvas.paste(grained)
