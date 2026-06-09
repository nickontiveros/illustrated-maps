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
from .text import draw_text_on_path, load_font
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


class Compositor:
    def __init__(self, plan: PlanDocument, assets_dir: Path | str, font_path: str | None = None):
        self.plan = plan
        self.assets_dir = Path(assets_dir)
        self.font_path = font_path
        self.palette = {k: hex_to_rgb(v) for k, v in plan.style.palette.items()}
        self._sprite_cache: dict[str, list[Image.Image]] = {}

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
        plan = self.plan
        width = max(1, int(round(plan.canvas.width_px * scale)))
        height = max(1, int(round(plan.canvas.height_px * scale)))
        wobble = plan.style.wobble_px * max(0.35, scale)

        canvas = Image.new("RGBA", (width, height), self.palette["paper"] + (255,))
        logger.info("Compositing %dx%d (scale %.2f)", width, height, scale)
        # Wobble wavelength tracks the poster, not absolute pixels, so a
        # preview render and the full-res render crinkle identically.
        self._wavelength = max(20.0, plan.canvas.width_px * 0.012 * scale)

        self._draw_base_land(canvas, scale)
        self._draw_ground(canvas, scale, wobble)
        self._draw_waterways(canvas, scale, wobble)
        self._draw_roads(canvas, scale, wobble)
        self._draw_buildings(canvas, scale)
        self._draw_scatter(canvas, scale)
        self._draw_pois(canvas, scale)
        self._draw_atmosphere(canvas, scale)
        self._draw_labels(canvas, scale)
        self._draw_frame(canvas, scale)
        self._apply_grain(canvas)
        return canvas.convert("RGB")

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

    def _tile_texture(self, texture: Image.Image, size: tuple[int, int], scale: float) -> Image.Image:
        """Tile a texture across the canvas, scaled so its print density holds."""
        tile_size = max(64, int(texture.width * max(0.25, scale)))
        tile = texture.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        out = Image.new("RGB", size)
        for y in range(0, size[1], tile_size):
            for x in range(0, size[0], tile_size):
                out.paste(tile, (x, y))
        return out

    def _scaled(self, points: list[Point], scale: float) -> list[Point]:
        return [(x * scale, y * scale) for x, y in points]

    def _draw_ground(self, canvas: Image.Image, scale: float, wobble: float) -> None:
        plan = self.plan
        # Water first, then vegetation/land classes on top.
        ordered = sorted(plan.ground, key=lambda g: g.cls != GroundClass.WATER)
        for idx, poly in enumerate(ordered):
            ring = wobble_ring(self._scaled(poly.exterior, scale), wobble * 2, self._wavelength * 2, seed=idx * 3.1)
            if len(ring) < 3:
                continue
            mask = Image.new("L", canvas.size, 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.polygon(ring, fill=255)
            for hole in poly.holes:
                hole_ring = wobble_ring(self._scaled(hole, scale), wobble * 2, self._wavelength * 2, seed=idx * 3.1 + 1)
                if len(hole_ring) >= 3:
                    mdraw.polygon(hole_ring, fill=0)

            texture = self._texture(poly.cls)
            fill_color = self.palette.get(poly.cls.value, self.palette["land"])
            layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            if texture is not None:
                layer.paste(self._tile_texture(texture, canvas.size, scale), (0, 0), mask)
            else:
                solid = Image.new("RGBA", canvas.size, fill_color + (255,))
                layer.paste(solid, (0, 0), mask)
            canvas.alpha_composite(layer)

            # Painterly edge: darkened wobbly outline, heavier for water.
            # Urban/land fills are subtle tone shifts, not washes -- no edge.
            if poly.cls in (GroundClass.URBAN, GroundClass.LAND):
                continue
            water = poly.cls == GroundClass.WATER
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
        """2.5D extrusion of footprints: roof shifted up, visible side faces."""
        draw = ImageDraw.Draw(canvas, "RGBA")
        wall = self.palette["building_wall"]
        roof = self.palette["building_roof"]
        outline = self.palette["outline"]
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
            draw.polygon(top, fill=shade(roof, 1.0 - 0.25 * b.depth) + (255,), outline=outline + (220,))

    def _draw_scatter(self, canvas: Image.Image, scale: float) -> None:
        for slot in sorted(self.plan.scatter, key=lambda s: s.y):
            sprites = self._sprites(slot.kind)
            if not sprites:
                continue
            sprite = sprites[slot.variant % len(sprites)]
            target_w = max(4, int(slot.width_px * scale))
            target_h = max(4, int(sprite.height * target_w / max(1, sprite.width)))
            resized = sprite.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x = int(slot.x * scale - target_w / 2)
            y = int(slot.y * scale - target_h)  # anchor at the base
            canvas.alpha_composite(resized, (x, y))

    def _draw_pois(self, canvas: Image.Image, scale: float) -> None:
        shadow_rgb = self.palette["shadow"]
        for slot in sorted(self.plan.pois, key=lambda s: s.anchor[1]):
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
            sprite = sprite.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
            # Soft drop shadow from the sprite's alpha, offset to the SE.
            alpha = sprite.getchannel("A").point(lambda a: int(a * 0.45))
            shadow = Image.new("RGBA", sprite.size, shadow_rgb + (0,))
            shadow.putalpha(alpha)
            shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, int(w * 0.015))))
            offset = max(2, int(w * 0.03))
            canvas.alpha_composite(shadow, (x + offset, y + offset))
            canvas.alpha_composite(sprite, (x, y))

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
        label_rgb = self.palette["label"]
        halo = self.palette["paper"]
        for label in self.plan.labels:
            size = max(7, int(label.font_size_px * scale))
            bold = label.kind in (LabelKind.TITLE, LabelKind.DISTRICT, LabelKind.POI)
            font = load_font(size, self.font_path, bold=bold)
            baseline = self._scaled(label.baseline, scale)
            if label.kind == LabelKind.TITLE:
                self._draw_title(canvas, label.text, baseline[0], size, scale)
                continue
            fill = self.palette["water_edge"] if label.kind == LabelKind.WATER else label_rgb
            draw_text_on_path(canvas, label.text, baseline, font, fill, halo=halo)

    def _draw_title(self, canvas: Image.Image, text: str, center: Point, size: int, scale: float) -> None:
        cartouche = self._asset("ornament_cartouche")
        font = load_font(size, self.font_path, bold=True)
        if cartouche is not None:
            cw = int(size * max(6, len(text) * 0.75))
            ch = int(cw * cartouche.height / max(1, cartouche.width))
            cartouche = cartouche.convert("RGBA").resize((cw, ch), Image.Resampling.LANCZOS)
            canvas.alpha_composite(cartouche, (int(center[0] - cw / 2), int(center[1] - ch / 2)))
        draw_text_on_path(canvas, text, [center], font, self.palette["label"], halo=self.palette["paper"])

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
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 1.0, (canvas.height, canvas.width, 1)).astype(np.float32)
        arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)
        out = np.clip(arr * (1.0 + strength * 0.35 * noise), 0, 255).astype(np.uint8)
        grained = Image.fromarray(out, "RGB").convert("RGBA")
        canvas.paste(grained)
