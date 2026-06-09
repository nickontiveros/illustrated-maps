"""Deterministic offline asset generator.

Produces procedural placeholder assets so the full pipeline (plan ->
assets -> compose) runs end-to-end with zero API cost: in tests, in CI,
and for layout review before committing AI spend. Output is seeded by
the asset id, so runs are reproducible.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from ..types import AssetKind, AssetSpec, StyleSpec, hex_to_rgb, shade
from .matting import KEY_COLOR


class StubAssetGenerator:
    """AssetGenerator that draws simple procedural stand-ins."""

    def generate(
        self,
        spec: AssetSpec,
        style: StyleSpec,
        style_reference: Optional[Image.Image] = None,
    ) -> Image.Image:
        rng = random.Random(spec.id)
        if spec.kind == AssetKind.TEXTURE:
            return self._texture(spec, style, rng)
        if spec.kind == AssetKind.SPRITE_SHEET:
            return self._sprite_sheet(spec, style, rng)
        if spec.kind == AssetKind.POI_SPRITE:
            return self._poi_building(spec, style, rng)
        if spec.kind == AssetKind.ORNAMENT:
            return self._ornament(spec, style, rng)
        return self._texture(spec, style, rng)  # style bible: a swatch

    # -- textures ---------------------------------------------------------

    def _texture(self, spec: AssetSpec, style: StyleSpec, rng: random.Random) -> Image.Image:
        base = hex_to_rgb(style.palette.get(spec.subject, style.palette["land"]))
        noise_rng = np.random.default_rng(abs(hash(spec.id)) % (2**32))
        noise = noise_rng.normal(0.0, 1.0, (spec.height_px // 4, spec.width_px // 4, 1))
        # Smooth the noise into soft blotches reminiscent of a wash.
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(noise, sigma=(6, 6, 0))
        noise = noise / (np.abs(noise).max() + 1e-9)
        arr = np.array(base, dtype=np.float32)[None, None, :] * (1.0 + 0.08 * noise)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        return img.resize((spec.width_px, spec.height_px), Image.Resampling.BILINEAR)

    # -- sprites ----------------------------------------------------------

    def _sprite_sheet(self, spec: AssetSpec, style: StyleSpec, rng: random.Random) -> Image.Image:
        cols, rows = spec.sheet_grid or (3, 2)
        img = Image.new("RGB", (spec.width_px, spec.height_px), KEY_COLOR)
        draw = ImageDraw.Draw(img)
        cell_w, cell_h = spec.width_px // cols, spec.height_px // rows
        for r in range(rows):
            for c in range(cols):
                cx, cy = c * cell_w + cell_w // 2, r * cell_h + cell_h // 2
                if spec.subject == "tree":
                    self._tree(draw, cx, cy, min(cell_w, cell_h), style, rng)
                elif spec.subject == "boat":
                    self._boat(draw, cx, cy, min(cell_w, cell_h), style, rng)
                else:
                    self._house(draw, cx, cy, min(cell_w, cell_h), style, rng)
        return img

    def _tree(self, draw, cx, cy, cell, style, rng):
        green = hex_to_rgb(style.palette["forest"])
        radius = cell * rng.uniform(0.22, 0.3)
        trunk = hex_to_rgb(style.palette["building_wall"])
        draw.rectangle([cx - cell * 0.03, cy, cx + cell * 0.03, cy + cell * 0.22], fill=trunk)
        draw.ellipse(
            [cx - radius, cy - radius * 1.6, cx + radius, cy + radius * 0.4],
            fill=shade(green, rng.uniform(0.9, 1.15)),
            outline=hex_to_rgb(style.palette["outline"]),
            width=3,
        )

    def _house(self, draw, cx, cy, cell, style, rng):
        w = cell * rng.uniform(0.32, 0.42)
        h = cell * rng.uniform(0.22, 0.3)
        wall = hex_to_rgb(style.palette["building_wall"])
        roof = hex_to_rgb(style.palette["building_roof"])
        outline = hex_to_rgb(style.palette["outline"])
        draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=wall, outline=outline, width=3)
        draw.polygon(
            [(cx - w / 2 - w * 0.08, cy - h / 2), (cx + w / 2 + w * 0.08, cy - h / 2), (cx, cy - h * 1.25)],
            fill=shade(roof, rng.uniform(0.9, 1.1)),
            outline=outline,
        )

    def _boat(self, draw, cx, cy, cell, style, rng):
        outline = hex_to_rgb(style.palette["outline"])
        hull = hex_to_rgb(style.palette["building_roof"])
        w = cell * 0.36
        draw.polygon(
            [(cx - w / 2, cy), (cx + w / 2, cy), (cx + w * 0.3, cy + w * 0.22), (cx - w * 0.3, cy + w * 0.22)],
            fill=hull,
            outline=outline,
        )
        draw.polygon([(cx, cy - w * 0.5), (cx, cy - w * 0.05), (cx + w * 0.3, cy - w * 0.05)], fill=(245, 240, 225), outline=outline)

    def _poi_building(self, spec: AssetSpec, style: StyleSpec, rng: random.Random) -> Image.Image:
        img = Image.new("RGB", (spec.width_px, spec.height_px), KEY_COLOR)
        draw = ImageDraw.Draw(img)
        w, h = spec.width_px, spec.height_px
        outline = hex_to_rgb(style.palette["outline"])
        wall = shade(hex_to_rgb(style.palette["building_wall"]), rng.uniform(0.85, 1.15))
        roof = shade(hex_to_rgb(style.palette["building_roof"]), rng.uniform(0.85, 1.15))
        # A simple oblique landmark block: two visible facades + roof.
        bw, bh = w * 0.55, h * rng.uniform(0.4, 0.62)
        x0, y1 = w * 0.22, h * 0.92
        skew = w * 0.13
        front = [(x0, y1), (x0 + bw, y1), (x0 + bw, y1 - bh), (x0, y1 - bh)]
        side = [(x0 + bw, y1), (x0 + bw + skew, y1 - skew * 0.5), (x0 + bw + skew, y1 - bh - skew * 0.5), (x0 + bw, y1 - bh)]
        top = [(x0, y1 - bh), (x0 + bw, y1 - bh), (x0 + bw + skew, y1 - bh - skew * 0.5), (x0 + skew, y1 - bh - skew * 0.5)]
        draw.polygon(front, fill=wall, outline=outline)
        draw.polygon(side, fill=shade(wall, 0.8), outline=outline)
        draw.polygon(top, fill=roof, outline=outline)
        # Windows.
        cols, rows = 4, max(2, int(bh / (h * 0.12)))
        for r in range(rows):
            for c in range(cols):
                wx = x0 + bw * (0.12 + 0.2 * c)
                wy = y1 - bh * (0.15 + 0.8 * r / rows)
                draw.rectangle([wx, wy - h * 0.025, wx + w * 0.06, wy], fill=shade(wall, 0.6))
        return img

    def _ornament(self, spec: AssetSpec, style: StyleSpec, rng: random.Random) -> Image.Image:
        img = Image.new("RGB", (spec.width_px, spec.height_px), KEY_COLOR)
        draw = ImageDraw.Draw(img)
        outline = hex_to_rgb(style.palette["outline"])
        cx, cy = spec.width_px / 2, spec.height_px / 2
        radius = min(spec.width_px, spec.height_px) * 0.4
        if "compass" in spec.subject:
            for i in range(8):
                import math

                a = i * math.pi / 4
                tip = (cx + radius * math.cos(a), cy + radius * math.sin(a))
                base_r = radius * (0.18 if i % 2 == 0 else 0.1)
                left = (cx + base_r * math.cos(a + math.pi / 2), cy + base_r * math.sin(a + math.pi / 2))
                right = (cx + base_r * math.cos(a - math.pi / 2), cy + base_r * math.sin(a - math.pi / 2))
                fill = outline if i % 2 == 0 else hex_to_rgb(style.palette["building_roof"])
                draw.polygon([left, tip, right], fill=fill, outline=outline)
            draw.ellipse([cx - radius * 0.08, cy - radius * 0.08, cx + radius * 0.08, cy + radius * 0.08], fill=outline)
        else:
            draw.rounded_rectangle(
                [spec.width_px * 0.05, spec.height_px * 0.12, spec.width_px * 0.95, spec.height_px * 0.88],
                radius=spec.height_px * 0.12,
                outline=outline,
                width=max(4, spec.width_px // 100),
            )
        return img
