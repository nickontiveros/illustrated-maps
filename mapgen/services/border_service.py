"""Service for rendering decorative borders, title cartouches, compass roses, and legends."""

import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..models.border import BorderSettings, BorderStyle, LegendItem
from ..models.project import BoundingBox
from .render_service import RenderService


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert a hex color string to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    elif len(hex_color) == 8:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        alpha = int(hex_color[6:8], 16)
    else:
        r, g, b = 0, 0, 0
    return (r, g, b, alpha)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a TrueType font, falling back to the PIL default.

    Tries several common system font paths on Linux/macOS/Windows.
    """
    font_candidates: list[str] = []
    if bold:
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
    else:
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # Last resort: PIL default bitmap font (ignores size parameter)
    return ImageFont.load_default()


class BorderService:
    """Renders decorative borders, title cartouches, compass roses, and legends.

    All rendering uses PIL (Pillow) ImageDraw primitives only -- no matplotlib,
    no SVG, no external renderers.
    """

    # Default border color when none is specified in settings
    DEFAULT_BORDER_COLOR = "#3C2415"

    def render_border(
        self,
        map_image: Image.Image,
        settings: BorderSettings,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        rotation_degrees: float = 0.0,
    ) -> Image.Image:
        """Render a full decorative border around the map image.

        Expands the canvas by ``settings.margin`` on every side, then draws
        the border frame, title cartouche, compass rose, and legend.

        Args:
            map_image: The source map image (RGBA recommended).
            settings: Border configuration.
            title: Map title text (displayed in the top-margin cartouche).
            subtitle: Subtitle text rendered below the title.
            bbox: Geographic bounding box, used for coordinate labels.
            rotation_degrees: Map rotation in degrees clockwise from north
                (affects compass rose north arrow).

        Returns:
            A new PIL Image with the border composited around the map.
        """
        margin = settings.margin
        bg_color = settings.background_color or "#FFF8F0"
        border_color = settings.border_color or self.DEFAULT_BORDER_COLOR
        ornament_alpha = int(settings.ornament_opacity * 255)

        # Step 1 -- expand canvas
        canvas = self._expand_canvas(map_image, margin, bg_color)
        draw = ImageDraw.Draw(canvas, "RGBA")
        canvas_size = canvas.size  # (width, height)

        # Step 2 -- border frame
        self._render_border_frame(draw, canvas_size, margin, settings.style, border_color, ornament_alpha)

        # Step 3 -- title cartouche
        if title:
            self._render_title_block(canvas, draw, title, subtitle, canvas_size, margin, border_color)

        # Step 4 -- compass rose
        if settings.show_compass:
            self._render_compass_rose(canvas, canvas_size, margin, rotation_degrees, border_color)

        # Step 5 -- legend
        if settings.show_legend:
            legend_items = settings.legend_items if settings.legend_items else self._auto_generate_legend_items()
            self._render_legend(canvas, draw, legend_items, canvas_size, margin, border_color)

        return canvas

    # ------------------------------------------------------------------
    # Canvas expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_canvas(
        image: Image.Image,
        margin: int,
        bg_color: str,
    ) -> Image.Image:
        """Create a larger canvas with the original image centred inside the margin.

        Args:
            image: Original map image.
            margin: Number of pixels to add on each side.
            bg_color: Background fill colour (hex string).

        Returns:
            New RGBA image with expanded dimensions.
        """
        orig_w, orig_h = image.size
        new_w = orig_w + 2 * margin
        new_h = orig_h + 2 * margin

        bg_rgba = _hex_to_rgba(bg_color)
        canvas = Image.new("RGBA", (new_w, new_h), bg_rgba)

        # Paste map centred in the new canvas
        source = image.convert("RGBA")
        canvas.paste(source, (margin, margin), source)
        return canvas

    # ------------------------------------------------------------------
    # Border frame
    # ------------------------------------------------------------------

    def _render_border_frame(
        self,
        draw: ImageDraw.ImageDraw,
        canvas_size: tuple[int, int],
        margin: int,
        style: BorderStyle,
        color: str,
        ornament_alpha: int = 204,
    ) -> None:
        """Dispatch to the correct border-frame renderer based on *style*."""
        cw, ch = canvas_size
        color_rgba = _hex_to_rgba(color, ornament_alpha)

        if style == BorderStyle.VINTAGE_SCROLL:
            self._frame_vintage_scroll(draw, cw, ch, margin, color_rgba)
        elif style == BorderStyle.ART_DECO:
            self._frame_art_deco(draw, cw, ch, margin, color_rgba)
        elif style == BorderStyle.MODERN_MINIMAL:
            self._frame_modern_minimal(draw, cw, ch, margin, color_rgba)
        elif style == BorderStyle.ORNATE_VICTORIAN:
            self._frame_ornate_victorian(draw, cw, ch, margin, color_rgba)

    # -- Vintage Scroll --------------------------------------------------

    def _frame_vintage_scroll(
        self,
        draw: ImageDraw.ImageDraw,
        cw: int,
        ch: int,
        margin: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """Double-line border with decorative corner ornaments."""
        outer_inset = int(margin * 0.15)
        inner_inset = int(margin * 0.30)
        line_width = max(2, margin // 60)

        # Outer rectangle
        draw.rectangle(
            [outer_inset, outer_inset, cw - 1 - outer_inset, ch - 1 - outer_inset],
            outline=color,
            width=line_width,
        )
        # Inner rectangle
        draw.rectangle(
            [inner_inset, inner_inset, cw - 1 - inner_inset, ch - 1 - inner_inset],
            outline=color,
            width=line_width,
        )

        # Corner ornaments -- decorative arcs and circles in each corner
        corner_radius = int(margin * 0.18)
        dot_r = max(3, margin // 30)
        corners = [
            (outer_inset, outer_inset),                          # top-left
            (cw - 1 - outer_inset, outer_inset),                 # top-right
            (outer_inset, ch - 1 - outer_inset),                 # bottom-left
            (cw - 1 - outer_inset, ch - 1 - outer_inset),       # bottom-right
        ]
        # Arc start angles for each corner (degrees, PIL convention)
        arc_angles = [
            (180, 270),  # top-left
            (270, 360),  # top-right
            (90, 180),   # bottom-left
            (0, 90),     # bottom-right
        ]
        for (cx, cy), (start, end) in zip(corners, arc_angles):
            # Large decorative arc
            arc_box = [cx - corner_radius, cy - corner_radius, cx + corner_radius, cy + corner_radius]
            draw.arc(arc_box, start, end, fill=color, width=line_width)
            # Smaller concentric arc
            sr = corner_radius // 2
            draw.arc([cx - sr, cy - sr, cx + sr, cy + sr], start, end, fill=color, width=line_width)
            # Centre dot
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=color)

        # Small decorative scrollwork lines connecting outer and inner rectangles
        # at midpoints of each side
        mid_x = cw // 2
        mid_y = ch // 2
        scroll_len = inner_inset - outer_inset
        dash_gap = max(4, scroll_len // 5)
        # Top midpoint
        for y in range(outer_inset, inner_inset, dash_gap * 2):
            draw.line([(mid_x - dash_gap, y), (mid_x + dash_gap, y)], fill=color, width=line_width)
        # Bottom midpoint
        for y in range(ch - 1 - inner_inset, ch - 1 - outer_inset, dash_gap * 2):
            draw.line([(mid_x - dash_gap, y), (mid_x + dash_gap, y)], fill=color, width=line_width)
        # Left midpoint
        for x in range(outer_inset, inner_inset, dash_gap * 2):
            draw.line([(x, mid_y - dash_gap), (x, mid_y + dash_gap)], fill=color, width=line_width)
        # Right midpoint
        for x in range(cw - 1 - inner_inset, cw - 1 - outer_inset, dash_gap * 2):
            draw.line([(x, mid_y - dash_gap), (x, mid_y + dash_gap)], fill=color, width=line_width)

    # -- Art Deco --------------------------------------------------------

    def _frame_art_deco(
        self,
        draw: ImageDraw.ImageDraw,
        cw: int,
        ch: int,
        margin: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """Geometric stepped border pattern inspired by Art Deco motifs."""
        line_width = max(2, margin // 50)
        step_count = 4
        step_size = int(margin * 0.12)
        base_inset = int(margin * 0.15)

        # Draw a series of concentric rectangles that "step" inward
        for i in range(step_count):
            inset = base_inset + i * step_size
            draw.rectangle(
                [inset, inset, cw - 1 - inset, ch - 1 - inset],
                outline=color,
                width=line_width,
            )

        # Geometric corner accents -- nested right-angle chevrons
        chevron_size = int(margin * 0.25)
        chevron_width = line_width + 1
        inner_inset = base_inset + (step_count - 1) * step_size

        # Top-left corner chevrons
        for k in range(3):
            offset = k * int(chevron_size * 0.35)
            x0 = inner_inset + offset
            y0 = inner_inset + offset
            # Horizontal arm
            draw.line([(x0, y0), (x0 + chevron_size - offset, y0)], fill=color, width=chevron_width)
            # Vertical arm
            draw.line([(x0, y0), (x0, y0 + chevron_size - offset)], fill=color, width=chevron_width)

        # Top-right corner chevrons
        for k in range(3):
            offset = k * int(chevron_size * 0.35)
            x0 = cw - 1 - inner_inset - offset
            y0 = inner_inset + offset
            draw.line([(x0, y0), (x0 - chevron_size + offset, y0)], fill=color, width=chevron_width)
            draw.line([(x0, y0), (x0, y0 + chevron_size - offset)], fill=color, width=chevron_width)

        # Bottom-left corner chevrons
        for k in range(3):
            offset = k * int(chevron_size * 0.35)
            x0 = inner_inset + offset
            y0 = ch - 1 - inner_inset - offset
            draw.line([(x0, y0), (x0 + chevron_size - offset, y0)], fill=color, width=chevron_width)
            draw.line([(x0, y0), (x0, y0 - chevron_size + offset)], fill=color, width=chevron_width)

        # Bottom-right corner chevrons
        for k in range(3):
            offset = k * int(chevron_size * 0.35)
            x0 = cw - 1 - inner_inset - offset
            y0 = ch - 1 - inner_inset - offset
            draw.line([(x0, y0), (x0 - chevron_size + offset, y0)], fill=color, width=chevron_width)
            draw.line([(x0, y0), (x0, y0 - chevron_size + offset)], fill=color, width=chevron_width)

        # Midpoint diamond accents on each side
        diamond_r = max(6, margin // 20)
        mid_x = cw // 2
        mid_y = ch // 2

        for cx, cy in [
            (mid_x, base_inset),                   # top
            (mid_x, ch - 1 - base_inset),           # bottom
            (base_inset, mid_y),                     # left
            (cw - 1 - base_inset, mid_y),            # right
        ]:
            diamond = [
                (cx, cy - diamond_r),
                (cx + diamond_r, cy),
                (cx, cy + diamond_r),
                (cx - diamond_r, cy),
            ]
            draw.polygon(diamond, fill=color, outline=color)

    # -- Modern Minimal --------------------------------------------------

    def _frame_modern_minimal(
        self,
        draw: ImageDraw.ImageDraw,
        cw: int,
        ch: int,
        margin: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """Single clean line border."""
        inset = int(margin * 0.25)
        line_width = max(2, margin // 80)
        draw.rectangle(
            [inset, inset, cw - 1 - inset, ch - 1 - inset],
            outline=color,
            width=line_width,
        )

    # -- Ornate Victorian ------------------------------------------------

    def _frame_ornate_victorian(
        self,
        draw: ImageDraw.ImageDraw,
        cw: int,
        ch: int,
        margin: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """Triple-line border with dot/circle ornaments at regular intervals."""
        line_width = max(2, margin // 70)
        insets = [int(margin * 0.12), int(margin * 0.22), int(margin * 0.32)]

        for inset in insets:
            draw.rectangle(
                [inset, inset, cw - 1 - inset, ch - 1 - inset],
                outline=color,
                width=line_width,
            )

        # Ornamental dots along the middle border line at regular intervals
        mid_inset = insets[1]
        dot_r = max(3, margin // 35)
        spacing = max(40, margin)  # spacing between ornament dots

        # Top edge
        x = mid_inset + spacing
        while x < cw - mid_inset - spacing // 2:
            draw.ellipse(
                [x - dot_r, mid_inset - dot_r, x + dot_r, mid_inset + dot_r],
                fill=color,
            )
            x += spacing

        # Bottom edge
        y_bot = ch - 1 - mid_inset
        x = mid_inset + spacing
        while x < cw - mid_inset - spacing // 2:
            draw.ellipse([x - dot_r, y_bot - dot_r, x + dot_r, y_bot + dot_r], fill=color)
            x += spacing

        # Left edge
        y = mid_inset + spacing
        while y < ch - mid_inset - spacing // 2:
            draw.ellipse([mid_inset - dot_r, y - dot_r, mid_inset + dot_r, y + dot_r], fill=color)
            y += spacing

        # Right edge
        x_right = cw - 1 - mid_inset
        y = mid_inset + spacing
        while y < ch - mid_inset - spacing // 2:
            draw.ellipse([x_right - dot_r, y - dot_r, x_right + dot_r, y + dot_r], fill=color)
            y += spacing

        # Larger circle ornaments at the four corners of the middle border
        corner_r = dot_r * 3
        corner_pts = [
            (mid_inset, mid_inset),
            (cw - 1 - mid_inset, mid_inset),
            (mid_inset, ch - 1 - mid_inset),
            (cw - 1 - mid_inset, ch - 1 - mid_inset),
        ]
        for cx, cy in corner_pts:
            draw.ellipse(
                [cx - corner_r, cy - corner_r, cx + corner_r, cy + corner_r],
                outline=color,
                width=line_width,
            )
            # Inner filled dot
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=color,
            )

    # ------------------------------------------------------------------
    # Title cartouche
    # ------------------------------------------------------------------

    def _render_title_block(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        title: str,
        subtitle: Optional[str],
        canvas_size: tuple[int, int],
        margin: int,
        border_color: str = "#3C2415",
    ) -> None:
        """Render title and optional subtitle in the top margin with a decorative cartouche.

        The title is centred horizontally. A rectangular cartouche with rounded
        feel (drawn with arcs at corners) frames the text block.
        """
        cw, ch = canvas_size
        color_rgba = _hex_to_rgba(border_color)

        # Font sizing relative to margin
        title_font_size = max(16, int(margin * 0.30))
        subtitle_font_size = max(12, int(margin * 0.18))

        title_font = _load_font(title_font_size, bold=True)
        subtitle_font = _load_font(subtitle_font_size, bold=False)

        # Measure title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]

        # Measure subtitle
        sub_w = 0
        sub_h = 0
        if subtitle:
            sub_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            sub_w = sub_bbox[2] - sub_bbox[0]
            sub_h = sub_bbox[3] - sub_bbox[1]

        # Total text block dimensions
        text_block_w = max(title_w, sub_w)
        gap = int(title_h * 0.3) if subtitle else 0
        text_block_h = title_h + gap + sub_h

        # Cartouche padding
        pad_x = int(margin * 0.25)
        pad_y = int(margin * 0.12)
        cart_w = text_block_w + 2 * pad_x
        cart_h = text_block_h + 2 * pad_y

        # Centre the cartouche horizontally in the top margin
        cart_x = (cw - cart_w) // 2
        cart_y = max(int(margin * 0.10), (margin - cart_h) // 2)

        # Draw cartouche background (slightly transparent parchment)
        bg_rgba = _hex_to_rgba("#FFF8F0", 220)
        draw.rectangle(
            [cart_x, cart_y, cart_x + cart_w, cart_y + cart_h],
            fill=bg_rgba,
        )

        # Cartouche border -- outer and inner rectangles with corner arcs
        line_w = max(2, margin // 80)
        draw.rectangle(
            [cart_x, cart_y, cart_x + cart_w, cart_y + cart_h],
            outline=color_rgba,
            width=line_w,
        )
        inner_pad = max(4, margin // 40)
        draw.rectangle(
            [
                cart_x + inner_pad,
                cart_y + inner_pad,
                cart_x + cart_w - inner_pad,
                cart_y + cart_h - inner_pad,
            ],
            outline=color_rgba,
            width=max(1, line_w // 2),
        )

        # Small decorative arcs at each corner of the cartouche
        arc_r = inner_pad * 2
        corner_arcs = [
            (cart_x, cart_y, 180, 270),
            (cart_x + cart_w, cart_y, 270, 360),
            (cart_x, cart_y + cart_h, 90, 180),
            (cart_x + cart_w, cart_y + cart_h, 0, 90),
        ]
        for cx, cy, sa, ea in corner_arcs:
            draw.arc(
                [cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r],
                sa, ea,
                fill=color_rgba,
                width=max(1, line_w // 2),
            )

        # Draw title text
        title_x = cart_x + (cart_w - title_w) // 2
        title_y = cart_y + pad_y
        draw.text((title_x, title_y), title, fill=color_rgba, font=title_font)

        # Draw subtitle text
        if subtitle:
            sub_x = cart_x + (cart_w - sub_w) // 2
            sub_y = title_y + title_h + gap
            # Slightly lighter colour for subtitle
            sub_color = _hex_to_rgba(border_color, 180)
            draw.text((sub_x, sub_y), subtitle, fill=sub_color, font=subtitle_font)

    # ------------------------------------------------------------------
    # Compass rose
    # ------------------------------------------------------------------

    def _render_compass_rose(
        self,
        image: Image.Image,
        canvas_size: tuple[int, int],
        margin: int,
        rotation_degrees: float = 0.0,
        border_color: str = "#3C2415",
    ) -> None:
        """Render a compass rose in the bottom-right margin area.

        Draws a circle with four cardinal-direction arrows (N/S/E/W).
        The north arrow is rotated by *rotation_degrees* so it points to
        true north regardless of the map's orientation.

        All drawing is done with PIL ImageDraw primitives.
        """
        draw = ImageDraw.Draw(image, "RGBA")
        cw, ch = canvas_size
        color_rgba = _hex_to_rgba(border_color)
        light_color = _hex_to_rgba(border_color, 140)

        # Compass centre: bottom-right margin area
        compass_r = int(margin * 0.30)
        cx = cw - margin // 2
        cy = ch - margin // 2

        # Outer circle
        line_w = max(2, margin // 80)
        draw.ellipse(
            [cx - compass_r, cy - compass_r, cx + compass_r, cy + compass_r],
            outline=color_rgba,
            width=line_w,
        )
        # Inner circle
        inner_r = int(compass_r * 0.85)
        draw.ellipse(
            [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
            outline=light_color,
            width=max(1, line_w // 2),
        )

        # Centre dot
        dot_r = max(3, compass_r // 12)
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=color_rgba)

        # Arrow geometry
        arrow_length = int(compass_r * 0.75)
        arrow_half_width = max(3, compass_r // 10)

        # Rotation in radians (PIL angles: 0 = right, we use math angles)
        # rotation_degrees is clockwise from north. In our coordinate system
        # north is "up" (negative Y), so base angle for north is -90 degrees
        # in standard math convention.
        rot_rad = math.radians(-rotation_degrees)

        # Cardinal directions as (angle_offset_from_north, label, is_primary)
        directions = [
            (0, "N", True),
            (180, "S", False),
            (90, "E", False),
            (270, "W", False),
        ]

        label_font = _load_font(max(10, compass_r // 4), bold=True)

        for angle_offset, label, is_primary in directions:
            # Angle in standard math convention (radians, counter-clockwise from right)
            angle = rot_rad + math.radians(-angle_offset) + math.radians(90)

            # Arrow tip
            tip_x = cx + arrow_length * math.cos(angle)
            tip_y = cy - arrow_length * math.sin(angle)

            # Arrow base (two points forming the base of a triangle)
            perp_angle = angle + math.pi / 2
            base_len = arrow_half_width if is_primary else arrow_half_width * 0.7

            b1_x = cx + base_len * math.cos(perp_angle)
            b1_y = cy - base_len * math.sin(perp_angle)
            b2_x = cx - base_len * math.cos(perp_angle)
            b2_y = cy + base_len * math.sin(perp_angle)

            arrow_points = [
                (tip_x, tip_y),
                (b1_x, b1_y),
                (b2_x, b2_y),
            ]

            if is_primary:
                # North arrow: filled solid
                draw.polygon(arrow_points, fill=color_rgba, outline=color_rgba)
            else:
                # Other arrows: outline only
                draw.polygon(arrow_points, outline=color_rgba)
                # Fill with light colour
                draw.polygon(arrow_points, fill=light_color)

            # Label placement: a bit beyond the arrow tip
            label_dist = arrow_length + max(8, compass_r // 5)
            lx = cx + label_dist * math.cos(angle)
            ly = cy - label_dist * math.sin(angle)

            # Centre the label on the computed point
            lbl_bbox = draw.textbbox((0, 0), label, font=label_font)
            lbl_w = lbl_bbox[2] - lbl_bbox[0]
            lbl_h = lbl_bbox[3] - lbl_bbox[1]
            draw.text(
                (lx - lbl_w / 2, ly - lbl_h / 2),
                label,
                fill=color_rgba,
                font=label_font,
            )

        # Small tick marks for intercardinal directions (NE, SE, SW, NW)
        tick_inner = int(compass_r * 0.60)
        tick_outer = int(compass_r * 0.75)
        for angle_offset in [45, 135, 225, 315]:
            angle = rot_rad + math.radians(-angle_offset) + math.radians(90)
            x1 = cx + tick_inner * math.cos(angle)
            y1 = cy - tick_inner * math.sin(angle)
            x2 = cx + tick_outer * math.cos(angle)
            y2 = cy - tick_outer * math.sin(angle)
            draw.line([(x1, y1), (x2, y2)], fill=color_rgba, width=max(1, line_w // 2))

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    def _render_legend(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        legend_items: list[LegendItem],
        canvas_size: tuple[int, int],
        margin: int,
        border_color: str = "#3C2415",
    ) -> None:
        """Render a legend panel in the bottom-left margin area.

        Each legend item is rendered as a colour swatch (rectangle, circle,
        solid line, or dashed line) followed by a text label.
        """
        if not legend_items:
            return

        cw, ch = canvas_size
        color_rgba = _hex_to_rgba(border_color)

        # Font
        font_size = max(11, int(margin * 0.12))
        font = _load_font(font_size)

        # Layout constants
        swatch_size = max(12, int(margin * 0.10))
        row_height = int(swatch_size * 1.8)
        text_gap = int(swatch_size * 0.6)

        # Compute total legend dimensions
        max_label_w = 0
        for item in legend_items:
            bbox = draw.textbbox((0, 0), item.label, font=font)
            w = bbox[2] - bbox[0]
            if w > max_label_w:
                max_label_w = w

        legend_w = swatch_size + text_gap + max_label_w + int(margin * 0.15)
        legend_h = len(legend_items) * row_height + int(margin * 0.10)

        # Position: bottom-left margin
        legend_x = int(margin * 0.20)
        legend_y = ch - margin + int(margin * 0.10)

        # Background panel
        panel_pad = int(margin * 0.06)
        panel_rgba = _hex_to_rgba("#FFF8F0", 210)
        draw.rectangle(
            [
                legend_x - panel_pad,
                legend_y - panel_pad,
                legend_x + legend_w + panel_pad,
                legend_y + legend_h + panel_pad,
            ],
            fill=panel_rgba,
        )
        draw.rectangle(
            [
                legend_x - panel_pad,
                legend_y - panel_pad,
                legend_x + legend_w + panel_pad,
                legend_y + legend_h + panel_pad,
            ],
            outline=color_rgba,
            width=max(1, margin // 100),
        )

        # "Legend" header
        header_font = _load_font(max(12, int(margin * 0.13)), bold=True)
        header_text = "Legend"
        hdr_bbox = draw.textbbox((0, 0), header_text, font=header_font)
        hdr_h = hdr_bbox[3] - hdr_bbox[1]
        draw.text((legend_x, legend_y), header_text, fill=color_rgba, font=header_font)

        # Items
        current_y = legend_y + hdr_h + int(margin * 0.06)

        for item in legend_items:
            item_color = _hex_to_rgba(item.color)
            swatch_x = legend_x
            swatch_cy = current_y + swatch_size // 2

            if item.symbol == "rect":
                draw.rectangle(
                    [swatch_x, current_y, swatch_x + swatch_size, current_y + swatch_size],
                    fill=item_color,
                    outline=color_rgba,
                    width=1,
                )
            elif item.symbol == "circle":
                r = swatch_size // 2
                draw.ellipse(
                    [swatch_x, current_y, swatch_x + swatch_size, current_y + swatch_size],
                    fill=item_color,
                    outline=color_rgba,
                    width=1,
                )
            elif item.symbol == "line":
                line_y = swatch_cy
                draw.line(
                    [(swatch_x, line_y), (swatch_x + swatch_size, line_y)],
                    fill=item_color,
                    width=max(2, swatch_size // 5),
                )
            elif item.symbol == "dashed":
                line_y = swatch_cy
                dash_len = max(3, swatch_size // 4)
                x = swatch_x
                while x < swatch_x + swatch_size:
                    x_end = min(x + dash_len, swatch_x + swatch_size)
                    draw.line(
                        [(x, line_y), (x_end, line_y)],
                        fill=item_color,
                        width=max(2, swatch_size // 5),
                    )
                    x += dash_len * 2
            else:
                # Fallback: rectangle
                draw.rectangle(
                    [swatch_x, current_y, swatch_x + swatch_size, current_y + swatch_size],
                    fill=item_color,
                    outline=color_rgba,
                    width=1,
                )

            # Label text
            text_x = swatch_x + swatch_size + text_gap
            # Vertically centre text with swatch
            lbl_bbox = draw.textbbox((0, 0), item.label, font=font)
            lbl_h = lbl_bbox[3] - lbl_bbox[1]
            text_y = current_y + (swatch_size - lbl_h) // 2
            draw.text((text_x, text_y), item.label, fill=color_rgba, font=font)

            current_y += row_height

    # ------------------------------------------------------------------
    # Auto-generated legend items
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_generate_legend_items() -> list[LegendItem]:
        """Return default legend items derived from RenderService.COLORS.

        Provides a standard set of map-feature legend entries:
        Water, Parks, Buildings, Major Roads, Minor Roads, Railways.
        """
        colors = RenderService.COLORS
        return [
            LegendItem(label="Water", color=colors["water"], symbol="rect"),
            LegendItem(label="Parks", color=colors["park"], symbol="rect"),
            LegendItem(label="Buildings", color=colors["building"], symbol="rect"),
            LegendItem(label="Major Roads", color=colors["road_major"], symbol="line"),
            LegendItem(label="Minor Roads", color=colors["road_minor"], symbol="line"),
            LegendItem(label="Railways", color=colors["railway"], symbol="dashed"),
        ]
