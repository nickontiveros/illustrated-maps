"""Typography service for map label extraction, placement, and rendering.

Handles label extraction from OSM data, curved text path rendering along roads,
collision-aware placement, and final rendering onto map images.
"""

import logging
import math
from typing import Optional, Union

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from ..models.project import BoundingBox
from ..models.typography import (
    FONT_COLORS,
    FONT_SIZES,
    FontTier,
    Label,
    TextPath,
    TypographySettings,
)
from ..utils.geo_utils import gps_to_pixel

logger = logging.getLogger(__name__)

# Reference image size for font scaling (A1 at 300 DPI)
REFERENCE_WIDTH = 7016
REFERENCE_HEIGHT = 9933

# Priority ordering for label tiers (lower number = placed first, wins collisions)
TIER_PRIORITY = {
    FontTier.TITLE: 0,
    FontTier.SUBTITLE: 1,
    FontTier.DISTRICT: 2,
    FontTier.ROAD_MAJOR: 3,
    FontTier.WATER: 4,
    FontTier.PARK: 5,
    FontTier.ROAD_MINOR: 6,
}

# Candidate system font paths to try, in preference order
_SYSTEM_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNSText.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


class TypographyService:
    """Service for extracting, placing, and rendering map labels.

    Supports:
    - Road name labels rendered along curved paths
    - Area labels (water bodies, parks) placed at centroids
    - Title and subtitle placement at the top of the map
    - Collision detection to prevent overlapping labels
    - Halo/outline rendering for legibility on busy backgrounds
    """

    def __init__(self, settings: TypographySettings):
        """Initialize the typography service.

        Args:
            settings: Typography configuration controlling which labels
                are enabled, font scaling, halo width, etc.
        """
        self.settings = settings
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._system_font_path: Optional[str] = self._find_system_font()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_labels(
        self,
        osm_data,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        distortion=None,
        rotation_degrees: float = 0.0,
    ) -> list[Union[Label, TextPath]]:
        """Extract all labels from OSM data, converting GPS to pixel coordinates.

        Processes roads, water bodies, and parks from the OSM data to produce
        a prioritised list of labels. Road names become ``TextPath`` objects
        following the road geometry; water and park names become point
        ``Label`` objects positioned at the feature centroid.

        Args:
            osm_data: ``OSMData`` instance with roads, water, parks GeoDataFrames.
            bbox: Geographic bounding box of the map.
            image_size: ``(width, height)`` of the output image in pixels.
            distortion: Optional ``DistortionService`` for non-linear coordinate
                mapping.
            rotation_degrees: Map rotation in degrees (clockwise from north).

        Returns:
            Combined list of ``Label`` and ``TextPath`` objects sorted by
            tier priority, capped at ``settings.max_labels``.
        """
        all_labels: list[Union[Label, TextPath]] = []

        # Title and subtitle always go first (highest priority)
        if self.settings.title_text:
            title_label = Label(
                text=self.settings.title_text,
                tier=FontTier.TITLE,
                # Place at top-centre of image (pixel coords stored as lat/lon
                # fields -- they will be overridden during rendering)
                latitude=bbox.north,
                longitude=(bbox.east + bbox.west) / 2,
            )
            all_labels.append(title_label)

        if self.settings.subtitle_text:
            subtitle_label = Label(
                text=self.settings.subtitle_text,
                tier=FontTier.SUBTITLE,
                latitude=bbox.north,
                longitude=(bbox.east + bbox.west) / 2,
            )
            all_labels.append(subtitle_label)

        # Road labels
        if self.settings.road_labels and osm_data.roads is not None and len(osm_data.roads) > 0:
            road_labels = self._extract_road_labels(
                osm_data.roads, bbox, image_size, distortion, rotation_degrees
            )
            all_labels.extend(road_labels)

        # Water labels
        if self.settings.water_labels and osm_data.water is not None and len(osm_data.water) > 0:
            water_labels = self._extract_area_labels(
                osm_data.water, FontTier.WATER, bbox, image_size, distortion, rotation_degrees
            )
            all_labels.extend(water_labels)

        # Park labels
        if self.settings.park_labels and osm_data.parks is not None and len(osm_data.parks) > 0:
            park_labels = self._extract_area_labels(
                osm_data.parks, FontTier.PARK, bbox, image_size, distortion, rotation_degrees
            )
            all_labels.extend(park_labels)

        # Sort by tier priority so higher-priority labels are placed first
        all_labels.sort(key=lambda lbl: TIER_PRIORITY.get(lbl.tier, 99))

        # Cap at max_labels
        if len(all_labels) > self.settings.max_labels:
            all_labels = all_labels[: self.settings.max_labels]

        logger.info("Extracted %d labels from OSM data", len(all_labels))
        return all_labels

    def render_labels(
        self,
        image: Image.Image,
        labels: list[Union[Label, TextPath]],
        perspective_params: Optional[dict] = None,
    ) -> Image.Image:
        """Render all labels onto the image.

        Labels are drawn in tier-priority order. Collision detection prevents
        overlapping labels -- if a label would overlap an already-placed one
        it is silently skipped.

        Args:
            image: PIL Image to render onto (not modified in-place).
            labels: List of ``Label`` and ``TextPath`` objects to render.
            perspective_params: Optional dict with ``convergence``,
                ``vertical_scale``, ``horizon_margin`` keys for coordinate
                transformation (not currently applied to individual labels
                but reserved for future perspective-aware placement).

        Returns:
            New PIL Image with labels rendered.
        """
        result = image.copy().convert("RGBA")
        image_size = result.size  # (width, height)
        draw = ImageDraw.Draw(result)

        # Track occupied rectangles for collision avoidance
        occupied_rects: list[tuple[int, int, int, int]] = []

        for label_obj in labels:
            if isinstance(label_obj, TextPath):
                font = self._get_font(label_obj.tier, label_obj.font_size)
                font_size = self._calculate_font_size(label_obj.tier, image_size)
                if label_obj.font_size is not None:
                    font_size = label_obj.font_size
                font = self._get_font(label_obj.tier, font_size)

                placed = self._render_text_path(
                    result, draw, label_obj, font, image_size, occupied_rects
                )
                if not placed:
                    logger.debug("Skipped road label '%s' due to collision", label_obj.text)

            elif isinstance(label_obj, Label):
                font_size = self._calculate_font_size(label_obj.tier, image_size)
                if label_obj.font_size is not None:
                    font_size = label_obj.font_size
                font = self._get_font(label_obj.tier, font_size)

                placed = self._render_label(
                    result, draw, label_obj, font, image_size, occupied_rects
                )
                if not placed:
                    logger.debug("Skipped label '%s' due to collision", label_obj.text)

        # Flatten back to RGB
        background = Image.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])
        # Composite onto the original image to preserve the map underneath
        final = image.copy().convert("RGBA")
        final = Image.alpha_composite(final, result)
        return final.convert("RGB")

    # ------------------------------------------------------------------
    # Label extraction helpers
    # ------------------------------------------------------------------

    def _extract_road_labels(
        self,
        roads_gdf,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        distortion=None,
        rotation_degrees: float = 0.0,
    ) -> list[TextPath]:
        """Extract road name labels as TextPaths along road geometry.

        Groups roads by name and picks the longest segment for each unique
        name. Points are sampled along the geometry at approximately 20-pixel
        intervals for smooth curved text rendering.

        Args:
            roads_gdf: GeoDataFrame with ``name``, ``road_class``, ``geometry``.
            bbox: Geographic bounding box.
            image_size: ``(width, height)`` in pixels.
            distortion: Optional ``DistortionService``.
            rotation_degrees: Map rotation in degrees.

        Returns:
            List of ``TextPath`` objects.
        """
        # Guard against missing 'name' column
        if "name" not in roads_gdf.columns:
            logger.warning("Roads GeoDataFrame has no 'name' column; skipping road labels")
            return []

        text_paths: list[TextPath] = []

        # Drop roads without a name
        named_roads = roads_gdf[roads_gdf["name"].notna()].copy()
        if len(named_roads) == 0:
            return []

        # Determine road class column
        has_road_class = "road_class" in named_roads.columns

        # Group by name -- for each unique road name, pick the longest geometry
        grouped = named_roads.groupby("name")

        for road_name, group in grouped:
            road_name_str = str(road_name).strip()
            if not road_name_str:
                continue

            # Find the longest geometry in this group
            best_geom = None
            best_length = 0.0
            best_road_class = "minor"

            for _, row in group.iterrows():
                geom = row.geometry
                lines = self._geometry_to_lines(geom)
                for line in lines:
                    if line.length > best_length:
                        best_length = line.length
                        best_geom = line
                        if has_road_class:
                            best_road_class = row.get("road_class", "minor")

            if best_geom is None:
                continue

            # Convert GPS coordinates along the line to pixel coordinates
            pixel_points = self._sample_line_to_pixels(
                best_geom, bbox, image_size, distortion, sample_distance_px=20
            )

            if len(pixel_points) < 2:
                continue

            # Calculate pixel path length
            path_length_px = self._pixel_path_length(pixel_points)
            if path_length_px < self.settings.min_road_length_px:
                continue

            # Determine tier
            tier = FontTier.ROAD_MAJOR if best_road_class == "major" else FontTier.ROAD_MINOR

            text_path = TextPath(
                text=road_name_str,
                tier=tier,
                points=pixel_points,  # stored as (x, y) pixel tuples
            )
            text_paths.append(text_path)

        return text_paths

    def _extract_area_labels(
        self,
        gdf,
        tier: FontTier,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        distortion=None,
        rotation_degrees: float = 0.0,
    ) -> list[Label]:
        """Extract labels for area features (water, parks) using centroid.

        Args:
            gdf: GeoDataFrame with ``name`` and ``geometry`` columns.
            tier: The ``FontTier`` to assign (e.g. WATER, PARK).
            bbox: Geographic bounding box.
            image_size: ``(width, height)`` in pixels.
            distortion: Optional ``DistortionService``.
            rotation_degrees: Map rotation in degrees.

        Returns:
            List of ``Label`` objects positioned at feature centroids.
        """
        labels: list[Label] = []

        if "name" not in gdf.columns:
            return []

        named_features = gdf[gdf["name"].notna()].copy()
        if len(named_features) == 0:
            return []

        # De-duplicate by name -- keep the largest feature for each name
        seen_names: set[str] = set()

        # Sort by geometry area descending so we process larger features first
        try:
            named_features = named_features.copy()
            named_features["_area"] = named_features.geometry.area
            named_features = named_features.sort_values("_area", ascending=False)
        except Exception:
            pass  # If area calculation fails, proceed unsorted

        for _, row in named_features.iterrows():
            name = str(row["name"]).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            geom = row.geometry
            try:
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
            except Exception:
                continue

            # Check that centroid is within the bounding box
            if not (bbox.south <= lat <= bbox.north and bbox.west <= lon <= bbox.east):
                continue

            # Convert to pixel to verify it falls on-screen
            px, py = gps_to_pixel(lat, lon, bbox, image_size, distortion=distortion)
            width, height = image_size
            if px < 0 or px >= width or py < 0 or py >= height:
                continue

            label = Label(
                text=name,
                tier=tier,
                latitude=lat,
                longitude=lon,
            )
            labels.append(label)

        return labels

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_text_path(
        self,
        image: Image.Image,
        draw: ImageDraw.Draw,
        text_path: TextPath,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        image_size: tuple[int, int],
        occupied_rects: list[tuple[int, int, int, int]],
    ) -> bool:
        """Render text along a curved path (for road names).

        For each character in the label text:
        1. Interpolate position along the path.
        2. Calculate the tangent angle at that point.
        3. Create a small RGBA image with the character.
        4. Rotate by the tangent angle.
        5. Paste onto the main image.

        A halo effect is applied by rendering the text in white offset by
        1-2 px in 8 compass directions before rendering the coloured text.

        Args:
            image: Target RGBA image to paste characters onto.
            draw: ``ImageDraw`` for the target image (unused for text-path
                rendering but kept for API consistency).
            text_path: The ``TextPath`` to render.
            font: PIL font to use.
            image_size: ``(width, height)`` in pixels.
            occupied_rects: Mutable list of occupied bounding rectangles.

        Returns:
            ``True`` if the label was placed, ``False`` if skipped due to
            collision.
        """
        points = text_path.points
        if len(points) < 2:
            return False

        text = text_path.text
        color = text_path.color or FONT_COLORS.get(text_path.tier, "#4A4A4A")
        halo_width = self.settings.halo_width

        # Compute cumulative distances along the pixel path
        cum_dists = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            cum_dists.append(cum_dists[-1] + math.hypot(dx, dy))

        total_length = cum_dists[-1]
        if total_length < 1:
            return False

        # Measure total text width
        char_widths = []
        for ch in text:
            try:
                bbox_ch = font.getbbox(ch)
                w = bbox_ch[2] - bbox_ch[0]
            except Exception:
                w = 10
            char_widths.append(w)

        total_text_width = sum(char_widths)

        # If text is longer than path, skip
        if total_text_width > total_length * 0.95:
            return False

        # Centre the text along the path
        start_offset = (total_length - total_text_width) / 2

        # Decide text direction: if the path goes predominantly right-to-left,
        # reverse the points so text reads left-to-right
        first_pt = points[0]
        last_pt = points[-1]
        if first_pt[0] > last_pt[0]:
            points = list(reversed(points))
            cum_dists_rev = [0.0]
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                cum_dists_rev.append(cum_dists_rev[-1] + math.hypot(dx, dy))
            cum_dists = cum_dists_rev

        # Pre-check collision using an approximate bounding rectangle of the
        # entire text path
        approx_rect = self._text_path_bounding_rect(points, char_widths, font)
        if self._check_label_collision(approx_rect, occupied_rects, padding=5):
            return False

        # Render each character
        current_dist = start_offset
        for i, ch in enumerate(text):
            cw = char_widths[i]
            char_centre_dist = current_dist + cw / 2

            # Interpolate position along path
            px, py = self._interpolate_along_path(points, cum_dists, char_centre_dist)

            # Calculate tangent angle
            angle = self._tangent_angle_at(points, cum_dists, char_centre_dist)

            # Render character with halo
            self._render_halo_text(
                image,
                ch,
                (int(px), int(py)),
                font,
                color,
                halo_width,
                halo_color="#FFFFFF",
                rotation=angle,
            )

            current_dist += cw

        # Record the occupied rectangle
        occupied_rects.append(approx_rect)
        return True

    def _render_label(
        self,
        image: Image.Image,
        draw: ImageDraw.Draw,
        label: Label,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        image_size: tuple[int, int],
        occupied_rects: list[tuple[int, int, int, int]],
    ) -> bool:
        """Render a single point label (for title, subtitle, districts, water, parks).

        Title and subtitle labels are placed at a fixed position near the top
        of the image, centred horizontally. Other labels are placed at the
        pixel position corresponding to their GPS coordinates.

        Args:
            image: Target RGBA image to render onto.
            draw: ``ImageDraw`` for the target image.
            label: The ``Label`` to render.
            font: PIL font to use.
            image_size: ``(width, height)`` in pixels.
            occupied_rects: Mutable list of occupied bounding rectangles.

        Returns:
            ``True`` if the label was placed, ``False`` if skipped due to
            collision.
        """
        width, height = image_size
        text = label.text
        color = label.color or FONT_COLORS.get(label.tier, "#4A4A4A")
        halo_width = self.settings.halo_width

        # Measure text dimensions
        try:
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w = len(text) * 10
            text_h = 14

        # Determine position
        if label.tier == FontTier.TITLE:
            # Title: centred horizontally, near the top
            px = width // 2 - text_w // 2
            py = int(height * 0.03)
        elif label.tier == FontTier.SUBTITLE:
            # Subtitle: centred horizontally, just below title area
            px = width // 2 - text_w // 2
            py = int(height * 0.03) + self._calculate_font_size(FontTier.TITLE, image_size) + 10
        else:
            # Geographic labels: use GPS-to-pixel
            px, py = gps_to_pixel(
                label.latitude,
                label.longitude,
                # We need the bbox -- reconstruct a minimal one from image context
                # The bbox is not stored on Label, so we rely on the fact that
                # _extract_area_labels already verified on-screen placement.
                # Here we re-derive pixel position using the label's lat/lon.
                self._current_bbox,
                image_size,
                distortion=self._current_distortion,
            )
            # Centre the text on the point
            px = px - text_w // 2
            py = py - text_h // 2

        # Build the label rectangle
        label_rect = (px, py, px + text_w, py + text_h)

        # Collision check
        if self._check_label_collision(label_rect, occupied_rects, padding=5):
            return False

        # Clamp to image bounds
        if px < 0 or py < 0 or px + text_w > width or py + text_h > height:
            # Allow title/subtitle to exceed slightly, but skip others
            if label.tier not in (FontTier.TITLE, FontTier.SUBTITLE):
                return False

        # Render with halo
        self._render_halo_text(
            image,
            text,
            (int(px), int(py)),
            font,
            color,
            halo_width,
            halo_color="#FFFFFF",
            rotation=label.rotation,
        )

        occupied_rects.append(label_rect)
        return True

    def _render_halo_text(
        self,
        image: Image.Image,
        text: str,
        position: tuple[int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        color: str,
        halo_width: int,
        halo_color: str = "#FFFFFF",
        rotation: float = 0.0,
    ) -> None:
        """Render text with a halo/outline for legibility.

        The halo is created by rendering the text in ``halo_color`` offset
        by ``halo_width`` pixels in 8 compass directions (N, NE, E, SE, S,
        SW, W, NW), then rendering the coloured text on top.

        If ``rotation`` is non-zero the entire character image is rotated
        before pasting.

        Args:
            image: Target RGBA image.
            text: Text string to render (may be a single character for
                curved-path rendering).
            position: ``(x, y)`` top-left position on the image.
            font: PIL font to use.
            color: Hex colour string for the main text.
            halo_width: Pixel width of the halo outline.
            halo_color: Hex colour string for the halo.
            rotation: Rotation angle in degrees (counter-clockwise).
        """
        if not text.strip():
            return

        # Measure the text
        try:
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            # Offset to account for font bearing
            x_offset = -text_bbox[0]
            y_offset = -text_bbox[1]
        except Exception:
            text_w = len(text) * 10
            text_h = 14
            x_offset = 0
            y_offset = 0

        # Create a small RGBA canvas with padding for the halo
        padding = halo_width + 2
        canvas_w = text_w + padding * 2
        canvas_h = text_h + padding * 2

        if canvas_w <= 0 or canvas_h <= 0:
            return

        char_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)

        draw_x = padding + x_offset
        draw_y = padding + y_offset

        # Draw halo: render text in 8 offset directions
        if halo_width > 0:
            offsets = [
                (0, -halo_width),   # N
                (halo_width, -halo_width),   # NE
                (halo_width, 0),    # E
                (halo_width, halo_width),    # SE
                (0, halo_width),    # S
                (-halo_width, halo_width),   # SW
                (-halo_width, 0),   # W
                (-halo_width, -halo_width),  # NW
            ]
            for dx, dy in offsets:
                char_draw.text(
                    (draw_x + dx, draw_y + dy),
                    text,
                    font=font,
                    fill=halo_color,
                )

        # Draw main text on top
        char_draw.text((draw_x, draw_y), text, font=font, fill=color)

        # Rotate if needed
        if abs(rotation) > 0.5:
            # Use bicubic resampling; expand=True so corners are not clipped
            char_img = char_img.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=True,
            )

        # Paste onto the main image, centred on the target position
        paste_x = position[0] - char_img.width // 2
        paste_y = position[1] - char_img.height // 2

        # Clip to image bounds to avoid errors
        img_w, img_h = image.size
        if (
            paste_x + char_img.width < 0
            or paste_y + char_img.height < 0
            or paste_x >= img_w
            or paste_y >= img_h
        ):
            return

        image.paste(char_img, (paste_x, paste_y), char_img)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_label_collision(
        self,
        new_rect: tuple[int, int, int, int],
        occupied_rects: list[tuple[int, int, int, int]],
        padding: int = 5,
    ) -> bool:
        """Check if a label rectangle collides with any existing labels.

        Args:
            new_rect: ``(x1, y1, x2, y2)`` bounding box of the new label.
            occupied_rects: List of already-placed label rectangles.
            padding: Extra pixels of clearance around each rectangle.

        Returns:
            ``True`` if there is a collision, ``False`` if placement is clear.
        """
        x1, y1, x2, y2 = new_rect
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        for ox1, oy1, ox2, oy2 in occupied_rects:
            # Axis-aligned bounding box overlap test
            if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                return True

        return False

    # ------------------------------------------------------------------
    # Font handling
    # ------------------------------------------------------------------

    def _get_font(
        self,
        tier: FontTier,
        size_override: Optional[int] = None,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get the appropriate font for a tier.

        Attempts to load a system TrueType font (DejaVu Sans, Liberation Sans,
        Arial, etc.). Falls back gracefully to the PIL built-in bitmap font
        if no TrueType font is available. Fonts are cached for reuse.

        Args:
            tier: The ``FontTier`` determining style/weight.
            size_override: If given, use this point size instead of the
                tier default.

        Returns:
            A PIL font object.
        """
        size = size_override if size_override is not None else FONT_SIZES[tier][0]
        cache_key = (tier.value, size)

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = self._load_font(size)
        self._font_cache[cache_key] = font
        return font

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load a font at the given size.

        Tries the cached system font path first, then each candidate path,
        then PIL's ``truetype`` with common font names, and finally falls
        back to the default bitmap font.

        Args:
            size: Point size.

        Returns:
            A PIL font object.
        """
        # Fast path: use the pre-discovered system font
        if self._system_font_path:
            try:
                return ImageFont.truetype(self._system_font_path, size)
            except Exception:
                pass

        # Try each candidate
        for path in _SYSTEM_FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

        # Try by font name (Pillow searches system paths)
        for name in ("DejaVuSans", "DejaVu Sans", "LiberationSans", "Arial", "Helvetica"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue

        # Ultimate fallback
        logger.warning("No TrueType font found; using PIL default bitmap font")
        return ImageFont.load_default()

    @staticmethod
    def _find_system_font() -> Optional[str]:
        """Probe the system for a usable TrueType font and return its path.

        Returns:
            Path string if found, ``None`` otherwise.
        """
        for path in _SYSTEM_FONT_CANDIDATES:
            try:
                ImageFont.truetype(path, 12)
                return path
            except Exception:
                continue

        # Try by name
        for name in ("DejaVuSans", "DejaVu Sans", "LiberationSans", "Arial", "Helvetica"):
            try:
                ImageFont.truetype(name, 12)
                return name
            except Exception:
                continue

        return None

    def _calculate_font_size(
        self,
        tier: FontTier,
        image_size: tuple[int, int],
    ) -> int:
        """Calculate appropriate font size based on tier and image dimensions.

        Scales the base font size from ``FONT_SIZES`` proportionally to the
        image dimensions relative to the A1-at-300-DPI reference size
        (7016 x 9933). The global ``font_scale`` multiplier from settings is
        also applied.

        Args:
            tier: The ``FontTier`` to look up base sizes for.
            image_size: ``(width, height)`` in pixels.

        Returns:
            Computed font size in points (clamped to at least 8).
        """
        min_size, max_size = FONT_SIZES[tier]
        base_size = (min_size + max_size) / 2

        width, height = image_size
        # Scale factor based on geometric mean of dimensions relative to reference
        ref_mean = math.sqrt(REFERENCE_WIDTH * REFERENCE_HEIGHT)
        img_mean = math.sqrt(width * height)
        scale = img_mean / ref_mean

        scaled = base_size * scale * self.settings.font_scale
        # Clamp between reasonable bounds
        result = max(8, min(int(scaled), max_size * 3))
        return result

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geometry_to_lines(geom) -> list[LineString]:
        """Convert a geometry to a list of LineString objects.

        Handles ``LineString``, ``MultiLineString``, and gracefully ignores
        unsupported geometry types.

        Args:
            geom: A shapely geometry object.

        Returns:
            List of ``LineString`` instances.
        """
        if isinstance(geom, LineString):
            return [geom]
        elif isinstance(geom, MultiLineString):
            return list(geom.geoms)
        else:
            return []

    def _sample_line_to_pixels(
        self,
        line: LineString,
        bbox: BoundingBox,
        image_size: tuple[int, int],
        distortion=None,
        sample_distance_px: int = 20,
    ) -> list[tuple[float, float]]:
        """Sample points along a LineString and convert to pixel coordinates.

        Points are sampled every ``sample_distance_px`` pixels (approximately)
        along the geographic line, plus the start and end points.

        Args:
            line: Shapely ``LineString`` in GPS coordinates (lon, lat).
            bbox: Geographic bounding box.
            image_size: ``(width, height)`` in pixels.
            distortion: Optional ``DistortionService``.
            sample_distance_px: Approximate spacing between samples in pixels.

        Returns:
            List of ``(x, y)`` pixel coordinate tuples.
        """
        if line.is_empty or line.length == 0:
            return []

        # First, convert endpoints to pixels to estimate pixel-space length
        coords = list(line.coords)
        start_px = gps_to_pixel(coords[0][1], coords[0][0], bbox, image_size, distortion=distortion)
        end_px = gps_to_pixel(coords[-1][1], coords[-1][0], bbox, image_size, distortion=distortion)
        approx_px_length = math.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])

        if approx_px_length < 1:
            return []

        # Determine number of samples
        num_samples = max(2, int(approx_px_length / sample_distance_px) + 1)

        # Sample at equal fractional distances along the line
        pixel_points: list[tuple[float, float]] = []
        for i in range(num_samples):
            frac = i / (num_samples - 1)
            point = line.interpolate(frac, normalized=True)
            lon, lat = point.x, point.y
            px, py = gps_to_pixel(lat, lon, bbox, image_size, distortion=distortion)
            pixel_points.append((float(px), float(py)))

        return pixel_points

    @staticmethod
    def _pixel_path_length(points: list[tuple[float, float]]) -> float:
        """Calculate the total length of a pixel path.

        Args:
            points: List of ``(x, y)`` pixel coordinates.

        Returns:
            Total path length in pixels.
        """
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += math.hypot(dx, dy)
        return total

    @staticmethod
    def _interpolate_along_path(
        points: list[tuple[float, float]],
        cum_dists: list[float],
        target_dist: float,
    ) -> tuple[float, float]:
        """Interpolate a position at a given distance along a polyline.

        Args:
            points: List of ``(x, y)`` coordinates.
            cum_dists: Cumulative distance at each point.
            target_dist: Distance along the path to interpolate at.

        Returns:
            ``(x, y)`` interpolated position.
        """
        if target_dist <= 0:
            return points[0]
        if target_dist >= cum_dists[-1]:
            return points[-1]

        # Find the segment containing target_dist
        for i in range(1, len(cum_dists)):
            if cum_dists[i] >= target_dist:
                seg_start = cum_dists[i - 1]
                seg_end = cum_dists[i]
                seg_len = seg_end - seg_start
                if seg_len < 1e-9:
                    return points[i]
                t = (target_dist - seg_start) / seg_len
                x = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
                y = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
                return (x, y)

        return points[-1]

    @staticmethod
    def _tangent_angle_at(
        points: list[tuple[float, float]],
        cum_dists: list[float],
        target_dist: float,
    ) -> float:
        """Calculate the tangent angle at a given distance along a polyline.

        The angle is measured in degrees, counter-clockwise from the positive
        X axis (suitable for PIL ``Image.rotate``).

        Args:
            points: List of ``(x, y)`` coordinates.
            cum_dists: Cumulative distance at each point.
            target_dist: Distance along the path.

        Returns:
            Angle in degrees (counter-clockwise from East).
        """
        # Sample two nearby points to compute a tangent
        delta = 2.0  # pixels
        d1 = max(0, target_dist - delta)
        d2 = min(cum_dists[-1], target_dist + delta)

        if d2 - d1 < 0.1:
            return 0.0

        # Interpolate
        p1_x, p1_y = TypographyService._interpolate_along_path(points, cum_dists, d1)
        p2_x, p2_y = TypographyService._interpolate_along_path(points, cum_dists, d2)

        dx = p2_x - p1_x
        dy = p2_y - p1_y

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0

        # atan2 gives angle from positive-X, counter-clockwise
        # PIL rotate is counter-clockwise, and image Y is inverted,
        # so we negate the angle.
        angle_rad = math.atan2(dy, dx)
        angle_deg = -math.degrees(angle_rad)
        return angle_deg

    def _text_path_bounding_rect(
        self,
        points: list[tuple[float, float]],
        char_widths: list[int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> tuple[int, int, int, int]:
        """Compute an approximate bounding rectangle for text along a path.

        Args:
            points: Pixel path points.
            char_widths: Width of each character in pixels.
            font: Font used for measuring height.

        Returns:
            ``(x1, y1, x2, y2)`` bounding rectangle.
        """
        if not points:
            return (0, 0, 0, 0)

        # Get approximate text height
        try:
            sample_bbox = font.getbbox("Ay")
            text_h = sample_bbox[3] - sample_bbox[1]
        except Exception:
            text_h = 14

        half_h = text_h // 2 + self.settings.halo_width + 4

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = int(min(xs)) - half_h
        y1 = int(min(ys)) - half_h
        x2 = int(max(xs)) + half_h
        y2 = int(max(ys)) + half_h

        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    def extract_and_render(
        self,
        image: Image.Image,
        osm_data,
        bbox: BoundingBox,
        distortion=None,
        rotation_degrees: float = 0.0,
        perspective_params: Optional[dict] = None,
    ) -> Image.Image:
        """Convenience method: extract labels and render them in one step.

        Stores ``bbox`` and ``distortion`` on the instance so that
        ``_render_label`` can convert GPS coordinates to pixel positions
        during rendering.

        Args:
            image: PIL Image to render labels onto.
            osm_data: ``OSMData`` instance.
            bbox: Geographic bounding box.
            distortion: Optional ``DistortionService``.
            rotation_degrees: Map rotation in degrees.
            perspective_params: Optional perspective parameters dict.

        Returns:
            New PIL Image with labels rendered.
        """
        if not self.settings.enabled:
            return image

        image_size = image.size

        # Store bbox and distortion for use in _render_label
        self._current_bbox = bbox
        self._current_distortion = distortion

        labels = self.extract_labels(
            osm_data, bbox, image_size, distortion=distortion, rotation_degrees=rotation_degrees
        )

        if not labels:
            return image

        result = self.render_labels(image, labels, perspective_params=perspective_params)

        # Clean up temporary state
        self._current_bbox = None
        self._current_distortion = None

        return result
