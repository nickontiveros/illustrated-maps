"""Palette enforcement service.

Extracts, applies, and enforces color palettes for illustrated map tiles.
Supports named presets, extraction from reference images, prompt generation
for Gemini, post-processing clamping, and compliance analysis.

Works in CIE-LAB color space for perceptually accurate color matching.
"""

import numpy as np
from PIL import Image

from .color_consistency_service import ColorConsistencyService


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, list[str]] = {
    "vintage_tourist": [
        "#E8D8C0",  # warm cream
        "#D5B896",  # tan
        "#7CB342",  # sage green
        "#4A90D9",  # sky blue
        "#F0E8E0",  # light ivory
        "#A89080",  # warm gray
    ],
    "modern_pop": [
        "#FF6B6B",  # coral red
        "#4ECDC4",  # teal
        "#FFE66D",  # sunny yellow
        "#2C3E50",  # dark navy
        "#95E1D3",  # mint
        "#F38181",  # salmon pink
    ],
    "ink_wash": [
        "#1A1A1A",  # near black
        "#4A4A4A",  # dark gray
        "#8A8A8A",  # medium gray
        "#C0C0C0",  # silver
        "#F0F0F0",  # near white
    ],
}

# Human-readable color descriptions keyed by hex value, used for prompt
# construction so Gemini gets meaningful names alongside the hex codes.
_COLOR_NAMES: dict[str, str] = {
    "#E8D8C0": "warm cream",
    "#D5B896": "tan",
    "#7CB342": "sage green",
    "#4A90D9": "sky blue",
    "#F0E8E0": "light ivory",
    "#A89080": "warm gray",
    "#FF6B6B": "coral red",
    "#4ECDC4": "teal",
    "#FFE66D": "sunny yellow",
    "#2C3E50": "dark navy",
    "#95E1D3": "mint",
    "#F38181": "salmon pink",
    "#1A1A1A": "near black",
    "#4A4A4A": "dark gray",
    "#8A8A8A": "medium gray",
    "#C0C0C0": "silver",
    "#F0F0F0": "near white",
}

# LAB distance threshold: pixels farther than this from every palette color
# are considered outliers in compliance analysis.  A value of 25 corresponds
# roughly to a clearly noticeable color difference.
_OUTLIER_THRESHOLD = 25.0


class PaletteService:
    """Enforce a color palette on generated map tiles.

    Workflow:
      1. Set a palette (from a preset, extracted from a reference image,
         or provided as a list of hex strings).
      2. Call ``build_prompt_instruction()`` to get a text instruction that
         can be prepended to the Gemini generation prompt.
      3. After generation, call ``clamp_to_palette()`` to post-process the
         tile so every pixel is nudged toward the nearest palette color.
      4. Optionally call ``analyze_compliance()`` to measure how well a
         tile already matches the palette.
    """

    def __init__(
        self,
        palette: list[str] | None = None,
        enforcement_strength: float = 0.5,
    ):
        """Initialize the palette service.

        Args:
            palette: List of hex color strings (e.g. ``["#FF0000", "#00FF00"]``).
                     If *None*, the service is created without a palette; one
                     must be set later via ``from_preset`` or
                     ``extract_palette_from_image``.
            enforcement_strength: Blending factor for ``clamp_to_palette``.
                0.0 leaves the image unchanged, 1.0 fully snaps every pixel
                to the nearest palette color.
        """
        self.palette: list[str] = [c.upper() for c in palette] if palette else []
        self.enforcement_strength = max(0.0, min(1.0, enforcement_strength))

        # Pre-compute LAB values for the palette so we don't redo them on
        # every call.  Stored as a (N, 3) float64 array.
        self._palette_lab: np.ndarray = self._compute_palette_lab()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls, preset_name: str, strength: float = 0.5
    ) -> "PaletteService":
        """Create a PaletteService from a named preset.

        Args:
            preset_name: One of ``"vintage_tourist"``, ``"modern_pop"``,
                         ``"ink_wash"``.
            strength: Enforcement strength (0.0 -- 1.0).

        Returns:
            A configured ``PaletteService``.

        Raises:
            ValueError: If *preset_name* is not a known preset.
        """
        if preset_name not in PRESETS:
            available = ", ".join(sorted(PRESETS.keys()))
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {available}"
            )
        return cls(palette=PRESETS[preset_name], enforcement_strength=strength)

    # ------------------------------------------------------------------
    # Palette extraction
    # ------------------------------------------------------------------

    def extract_palette_from_image(
        self, image: Image.Image, n_colors: int = 8
    ) -> list[str]:
        """Extract a palette from an image and store it on this service.

        Uses ``ColorConsistencyService.extract_palette()`` (K-means in RGB
        space) under the hood, then converts the resulting RGB tuples to
        hex strings.

        Args:
            image: PIL Image to analyse.
            n_colors: Number of dominant colors to extract.

        Returns:
            List of hex color strings (e.g. ``["#E8D8C0", "#7CB342", ...]``).
        """
        ccs = ColorConsistencyService()
        rgb_tuples = ccs.extract_palette(image, n_colors=n_colors)
        hex_colors = [self.rgb_to_hex(r, g, b) for r, g, b in rgb_tuples]
        self.palette = hex_colors
        self._palette_lab = self._compute_palette_lab()
        return hex_colors

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt_instruction(self) -> str:
        """Build a Gemini prompt instruction string listing the palette.

        The instruction asks the model to restrict its color usage to the
        palette colors and derive any shading or variation from them.

        Returns:
            A multi-sentence instruction string.  If the palette is empty
            an empty string is returned.
        """
        if not self.palette:
            return ""

        color_parts: list[str] = []
        for hex_color in self.palette:
            name = _COLOR_NAMES.get(hex_color)
            if name:
                color_parts.append(f"{hex_color} ({name})")
            else:
                color_parts.append(hex_color)

        colors_str = ", ".join(color_parts)
        return (
            f"Use ONLY these colors in your illustration: {colors_str}. "
            "All variations and shading should be derived from these base colors."
        )

    # ------------------------------------------------------------------
    # Post-processing: clamp to palette
    # ------------------------------------------------------------------

    def clamp_to_palette(self, image: Image.Image) -> Image.Image:
        """Clamp every pixel in *image* to the nearest palette color.

        The operation is performed in CIE-LAB space for perceptual accuracy.
        The result is blended with the original image according to
        ``enforcement_strength``:

            output = (1 - strength) * original + strength * clamped

        The alpha channel, if present, is preserved unchanged.

        Args:
            image: PIL Image (RGB or RGBA).

        Returns:
            A new PIL Image with colors nudged toward the palette.

        Raises:
            ValueError: If the palette is empty.
        """
        if not self.palette:
            raise ValueError("Cannot clamp: palette is empty.")

        has_alpha = image.mode == "RGBA"
        alpha_channel: np.ndarray | None = None

        img_rgb = np.array(image.convert("RGB"), dtype=np.float64)
        if has_alpha:
            alpha_channel = np.array(image)[:, :, 3]

        h, w, _ = img_rgb.shape

        # Convert image pixels to LAB
        img_lab = ColorConsistencyService.rgb_to_lab(img_rgb)  # (H, W, 3)
        pixels_lab = img_lab.reshape(-1, 3)  # (N, 3)

        # Compute distance from each pixel to each palette color
        # pixels_lab: (N, 3), palette_lab: (P, 3)
        # diff: (N, P, 3)
        palette_lab = self._palette_lab  # (P, 3)
        diff = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)  # (N, P)

        # Find nearest palette color index for each pixel
        nearest_idx = np.argmin(distances, axis=1)  # (N,)

        # Build clamped LAB image
        clamped_lab = palette_lab[nearest_idx]  # (N, 3)

        # Convert clamped LAB back to RGB
        clamped_rgb = ColorConsistencyService.lab_to_rgb(
            clamped_lab.reshape(h, w, 3)
        )  # (H, W, 3)

        # Blend with original
        blended = (
            (1.0 - self.enforcement_strength) * img_rgb
            + self.enforcement_strength * clamped_rgb
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if has_alpha and alpha_channel is not None:
            result = np.dstack([blended, alpha_channel])
            return Image.fromarray(result, "RGBA")
        return Image.fromarray(blended, "RGB")

    # ------------------------------------------------------------------
    # Compliance analysis
    # ------------------------------------------------------------------

    def analyze_compliance(self, image: Image.Image) -> dict:
        """Analyse how well *image* matches the target palette.

        All distance calculations use CIE-LAB Euclidean distance.

        Args:
            image: PIL Image to analyse.

        Returns:
            A dict with the following keys:

            - ``mean_distance`` (float): Average LAB Euclidean distance from
              each pixel to its nearest palette color.
            - ``compliance_score`` (float): 0.0 -- 1.0 where 1.0 means every
              pixel sits exactly on a palette color.  Computed as
              ``1 / (1 + mean_distance)``.
            - ``per_color_coverage`` (dict[str, float]): Mapping from each
              palette hex color to the fraction of image pixels (0.0 -- 1.0)
              for which it is the nearest palette color.
            - ``outlier_percentage`` (float): Fraction (0.0 -- 1.0) of pixels
              whose LAB distance to the nearest palette color exceeds the
              outlier threshold.

        Raises:
            ValueError: If the palette is empty.
        """
        if not self.palette:
            raise ValueError("Cannot analyse compliance: palette is empty.")

        img_rgb = np.array(image.convert("RGB"), dtype=np.float64)
        img_lab = ColorConsistencyService.rgb_to_lab(img_rgb)
        pixels_lab = img_lab.reshape(-1, 3)

        palette_lab = self._palette_lab
        diff = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
        sq_distances = np.sum(diff ** 2, axis=2)  # (N, P)

        nearest_idx = np.argmin(sq_distances, axis=1)
        nearest_dist = np.sqrt(sq_distances[np.arange(len(nearest_idx)), nearest_idx])

        mean_distance = float(np.mean(nearest_dist))
        compliance_score = 1.0 / (1.0 + mean_distance)

        n_pixels = len(nearest_idx)

        # Per-color coverage
        per_color_coverage: dict[str, float] = {}
        for i, hex_color in enumerate(self.palette):
            count = int(np.sum(nearest_idx == i))
            per_color_coverage[hex_color] = count / n_pixels

        # Outlier percentage
        outlier_count = int(np.sum(nearest_dist > _OUTLIER_THRESHOLD))
        outlier_percentage = outlier_count / n_pixels

        return {
            "mean_distance": mean_distance,
            "compliance_score": compliance_score,
            "per_color_coverage": per_color_coverage,
            "outlier_percentage": outlier_percentage,
        }

    # ------------------------------------------------------------------
    # Static colour-space helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert a hex color string to an (R, G, B) tuple.

        Accepts strings with or without a leading ``#``.

        Args:
            hex_color: e.g. ``"#FF0000"`` or ``"FF0000"``.

        Returns:
            Tuple of ints in 0 -- 255.
        """
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert an (R, G, B) tuple to an uppercase hex string.

        Args:
            r: Red channel (0 -- 255).
            g: Green channel (0 -- 255).
            b: Blue channel (0 -- 255).

        Returns:
            Hex string, e.g. ``"#FF0000"``.
        """
        return f"#{r:02X}{g:02X}{b:02X}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_palette_lab(self) -> np.ndarray:
        """Convert the current hex palette to a (N, 3) LAB array."""
        if not self.palette:
            return np.empty((0, 3), dtype=np.float64)

        rgb_array = np.array(
            [self.hex_to_rgb(c) for c in self.palette], dtype=np.float64
        )  # (N, 3)
        # rgb_to_lab expects (..., 3), so shape (N, 3) works directly.
        return ColorConsistencyService.rgb_to_lab(rgb_array)
