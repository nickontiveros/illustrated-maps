"""Cross-tile color consistency service.

Provides histogram matching and global color grading to reduce
palette and style drift between independently generated tiles.
"""

import numpy as np
from PIL import Image
from typing import Optional


# D65 illuminant reference white point
_D65_X = 0.95047
_D65_Y = 1.00000
_D65_Z = 1.08883

# CIE LAB constants
_LAB_EPSILON = 0.008856
_LAB_KAPPA = 903.3


class ColorConsistencyService:
    """Service for enforcing color consistency across tiles."""

    def __init__(self, strength: float = 0.5):
        """Initialize.

        Args:
            strength: Blend strength 0.0 (off) to 1.0 (full correction).
        """
        self.strength = max(0.0, min(1.0, strength))

    # ------------------------------------------------------------------
    # Histogram matching
    # ------------------------------------------------------------------

    def histogram_match(
        self, source: Image.Image, reference: Image.Image
    ) -> Image.Image:
        """Match the histogram of source to reference.

        Per-channel CDF transfer: for each RGB channel, map source pixel values
        through the inverse CDF of the reference tile using np.interp() with
        cumulative histograms.

        Args:
            source: Image to adjust.
            reference: Target image to match.

        Returns:
            Color-matched image (blended with original by self.strength).
        """
        has_alpha = source.mode == "RGBA"
        alpha_channel: Optional[np.ndarray] = None

        # Convert to RGB arrays
        src_rgb = np.array(source.convert("RGB"), dtype=np.float64)
        ref_rgb = np.array(reference.convert("RGB"), dtype=np.float64)

        if has_alpha:
            alpha_channel = np.array(source)[:, :, 3]

        matched = np.empty_like(src_rgb)
        for ch in range(3):
            matched[:, :, ch] = self._match_channel(
                src_rgb[:, :, ch], ref_rgb[:, :, ch]
            )

        # Blend with original according to strength
        blended = (1.0 - self.strength) * src_rgb + self.strength * matched
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if has_alpha and alpha_channel is not None:
            result = np.dstack([blended, alpha_channel])
            return Image.fromarray(result, "RGBA")
        return Image.fromarray(blended, "RGB")

    def _match_channel(
        self, source_ch: np.ndarray, reference_ch: np.ndarray
    ) -> np.ndarray:
        """Match histogram of a single channel.

        Args:
            source_ch: 2-D array of source pixel values (0-255 float).
            reference_ch: 2-D array of reference pixel values (0-255 float).

        Returns:
            2-D array with matched values.
        """
        # Flatten
        src_flat = source_ch.ravel()
        ref_flat = reference_ch.ravel()

        # Compute histograms (256 bins, range 0-255)
        src_hist, bin_edges = np.histogram(src_flat, bins=256, range=(0, 255))
        ref_hist, _ = np.histogram(ref_flat, bins=256, range=(0, 255))

        # Compute CDFs
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]  # normalise to [0, 1]

        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # Build mapping: for each source intensity, find the reference intensity
        # whose CDF value is closest.  np.interp maps source CDF values through
        # the inverse of the reference CDF.
        bin_centers = np.arange(256)
        mapping = np.interp(src_cdf, ref_cdf, bin_centers)

        # Apply mapping
        src_indices = np.clip(src_flat, 0, 255).astype(np.int64)
        matched_flat = mapping[src_indices]
        return matched_flat.reshape(source_ch.shape)

    def match_tiles_to_reference(
        self,
        tiles: list[Image.Image],
        reference: Image.Image,
    ) -> list[Image.Image]:
        """Match all tiles to a single reference tile.

        Args:
            tiles: List of tile images to adjust.
            reference: Reference tile (typically the central tile).

        Returns:
            List of color-matched tiles.
        """
        return [self.histogram_match(tile, reference) for tile in tiles]

    # ------------------------------------------------------------------
    # 3-D Color LUT
    # ------------------------------------------------------------------

    def build_color_lut(
        self, reference: Image.Image, lut_size: int = 32
    ) -> np.ndarray:
        """Build a 3D color lookup table from a reference image.

        Creates a reduced-resolution 3D LUT (default 32x32x32) that maps
        input RGB values to the average output color seen in the reference.

        The LUT is constructed by quantising every pixel in the reference image
        into one of ``lut_size`` bins per channel, then computing the mean colour
        within each bin.  Bins that receive no pixels are filled with the
        identity mapping (i.e. the bin's own centre colour).

        Args:
            reference: Reference image to build LUT from.
            lut_size: Size of each LUT dimension (32 = 32x32x32 = 32768 entries).

        Returns:
            LUT as numpy array of shape (lut_size, lut_size, lut_size, 3),
            with float64 values in range 0-255.
        """
        ref_rgb = np.array(reference.convert("RGB"), dtype=np.float64)
        pixels = ref_rgb.reshape(-1, 3)

        # Quantise each channel to bin indices
        bin_width = 256.0 / lut_size
        indices = np.clip((pixels / bin_width).astype(np.int64), 0, lut_size - 1)

        # Accumulate sums and counts per bin
        lut_sum = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float64)
        lut_count = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)

        # Use np.add.at for unbuffered accumulation
        flat_idx = (
            indices[:, 0] * lut_size * lut_size
            + indices[:, 1] * lut_size
            + indices[:, 2]
        )
        flat_lut_sum = lut_sum.reshape(-1, 3)
        flat_lut_count = lut_count.ravel()

        np.add.at(flat_lut_sum, flat_idx, pixels)
        np.add.at(flat_lut_count, flat_idx, 1.0)

        # Compute mean colour per bin
        mask = lut_count > 0
        lut_sum[mask] /= lut_count[mask, np.newaxis]

        # Fill empty bins with identity (the bin centre colour)
        r_centers = (np.arange(lut_size) + 0.5) * bin_width
        g_centers = r_centers.copy()
        b_centers = r_centers.copy()
        rr, gg, bb = np.meshgrid(r_centers, g_centers, b_centers, indexing="ij")
        identity = np.stack([rr, gg, bb], axis=-1)

        lut_sum[~mask] = identity[~mask]

        return lut_sum

    def apply_color_lut(self, image: Image.Image, lut: np.ndarray) -> Image.Image:
        """Apply a 3D color LUT to an image using trilinear interpolation.

        Blended with original by self.strength.

        Args:
            image: Image to apply LUT to.
            lut: 3D LUT from build_color_lut().

        Returns:
            Color-graded image.
        """
        has_alpha = image.mode == "RGBA"
        alpha_channel: Optional[np.ndarray] = None

        img_rgb = np.array(image.convert("RGB"), dtype=np.float64)
        if has_alpha:
            alpha_channel = np.array(image)[:, :, 3]

        h, w, _ = img_rgb.shape
        pixels = img_rgb.reshape(-1, 3)

        lut_size = lut.shape[0]
        bin_width = 256.0 / lut_size

        # Continuous coordinates into the LUT
        coords = pixels / bin_width - 0.5
        coords = np.clip(coords, 0, lut_size - 1 - 1e-6)

        # Integer (floor) and fractional parts
        low = coords.astype(np.int64)
        low = np.clip(low, 0, lut_size - 2)
        frac = coords - low

        # The eight corners for trilinear interpolation
        r0, g0, b0 = low[:, 0], low[:, 1], low[:, 2]
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

        fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]

        # Look up the eight corner values
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        # Trilinear interpolation
        fr = fr[:, np.newaxis]
        fg = fg[:, np.newaxis]
        fb = fb[:, np.newaxis]

        c00 = c000 * (1 - fb) + c001 * fb
        c01 = c010 * (1 - fb) + c011 * fb
        c10 = c100 * (1 - fb) + c101 * fb
        c11 = c110 * (1 - fb) + c111 * fb

        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg

        mapped = c0 * (1 - fr) + c1 * fr
        mapped = mapped.reshape(h, w, 3)

        # Blend with original
        blended = (1.0 - self.strength) * img_rgb + self.strength * mapped
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if has_alpha and alpha_channel is not None:
            result = np.dstack([blended, alpha_channel])
            return Image.fromarray(result, "RGBA")
        return Image.fromarray(blended, "RGB")

    def apply_global_grading(
        self, assembled: Image.Image, reference: Image.Image
    ) -> Image.Image:
        """Apply global color grading to an assembled image.

        Builds a LUT from the reference and applies it to the assembled image.
        Result is: output = (1-strength) * original + strength * lut_mapped

        Args:
            assembled: Full assembled image.
            reference: Style reference tile.

        Returns:
            Color-graded image.
        """
        lut = self.build_color_lut(reference)
        return self.apply_color_lut(assembled, lut)

    # ------------------------------------------------------------------
    # Palette extraction (K-means)
    # ------------------------------------------------------------------

    def extract_palette(
        self, image: Image.Image, n_colors: int = 8
    ) -> list[tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering.

        Uses a simple iterative K-means (no sklearn dependency).  Large images
        are downsampled before clustering for speed.

        Args:
            image: Image to analyse.
            n_colors: Number of colours to extract.

        Returns:
            List of (R, G, B) tuples for dominant colours, sorted by cluster
            population (most frequent first).
        """
        img_rgb = np.array(image.convert("RGB"), dtype=np.float64)
        pixels = img_rgb.reshape(-1, 3)

        # Downsample if there are too many pixels
        max_pixels = 50_000
        if len(pixels) > max_pixels:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[indices]

        n_pixels = len(pixels)
        n_colors = min(n_colors, n_pixels)

        # Initialise centres by sampling random pixels
        rng = np.random.default_rng(0)
        center_indices = rng.choice(n_pixels, n_colors, replace=False)
        centers = pixels[center_indices].copy()

        n_iterations = 20
        for _ in range(n_iterations):
            # Assign each pixel to nearest centre
            # distances shape: (n_pixels, n_colors)
            diffs = pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.sum(diffs ** 2, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centres
            new_centers = np.empty_like(centers)
            counts = np.empty(n_colors, dtype=np.float64)
            for k in range(n_colors):
                mask = labels == k
                count = np.sum(mask)
                counts[k] = count
                if count > 0:
                    new_centers[k] = pixels[mask].mean(axis=0)
                else:
                    # Re-seed empty cluster from a random pixel
                    new_centers[k] = pixels[rng.integers(n_pixels)]
                    counts[k] = 0

            # Check for convergence
            shift = np.max(np.abs(new_centers - centers))
            centers = new_centers
            if shift < 0.5:
                break

        # Final assignment to get counts
        diffs = pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        final_counts = np.array(
            [np.sum(labels == k) for k in range(n_colors)], dtype=np.float64
        )

        # Sort by population (most frequent first)
        order = np.argsort(-final_counts)
        palette: list[tuple[int, int, int]] = []
        for idx in order:
            r, g, b = centers[idx]
            palette.append((int(round(r)), int(round(g)), int(round(b))))

        return palette

    # ------------------------------------------------------------------
    # Perceptual colour distance
    # ------------------------------------------------------------------

    def compute_color_distance(
        self, image: Image.Image, reference: Image.Image
    ) -> float:
        """Compute average colour distance between two images.

        Uses mean absolute difference in LAB colour space for perceptual
        accuracy.  If the images differ in size, the larger one is resized
        to match the smaller.

        Args:
            image: First image.
            reference: Second image.

        Returns:
            Average perceptual colour distance (0.0 = identical).
        """
        img_rgb = np.array(image.convert("RGB"), dtype=np.float64)
        ref_rgb = np.array(reference.convert("RGB"), dtype=np.float64)

        # Resize to common dimensions if needed
        if img_rgb.shape[:2] != ref_rgb.shape[:2]:
            target_h = min(img_rgb.shape[0], ref_rgb.shape[0])
            target_w = min(img_rgb.shape[1], ref_rgb.shape[1])
            img_pil = Image.fromarray(img_rgb.astype(np.uint8)).resize(
                (target_w, target_h), Image.LANCZOS
            )
            ref_pil = Image.fromarray(ref_rgb.astype(np.uint8)).resize(
                (target_w, target_h), Image.LANCZOS
            )
            img_rgb = np.array(img_pil, dtype=np.float64)
            ref_rgb = np.array(ref_pil, dtype=np.float64)

        img_lab = self.rgb_to_lab(img_rgb)
        ref_lab = self.rgb_to_lab(ref_rgb)

        diff = np.abs(img_lab - ref_lab)
        return float(np.mean(diff))

    # ------------------------------------------------------------------
    # Colour space conversions  (RGB <-> XYZ <-> LAB)
    # ------------------------------------------------------------------

    @staticmethod
    def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB array to CIE-LAB colour space.

        Standard pipeline: sRGB linearisation -> XYZ (D65) -> LAB.

        Args:
            rgb: Array of shape (..., 3) with values 0-255.

        Returns:
            Array of shape (..., 3) with L(0-100), a(-128..127), b(-128..127).
        """
        # 1. Normalise to [0, 1] and apply inverse sRGB companding
        rgb_norm = rgb / 255.0
        mask = rgb_norm > 0.04045
        linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

        # 2. Linear RGB -> XYZ  (sRGB matrix, D65 illuminant)
        r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # 3. Normalise by D65 white point
        x /= _D65_X
        y /= _D65_Y
        z /= _D65_Z

        # 4. XYZ -> LAB
        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > _LAB_EPSILON, np.cbrt(t), (_LAB_KAPPA * t + 16.0) / 116.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_ch = 200.0 * (fy - fz)

        return np.stack([L, a, b_ch], axis=-1)

    @staticmethod
    def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        """Convert CIE-LAB array back to sRGB (0-255).

        Args:
            lab: Array of shape (..., 3) with L, a, b channels.

        Returns:
            Array of shape (..., 3) with values 0-255 (float64, clipped).
        """
        L, a, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

        # 1. LAB -> XYZ
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b_ch / 200.0

        def f_inv(t: np.ndarray) -> np.ndarray:
            t3 = t ** 3
            return np.where(t3 > _LAB_EPSILON, t3, (116.0 * t - 16.0) / _LAB_KAPPA)

        x = f_inv(fx) * _D65_X
        y = f_inv(fy) * _D65_Y
        z = f_inv(fz) * _D65_Z

        # 2. XYZ -> linear RGB (inverse of sRGB matrix)
        r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

        # 3. sRGB companding
        def gamma(c: np.ndarray) -> np.ndarray:
            return np.where(c > 0.0031308, 1.055 * np.power(np.maximum(c, 0), 1.0 / 2.4) - 0.055, 12.92 * c)

        r_srgb = gamma(r_lin)
        g_srgb = gamma(g_lin)
        b_srgb = gamma(b_lin)

        rgb = np.stack([r_srgb, g_srgb, b_srgb], axis=-1)
        return np.clip(rgb * 255.0, 0, 255)
