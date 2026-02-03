"""PSD export service for layered map output."""

from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


class PSDService:
    """Service for exporting maps as layered PSD files."""

    def __init__(self):
        """Initialize PSD service."""
        self._psd_tools_available = False
        self._pytoshop_available = False

        try:
            import psd_tools

            self._psd_tools_available = True
        except ImportError:
            pass

        try:
            import pytoshop

            self._pytoshop_available = True
        except ImportError:
            pass

    def create_layered_psd(
        self,
        layers: dict[str, tuple[Image.Image, tuple[int, int]]],
        output_path: Union[str, Path],
        canvas_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Create a layered PSD file.

        Args:
            layers: Dictionary mapping layer names to (image, position) tuples
            output_path: Output file path
            canvas_size: Optional (width, height) for canvas
        """
        if self._pytoshop_available:
            self._create_psd_pytoshop(layers, output_path, canvas_size)
        else:
            # Fallback: create a simple layered TIFF or multi-layer PNG
            self._create_layered_tiff(layers, output_path, canvas_size)

    def _create_psd_pytoshop(
        self,
        layers: dict[str, tuple[Image.Image, tuple[int, int]]],
        output_path: Union[str, Path],
        canvas_size: Optional[tuple[int, int]],
    ) -> None:
        """Create PSD using pytoshop library."""
        import pytoshop
        from pytoshop import layers as psd_layers
        from pytoshop.enums import Compression

        # Determine canvas size
        if canvas_size is None:
            max_width = 0
            max_height = 0
            for img, pos in layers.values():
                right = pos[0] + img.width
                bottom = pos[1] + img.height
                max_width = max(max_width, right)
                max_height = max(max_height, bottom)
            canvas_size = (max_width, max_height)

        width, height = canvas_size

        # Create PSD document
        psd = pytoshop.core.PsdFile(num_channels=4, height=height, width=width)

        # Add layers in reverse order (first in dict = top layer in PSD)
        layer_list = list(layers.items())
        for name, (img, pos) in reversed(layer_list):
            # Ensure RGBA
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Convert to numpy array
            arr = np.array(img)

            # Create ChannelImageData for each channel (using raw to avoid pytoshop RLE bug)
            r = psd_layers.ChannelImageData(
                image=arr[:, :, 0], compression=Compression.raw
            )
            g = psd_layers.ChannelImageData(
                image=arr[:, :, 1], compression=Compression.raw
            )
            b = psd_layers.ChannelImageData(
                image=arr[:, :, 2], compression=Compression.raw
            )
            a = psd_layers.ChannelImageData(
                image=arr[:, :, 3], compression=Compression.raw
            )

            # Create layer with position using LayerRecord
            layer = psd_layers.LayerRecord(
                name=name,
                top=pos[1],
                left=pos[0],
                bottom=pos[1] + img.height,
                right=pos[0] + img.width,
                channels={
                    0: r,  # Red
                    1: g,  # Green
                    2: b,  # Blue
                    -1: a,  # Alpha
                },
            )
            psd.layer_and_mask_info.layer_info.layer_records.append(layer)

        # Save PSD
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            psd.write(f)

    def _create_layered_tiff(
        self,
        layers: dict[str, tuple[Image.Image, tuple[int, int]]],
        output_path: Union[str, Path],
        canvas_size: Optional[tuple[int, int]],
    ) -> None:
        """
        Fallback: Create a multi-page TIFF with layers.

        Note: This is not a true PSD but provides layer separation.
        """
        output_path = Path(output_path)

        # Determine canvas size
        if canvas_size is None:
            max_width = 0
            max_height = 0
            for img, pos in layers.values():
                right = pos[0] + img.width
                bottom = pos[1] + img.height
                max_width = max(max_width, right)
                max_height = max(max_height, bottom)
            canvas_size = (max_width, max_height)

        width, height = canvas_size
        images = []

        for name, (img, pos) in layers.items():
            # Create full-size canvas with transparency
            canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

            # Ensure RGBA
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Paste at position
            canvas.paste(img, pos, img)

            # Add name as metadata (will be lost in TIFF but useful for debugging)
            canvas.info["name"] = name
            images.append(canvas)

        # Save as TIFF (change extension)
        tiff_path = output_path.with_suffix(".tiff")
        tiff_path.parent.mkdir(parents=True, exist_ok=True)

        if images:
            # Save first image with others appended
            images[0].save(
                tiff_path,
                save_all=True,
                append_images=images[1:],
                compression="tiff_lzw",
            )

        print(f"Note: PSD library not available. Saved as layered TIFF: {tiff_path}")

    def export_layers_separately(
        self,
        layers: dict[str, tuple[Image.Image, tuple[int, int]]],
        output_dir: Union[str, Path],
        canvas_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, Path]:
        """
        Export each layer as a separate PNG file.

        Args:
            layers: Dictionary mapping layer names to (image, position) tuples
            output_dir: Output directory
            canvas_size: Optional canvas size

        Returns:
            Dictionary mapping layer names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine canvas size
        if canvas_size is None:
            max_width = 0
            max_height = 0
            for img, pos in layers.values():
                right = pos[0] + img.width
                bottom = pos[1] + img.height
                max_width = max(max_width, right)
                max_height = max(max_height, bottom)
            canvas_size = (max_width, max_height)

        exported = {}

        for i, (name, (img, pos)) in enumerate(layers.items()):
            # Create full-size canvas
            canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

            if img.mode != "RGBA":
                img = img.convert("RGBA")

            canvas.paste(img, pos, img)

            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            filename = f"{i:02d}_{safe_name}.png"
            filepath = output_dir / filename

            canvas.save(filepath)
            exported[name] = filepath

        return exported

    def create_layer_manifest(
        self,
        layers: dict[str, tuple[Image.Image, tuple[int, int]]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Create a JSON manifest describing the layer structure.

        Args:
            layers: Dictionary mapping layer names to (image, position) tuples
            output_path: Output path for manifest JSON
        """
        import json

        manifest = {
            "layers": [],
            "canvas_size": None,
        }

        max_width = 0
        max_height = 0

        for name, (img, pos) in layers.items():
            layer_info = {
                "name": name,
                "position": {"x": pos[0], "y": pos[1]},
                "size": {"width": img.width, "height": img.height},
            }
            manifest["layers"].append(layer_info)

            right = pos[0] + img.width
            bottom = pos[1] + img.height
            max_width = max(max_width, right)
            max_height = max(max_height, bottom)

        manifest["canvas_size"] = {"width": max_width, "height": max_height}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
