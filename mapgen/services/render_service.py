"""Render service for converting OSM data to styled base map images."""

from io import BytesIO
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from ..models.project import BoundingBox, Project
from .osm_service import OSMData
from .perspective_service import PerspectiveService
from .terrain_service import ElevationData


class RenderService:
    """Service for rendering OSM data to styled base map images."""

    # Color scheme inspired by Disneyland maps
    COLORS = {
        # Water
        "water": "#4A90D9",
        "waterway": "#5BA3E0",
        # Parks and green areas
        "park": "#7CB342",
        "forest": "#558B2F",
        "grass": "#8BC34A",
        # Urban areas
        "urban": "#E8E0D5",
        "residential": "#F5F0E8",
        "commercial": "#F0E8E0",
        "industrial": "#E0D8D0",
        # Buildings
        "building": "#D5C8B8",
        "building_shadow": "#B8A898",
        # Roads
        "road_major": "#FFFFFF",
        "road_major_outline": "#D0D0D0",
        "road_minor": "#F8F8F8",
        "road_minor_outline": "#E0E0E0",
        # Railways
        "railway": "#606060",
        # Background
        "background": "#E8F4FC",
        # Desert/sand
        "desert": "#F4E4B8",
        "sand": "#FAF0D0",
    }

    # Line widths for different road types (in points)
    ROAD_WIDTHS = {
        "major": 4.0,
        "minor": 2.0,
        "other": 1.0,
    }

    def __init__(self, perspective_service: Optional[PerspectiveService] = None):
        """
        Initialize render service.

        Args:
            perspective_service: Optional perspective transformation service
        """
        self.perspective = perspective_service or PerspectiveService()

    def render_base_map(
        self,
        osm_data: OSMData,
        bbox: BoundingBox,
        output_size: tuple[int, int],
        elevation_data: Optional[ElevationData] = None,
        apply_perspective: bool = True,
        dpi: int = 100,
    ) -> Image.Image:
        """
        Render OSM data to a styled base map image.

        Args:
            osm_data: OSM data layers
            bbox: Bounding box
            output_size: (width, height) in pixels
            elevation_data: Optional elevation data for hillshade
            apply_perspective: Whether to apply isometric perspective
            dpi: DPI for rendering

        Returns:
            PIL Image with rendered map
        """
        width, height = output_size

        # Calculate figure size in inches
        fig_width = width / dpi
        fig_height = height / dpi

        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.set_xlim(bbox.west, bbox.east)
        ax.set_ylim(bbox.south, bbox.north)
        ax.set_aspect("equal")
        ax.axis("off")

        # Set background color
        fig.patch.set_facecolor(self.COLORS["background"])
        ax.set_facecolor(self.COLORS["background"])

        # Get elevation array for perspective if available
        elev_array = elevation_data.dem if elevation_data else None

        # Render hillshade if elevation data available
        if elevation_data is not None:
            self._render_hillshade(ax, elevation_data, bbox)

        # Render layers in order (back to front)
        # 1. Terrain types (landuse)
        if osm_data.terrain_types is not None:
            terrain = osm_data.terrain_types
            if apply_perspective:
                terrain = self.perspective.transform_geodataframe(terrain, bbox, elev_array)
            self._render_terrain(ax, terrain)

        # 2. Water bodies
        if osm_data.water is not None:
            water = osm_data.water
            if apply_perspective:
                water = self.perspective.transform_geodataframe(water, bbox, elev_array)
            self._render_water(ax, water)

        # 3. Parks
        if osm_data.parks is not None:
            parks = osm_data.parks
            if apply_perspective:
                parks = self.perspective.transform_geodataframe(parks, bbox, elev_array)
            self._render_parks(ax, parks)

        # 4. Buildings
        if osm_data.buildings is not None:
            buildings = osm_data.buildings
            if apply_perspective:
                buildings = self.perspective.transform_geodataframe(buildings, bbox, elev_array)
            self._render_buildings(ax, buildings)

        # 5. Railways
        if osm_data.railways is not None:
            railways = osm_data.railways
            if apply_perspective:
                railways = self.perspective.transform_geodataframe(railways, bbox, elev_array)
            self._render_railways(ax, railways)

        # 6. Roads (on top)
        if osm_data.roads is not None:
            roads = osm_data.roads
            if apply_perspective:
                roads = self.perspective.transform_geodataframe(roads, bbox, elev_array)
            self._render_roads(ax, roads)

        # Convert to PIL Image
        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor=fig.get_facecolor(),
        )
        buf.seek(0)
        image = Image.open(buf).convert("RGBA")

        # Resize to exact output size if needed
        if image.size != output_size:
            image = image.resize(output_size, Image.Resampling.LANCZOS)

        plt.close(fig)
        return image

    def _render_hillshade(
        self,
        ax,
        elevation_data: ElevationData,
        bbox: BoundingBox,
    ) -> None:
        """Render hillshade as background layer."""
        from .terrain_service import TerrainService

        terrain_service = TerrainService()
        hillshade = terrain_service.compute_hillshade(elevation_data)

        # Create extent for imshow
        extent = [bbox.west, bbox.east, bbox.south, bbox.north]

        # Render as semi-transparent gray layer
        ax.imshow(
            hillshade,
            extent=extent,
            cmap="gray",
            alpha=0.3,
            origin="upper",
            aspect="auto",
        )

    def _render_terrain(self, ax, terrain: gpd.GeoDataFrame) -> None:
        """Render terrain type polygons."""
        if len(terrain) == 0:
            return

        # Color mapping for terrain classes
        terrain_colors = {
            "urban": self.COLORS["urban"],
            "forest": self.COLORS["forest"],
            "grassland": self.COLORS["grass"],
            "water": self.COLORS["water"],
            "desert": self.COLORS["desert"],
            "other": self.COLORS["residential"],
        }

        for terrain_class, color in terrain_colors.items():
            subset = terrain[terrain["terrain_class"] == terrain_class]
            if len(subset) == 0:
                continue

            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                self._render_polygon(ax, geom, facecolor=color, edgecolor="none", alpha=0.7)

    def _render_water(self, ax, water: gpd.GeoDataFrame) -> None:
        """Render water bodies and waterways."""
        if len(water) == 0:
            return

        for idx, row in water.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            water_type = row.get("water_type", "water")
            color = self.COLORS.get(water_type, self.COLORS["water"])

            if geom.geom_type in ["Polygon", "MultiPolygon"]:
                self._render_polygon(ax, geom, facecolor=color, edgecolor=color, alpha=0.9)
            elif geom.geom_type in ["LineString", "MultiLineString"]:
                self._render_linestring(ax, geom, color=color, linewidth=2)

    def _render_parks(self, ax, parks: gpd.GeoDataFrame) -> None:
        """Render park areas."""
        if len(parks) == 0:
            return

        for geom in parks.geometry:
            if geom is None or geom.is_empty:
                continue
            self._render_polygon(
                ax,
                geom,
                facecolor=self.COLORS["park"],
                edgecolor=self.COLORS["forest"],
                linewidth=0.5,
                alpha=0.8,
            )

    def _render_buildings(self, ax, buildings: gpd.GeoDataFrame) -> None:
        """Render building footprints with shadows."""
        if len(buildings) == 0:
            return

        # Sort by centroid y-coordinate for proper overlapping
        buildings = buildings.copy()
        buildings["centroid_y"] = buildings.geometry.centroid.y
        buildings = buildings.sort_values("centroid_y", ascending=False)

        shadow_offset = 0.00005  # Small offset for shadow effect

        for geom in buildings.geometry:
            if geom is None or geom.is_empty:
                continue

            # Render shadow first
            from shapely import affinity

            shadow = affinity.translate(geom, xoff=shadow_offset, yoff=-shadow_offset)
            self._render_polygon(
                ax,
                shadow,
                facecolor=self.COLORS["building_shadow"],
                edgecolor="none",
                alpha=0.5,
            )

            # Render building
            self._render_polygon(
                ax,
                geom,
                facecolor=self.COLORS["building"],
                edgecolor="#A09080",
                linewidth=0.3,
                alpha=0.9,
            )

    def _render_roads(self, ax, roads: gpd.GeoDataFrame) -> None:
        """Render road network."""
        if len(roads) == 0:
            return

        # Render roads by class (major roads on top)
        for road_class in ["other", "minor", "major"]:
            subset = roads[roads.get("road_class", "other") == road_class]
            if len(subset) == 0:
                continue

            width = self.ROAD_WIDTHS.get(road_class, 1.0)
            color = self.COLORS.get(f"road_{road_class}", self.COLORS["road_minor"])
            outline = self.COLORS.get(f"road_{road_class}_outline", "#E0E0E0")

            # Render outline first
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                self._render_linestring(ax, geom, color=outline, linewidth=width + 1)

            # Render road fill
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                self._render_linestring(ax, geom, color=color, linewidth=width)

    def _render_railways(self, ax, railways: gpd.GeoDataFrame) -> None:
        """Render railway lines."""
        if len(railways) == 0:
            return

        for geom in railways.geometry:
            if geom is None or geom.is_empty:
                continue
            # Dashed line for railways
            self._render_linestring(
                ax,
                geom,
                color=self.COLORS["railway"],
                linewidth=1.5,
                linestyle="--",
            )

    def _render_polygon(
        self,
        ax,
        geom,
        facecolor: str,
        edgecolor: str,
        linewidth: float = 1,
        alpha: float = 1,
    ) -> None:
        """Render a polygon or multipolygon."""
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                self._render_single_polygon(ax, poly, facecolor, edgecolor, linewidth, alpha)
        elif isinstance(geom, Polygon):
            self._render_single_polygon(ax, geom, facecolor, edgecolor, linewidth, alpha)

    def _render_single_polygon(
        self,
        ax,
        poly: Polygon,
        facecolor: str,
        edgecolor: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        """Render a single polygon."""
        if poly.is_empty:
            return

        # Create path for polygon with holes
        exterior = np.array(poly.exterior.coords)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(exterior) - 2) + [MplPath.CLOSEPOLY]
        vertices = list(exterior)

        for interior in poly.interiors:
            interior_coords = np.array(interior.coords)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(interior_coords) - 2) + [MplPath.CLOSEPOLY]
            vertices.extend(interior_coords)

        path = MplPath(vertices, codes)
        patch = PathPatch(
            path,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(patch)

    def _render_linestring(
        self,
        ax,
        geom,
        color: str,
        linewidth: float = 1,
        linestyle: str = "-",
    ) -> None:
        """Render a linestring or multilinestring."""
        from shapely.geometry import LineString, MultiLineString

        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                self._render_single_linestring(ax, line, color, linewidth, linestyle)
        elif isinstance(geom, LineString):
            self._render_single_linestring(ax, geom, color, linewidth, linestyle)

    def _render_single_linestring(
        self,
        ax,
        line,
        color: str,
        linewidth: float,
        linestyle: str,
    ) -> None:
        """Render a single linestring."""
        if line.is_empty:
            return

        coords = np.array(line.coords)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    def render_tile(
        self,
        osm_data: OSMData,
        bbox: BoundingBox,
        tile_bbox: BoundingBox,
        tile_size: int,
        elevation_data: Optional[ElevationData] = None,
        apply_perspective: bool = True,
    ) -> Image.Image:
        """
        Render a single tile from the map.

        Args:
            osm_data: Full OSM data
            bbox: Full map bounding box
            tile_bbox: Tile bounding box
            tile_size: Tile size in pixels
            elevation_data: Optional elevation data
            apply_perspective: Whether to apply perspective

        Returns:
            PIL Image for tile
        """
        # Render at tile size
        return self.render_base_map(
            osm_data=osm_data,
            bbox=tile_bbox,
            output_size=(tile_size, tile_size),
            elevation_data=elevation_data,
            apply_perspective=apply_perspective,
        )
