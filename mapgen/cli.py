"""Command-line interface for map generator."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_config
from .models.project import BoundingBox, DetailLevel, Project, get_recommended_detail_level


def timestamped_filename(base_name: str, extension: str = "png") -> str:
    """Generate a filename with timestamp to avoid overwrites.

    Args:
        base_name: Base name for the file (e.g., 'NYC_Map_illustrated')
        extension: File extension without dot (default: 'png')

    Returns:
        Filename like 'NYC_Map_illustrated_20240201_143052.png'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Illustrated Map Generator - Create theme park style maps."""
    pass


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--skip-tiles", is_flag=True, help="Skip tile generation (use cached)")
@click.option("--skip-landmarks", is_flag=True, help="Skip landmark generation")
@click.option("--preview-only", is_flag=True, help="Generate preview only (no full res)")
def generate(
    project_path: str,
    output: Optional[str],
    skip_tiles: bool,
    skip_landmarks: bool,
    preview_only: bool,
):
    """Generate an illustrated map from a project configuration."""
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    if not project_file.exists():
        console.print(f"[red]Error:[/red] Project file not found: {project_file}")
        raise SystemExit(1)

    console.print(f"[bold]Loading project:[/bold] {project_file}")
    project = Project.from_yaml(project_file)
    project.ensure_directories()

    output_dir = Path(output) if output else project.output_dir

    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Region:[/bold] {project.region.north:.4f}N to {project.region.south:.4f}S, "
                  f"{project.region.west:.4f}W to {project.region.east:.4f}E")

    # Calculate tile grid
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    total_tiles = cols * rows

    console.print(f"[bold]Output size:[/bold] {project.output.width} x {project.output.height} px")
    console.print(f"[bold]Tile grid:[/bold] {cols} x {rows} = {total_tiles} tiles")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Fetch OSM data
        task = progress.add_task("Fetching OSM data...", total=None)
        osm_data = _fetch_osm_data(project)
        progress.update(task, completed=True, description="[green]OSM data fetched")

        # Step 2: Fetch elevation data
        task = progress.add_task("Fetching elevation data...", total=None)
        elevation_data = _fetch_elevation_data(project)
        progress.update(task, completed=True, description="[green]Elevation data fetched")

        # Step 3: Render base map
        task = progress.add_task("Rendering base map...", total=None)
        if preview_only:
            base_map = _render_preview(project, osm_data, elevation_data)
        else:
            base_map = _render_base_map(project, osm_data, elevation_data)
        progress.update(task, completed=True, description="[green]Base map rendered")

        # Save result
        base_name = project.name.replace(' ', '_') + "_base_map"
        output_path = output_dir / timestamped_filename(base_name)
        base_map.save(output_path)
        console.print(f"[green]Saved:[/green] {output_path}")


@main.command()
@click.option("--name", "-n", required=True, help="Project name")
@click.option("--north", type=float, required=True, help="North latitude")
@click.option("--south", type=float, required=True, help="South latitude")
@click.option("--east", type=float, required=True, help="East longitude")
@click.option("--west", type=float, required=True, help="West longitude")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def init(name: str, north: float, south: float, east: float, west: float, output: Optional[str]):
    """Initialize a new map project."""
    output_dir = Path(output) if output else Path.cwd() / name.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    project = Project(
        name=name,
        region=BoundingBox(north=north, south=south, east=east, west=west),
    )
    project.project_dir = output_dir

    # Save project file
    project_file = output_dir / "project.yaml"
    project.to_yaml(project_file)

    # Create subdirectories
    project.ensure_directories()

    console.print(f"[green]Created project:[/green] {project_file}")
    console.print(f"[dim]Add landmark photos to:[/dim] {project.landmarks_dir}")
    console.print(f"[dim]Add logo PNGs to:[/dim] {project.logos_dir}")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
def info(project_path: str):
    """Show information about a project."""
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    # Calculate region area and recommended detail level
    area_km2 = project.region.calculate_area_km2()
    recommended_detail = project.region.get_recommended_detail_level()

    table = Table(title=f"Project: {project.name}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Region North", f"{project.region.north:.6f}")
    table.add_row("Region South", f"{project.region.south:.6f}")
    table.add_row("Region East", f"{project.region.east:.6f}")
    table.add_row("Region West", f"{project.region.west:.6f}")
    table.add_row("Region Area", f"{area_km2:,.0f} km²")
    table.add_row("Output Width", f"{project.output.width} px")
    table.add_row("Output Height", f"{project.output.height} px")
    table.add_row("Output DPI", str(project.output.dpi))
    table.add_row("Tile Size", f"{project.tiles.size} px")
    table.add_row("Tile Overlap", f"{project.tiles.overlap} px")

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    table.add_row("Tile Grid", f"{cols} x {rows} = {cols * rows} tiles")

    console.print(table)

    # Show detail level recommendation with warning for large regions
    console.print(f"\n[bold]Recommended Detail Level:[/bold] {recommended_detail.value}")
    if recommended_detail == DetailLevel.COUNTRY:
        console.print("[yellow]Warning:[/yellow] Region is very large. Only motorways and major cities will be rendered.")
        console.print("[dim]Consider using a smaller region for more detail.[/dim]")
    elif recommended_detail == DetailLevel.REGIONAL:
        console.print("[yellow]Note:[/yellow] Region is large. Primary roads and named landmarks will be rendered.")
    elif recommended_detail == DetailLevel.SIMPLIFIED:
        console.print("[dim]Major roads and notable buildings will be rendered.[/dim]")
    else:
        console.print("[dim]Full detail - all roads and buildings will be rendered.[/dim]")

    # Show per-tile OSM info for large regions
    if area_km2 > 10_000:
        console.print(f"\n[bold]Per-tile OSM fetching:[/bold] enabled (region > 10,000 km²)")
        console.print("[dim]OSM data will be fetched per-tile to avoid query size limits.[/dim]")

    # Estimate costs
    from .services.gemini_service import GeminiService

    gemini = GeminiService.__new__(GeminiService)
    costs = gemini.estimate_cost(cols * rows, 0)

    console.print(f"\n[bold]Estimated generation cost:[/bold] ${costs['total_cost']:.2f}")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
def preview_osm(project_path: str):
    """Generate a preview of OSM data extraction."""
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    console.print(f"[bold]Fetching OSM data for:[/bold] {project.name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching OSM data...", total=None)
        osm_data = _fetch_osm_data(project)
        progress.update(task, completed=True)

    # Print summary
    console.print("\n[bold]OSM Data Summary:[/bold]")
    if osm_data.roads is not None:
        console.print(f"  Roads: {len(osm_data.roads)} features")
    if osm_data.buildings is not None:
        console.print(f"  Buildings: {len(osm_data.buildings)} features")
    if osm_data.water is not None:
        console.print(f"  Water: {len(osm_data.water)} features")
    if osm_data.parks is not None:
        console.print(f"  Parks: {len(osm_data.parks)} features")
    if osm_data.terrain_types is not None:
        console.print(f"  Terrain: {len(osm_data.terrain_types)} features")

    # Render preview
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Rendering preview...", total=None)
        preview = _render_preview(project, osm_data, None)
        progress.update(task, completed=True)

    preview_path = project.output_dir / timestamped_filename("osm_preview")
    preview.save(preview_path)
    console.print(f"\n[green]Saved preview:[/green] {preview_path}")


def _fetch_osm_data(project: Project):
    """Fetch OSM data for project."""
    from .services.osm_service import OSMService

    config = get_config()
    osm_service = OSMService(cache_dir=str(config.cache_dir / "osm"))
    return osm_service.fetch_region_data(project.region)


def _fetch_elevation_data(project: Project):
    """Fetch elevation data for project."""
    from .services.terrain_service import TerrainService

    config = get_config()
    terrain_service = TerrainService(cache_dir=str(config.cache_dir / "dem"))

    try:
        return terrain_service.fetch_elevation_data(project.region)
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not fetch elevation: {e}")
        return None


def _render_preview(project: Project, osm_data, elevation_data):
    """Render a low-resolution preview."""
    from .services.perspective_service import PerspectiveService
    from .services.render_service import RenderService

    perspective = PerspectiveService(angle=project.style.perspective_angle)
    render = RenderService(perspective_service=perspective)

    # Render at 1/4 size for preview
    preview_size = (project.output.width // 4, project.output.height // 4)

    return render.render_base_map(
        osm_data=osm_data,
        bbox=project.region,
        output_size=preview_size,
        elevation_data=elevation_data,
        apply_perspective=True,
        dpi=72,
    )


def _render_base_map(project: Project, osm_data, elevation_data):
    """Render the full base map."""
    from .services.perspective_service import PerspectiveService
    from .services.render_service import RenderService

    perspective = PerspectiveService(angle=project.style.perspective_angle)
    render = RenderService(perspective_service=perspective)

    return render.render_base_map(
        osm_data=osm_data,
        bbox=project.region,
        output_size=(project.output.width, project.output.height),
        elevation_data=elevation_data,
        apply_perspective=True,
        dpi=project.output.dpi,
    )


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--size", "-s", default=2048, help="Output size in pixels (square)")
@click.option("--perspective/--no-perspective", default=True, help="Apply aerial perspective")
@click.option("--convergence", "-c", default=0.7, help="Top edge convergence (0.0-1.0)")
@click.option("--vertical-scale", "-v", default=0.4, help="Vertical compression at top (0.0-1.0)")
@click.option("--horizon-margin", "-m", default=0.15, help="Extra space for horizon (0.0-0.5)")
def preview_composite(
    project_path: str,
    size: int,
    perspective: bool,
    convergence: float,
    vertical_scale: float,
    horizon_margin: float,
):
    """Generate a composite reference preview (satellite + simplified OSM).

    The perspective transform creates an aerial view effect where:
    - The top of the map appears farther away (narrower, compressed)
    - A horizon/sky area is visible at the top
    - The bottom of the map is closer to the viewer

    Adjust --convergence (how much top narrows), --vertical-scale (compression),
    and --horizon-margin (sky space) to tune the perspective effect.
    """
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    console.print(f"[bold]Generating composite reference for:[/bold] {project.name}")
    if perspective:
        console.print(f"[dim]Perspective: convergence={convergence}, vertical_scale={vertical_scale}, horizon_margin={horizon_margin}[/dim]")

    output_size = (size, size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Fetch satellite imagery
        task = progress.add_task("Fetching satellite imagery...", total=None)
        satellite_image = _fetch_satellite_imagery(project, output_size)
        progress.update(task, completed=True, description="[green]Satellite imagery fetched")

        # Step 2: Fetch simplified OSM data
        task = progress.add_task("Fetching simplified OSM data...", total=None)
        osm_data = _fetch_simplified_osm_data(project)
        progress.update(task, completed=True, description="[green]Simplified OSM data fetched")

        # Step 3: Create composite
        task = progress.add_task("Creating composite reference...", total=None)
        composite = _render_composite(
            project,
            satellite_image,
            osm_data,
            output_size,
            apply_perspective=perspective,
            convergence=convergence,
            vertical_scale=vertical_scale,
            horizon_margin=horizon_margin,
        )
        progress.update(task, completed=True, description="[green]Composite created")

    # Print summary
    console.print("\n[bold]Data Summary:[/bold]")
    if osm_data.roads is not None:
        console.print(f"  Major roads: {len(osm_data.roads)} features")
    if osm_data.buildings is not None:
        console.print(f"  Notable buildings: {len(osm_data.buildings)} features")
    if osm_data.water is not None:
        console.print(f"  Water: {len(osm_data.water)} features")
    if osm_data.parks is not None:
        console.print(f"  Parks: {len(osm_data.parks)} features")

    # Save outputs
    satellite_path = project.output_dir / timestamped_filename("satellite")
    if perspective:
        composite_path = project.output_dir / timestamped_filename("composite_perspective")
    else:
        composite_path = project.output_dir / timestamped_filename("composite_flat")

    satellite_image.save(satellite_path)
    composite.save(composite_path)

    console.print(f"\n[green]Saved satellite image:[/green] {satellite_path}")
    console.print(f"[green]Saved composite reference:[/green] {composite_path}")
    if perspective:
        # Report final image size since perspective adds margin
        console.print(f"[dim]Output size: {composite.width} x {composite.height} px (includes horizon margin)[/dim]")
    console.print("\n[dim]The composite reference can be sent to Gemini for illustration.[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--col", "-c", default=0, help="Column index of tile to generate")
@click.option("--row", "-r", default=0, help="Row index of tile to generate")
@click.option("--save-reference/--no-save-reference", default=True, help="Save reference image")
def test_tile(project_path: str, col: int, row: int, save_reference: bool):
    """Generate a single test tile to validate the pipeline.

    This is useful for testing before running full generation. It generates
    one tile using Gemini to verify the style and quality are acceptable.

    Requires GOOGLE_API_KEY environment variable to be set.
    """
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    # Import generation service
    from .services.generation_service import GenerationService

    config = get_config()
    gen_service = GenerationService(
        project=project,
        cache_dir=config.cache_dir / "generation",
    )

    # Show grid info
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Tile grid:[/bold] {cols} x {rows} = {cols * rows} tiles")
    console.print(f"[bold]Testing tile:[/bold] ({col}, {row})")

    if col >= cols or row >= rows:
        console.print(f"[red]Error:[/red] Tile ({col}, {row}) is outside grid bounds")
        raise SystemExit(1)

    # Estimate cost
    cost = gen_service.estimate_cost()
    console.print(f"[dim]Single tile cost: ~$0.13 | Full map: ~${cost['total_cost']:.2f}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Generate reference
        task = progress.add_task("Generating reference image...", total=None)

        specs = gen_service.calculate_tile_specs()
        spec = next(s for s in specs if s.col == col and s.row == row)

        reference = gen_service.generate_tile_reference(spec)
        progress.update(task, completed=True, description="[green]Reference generated")

        if save_reference:
            ref_path = project.output_dir / f"test_tile_{col}_{row}_reference.png"
            reference.save(ref_path)
            console.print(f"[dim]Saved reference:[/dim] {ref_path}")

        # Generate with Gemini
        task = progress.add_task("Calling Gemini (this may take 30-60 seconds)...", total=None)

        result = gen_service.generate_tile(spec)

        if result.error:
            progress.update(task, description=f"[red]Failed: {result.error}")
            raise SystemExit(1)

        progress.update(task, completed=True, description=f"[green]Generated in {result.generation_time:.1f}s")

    # Save result
    output_path = project.output_dir / timestamped_filename(f"test_tile_{col}_{row}")
    result.generated_image.save(output_path)

    console.print(f"\n[green]Success![/green] Generated tile saved to: {output_path}")
    console.print(f"[dim]Generation time: {result.generation_time:.1f}s | Retries: {result.retries}[/dim]")
    console.print("\n[bold]Review the output to check style quality before running full generation.[/bold]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--perspective/--no-perspective", default=True, help="Apply perspective to final")
@click.option("--skip-existing", is_flag=True, help="Skip tiles that are already cached")
@click.option("--dry-run", is_flag=True, help="Show what would be generated without calling Gemini")
@click.option("--detail-level", "-d",
              type=click.Choice(["full", "simplified", "regional", "country", "auto"]),
              default="auto",
              help="OSM detail level (auto selects based on region size)")
def generate_tiles(
    project_path: str,
    output: Optional[str],
    perspective: bool,
    skip_existing: bool,
    dry_run: bool,
    detail_level: str,
):
    """Generate all illustrated map tiles using Gemini.

    This command generates the full illustrated map by:
    1. Splitting the region into overlapping tiles
    2. Creating reference images (satellite + OSM) for each tile
    3. Sending each tile to Gemini for illustration
    4. Blending all tiles into the final image

    Detail levels:
    - full: All features (for small areas < 100 km²)
    - simplified: Major roads, notable buildings (100-1,000 km²)
    - regional: Primary roads, landmarks only (1,000-50,000 km²)
    - country: Motorways, major cities only (> 50,000 km²)
    - auto: Automatically select based on region size (default)

    Requires GOOGLE_API_KEY environment variable to be set.
    """
    project_file = Path(project_path)

    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    # Import generation service
    from .services.generation_service import GenerationService

    config = get_config()

    # Parse detail level
    detail_level_enum = None
    if detail_level != "auto":
        detail_level_enum = DetailLevel(detail_level)

    gen_service = GenerationService(
        project=project,
        cache_dir=config.cache_dir / "generation",
        detail_level=detail_level_enum,
    )

    # Show info
    specs = gen_service.calculate_tile_specs()
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    cost = gen_service.estimate_cost()

    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Output size:[/bold] {project.output.width} x {project.output.height} px")
    console.print(f"[bold]Region area:[/bold] {gen_service.region_area_km2:,.0f} km²")
    console.print(f"[bold]Detail level:[/bold] {gen_service.detail_level.value}")
    if gen_service.uses_per_tile_osm:
        console.print(f"[bold]Per-tile OSM:[/bold] [yellow]enabled[/yellow] (region > 10,000 km²)")
    console.print(f"[bold]Tile grid:[/bold] {cols} x {rows} = {len(specs)} tiles")
    console.print(f"[bold]Tile size:[/bold] {project.tiles.size}px with {project.tiles.overlap}px overlap")
    console.print(f"[bold]Estimated cost:[/bold] ${cost['total_cost']:.2f}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no tiles will be generated[/yellow]")
        console.print("\n[bold]Tile specifications:[/bold]")
        table = Table()
        table.add_column("Tile", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Bounds")

        for spec in specs[:10]:  # Show first 10
            table.add_row(
                f"({spec.col}, {spec.row})",
                spec.position_desc,
                f"N{spec.bbox.north:.4f} S{spec.bbox.south:.4f}",
            )

        if len(specs) > 10:
            table.add_row("...", f"({len(specs) - 10} more)", "...")

        console.print(table)
        return

    # Confirm before proceeding
    if not click.confirm(f"\nThis will cost approximately ${cost['total_cost']:.2f}. Continue?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Pre-fetch OSM data (only for small regions; large regions fetch per-tile)
    if not gen_service.uses_per_tile_osm:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching OSM data for region...", total=None)
            gen_service.fetch_osm_data()
            progress.update(task, completed=True, description="[green]OSM data fetched")
    else:
        console.print("[dim]OSM data will be fetched per-tile (large region mode)[/dim]")

    # Generate tiles with progress
    from rich.progress import BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    console.print("\n[bold]Generating tiles...[/bold]")

    results = []
    failed = []
    current_status = ["Initializing..."]

    def make_status_panel(tile_num: int, total: int, spec, status: str) -> Panel:
        """Create a status panel showing current tile progress."""
        content = Text()
        content.append(f"Tile {tile_num}/{total}: ", style="bold")
        content.append(f"({spec.col}, {spec.row}) - {spec.position_desc}\n", style="cyan")
        content.append(f"Status: ", style="dim")
        content.append(status, style="yellow")
        return Panel(content, title="[bold]Generating[/bold]", border_style="blue")

    with Live(console=console, refresh_per_second=4) as live:
        for i, spec in enumerate(specs):
            tile_num = i + 1

            # Create progress callback that updates the live display
            def progress_callback(status: str, spec=spec, tile_num=tile_num):
                live.update(make_status_panel(tile_num, len(specs), spec, status))

            progress_callback("Starting...")

            result = gen_service.generate_tile(spec, progress_callback=progress_callback)
            results.append(result)

            if result.error:
                failed.append(result)
                console.print(f"  [red]Failed:[/red] ({spec.col}, {spec.row}) - {result.error}")
            else:
                console.print(f"  [green]Done:[/green] ({spec.col}, {spec.row}) in {result.generation_time:.1f}s")

    # Report results
    console.print(f"\n[bold]Generation complete:[/bold]")
    console.print(f"  Successful: {len(results) - len(failed)} / {len(specs)}")
    if failed:
        console.print(f"  [red]Failed: {len(failed)}[/red]")

    # Assemble tiles
    if len(results) - len(failed) < len(specs) * 0.8:
        console.print("[red]Too many failures to assemble. Fix errors and retry.[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Assembling final image...", total=None)

        final_image = gen_service.assemble_tiles(results, apply_perspective=perspective)

        if final_image is None:
            progress.update(task, description="[red]Assembly failed")
            raise SystemExit(1)

        progress.update(task, completed=True, description="[green]Assembly complete")

    # Save result
    if output:
        output_path = Path(output)
    else:
        base_name = project.name.replace(' ', '_') + "_illustrated"
        output_path = project.output_dir / timestamped_filename(base_name)
    final_image.save(output_path)

    console.print(f"\n[green]Success![/green] Final image saved to: {output_path}")
    console.print(f"[dim]Size: {final_image.width} x {final_image.height} px[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
def save_tiles(project_path: str):
    """Copy generated tiles from cache to project folder for permanent storage.

    This copies all cached tiles (both generated and reference) to the project's
    output/tiles/ directory so they persist across cache clears.
    """
    import shutil

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    config = get_config()
    cache_dir = config.cache_dir / "generation"

    # Create tiles directories in project
    tiles_dir = project.output_dir / "tiles"
    generated_dir = tiles_dir / "generated"
    references_dir = tiles_dir / "references"
    generated_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)

    # Copy generated tiles
    src_generated = cache_dir / "generated"
    if src_generated.exists():
        copied = 0
        for tile_file in src_generated.glob("tile_*.png"):
            shutil.copy2(tile_file, generated_dir / tile_file.name)
            copied += 1
        console.print(f"[green]Copied {copied} generated tiles[/green] to {generated_dir}")
    else:
        console.print("[yellow]No generated tiles found in cache[/yellow]")

    # Copy reference tiles
    src_references = cache_dir / "references"
    if src_references.exists():
        copied = 0
        for ref_file in src_references.glob("tile_*_ref.png"):
            shutil.copy2(ref_file, references_dir / ref_file.name)
            copied += 1
        console.print(f"[green]Copied {copied} reference tiles[/green] to {references_dir}")

    console.print(f"\n[bold]Tiles saved to:[/bold] {tiles_dir}")
    console.print("[dim]These tiles will persist even if the cache is cleared.[/dim]")


def _fetch_satellite_imagery(project: Project, output_size: tuple[int, int]):
    """Fetch Mapbox satellite imagery for project."""
    from .services.satellite_service import SatelliteService

    config = get_config()
    satellite_service = SatelliteService(cache_dir=str(config.cache_dir / "satellite"))

    return satellite_service.fetch_satellite_imagery(
        bbox=project.region,
        output_size=output_size,
    )


def _fetch_simplified_osm_data(project: Project):
    """Fetch simplified OSM data for project."""
    from .services.osm_service import OSMService

    config = get_config()
    osm_service = OSMService(cache_dir=str(config.cache_dir / "osm"))
    return osm_service.fetch_region_data(project.region, detail_level="simplified")


def _render_composite(
    project: Project,
    satellite_image,
    osm_data,
    output_size,
    apply_perspective: bool = True,
    convergence: float = 0.7,
    vertical_scale: float = 0.4,
    horizon_margin: float = 0.15,
):
    """Render composite reference image with optional aerial perspective."""
    from .services.perspective_service import PerspectiveService
    from .services.render_service import RenderService

    perspective = PerspectiveService(
        angle=project.style.perspective_angle,
        convergence=convergence,
        vertical_scale=vertical_scale,
        horizon_margin=horizon_margin,
    )
    render = RenderService(perspective_service=perspective)

    return render.render_composite_reference(
        satellite_image=satellite_image,
        osm_data=osm_data,
        bbox=project.region,
        output_size=output_size,
        osm_opacity=0.5,
        apply_perspective=apply_perspective,
    )


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
def list_seams(project_path: str):
    """List all seam locations in the tile grid.

    Shows tile coordinates for each seam that can be repaired.
    Use the output with repair-seam to fix specific discontinuities.
    """
    from rich.table import Table

    from .services.seam_repair_service import SeamRepairService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    # Calculate grid dimensions
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)

    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Tile grid:[/bold] {cols} columns x {rows} rows")
    console.print(f"[bold]Tile size:[/bold] {project.tiles.size}px with {project.tiles.overlap}px overlap\n")

    # Get all seams
    repair_service = SeamRepairService(
        tile_size=project.tiles.size,
        overlap=project.tiles.overlap,
    )
    seams = repair_service.identify_seams(cols, rows)

    # Split into horizontal and vertical
    h_seams = [s for s in seams if s.orientation == "horizontal"]
    v_seams = [s for s in seams if s.orientation == "vertical"]

    # Horizontal seams table
    console.print(f"[bold cyan]Horizontal Seams ({len(h_seams)}):[/bold cyan]")
    h_table = Table()
    h_table.add_column("Seam ID", style="cyan")
    h_table.add_column("Tiles", style="green")
    h_table.add_column("Position (x, y)")
    h_table.add_column("Size")

    for seam in h_seams:
        h_table.add_row(
            seam.id,
            seam.description,
            f"({seam.x}, {seam.y})",
            f"{seam.width}x{seam.height}",
        )
    console.print(h_table)

    # Vertical seams table
    console.print(f"\n[bold cyan]Vertical Seams ({len(v_seams)}):[/bold cyan]")
    v_table = Table()
    v_table.add_column("Seam ID", style="cyan")
    v_table.add_column("Tiles", style="green")
    v_table.add_column("Position (x, y)")
    v_table.add_column("Size")

    for seam in v_seams:
        v_table.add_row(
            seam.id,
            seam.description,
            f"({seam.x}, {seam.y})",
            f"{seam.width}x{seam.height}",
        )
    console.print(v_table)

    console.print(f"\n[bold]Total:[/bold] {len(seams)} seams")
    console.print("\n[dim]Use 'mapgen repair-seam project/ --tiles COL,ROW-COL,ROW' to repair a seam[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--tiles", "-t", multiple=True, required=True,
              help="Seam to repair in format 'COL,ROW-COL,ROW' (e.g., '1,2-2,2')")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Input assembled image (defaults to latest in output/)")
@click.option("--output", "-o", type=click.Path(),
              help="Output file path")
def repair_seam(
    project_path: str,
    tiles: tuple[str],
    input_file: Optional[str],
    output: Optional[str],
):
    """Repair specific seams between tiles using Gemini inpainting.

    Specify seams by tile coordinates, e.g.:
        mapgen repair-seam project/ --tiles 1,2-2,2
        mapgen repair-seam project/ --tiles 0,0-1,0 --tiles 1,0-2,0
    """
    from .config import get_config
    from .services.gemini_service import GeminiService
    from .services.seam_repair_service import SeamRepairService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    config = get_config()

    # Calculate grid dimensions
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)

    # Create services
    repair_service = SeamRepairService(
        tile_size=project.tiles.size,
        overlap=project.tiles.overlap,
    )
    gemini = GeminiService()

    # Get all seams
    all_seams = repair_service.identify_seams(cols, rows)

    # Parse requested seams
    seams_to_repair = []
    for spec in tiles:
        try:
            tile_a, tile_b = repair_service.parse_seam_spec(spec)
            seam = repair_service.find_seam(all_seams, tile_a, tile_b)
            if seam is None:
                console.print(f"[yellow]Warning:[/yellow] No seam found for {spec}")
            else:
                seams_to_repair.append(seam)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)

    if not seams_to_repair:
        console.print("[red]No valid seams to repair[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Repairing {len(seams_to_repair)} seam(s)...[/bold]")

    # Load input image
    if input_file:
        input_path = Path(input_file)
    else:
        # Find latest assembled image
        output_dir = project.output_dir
        candidates = list(output_dir.glob("*illustrated*.png")) + list(output_dir.glob("*assembled*.png"))
        if not candidates:
            console.print("[red]No assembled image found. Specify --input or run generate-tiles first.[/red]")
            raise SystemExit(1)
        input_path = max(candidates, key=lambda p: p.stat().st_mtime)
        console.print(f"[dim]Using: {input_path}[/dim]")

    assembled = Image.open(input_path).convert("RGBA")
    cache_dir = config.cache_dir / "generation"

    # Repair seams
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for seam in seams_to_repair:
            task = progress.add_task(f"Repairing {seam.id}...", total=None)

            # Load tiles
            tile_a = repair_service.load_tile(cache_dir, seam.tile_a[0], seam.tile_a[1])
            tile_b = repair_service.load_tile(cache_dir, seam.tile_b[0], seam.tile_b[1])

            if tile_a is None or tile_b is None:
                progress.update(task, description=f"[red]Missing tiles for {seam.id}[/red]")
                continue

            result = repair_service.repair_seam(seam, tile_a, tile_b, gemini)

            if result.error:
                progress.update(task, description=f"[red]Failed: {seam.id} - {result.error}[/red]")
            else:
                assembled = repair_service.apply_repair(assembled, seam, result.repaired_region)
                progress.update(task, completed=True,
                               description=f"[green]Repaired {seam.id}[/green] ({result.generation_time:.1f}s)")

    # Save result
    if output:
        output_path = Path(output)
    else:
        output_path = project.output_dir / timestamped_filename("repaired")
    assembled.save(output_path)

    console.print(f"\n[green]Saved:[/green] {output_path}")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--tile", "-t", multiple=True, required=True,
              help="Tile to regenerate in format 'COL,ROW' (e.g., '2,3')")
@click.option("--no-cache", is_flag=True, help="Force regeneration even if cached")
def regenerate_tile(
    project_path: str,
    tile: tuple[str],
    no_cache: bool,
):
    """Regenerate specific tiles using Gemini.

    Use this to fix tiles that have issues. After regenerating,
    run 'assemble' to rebuild the final image.

    Examples:
        mapgen regenerate-tile project/ --tile 2,3
        mapgen regenerate-tile project/ --tile 2,3 --tile 2,4 --no-cache
    """
    from .config import get_config
    from .services.generation_service import GenerationService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    config = get_config()

    # Parse tile coordinates
    tiles_to_regen = []
    for spec in tile:
        try:
            parts = spec.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid format: {spec}")
            col, row = int(parts[0]), int(parts[1])
            tiles_to_regen.append((col, row))
        except (ValueError, IndexError):
            console.print(f"[red]Error:[/red] Invalid tile spec '{spec}'. Expected format: COL,ROW")
            raise SystemExit(1)

    # Create generation service
    gen_service = GenerationService(
        project=project,
        cache_dir=config.cache_dir / "generation",
    )

    # Get all tile specs
    all_specs = gen_service.calculate_tile_specs()

    # Find matching specs
    specs_to_regen = []
    for col, row in tiles_to_regen:
        found = False
        for spec in all_specs:
            if spec.col == col and spec.row == row:
                specs_to_regen.append(spec)
                found = True
                break
        if not found:
            console.print(f"[yellow]Warning:[/yellow] Tile ({col}, {row}) not in grid")

    if not specs_to_regen:
        console.print("[red]No valid tiles to regenerate[/red]")
        raise SystemExit(1)

    # Delete cached tiles if --no-cache
    if no_cache:
        cache_dir = config.cache_dir / "generation" / "generated"
        for spec in specs_to_regen:
            cache_file = cache_dir / f"tile_{spec.col}_{spec.row}.png"
            if cache_file.exists():
                cache_file.unlink()
                console.print(f"[dim]Cleared cache for ({spec.col}, {spec.row})[/dim]")

    console.print(f"[bold]Regenerating {len(specs_to_regen)} tile(s)...[/bold]")

    # Pre-fetch OSM data
    gen_service.fetch_osm_data()

    # Regenerate tiles
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for spec in specs_to_regen:
            task = progress.add_task(f"Regenerating ({spec.col}, {spec.row})...", total=None)

            result = gen_service.generate_tile(spec)

            if result.error:
                progress.update(task, description=f"[red]Failed ({spec.col}, {spec.row}): {result.error}[/red]")
            else:
                progress.update(
                    task, completed=True,
                    description=f"[green]Done ({spec.col}, {spec.row})[/green] in {result.generation_time:.1f}s"
                )

    console.print("\n[dim]Run 'mapgen assemble project/' to rebuild the final image.[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--perspective/--no-perspective", default=True, help="Apply aerial perspective")
def assemble(
    project_path: str,
    output: Optional[str],
    perspective: bool,
):
    """Assemble cached tiles into final image.

    Uses previously generated tiles from cache to build the final map.
    Useful after regenerating specific tiles or when you want to
    rebuild with different perspective settings.
    """
    from .config import get_config
    from .services.blending_service import BlendingService, TileInfo
    from .services.generation_service import GenerationService, TileResult
    from .services.perspective_service import PerspectiveService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    config = get_config()

    # Create services
    gen_service = GenerationService(
        project=project,
        cache_dir=config.cache_dir / "generation",
    )

    # Get tile specs
    specs = gen_service.calculate_tile_specs()
    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)

    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Tile grid:[/bold] {cols} x {rows} = {len(specs)} tiles")

    # Load cached tiles
    cache_dir = config.cache_dir / "generation" / "generated"
    loaded = []
    missing = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading cached tiles...", total=len(specs))

        for spec in specs:
            tile_path = cache_dir / f"tile_{spec.col}_{spec.row}.png"
            if tile_path.exists():
                tile_img = Image.open(tile_path).convert("RGBA")
                result = TileResult(spec=spec, generated_image=tile_img)
                loaded.append(result)
            else:
                missing.append(spec)
            progress.advance(task)

    console.print(f"[green]Loaded {len(loaded)} tiles[/green]")
    if missing:
        console.print(f"[yellow]Missing {len(missing)} tiles:[/yellow]")
        for spec in missing[:5]:
            console.print(f"  - ({spec.col}, {spec.row})")
        if len(missing) > 5:
            console.print(f"  ... and {len(missing) - 5} more")

    if len(loaded) < len(specs) * 0.8:
        console.print("[red]Too many missing tiles. Run generate-tiles first.[/red]")
        raise SystemExit(1)

    # Assemble
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Assembling final image...", total=None)

        final_image = gen_service.assemble_tiles(loaded, apply_perspective=perspective)

        if final_image is None:
            progress.update(task, description="[red]Assembly failed[/red]")
            raise SystemExit(1)

        progress.update(task, completed=True, description="[green]Assembly complete[/green]")

    # Save result
    if output:
        output_path = Path(output)
    else:
        base_name = project.name.replace(' ', '_') + "_assembled"
        output_path = project.output_dir / timestamped_filename(base_name)
    final_image.save(output_path)

    console.print(f"\n[green]Saved:[/green] {output_path}")
    console.print(f"[dim]Size: {final_image.width} x {final_image.height} px[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Landmark name")
@click.option("--lat", type=float, required=True, help="Latitude coordinate")
@click.option("--lon", type=float, required=True, help="Longitude coordinate")
@click.option("--photo", "-p", type=click.Path(), help="Path to landmark photo (relative to project dir)")
@click.option("--logo", "-l", type=click.Path(), help="Path to logo PNG (relative to project dir)")
@click.option("--scale", "-s", type=float, default=1.5, help="Scale factor (1.0=actual, >1=exaggerated)")
@click.option("--z-index", "-z", type=int, default=0, help="Z-index for layering")
def add_landmark(
    project_path: str,
    name: str,
    lat: float,
    lon: float,
    photo: Optional[str],
    logo: Optional[str],
    scale: float,
    z_index: int,
):
    """Add a landmark to the project.

    The landmark will be added to project.yaml and can be illustrated
    and composited onto the map.

    Examples:
        mapgen add-landmark project/ --name "Empire State Building" \\
            --lat 40.7484 --lon -73.9857 \\
            --photo landmarks/empire_state.jpg \\
            --logo logos/empire_state.png --scale 2.0
    """
    from .models.landmark import Landmark

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    # Check if landmark already exists
    existing = [l for l in project.landmarks if l.name.lower() == name.lower()]
    if existing:
        console.print(f"[yellow]Warning:[/yellow] Landmark '{name}' already exists. Use remove-landmark first.")
        raise SystemExit(1)

    # Validate photo path exists if provided
    if photo:
        photo_path = project.project_dir / photo
        if not photo_path.exists():
            console.print(f"[yellow]Warning:[/yellow] Photo file not found: {photo_path}")

    # Validate logo path exists if provided
    if logo:
        logo_path = project.project_dir / logo
        if not logo_path.exists():
            console.print(f"[yellow]Warning:[/yellow] Logo file not found: {logo_path}")

    # Create landmark
    landmark = Landmark(
        name=name,
        latitude=lat,
        longitude=lon,
        photo=photo,
        logo=logo,
        scale=scale,
        z_index=z_index,
    )

    # Add to project and save
    project.landmarks.append(landmark)
    project.to_yaml(project_file)

    console.print(f"[green]Added landmark:[/green] {name}")
    console.print(f"  Location: ({lat:.6f}, {lon:.6f})")
    if photo:
        console.print(f"  Photo: {photo}")
    if logo:
        console.print(f"  Logo: {logo}")
    console.print(f"  Scale: {scale}x | Z-index: {z_index}")
    console.print(f"\n[dim]Total landmarks: {len(project.landmarks)}[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
def list_landmarks(project_path: str):
    """List all landmarks in the project."""
    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    if not project.landmarks:
        console.print("[yellow]No landmarks defined in this project.[/yellow]")
        console.print("[dim]Use 'mapgen add-landmark' to add landmarks.[/dim]")
        return

    table = Table(title=f"Landmarks in {project.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Latitude", style="green")
    table.add_column("Longitude", style="green")
    table.add_column("Photo", style="dim")
    table.add_column("Logo", style="dim")
    table.add_column("Scale")
    table.add_column("Z")

    for landmark in project.landmarks:
        photo_status = "✓" if landmark.photo else "–"
        logo_status = "✓" if landmark.logo else "–"

        table.add_row(
            landmark.name,
            f"{landmark.latitude:.6f}",
            f"{landmark.longitude:.6f}",
            photo_status,
            logo_status,
            f"{landmark.scale}x",
            str(landmark.z_index),
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(project.landmarks)} landmarks")

    # Estimate illustration cost
    from .services.landmark_service import LandmarkService

    landmark_service = LandmarkService(project)
    cost = landmark_service.estimate_cost()

    if cost["landmarks_with_photos"] > 0:
        console.print(f"[dim]Illustration cost: ~${cost['estimated_cost']:.2f} "
                     f"({cost['landmarks_with_photos']} with photos)[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Landmark name to remove")
def remove_landmark(project_path: str, name: str):
    """Remove a landmark from the project."""
    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    # Find landmark by name (case-insensitive)
    original_count = len(project.landmarks)
    project.landmarks = [l for l in project.landmarks if l.name.lower() != name.lower()]

    if len(project.landmarks) == original_count:
        console.print(f"[yellow]Landmark not found:[/yellow] {name}")
        raise SystemExit(1)

    # Save updated project
    project.to_yaml(project_file)

    console.print(f"[green]Removed landmark:[/green] {name}")
    console.print(f"[dim]Remaining landmarks: {len(project.landmarks)}[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Illustrate specific landmark by name (default: all)")
@click.option("--style-ref", type=click.Path(exists=True), help="Path to style reference image")
def illustrate_landmarks(
    project_path: str,
    name: Optional[str],
    style_ref: Optional[str],
):
    """Illustrate landmarks using Gemini AI.

    Transforms landmark photos into illustrated style matching the map.
    Results are saved to output/landmarks/ directory.

    Examples:
        mapgen illustrate-landmarks project/
        mapgen illustrate-landmarks project/ --name "Empire State Building"
    """
    from .services.landmark_service import LandmarkService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()

    landmark_service = LandmarkService(project)

    # Get landmarks to illustrate
    if name:
        landmark = landmark_service.get_landmark_by_name(name)
        if landmark is None:
            console.print(f"[red]Landmark not found:[/red] {name}")
            raise SystemExit(1)
        landmarks = [landmark]
    else:
        landmarks = landmark_service.get_landmarks()

    if not landmarks:
        console.print("[yellow]No landmarks to illustrate.[/yellow]")
        return

    # Check for photos
    landmarks_with_photos = [l for l in landmarks if l.photo is not None]
    if not landmarks_with_photos:
        console.print("[yellow]No landmarks have photos. Add photos first.[/yellow]")
        return

    # Load style reference
    style_reference = None
    if style_ref:
        style_reference = Image.open(style_ref).convert("RGBA")
        console.print(f"[dim]Using style reference: {style_ref}[/dim]")
    else:
        style_reference = landmark_service.get_style_reference()
        if style_reference:
            console.print("[dim]Using generated tile as style reference[/dim]")

    console.print(f"[bold]Illustrating {len(landmarks_with_photos)} landmark(s)...[/bold]")

    # Estimate cost
    cost = len(landmarks_with_photos) * 0.13
    console.print(f"[dim]Estimated cost: ~${cost:.2f}[/dim]")

    if not click.confirm("Continue?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Illustrate with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        def progress_callback(current: int, total: int, landmark):
            pass  # Progress is handled by the task update

        for i, landmark in enumerate(landmarks_with_photos):
            task = progress.add_task(f"Illustrating {landmark.name}...", total=None)

            result = landmark_service.illustrate_landmark(landmark, style_reference)

            if result.error:
                progress.update(task, description=f"[red]Failed: {landmark.name} - {result.error}[/red]")
            else:
                # Save the result
                output_path = landmark_service.save_illustrated(landmark, result.image)
                progress.update(
                    task, completed=True,
                    description=f"[green]{landmark.name}[/green] ({result.generation_time:.1f}s)"
                )

    console.print(f"\n[green]Done![/green] Illustrations saved to: {landmark_service.output_dir}")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--base-map", "-b", type=click.Path(exists=True), required=True,
              help="Base map image to composite landmarks onto")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--with-logos/--no-logos", default=True, help="Include logos next to landmarks")
@click.option("--with-shadows/--no-shadows", default=True, help="Add drop shadows")
@click.option("--perspective/--no-perspective", default=True,
              help="Apply perspective transform to landmark positions (for perspective maps)")
@click.option("--convergence", type=float, default=0.7,
              help="Perspective convergence (must match map generation)")
@click.option("--vertical-scale", type=float, default=0.4,
              help="Perspective vertical scale (must match map generation)")
@click.option("--horizon-margin", type=float, default=0.15,
              help="Perspective horizon margin (must match map generation)")
def compose(
    project_path: str,
    base_map: str,
    output: Optional[str],
    with_logos: bool,
    with_shadows: bool,
    perspective: bool,
    convergence: float,
    vertical_scale: float,
    horizon_margin: float,
):
    """Composite illustrated landmarks onto base map.

    Places landmarks at their GPS coordinates with optional logos and shadows.
    For perspective-warped maps, use --perspective to adjust landmark positions
    and scales to match the map's perspective.

    Examples:
        mapgen compose project/ --base-map output/illustrated.png
        mapgen compose project/ -b output/illustrated.png -o final.png --no-logos
        mapgen compose project/ -b output/flat.png --no-perspective
    """
    from .services.composition_service import CompositionService
    from .services.landmark_service import LandmarkService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    landmark_service = LandmarkService(project)
    landmarks = landmark_service.get_landmarks()

    if not landmarks:
        console.print("[yellow]No landmarks in project.[/yellow]")
        return

    # Load base map
    base_image = Image.open(base_map).convert("RGBA")
    console.print(f"[bold]Base map:[/bold] {base_map}")
    console.print(f"[bold]Size:[/bold] {base_image.width} x {base_image.height}")
    console.print(f"[bold]Landmarks:[/bold] {len(landmarks)}")
    console.print(f"[bold]Perspective:[/bold] {'enabled' if perspective else 'disabled'}")

    # Create composition service with perspective parameters
    composition = CompositionService(
        perspective_convergence=convergence,
        perspective_vertical_scale=vertical_scale,
        perspective_horizon_margin=horizon_margin,
    )
    base_map_size = (base_image.width, base_image.height)

    # Load and place landmarks
    placed_landmarks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading and placing landmarks...", total=len(landmarks))

        for landmark in landmarks:
            # Load illustrated image
            illustration = landmark_service.load_illustrated(landmark)
            if illustration is None:
                console.print(f"[yellow]Skipping {landmark.name} - no illustration found[/yellow]")
                progress.advance(task)
                continue

            # Place landmark on map with perspective adjustment
            placed = composition.place_landmark(
                landmark=landmark,
                illustration=illustration,
                base_map_size=base_map_size,
                bbox=project.region,
                apply_perspective=perspective,
                remove_background=True,
            )

            # Load and place logo if enabled
            if with_logos:
                logo = landmark_service.load_logo(landmark)
                if logo is not None:
                    composition.place_logo(placed, logo, max_width=300)

            placed_landmarks.append(placed)
            progress.advance(task)

    console.print(f"[dim]Placed {len(placed_landmarks)} landmarks[/dim]")

    if not placed_landmarks:
        console.print("[yellow]No illustrated landmarks found. Run illustrate-landmarks first.[/yellow]")
        return

    # Avoid logo collisions
    placed_landmarks = composition.avoid_collisions(placed_landmarks, base_map_size)

    # Composite onto base map
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Compositing landmarks...", total=None)

        result = composition.composite_map(
            base_map=base_image,
            placed_landmarks=placed_landmarks,
            add_shadows=with_shadows,
        )

        progress.update(task, completed=True, description="[green]Composition complete[/green]")

    # Save result
    if output:
        output_path = Path(output)
    else:
        base_name = project.name.replace(' ', '_') + "_with_landmarks"
        output_path = project.output_dir / timestamped_filename(base_name)

    result.save(output_path)

    console.print(f"\n[green]Saved:[/green] {output_path}")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--skip-tiles", is_flag=True, help="Skip tile generation (use cached)")
@click.option("--skip-landmarks", is_flag=True, help="Skip landmark illustration")
@click.option("--perspective/--no-perspective", default=True, help="Apply aerial perspective")
def generate_full(
    project_path: str,
    output: Optional[str],
    skip_tiles: bool,
    skip_landmarks: bool,
    perspective: bool,
):
    """Generate full illustrated map with landmarks.

    Complete pipeline:
    1. Generate all tiles using Gemini
    2. Assemble tiles into base map
    3. Illustrate landmarks using Gemini
    4. Composite landmarks onto base map

    Examples:
        mapgen generate-full project/
        mapgen generate-full project/ --skip-tiles  # Use cached tiles
    """
    from .config import get_config
    from .services.composition_service import CompositionService
    from .services.generation_service import GenerationService
    from .services.landmark_service import LandmarkService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    project.ensure_directories()
    config = get_config()

    # Initialize services
    gen_service = GenerationService(
        project=project,
        cache_dir=config.cache_dir / "generation",
    )
    landmark_service = LandmarkService(project)

    # Show summary
    specs = gen_service.calculate_tile_specs()
    landmarks = landmark_service.get_landmarks()
    landmarks_with_photos = [l for l in landmarks if l.photo is not None]

    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Tiles:[/bold] {len(specs)}")
    console.print(f"[bold]Landmarks:[/bold] {len(landmarks)} ({len(landmarks_with_photos)} with photos)")

    # Estimate total cost
    tile_cost = 0 if skip_tiles else len(specs) * 0.13
    landmark_cost = 0 if skip_landmarks else len(landmarks_with_photos) * 0.13
    total_cost = tile_cost + landmark_cost

    console.print(f"[bold]Estimated cost:[/bold] ${total_cost:.2f}")
    console.print(f"  Tiles: ${tile_cost:.2f} {'(skipped)' if skip_tiles else ''}")
    console.print(f"  Landmarks: ${landmark_cost:.2f} {'(skipped)' if skip_landmarks else ''}")

    if total_cost > 0 and not click.confirm("\nContinue?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Phase 1: Generate tiles
    if not skip_tiles:
        console.print("\n[bold cyan]Phase 1: Generating tiles[/bold cyan]")

        # Pre-fetch OSM data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching OSM data...", total=None)
            gen_service.fetch_osm_data()
            progress.update(task, completed=True, description="[green]OSM data fetched")

        from rich.progress import BarColumn, TaskProgressColumn, TimeRemainingColumn

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating tiles", total=len(specs))

            for spec in specs:
                result = gen_service.generate_tile(spec)
                results.append(result)

                if result.error:
                    console.print(f"  [red]Failed:[/red] ({spec.col}, {spec.row})")
                progress.advance(task)

        successful = [r for r in results if not r.error]
        console.print(f"[green]Generated {len(successful)}/{len(specs)} tiles[/green]")
    else:
        console.print("\n[bold cyan]Phase 1: Loading cached tiles[/bold cyan]")
        # Load cached tiles for assembly
        from .services.generation_service import TileResult

        cache_dir = config.cache_dir / "generation" / "generated"
        results = []
        for spec in specs:
            tile_path = cache_dir / f"tile_{spec.col}_{spec.row}.png"
            if tile_path.exists():
                tile_img = Image.open(tile_path).convert("RGBA")
                results.append(TileResult(spec=spec, generated_image=tile_img))
        console.print(f"[green]Loaded {len(results)} cached tiles[/green]")

    # Phase 2: Assemble tiles
    console.print("\n[bold cyan]Phase 2: Assembling base map[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Assembling...", total=None)
        base_map = gen_service.assemble_tiles(results, apply_perspective=perspective)
        progress.update(task, completed=True, description="[green]Assembly complete")

    if base_map is None:
        console.print("[red]Assembly failed[/red]")
        raise SystemExit(1)

    # Save base map
    base_map_path = project.output_dir / timestamped_filename(
        project.name.replace(' ', '_') + "_base"
    )
    base_map.save(base_map_path)
    console.print(f"[dim]Base map saved: {base_map_path}[/dim]")

    # Phase 3: Illustrate landmarks
    if not skip_landmarks and landmarks_with_photos:
        console.print("\n[bold cyan]Phase 3: Illustrating landmarks[/bold cyan]")

        # Get style reference from first tile
        style_reference = landmark_service.get_style_reference()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for landmark in landmarks_with_photos:
                task = progress.add_task(f"Illustrating {landmark.name}...", total=None)

                result = landmark_service.illustrate_landmark(landmark, style_reference)

                if result.error:
                    progress.update(task, description=f"[red]Failed: {landmark.name}[/red]")
                else:
                    landmark_service.save_illustrated(landmark, result.image)
                    progress.update(
                        task, completed=True,
                        description=f"[green]{landmark.name}[/green]"
                    )

    # Phase 4: Composite landmarks
    if landmarks:
        console.print("\n[bold cyan]Phase 4: Compositing landmarks[/bold cyan]")

        # Use perspective parameters matching the map generation
        composition = CompositionService(
            perspective_convergence=0.7,
            perspective_vertical_scale=0.4,
            perspective_horizon_margin=0.15,
        )
        base_map_size = (base_map.width, base_map.height)

        # Place landmarks with perspective adjustment
        placed_landmarks = []
        for landmark in landmarks:
            illustration = landmark_service.load_illustrated(landmark)
            if illustration is None:
                continue

            placed = composition.place_landmark(
                landmark=landmark,
                illustration=illustration,
                base_map_size=base_map_size,
                bbox=project.region,
                apply_perspective=perspective,
                remove_background=True,
            )

            logo = landmark_service.load_logo(landmark)
            if logo is not None:
                composition.place_logo(placed, logo, max_width=300)

            placed_landmarks.append(placed)

        if placed_landmarks:
            # Avoid collisions and composite
            placed_landmarks = composition.avoid_collisions(placed_landmarks, base_map_size)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Compositing...", total=None)

                final_image = composition.composite_map(
                    base_map=base_map,
                    placed_landmarks=placed_landmarks,
                    add_shadows=True,
                )

                progress.update(task, completed=True, description="[green]Composition complete[/green]")
        else:
            console.print("[yellow]No illustrated landmarks to composite[/yellow]")
            final_image = base_map
    else:
        final_image = base_map

    # Save final result
    if output:
        output_path = Path(output)
    else:
        base_name = project.name.replace(' ', '_') + "_final"
        output_path = project.output_dir / timestamped_filename(base_name)

    final_image.save(output_path)

    console.print(f"\n[green]Complete![/green]")
    console.print(f"[bold]Final image:[/bold] {output_path}")
    console.print(f"[dim]Size: {final_image.width} x {final_image.height} px[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Input perspective-transformed map image")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--convergence", type=float, default=0.7,
              help="Perspective convergence factor (0.0-1.0)")
@click.option("--vertical-scale", type=float, default=0.4,
              help="Vertical compression at top (0.0-1.0)")
@click.option("--horizon-margin", type=float, default=0.15,
              help="Horizon band height as fraction of content height")
@click.option("--fill-color", type=str, default="#808080",
              help="Fill color for empty regions (hex, e.g., #808080)")
@click.option("--dry-run", is_flag=True, help="Show info without generating")
def outpaint(
    project_path: str,
    input_file: Optional[str],
    output: Optional[str],
    convergence: float,
    vertical_scale: float,
    horizon_margin: float,
    fill_color: str,
    dry_run: bool,
):
    """Outpaint empty regions in a perspective-transformed map.

    After applying perspective transform, the map becomes trapezoidal with
    empty regions (horizon band at top, triangles on sides).

    This command uses a single-generation + upscale approach:
    1. Fill empty regions with grey
    2. Downscale image to fit Gemini limits
    3. Single Gemini call to complete the image
    4. Upscale with Real-ESRGAN
    5. Merge with original high-res content

    Example:
        mapgen outpaint projects/nyc/ -i output/final_with_perspective.png
    """
    from .services.outpainting_service import OutpaintingService

    console = Console()
    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)
    console.print(f"[bold]Project:[/bold] {project.name}")

    # Find input image
    if input_file:
        image_path = Path(input_file)
    else:
        # Look for perspective-transformed image in output
        output_dir = project.output_dir
        candidates = list(output_dir.glob("*perspective*.png"))
        candidates.extend(output_dir.glob("*final*.png"))
        if not candidates:
            console.print("[red]No input image specified and none found in output/[/red]")
            console.print("[dim]Use --input to specify the perspective-transformed map[/dim]")
            raise SystemExit(1)
        # Use most recent
        image_path = max(candidates, key=lambda p: p.stat().st_mtime)
        console.print(f"[dim]Using: {image_path}[/dim]")

    # Load image
    image = Image.open(image_path).convert("RGBA")
    console.print(f"[dim]Image size: {image.width} x {image.height} px[/dim]")

    # Parse fill color
    try:
        fill_color_rgb = tuple(int(fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        console.print(f"[red]Invalid fill color: {fill_color}[/red]")
        raise SystemExit(1)

    # Initialize service
    outpainting = OutpaintingService(
        convergence=convergence,
        vertical_scale=vertical_scale,
        horizon_margin=horizon_margin,
        fill_color=fill_color_rgb,
    )

    # Calculate and display info
    scale_factor = outpainting.calculate_scale_factor(image.size)
    downscaled_size = (
        int(image.width * scale_factor),
        int(image.height * scale_factor),
    )

    console.print(f"\n[bold]Outpainting Strategy:[/bold]")
    console.print(f"  Fill color: {fill_color}")
    console.print(f"  Downscale: {image.width}x{image.height} → {downscaled_size[0]}x{downscaled_size[1]} (factor: {scale_factor:.3f})")
    console.print(f"  Upscale factor: {1/scale_factor:.1f}x with Real-ESRGAN")

    # Show cost estimate
    cost = outpainting.estimate_cost()
    console.print(f"\n[bold]Estimated cost:[/bold] ${cost['total_cost']:.2f}")
    console.print(f"  Single Gemini generation: ${cost['gemini_cost']:.2f}")
    console.print(f"  Real-ESRGAN upscale: ${cost['upscale_cost']:.2f} (local)")

    if dry_run:
        console.print("\n[yellow]Dry run - no images generated[/yellow]")
        return

    # Confirm
    if not click.confirm("\nProceed with outpainting?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Run outpainting with progress
    console.print("\n[bold cyan]Outpainting...[/bold cyan]")

    def progress_callback(message: str, progress: float):
        console.print(f"  [dim]{message}[/dim]")

    result = outpainting.outpaint_image(
        image=image,
        bbox=project.region,
        progress_callback=progress_callback,
    )

    # Save result
    if output:
        output_path = Path(output)
    else:
        base_name = image_path.stem + "_outpainted"
        output_path = project.output_dir / f"{base_name}.png"

    result.save(output_path)
    console.print(f"\n[green]Complete![/green]")
    console.print(f"[bold]Output:[/bold] {output_path}")
    console.print(f"[dim]Size: {result.width} x {result.height} px[/dim]")


@main.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--base-map", "-b", type=click.Path(exists=True), required=True,
              help="Base map image (final composed map)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "export_format", type=click.Choice(["psd", "layers"]), default="psd",
              help="Export format: psd (layered PSD) or layers (separate PNGs)")
@click.option("--with-logos/--no-logos", default=True, help="Include logo labels")
@click.option("--perspective/--no-perspective", default=True,
              help="Use perspective-adjusted positions")
@click.option("--convergence", default=0.7, help="Perspective convergence (0.0-1.0)")
@click.option("--vertical-scale", default=0.4, help="Vertical scale at top (0.0-1.0)")
@click.option("--horizon-margin", default=0.15, help="Horizon margin fraction")
def export_psd(
    project_path: str,
    base_map: str,
    output: Optional[str],
    export_format: str,
    with_logos: bool,
    perspective: bool,
    convergence: float,
    vertical_scale: float,
    horizon_margin: float,
):
    """Export map as layered PSD file for editing in Photoshop/GIMP.

    Creates a PSD with separate layers for the base map and each landmark,
    allowing for post-editing of individual elements.

    Examples:
        mapgen export-psd project/ --base-map output/final.png -o map.psd
        mapgen export-psd project/ -b output/final.png --format layers -o layers/
    """
    from .services.composition_service import CompositionService
    from .services.landmark_service import LandmarkService
    from .services.psd_service import PSDService

    project_file = Path(project_path)
    if project_file.is_dir():
        project_file = project_file / "project.yaml"

    project = Project.from_yaml(project_file)

    # Load services
    landmark_service = LandmarkService(project)
    landmarks = landmark_service.get_landmarks()

    # Load base map
    base_image = Image.open(base_map).convert("RGBA")
    console.print(f"[bold]Project:[/bold] {project.name}")
    console.print(f"[bold]Base map:[/bold] {base_map}")
    console.print(f"[bold]Size:[/bold] {base_image.width} x {base_image.height}")
    console.print(f"[bold]Landmarks:[/bold] {len(landmarks)}")
    console.print(f"[bold]Format:[/bold] {export_format.upper()}")

    # Create composition service
    composition = CompositionService(
        perspective_convergence=convergence,
        perspective_vertical_scale=vertical_scale,
        perspective_horizon_margin=horizon_margin,
    )
    base_map_size = (base_image.width, base_image.height)

    # Load and place landmarks
    placed_landmarks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading landmarks...", total=len(landmarks))

        for landmark in landmarks:
            # Load illustrated image
            illustration = landmark_service.load_illustrated(landmark)
            if illustration is None:
                console.print(f"[yellow]Skipping {landmark.name} - no illustration[/yellow]")
                progress.advance(task)
                continue

            # Place landmark
            placed = composition.place_landmark(
                landmark=landmark,
                illustration=illustration,
                base_map_size=base_map_size,
                bbox=project.region,
                apply_perspective=perspective,
                remove_background=True,
            )

            # Load and attach logo
            if with_logos:
                logo = landmark_service.load_logo(landmark)
                if logo is not None:
                    composition.place_logo(placed, logo, max_width=300)

            placed_landmarks.append(placed)
            progress.advance(task)

    console.print(f"[dim]Found {len(placed_landmarks)} illustrated landmarks[/dim]")

    if not placed_landmarks:
        console.print("[yellow]No illustrated landmarks found.[/yellow]")
        console.print("[dim]Run 'illustrate-landmarks' first, or export without landmarks.[/dim]")

    # Avoid logo collisions
    if placed_landmarks:
        placed_landmarks = composition.avoid_collisions(placed_landmarks, base_map_size)

    # Create layer stack
    layers = composition.create_layer_stack(base_image, placed_landmarks)
    console.print(f"[dim]Created {len(layers)} layers[/dim]")

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        if export_format == "psd":
            output_path = project.output_dir / f"{project.name.replace(' ', '_')}_layered.psd"
        else:
            output_path = project.output_dir / "layers"

    # Export
    psd_service = PSDService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if export_format == "psd":
            task = progress.add_task("Exporting PSD...", total=None)
            psd_service.create_layered_psd(layers, output_path, canvas_size=base_map_size)
            progress.update(task, completed=True, description="[green]PSD exported")
            console.print(f"\n[green]Success![/green] Layered PSD saved to: {output_path}")
        else:
            task = progress.add_task("Exporting layers...", total=None)
            exported = psd_service.export_layers_separately(layers, output_path, canvas_size=base_map_size)
            progress.update(task, completed=True, description="[green]Layers exported")
            console.print(f"\n[green]Success![/green] {len(exported)} layers saved to: {output_path}")

            # Also create manifest
            manifest_path = output_path / "layers.json"
            psd_service.create_layer_manifest(layers, manifest_path)
            console.print(f"[dim]Layer manifest: {manifest_path}[/dim]")

    # Show layer summary
    console.print("\n[bold]Layer structure:[/bold]")
    for name in layers.keys():
        console.print(f"  • {name}")


if __name__ == "__main__":
    main()
