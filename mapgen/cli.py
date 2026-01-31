"""Command-line interface for map generator."""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_config
from .models.project import BoundingBox, Project


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
        output_path = output_dir / f"{project.name.replace(' ', '_')}_base_map.png"
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

    table = Table(title=f"Project: {project.name}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Region North", f"{project.region.north:.6f}")
    table.add_row("Region South", f"{project.region.south:.6f}")
    table.add_row("Region East", f"{project.region.east:.6f}")
    table.add_row("Region West", f"{project.region.west:.6f}")
    table.add_row("Output Width", f"{project.output.width} px")
    table.add_row("Output Height", f"{project.output.height} px")
    table.add_row("Output DPI", str(project.output.dpi))
    table.add_row("Tile Size", f"{project.tiles.size} px")
    table.add_row("Tile Overlap", f"{project.tiles.overlap} px")

    cols, rows = project.tiles.calculate_grid(project.output.width, project.output.height)
    table.add_row("Tile Grid", f"{cols} x {rows} = {cols * rows} tiles")

    console.print(table)

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

    preview_path = project.output_dir / "osm_preview.png"
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


if __name__ == "__main__":
    main()
