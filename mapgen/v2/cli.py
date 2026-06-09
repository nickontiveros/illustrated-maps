"""V2 CLI: plan -> assets -> compose, each independently runnable.

    mapgen v2 plan PROJECT_DIR        # plan.json + preview.svg (free)
    mapgen v2 assets PROJECT_DIR      # AI asset generation (cached)
    mapgen v2 compose PROJECT_DIR     # render the poster
    mapgen v2 generate PROJECT_DIR    # all three

PROJECT_DIR must contain project.yaml (see V2Project). Use --stub to run
the asset stage with the offline procedural generator (no API key, no
cost) -- useful for layout review and CI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from . import pipeline
from .types import PlanDocument

logger = logging.getLogger(__name__)


@click.group(name="v2")
def v2() -> None:
    """V2 asset-composition map generator (see V2_DESIGN.md)."""


def _load_project(project_dir: str) -> tuple[pipeline.V2Project, Path]:
    directory = Path(project_dir)
    config = directory / "project.yaml"
    if not config.exists():
        raise click.ClickException(f"No project.yaml in {directory}")
    return pipeline.V2Project.load(config), directory


def _load_plan(directory: Path) -> PlanDocument:
    plan_path = directory / pipeline.PLAN_FILENAME
    if not plan_path.exists():
        raise click.ClickException(f"No {pipeline.PLAN_FILENAME} in {directory}; run `mapgen v2 plan` first")
    return PlanDocument.load(plan_path)


def _make_generator(stub: bool):
    if stub:
        from .assets.stub import StubAssetGenerator

        return StubAssetGenerator()
    from .assets.gemini_client import GeminiAssetGenerator

    return GeminiAssetGenerator()


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
def plan(project_dir: str) -> None:
    """Build plan.json and preview.svg from live OSM data (free, no AI)."""
    project, directory = _load_project(project_dir)
    click.echo(f"Fetching OSM data for {project.name}...")
    source = pipeline.fetch_source(project, cache_dir=directory / "cache")
    document = pipeline.build_plan(project, source)
    plan_path, preview_path = pipeline.write_plan(document, directory)
    click.echo(f"Plan: {plan_path}")
    click.echo(f"Preview: {preview_path}")
    click.echo(
        f"  {len(document.roads)} roads, {len(document.ground)} ground polygons, "
        f"{len(document.pois)} POIs, {len(document.labels)} labels, "
        f"{len(document.manifest)} assets to generate"
    )


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stub", is_flag=True, help="Use the offline procedural generator (no API cost).")
@click.option("--force", is_flag=True, help="Regenerate even if cached.")
@click.option("--only", "only_ids", multiple=True, help="Regenerate only these asset ids.")
def assets(project_dir: str, stub: bool, force: bool, only_ids: tuple[str, ...]) -> None:
    """Generate all assets in the plan manifest (cached by content hash)."""
    _, directory = _load_project(project_dir)
    document = _load_plan(directory)
    generator = _make_generator(stub)

    def report(asset_id: str, i: int, total: int) -> None:
        if asset_id != "done":
            click.echo(f"  [{i + 1}/{total}] {asset_id}")

    studio = pipeline.AssetStudio(generator, directory / pipeline.ASSETS_DIRNAME)
    paths = studio.generate_all(
        document, force=force, only_ids=set(only_ids) or None, progress=report
    )
    click.echo(f"{len(paths)} assets ready in {directory / pipeline.ASSETS_DIRNAME}")


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--scale", default=1.0, show_default=True, help="Render scale (0.1 = quick preview).")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
def compose(project_dir: str, scale: float, output: str | None) -> None:
    """Render the poster from plan.json + generated assets."""
    _, directory = _load_project(project_dir)
    document = _load_plan(directory)
    out = pipeline.compose_poster(
        document, directory, scale=scale, out_path=Path(output) if output else None
    )
    click.echo(f"Poster: {out}")


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stub", is_flag=True, help="Use the offline procedural generator (no API cost).")
@click.option("--scale", default=1.0, show_default=True)
def generate(project_dir: str, stub: bool, scale: float) -> None:
    """Full pipeline: plan -> assets -> compose."""
    project, directory = _load_project(project_dir)
    click.echo(f"[1/3] Planning {project.name}...")
    source = pipeline.fetch_source(project, cache_dir=directory / "cache")
    document = pipeline.build_plan(project, source)
    pipeline.write_plan(document, directory)
    click.echo("[2/3] Generating assets...")
    pipeline.generate_assets(document, directory, _make_generator(stub))
    click.echo("[3/3] Composing poster...")
    out = pipeline.compose_poster(document, directory, scale=scale)
    click.echo(f"Done: {out}")
