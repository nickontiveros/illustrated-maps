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
    _echo_warnings(document)


def _echo_warnings(document: PlanDocument) -> None:
    for message in document.warnings:
        click.secho(f"  WARNING: {message}", fg="yellow", err=True)


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stub", is_flag=True, help="Use the offline procedural generator (no API cost).")
@click.option("--force", is_flag=True, help="Regenerate even if cached.")
@click.option("--only", "only_ids", multiple=True, help="Regenerate only these asset ids.")
@click.option(
    "--reprocess",
    is_flag=True,
    help="Re-run matting/post-processing from the saved raw generations (no API cost).",
)
def assets(project_dir: str, stub: bool, force: bool, only_ids: tuple[str, ...], reprocess: bool) -> None:
    """Generate all assets in the plan manifest (cached by content hash)."""
    _, directory = _load_project(project_dir)
    document = _load_plan(directory)

    def report(asset_id: str, i: int, total: int) -> None:
        if asset_id != "done":
            click.echo(f"  [{i + 1}/{total}] {asset_id}")

    studio = pipeline.AssetStudio(_make_generator(stub or reprocess), directory / pipeline.ASSETS_DIRNAME)
    if reprocess:
        paths = studio.reprocess_all(document, only_ids=set(only_ids) or None, progress=report)
        click.echo(f"{len(paths)} assets reprocessed from raw in {directory / pipeline.ASSETS_DIRNAME}")
        return

    paths = studio.generate_all(
        document, force=force, only_ids=set(only_ids) or None, progress=report
    )
    click.echo(f"{len(paths)} assets ready in {directory / pipeline.ASSETS_DIRNAME}")


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("title")
def retitle(project_dir: str, title: str) -> None:
    """Change the poster title without re-planning (then re-run compose)."""
    _, directory = _load_project(project_dir)
    patched = pipeline.retitle_project(directory, title)
    if patched:
        click.echo(f"Title set to {title!r}; plan patched. Run `mapgen v2 compose` to re-render.")
    else:
        click.echo(f"Title set to {title!r} (no plan yet; it will apply when you plan).")


def _make_mood_pass(harmonize: bool):
    if not harmonize:
        return None
    from .compose.harmonize import GeminiMoodPass

    return GeminiMoodPass()


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--scale", default=1.0, show_default=True, help="Render scale (0.1 = quick preview).")
@click.option("--harmonize", is_flag=True, help="Apply the low-frequency AI mood pass (one Gemini call).")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
def compose(project_dir: str, scale: float, harmonize: bool, output: str | None) -> None:
    """Render the poster from plan.json + generated assets."""
    _, directory = _load_project(project_dir)
    document = _load_plan(directory)
    out = pipeline.compose_poster(
        document,
        directory,
        scale=scale,
        out_path=Path(output) if output else None,
        mood_pass=_make_mood_pass(harmonize),
    )
    click.echo(f"Poster: {out}")


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--scale", default=1.0, show_default=True, help="Final poster scale.")
@click.option("--strength", default=1.0, show_default=True, help="Texture blend strength (single mode).")
@click.option("--tiled", is_flag=True, help="EXPERIMENTAL: window-by-window infill engine; seam-free only with a fine-tuned exact-infill painter (zero-shot models redraw context pixels).")
@click.option("--repaint-scale", default=0.5, show_default=True, help="Tiled mode: AI repaint resolution (fraction of full size).")
@click.option("--max-calls", type=int, default=None, help="Tiled mode: hard budget of model calls; stop (resumably) when reached.")
@click.option("--dry-run", is_flag=True, help="Print the planned call count and cost; spend nothing.")
@click.option("--stub", is_flag=True, help="Identity painter: exercise the machinery without API calls.")
def repaint(project_dir: str, scale: float, strength: float, tiled: bool, repaint_scale: float, max_calls: int | None, dry_run: bool, stub: bool) -> None:
    """AI hand-painted texture pass over the poster base.

    Default (single) mode: ONE whole-base img2img call; only its low/mid
    frequency texture is blended over the native render, so geometry and
    linework stay pixel-exact, no seams are possible, and labels/frame/grain
    composite on top at native resolution. ~$0.13 per poster.
    """
    _, directory = _load_project(project_dir)
    document = _load_plan(directory)

    def report(i: int, total: int, detail: str) -> None:
        click.echo(f"  [{min(i + 1, total)}/{total}] {detail}")

    if not tiled:
        if dry_run:
            click.echo(f"Single texture pass: 1 call (~${pipeline.REPAINT_COST_PER_CALL})")
            return
        if stub:
            from .repaint import IdentityTexturePass

            painter = IdentityTexturePass()
        else:
            from .repaint import GeminiTexturePass

            painter = GeminiTexturePass()
        out = pipeline.texture_poster(
            document, directory, painter, scale=scale, strength=strength, progress=report
        )
        click.echo(f"Poster: {out}")
        return

    if dry_run:
        info = pipeline.plan_repaint(document, directory, repaint_scale=repaint_scale)
        click.echo(
            f"Grid {info['grid']['cols']}x{info['grid']['rows']} at scale {repaint_scale}: "
            f"{info['calls_planned']} calls (~${info['estimated_cost_usd']})"
        )
        return
    if stub:
        from .repaint import IdentityPainter

        painter = IdentityPainter()
    else:
        from .repaint import GeminiPainter

        painter = GeminiPainter()
    out, result = pipeline.repaint_poster(
        document, directory, painter,
        scale=scale, repaint_scale=repaint_scale, max_calls=max_calls, progress=report,
    )
    state = "complete" if result.completed else f"stopped at budget ({result.calls_made} calls); re-run to resume"
    click.echo(f"Poster: {out} ({result.calls_made} calls, {state})")


@v2.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stub", is_flag=True, help="Use the offline procedural generator (no API cost).")
@click.option("--harmonize", is_flag=True, help="Apply the low-frequency AI mood pass (one Gemini call).")
@click.option("--scale", default=1.0, show_default=True)
def generate(project_dir: str, stub: bool, harmonize: bool, scale: float) -> None:
    """Full pipeline: plan -> assets -> compose."""
    project, directory = _load_project(project_dir)
    click.echo(f"[1/3] Planning {project.name}...")
    source = pipeline.fetch_source(project, cache_dir=directory / "cache")
    document = pipeline.build_plan(project, source)
    pipeline.write_plan(document, directory)
    _echo_warnings(document)
    click.echo("[2/3] Generating assets...")
    pipeline.generate_assets(document, directory, _make_generator(stub))
    click.echo("[3/3] Composing poster...")
    out = pipeline.compose_poster(
        document, directory, scale=scale, mood_pass=_make_mood_pass(harmonize and not stub)
    )
    click.echo(f"Done: {out}")
