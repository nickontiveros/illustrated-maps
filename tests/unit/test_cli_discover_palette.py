"""Tests for CLI commands: discover-landmarks, palette (list-presets, extract)."""

import pytest
from click.testing import CliRunner

from mapgen.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_project(tmp_path):
    """Create a minimal project directory with project.yaml."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    yaml_content = """
name: test-project
region:
  north: 40.775
  south: 40.768
  east: -73.968
  west: -73.978
output:
  width: 1024
  height: 1024
  dpi: 72
tiles:
  size: 512
  overlap: 64
"""
    (project_dir / "project.yaml").write_text(yaml_content)
    return str(project_dir)


@pytest.fixture
def fake_image(tmp_path):
    """Create a minimal PNG image file."""
    from PIL import Image

    img = Image.new("RGBA", (64, 64), (128, 64, 200, 255))
    path = tmp_path / "sample.png"
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# discover-landmarks command
# ---------------------------------------------------------------------------

class TestDiscoverLandmarksCommand:
    def test_missing_project_path_fails(self, runner):
        result = runner.invoke(main, ["discover-landmarks"])
        assert result.exit_code != 0

    def test_nonexistent_project_path_fails(self, runner, tmp_path):
        bad_path = str(tmp_path / "nonexistent")
        result = runner.invoke(main, ["discover-landmarks", bad_path])
        assert result.exit_code != 0

    def test_help_flag(self, runner):
        result = runner.invoke(main, ["discover-landmarks", "--help"])
        assert result.exit_code == 0
        assert "min-score" in result.output or "max-landmarks" in result.output or "save" in result.output


# ---------------------------------------------------------------------------
# palette group
# ---------------------------------------------------------------------------

class TestPaletteGroup:
    def test_palette_help(self, runner):
        result = runner.invoke(main, ["palette", "--help"])
        assert result.exit_code == 0

    def test_palette_no_subcommand(self, runner):
        result = runner.invoke(main, ["palette"])
        # Should show help or list subcommands, not crash
        assert result.exit_code in (0, 2)  # 2 is Click's "missing subcommand" exit code


# ---------------------------------------------------------------------------
# palette list-presets
# ---------------------------------------------------------------------------

class TestPaletteListPresets:
    def test_list_presets_succeeds(self, runner):
        result = runner.invoke(main, ["palette", "list-presets"])
        assert result.exit_code == 0

    def test_list_presets_shows_presets(self, runner):
        result = runner.invoke(main, ["palette", "list-presets"])
        # Should mention at least one preset name
        output = result.output.lower()
        assert "vintage" in output or "modern" in output or "ink" in output


# ---------------------------------------------------------------------------
# palette extract
# ---------------------------------------------------------------------------

class TestPaletteExtract:
    def test_missing_image_path_fails(self, runner):
        result = runner.invoke(main, ["palette", "extract"])
        assert result.exit_code != 0

    def test_nonexistent_image_fails(self, runner):
        result = runner.invoke(main, ["palette", "extract", "/nonexistent/image.png"])
        assert result.exit_code != 0

    def test_extract_help(self, runner):
        result = runner.invoke(main, ["palette", "extract", "--help"])
        assert result.exit_code == 0
        assert "n-colors" in result.output or "colors" in result.output.lower()
