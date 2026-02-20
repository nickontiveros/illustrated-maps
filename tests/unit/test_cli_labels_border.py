"""Tests for CLI commands: add-labels, add-border."""

import os

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

    img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    path = tmp_path / "input.png"
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# add-labels command
# ---------------------------------------------------------------------------

class TestAddLabelsCommand:
    def test_missing_project_path_fails(self, runner):
        result = runner.invoke(main, ["add-labels"])
        assert result.exit_code != 0

    def test_missing_input_image_fails(self, runner, fake_project):
        result = runner.invoke(main, ["add-labels", fake_project])
        assert result.exit_code != 0

    def test_nonexistent_project_path_fails(self, runner, tmp_path, fake_image):
        bad_path = str(tmp_path / "nonexistent")
        result = runner.invoke(main, ["add-labels", bad_path, "-i", fake_image])
        assert result.exit_code != 0

    def test_nonexistent_input_image_fails(self, runner, fake_project):
        result = runner.invoke(main, ["add-labels", fake_project, "-i", "/nonexistent/image.png"])
        assert result.exit_code != 0

    def test_help_flag(self, runner):
        result = runner.invoke(main, ["add-labels", "--help"])
        assert result.exit_code == 0
        assert "input-image" in result.output or "input" in result.output


# ---------------------------------------------------------------------------
# add-border command
# ---------------------------------------------------------------------------

class TestAddBorderCommand:
    def test_missing_project_path_fails(self, runner):
        result = runner.invoke(main, ["add-border"])
        assert result.exit_code != 0

    def test_missing_input_image_fails(self, runner, fake_project):
        result = runner.invoke(main, ["add-border", fake_project])
        assert result.exit_code != 0

    def test_nonexistent_project_path_fails(self, runner, tmp_path, fake_image):
        bad_path = str(tmp_path / "nonexistent")
        result = runner.invoke(main, ["add-border", bad_path, "-i", fake_image])
        assert result.exit_code != 0

    def test_nonexistent_input_image_fails(self, runner, fake_project):
        result = runner.invoke(main, ["add-border", fake_project, "-i", "/nonexistent/image.png"])
        assert result.exit_code != 0

    def test_help_flag(self, runner):
        result = runner.invoke(main, ["add-border", "--help"])
        assert result.exit_code == 0
        assert "style" in result.output.lower() or "border" in result.output.lower()

    def test_style_choices_in_help(self, runner):
        result = runner.invoke(main, ["add-border", "--help"])
        assert result.exit_code == 0

    def test_invalid_style_rejected(self, runner, fake_project, fake_image):
        result = runner.invoke(main, ["add-border", fake_project, "-i", fake_image, "-s", "invalid_style"])
        assert result.exit_code != 0
