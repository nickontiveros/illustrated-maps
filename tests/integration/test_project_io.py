"""Integration tests for Project YAML I/O."""

import pytest
import yaml

from mapgen.models.landmark import FeatureType, Landmark
from mapgen.models.project import (
    BoundingBox,
    CardinalDirection,
    OutputSettings,
    Project,
    StyleSettings,
    TileSettings,
)


class TestProjectYamlRoundtrip:
    """Test saving and loading projects from YAML."""

    def test_basic_roundtrip(self, tmp_path, sample_bbox):
        project = Project(
            name="roundtrip-test",
            region=sample_bbox,
            output=OutputSettings(width=2048, height=2048, dpi=150),
            tiles=TileSettings(size=1024, overlap=128),
        )

        yaml_path = tmp_path / "project.yaml"
        project.to_yaml(yaml_path)

        loaded = Project.from_yaml(yaml_path)
        assert loaded.name == "roundtrip-test"
        assert loaded.region.north == sample_bbox.north
        assert loaded.output.width == 2048
        assert loaded.tiles.size == 1024
        assert loaded.project_dir == tmp_path

    def test_roundtrip_with_landmarks(self, tmp_path, sample_bbox):
        landmarks = [
            Landmark(name="Building A", latitude=40.77, longitude=-73.97, scale=2.0),
            Landmark(name="Park B", latitude=40.77, longitude=-73.975, feature_type=FeatureType.PARK),
        ]
        project = Project(
            name="landmark-test",
            region=sample_bbox,
            landmarks=landmarks,
        )

        yaml_path = tmp_path / "project.yaml"
        project.to_yaml(yaml_path)

        loaded = Project.from_yaml(yaml_path)
        assert len(loaded.landmarks) == 2
        assert loaded.landmarks[0].name == "Building A"
        assert loaded.landmarks[1].feature_type == FeatureType.PARK

    def test_roundtrip_with_style(self, tmp_path, sample_bbox):
        project = Project(
            name="style-test",
            region=sample_bbox,
            style=StyleSettings(
                perspective_angle=45.0,
                orientation=CardinalDirection.EAST,
                prompt="Custom prompt",
            ),
        )

        yaml_path = tmp_path / "project.yaml"
        project.to_yaml(yaml_path)

        loaded = Project.from_yaml(yaml_path)
        assert loaded.style.perspective_angle == 45.0
        assert loaded.style.orientation == CardinalDirection.EAST
        assert loaded.style.prompt == "Custom prompt"

    def test_roundtrip_orientation_degrees(self, tmp_path, sample_bbox):
        project = Project(
            name="orient-test",
            region=sample_bbox,
            style=StyleSettings(orientation_degrees=135.0),
        )

        yaml_path = tmp_path / "project.yaml"
        project.to_yaml(yaml_path)

        loaded = Project.from_yaml(yaml_path)
        assert loaded.style.orientation_degrees == 135.0
        assert loaded.style.effective_rotation_degrees == 135.0


class TestProjectYamlErrors:
    """Test error handling for YAML I/O."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Project.from_yaml(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("name: test\nregion: not-a-dict\n")
        with pytest.raises(Exception):
            Project.from_yaml(bad_path)

    def test_missing_required_fields_raises(self, tmp_path):
        incomplete_path = tmp_path / "incomplete.yaml"
        incomplete_path.write_text("name: test\n")
        with pytest.raises(Exception):
            Project.from_yaml(incomplete_path)


class TestEnsureDirectories:
    """Test directory creation."""

    def test_creates_all_dirs(self, sample_project):
        sample_project.ensure_directories()
        assert (sample_project.project_dir / "landmarks").exists()
        assert (sample_project.project_dir / "logos").exists()
        assert (sample_project.project_dir / "output").exists()

    def test_idempotent(self, sample_project):
        sample_project.ensure_directories()
        sample_project.ensure_directories()
        assert sample_project.landmarks_dir.exists()

    def test_raises_without_project_dir(self, sample_bbox):
        project = Project(name="test", region=sample_bbox)
        with pytest.raises(ValueError):
            project.ensure_directories()
