"""E2E tests for landmark CRUD and illustration."""

import io

import pytest
from PIL import Image

from .conftest import A1_BBOX, API_KEY_HEADERS, start_and_wait

pytestmark = pytest.mark.e2e


def _create_landmark(client, project_name: str, name: str, **kwargs) -> dict:
    """Helper to create a landmark via API."""
    body = {
        "name": name,
        "latitude": 40.5,
        "longitude": -73.5,
        **kwargs,
    }
    resp = client.post(f"/api/projects/{project_name}/landmarks", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestLandmarkCRUD:
    def test_create_landmark(self, stubbed_client, e2e_project_flat):
        data = _create_landmark(stubbed_client, "e2e-flat", "Empire State Building")
        assert data["name"] == "Empire State Building"
        assert data["latitude"] == 40.5

    def test_list_landmarks(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "LM-A")
        _create_landmark(stubbed_client, "e2e-flat", "LM-B")
        _create_landmark(stubbed_client, "e2e-flat", "LM-C")
        resp = stubbed_client.get("/api/projects/e2e-flat/landmarks")
        assert resp.status_code == 200
        assert len(resp.json()) >= 3

    def test_get_landmark_detail(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "Central Park")
        resp = stubbed_client.get("/api/projects/e2e-flat/landmarks/Central Park")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Central Park"

    def test_update_landmark(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "Bridge")
        resp = stubbed_client.put(
            "/api/projects/e2e-flat/landmarks/Bridge",
            json={"scale": 2.5, "z_index": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scale"] == 2.5
        assert data["z_index"] == 10

    def test_delete_landmark(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "ToDelete")
        resp = stubbed_client.delete("/api/projects/e2e-flat/landmarks/ToDelete")
        assert resp.status_code == 200
        # Verify gone
        resp = stubbed_client.get("/api/projects/e2e-flat/landmarks/ToDelete")
        assert resp.status_code == 404


class TestLandmarkPhotos:
    def test_upload_landmark_photo(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "PhotoTest")
        img = Image.new("RGB", (200, 200), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = stubbed_client.post(
            "/api/projects/e2e-flat/landmarks/PhotoTest/upload-photo",
            files={"file": ("photo.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code == 200

        # Verify photo is accessible
        resp = stubbed_client.get("/api/projects/e2e-flat/landmarks/PhotoTest/photo")
        assert resp.status_code == 200

    def test_fetch_wikipedia_photo(self, stubbed_client, e2e_project_flat):
        _create_landmark(stubbed_client, "e2e-flat", "WikiTest",
                         wikipedia_url="https://en.wikipedia.org/wiki/Test")
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/landmarks/WikiTest/fetch-photo",
        )
        assert resp.status_code == 200

        # Verify photo saved
        resp = stubbed_client.get("/api/projects/e2e-flat/landmarks/WikiTest/photo")
        assert resp.status_code == 200


class TestLandmarkIllustration:
    def test_illustrate_landmark(self, stubbed_client, generated_flat_project):
        """Illustrating a landmark requires generated tiles for style reference."""
        name = "e2e-flat"
        _create_landmark(stubbed_client, name, "IllustTest")

        # Upload a photo first
        img = Image.new("RGB", (200, 200), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        stubbed_client.post(
            f"/api/projects/{name}/landmarks/IllustTest/upload-photo",
            files={"file": ("photo.jpg", buf, "image/jpeg")},
        )

        resp = stubbed_client.post(
            f"/api/projects/{name}/landmarks/IllustTest/illustrate",
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200

    def test_illustrate_all(self, stubbed_client, generated_flat_project):
        name = "e2e-flat"
        # Create two landmarks with photos
        for lm_name in ["IllAll-A", "IllAll-B"]:
            _create_landmark(stubbed_client, name, lm_name)
            img = Image.new("RGB", (100, 100), (50, 50, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            stubbed_client.post(
                f"/api/projects/{name}/landmarks/{lm_name}/upload-photo",
                files={"file": ("photo.jpg", buf, "image/jpeg")},
            )

        resp = stubbed_client.post(
            f"/api/projects/{name}/landmarks/illustrate-all",
            headers=API_KEY_HEADERS,
        )
        assert resp.status_code == 200


class TestLandmarkDiscovery:
    def test_discover_landmarks(self, stubbed_client, e2e_project_flat):
        resp = stubbed_client.post(
            "/api/projects/e2e-flat/landmarks/discover",
            json={"min_importance_score": 0.1, "max_landmarks": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "discovered" in data
        assert "landmarks" in data
