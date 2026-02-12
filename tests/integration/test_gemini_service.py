"""Integration tests for GeminiService with mocked API client."""

import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mapgen.services.gemini_service import GeminiService, GenerationResult


def _make_fake_response(width=256, height=256):
    """Create a fake Gemini API response with an image."""
    img = Image.new("RGBA", (width, height), (100, 150, 200, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    inline_data = MagicMock()
    inline_data.data = image_bytes

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


@pytest.fixture
def mock_genai_types():
    """Mock google.genai.types to avoid import errors in restricted environments."""
    mock_types = MagicMock()
    mock_types.GenerateContentConfig = MagicMock
    # Insert into sys.modules so `from google.genai import types` finds it
    with patch.dict(sys.modules, {"google.genai": MagicMock(types=mock_types)}):
        yield mock_types


@pytest.fixture
def gemini_service():
    """GeminiService with a fake API key."""
    return GeminiService(api_key="fake-key-for-testing")


class TestGeminiServiceInit:
    """Test service initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                GeminiService(api_key=None)

    def test_accepts_explicit_key(self):
        svc = GeminiService(api_key="test-key")
        assert svc.api_key == "test-key"

    def test_lazy_client(self, gemini_service):
        assert gemini_service._client is None


class TestEstimateCost:
    """Test cost estimation (no API needed)."""

    def test_basic_estimate(self, gemini_service):
        cost = gemini_service.estimate_cost(10, 5)
        assert cost["num_tiles"] == 10
        assert cost["num_landmarks"] == 5
        assert cost["total_cost"] > 0

    def test_total_is_sum(self, gemini_service):
        cost = gemini_service.estimate_cost(10, 5)
        expected = cost["tile_cost"] + cost["landmark_cost"] + cost["seam_cost"]
        assert cost["total_cost"] == pytest.approx(expected)

    def test_zero_inputs(self, gemini_service):
        cost = gemini_service.estimate_cost(0, 0)
        assert cost["total_cost"] == 0.0


class TestGenerateBaseTile:
    """Test base tile generation with mocked client."""

    def test_calls_api(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        ref_image = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
        result = gemini_service.generate_base_tile(ref_image)

        assert isinstance(result, GenerationResult)
        assert result.image is not None
        assert result.image.mode == "RGBA"
        mock_client.models.generate_content.assert_called_once()

    def test_includes_position_in_prompt(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        ref_image = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
        gemini_service.generate_base_tile(ref_image, tile_position="top-left corner")

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        prompt = contents[-1]
        assert "top-left corner" in prompt

    def test_includes_style_reference(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        ref = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
        style = Image.new("RGBA", (256, 256), (200, 100, 50, 255))
        gemini_service.generate_base_tile(ref, style_reference=style)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        # Should have ref image, style image, and prompt
        assert len(contents) == 3

    def test_resizes_large_input(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        large_image = Image.new("RGBA", (4096, 4096), (128, 128, 128, 255))
        gemini_service.generate_base_tile(large_image)

        mock_client.models.generate_content.assert_called_once()


class TestStylizeLandmark:
    """Test landmark stylization with mocked client."""

    def test_calls_api(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        photo = Image.new("RGBA", (256, 256), (128, 128, 128, 255))
        result = gemini_service.stylize_landmark(photo, landmark_name="Empire State Building")

        assert isinstance(result, GenerationResult)
        assert result.image is not None
        mock_client.models.generate_content.assert_called_once()

    def test_includes_landmark_name(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response()
        gemini_service._client = mock_client

        photo = Image.new("RGBA", (256, 256), (128, 128, 128, 255))
        gemini_service.stylize_landmark(photo, landmark_name="Statue of Liberty")

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        prompt = contents[-1]
        assert "Statue of Liberty" in prompt


class TestInpaintSeam:
    """Test seam inpainting with mocked client."""

    def test_calls_api(self, gemini_service, mock_genai_types):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_fake_response(256, 64)
        gemini_service._client = mock_client

        seam_region = Image.new("RGBA", (256, 64), (128, 128, 128, 255))
        result = gemini_service.inpaint_seam(seam_region, orientation="horizontal")

        assert isinstance(result, GenerationResult)
        assert result.image is not None


class TestExtractImage:
    """Test image extraction from various response formats."""

    def test_extracts_from_candidates(self, gemini_service):
        response = _make_fake_response()
        img = gemini_service._extract_image_from_response(response)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"

    def test_raises_on_empty_response(self, gemini_service):
        response = MagicMock()
        response.candidates = []
        response.images = None
        # Make parts behave as a non-iterable to trigger the ValueError
        del response.parts
        with pytest.raises((ValueError, AttributeError)):
            gemini_service._extract_image_from_response(response)
