"""
Tests for the vision module.

Tests vision model prompting, response parsing, and Ollama integration.
Uses mocks for Ollama API calls to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
import httpx

from sims.vision import (
    DESCRIPTION_PROMPT,
    parse_vision_response,
    build_full_description,
    describe_image,
    describe_image_simple,
    check_ollama_available,
    check_model_available,
    wait_for_ollama,
    OllamaError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaModelError,
)


# Sample responses for testing
SAMPLE_VISION_RESPONSE = """SCENE: A golden retriever running through a grassy field with mountains in the background. The dog appears joyful with its tongue out, captured mid-stride as it bounds across the meadow.

MOOD: Joyful, energetic

TAGS: dog, golden retriever, field, grass, mountains, running, pet, outdoor, nature, sunny

COLORS: golden, green, blue, white, brown

TIME: afternoon"""


SAMPLE_MINIMAL_RESPONSE = """SCENE: A beach at sunset.
MOOD: Peaceful
TAGS: beach, sunset
COLORS: orange, purple
TIME: sunset"""


SAMPLE_MALFORMED_RESPONSE = """This is just a plain text description without the expected format.
The image shows a mountain landscape."""


class TestDescriptionPrompt:
    """Tests for the description prompt."""

    def test_prompt_contains_required_sections(self) -> None:
        """Test that prompt asks for all required sections."""
        assert "SCENE" in DESCRIPTION_PROMPT
        assert "MOOD" in DESCRIPTION_PROMPT
        assert "TAGS" in DESCRIPTION_PROMPT
        assert "COLORS" in DESCRIPTION_PROMPT
        assert "TIME" in DESCRIPTION_PROMPT

    def test_prompt_has_clear_format_instructions(self) -> None:
        """Test that prompt includes format instructions."""
        assert "Respond in this exact format" in DESCRIPTION_PROMPT
        assert "SCENE:" in DESCRIPTION_PROMPT
        assert "MOOD:" in DESCRIPTION_PROMPT


class TestParseVisionResponse:
    """Tests for parse_vision_response function."""

    def test_parse_complete_response(self) -> None:
        """Test parsing a complete well-formatted response."""
        result = parse_vision_response(SAMPLE_VISION_RESPONSE)

        assert "golden retriever" in result["scene"].lower()
        assert "joyful" in result["mood"].lower()
        assert "dog" in result["tags"]
        assert "golden retriever" in result["tags"]
        assert len(result["tags"]) >= 5
        assert "golden" in result["colors"]
        assert "afternoon" in result["time_of_day"]

    def test_parse_minimal_response(self) -> None:
        """Test parsing a minimal response."""
        result = parse_vision_response(SAMPLE_MINIMAL_RESPONSE)

        assert "beach" in result["scene"].lower()
        assert "peaceful" in result["mood"].lower()
        assert "beach" in result["tags"]
        assert "sunset" in result["tags"]
        assert "orange" in result["colors"]
        assert "sunset" in result["time_of_day"]

    def test_parse_malformed_response(self) -> None:
        """Test parsing a response that doesn't follow the format."""
        result = parse_vision_response(SAMPLE_MALFORMED_RESPONSE)

        # Should return empty/default values for missing fields
        assert result["scene"] == ""
        assert result["mood"] == ""
        assert result["tags"] == []

    def test_parse_empty_response(self) -> None:
        """Test parsing empty response."""
        result = parse_vision_response("")

        assert result["scene"] == ""
        assert result["mood"] == ""
        assert result["tags"] == []
        assert result["colors"] == []
        assert result["time_of_day"] == ""

    def test_parse_tags_with_various_separators(self) -> None:
        """Test parsing tags with different separators."""
        response = """SCENE: Test scene
MOOD: Test mood
TAGS: tag1; tag2, tag3
tag4
COLORS: red
TIME: morning"""

        result = parse_vision_response(response)
        assert "tag1" in result["tags"]
        assert "tag2" in result["tags"]
        assert "tag3" in result["tags"]

    def test_parse_case_insensitive_labels(self) -> None:
        """Test that field labels are case-insensitive."""
        response = """scene: A test scene
mood: happy
tags: test
colors: blue
time: day"""

        result = parse_vision_response(response)
        assert result["scene"] != ""
        assert result["mood"] != ""


class TestBuildFullDescription:
    """Tests for build_full_description function."""

    def test_build_complete_description(self) -> None:
        """Test building description with all fields."""
        parsed = {
            "scene": "A dog in a field",
            "mood": "happy",
            "tags": ["dog", "field"],
            "colors": ["green", "brown"],
            "time_of_day": "afternoon",
        }

        result = build_full_description(parsed)

        assert "A dog in a field" in result
        assert "Mood: happy" in result
        assert "Tags: dog, field" in result
        assert "Colors: green, brown" in result
        assert "Time: afternoon" in result

    def test_build_partial_description(self) -> None:
        """Test building description with some fields missing."""
        parsed = {
            "scene": "A beach",
            "mood": "",
            "tags": [],
            "colors": ["blue"],
            "time_of_day": "",
        }

        result = build_full_description(parsed)

        assert "A beach" in result
        assert "Colors: blue" in result
        assert "Mood" not in result  # Empty mood should be excluded


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available function."""

    @pytest.mark.asyncio
    async def test_ollama_available(self) -> None:
        """Test when Ollama is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_ollama_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_ollama_unavailable(self) -> None:
        """Test when Ollama is not available."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_ollama_available()
            assert result is False


class TestCheckModelAvailable:
    """Tests for check_model_available function."""

    @pytest.mark.asyncio
    async def test_model_available(self) -> None:
        """Test when model is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {
            "models": [
                {"name": "qwen3-vl:8b"},
                {"name": "nomic-embed-text"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_model_available("qwen3-vl:8b")
            assert result is True

    @pytest.mark.asyncio
    async def test_model_not_available(self) -> None:
        """Test when model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {
            "models": [{"name": "other-model"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_model_available("qwen3-vl:8b")
            assert result is False


class TestDescribeImage:
    """Tests for describe_image function."""

    @pytest.mark.asyncio
    async def test_describe_image_success(self, fixtures_dir: Path) -> None:
        """Test successful image description."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": SAMPLE_VISION_RESPONSE}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await describe_image(fixtures_dir / "sample.jpg")

            assert "raw_response" in result
            assert "scene" in result
            assert "mood" in result
            assert "tags" in result
            assert "colors" in result
            assert "time_of_day" in result
            assert "full_description" in result

            assert "golden retriever" in result["scene"].lower()
            assert len(result["tags"]) > 0
            assert result["full_description"] != ""

    @pytest.mark.asyncio
    async def test_describe_image_file_not_found(self) -> None:
        """Test error when image file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            await describe_image(Path("/nonexistent/image.jpg"))

    @pytest.mark.asyncio
    async def test_describe_image_model_not_found(self, fixtures_dir: Path) -> None:
        """Test error when model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(OllamaModelError):
                await describe_image(fixtures_dir / "sample.jpg", max_retries=1)

    @pytest.mark.asyncio
    async def test_describe_image_connection_error_retries(self, fixtures_dir: Path) -> None:
        """Test retry behavior on connection errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(OllamaConnectionError):
                await describe_image(fixtures_dir / "sample.jpg", max_retries=2)

            # Should have attempted twice
            assert mock_instance.post.call_count == 2

    @pytest.mark.asyncio
    async def test_describe_image_timeout_retries(self, fixtures_dir: Path) -> None:
        """Test retry behavior on timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(OllamaTimeoutError):
                await describe_image(fixtures_dir / "sample.jpg", max_retries=2)


class TestDescribeImageSimple:
    """Tests for describe_image_simple function."""

    @pytest.mark.asyncio
    async def test_simple_description(self, fixtures_dir: Path) -> None:
        """Test simple description without structured parsing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": "A simple image description."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await describe_image_simple(fixtures_dir / "sample.jpg")

            assert result == "A simple image description."

    @pytest.mark.asyncio
    async def test_simple_description_custom_prompt(self, fixtures_dir: Path) -> None:
        """Test simple description with custom prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": "Custom response."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await describe_image_simple(
                fixtures_dir / "sample.jpg",
                prompt="What colors are in this image?"
            )

            assert result == "Custom response."


class TestWaitForOllama:
    """Tests for wait_for_ollama function."""

    @pytest.mark.asyncio
    async def test_wait_success_immediate(self) -> None:
        """Test wait succeeds immediately when Ollama is available."""
        with patch("sims.vision.check_ollama_available", return_value=True):
            result = await wait_for_ollama(timeout=5)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_timeout(self) -> None:
        """Test wait times out when Ollama not available."""
        with patch("sims.vision.check_ollama_available", return_value=False):
            result = await wait_for_ollama(timeout=0.1, check_interval=0.05)
            assert result is False


class TestModelOverride:
    """Tests for runtime model override functionality."""

    def test_config_model_override(self) -> None:
        """Test that model override works via Config."""
        from sims.config import Config

        original = Config.get_vision_model()
        Config.set_vision_model("test-model:latest")

        assert Config.get_vision_model() == "test-model:latest"

        Config.reset_vision_model()
        assert Config.get_vision_model() == original

    def test_config_model_override_in_as_dict(self) -> None:
        """Test that as_dict uses the overridden model."""
        from sims.config import Config

        Config.set_vision_model("override-model:1b")
        config_dict = Config.as_dict()

        assert config_dict["VISION_MODEL"] == "override-model:1b"

        Config.reset_vision_model()

    @pytest.mark.asyncio
    async def test_describe_image_with_custom_model(self, fixtures_dir: Path) -> None:
        """Test describe_image uses custom model when specified."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": SAMPLE_VISION_RESPONSE}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            await describe_image(
                fixtures_dir / "sample.jpg",
                model="qwen3-vl:4b"
            )

            # Verify the custom model was used in the request
            call_args = mock_instance.post.call_args
            request_data = call_args.kwargs.get('json') or call_args[1].get('json')
            assert request_data["model"] == "qwen3-vl:4b"

    @pytest.mark.asyncio
    async def test_describe_image_uses_config_default(self, fixtures_dir: Path) -> None:
        """Test describe_image uses config default when no model specified."""
        from sims.config import Config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": SAMPLE_VISION_RESPONSE}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            await describe_image(fixtures_dir / "sample.jpg")

            # Verify the default model was used
            call_args = mock_instance.post.call_args
            request_data = call_args.kwargs.get('json') or call_args[1].get('json')
            assert request_data["model"] == Config.get_vision_model()

    @pytest.mark.asyncio
    async def test_describe_image_simple_with_custom_model(self, fixtures_dir: Path) -> None:
        """Test describe_image_simple uses custom model when specified."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"response": "A simple description."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            await describe_image_simple(
                fixtures_dir / "sample.jpg",
                model="qwen3-vl:4b"
            )

            # Verify the custom model was used in the request
            call_args = mock_instance.post.call_args
            request_data = call_args.kwargs.get('json') or call_args[1].get('json')
            assert request_data["model"] == "qwen3-vl:4b"
