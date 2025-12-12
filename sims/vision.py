# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Ollama vision model interface for SImS.

Provides image description generation using qwen3-vl via the Ollama API.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from sims.config import Config

logger = logging.getLogger(__name__)


# Vision model prompt for structured image descriptions
DESCRIPTION_PROMPT = """Analyze this image and provide a structured description.

Describe:
1. SCENE: What is shown in the image? Be specific about subjects, setting, and composition.
2. MOOD: What is the overall mood or atmosphere? (e.g., peaceful, energetic, melancholic, joyful)
3. TAGS: List 5-10 descriptive tags as keywords.
4. COLORS: Dominant color palette.
5. TIME OF DAY: If discernible (morning, afternoon, sunset, night, unclear).

Respond in this exact format:
SCENE: [description]
MOOD: [mood]
TAGS: [tag1, tag2, tag3, ...]
COLORS: [color1, color2, ...]
TIME: [time of day]"""


class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to Ollama."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request times out."""
    pass


class OllamaModelError(OllamaError):
    """Raised when model is not available."""
    pass


async def check_ollama_available(timeout: float = 5.0) -> bool:
    """
    Check if Ollama is running and accessible.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        True if Ollama is accessible.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{Config.OLLAMA_HOST}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


async def check_model_available(model: str, timeout: float = 5.0) -> bool:
    """
    Check if a specific model is available in Ollama.

    Args:
        model: Model name (e.g., "qwen3-vl:4b").
        timeout: Request timeout in seconds.

    Returns:
        True if the model is available.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{Config.OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return model in models or any(model.split(":")[0] in m for m in models)
    except Exception:
        return False


async def wait_for_ollama(timeout: int = 60, check_interval: float = 2.0) -> bool:
    """
    Wait for Ollama to become available.

    Args:
        timeout: Maximum time to wait in seconds.
        check_interval: Time between checks in seconds.

    Returns:
        True if Ollama became available within timeout.
    """
    elapsed = 0.0
    while elapsed < timeout:
        if await check_ollama_available():
            logger.info("Ollama is available")
            return True

        logger.debug(f"Waiting for Ollama... ({elapsed:.1f}s)")
        await asyncio.sleep(check_interval)
        elapsed += check_interval

    logger.warning(f"Ollama not available after {timeout}s")
    return False


def parse_vision_response(response: str) -> dict:
    """
    Parse the structured response from the vision model.

    Args:
        response: Raw text response from the model.

    Returns:
        Dictionary with parsed fields:
            - scene: Scene description
            - mood: Mood/atmosphere
            - tags: List of tags
            - colors: List of colors
            - time_of_day: Time of day
    """
    result = {
        "scene": "",
        "mood": "",
        "tags": [],
        "colors": [],
        "time_of_day": "",
    }

    if not response:
        return result

    # Normalize line endings and clean up
    text = response.replace('\r\n', '\n').strip()

    # Parse each field using regex
    # SCENE: can span multiple lines until the next field
    scene_match = re.search(
        r'SCENE:\s*(.+?)(?=\n(?:MOOD|TAGS|COLORS|TIME):|$)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if scene_match:
        result["scene"] = scene_match.group(1).strip()

    # MOOD: single line typically
    mood_match = re.search(r'MOOD:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
    if mood_match:
        result["mood"] = mood_match.group(1).strip()

    # TAGS: comma-separated list
    tags_match = re.search(r'TAGS:\s*(.+?)(?=\n(?:COLORS|TIME):|$)', text, re.IGNORECASE | re.DOTALL)
    if tags_match:
        tags_str = tags_match.group(1).strip()
        # Handle various separators: comma, semicolon, newline
        tags = re.split(r'[,;\n]+', tags_str)
        result["tags"] = [t.strip().lower() for t in tags if t.strip()]

    # COLORS: comma-separated list
    colors_match = re.search(r'COLORS:\s*(.+?)(?=\n(?:TIME):|$)', text, re.IGNORECASE | re.DOTALL)
    if colors_match:
        colors_str = colors_match.group(1).strip()
        colors = re.split(r'[,;\n]+', colors_str)
        result["colors"] = [c.strip().lower() for c in colors if c.strip()]

    # TIME: single value
    time_match = re.search(r'TIME(?:\s+OF\s+DAY)?:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
    if time_match:
        result["time_of_day"] = time_match.group(1).strip().lower()

    return result


def build_full_description(parsed: dict) -> str:
    """
    Build a full description string for embedding from parsed response.

    Args:
        parsed: Parsed vision response dictionary.

    Returns:
        Concatenated description suitable for embedding.
    """
    parts = []

    if parsed.get("scene"):
        parts.append(parsed["scene"])

    if parsed.get("mood"):
        parts.append(f"Mood: {parsed['mood']}")

    if parsed.get("tags"):
        parts.append(f"Tags: {', '.join(parsed['tags'])}")

    if parsed.get("colors"):
        parts.append(f"Colors: {', '.join(parsed['colors'])}")

    if parsed.get("time_of_day"):
        parts.append(f"Time: {parsed['time_of_day']}")

    return ". ".join(parts)


async def describe_image(
    image_path: Path,
    max_retries: int = None,
    timeout: float = None,
    model: Optional[str] = None,
) -> dict:
    """
    Generate a structured description of an image using the vision model.

    Args:
        image_path: Path to the image file.
        max_retries: Maximum retry attempts (default from config).
        timeout: Request timeout in seconds (default from config).
        model: Vision model to use (default from config).

    Returns:
        Dictionary containing:
            - raw_response: Original model response
            - scene: Scene description
            - mood: Mood/atmosphere
            - tags: List of tags
            - colors: List of colors
            - time_of_day: Time of day
            - full_description: Concatenated description for embedding

    Raises:
        FileNotFoundError: If image file doesn't exist.
        OllamaConnectionError: If unable to connect to Ollama.
        OllamaTimeoutError: If request times out after retries.
        OllamaModelError: If model is not available.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if max_retries is None:
        max_retries = Config.MAX_RETRIES

    if timeout is None:
        timeout = Config.OLLAMA_TIMEOUT

    # Determine which model to use
    vision_model = model or Config.get_vision_model()

    # Read and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Prepare request
    request_data = {
        "model": vision_model,
        "prompt": DESCRIPTION_PROMPT,
        "images": [image_b64],
        "stream": False,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Describing image {image_path.name} "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{Config.OLLAMA_HOST}/api/generate",
                    json=request_data,
                )

                if response.status_code == 404:
                    raise OllamaModelError(
                        f"Model {vision_model} not found. "
                        f"Run: ollama pull {vision_model}"
                    )

                response.raise_for_status()
                data = response.json()

            raw_response = data.get("response", "")
            parsed = parse_vision_response(raw_response)
            full_description = build_full_description(parsed)

            logger.debug(f"Successfully described {image_path.name}")

            return {
                "raw_response": raw_response,
                "scene": parsed["scene"],
                "mood": parsed["mood"],
                "tags": parsed["tags"],
                "colors": parsed["colors"],
                "time_of_day": parsed["time_of_day"],
                "full_description": full_description,
            }

        except httpx.ConnectError as e:
            last_error = OllamaConnectionError(
                f"Cannot connect to Ollama at {Config.OLLAMA_HOST}: {e}"
            )
            logger.warning(f"Connection error (attempt {attempt + 1}): {e}")

        except httpx.TimeoutException as e:
            last_error = OllamaTimeoutError(
                f"Request timed out after {timeout}s: {e}"
            )
            logger.warning(f"Timeout (attempt {attempt + 1}): {e}")

        except OllamaModelError:
            raise

        except httpx.HTTPStatusError as e:
            last_error = OllamaError(f"HTTP error: {e}")
            logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")

        except Exception as e:
            last_error = OllamaError(f"Unexpected error: {e}")
            logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")

        # Exponential backoff before retry
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logger.debug(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    # All retries exhausted
    raise last_error or OllamaError("Failed to describe image after retries")


async def describe_image_simple(
    image_path: Path,
    prompt: str = "Describe this image in detail.",
    timeout: float = None,
    model: Optional[str] = None,
) -> str:
    """
    Get a simple text description of an image without structured parsing.

    Args:
        image_path: Path to the image file.
        prompt: Custom prompt for the model.
        timeout: Request timeout in seconds.
        model: Vision model to use (default from config).

    Returns:
        Raw text response from the model.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if timeout is None:
        timeout = Config.OLLAMA_TIMEOUT

    # Determine which model to use
    vision_model = model or Config.get_vision_model()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{Config.OLLAMA_HOST}/api/generate",
            json={
                "model": vision_model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")
