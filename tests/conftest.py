"""
Pytest fixtures for SImS tests.

Provides common fixtures for temporary directories, mock data, and test configuration.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="sims_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_db_path(temp_data_dir: Path) -> Path:
    """Return path for a temporary SQLite database."""
    return temp_data_dir / "test_sims.db"


@pytest.fixture
def temp_chroma_path(temp_data_dir: Path) -> Path:
    """Return path for a temporary ChromaDB directory."""
    chroma_path = temp_data_dir / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


@pytest.fixture
def sample_image_dir(temp_dir: Path) -> Path:
    """Create a directory structure with sample test images."""
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for testing recursive traversal
    subdir = images_dir / "subdir"
    subdir.mkdir(parents=True, exist_ok=True)

    return images_dir


@pytest.fixture
def mock_config(
    temp_cache_dir: Path,
    temp_data_dir: Path,
    temp_db_path: Path,
    temp_chroma_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set environment variables to use temporary directories for testing."""
    monkeypatch.setenv("SIMS_CACHE_DIR", str(temp_cache_dir))
    monkeypatch.setenv("SIMS_DATA_DIR", str(temp_data_dir))
    monkeypatch.setenv("SIMS_DB_PATH", str(temp_db_path))
    monkeypatch.setenv("SIMS_CHROMA_PATH", str(temp_chroma_path))


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# Mock response fixtures for Ollama
@pytest.fixture
def mock_vision_response() -> str:
    """Sample vision model response for testing."""
    return """SCENE: A golden retriever running through a grassy field with mountains in the background. The dog appears joyful with its tongue out.

MOOD: Joyful, energetic

TAGS: dog, golden retriever, field, grass, mountains, running, pet, outdoor, nature, sunny

COLORS: Golden, green, blue, white

TIME OF DAY: Afternoon"""


@pytest.fixture
def mock_embedding() -> list[float]:
    """Sample 768-dimensional embedding for testing."""
    import random

    random.seed(42)  # Reproducible
    return [random.uniform(-1, 1) for _ in range(768)]
