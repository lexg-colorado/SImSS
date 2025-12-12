"""
Tests for the agent module.

Tests ingestion orchestration, pipeline stages, and CLI functionality.
Uses mocks for Ollama API calls to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from sims import db
from sims import vectorstore
from sims.agent import (
    IngestionAgent,
    ProcessingStats,
    JobStatus,
    search_images,
    get_system_stats,
)
from sims.config import Config


def generate_mock_embedding(seed: int = 42) -> list[float]:
    """Generate a deterministic mock embedding for testing."""
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(768)]


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory(prefix="sims_agent_test_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        images_dir = tmpdir / "images"
        images_dir.mkdir()

        # Create a simple test image
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img.save(images_dir / f"test_{i}.jpg", "JPEG")

        # Create subdirectory with more images
        subdir = images_dir / "subdir"
        subdir.mkdir()
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        img.save(subdir / "sub_test.jpg", "JPEG")

        yield tmpdir


@pytest.fixture
def isolated_environment(
    temp_test_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path, None, None]:
    """Set up isolated environment for agent testing."""
    # Create data directories
    data_dir = temp_test_dir / "data"
    data_dir.mkdir()
    cache_dir = temp_test_dir / "cache"
    cache_dir.mkdir()
    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir()
    db_path = data_dir / "test.db"

    # Set environment
    monkeypatch.setenv("SIMS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SIMS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("SIMS_DB_PATH", str(db_path))
    monkeypatch.setenv("SIMS_CHROMA_PATH", str(chroma_dir))

    # Update config
    monkeypatch.setattr(Config, "DATA_DIR", data_dir)
    monkeypatch.setattr(Config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    monkeypatch.setattr(Config, "CHROMA_PATH", chroma_dir)

    # Clear vectorstore cache
    vectorstore.close()
    vectorstore._client = None
    vectorstore._collection = None

    yield temp_test_dir

    # Cleanup
    vectorstore.close()
    vectorstore._client = None
    vectorstore._collection = None


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_initial_values(self) -> None:
        """Test default values."""
        stats = ProcessingStats()
        assert stats.total == 0
        assert stats.processed == 0
        assert stats.failed == 0
        assert stats.skipped == 0

    def test_pending_calculation(self) -> None:
        """Test pending property calculation."""
        stats = ProcessingStats(total=100, processed=30, failed=10, skipped=20)
        assert stats.pending == 40

    def test_all_complete(self) -> None:
        """Test when all items are complete."""
        stats = ProcessingStats(total=100, processed=90, failed=10, skipped=0)
        assert stats.pending == 0


class TestJobStatus:
    """Tests for JobStatus dataclass."""

    def test_progress_percent_normal(self) -> None:
        """Test progress percentage calculation."""
        status = JobStatus(
            job_id=1,
            root_path="/test",
            status="running",
            total_files=100,
            processed_files=50,
            failed_files=10,
        )
        assert status.progress_percent == 60.0

    def test_progress_percent_zero_total(self) -> None:
        """Test progress with zero total files."""
        status = JobStatus(
            job_id=1,
            root_path="/test",
            status="completed",
            total_files=0,
            processed_files=0,
            failed_files=0,
        )
        assert status.progress_percent == 0.0

    def test_progress_percent_complete(self) -> None:
        """Test progress at 100%."""
        status = JobStatus(
            job_id=1,
            root_path="/test",
            status="completed",
            total_files=100,
            processed_files=95,
            failed_files=5,
        )
        assert status.progress_percent == 100.0


class TestIngestionAgentInit:
    """Tests for IngestionAgent initialization."""

    def test_default_batch_size(self, isolated_environment: Path) -> None:
        """Test default batch size from config."""
        agent = IngestionAgent()
        assert agent.batch_size == Config.BATCH_SIZE

    def test_custom_batch_size(self, isolated_environment: Path) -> None:
        """Test custom batch size."""
        agent = IngestionAgent(batch_size=5)
        assert agent.batch_size == 5

    def test_with_progress_callback(self, isolated_environment: Path) -> None:
        """Test with progress callback."""
        callback = MagicMock()
        agent = IngestionAgent(on_progress=callback)
        assert agent.on_progress is callback


class TestDiscoveryStage:
    """Tests for image discovery stage."""

    @pytest.mark.asyncio
    async def test_discover_new_images(self, isolated_environment: Path) -> None:
        """Test discovering new images."""
        images_dir = isolated_environment / "images"

        # Initialize DB
        db.init_db()

        agent = IngestionAgent()
        stats = ProcessingStats(total=4)

        # Run discovery
        discovered = await agent._discover_stage(images_dir, recursive=True, stats=stats)

        assert len(discovered) == 4  # 3 in root + 1 in subdir
        assert stats.skipped == 0

    @pytest.mark.asyncio
    async def test_discover_skips_existing(self, isolated_environment: Path) -> None:
        """Test that unchanged images are skipped."""
        images_dir = isolated_environment / "images"

        # Initialize DB and add one image
        db.init_db()
        test_image = images_dir / "test_0.jpg"
        from sims.walker import compute_file_hash

        db.register_image(
            original_path=str(test_image),
            file_hash=compute_file_hash(test_image),
            file_size=test_image.stat().st_size,
            format="jpeg",
        )

        agent = IngestionAgent()
        stats = ProcessingStats(total=4)

        # Run discovery
        discovered = await agent._discover_stage(images_dir, recursive=True, stats=stats)

        assert len(discovered) == 3  # One was already registered
        assert stats.skipped == 1


class TestThumbnailStage:
    """Tests for thumbnail generation stage."""

    @pytest.mark.asyncio
    async def test_thumbnail_generation(self, isolated_environment: Path) -> None:
        """Test thumbnail is generated and path stored."""
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"

        # Initialize DB
        db.init_db()
        from sims.walker import compute_file_hash

        image_id = db.register_image(
            original_path=str(test_image),
            file_hash=compute_file_hash(test_image),
            file_size=test_image.stat().st_size,
            format="jpeg",
        )

        image = db.get_image_by_id(image_id)

        agent = IngestionAgent()
        await agent._thumbnail_stage(image)

        # Check thumbnail was created
        updated = db.get_image_by_id(image_id)
        assert updated["cached_path"] is not None
        assert updated["thumbnail_at"] is not None
        assert Path(updated["cached_path"]).exists()


class TestDescribeStage:
    """Tests for metadata extraction and description stage."""

    @pytest.mark.asyncio
    async def test_describe_with_mock(self, isolated_environment: Path) -> None:
        """Test description stage with mocked vision model."""
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"

        # Initialize DB
        db.init_db()
        from sims.walker import compute_file_hash
        from sims.thumbnail import generate_thumbnail

        image_id = db.register_image(
            original_path=str(test_image),
            file_hash=compute_file_hash(test_image),
            file_size=test_image.stat().st_size,
            format="jpeg",
        )

        # Generate thumbnail first
        file_hash = compute_file_hash(test_image)
        thumbnail_path = generate_thumbnail(test_image, file_hash)
        db.update_image_thumbnail(image_id, str(thumbnail_path))

        image = db.get_image_by_id(image_id)

        # Mock vision model
        mock_result = {
            "scene": "A test image",
            "mood": "neutral",
            "tags": ["test", "image"],
            "colors": ["red", "blue"],
            "time_of_day": "unclear",
            "full_description": "A test image. Mood: neutral. Tags: test, image.",
            "raw_response": "Test response",
        }

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            mock_describe.return_value = mock_result

            agent = IngestionAgent()
            await agent._describe_stage(image)

        # Check description was stored
        updated = db.get_image_by_id(image_id)
        assert updated["description"] is not None
        assert updated["described_at"] is not None
        assert "test" in updated["description"].lower()


class TestEmbedStage:
    """Tests for embedding generation stage."""

    @pytest.mark.asyncio
    async def test_embed_with_mock(self, isolated_environment: Path) -> None:
        """Test embedding stage with mocked embedding model."""
        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Create a fake image record with description
        image_id = db.register_image(
            original_path="/fake/test.jpg",
            file_hash="fakehash123",
            file_size=1000,
            format="jpeg",
        )
        db.update_image_description(
            image_id=image_id,
            description="A beautiful sunset over the ocean",
            tags=["sunset", "ocean"],
            mood="peaceful",
        )

        image = db.get_image_by_id(image_id)

        # Mock embedding model
        mock_embedding = generate_mock_embedding(seed=100)

        with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = mock_embedding

            agent = IngestionAgent()
            await agent._embed_stage(image)

        # Check embedding was stored
        updated = db.get_image_by_id(image_id)
        assert updated["embedded_at"] is not None

        # Check vector store
        stored = vectorstore.get_embedding(image_id)
        assert stored is not None
        assert len(stored["embedding"]) == 768


class TestProcessSingleImage:
    """Tests for processing a single image."""

    @pytest.mark.asyncio
    async def test_process_single_image(self, isolated_environment: Path) -> None:
        """Test full pipeline on single image with mocks."""
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"

        mock_vision_result = {
            "scene": "A colorful test image",
            "mood": "vibrant",
            "tags": ["test", "colorful"],
            "colors": ["red", "green", "blue"],
            "time_of_day": "day",
            "full_description": "A colorful test image. Mood: vibrant.",
            "raw_response": "Response",
        }
        mock_embedding = generate_mock_embedding(seed=200)

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
                mock_describe.return_value = mock_vision_result
                mock_embed.return_value = mock_embedding

                agent = IngestionAgent()
                result = await agent.process_single_image(test_image)

        assert result["status"] == "processed"
        assert result["image_id"] is not None
        assert result["description"] is not None
        assert "colorful" in result["description"].lower()

    @pytest.mark.asyncio
    async def test_process_single_image_already_processed(self, isolated_environment: Path) -> None:
        """Test that already processed image returns quickly."""
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"

        # First, process the image
        mock_vision_result = {
            "scene": "A test",
            "mood": "neutral",
            "tags": ["test"],
            "colors": ["gray"],
            "time_of_day": "unclear",
            "full_description": "A test image.",
            "raw_response": "Response",
        }
        mock_embedding = generate_mock_embedding(seed=300)

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
                mock_describe.return_value = mock_vision_result
                mock_embed.return_value = mock_embedding

                agent = IngestionAgent()
                result1 = await agent.process_single_image(test_image)
                assert result1["status"] == "processed"

                # Try again - should be already processed
                result2 = await agent.process_single_image(test_image)
                assert result2["status"] == "already_processed"

                # Vision should only be called once
                assert mock_describe.call_count == 1


class TestProcessDirectory:
    """Tests for directory processing."""

    @pytest.mark.asyncio
    async def test_process_directory_full(self, isolated_environment: Path) -> None:
        """Test full directory processing with mocks."""
        images_dir = isolated_environment / "images"

        mock_vision_result = {
            "scene": "A test",
            "mood": "neutral",
            "tags": ["test"],
            "colors": ["gray"],
            "time_of_day": "unclear",
            "full_description": "A test image.",
            "raw_response": "Response",
        }
        mock_embedding = generate_mock_embedding(seed=400)

        progress_reports = []

        def on_progress(stats: ProcessingStats):
            progress_reports.append(stats.processed)

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
                mock_describe.return_value = mock_vision_result
                mock_embed.return_value = mock_embedding

                agent = IngestionAgent(on_progress=on_progress)
                job_id = await agent.process_directory(images_dir, recursive=True)

        assert job_id > 0

        # Check job status
        status = agent.get_job_status(job_id)
        assert status is not None
        assert status.status == "completed"
        assert status.processed_files == 4  # 3 + 1 in subdir
        assert status.failed_files == 0

        # Check progress was reported
        assert len(progress_reports) > 0

    @pytest.mark.asyncio
    async def test_process_directory_non_recursive(self, isolated_environment: Path) -> None:
        """Test non-recursive directory processing."""
        images_dir = isolated_environment / "images"

        mock_vision_result = {
            "scene": "A test",
            "mood": "neutral",
            "tags": ["test"],
            "colors": ["gray"],
            "time_of_day": "unclear",
            "full_description": "A test image.",
            "raw_response": "Response",
        }
        mock_embedding = generate_mock_embedding(seed=500)

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
                mock_describe.return_value = mock_vision_result
                mock_embed.return_value = mock_embedding

                agent = IngestionAgent()
                job_id = await agent.process_directory(images_dir, recursive=False)

        status = agent.get_job_status(job_id)
        assert status.processed_files == 3  # Only root directory


class TestResumeIncomplete:
    """Tests for resuming incomplete processing."""

    @pytest.mark.asyncio
    async def test_resume_incomplete(self, isolated_environment: Path) -> None:
        """Test resuming incomplete images."""
        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Create an image that has thumbnail but no description
        from sims.thumbnail import generate_thumbnail
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"
        from sims.walker import compute_file_hash

        file_hash = compute_file_hash(test_image)
        image_id = db.register_image(
            original_path=str(test_image),
            file_hash=file_hash,
            file_size=test_image.stat().st_size,
            format="jpeg",
        )
        thumbnail_path = generate_thumbnail(test_image, file_hash)
        db.update_image_thumbnail(image_id, str(thumbnail_path))

        # Now resume should pick this up
        mock_vision_result = {
            "scene": "Resumed test",
            "mood": "neutral",
            "tags": ["resumed"],
            "colors": ["gray"],
            "time_of_day": "unclear",
            "full_description": "A resumed test image.",
            "raw_response": "Response",
        }
        mock_embedding = generate_mock_embedding(seed=600)

        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
                mock_describe.return_value = mock_vision_result
                mock_embed.return_value = mock_embedding

                agent = IngestionAgent()
                stats = await agent.resume_incomplete()

        assert stats.processed >= 1

        # Check image is now fully processed
        updated = db.get_image_by_id(image_id)
        assert updated["embedded_at"] is not None


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_in_describe_stage(self, isolated_environment: Path) -> None:
        """Test that errors in describe stage are recorded."""
        images_dir = isolated_environment / "images"
        test_image = images_dir / "test_0.jpg"

        # Initialize DB
        db.init_db()
        vectorstore.init_vectorstore()

        from sims.walker import compute_file_hash
        from sims.thumbnail import generate_thumbnail

        file_hash = compute_file_hash(test_image)
        image_id = db.register_image(
            original_path=str(test_image),
            file_hash=file_hash,
            file_size=test_image.stat().st_size,
            format="jpeg",
        )
        thumbnail_path = generate_thumbnail(test_image, file_hash)
        db.update_image_thumbnail(image_id, str(thumbnail_path))

        image = db.get_image_by_id(image_id)

        # Make describe raise an error
        with patch("sims.agent.vision.describe_image", new_callable=AsyncMock) as mock_describe:
            mock_describe.side_effect = Exception("Vision model error")

            agent = IngestionAgent()
            with pytest.raises(Exception, match="Vision model error"):
                await agent._describe_stage(image)

    @pytest.mark.asyncio
    async def test_invalid_directory(self, isolated_environment: Path) -> None:
        """Test error handling for invalid directory."""
        agent = IngestionAgent()

        with pytest.raises(ValueError, match="Not a directory"):
            await agent.process_directory(Path("/nonexistent/path"))


class TestSearchImages:
    """Tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_with_results(self, isolated_environment: Path) -> None:
        """Test search returns results."""
        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Add some test images
        for i in range(3):
            image_id = db.register_image(
                original_path=f"/test/image_{i}.jpg",
                file_hash=f"hash_{i}",
                file_size=1000,
                format="jpeg",
            )
            db.update_image_description(
                image_id=image_id,
                description=f"Test image {i} with beach and sunset",
                tags=["beach", "sunset"],
                mood="peaceful",
            )
            db.mark_image_embedded(image_id)

            # Add to vector store
            embedding = generate_mock_embedding(seed=1000 + i)
            vectorstore.add_embedding(
                image_id=image_id,
                embedding=embedding,
                description=f"Test image {i} with beach and sunset",
                metadata={
                    "original_path": f"/test/image_{i}.jpg",
                    "has_gps": False,
                },
            )

        # Search
        query_embedding = generate_mock_embedding(seed=1000)

        with patch("sims.agent.embeddings.embed_text", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = query_embedding
            results = await search_images("beach sunset", limit=5)

        assert len(results) == 3
        assert results[0]["score"] > 0.9  # First should be very similar


class TestGetSystemStats:
    """Tests for system statistics."""

    def test_get_stats(self, isolated_environment: Path) -> None:
        """Test getting system statistics."""
        db.init_db()
        vectorstore.init_vectorstore()

        # Add some data
        image_id = db.register_image(
            original_path="/test/stats.jpg",
            file_hash="stats_hash",
            file_size=5000,
            format="jpeg",
        )

        stats = get_system_stats()

        assert "total_images" in stats
        assert stats["total_images"] >= 1
        assert "embeddings_count" in stats
        assert "cache_size_bytes" in stats
        assert "db_path" in stats


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self) -> None:
        """Test CLI help displays correctly."""
        from sims.__main__ import main
        import sys

        # Capture the exit
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["sims", "--help"]
            main()

        # Help should exit with 0
        assert exc_info.value.code == 0

    def test_cli_init(self, isolated_environment: Path) -> None:
        """Test CLI init command."""
        from sims.__main__ import cmd_init
        import argparse

        args = argparse.Namespace()
        result = cmd_init(args)

        assert result == 0
        assert Config.DB_PATH.exists()

    def test_cli_stats(self, isolated_environment: Path) -> None:
        """Test CLI stats command."""
        from sims.__main__ import cmd_stats
        import argparse

        # Initialize first
        db.init_db()

        args = argparse.Namespace(json=False, verbose=False)
        result = cmd_stats(args)

        assert result == 0
