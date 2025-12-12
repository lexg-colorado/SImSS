"""
Tests for sims.db module.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from sims import db


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_init_db_creates_database(self, temp_db_path: Path):
        """Test that init_db creates a database file."""
        assert not temp_db_path.exists()
        db.init_db(temp_db_path)
        assert temp_db_path.exists()

    def test_init_db_creates_tables(self, temp_db_path: Path):
        """Test that init_db creates the required tables."""
        db.init_db(temp_db_path)

        with db.get_db(temp_db_path) as conn:
            # Check images table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='images'"
            )
            assert cursor.fetchone() is not None

            # Check ingestion_jobs table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_jobs'"
            )
            assert cursor.fetchone() is not None

    def test_init_db_creates_indexes(self, temp_db_path: Path):
        """Test that init_db creates the required indexes."""
        db.init_db(temp_db_path)

        with db.get_db(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = {row["name"] for row in cursor.fetchall()}

            assert "idx_images_date" in indexes
            assert "idx_images_format" in indexes
            assert "idx_images_processing" in indexes
            assert "idx_images_hash" in indexes

    def test_init_db_idempotent(self, temp_db_path: Path):
        """Test that init_db can be called multiple times safely."""
        db.init_db(temp_db_path)
        db.init_db(temp_db_path)  # Should not raise
        assert temp_db_path.exists()

    def test_get_connection(self, temp_db_path: Path):
        """Test that get_connection returns a working connection."""
        db.init_db(temp_db_path)
        conn = db.get_connection(temp_db_path)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()


class TestImageOperations:
    """Tests for image CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_db_path: Path):
        """Initialize database before each test."""
        self.db_path = temp_db_path
        db.init_db(temp_db_path)

    def test_register_image(self):
        """Test registering a new image."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )
        assert image_id == 1

    def test_register_image_duplicate_path_fails(self):
        """Test that registering duplicate paths raises an error."""
        db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        with pytest.raises(sqlite3.IntegrityError):
            db.register_image(
                original_path="/photos/test.jpg",
                file_hash="def456",
                file_size=2048,
                format="jpeg",
                db_path=self.db_path,
            )

    def test_get_image_by_path(self):
        """Test retrieving an image by path."""
        db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        image = db.get_image_by_path("/photos/test.jpg", db_path=self.db_path)
        assert image is not None
        assert image["original_path"] == "/photos/test.jpg"
        assert image["file_hash"] == "abc123"
        assert image["file_size"] == 1024
        assert image["format"] == "jpeg"

    def test_get_image_by_path_not_found(self):
        """Test retrieving a non-existent image by path."""
        image = db.get_image_by_path("/photos/nonexistent.jpg", db_path=self.db_path)
        assert image is None

    def test_get_image_by_id(self):
        """Test retrieving an image by ID."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image is not None
        assert image["id"] == image_id

    def test_get_image_by_id_not_found(self):
        """Test retrieving a non-existent image by ID."""
        image = db.get_image_by_id(999, db_path=self.db_path)
        assert image is None

    def test_get_image_by_hash(self):
        """Test retrieving an image by hash."""
        db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        image = db.get_image_by_hash("abc123", db_path=self.db_path)
        assert image is not None
        assert image["file_hash"] == "abc123"

    def test_get_image_by_hash_not_found(self):
        """Test retrieving a non-existent image by hash."""
        image = db.get_image_by_hash("nonexistent", db_path=self.db_path)
        assert image is None

    def test_update_image_thumbnail(self):
        """Test updating the thumbnail path for an image."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        db.update_image_thumbnail(
            image_id,
            cached_path="/cache/abc123.jpg",
            db_path=self.db_path,
        )

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["cached_path"] == "/cache/abc123.jpg"
        assert image["thumbnail_at"] is not None

    def test_update_image_metadata(self):
        """Test updating EXIF metadata for an image."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        date_taken = datetime(2023, 7, 15, 14, 30, 0)
        db.update_image_metadata(
            image_id,
            date_taken=date_taken,
            gps_lat=37.7749,
            gps_lon=-122.4194,
            camera_make="Canon",
            camera_model="EOS R5",
            db_path=self.db_path,
        )

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["gps_lat"] == pytest.approx(37.7749)
        assert image["gps_lon"] == pytest.approx(-122.4194)
        assert image["camera_make"] == "Canon"
        assert image["camera_model"] == "EOS R5"

    def test_update_image_description(self):
        """Test updating the description for an image."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        tags = ["beach", "sunset", "ocean"]
        db.update_image_description(
            image_id,
            description="A beautiful sunset over the ocean.",
            tags=tags,
            mood="peaceful",
            db_path=self.db_path,
        )

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["description"] == "A beautiful sunset over the ocean."
        assert json.loads(image["tags"]) == tags
        assert image["mood"] == "peaceful"
        assert image["described_at"] is not None

    def test_mark_image_embedded(self):
        """Test marking an image as embedded."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        db.mark_image_embedded(image_id, db_path=self.db_path)

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["embedded_at"] is not None

    def test_mark_image_error(self):
        """Test marking an image as having an error."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        db.mark_image_error(image_id, "Test error message", db_path=self.db_path)

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["error_message"] == "Test error message"
        assert image["retry_count"] == 1

        # Mark error again to test retry count increment
        db.mark_image_error(image_id, "Another error", db_path=self.db_path)
        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["retry_count"] == 2

    def test_clear_image_error(self):
        """Test clearing an image error."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        db.mark_image_error(image_id, "Test error", db_path=self.db_path)
        db.clear_image_error(image_id, db_path=self.db_path)

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image["error_message"] is None
        # retry_count should remain
        assert image["retry_count"] == 1

    def test_delete_image(self):
        """Test deleting an image."""
        image_id = db.register_image(
            original_path="/photos/test.jpg",
            file_hash="abc123",
            file_size=1024,
            format="jpeg",
            db_path=self.db_path,
        )

        result = db.delete_image(image_id, db_path=self.db_path)
        assert result is True

        image = db.get_image_by_id(image_id, db_path=self.db_path)
        assert image is None

    def test_delete_image_not_found(self):
        """Test deleting a non-existent image."""
        result = db.delete_image(999, db_path=self.db_path)
        assert result is False


class TestPendingImages:
    """Tests for get_pending_images function."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_db_path: Path):
        """Initialize database and create test images."""
        self.db_path = temp_db_path
        db.init_db(temp_db_path)

        # Create images at different stages
        # Image 1: Just discovered (pending thumbnail)
        self.img1 = db.register_image(
            original_path="/photos/img1.jpg",
            file_hash="hash1",
            file_size=1000,
            format="jpeg",
            db_path=self.db_path,
        )

        # Image 2: Has thumbnail (pending describe)
        self.img2 = db.register_image(
            original_path="/photos/img2.jpg",
            file_hash="hash2",
            file_size=2000,
            format="jpeg",
            db_path=self.db_path,
        )
        db.update_image_thumbnail(self.img2, "/cache/hash2.jpg", db_path=self.db_path)

        # Image 3: Has description (pending embed)
        self.img3 = db.register_image(
            original_path="/photos/img3.jpg",
            file_hash="hash3",
            file_size=3000,
            format="jpeg",
            db_path=self.db_path,
        )
        db.update_image_thumbnail(self.img3, "/cache/hash3.jpg", db_path=self.db_path)
        db.update_image_description(
            self.img3,
            description="Test",
            tags=["test"],
            mood="neutral",
            db_path=self.db_path,
        )

        # Image 4: Fully processed
        self.img4 = db.register_image(
            original_path="/photos/img4.jpg",
            file_hash="hash4",
            file_size=4000,
            format="jpeg",
            db_path=self.db_path,
        )
        db.update_image_thumbnail(self.img4, "/cache/hash4.jpg", db_path=self.db_path)
        db.update_image_description(
            self.img4,
            description="Test",
            tags=["test"],
            mood="neutral",
            db_path=self.db_path,
        )
        db.mark_image_embedded(self.img4, db_path=self.db_path)

    def test_get_pending_thumbnail(self):
        """Test getting images pending thumbnail generation."""
        pending = db.get_pending_images("thumbnail", db_path=self.db_path)
        assert len(pending) == 1
        assert pending[0]["id"] == self.img1

    def test_get_pending_describe(self):
        """Test getting images pending description generation."""
        pending = db.get_pending_images("describe", db_path=self.db_path)
        assert len(pending) == 1
        assert pending[0]["id"] == self.img2

    def test_get_pending_embed(self):
        """Test getting images pending embedding."""
        pending = db.get_pending_images("embed", db_path=self.db_path)
        assert len(pending) == 1
        assert pending[0]["id"] == self.img3

    def test_get_pending_with_limit(self):
        """Test getting pending images with a limit."""
        # Add more images pending thumbnail
        for i in range(5):
            db.register_image(
                original_path=f"/photos/extra{i}.jpg",
                file_hash=f"extrahash{i}",
                file_size=1000,
                format="jpeg",
                db_path=self.db_path,
            )

        pending = db.get_pending_images("thumbnail", limit=3, db_path=self.db_path)
        assert len(pending) == 3

    def test_get_pending_invalid_stage(self):
        """Test that invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):
            db.get_pending_images("invalid", db_path=self.db_path)

    def test_get_pending_excludes_errored_images(self):
        """Test that images with errors exceeding max retries are excluded."""
        # Mark img1 with max retries exceeded
        for _ in range(4):  # MAX_RETRIES is 3 by default
            db.mark_image_error(self.img1, "Error", db_path=self.db_path)

        pending = db.get_pending_images("thumbnail", db_path=self.db_path)
        assert len(pending) == 0


class TestJobOperations:
    """Tests for job CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_db_path: Path):
        """Initialize database before each test."""
        self.db_path = temp_db_path
        db.init_db(temp_db_path)

    def test_create_job(self):
        """Test creating a new job."""
        job_id = db.create_job(
            root_path="/photos",
            total_files=100,
            db_path=self.db_path,
        )
        assert job_id == 1

    def test_get_job(self):
        """Test retrieving a job by ID."""
        job_id = db.create_job(
            root_path="/photos",
            total_files=100,
            db_path=self.db_path,
        )

        job = db.get_job(job_id, db_path=self.db_path)
        assert job is not None
        assert job["root_path"] == "/photos"
        assert job["total_files"] == 100
        assert job["status"] == "running"
        assert job["processed_files"] == 0
        assert job["failed_files"] == 0

    def test_get_job_not_found(self):
        """Test retrieving a non-existent job."""
        job = db.get_job(999, db_path=self.db_path)
        assert job is None

    def test_update_job_progress(self):
        """Test updating job progress."""
        job_id = db.create_job(
            root_path="/photos",
            total_files=100,
            db_path=self.db_path,
        )

        db.update_job_progress(job_id, processed=50, failed=5, db_path=self.db_path)

        job = db.get_job(job_id, db_path=self.db_path)
        assert job["processed_files"] == 50
        assert job["failed_files"] == 5

    def test_complete_job(self):
        """Test completing a job."""
        job_id = db.create_job(
            root_path="/photos",
            total_files=100,
            db_path=self.db_path,
        )

        db.complete_job(job_id, status="completed", db_path=self.db_path)

        job = db.get_job(job_id, db_path=self.db_path)
        assert job["status"] == "completed"
        assert job["completed_at"] is not None

    def test_get_active_jobs(self):
        """Test getting active jobs."""
        # Create multiple jobs
        job1 = db.create_job("/photos1", 100, db_path=self.db_path)
        job2 = db.create_job("/photos2", 200, db_path=self.db_path)
        job3 = db.create_job("/photos3", 300, db_path=self.db_path)

        # Complete one job
        db.complete_job(job2, db_path=self.db_path)

        active = db.get_active_jobs(db_path=self.db_path)
        assert len(active) == 2
        active_ids = {job["id"] for job in active}
        assert job1 in active_ids
        assert job3 in active_ids
        assert job2 not in active_ids


class TestStatistics:
    """Tests for statistics functions."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_db_path: Path):
        """Initialize database and create test data."""
        self.db_path = temp_db_path
        db.init_db(temp_db_path)

    def test_get_stats_empty_db(self):
        """Test statistics on an empty database."""
        stats = db.get_stats(db_path=self.db_path)
        assert stats["total_images"] == 0
        assert stats["processed"] == 0
        assert stats["pending"] == 0
        assert stats["errors"] == 0

    def test_get_stats_with_images(self):
        """Test statistics with various images."""
        # Create images at different stages
        img1 = db.register_image("/p/1.jpg", "h1", 1000, "jpeg", db_path=self.db_path)
        img2 = db.register_image("/p/2.jpg", "h2", 2000, "png", db_path=self.db_path)
        img3 = db.register_image("/p/3.jpg", "h3", 3000, "jpeg", db_path=self.db_path)

        # Progress img2 to thumbnail stage
        db.update_image_thumbnail(img2, "/c/h2.jpg", db_path=self.db_path)

        # Progress img3 to fully processed
        db.update_image_thumbnail(img3, "/c/h3.jpg", db_path=self.db_path)
        db.update_image_description(img3, "Test", ["tag"], "mood", db_path=self.db_path)
        db.mark_image_embedded(img3, db_path=self.db_path)

        stats = db.get_stats(db_path=self.db_path)
        assert stats["total_images"] == 3
        assert stats["processed"] == 1
        assert stats["pending_thumbnail"] == 1
        assert stats["pending_describe"] == 1
        assert stats["pending_embed"] == 0
        assert stats["pending"] == 2
        assert stats["by_format"]["jpeg"] == 2
        assert stats["by_format"]["png"] == 1
        assert stats["total_size_bytes"] == 6000

    def test_get_stats_with_gps(self):
        """Test GPS statistics."""
        img1 = db.register_image("/p/1.jpg", "h1", 1000, "jpeg", db_path=self.db_path)
        img2 = db.register_image("/p/2.jpg", "h2", 2000, "jpeg", db_path=self.db_path)

        db.update_image_metadata(
            img1,
            gps_lat=37.7749,
            gps_lon=-122.4194,
            db_path=self.db_path,
        )

        stats = db.get_stats(db_path=self.db_path)
        assert stats["with_gps"] == 1

    def test_get_stats_with_errors(self):
        """Test error statistics."""
        img1 = db.register_image("/p/1.jpg", "h1", 1000, "jpeg", db_path=self.db_path)
        db.mark_image_error(img1, "Test error", db_path=self.db_path)

        stats = db.get_stats(db_path=self.db_path)
        assert stats["errors"] == 1

    def test_get_stats_with_jobs(self):
        """Test job statistics."""
        db.create_job("/photos1", 100, db_path=self.db_path)
        job2 = db.create_job("/photos2", 200, db_path=self.db_path)
        db.complete_job(job2, db_path=self.db_path)

        stats = db.get_stats(db_path=self.db_path)
        assert stats["total_jobs"] == 2
        assert stats["active_jobs"] == 1


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_db_path: Path):
        """Initialize database before each test."""
        self.db_path = temp_db_path
        db.init_db(temp_db_path)

    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        errors = []
        results = []

        def register_images(thread_id: int, count: int):
            try:
                for i in range(count):
                    img_id = db.register_image(
                        original_path=f"/photos/thread{thread_id}/img{i}.jpg",
                        file_hash=f"hash_{thread_id}_{i}",
                        file_size=1000,
                        format="jpeg",
                        db_path=self.db_path,
                    )
                    results.append(img_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(3):
            thread = threading.Thread(target=register_images, args=(t, 10))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 30

        stats = db.get_stats(db_path=self.db_path)
        assert stats["total_images"] == 30

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read and write operations."""
        # Pre-populate some data
        for i in range(10):
            db.register_image(f"/p/{i}.jpg", f"h{i}", 1000, "jpeg", db_path=self.db_path)

        errors = []
        read_results = []

        def writer(count: int):
            try:
                for i in range(count):
                    db.register_image(
                        f"/new/{i}.jpg", f"new_h{i}", 1000, "jpeg",
                        db_path=self.db_path
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", e))

        def reader(count: int):
            try:
                for _ in range(count):
                    stats = db.get_stats(db_path=self.db_path)
                    read_results.append(stats["total_images"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("reader", e))

        writer_thread = threading.Thread(target=writer, args=(20,))
        reader_thread = threading.Thread(target=reader, args=(20,))

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(r >= 10 for r in read_results)  # At least initial count


class TestContextManager:
    """Tests for the get_db context manager."""

    def test_get_db_commits_on_success(self, temp_db_path: Path):
        """Test that get_db commits changes on success."""
        db.init_db(temp_db_path)

        with db.get_db(temp_db_path) as conn:
            conn.execute(
                "INSERT INTO images (original_path, file_hash, file_size, format) VALUES (?, ?, ?, ?)",
                ("/test.jpg", "hash", 1000, "jpeg"),
            )

        # Verify data persisted
        image = db.get_image_by_path("/test.jpg", db_path=temp_db_path)
        assert image is not None

    def test_get_db_rollback_on_error(self, temp_db_path: Path):
        """Test that get_db rolls back changes on error."""
        db.init_db(temp_db_path)

        try:
            with db.get_db(temp_db_path) as conn:
                conn.execute(
                    "INSERT INTO images (original_path, file_hash, file_size, format) VALUES (?, ?, ?, ?)",
                    ("/test.jpg", "hash", 1000, "jpeg"),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify data was rolled back
        image = db.get_image_by_path("/test.jpg", db_path=temp_db_path)
        assert image is None
