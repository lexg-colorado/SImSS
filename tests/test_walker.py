"""
Tests for the walker module.

Tests file hash computation, format detection, directory traversal,
and change detection functionality.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator

import pytest

from sims.walker import (
    compute_file_hash,
    is_supported_format,
    get_format,
    discover_images,
    get_new_or_changed_images,
    count_images,
)
from sims import db


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_hash_computation(self, fixtures_dir: Path) -> None:
        """Test that hash is computed correctly."""
        sample_jpg = fixtures_dir / "sample.jpg"
        hash_result = compute_file_hash(sample_jpg)

        # Should be a 64-character hex string (SHA256)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_hash_is_deterministic(self, fixtures_dir: Path) -> None:
        """Test that same file produces same hash."""
        sample_jpg = fixtures_dir / "sample.jpg"
        hash1 = compute_file_hash(sample_jpg)
        hash2 = compute_file_hash(sample_jpg)
        assert hash1 == hash2

    def test_different_files_different_hashes(self, fixtures_dir: Path) -> None:
        """Test that different files produce different hashes."""
        hash_jpg = compute_file_hash(fixtures_dir / "sample.jpg")
        hash_png = compute_file_hash(fixtures_dir / "sample.png")
        assert hash_jpg != hash_png

    def test_hash_matches_manual_computation(self, fixtures_dir: Path) -> None:
        """Test that hash matches manual SHA256 computation."""
        sample_jpg = fixtures_dir / "sample.jpg"

        # Compute hash manually
        sha256 = hashlib.sha256()
        with open(sample_jpg, "rb") as f:
            sha256.update(f.read())
        expected_hash = sha256.hexdigest()

        assert compute_file_hash(sample_jpg) == expected_hash

    def test_nonexistent_file_raises_error(self) -> None:
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(Path("/nonexistent/file.jpg"))


class TestIsSupportedFormat:
    """Tests for is_supported_format function."""

    @pytest.mark.parametrize("extension", [
        ".jpg", ".jpeg", ".png", ".heic", ".heif",
        ".cr2", ".nef", ".arw", ".dng",
    ])
    def test_supported_formats(self, tmp_path: Path, extension: str) -> None:
        """Test that all documented formats are supported."""
        test_file = tmp_path / f"test{extension}"
        test_file.touch()
        assert is_supported_format(test_file) is True

    @pytest.mark.parametrize("extension", [
        ".JPG", ".JPEG", ".PNG", ".HEIC", ".CR2",
    ])
    def test_case_insensitive(self, tmp_path: Path, extension: str) -> None:
        """Test that format detection is case-insensitive."""
        test_file = tmp_path / f"test{extension}"
        test_file.touch()
        assert is_supported_format(test_file) is True

    @pytest.mark.parametrize("extension", [
        ".txt", ".pdf", ".doc", ".gif", ".bmp", ".tiff", ".webp",
    ])
    def test_unsupported_formats(self, tmp_path: Path, extension: str) -> None:
        """Test that unsupported formats return False."""
        test_file = tmp_path / f"test{extension}"
        test_file.touch()
        assert is_supported_format(test_file) is False


class TestGetFormat:
    """Tests for get_format function."""

    def test_jpg_normalized_to_jpeg(self, tmp_path: Path) -> None:
        """Test that .jpg is normalized to 'jpeg'."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        assert get_format(test_file) == "jpeg"

    def test_jpeg_stays_jpeg(self, tmp_path: Path) -> None:
        """Test that .jpeg stays as 'jpeg'."""
        test_file = tmp_path / "test.jpeg"
        test_file.touch()
        assert get_format(test_file) == "jpeg"

    def test_png_format(self, tmp_path: Path) -> None:
        """Test PNG format detection."""
        test_file = tmp_path / "test.png"
        test_file.touch()
        assert get_format(test_file) == "png"

    def test_heic_format(self, tmp_path: Path) -> None:
        """Test HEIC format detection."""
        test_file = tmp_path / "test.heic"
        test_file.touch()
        assert get_format(test_file) == "heic"

    def test_raw_formats(self, tmp_path: Path) -> None:
        """Test RAW format detection."""
        for ext, expected in [(".cr2", "cr2"), (".nef", "nef"), (".arw", "arw"), (".dng", "dng")]:
            test_file = tmp_path / f"test{ext}"
            test_file.touch()
            assert get_format(test_file) == expected


class TestDiscoverImages:
    """Tests for discover_images function."""

    def test_discover_in_flat_directory(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test discovering images in a flat directory."""
        # Copy sample images to test directory
        import shutil
        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", sample_image_dir / "photo2.png")

        images = list(discover_images(sample_image_dir))

        assert len(images) == 2
        paths = {img["path"].name for img in images}
        assert paths == {"photo1.jpg", "photo2.png"}

    def test_discover_recursive(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test recursive discovery finds images in subdirectories."""
        import shutil

        # Create subdirectory with image
        subdir = sample_image_dir / "subdir"
        subdir.mkdir(exist_ok=True)

        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", subdir / "photo2.png")

        images = list(discover_images(sample_image_dir, recursive=True))

        assert len(images) == 2
        paths = {img["path"].name for img in images}
        assert paths == {"photo1.jpg", "photo2.png"}

    def test_discover_non_recursive(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test non-recursive discovery ignores subdirectories."""
        import shutil

        subdir = sample_image_dir / "subdir"
        subdir.mkdir(exist_ok=True)

        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", subdir / "photo2.png")

        images = list(discover_images(sample_image_dir, recursive=False))

        assert len(images) == 1
        assert images[0]["path"].name == "photo1.jpg"

    def test_discover_skips_unsupported_files(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test that unsupported file types are skipped."""
        import shutil

        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo.jpg")
        (sample_image_dir / "document.txt").write_text("hello")
        (sample_image_dir / "data.pdf").write_bytes(b"PDF")

        images = list(discover_images(sample_image_dir))

        assert len(images) == 1
        assert images[0]["path"].name == "photo.jpg"

    def test_discover_yields_correct_structure(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test that discovered images have correct structure."""
        import shutil
        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "test.jpg")

        images = list(discover_images(sample_image_dir))

        assert len(images) == 1
        img = images[0]

        assert "path" in img
        assert "hash" in img
        assert "size" in img
        assert "format" in img

        assert isinstance(img["path"], Path)
        assert len(img["hash"]) == 64  # SHA256
        assert isinstance(img["size"], int)
        assert img["size"] > 0
        assert img["format"] == "jpeg"

    def test_not_a_directory_error(self, fixtures_dir: Path) -> None:
        """Test that NotADirectoryError is raised for files."""
        with pytest.raises(NotADirectoryError):
            list(discover_images(fixtures_dir / "sample.jpg"))


class TestGetNewOrChangedImages:
    """Tests for get_new_or_changed_images function."""

    def test_detects_new_images(
        self, sample_image_dir: Path, fixtures_dir: Path, temp_db_path: Path
    ) -> None:
        """Test that new images are detected."""
        import shutil
        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "new_photo.jpg")

        # Initialize empty database
        db.init_db(temp_db_path)

        images = list(get_new_or_changed_images(
            sample_image_dir, db_path=temp_db_path
        ))

        assert len(images) == 1
        assert images[0]["status"] == "new"
        assert images[0]["existing_id"] is None

    def test_detects_unchanged_images(
        self, sample_image_dir: Path, fixtures_dir: Path, temp_db_path: Path
    ) -> None:
        """Test that unchanged images are skipped."""
        import shutil
        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "existing.jpg")

        # Initialize database and register the image
        db.init_db(temp_db_path)
        from sims.walker import compute_file_hash
        file_path = sample_image_dir / "existing.jpg"
        file_hash = compute_file_hash(file_path)
        db.register_image(
            original_path=str(file_path),
            file_hash=file_hash,
            file_size=file_path.stat().st_size,
            format="jpeg",
            db_path=temp_db_path,
        )

        # Should not find any new/changed images
        images = list(get_new_or_changed_images(
            sample_image_dir, db_path=temp_db_path
        ))

        assert len(images) == 0

    def test_detects_changed_images(
        self, sample_image_dir: Path, fixtures_dir: Path, temp_db_path: Path
    ) -> None:
        """Test that changed images are detected."""
        import shutil
        target_path = sample_image_dir / "changing.jpg"
        shutil.copy(fixtures_dir / "sample.jpg", target_path)

        # Initialize database and register with original hash
        db.init_db(temp_db_path)
        original_hash = compute_file_hash(target_path)
        image_id = db.register_image(
            original_path=str(target_path),
            file_hash=original_hash,
            file_size=target_path.stat().st_size,
            format="jpeg",
            db_path=temp_db_path,
        )

        # Replace with different file
        shutil.copy(fixtures_dir / "sample.png", target_path)

        # Should detect as changed
        images = list(get_new_or_changed_images(
            sample_image_dir, db_path=temp_db_path
        ))

        assert len(images) == 1
        assert images[0]["status"] == "changed"
        assert images[0]["existing_id"] == image_id


class TestCountImages:
    """Tests for count_images function."""

    def test_count_images(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test counting images in a directory."""
        import shutil
        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", sample_image_dir / "photo2.png")
        (sample_image_dir / "document.txt").write_text("hello")

        count = count_images(sample_image_dir)
        assert count == 2

    def test_count_recursive(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test recursive counting."""
        import shutil
        subdir = sample_image_dir / "subdir"
        subdir.mkdir(exist_ok=True)

        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", subdir / "photo2.png")

        count = count_images(sample_image_dir, recursive=True)
        assert count == 2

    def test_count_non_recursive(self, sample_image_dir: Path, fixtures_dir: Path) -> None:
        """Test non-recursive counting."""
        import shutil
        subdir = sample_image_dir / "subdir"
        subdir.mkdir(exist_ok=True)

        shutil.copy(fixtures_dir / "sample.jpg", sample_image_dir / "photo1.jpg")
        shutil.copy(fixtures_dir / "sample.png", subdir / "photo2.png")

        count = count_images(sample_image_dir, recursive=False)
        assert count == 1

    def test_count_empty_directory(self, tmp_path: Path) -> None:
        """Test counting in empty directory."""
        count = count_images(tmp_path)
        assert count == 0

    def test_count_not_a_directory(self, fixtures_dir: Path) -> None:
        """Test that NotADirectoryError is raised for files."""
        with pytest.raises(NotADirectoryError):
            count_images(fixtures_dir / "sample.jpg")
