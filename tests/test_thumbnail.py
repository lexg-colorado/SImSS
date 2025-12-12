"""
Tests for the thumbnail module.

Tests thumbnail generation, format conversion, aspect ratio preservation,
and cache path generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sims.thumbnail import (
    generate_thumbnail,
    generate_thumbnail_for_hash,
    get_cache_path,
    ensure_cache_dir,
    get_thumbnail_info,
    ThumbnailError,
    UnsupportedFormatError,
    _calculate_thumbnail_size,
    _apply_exif_orientation,
    HEIF_AVAILABLE,
    RAWPY_AVAILABLE,
)
from sims.config import Config


class TestEnsureCacheDir:
    """Tests for ensure_cache_dir function."""

    def test_creates_cache_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that cache directory is created."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(Config, "CACHE_DIR", cache_dir)

        result = ensure_cache_dir()

        assert cache_dir.exists()
        assert result == cache_dir

    def test_returns_existing_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that existing directory is returned without error."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Config, "CACHE_DIR", cache_dir)

        result = ensure_cache_dir()

        assert result == cache_dir


class TestGetCachePath:
    """Tests for get_cache_path function."""

    def test_returns_path_with_hash(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that cache path includes hash."""
        monkeypatch.setattr(Config, "CACHE_DIR", tmp_path)

        file_hash = "abc123def456"
        result = get_cache_path(file_hash)

        assert result == tmp_path / "abc123def456.jpg"

    def test_custom_extension(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test custom extension."""
        monkeypatch.setattr(Config, "CACHE_DIR", tmp_path)

        result = get_cache_path("hash123", extension=".png")
        assert result == tmp_path / "hash123.png"


class TestCalculateThumbnailSize:
    """Tests for _calculate_thumbnail_size function."""

    def test_landscape_image(self) -> None:
        """Test resizing landscape image."""
        original = (1200, 800)
        result = _calculate_thumbnail_size(original, max_size=768)
        assert result == (768, 512)

    def test_portrait_image(self) -> None:
        """Test resizing portrait image."""
        original = (800, 1200)
        result = _calculate_thumbnail_size(original, max_size=768)
        assert result == (512, 768)

    def test_square_image(self) -> None:
        """Test resizing square image."""
        original = (1000, 1000)
        result = _calculate_thumbnail_size(original, max_size=768)
        assert result == (768, 768)

    def test_smaller_than_max_unchanged(self) -> None:
        """Test that images smaller than max size are unchanged."""
        original = (500, 400)
        result = _calculate_thumbnail_size(original, max_size=768)
        assert result == (500, 400)

    def test_exactly_max_size_unchanged(self) -> None:
        """Test that images exactly at max size are unchanged."""
        original = (768, 600)
        result = _calculate_thumbnail_size(original, max_size=768)
        assert result == (768, 600)

    def test_aspect_ratio_preserved(self) -> None:
        """Test that aspect ratio is preserved."""
        original = (1920, 1080)
        result = _calculate_thumbnail_size(original, max_size=768)

        original_ratio = original[0] / original[1]
        result_ratio = result[0] / result[1]

        # Allow small rounding error
        assert abs(original_ratio - result_ratio) < 0.01


class TestGenerateThumbnail:
    """Tests for generate_thumbnail function."""

    def test_jpeg_thumbnail(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test generating thumbnail from JPEG."""
        source = fixtures_dir / "sample.jpg"
        output = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source, output)

        assert result == output
        assert output.exists()

        # Verify it's a valid image
        with Image.open(output) as img:
            assert img.format == "JPEG"

    def test_png_thumbnail(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test generating thumbnail from PNG."""
        source = fixtures_dir / "sample.png"
        output = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source, output)

        assert output.exists()
        with Image.open(output) as img:
            assert img.format == "JPEG"

    def test_large_image_resized(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that large images are resized correctly."""
        source = fixtures_dir / "large_image.jpg"  # 1200x800
        output = tmp_path / "thumb.jpg"

        generate_thumbnail(source, output, max_size=768)

        with Image.open(output) as img:
            # Longest edge should be 768
            assert max(img.width, img.height) == 768
            # Aspect ratio should be preserved (1200:800 = 3:2)
            assert img.width == 768
            assert img.height == 512

    def test_small_image_not_enlarged(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that small images are not enlarged."""
        source = fixtures_dir / "sample.jpg"  # 100x100
        output = tmp_path / "thumb.jpg"

        generate_thumbnail(source, output, max_size=768)

        with Image.open(output) as img:
            assert img.width == 100
            assert img.height == 100

    def test_rgba_converted_to_rgb(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that RGBA images are converted to RGB."""
        source = fixtures_dir / "transparent.png"  # RGBA
        output = tmp_path / "thumb.jpg"

        generate_thumbnail(source, output)

        with Image.open(output) as img:
            assert img.mode == "RGB"
            assert img.format == "JPEG"

    def test_quality_setting(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that quality setting affects file size."""
        source = fixtures_dir / "large_image.jpg"

        high_quality = tmp_path / "high.jpg"
        low_quality = tmp_path / "low.jpg"

        generate_thumbnail(source, high_quality, quality=95)
        generate_thumbnail(source, low_quality, quality=30)

        # Higher quality should produce larger file
        assert high_quality.stat().st_size > low_quality.stat().st_size

    def test_nonexistent_source_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent source raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            generate_thumbnail(
                tmp_path / "nonexistent.jpg",
                tmp_path / "thumb.jpg"
            )

    def test_creates_output_directory(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that output directory is created if needed."""
        source = fixtures_dir / "sample.jpg"
        output = tmp_path / "subdir" / "nested" / "thumb.jpg"

        generate_thumbnail(source, output)

        assert output.exists()

    def test_auto_cache_path(
        self, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test automatic cache path generation when output_path is None."""
        monkeypatch.setattr(Config, "CACHE_DIR", tmp_path)
        source = fixtures_dir / "sample.jpg"

        result = generate_thumbnail(source, output_path=None)

        assert result.parent == tmp_path
        assert result.suffix == ".jpg"
        assert result.exists()


class TestGenerateThumbnailForHash:
    """Tests for generate_thumbnail_for_hash function."""

    def test_uses_provided_hash(
        self, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that provided hash is used for cache path."""
        monkeypatch.setattr(Config, "CACHE_DIR", tmp_path)

        source = fixtures_dir / "sample.jpg"
        file_hash = "known_hash_12345"

        result = generate_thumbnail_for_hash(source, file_hash)

        assert result == tmp_path / "known_hash_12345.jpg"
        assert result.exists()


class TestGetThumbnailInfo:
    """Tests for get_thumbnail_info function."""

    def test_returns_correct_info(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test that thumbnail info is correct."""
        # Generate a thumbnail first
        source = fixtures_dir / "large_image.jpg"
        thumb = tmp_path / "thumb.jpg"
        generate_thumbnail(source, thumb, max_size=768)

        info = get_thumbnail_info(thumb)

        assert info["width"] == 768
        assert info["height"] == 512
        assert info["size_bytes"] > 0
        assert info["format"] == "JPEG"

    def test_nonexistent_thumbnail_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent thumbnail raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_thumbnail_info(tmp_path / "nonexistent.jpg")


class TestApplyExifOrientation:
    """Tests for _apply_exif_orientation function."""

    def test_no_exif_unchanged(self) -> None:
        """Test that images without EXIF are unchanged."""
        img = Image.new("RGB", (100, 100), color="red")
        result = _apply_exif_orientation(img)
        assert result.size == (100, 100)

    def test_handles_missing_orientation(self) -> None:
        """Test handling of EXIF without orientation tag."""
        img = Image.new("RGB", (100, 100), color="red")
        # Image has no EXIF data, should return unchanged
        result = _apply_exif_orientation(img)
        assert result.size == (100, 100)


class TestHeifSupport:
    """Tests for HEIC/HEIF format support."""

    @pytest.mark.skipif(not HEIF_AVAILABLE, reason="pillow-heif not installed")
    def test_heif_available_flag(self) -> None:
        """Test that HEIF_AVAILABLE is True when pillow-heif is installed."""
        assert HEIF_AVAILABLE is True

    @pytest.mark.skipif(HEIF_AVAILABLE, reason="Only test when pillow-heif not available")
    def test_heic_raises_unsupported_error(self, tmp_path: Path) -> None:
        """Test that HEIC files raise UnsupportedFormatError without pillow-heif."""
        # Create a fake HEIC file
        fake_heic = tmp_path / "test.heic"
        fake_heic.write_bytes(b"fake heic data")

        with pytest.raises(UnsupportedFormatError):
            generate_thumbnail(fake_heic, tmp_path / "thumb.jpg")


class TestRawSupport:
    """Tests for RAW format support."""

    @pytest.mark.skipif(not RAWPY_AVAILABLE, reason="rawpy not installed")
    def test_rawpy_available_flag(self) -> None:
        """Test that RAWPY_AVAILABLE is True when rawpy is installed."""
        assert RAWPY_AVAILABLE is True

    @pytest.mark.skipif(RAWPY_AVAILABLE, reason="Only test when rawpy not available")
    def test_raw_raises_unsupported_error(self, tmp_path: Path) -> None:
        """Test that RAW files raise UnsupportedFormatError without rawpy."""
        fake_raw = tmp_path / "test.cr2"
        fake_raw.write_bytes(b"fake raw data")

        with pytest.raises(UnsupportedFormatError):
            generate_thumbnail(fake_raw, tmp_path / "thumb.jpg")


class TestIntegrationWithRealImages:
    """Integration tests with real images from a configured test directory."""

    def test_thumbnail_from_pictures_jpg(self, tmp_path: Path) -> None:
        """Test thumbnail generation from a real JPG in test directory."""
        import os
        pictures_path_str = os.getenv("SIMS_TEST_PICTURES_DIR")
        if not pictures_path_str:
            pytest.skip("SIMS_TEST_PICTURES_DIR not set")

        pictures_path = Path(pictures_path_str)
        if not pictures_path.exists():
            pytest.skip("Test pictures directory not available")

        jpg_files = list(pictures_path.glob("*.JPG")) + list(pictures_path.glob("*.jpg"))

        if not jpg_files:
            pytest.skip("No JPG files found in test pictures directory")

        source = jpg_files[0]
        output = tmp_path / "real_thumb.jpg"

        result = generate_thumbnail(source, output, max_size=256)

        assert result.exists()
        with Image.open(result) as img:
            assert max(img.width, img.height) <= 256

    def test_thumbnail_from_pictures_png(self, tmp_path: Path) -> None:
        """Test thumbnail generation from a real PNG in test directory."""
        import os
        pictures_path_str = os.getenv("SIMS_TEST_PICTURES_DIR")
        if not pictures_path_str:
            pytest.skip("SIMS_TEST_PICTURES_DIR not set")

        pictures_path = Path(pictures_path_str)
        if not pictures_path.exists():
            pytest.skip("Test pictures directory not available")

        png_files = list(pictures_path.glob("*.PNG")) + list(pictures_path.glob("*.png"))

        if not png_files:
            pytest.skip("No PNG files found in test pictures directory")

        source = png_files[0]
        output = tmp_path / "real_thumb.jpg"

        result = generate_thumbnail(source, output, max_size=256)

        assert result.exists()
        with Image.open(result) as img:
            assert img.format == "JPEG"
            assert max(img.width, img.height) <= 256
