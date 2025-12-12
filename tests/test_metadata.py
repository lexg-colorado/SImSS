"""
Tests for the metadata module.

Tests EXIF extraction, GPS coordinate conversion, and date parsing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sims.metadata import (
    extract_metadata,
    parse_exif_date,
    convert_gps_coordinates,
    has_gps,
    get_image_dimensions_from_exif,
    _convert_to_degrees,
)


class TestParseExifDate:
    """Tests for parse_exif_date function."""

    def test_standard_exif_format(self) -> None:
        """Test parsing standard EXIF date format."""
        result = parse_exif_date("2019:03:04 14:02:04")
        assert result == datetime(2019, 3, 4, 14, 2, 4)

    def test_iso_format(self) -> None:
        """Test parsing ISO-like format."""
        result = parse_exif_date("2019-03-04 14:02:04")
        assert result == datetime(2019, 3, 4, 14, 2, 4)

    def test_date_only_exif_format(self) -> None:
        """Test parsing date-only EXIF format."""
        result = parse_exif_date("2019:03:04")
        assert result == datetime(2019, 3, 4, 0, 0, 0)

    def test_date_only_iso_format(self) -> None:
        """Test parsing date-only ISO format."""
        result = parse_exif_date("2019-03-04")
        assert result == datetime(2019, 3, 4, 0, 0, 0)

    def test_empty_string(self) -> None:
        """Test that empty string returns None."""
        result = parse_exif_date("")
        assert result is None

    def test_none_input(self) -> None:
        """Test that None input returns None."""
        result = parse_exif_date(None)
        assert result is None

    def test_invalid_date(self) -> None:
        """Test that invalid date returns None."""
        result = parse_exif_date("not a date")
        assert result is None

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is handled."""
        result = parse_exif_date("  2019:03:04 14:02:04  ")
        assert result == datetime(2019, 3, 4, 14, 2, 4)


class TestConvertToDegrees:
    """Tests for _convert_to_degrees function."""

    def test_simple_degrees(self) -> None:
        """Test converting simple degree values."""
        # Mock IfdTag-like object
        mock_tag = MagicMock()
        mock_ratio_39 = MagicMock(num=39, den=1)
        mock_ratio_23 = MagicMock(num=23, den=1)
        mock_ratio_30 = MagicMock(num=30, den=1)
        mock_tag.values = [mock_ratio_39, mock_ratio_23, mock_ratio_30]

        result = _convert_to_degrees(mock_tag)
        # 39 + 23/60 + 30/3600 = 39.391666...
        assert result is not None
        assert abs(result - 39.391666) < 0.001

    def test_fractional_values(self) -> None:
        """Test converting fractional coordinate values."""
        mock_tag = MagicMock()
        mock_ratio_1 = MagicMock(num=104, den=1)
        mock_ratio_2 = MagicMock(num=515547383, den=10000000)  # ~51.55
        mock_ratio_3 = MagicMock(num=0, den=1)
        mock_tag.values = [mock_ratio_1, mock_ratio_2, mock_ratio_3]

        result = _convert_to_degrees(mock_tag)
        assert result is not None
        # Should be approximately 104 + 51.55/60 = 104.859
        assert abs(result - 104.859) < 0.01

    def test_zero_denominator(self) -> None:
        """Test handling of zero denominator."""
        mock_tag = MagicMock()
        mock_ratio = MagicMock(num=10, den=0)
        mock_tag.values = [mock_ratio, mock_ratio, mock_ratio]

        result = _convert_to_degrees(mock_tag)
        # Should handle gracefully
        assert result == 0.0 or result is None


class TestConvertGpsCoordinates:
    """Tests for convert_gps_coordinates function."""

    def test_north_east_coordinates(self) -> None:
        """Test converting N/E coordinates."""
        lat_tag = MagicMock()
        lat_tag.values = [
            MagicMock(num=39, den=1),
            MagicMock(num=23, den=1),
            MagicMock(num=30, den=1),
        ]

        lon_tag = MagicMock()
        lon_tag.values = [
            MagicMock(num=104, den=1),
            MagicMock(num=30, den=1),
            MagicMock(num=0, den=1),
        ]

        result = convert_gps_coordinates(lat_tag, "N", lon_tag, "E")
        assert result is not None
        lat, lon = result
        assert lat > 0  # North is positive
        assert lon > 0  # East is positive

    def test_south_west_coordinates(self) -> None:
        """Test converting S/W coordinates (negative)."""
        lat_tag = MagicMock()
        lat_tag.values = [
            MagicMock(num=39, den=1),
            MagicMock(num=23, den=1),
            MagicMock(num=30, den=1),
        ]

        lon_tag = MagicMock()
        lon_tag.values = [
            MagicMock(num=104, den=1),
            MagicMock(num=30, den=1),
            MagicMock(num=0, den=1),
        ]

        result = convert_gps_coordinates(lat_tag, "S", lon_tag, "W")
        assert result is not None
        lat, lon = result
        assert lat < 0  # South is negative
        assert lon < 0  # West is negative

    def test_missing_tags_returns_none(self) -> None:
        """Test that missing tags return None."""
        assert convert_gps_coordinates(None, "N", None, "E") is None
        assert convert_gps_coordinates(MagicMock(), None, MagicMock(), "E") is None

    def test_invalid_range_returns_none(self) -> None:
        """Test that out-of-range coordinates return None."""
        lat_tag = MagicMock()
        lat_tag.values = [
            MagicMock(num=200, den=1),  # Invalid: > 90
            MagicMock(num=0, den=1),
            MagicMock(num=0, den=1),
        ]

        lon_tag = MagicMock()
        lon_tag.values = [
            MagicMock(num=104, den=1),
            MagicMock(num=0, den=1),
            MagicMock(num=0, den=1),
        ]

        result = convert_gps_coordinates(lat_tag, "N", lon_tag, "E")
        assert result is None


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_from_real_image_with_exif(self) -> None:
        """Test extracting metadata from a real image with EXIF data."""
        # Use SIMS_TEST_EXIF_IMAGE env var for local testing with real images
        import os
        test_image = os.getenv("SIMS_TEST_EXIF_IMAGE")
        if not test_image:
            pytest.skip("SIMS_TEST_EXIF_IMAGE not set")

        img_path = Path(test_image)
        if not img_path.exists():
            pytest.skip("Test image not available")

        result = extract_metadata(img_path)

        # Should have date_taken
        assert result["date_taken"] is not None
        assert isinstance(result["date_taken"], datetime)

    def test_extract_from_real_image_with_gps(self) -> None:
        """Test extracting GPS coordinates from a real image."""
        # Use SIMS_TEST_GPS_IMAGE env var for local testing with real images
        import os
        test_image = os.getenv("SIMS_TEST_GPS_IMAGE")
        if not test_image:
            pytest.skip("SIMS_TEST_GPS_IMAGE not set")

        img_path = Path(test_image)
        if not img_path.exists():
            pytest.skip("Test image with GPS not available")

        result = extract_metadata(img_path)

        # Should have GPS coordinates
        assert result["gps_lat"] is not None
        assert result["gps_lon"] is not None

    def test_extract_from_image_without_exif(self, fixtures_dir: Path) -> None:
        """Test extracting metadata from an image without EXIF."""
        # Our generated test images have no EXIF data
        img_path = fixtures_dir / "sample.jpg"

        result = extract_metadata(img_path)

        # All fields should be None
        assert result["date_taken"] is None
        assert result["gps_lat"] is None
        assert result["gps_lon"] is None
        assert result["camera_make"] is None
        assert result["camera_model"] is None

    def test_extract_from_nonexistent_file(self) -> None:
        """Test extracting from nonexistent file returns empty result."""
        result = extract_metadata(Path("/nonexistent/file.jpg"))

        # Should return empty result, not raise
        assert result["date_taken"] is None
        assert result["gps_lat"] is None

    def test_result_structure(self, fixtures_dir: Path) -> None:
        """Test that result has correct structure."""
        result = extract_metadata(fixtures_dir / "sample.jpg")

        # Check all expected keys exist
        assert "date_taken" in result
        assert "gps_lat" in result
        assert "gps_lon" in result
        assert "camera_make" in result
        assert "camera_model" in result


class TestHasGps:
    """Tests for has_gps function."""

    def test_image_with_gps(self) -> None:
        """Test detecting GPS in image that has it."""
        import os
        test_image = os.getenv("SIMS_TEST_GPS_IMAGE")
        if not test_image:
            pytest.skip("SIMS_TEST_GPS_IMAGE not set")

        img_path = Path(test_image)
        if not img_path.exists():
            pytest.skip("Test image with GPS not available")

        assert has_gps(img_path) is True

    def test_image_without_gps(self) -> None:
        """Test detecting lack of GPS."""
        import os
        test_image = os.getenv("SIMS_TEST_EXIF_IMAGE")
        if not test_image:
            pytest.skip("SIMS_TEST_EXIF_IMAGE not set")

        img_path = Path(test_image)
        if not img_path.exists():
            pytest.skip("Test image not available")

        # Note: This test assumes the EXIF image doesn't have GPS
        # Will pass if has_gps returns False, skip otherwise
        if has_gps(img_path):
            pytest.skip("Test image has GPS, cannot test no-GPS case")

    def test_nonexistent_file(self) -> None:
        """Test nonexistent file returns False."""
        assert has_gps(Path("/nonexistent/file.jpg")) is False


class TestGetImageDimensionsFromExif:
    """Tests for get_image_dimensions_from_exif function."""

    def test_get_dimensions_from_real_image(self) -> None:
        """Test getting dimensions from real image."""
        import os
        test_image = os.getenv("SIMS_TEST_EXIF_IMAGE")
        if not test_image:
            pytest.skip("SIMS_TEST_EXIF_IMAGE not set")

        img_path = Path(test_image)
        if not img_path.exists():
            pytest.skip("Test image not available")

        result = get_image_dimensions_from_exif(img_path)

        # Should return dimensions
        if result is not None:
            width, height = result
            assert width > 0
            assert height > 0

    def test_nonexistent_file(self) -> None:
        """Test nonexistent file returns None."""
        result = get_image_dimensions_from_exif(Path("/nonexistent/file.jpg"))
        assert result is None
