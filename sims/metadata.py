# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
EXIF metadata extraction for SImS.

Extracts date taken, GPS coordinates, and camera information from images.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import exifread

logger = logging.getLogger(__name__)


def parse_exif_date(date_string: str) -> Optional[datetime]:
    """
    Parse EXIF date string to datetime object.

    EXIF dates are typically in format: "YYYY:MM:DD HH:MM:SS"

    Args:
        date_string: Date string from EXIF data.

    Returns:
        datetime object or None if parsing fails.
    """
    if not date_string:
        return None

    # Common EXIF date formats
    formats = [
        "%Y:%m:%d %H:%M:%S",      # Standard EXIF format
        "%Y-%m-%d %H:%M:%S",      # ISO-like format
        "%Y:%m:%d",               # Date only
        "%Y-%m-%d",               # ISO date only
    ]

    # Clean up the string
    date_string = str(date_string).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    logger.debug(f"Could not parse date: {date_string}")
    return None


def _convert_to_degrees(value) -> Optional[float]:
    """
    Convert EXIF GPS coordinate to decimal degrees.

    EXIF stores GPS as [degrees, minutes, seconds] with each as a Ratio.

    Args:
        value: EXIF IfdTag value containing GPS coordinate.

    Returns:
        Decimal degrees as float, or None if conversion fails.
    """
    try:
        # Handle exifread's IfdTag format
        if hasattr(value, 'values'):
            values = value.values
        else:
            values = value

        if len(values) < 3:
            return None

        # Each value might be a Ratio object with num/den attributes
        def to_float(v):
            if hasattr(v, 'num') and hasattr(v, 'den'):
                if v.den == 0:
                    return 0.0
                return float(v.num) / float(v.den)
            return float(v)

        degrees = to_float(values[0])
        minutes = to_float(values[1])
        seconds = to_float(values[2])

        return degrees + (minutes / 60.0) + (seconds / 3600.0)

    except (TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        logger.debug(f"Error converting GPS coordinate: {e}")
        return None


def convert_gps_coordinates(
    lat_tag,
    lat_ref_tag,
    lon_tag,
    lon_ref_tag,
) -> Optional[tuple[float, float]]:
    """
    Convert EXIF GPS tags to decimal latitude and longitude.

    Args:
        lat_tag: GPS latitude EXIF tag.
        lat_ref_tag: GPS latitude reference (N/S).
        lon_tag: GPS longitude EXIF tag.
        lon_ref_tag: GPS longitude reference (E/W).

    Returns:
        Tuple of (latitude, longitude) in decimal degrees,
        or None if conversion fails.
    """
    if not all([lat_tag, lat_ref_tag, lon_tag, lon_ref_tag]):
        return None

    lat = _convert_to_degrees(lat_tag)
    lon = _convert_to_degrees(lon_tag)

    if lat is None or lon is None:
        return None

    # Get reference directions
    lat_ref = str(lat_ref_tag).strip().upper()
    lon_ref = str(lon_ref_tag).strip().upper()

    # Apply direction (S and W are negative)
    if lat_ref == 'S':
        lat = -lat
    if lon_ref == 'W':
        lon = -lon

    # Validate ranges
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.debug(f"GPS coordinates out of range: lat={lat}, lon={lon}")
        return None

    return (lat, lon)


def extract_metadata(image_path: Path) -> dict:
    """
    Extract EXIF metadata from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary containing:
            - date_taken: datetime when photo was taken, or None
            - gps_lat: GPS latitude in decimal degrees, or None
            - gps_lon: GPS longitude in decimal degrees, or None
            - camera_make: Camera manufacturer, or None
            - camera_model: Camera model, or None
    """
    image_path = Path(image_path)
    result = {
        "date_taken": None,
        "gps_lat": None,
        "gps_lon": None,
        "camera_make": None,
        "camera_model": None,
    }

    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return result

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        if not tags:
            logger.debug(f"No EXIF data found: {image_path}")
            return result

        # Extract date taken
        # Try multiple date tags in order of preference
        date_tags = [
            'EXIF DateTimeOriginal',
            'EXIF DateTimeDigitized',
            'Image DateTime',
        ]
        for tag_name in date_tags:
            if tag_name in tags:
                result["date_taken"] = parse_exif_date(str(tags[tag_name]))
                if result["date_taken"]:
                    break

        # Extract GPS coordinates
        gps_coords = convert_gps_coordinates(
            tags.get('GPS GPSLatitude'),
            tags.get('GPS GPSLatitudeRef'),
            tags.get('GPS GPSLongitude'),
            tags.get('GPS GPSLongitudeRef'),
        )
        if gps_coords:
            result["gps_lat"], result["gps_lon"] = gps_coords

        # Extract camera info
        if 'Image Make' in tags:
            result["camera_make"] = str(tags['Image Make']).strip()
        if 'Image Model' in tags:
            result["camera_model"] = str(tags['Image Model']).strip()

        logger.debug(
            f"Extracted metadata from {image_path.name}: "
            f"date={result['date_taken']}, "
            f"gps=({result['gps_lat']}, {result['gps_lon']}), "
            f"camera={result['camera_make']} {result['camera_model']}"
        )

    except Exception as e:
        logger.warning(f"Error extracting metadata from {image_path}: {e}")

    return result


def has_gps(image_path: Path) -> bool:
    """
    Quick check if an image has GPS coordinates.

    Args:
        image_path: Path to the image file.

    Returns:
        True if GPS coordinates are present.
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(
                f,
                details=False,
                stop_tag='GPS GPSLongitude'
            )
        return 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags
    except Exception:
        return False


def get_image_dimensions_from_exif(image_path: Path) -> Optional[tuple[int, int]]:
    """
    Get image dimensions from EXIF data without loading the full image.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (width, height) or None if not available.
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        width = tags.get('EXIF ExifImageWidth') or tags.get('Image ImageWidth')
        height = tags.get('EXIF ExifImageLength') or tags.get('Image ImageLength')

        if width and height:
            return (int(str(width)), int(str(height)))
    except Exception:
        pass

    return None
