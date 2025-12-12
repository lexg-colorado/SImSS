# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Image thumbnail generation for SImS.

Handles resizing and format conversion for various image formats including
JPEG, PNG, HEIC, and RAW formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags

from sims.config import Config, CACHE_DIR, THUMBNAIL_MAX_SIZE, THUMBNAIL_QUALITY

logger = logging.getLogger(__name__)

# Optional imports for specialized formats
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    logger.debug("pillow-heif not available, HEIC/HEIF support disabled")

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False
    logger.debug("rawpy not available, RAW format support disabled")


# RAW format extensions
RAW_FORMATS = frozenset({".cr2", ".nef", ".arw", ".dng"})

# HEIF format extensions
HEIF_FORMATS = frozenset({".heic", ".heif"})


class ThumbnailError(Exception):
    """Exception raised when thumbnail generation fails."""
    pass


class UnsupportedFormatError(ThumbnailError):
    """Exception raised when an image format is not supported."""
    pass


def ensure_cache_dir() -> Path:
    """
    Ensure the cache directory exists.

    Returns:
        Path to the cache directory.
    """
    cache_dir = Config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(file_hash: str, extension: str = ".jpg") -> Path:
    """
    Get the cache path for a thumbnail based on its file hash.

    Args:
        file_hash: SHA256 hash of the original file.
        extension: File extension for the thumbnail (default .jpg).

    Returns:
        Path to the cached thumbnail file.
    """
    return Config.CACHE_DIR / f"{file_hash}{extension}"


def _load_standard_image(path: Path) -> Image.Image:
    """
    Load a standard image (JPEG, PNG) using Pillow.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        ThumbnailError: If the image cannot be loaded.
    """
    try:
        img = Image.open(path)
        # Handle EXIF orientation
        img = _apply_exif_orientation(img)
        return img
    except Exception as e:
        raise ThumbnailError(f"Failed to load image {path}: {e}") from e


def _load_heic_image(path: Path) -> Image.Image:
    """
    Load a HEIC/HEIF image.

    Args:
        path: Path to the HEIC/HEIF file.

    Returns:
        PIL Image object.

    Raises:
        UnsupportedFormatError: If pillow-heif is not available.
        ThumbnailError: If the image cannot be loaded.
    """
    if not HEIF_AVAILABLE:
        raise UnsupportedFormatError(
            f"HEIC/HEIF support requires pillow-heif: {path}"
        )

    try:
        # pillow-heif registers itself with PIL, so we can use Image.open
        img = Image.open(path)
        img = _apply_exif_orientation(img)
        return img
    except Exception as e:
        raise ThumbnailError(f"Failed to load HEIC image {path}: {e}") from e


def _load_raw_image(path: Path) -> Image.Image:
    """
    Load a RAW image (CR2, NEF, ARW, DNG).

    Args:
        path: Path to the RAW file.

    Returns:
        PIL Image object.

    Raises:
        UnsupportedFormatError: If rawpy is not available.
        ThumbnailError: If the image cannot be loaded.
    """
    if not RAWPY_AVAILABLE:
        raise UnsupportedFormatError(
            f"RAW format support requires rawpy: {path}"
        )

    try:
        with rawpy.imread(str(path)) as raw:
            # Process with default parameters (auto white balance, etc.)
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster processing for thumbnails
                no_auto_bright=False,
            )
        return Image.fromarray(rgb)
    except Exception as e:
        raise ThumbnailError(f"Failed to load RAW image {path}: {e}") from e


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to an image.

    Some cameras store images rotated with an EXIF orientation tag.
    This function applies the rotation so the image displays correctly.

    Args:
        img: PIL Image object.

    Returns:
        Correctly oriented PIL Image object.
    """
    try:
        # Get EXIF data
        exif = img.getexif()
        if not exif:
            return img

        # Find orientation tag
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_key = tag
                break

        if orientation_key is None or orientation_key not in exif:
            return img

        orientation = exif[orientation_key]

        # Apply rotation/flip based on orientation value
        if orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 7:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

        return img

    except Exception as e:
        logger.debug(f"Could not apply EXIF orientation: {e}")
        return img


def _calculate_thumbnail_size(
    original_size: tuple[int, int],
    max_size: int,
) -> tuple[int, int]:
    """
    Calculate thumbnail dimensions preserving aspect ratio.

    Args:
        original_size: Original (width, height) of the image.
        max_size: Maximum dimension for longest edge.

    Returns:
        New (width, height) tuple.
    """
    width, height = original_size

    # If already smaller than max_size, return original dimensions
    if width <= max_size and height <= max_size:
        return original_size

    # Scale based on longest edge
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return (new_width, new_height)


def generate_thumbnail(
    source_path: Path,
    output_path: Optional[Path] = None,
    max_size: Optional[int] = None,
    quality: Optional[int] = None,
) -> Path:
    """
    Generate a thumbnail for an image.

    Supports JPEG, PNG, HEIC/HEIF (with pillow-heif), and RAW formats
    (with rawpy). The thumbnail is always saved as JPEG.

    Args:
        source_path: Path to the source image file.
        output_path: Path to save the thumbnail. If None, generates path
                     based on file hash in cache directory.
        max_size: Maximum dimension for longest edge (default from config).
        quality: JPEG quality 1-100 (default from config).

    Returns:
        Path to the generated thumbnail.

    Raises:
        FileNotFoundError: If source file does not exist.
        UnsupportedFormatError: If format not supported.
        ThumbnailError: If thumbnail generation fails.
    """
    source_path = Path(source_path).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if max_size is None:
        max_size = Config.THUMBNAIL_MAX_SIZE

    if quality is None:
        quality = Config.THUMBNAIL_QUALITY

    suffix = source_path.suffix.lower()

    logger.debug(f"Generating thumbnail for {source_path} (format: {suffix})")

    # Load image based on format
    if suffix in RAW_FORMATS:
        img = _load_raw_image(source_path)
    elif suffix in HEIF_FORMATS:
        img = _load_heic_image(source_path)
    else:
        img = _load_standard_image(source_path)

    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Calculate new size
    new_size = _calculate_thumbnail_size(img.size, max_size)

    # Resize using high-quality resampling
    if new_size != img.size:
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized from {img.size} to {new_size}")

    # Determine output path if not provided
    if output_path is None:
        # Need to compute hash - import here to avoid circular import
        from sims.walker import compute_file_hash
        file_hash = compute_file_hash(source_path)
        output_path = get_cache_path(file_hash)

    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JPEG
    try:
        img.save(output_path, format="JPEG", quality=quality, optimize=True)
        logger.debug(f"Saved thumbnail to {output_path}")
    except Exception as e:
        raise ThumbnailError(f"Failed to save thumbnail: {e}") from e
    finally:
        img.close()

    return output_path


def generate_thumbnail_for_hash(
    source_path: Path,
    file_hash: str,
    max_size: Optional[int] = None,
    quality: Optional[int] = None,
) -> Path:
    """
    Generate a thumbnail with a known file hash.

    This is more efficient than generate_thumbnail() when the hash
    is already computed (avoids recomputing).

    Args:
        source_path: Path to the source image file.
        file_hash: Pre-computed SHA256 hash of the file.
        max_size: Maximum dimension for longest edge.
        quality: JPEG quality 1-100.

    Returns:
        Path to the generated thumbnail.
    """
    output_path = get_cache_path(file_hash)
    return generate_thumbnail(
        source_path=source_path,
        output_path=output_path,
        max_size=max_size,
        quality=quality,
    )


def get_thumbnail_info(thumbnail_path: Path) -> dict:
    """
    Get information about a thumbnail file.

    Args:
        thumbnail_path: Path to the thumbnail.

    Returns:
        Dictionary with thumbnail info:
            - width: Image width in pixels
            - height: Image height in pixels
            - size_bytes: File size in bytes
            - format: Image format (JPEG)
    """
    thumbnail_path = Path(thumbnail_path)

    if not thumbnail_path.exists():
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")

    with Image.open(thumbnail_path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "size_bytes": thumbnail_path.stat().st_size,
            "format": img.format,
        }
