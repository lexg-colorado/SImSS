# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Directory traversal and file discovery for SImS.

Provides functions for recursively walking directories, computing file hashes,
and detecting new or changed image files.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

from sims.config import Config, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: Path to the file.
        chunk_size: Size of chunks to read at a time (default 64KB).

    Returns:
        Hexadecimal string of the SHA256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_supported_format(path: Path) -> bool:
    """
    Check if a file has a supported image format.

    Args:
        path: Path to the file.

    Returns:
        True if the file extension is in SUPPORTED_FORMATS.
    """
    return path.suffix.lower() in SUPPORTED_FORMATS


def get_format(path: Path) -> str:
    """
    Get the normalized format string for an image file.

    Args:
        path: Path to the image file.

    Returns:
        Lowercase format string (e.g., 'jpeg', 'png', 'heic').
    """
    suffix = path.suffix.lower()
    # Normalize common variations
    format_map = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".heic": "heic",
        ".heif": "heif",
        ".cr2": "cr2",
        ".nef": "nef",
        ".arw": "arw",
        ".dng": "dng",
    }
    return format_map.get(suffix, suffix.lstrip("."))


def discover_images(
    root_path: Path,
    recursive: bool = True,
) -> Iterator[dict]:
    """
    Discover image files in a directory.

    Walks the directory tree (optionally recursively) and yields information
    about each supported image file found.

    Args:
        root_path: Root directory to search.
        recursive: If True, search subdirectories recursively.

    Yields:
        Dictionary for each image found:
            - path: Path object to the file
            - hash: SHA256 hash of the file contents
            - size: File size in bytes
            - format: Normalized format string (e.g., 'jpeg', 'png')

    Raises:
        NotADirectoryError: If root_path is not a directory.
    """
    root_path = Path(root_path).resolve()

    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    logger.info(f"Discovering images in {root_path} (recursive={recursive})")

    if recursive:
        file_iterator = root_path.rglob("*")
    else:
        file_iterator = root_path.glob("*")

    discovered_count = 0
    skipped_count = 0

    for file_path in file_iterator:
        # Skip directories
        if not file_path.is_file():
            continue

        # Skip unsupported formats
        if not is_supported_format(file_path):
            skipped_count += 1
            continue

        try:
            stat = file_path.stat()
            file_hash = compute_file_hash(file_path)

            discovered_count += 1
            logger.debug(f"Found: {file_path}")

            yield {
                "path": file_path,
                "hash": file_hash,
                "size": stat.st_size,
                "format": get_format(file_path),
            }

        except PermissionError:
            logger.warning(f"Permission denied: {file_path}")
        except OSError as e:
            logger.warning(f"Error reading {file_path}: {e}")

    logger.info(
        f"Discovery complete: {discovered_count} images found, {skipped_count} files skipped"
    )


def get_new_or_changed_images(
    root_path: Path,
    db_path: Optional[Path] = None,
    recursive: bool = True,
) -> Iterator[dict]:
    """
    Discover images that are new or have changed since last scan.

    Compares discovered files against the database to find:
    - New files (path not in database)
    - Changed files (path exists but hash differs)

    Args:
        root_path: Root directory to search.
        db_path: Path to the database (uses default if not provided).
        recursive: If True, search subdirectories recursively.

    Yields:
        Dictionary for each new/changed image:
            - path: Path object to the file
            - hash: SHA256 hash of the file contents
            - size: File size in bytes
            - format: Normalized format string
            - status: 'new' or 'changed'
            - existing_id: Image ID if changed (None if new)
    """
    # Import here to avoid circular import
    from sims.db import get_image_by_path

    root_path = Path(root_path).resolve()
    logger.info(f"Scanning for new/changed images in {root_path}")

    new_count = 0
    changed_count = 0
    unchanged_count = 0

    for image_info in discover_images(root_path, recursive=recursive):
        path_str = str(image_info["path"])

        # Check if image exists in database
        existing = get_image_by_path(path_str, db_path=db_path)

        if existing is None:
            # New image
            new_count += 1
            yield {
                **image_info,
                "status": "new",
                "existing_id": None,
            }

        elif existing["file_hash"] != image_info["hash"]:
            # Changed image (hash differs)
            changed_count += 1
            logger.debug(f"File changed: {path_str}")
            yield {
                **image_info,
                "status": "changed",
                "existing_id": existing["id"],
            }

        else:
            # Unchanged - skip
            unchanged_count += 1
            logger.debug(f"Unchanged: {path_str}")

    logger.info(
        f"Scan complete: {new_count} new, {changed_count} changed, "
        f"{unchanged_count} unchanged"
    )


def count_images(root_path: Path, recursive: bool = True) -> int:
    """
    Count the number of supported image files in a directory.

    This is faster than discover_images() as it doesn't compute hashes.

    Args:
        root_path: Root directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        Number of supported image files found.
    """
    root_path = Path(root_path).resolve()

    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    if recursive:
        file_iterator = root_path.rglob("*")
    else:
        file_iterator = root_path.glob("*")

    count = sum(
        1 for f in file_iterator
        if f.is_file() and is_supported_format(f)
    )

    logger.debug(f"Counted {count} images in {root_path}")
    return count
