# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
SQLite database operations for SImS.

Provides the data layer for tracking images and ingestion jobs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from sims.config import Config

logger = logging.getLogger(__name__)

# Register datetime adapter/converter for Python 3.12+ compatibility
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

# SQL Schema
SCHEMA = """
-- Core image registry
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_path TEXT UNIQUE NOT NULL,
    cached_path TEXT,
    file_hash TEXT NOT NULL,
    file_size INTEGER,
    format TEXT,

    -- EXIF metadata
    date_taken TIMESTAMP,
    gps_lat REAL,
    gps_lon REAL,
    camera_make TEXT,
    camera_model TEXT,

    -- LLM outputs
    description TEXT,
    tags TEXT,
    mood TEXT,

    -- Processing state
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    thumbnail_at TIMESTAMP,
    described_at TIMESTAMP,
    embedded_at TIMESTAMP,

    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- Ingestion jobs (for tracking batch progress)
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_files INTEGER,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_images_date ON images(date_taken);
CREATE INDEX IF NOT EXISTS idx_images_format ON images(format);
CREATE INDEX IF NOT EXISTS idx_images_processing ON images(embedded_at);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
"""


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection.

    Args:
        db_path: Optional path to database. Uses Config.DB_PATH if not provided.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    if db_path is None:
        db_path = Config.DB_PATH

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.

    Args:
        db_path: Optional path to database.

    Yields:
        sqlite3.Connection
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database schema.

    Args:
        db_path: Optional path to database. Uses Config.DB_PATH if not provided.
    """
    if db_path is None:
        db_path = Config.DB_PATH

    logger.info(f"Initializing database at {db_path}")

    with get_db(db_path) as conn:
        conn.executescript(SCHEMA)

    logger.info("Database initialized successfully")


# =============================================================================
# Image Operations
# =============================================================================

def register_image(
    original_path: str,
    file_hash: str,
    file_size: int,
    format: str,
    db_path: Optional[Path] = None,
) -> int:
    """
    Register a new image in the database.

    Args:
        original_path: Absolute path to the original image file.
        file_hash: SHA256 hash of the file contents.
        file_size: Size of the file in bytes.
        format: Image format (e.g., 'jpeg', 'png', 'heic').
        db_path: Optional path to database.

    Returns:
        The ID of the newly registered image.

    Raises:
        sqlite3.IntegrityError: If the image path already exists.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO images (original_path, file_hash, file_size, format)
            VALUES (?, ?, ?, ?)
            """,
            (original_path, file_hash, file_size, format),
        )
        return cursor.lastrowid


def get_image_by_path(path: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get an image record by its original path.

    Args:
        path: The original path of the image.
        db_path: Optional path to database.

    Returns:
        Image record as a dictionary, or None if not found.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM images WHERE original_path = ?",
            (path,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_image_by_id(image_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get an image record by its ID.

    Args:
        image_id: The image ID.
        db_path: Optional path to database.

    Returns:
        Image record as a dictionary, or None if not found.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM images WHERE id = ?",
            (image_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_image_by_hash(file_hash: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get an image record by its file hash.

    Args:
        file_hash: The SHA256 hash of the file.
        db_path: Optional path to database.

    Returns:
        Image record as a dictionary, or None if not found.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM images WHERE file_hash = ?",
            (file_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_image_thumbnail(
    image_id: int,
    cached_path: str,
    db_path: Optional[Path] = None,
) -> None:
    """
    Update the cached thumbnail path for an image.

    Args:
        image_id: The image ID.
        cached_path: Path to the cached thumbnail.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET cached_path = ?, thumbnail_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (cached_path, image_id),
        )


def update_image_metadata(
    image_id: int,
    date_taken: Optional[datetime] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    camera_make: Optional[str] = None,
    camera_model: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Update EXIF metadata for an image.

    Args:
        image_id: The image ID.
        date_taken: When the photo was taken.
        gps_lat: GPS latitude.
        gps_lon: GPS longitude.
        camera_make: Camera manufacturer.
        camera_model: Camera model.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET date_taken = ?, gps_lat = ?, gps_lon = ?,
                camera_make = ?, camera_model = ?
            WHERE id = ?
            """,
            (date_taken, gps_lat, gps_lon, camera_make, camera_model, image_id),
        )


def update_image_description(
    image_id: int,
    description: str,
    tags: list[str],
    mood: str,
    db_path: Optional[Path] = None,
) -> None:
    """
    Update the LLM-generated description for an image.

    Args:
        image_id: The image ID.
        description: The generated description text.
        tags: List of extracted tags.
        mood: The mood/atmosphere of the image.
        db_path: Optional path to database.
    """
    tags_json = json.dumps(tags)
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET description = ?, tags = ?, mood = ?, described_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (description, tags_json, mood, image_id),
        )


def mark_image_embedded(image_id: int, db_path: Optional[Path] = None) -> None:
    """
    Mark an image as having been embedded in the vector store.

    Args:
        image_id: The image ID.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET embedded_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (image_id,),
        )


def mark_image_error(
    image_id: int,
    error_message: str,
    db_path: Optional[Path] = None,
) -> None:
    """
    Mark an image as having encountered an error during processing.

    Args:
        image_id: The image ID.
        error_message: Description of the error.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET error_message = ?, retry_count = retry_count + 1
            WHERE id = ?
            """,
            (error_message, image_id),
        )


def clear_image_error(image_id: int, db_path: Optional[Path] = None) -> None:
    """
    Clear the error state for an image.

    Args:
        image_id: The image ID.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE images
            SET error_message = NULL
            WHERE id = ?
            """,
            (image_id,),
        )


def reset_errors(db_path: Optional[Path] = None) -> int:
    """
    Reset error state for all images that have errors.

    Clears error_message and resets retry_count to 0 for all images
    that have not completed processing (embedded_at IS NULL).
    This allows failed images to be retried.

    Args:
        db_path: Optional path to database.

    Returns:
        Number of images that were reset.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE images
            SET error_message = NULL, retry_count = 0
            WHERE error_message IS NOT NULL AND embedded_at IS NULL
            """
        )
        count = cursor.rowcount
        logger.info(f"Reset errors for {count} images")
        return count


def get_pending_images(
    stage: str,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Get images pending processing at a specific stage.

    Args:
        stage: One of 'thumbnail', 'describe', 'embed'.
        limit: Maximum number of images to return.
        db_path: Optional path to database.

    Returns:
        List of image records pending at the specified stage.

    Raises:
        ValueError: If stage is not valid.
    """
    stage_conditions = {
        "thumbnail": "thumbnail_at IS NULL",
        "describe": "thumbnail_at IS NOT NULL AND described_at IS NULL",
        "embed": "described_at IS NOT NULL AND embedded_at IS NULL",
    }

    if stage not in stage_conditions:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {list(stage_conditions.keys())}")

    condition = stage_conditions[stage]
    query = f"""
        SELECT * FROM images
        WHERE {condition} AND (error_message IS NULL OR retry_count < ?)
        ORDER BY discovered_at ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    with get_db(db_path) as conn:
        cursor = conn.execute(query, (Config.MAX_RETRIES,))
        return [dict(row) for row in cursor.fetchall()]


def delete_image(image_id: int, db_path: Optional[Path] = None) -> bool:
    """
    Delete an image record from the database.

    Args:
        image_id: The image ID.
        db_path: Optional path to database.

    Returns:
        True if an image was deleted, False otherwise.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM images WHERE id = ?",
            (image_id,),
        )
        return cursor.rowcount > 0


def get_all_images(db_path: Optional[Path] = None) -> list[dict]:
    """
    Get all image records from the database.

    Args:
        db_path: Optional path to database.

    Returns:
        List of all image records as dictionaries.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute("SELECT * FROM images ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]


def delete_images_batch(image_ids: list[int], db_path: Optional[Path] = None) -> int:
    """
    Delete multiple image records from the database.

    Args:
        image_ids: List of image IDs to delete.
        db_path: Optional path to database.

    Returns:
        Number of images deleted.
    """
    if not image_ids:
        return 0

    with get_db(db_path) as conn:
        placeholders = ",".join("?" * len(image_ids))
        cursor = conn.execute(
            f"DELETE FROM images WHERE id IN ({placeholders})",
            image_ids,
        )
        count = cursor.rowcount
        logger.info(f"Deleted {count} images from database")
        return count


# =============================================================================
# Job Operations
# =============================================================================

def create_job(
    root_path: str,
    total_files: int,
    db_path: Optional[Path] = None,
) -> int:
    """
    Create a new ingestion job.

    Args:
        root_path: The root directory being processed.
        total_files: Total number of files to process.
        db_path: Optional path to database.

    Returns:
        The ID of the newly created job.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO ingestion_jobs (root_path, total_files)
            VALUES (?, ?)
            """,
            (root_path, total_files),
        )
        return cursor.lastrowid


def update_job_progress(
    job_id: int,
    processed: int,
    failed: int,
    db_path: Optional[Path] = None,
) -> None:
    """
    Update the progress of an ingestion job.

    Args:
        job_id: The job ID.
        processed: Number of files processed so far.
        failed: Number of files that failed processing.
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET processed_files = ?, failed_files = ?
            WHERE id = ?
            """,
            (processed, failed, job_id),
        )


def complete_job(
    job_id: int,
    status: str = "completed",
    db_path: Optional[Path] = None,
) -> None:
    """
    Mark an ingestion job as complete.

    Args:
        job_id: The job ID.
        status: Final status ('completed', 'failed', 'cancelled').
        db_path: Optional path to database.
    """
    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET completed_at = CURRENT_TIMESTAMP, status = ?
            WHERE id = ?
            """,
            (status, job_id),
        )


def get_job(job_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get an ingestion job by ID.

    Args:
        job_id: The job ID.
        db_path: Optional path to database.

    Returns:
        Job record as a dictionary, or None if not found.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE id = ?",
            (job_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_active_jobs(db_path: Optional[Path] = None) -> list[dict]:
    """
    Get all active (running) ingestion jobs.

    Args:
        db_path: Optional path to database.

    Returns:
        List of active job records.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE status = 'running' ORDER BY started_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


def mark_running_jobs_interrupted(db_path: Optional[Path] = None) -> int:
    """
    Mark all running jobs as 'interrupted'.

    This should be called on startup to clean up jobs that were
    left in 'running' state from previous crashes or interrupts.

    Args:
        db_path: Optional path to database.

    Returns:
        Number of jobs marked as interrupted.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'interrupted', completed_at = CURRENT_TIMESTAMP
            WHERE status = 'running'
            """
        )
        count = cursor.rowcount
        if count > 0:
            logger.warning(f"Marked {count} stale running jobs as interrupted")
        return count


# =============================================================================
# Statistics
# =============================================================================

def get_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get statistics about the image database.

    Args:
        db_path: Optional path to database.

    Returns:
        Dictionary containing various statistics.
    """
    with get_db(db_path) as conn:
        stats = {}

        # Total images
        cursor = conn.execute("SELECT COUNT(*) as count FROM images")
        stats["total_images"] = cursor.fetchone()["count"]

        # Fully processed (embedded)
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE embedded_at IS NOT NULL"
        )
        stats["processed"] = cursor.fetchone()["count"]

        # Pending at each stage
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE thumbnail_at IS NULL"
        )
        stats["pending_thumbnail"] = cursor.fetchone()["count"]

        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE thumbnail_at IS NOT NULL AND described_at IS NULL"
        )
        stats["pending_describe"] = cursor.fetchone()["count"]

        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE described_at IS NOT NULL AND embedded_at IS NULL"
        )
        stats["pending_embed"] = cursor.fetchone()["count"]

        # Total pending
        stats["pending"] = (
            stats["pending_thumbnail"] +
            stats["pending_describe"] +
            stats["pending_embed"]
        )

        # Errored images
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE error_message IS NOT NULL"
        )
        stats["errors"] = cursor.fetchone()["count"]

        # By format
        cursor = conn.execute(
            "SELECT format, COUNT(*) as count FROM images GROUP BY format"
        )
        stats["by_format"] = {row["format"]: row["count"] for row in cursor.fetchall()}

        # Images with GPS
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM images WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
        )
        stats["with_gps"] = cursor.fetchone()["count"]

        # Total storage (file sizes)
        cursor = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) as total FROM images"
        )
        stats["total_size_bytes"] = cursor.fetchone()["total"]

        # Job stats
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM ingestion_jobs WHERE status = 'running'"
        )
        stats["active_jobs"] = cursor.fetchone()["count"]

        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM ingestion_jobs"
        )
        stats["total_jobs"] = cursor.fetchone()["count"]

        return stats


# =============================================================================
# CLI Support
# =============================================================================

def main() -> None:
    """Command-line interface for database operations."""
    import sys

    Config.configure_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m sims.db <command>")
        print("\nCommands:")
        print("  init    Initialize the database")
        print("  stats   Show database statistics")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        Config.ensure_directories()
        init_db()
        print(f"Database initialized at {Config.DB_PATH}")
    elif command == "stats":
        if not Config.DB_PATH.exists():
            print(f"Database not found at {Config.DB_PATH}")
            print("Run 'python -m sims.db init' first.")
            sys.exit(1)
        stats = get_stats()
        print("\nSImS Database Statistics")
        print("=" * 40)
        print(f"Total images:      {stats['total_images']}")
        print(f"Processed:         {stats['processed']}")
        print(f"Pending:           {stats['pending']}")
        print(f"  - Thumbnail:     {stats['pending_thumbnail']}")
        print(f"  - Describe:      {stats['pending_describe']}")
        print(f"  - Embed:         {stats['pending_embed']}")
        print(f"Errors:            {stats['errors']}")
        print(f"With GPS:          {stats['with_gps']}")
        print(f"Total size:        {stats['total_size_bytes'] / (1024*1024):.2f} MB")
        print(f"\nBy format:")
        for fmt, count in stats.get("by_format", {}).items():
            print(f"  - {fmt or 'unknown'}: {count}")
        print(f"\nJobs: {stats['active_jobs']} active / {stats['total_jobs']} total")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
