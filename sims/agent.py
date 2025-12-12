# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Ingestion agent for SImS.

Orchestrates the full image ingestion pipeline: discovery, thumbnail generation,
metadata extraction, vision model description, and vector embedding.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from sims.config import Config
from sims import db
from sims import walker
from sims import thumbnail
from sims import metadata
from sims import vision
from sims import embeddings
from sims import vectorstore

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""
    total: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    missing: int = 0  # Files that no longer exist

    @property
    def pending(self) -> int:
        return self.total - self.processed - self.failed - self.skipped - self.missing

    def summary(self) -> str:
        """Return a human-readable summary of the statistics."""
        parts = [f"processed={self.processed}"]
        if self.failed:
            parts.append(f"failed={self.failed}")
        if self.skipped:
            parts.append(f"skipped={self.skipped}")
        if self.missing:
            parts.append(f"missing={self.missing}")
        return ", ".join(parts)


@dataclass
class JobStatus:
    """Status information for an ingestion job."""
    job_id: int
    root_path: str
    status: str
    total_files: int
    processed_files: int
    failed_files: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files + self.failed_files) / self.total_files * 100


class IngestionAgent:
    """
    Main ingestion orchestration agent.

    Processes directories of images through the full pipeline:
    1. Discovery - Find and register new/changed images
    2. Thumbnail - Generate thumbnails for processing
    3. Metadata - Extract EXIF data
    4. Describe - Generate descriptions using vision model
    5. Embed - Create vector embeddings and store in ChromaDB
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        vision_model: Optional[str] = None,
    ):
        """
        Initialize the ingestion agent.

        Args:
            batch_size: Number of images to process per batch.
            on_progress: Optional callback for progress updates.
            vision_model: Vision model to use (default from config).
        """
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.on_progress = on_progress
        self.vision_model = vision_model
        self._shutdown_requested = False
        self._current_job_id: Optional[int] = None

        # Set model override if provided
        if vision_model:
            Config.set_vision_model(vision_model)

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, requesting graceful shutdown...")
            self._shutdown_requested = True

        # Only set handlers if running in main thread
        try:
            signal.signal(signal.SIGINT, handle_shutdown)
            signal.signal(signal.SIGTERM, handle_shutdown)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass

    def _report_progress(self, stats: ProcessingStats) -> None:
        """Report progress via callback if set."""
        if self.on_progress:
            self.on_progress(stats)

    def _validate_file_exists(self, image: dict) -> bool:
        """
        Validate that the original image file exists.

        Args:
            image: Image record dict with 'original_path' and 'id'.

        Returns:
            True if file exists, False otherwise.

        Side effect:
            If file doesn't exist, marks the image with an error in the database.
        """
        original_path = Path(image["original_path"])
        if not original_path.exists():
            error_msg = f"Original file not found: {original_path}"
            logger.warning(error_msg)
            db.mark_image_error(image["id"], error_msg)
            return False
        return True

    async def process_directory(
        self,
        root_path: Path,
        recursive: bool = True,
    ) -> int:
        """
        Process all images in a directory.

        Args:
            root_path: Path to directory to process.
            recursive: Whether to process subdirectories.

        Returns:
            Job ID for tracking progress.
        """
        root_path = Path(root_path).resolve()

        if not root_path.is_dir():
            raise ValueError(f"Not a directory: {root_path}")

        logger.info(f"Starting ingestion from {root_path} (recursive={recursive})")

        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Clean up any stale running jobs from previous crashes
        interrupted_count = db.mark_running_jobs_interrupted()
        if interrupted_count > 0:
            logger.info(f"Cleaned up {interrupted_count} interrupted jobs from previous run")

        # Count total images
        total_images = walker.count_images(root_path, recursive=recursive)
        logger.info(f"Found {total_images} images to process")

        if total_images == 0:
            logger.info("No images found, nothing to do")
            return 0

        # Create job
        job_id = db.create_job(str(root_path), total_images)
        self._current_job_id = job_id
        logger.info(f"Created job {job_id}")

        stats = ProcessingStats(total=total_images)

        try:
            # Discovery stage
            logger.info("Stage 1/5: Discovery")
            discovered_ids = await self._discover_stage(root_path, recursive, stats)

            if self._shutdown_requested:
                logger.info("Shutdown requested after discovery")
                db.complete_job(job_id, "cancelled")
                return job_id

            # Process in batches through remaining stages
            await self._process_pipeline(job_id, stats)

            # Complete job
            final_status = "cancelled" if self._shutdown_requested else "completed"
            db.complete_job(job_id, final_status)
            logger.info(f"Job {job_id} {final_status}: {stats.processed} processed, {stats.failed} failed")

        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {e}")
            db.complete_job(job_id, "failed")
            raise
        finally:
            self._current_job_id = None

        return job_id

    async def _discover_stage(
        self,
        root_path: Path,
        recursive: bool,
        stats: ProcessingStats,
    ) -> list[int]:
        """
        Discover and register new/changed images.

        Returns:
            List of image IDs that were registered.
        """
        discovered_ids = []

        for image_info in walker.discover_images(root_path, recursive=recursive):
            if self._shutdown_requested:
                break

            path = image_info["path"]
            file_hash = image_info["hash"]

            # Check if already registered
            existing = db.get_image_by_path(str(path))

            if existing:
                if existing["file_hash"] == file_hash:
                    # Unchanged, skip
                    stats.skipped += 1
                    continue
                else:
                    # File changed, will reprocess
                    # Delete old vector embedding if exists
                    try:
                        vectorstore.delete_embedding(existing["id"])
                    except vectorstore.EmbeddingNotFoundError:
                        pass
                    # Delete old record to create fresh one
                    db.delete_image(existing["id"])

            # Register new image
            try:
                image_id = db.register_image(
                    original_path=str(path),
                    file_hash=file_hash,
                    file_size=image_info["size"],
                    format=image_info["format"],
                )
                discovered_ids.append(image_id)
                logger.debug(f"Registered image {image_id}: {path}")
            except Exception as e:
                logger.error(f"Failed to register {path}: {e}")
                stats.failed += 1

        logger.info(f"Discovered {len(discovered_ids)} new images, skipped {stats.skipped} unchanged")
        return discovered_ids

    async def _process_pipeline(self, job_id: int, stats: ProcessingStats) -> None:
        """Process images through thumbnail, metadata, describe, embed stages."""

        stages = [
            ("thumbnail", "Stage 2/5: Thumbnails", self._thumbnail_stage),
            ("describe", "Stage 3/5: Metadata + Description", self._describe_stage),
            ("embed", "Stage 4/5: Embedding", self._embed_stage),
        ]

        for stage_name, stage_label, stage_func in stages:
            if self._shutdown_requested:
                break

            logger.info(stage_label)

            while True:
                if self._shutdown_requested:
                    break

                # Get batch of pending images for this stage
                pending = db.get_pending_images(stage_name, limit=self.batch_size)

                if not pending:
                    break

                # Process batch
                for image in pending:
                    if self._shutdown_requested:
                        break

                    try:
                        await stage_func(image)
                        # Only count as processed when fully embedded
                        if stage_name == "embed":
                            stats.processed += 1
                    except FileNotFoundError as e:
                        # File no longer exists - already marked as error in _validate_file_exists
                        logger.warning(f"Image {image['id']} file missing: {e}")
                        stats.missing += 1
                    except Exception as e:
                        logger.error(f"Error processing image {image['id']} at {stage_name}: {e}")
                        db.mark_image_error(image["id"], str(e))
                        stats.failed += 1

                    # Update job progress (only count fully processed images)
                    db.update_job_progress(job_id, stats.processed, stats.failed)
                    self._report_progress(stats)

    async def _thumbnail_stage(self, image: dict) -> None:
        """Generate thumbnail for an image."""
        # Early validation - ensure file exists
        if not self._validate_file_exists(image):
            raise FileNotFoundError(f"Original file not found: {image['original_path']}")

        image_id = image["id"]
        original_path = Path(image["original_path"])
        file_hash = image["file_hash"]

        # Generate thumbnail using hash for cache path
        thumbnail_path = thumbnail.generate_thumbnail_for_hash(
            source_path=original_path,
            file_hash=file_hash,
        )

        # Update database
        db.update_image_thumbnail(image_id, str(thumbnail_path))
        logger.debug(f"Generated thumbnail for {image_id}")

    async def _describe_stage(self, image: dict) -> None:
        """Extract metadata and generate description for an image."""
        # Early validation - need original for metadata extraction
        if not self._validate_file_exists(image):
            raise FileNotFoundError(f"Original file not found: {image['original_path']}")

        image_id = image["id"]
        original_path = Path(image["original_path"])
        cached_path = Path(image["cached_path"]) if image["cached_path"] else None

        # Extract EXIF metadata from original
        meta = metadata.extract_metadata(original_path)

        # Update metadata in database
        db.update_image_metadata(
            image_id=image_id,
            date_taken=meta.get("date_taken"),
            gps_lat=meta.get("gps_lat"),
            gps_lon=meta.get("gps_lon"),
            camera_make=meta.get("camera_make"),
            camera_model=meta.get("camera_model"),
        )

        # Generate description from thumbnail (faster than full image)
        image_to_describe = cached_path if cached_path and cached_path.exists() else original_path

        description_result = await vision.describe_image(image_to_describe)

        # Update description in database
        db.update_image_description(
            image_id=image_id,
            description=description_result["full_description"],
            tags=description_result["tags"],
            mood=description_result["mood"],
        )
        logger.debug(f"Described image {image_id}")

    async def _embed_stage(self, image: dict) -> None:
        """Generate embedding and store in vector database."""
        image_id = image["id"]
        description = image["description"]

        if not description:
            raise ValueError(f"Image {image_id} has no description")

        # Generate embedding
        embedding = await embeddings.embed_text(description)

        # Prepare metadata for vector store
        vs_metadata = {
            "original_path": image["original_path"],
            "date_taken": image["date_taken"].isoformat() if image["date_taken"] else "",
            "has_gps": image["gps_lat"] is not None and image["gps_lon"] is not None,
        }

        # Store in vector database
        vectorstore.upsert_embedding(
            image_id=image_id,
            embedding=embedding,
            description=description,
            metadata=vs_metadata,
        )

        # Mark as embedded in SQLite
        db.mark_image_embedded(image_id)
        logger.debug(f"Embedded image {image_id}")

    async def process_single_image(self, image_path: Path) -> dict:
        """
        Process a single image through the full pipeline.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with processing results including image_id, description, etc.
        """
        image_path = Path(image_path).resolve()

        if not image_path.is_file():
            raise ValueError(f"Not a file: {image_path}")

        if not walker.is_supported_format(image_path):
            raise ValueError(f"Unsupported format: {image_path.suffix}")

        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Compute hash
        file_hash = walker.compute_file_hash(image_path)

        # Check if already exists
        existing = db.get_image_by_path(str(image_path))
        if existing and existing["file_hash"] == file_hash and existing["embedded_at"]:
            logger.info(f"Image already processed: {image_path}")
            return {
                "image_id": existing["id"],
                "status": "already_processed",
                "description": existing["description"],
            }

        # Register or get image
        if existing:
            # Hash changed or not fully processed, delete old
            try:
                vectorstore.delete_embedding(existing["id"])
            except vectorstore.EmbeddingNotFoundError:
                pass
            db.delete_image(existing["id"])

        # Register new
        image_id = db.register_image(
            original_path=str(image_path),
            file_hash=file_hash,
            file_size=image_path.stat().st_size,
            format=walker.get_format(image_path),
        )

        # Get full image record
        image = db.get_image_by_id(image_id)

        # Process through all stages
        await self._thumbnail_stage(image)
        image = db.get_image_by_id(image_id)  # Refresh

        await self._describe_stage(image)
        image = db.get_image_by_id(image_id)  # Refresh

        await self._embed_stage(image)
        image = db.get_image_by_id(image_id)  # Refresh

        logger.info(f"Processed single image: {image_path}")

        return {
            "image_id": image_id,
            "status": "processed",
            "description": image["description"],
            "tags": image["tags"],
            "mood": image["mood"],
            "cached_path": image["cached_path"],
        }

    async def resume_incomplete(self) -> ProcessingStats:
        """
        Resume processing of incomplete images.

        Finds and processes all images where embedded_at IS NULL.

        Returns:
            Processing statistics.
        """
        logger.info("Resuming incomplete processing")

        # Initialize stores
        db.init_db()
        vectorstore.init_vectorstore()

        # Clean up any stale running jobs from previous crashes
        interrupted_count = db.mark_running_jobs_interrupted()
        if interrupted_count > 0:
            logger.info(f"Cleaned up {interrupted_count} interrupted jobs from previous run")

        # Count pending at each stage
        pending_thumbnail = len(db.get_pending_images("thumbnail"))
        pending_describe = len(db.get_pending_images("describe"))
        pending_embed = len(db.get_pending_images("embed"))

        total_pending = pending_thumbnail + pending_describe + pending_embed

        if total_pending == 0:
            logger.info("No incomplete images to process")
            return ProcessingStats()

        logger.info(f"Found {total_pending} incomplete images "
                   f"(thumbnail: {pending_thumbnail}, describe: {pending_describe}, embed: {pending_embed})")

        stats = ProcessingStats(total=total_pending)

        # Create a pseudo-job for tracking
        job_id = db.create_job("resume", total_pending)
        self._current_job_id = job_id

        try:
            await self._process_pipeline(job_id, stats)

            final_status = "cancelled" if self._shutdown_requested else "completed"
            db.complete_job(job_id, final_status)

        except Exception as e:
            logger.error(f"Resume failed with error: {e}")
            db.complete_job(job_id, "failed")
            raise
        finally:
            self._current_job_id = None

        logger.info(f"Resume completed: {stats.processed} processed, {stats.failed} failed")
        return stats

    def get_job_status(self, job_id: int) -> Optional[JobStatus]:
        """
        Get status of an ingestion job.

        Args:
            job_id: The job ID.

        Returns:
            JobStatus object or None if job not found.
        """
        job = db.get_job(job_id)
        if not job:
            return None

        return JobStatus(
            job_id=job["id"],
            root_path=job["root_path"],
            status=job["status"],
            total_files=job["total_files"],
            processed_files=job["processed_files"],
            failed_files=job["failed_files"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
        )

    async def cancel_job(self, job_id: int) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if job was cancelled, False if not found or already completed.
        """
        job = db.get_job(job_id)
        if not job:
            return False

        if job["status"] != "running":
            return False

        # If this is the current job, request shutdown
        if self._current_job_id == job_id:
            self._shutdown_requested = True
        else:
            # Just mark as cancelled in DB
            db.complete_job(job_id, "cancelled")

        return True


async def search_images(
    query: str,
    limit: int = 20,
    filters: Optional[dict] = None,
    min_score: Optional[float] = None,
) -> list[dict]:
    """
    Search for images using natural language query.

    Args:
        query: Natural language search query.
        limit: Maximum number of results.
        filters: Optional filters (date_after, date_before, has_gps).
        min_score: Optional minimum similarity score threshold (0.0 to 1.0).
            Results with scores below this threshold are excluded.

    Returns:
        List of search results with image info and scores.
    """
    # Initialize stores
    db.init_db()
    vectorstore.init_vectorstore()

    # Generate query embedding
    query_embedding = await embeddings.embed_text(query)

    # Search vector store
    results = vectorstore.search(
        query_embedding=query_embedding,
        limit=limit,
        filters=filters,
        min_score=min_score,
    )

    # Enrich results with full image data from SQLite
    enriched_results = []
    for result in results:
        image = db.get_image_by_id(result["id"])
        if image:
            enriched_results.append({
                "id": result["id"],
                "score": result["score"],
                "original_path": image["original_path"],
                "cached_path": image["cached_path"],
                "description": image["description"],
                "tags": image["tags"],
                "mood": image["mood"],
                "date_taken": image["date_taken"],
                "gps_lat": image["gps_lat"],
                "gps_lon": image["gps_lon"],
                "camera_make": image["camera_make"],
                "camera_model": image["camera_model"],
            })

    return enriched_results


def get_system_stats() -> dict:
    """
    Get system-wide statistics.

    Returns:
        Dict with statistics about images, processing status, storage, etc.
    """
    # Initialize DB (vectorstore optional)
    db.init_db()

    # Get DB stats
    db_stats = db.get_stats()

    # Get vector store stats if available
    try:
        vectorstore.init_vectorstore()
        vs_stats = vectorstore.get_collection_stats()
    except Exception:
        vs_stats = {"total_embeddings": 0}

    # Calculate storage used by thumbnails
    cache_size = 0
    if Config.CACHE_DIR.exists():
        for f in Config.CACHE_DIR.iterdir():
            if f.is_file():
                cache_size += f.stat().st_size

    return {
        **db_stats,
        "embeddings_count": vs_stats["total_embeddings"],
        "cache_size_bytes": cache_size,
        "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        "cache_dir": str(Config.CACHE_DIR),
        "db_path": str(Config.DB_PATH),
        "chroma_path": str(Config.CHROMA_PATH),
    }


def cleanup_orphaned_images() -> dict:
    """
    Remove database entries for images whose files no longer exist.

    Scans all registered images and removes entries where the original
    file is missing. Also removes corresponding vector embeddings.

    Returns:
        Dict with cleanup results:
            - total_checked: Number of images checked
            - orphaned_count: Number of orphaned entries found
            - deleted_db: Number of DB records deleted
            - deleted_vectors: Number of vector embeddings deleted
    """
    # Initialize stores
    db.init_db()
    vectorstore.init_vectorstore()

    all_images = db.get_all_images()

    orphaned_ids = []
    for image in all_images:
        original_path = Path(image["original_path"])
        if not original_path.exists():
            orphaned_ids.append(image["id"])
            logger.debug(f"Orphaned: {original_path}")

    results = {
        "total_checked": len(all_images),
        "orphaned_count": len(orphaned_ids),
        "deleted_db": 0,
        "deleted_vectors": 0,
    }

    if orphaned_ids:
        # Delete from vector store first (may not have all embeddings)
        results["deleted_vectors"] = vectorstore.delete_embeddings_batch(orphaned_ids)
        # Delete from database
        results["deleted_db"] = db.delete_images_batch(orphaned_ids)

    logger.info(f"Cleanup complete: {results['orphaned_count']} orphaned of {results['total_checked']} checked")
    return results
