# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
CLI entry point for SImS.

Provides command-line interface for image ingestion, search, and management.

Usage:
    python -m sims <command> [options]

Commands:
    init            Initialize databases
    ingest <path>   Start ingestion from directory
    resume          Resume incomplete processing
    search <query>  Semantic search for images
    stats           Show system statistics
    status <job>    Check job status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sims.config import Config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_datetime(dt: Optional[datetime]) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize databases."""
    from sims import db
    from sims import vectorstore

    print("Initializing SImS...")

    # Ensure directories exist
    Config.ensure_directories()
    print(f"  Cache directory: {Config.CACHE_DIR}")
    print(f"  Data directory: {Config.DATA_DIR}")

    # Initialize SQLite
    db.init_db()
    print(f"  Database: {Config.DB_PATH}")

    # Initialize ChromaDB
    vectorstore.init_vectorstore()
    print(f"  Vector store: {Config.CHROMA_PATH}")

    print("\nInitialization complete!")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run ingestion on a directory."""
    from sims.agent import IngestionAgent, ProcessingStats
    from sims import vision

    path = Path(args.path).resolve()

    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        return 1

    if not path.is_dir():
        print(f"Error: Not a directory: {path}", file=sys.stderr)
        return 1

    # Validate vision model if specified
    if args.vision_model:
        print(f"Validating vision model: {args.vision_model}")
        model_available = asyncio.run(
            vision.check_model_available(args.vision_model)
        )
        if not model_available:
            print(
                f"Error: Vision model '{args.vision_model}' not found in Ollama.",
                file=sys.stderr,
            )
            print(f"Pull it with: ollama pull {args.vision_model}", file=sys.stderr)
            return 1

    # Progress callback with enhanced reporting
    last_report = [0]  # Use list to allow mutation in closure
    final_stats = [None]  # Track final stats for end summary

    def on_progress(stats: ProcessingStats):
        final_stats[0] = stats  # Keep reference to latest stats
        # Report every 10 images or on completion
        current = stats.processed + stats.failed + stats.missing
        if current - last_report[0] >= 10 or current == stats.total:
            percent = (current / stats.total * 100) if stats.total > 0 else 0
            # Enhanced output showing all categories
            status_parts = [f"{stats.processed} processed"]
            if stats.failed:
                status_parts.append(f"{stats.failed} failed")
            if stats.missing:
                status_parts.append(f"{stats.missing} missing")
            status_str = ", ".join(status_parts)
            print(f"  Progress: {current}/{stats.total} ({percent:.1f}%) - {status_str}")
            last_report[0] = current

    agent = IngestionAgent(
        batch_size=args.batch_size,
        on_progress=on_progress if not args.quiet else None,
        vision_model=args.vision_model,
    )

    print(f"Starting ingestion from: {path}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Batch size: {args.batch_size}")
    if args.vision_model:
        print(f"  Vision model: {args.vision_model}")
    else:
        print(f"  Vision model: {Config.VISION_MODEL} (default)")
    print()

    try:
        job_id = asyncio.run(agent.process_directory(path, recursive=args.recursive))

        if job_id:
            status = agent.get_job_status(job_id)
            stats = final_stats[0]
            print(f"\nJob {job_id} completed!")
            print(f"  Status: {status.status}")
            print(f"  Processed: {stats.processed if stats else status.processed_files}")
            print(f"  Failed: {stats.failed if stats else status.failed_files}")
            if stats:
                if stats.skipped:
                    print(f"  Skipped (unchanged): {stats.skipped}")
                if stats.missing:
                    print(f"  Missing (file not found): {stats.missing}")
        else:
            print("\nNo images found to process.")

        return 0

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume incomplete processing."""
    from sims.agent import IngestionAgent, ProcessingStats
    from sims import vision

    # Validate vision model if specified
    if args.vision_model:
        print(f"Validating vision model: {args.vision_model}")
        model_available = asyncio.run(
            vision.check_model_available(args.vision_model)
        )
        if not model_available:
            print(
                f"Error: Vision model '{args.vision_model}' not found in Ollama.",
                file=sys.stderr,
            )
            print(f"Pull it with: ollama pull {args.vision_model}", file=sys.stderr)
            return 1

    # Progress callback with enhanced reporting
    def on_progress(stats: ProcessingStats):
        current = stats.processed + stats.failed + stats.missing
        if stats.total > 0:
            percent = current / stats.total * 100
            status_parts = [f"{stats.processed} processed"]
            if stats.failed:
                status_parts.append(f"{stats.failed} failed")
            if stats.missing:
                status_parts.append(f"{stats.missing} missing")
            status_str = ", ".join(status_parts)
            print(f"  Progress: {current}/{stats.total} ({percent:.1f}%) - {status_str}")

    agent = IngestionAgent(
        batch_size=args.batch_size,
        on_progress=on_progress if not args.quiet else None,
        vision_model=args.vision_model,
    )

    print("Resuming incomplete processing...")
    if args.vision_model:
        print(f"  Vision model: {args.vision_model}")
    else:
        print(f"  Vision model: {Config.VISION_MODEL} (default)")

    try:
        stats = asyncio.run(agent.resume_incomplete())

        print(f"\nResume completed!")
        print(f"  Processed: {stats.processed}")
        print(f"  Failed: {stats.failed}")
        if stats.missing:
            print(f"  Missing (file not found): {stats.missing}")

        return 0

    except KeyboardInterrupt:
        print("\n\nResume interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during resume: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for images."""
    from sims.agent import search_images

    query = " ".join(args.query)

    if not query.strip():
        print("Error: Search query cannot be empty", file=sys.stderr)
        return 1

    # Build filters
    filters = {}
    if args.date_after:
        filters["date_after"] = args.date_after
    if args.date_before:
        filters["date_before"] = args.date_before
    if args.has_gps:
        filters["has_gps"] = True

    # Convert min_score percentage to decimal (0-100 -> 0.0-1.0)
    min_score = args.min_score / 100.0 if args.min_score is not None else None

    print(f"Searching for: {query}")
    if filters:
        print(f"  Filters: {filters}")
    if min_score is not None:
        print(f"  Minimum score: {args.min_score}%")
    print()

    try:
        results = asyncio.run(search_images(
            query=query,
            limit=args.limit,
            filters=filters if filters else None,
            min_score=min_score,
        ))

        if not results:
            if min_score is not None:
                print(f"No results found matching '{query}' with score >= {args.min_score}%.")
            else:
                print("No results found.")
            return 0

        print(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            score_pct = result["score"] * 100
            print(f"{i}. [{score_pct:.1f}%] {result['original_path']}")

            if args.verbose:
                print(f"   Description: {result['description'][:100]}...")
                if result["tags"]:
                    tags = result["tags"] if isinstance(result["tags"], list) else json.loads(result["tags"])
                    print(f"   Tags: {', '.join(tags[:5])}")
                if result["date_taken"]:
                    print(f"   Date: {format_datetime(result['date_taken'])}")
                if result["gps_lat"] and result["gps_lon"]:
                    print(f"   GPS: {result['gps_lat']:.4f}, {result['gps_lon']:.4f}")
                print()

        # JSON output option
        if args.json:
            print("\n--- JSON Output ---")
            # Convert datetime objects for JSON serialization
            for r in results:
                if r.get("date_taken"):
                    r["date_taken"] = r["date_taken"].isoformat() if hasattr(r["date_taken"], "isoformat") else str(r["date_taken"])
            print(json.dumps(results, indent=2))

        return 0

    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show system statistics."""
    from sims.agent import get_system_stats

    print("SImS Statistics")
    print("=" * 40)

    try:
        stats = get_system_stats()

        print(f"\nImages:")
        print(f"  Total: {stats.get('total', 0)}")
        print(f"  Processed: {stats.get('processed', 0)}")
        print(f"  Pending: {stats.get('pending', 0)}")
        print(f"  With errors: {stats.get('with_errors', 0)}")

        print(f"\nVector Store:")
        print(f"  Embeddings: {stats.get('embeddings_count', 0)}")

        print(f"\nMetadata:")
        print(f"  With GPS: {stats.get('with_gps', 0)}")
        print(f"  With date: {stats.get('with_date', 0)}")

        print(f"\nFormats:")
        for fmt, count in stats.get("by_format", {}).items():
            print(f"  {fmt}: {count}")

        print(f"\nStorage:")
        print(f"  Cache: {format_size(stats.get('cache_size_bytes', 0))}")
        print(f"  Cache dir: {stats.get('cache_dir', 'N/A')}")
        print(f"  Database: {stats.get('db_path', 'N/A')}")
        print(f"  Vector store: {stats.get('chroma_path', 'N/A')}")

        # JSON output option
        if args.json:
            print("\n--- JSON Output ---")
            print(json.dumps(stats, indent=2, default=str))

        return 0

    except Exception as e:
        print(f"Error getting stats: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Check job status."""
    from sims.agent import IngestionAgent

    agent = IngestionAgent()
    status = agent.get_job_status(args.job_id)

    if not status:
        print(f"Job {args.job_id} not found.", file=sys.stderr)
        return 1

    print(f"Job {status.job_id}")
    print("=" * 40)
    print(f"  Root path: {status.root_path}")
    print(f"  Status: {status.status}")
    print(f"  Progress: {status.progress_percent:.1f}%")
    print(f"  Total files: {status.total_files}")
    print(f"  Processed: {status.processed_files}")
    print(f"  Failed: {status.failed_files}")
    print(f"  Started: {format_datetime(status.started_at)}")
    print(f"  Completed: {format_datetime(status.completed_at)}")

    # JSON output option
    if args.json:
        print("\n--- JSON Output ---")
        job_dict = {
            "job_id": status.job_id,
            "root_path": status.root_path,
            "status": status.status,
            "progress_percent": status.progress_percent,
            "total_files": status.total_files,
            "processed_files": status.processed_files,
            "failed_files": status.failed_files,
            "started_at": format_datetime(status.started_at),
            "completed_at": format_datetime(status.completed_at),
        }
        print(json.dumps(job_dict, indent=2))

    return 0


def cmd_reset_errors(args: argparse.Namespace) -> int:
    """Reset error state for failed images."""
    from sims import db

    print("Resetting error state for failed images...")

    try:
        count = db.reset_errors()

        if count == 0:
            print("No images with errors found.")
        else:
            print(f"Reset {count} images.")
            print("\nRun 'python -m sims resume' to retry processing.")

        return 0

    except Exception as e:
        print(f"Error resetting errors: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Remove orphaned database entries for missing files."""
    from sims import db
    from sims.agent import cleanup_orphaned_images

    print("Scanning for orphaned database entries...")

    try:
        # First, do a dry-run check to show what will be removed
        db.init_db()
        all_images = db.get_all_images()

        orphaned = []
        for image in all_images:
            if not Path(image["original_path"]).exists():
                orphaned.append(image)

        if not orphaned:
            print("No orphaned entries found.")
            return 0

        print(f"\nFound {len(orphaned)} orphaned entries:")
        # Show first 10 entries
        for img in orphaned[:10]:
            print(f"  - {img['original_path']}")
        if len(orphaned) > 10:
            print(f"  ... and {len(orphaned) - 10} more")

        # Confirm unless --force
        if not args.force:
            response = input(f"\nRemove {len(orphaned)} orphaned entries? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0

        # Perform cleanup
        results = cleanup_orphaned_images()

        print(f"\nCleanup complete:")
        print(f"  Database records removed: {results['deleted_db']}")
        print(f"  Vector embeddings removed: {results['deleted_vectors']}")

        return 0

    except Exception as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="sims",
        description="SImS - Semantic Image Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sims init                          Initialize databases
  python -m sims ingest ~/Photos               Ingest images from directory
  python -m sims ingest ~/Photos --no-recursive  Only process top-level directory
  python -m sims resume                        Resume incomplete processing
  python -m sims search sunset beach           Search for sunset beach images
  python -m sims search "mountain landscape" --limit 5
  python -m sims stats                         Show system statistics
  python -m sims status 1                      Check status of job 1
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize databases")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest images from directory")
    ingest_parser.add_argument("path", help="Path to directory containing images")
    ingest_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't process subdirectories",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help=f"Batch size for processing (default: {Config.BATCH_SIZE})",
    )
    ingest_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    ingest_parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        metavar="MODEL",
        help=f"Vision model to use (default: {Config.VISION_MODEL})",
    )

    # resume command
    resume_parser = subparsers.add_parser("resume", help="Resume incomplete processing")
    resume_parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help=f"Batch size for processing (default: {Config.BATCH_SIZE})",
    )
    resume_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    resume_parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        metavar="MODEL",
        help=f"Vision model to use (default: {Config.VISION_MODEL})",
    )

    # search command
    search_parser = subparsers.add_parser("search", help="Search for images")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    search_parser.add_argument(
        "--date-after",
        help="Only show images taken after this date (YYYY-MM-DD)",
    )
    search_parser.add_argument(
        "--date-before",
        help="Only show images taken before this date (YYYY-MM-DD)",
    )
    search_parser.add_argument(
        "--has-gps",
        action="store_true",
        help="Only show images with GPS coordinates",
    )
    search_parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        metavar="PERCENT",
        help="Minimum match score percentage (0-100). Results below this threshold are excluded.",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", type=int, help="Job ID to check")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # reset-errors command
    reset_errors_parser = subparsers.add_parser(
        "reset-errors",
        help="Reset error state for failed images to allow retry"
    )

    # cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove orphaned database entries for missing files"
    )
    cleanup_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Route to command handler
    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "resume": cmd_resume,
        "search": cmd_search,
        "stats": cmd_stats,
        "status": cmd_status,
        "reset-errors": cmd_reset_errors,
        "cleanup": cmd_cleanup,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
