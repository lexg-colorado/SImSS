# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Configuration management for SImS.

Configuration can be set via environment variables or by modifying the Config class.
Environment variables take precedence over defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar, Optional


class Config:
    """Central configuration for SImS."""

    # Base directories
    SIMS_HOME: ClassVar[Path] = Path(
        os.getenv("SIMS_HOME", str(Path.home() / "sims"))
    )
    CACHE_DIR: ClassVar[Path] = Path(
        os.getenv("SIMS_CACHE_DIR", str(SIMS_HOME / "cache"))
    )
    DATA_DIR: ClassVar[Path] = Path(
        os.getenv("SIMS_DATA_DIR", str(SIMS_HOME / "data"))
    )

    # Database paths
    DB_PATH: ClassVar[Path] = Path(
        os.getenv("SIMS_DB_PATH", str(DATA_DIR / "sims.db"))
    )
    CHROMA_PATH: ClassVar[Path] = Path(
        os.getenv("SIMS_CHROMA_PATH", str(DATA_DIR / "chroma"))
    )

    # Ollama configuration
    OLLAMA_HOST: ClassVar[str] = os.getenv(
        "SIMS_OLLAMA_HOST", "http://localhost:11434"
    )
    VISION_MODEL: ClassVar[str] = os.getenv(
        "SIMS_VISION_MODEL", "qwen3-vl:8b"
    )
    _vision_model_override: ClassVar[Optional[str]] = None

    EMBED_MODEL: ClassVar[str] = os.getenv(
        "SIMS_EMBED_MODEL", "nomic-embed-text"
    )

    # Thumbnail settings
    THUMBNAIL_MAX_SIZE: ClassVar[int] = int(
        os.getenv("SIMS_THUMBNAIL_MAX_SIZE", "768")
    )
    THUMBNAIL_QUALITY: ClassVar[int] = int(
        os.getenv("SIMS_THUMBNAIL_QUALITY", "85")
    )

    # Processing settings
    BATCH_SIZE: ClassVar[int] = int(
        os.getenv("SIMS_BATCH_SIZE", "10")
    )
    MAX_RETRIES: ClassVar[int] = int(
        os.getenv("SIMS_MAX_RETRIES", "3")
    )
    OLLAMA_TIMEOUT: ClassVar[int] = int(
        os.getenv("SIMS_OLLAMA_TIMEOUT", "120")
    )

    # Supported image formats
    SUPPORTED_FORMATS: ClassVar[tuple[str, ...]] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".cr2",
        ".nef",
        ".arw",
        ".dng",
    )

    # Logging configuration
    LOG_LEVEL: ClassVar[str] = os.getenv("SIMS_LOG_LEVEL", "INFO")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_vision_model(cls) -> str:
        """Get the current vision model, respecting runtime overrides."""
        if cls._vision_model_override is not None:
            return cls._vision_model_override
        return cls.VISION_MODEL

    @classmethod
    def set_vision_model(cls, model: str) -> None:
        """Set a runtime override for the vision model."""
        cls._vision_model_override = model

    @classmethod
    def reset_vision_model(cls) -> None:
        """Reset vision model to default (environment/config)."""
        cls._vision_model_override = None

    @classmethod
    def configure_logging(cls) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @classmethod
    def as_dict(cls) -> dict:
        """Return configuration as a dictionary."""
        return {
            "SIMS_HOME": str(cls.SIMS_HOME),
            "CACHE_DIR": str(cls.CACHE_DIR),
            "DATA_DIR": str(cls.DATA_DIR),
            "DB_PATH": str(cls.DB_PATH),
            "CHROMA_PATH": str(cls.CHROMA_PATH),
            "OLLAMA_HOST": cls.OLLAMA_HOST,
            "VISION_MODEL": cls.get_vision_model(),
            "EMBED_MODEL": cls.EMBED_MODEL,
            "THUMBNAIL_MAX_SIZE": cls.THUMBNAIL_MAX_SIZE,
            "THUMBNAIL_QUALITY": cls.THUMBNAIL_QUALITY,
            "BATCH_SIZE": cls.BATCH_SIZE,
            "MAX_RETRIES": cls.MAX_RETRIES,
            "OLLAMA_TIMEOUT": cls.OLLAMA_TIMEOUT,
            "SUPPORTED_FORMATS": cls.SUPPORTED_FORMATS,
            "LOG_LEVEL": cls.LOG_LEVEL,
        }


# Convenience exports for direct imports
SIMS_HOME = Config.SIMS_HOME
CACHE_DIR = Config.CACHE_DIR
DATA_DIR = Config.DATA_DIR
DB_PATH = Config.DB_PATH
CHROMA_PATH = Config.CHROMA_PATH
OLLAMA_HOST = Config.OLLAMA_HOST
VISION_MODEL = Config.VISION_MODEL
EMBED_MODEL = Config.EMBED_MODEL
THUMBNAIL_MAX_SIZE = Config.THUMBNAIL_MAX_SIZE
THUMBNAIL_QUALITY = Config.THUMBNAIL_QUALITY
BATCH_SIZE = Config.BATCH_SIZE
MAX_RETRIES = Config.MAX_RETRIES
OLLAMA_TIMEOUT = Config.OLLAMA_TIMEOUT
SUPPORTED_FORMATS = Config.SUPPORTED_FORMATS
LOG_LEVEL = Config.LOG_LEVEL
