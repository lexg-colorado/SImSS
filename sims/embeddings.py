# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
Ollama text embedding interface for SImS.

Provides text embedding generation using nomic-embed-text via the Ollama API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from sims.config import Config

logger = logging.getLogger(__name__)

# Expected embedding dimension for nomic-embed-text
EMBEDDING_DIMENSION = 768


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingConnectionError(EmbeddingError):
    """Raised when unable to connect to Ollama."""
    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding request times out."""
    pass


class EmbeddingModelError(EmbeddingError):
    """Raised when embedding model is not available."""
    pass


class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding has unexpected dimensions."""
    pass


async def check_embed_model_available(timeout: float = 5.0) -> bool:
    """
    Check if the embedding model is available in Ollama.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        True if the embedding model is available.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{Config.OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_name = Config.EMBED_MODEL

            return model_name in models or any(
                model_name.split(":")[0] in m for m in models
            )
    except Exception:
        return False


async def embed_text(
    text: str,
    max_retries: int = None,
    timeout: float = None,
    validate_dimension: bool = True,
) -> list[float]:
    """
    Generate an embedding vector for text.

    Args:
        text: Text to embed.
        max_retries: Maximum retry attempts (default from config).
        timeout: Request timeout in seconds (default from config).
        validate_dimension: If True, validate embedding dimension.

    Returns:
        768-dimensional embedding vector as list of floats.

    Raises:
        ValueError: If text is empty.
        EmbeddingConnectionError: If unable to connect to Ollama.
        EmbeddingTimeoutError: If request times out after retries.
        EmbeddingModelError: If embedding model is not available.
        EmbeddingDimensionError: If embedding has wrong dimensions.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if max_retries is None:
        max_retries = Config.MAX_RETRIES

    if timeout is None:
        # Embedding is much faster than vision, use shorter timeout
        timeout = min(30.0, Config.OLLAMA_TIMEOUT)

    request_data = {
        "model": Config.EMBED_MODEL,
        "prompt": text,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Generating embedding (attempt {attempt + 1}/{max_retries})"
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{Config.OLLAMA_HOST}/api/embeddings",
                    json=request_data,
                )

                if response.status_code == 404:
                    raise EmbeddingModelError(
                        f"Model {Config.EMBED_MODEL} not found. "
                        f"Run: ollama pull {Config.EMBED_MODEL}"
                    )

                response.raise_for_status()
                data = response.json()

            embedding = data.get("embedding", [])

            if validate_dimension and len(embedding) != EMBEDDING_DIMENSION:
                raise EmbeddingDimensionError(
                    f"Expected {EMBEDDING_DIMENSION}-dimensional embedding, "
                    f"got {len(embedding)}"
                )

            logger.debug(
                f"Generated {len(embedding)}-dimensional embedding"
            )
            return embedding

        except httpx.ConnectError as e:
            last_error = EmbeddingConnectionError(
                f"Cannot connect to Ollama at {Config.OLLAMA_HOST}: {e}"
            )
            logger.warning(f"Connection error (attempt {attempt + 1}): {e}")

        except httpx.TimeoutException as e:
            last_error = EmbeddingTimeoutError(
                f"Request timed out after {timeout}s: {e}"
            )
            logger.warning(f"Timeout (attempt {attempt + 1}): {e}")

        except (EmbeddingModelError, EmbeddingDimensionError):
            raise

        except httpx.HTTPStatusError as e:
            last_error = EmbeddingError(f"HTTP error: {e}")
            logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")

        except Exception as e:
            last_error = EmbeddingError(f"Unexpected error: {e}")
            logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")

        # Exponential backoff before retry
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logger.debug(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    # All retries exhausted
    raise last_error or EmbeddingError("Failed to generate embedding after retries")


async def embed_batch(
    texts: list[str],
    max_retries: int = None,
    timeout: float = None,
    validate_dimension: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Note: Ollama doesn't have native batch embedding support,
    so this processes texts sequentially. For better performance,
    consider using asyncio.gather with a semaphore for concurrent requests.

    Args:
        texts: List of texts to embed.
        max_retries: Maximum retry attempts per text.
        timeout: Request timeout in seconds.
        validate_dimension: If True, validate embedding dimensions.

    Returns:
        List of embedding vectors.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    embeddings = []

    for i, text in enumerate(texts):
        logger.debug(f"Embedding text {i + 1}/{len(texts)}")
        embedding = await embed_text(
            text,
            max_retries=max_retries,
            timeout=timeout,
            validate_dimension=validate_dimension,
        )
        embeddings.append(embedding)

    return embeddings


async def embed_batch_concurrent(
    texts: list[str],
    max_concurrent: int = 3,
    max_retries: int = None,
    timeout: float = None,
    validate_dimension: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts with limited concurrency.

    Args:
        texts: List of texts to embed.
        max_concurrent: Maximum concurrent requests.
        max_retries: Maximum retry attempts per text.
        timeout: Request timeout in seconds.
        validate_dimension: If True, validate embedding dimensions.

    Returns:
        List of embedding vectors in same order as input texts.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_with_semaphore(text: str, index: int) -> tuple[int, list[float]]:
        async with semaphore:
            logger.debug(f"Embedding text {index + 1}/{len(texts)}")
            embedding = await embed_text(
                text,
                max_retries=max_retries,
                timeout=timeout,
                validate_dimension=validate_dimension,
            )
            return (index, embedding)

    # Create tasks with indices to preserve order
    tasks = [
        embed_with_semaphore(text, i)
        for i, text in enumerate(texts)
    ]

    # Run concurrently and gather results
    results = await asyncio.gather(*tasks)

    # Sort by index to return in original order
    sorted_results = sorted(results, key=lambda x: x[0])
    return [embedding for _, embedding in sorted_results]


def validate_embedding(embedding: list[float]) -> bool:
    """
    Validate that an embedding has the correct format.

    Args:
        embedding: Embedding vector to validate.

    Returns:
        True if embedding is valid.
    """
    if not isinstance(embedding, list):
        return False

    if len(embedding) != EMBEDDING_DIMENSION:
        return False

    if not all(isinstance(x, (int, float)) for x in embedding):
        return False

    return True


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have same length: {len(vec1)} != {len(vec2)}"
        )

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
