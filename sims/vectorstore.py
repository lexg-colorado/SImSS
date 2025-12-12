# Copyright (c) 2025 Lex Gaines
# Licensed under the Apache License, Version 2.0

"""
ChromaDB vector store operations for SImS.

Provides vector storage and semantic search using ChromaDB with persistent storage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from sims.config import Config

logger = logging.getLogger(__name__)

# Collection name for SImS embeddings
COLLECTION_NAME = "sims_embeddings"

# Expected embedding dimension
EMBEDDING_DIMENSION = 768

# Module-level client and collection cache
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class CollectionNotInitializedError(VectorStoreError):
    """Raised when collection is accessed before initialization."""
    pass


class EmbeddingExistsError(VectorStoreError):
    """Raised when trying to add an embedding that already exists."""
    pass


class EmbeddingNotFoundError(VectorStoreError):
    """Raised when an embedding is not found."""
    pass


def _get_client() -> chromadb.PersistentClient:
    """
    Get or create the ChromaDB client.

    Returns:
        PersistentClient instance.
    """
    global _client
    if _client is None:
        Config.ensure_directories()
        _client = chromadb.PersistentClient(
            path=str(Config.CHROMA_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        logger.debug(f"ChromaDB client initialized at {Config.CHROMA_PATH}")
    return _client


def init_vectorstore() -> chromadb.Collection:
    """
    Initialize the vector store and return the collection.

    Creates the collection if it doesn't exist, or gets the existing one.

    Returns:
        The ChromaDB collection for SImS embeddings.
    """
    global _collection
    client = _get_client()
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SImS image embeddings",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "hnsw:space": "cosine",  # Use cosine similarity
        },
    )
    logger.info(f"Vector store initialized: {COLLECTION_NAME}")
    return _collection


def get_collection() -> chromadb.Collection:
    """
    Get the initialized collection.

    Returns:
        The ChromaDB collection.

    Raises:
        CollectionNotInitializedError: If init_vectorstore() hasn't been called.
    """
    global _collection
    if _collection is None:
        # Try to initialize if not done yet
        _collection = init_vectorstore()
    return _collection


def _make_document_id(image_id: int) -> str:
    """
    Create a document ID from an image ID.

    Args:
        image_id: The database image ID.

    Returns:
        Document ID string in format 'img_{id}'.
    """
    return f"img_{image_id}"


def _parse_document_id(doc_id: str) -> int:
    """
    Parse an image ID from a document ID.

    Args:
        doc_id: Document ID string.

    Returns:
        The image ID as integer.

    Raises:
        ValueError: If doc_id is not in expected format.
    """
    if not doc_id.startswith("img_"):
        raise ValueError(f"Invalid document ID format: {doc_id}")
    return int(doc_id[4:])


def add_embedding(
    image_id: int,
    embedding: list[float],
    description: str,
    metadata: dict[str, Any],
) -> None:
    """
    Add an embedding to the vector store.

    Args:
        image_id: The database image ID.
        embedding: 768-dimensional embedding vector.
        description: The image description text.
        metadata: Additional metadata dict with keys:
            - original_path: str - Path to original image
            - date_taken: str | None - ISO format date string
            - has_gps: bool - Whether image has GPS coordinates

    Raises:
        VectorStoreError: If embedding dimension is wrong.
        EmbeddingExistsError: If embedding already exists for this image.
    """
    if len(embedding) != EMBEDDING_DIMENSION:
        raise VectorStoreError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
            f"got {len(embedding)}"
        )

    collection = get_collection()
    doc_id = _make_document_id(image_id)

    # Check if embedding already exists
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        raise EmbeddingExistsError(f"Embedding already exists for image {image_id}")

    # Prepare metadata for ChromaDB (must be flat key-value pairs)
    chroma_metadata = {
        "original_path": str(metadata.get("original_path", "")),
        "has_gps": metadata.get("has_gps", False),
    }

    # Handle date_taken - ChromaDB doesn't support datetime natively
    date_taken = metadata.get("date_taken")
    if date_taken:
        if isinstance(date_taken, datetime):
            chroma_metadata["date_taken"] = date_taken.isoformat()
        else:
            chroma_metadata["date_taken"] = str(date_taken)
    else:
        chroma_metadata["date_taken"] = ""

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[chroma_metadata],
    )
    logger.debug(f"Added embedding for image {image_id}")


def update_embedding(
    image_id: int,
    embedding: list[float],
    description: str,
    metadata: dict[str, Any],
) -> None:
    """
    Update an existing embedding in the vector store.

    Args:
        image_id: The database image ID.
        embedding: 768-dimensional embedding vector.
        description: The image description text.
        metadata: Additional metadata dict.

    Raises:
        VectorStoreError: If embedding dimension is wrong.
        EmbeddingNotFoundError: If embedding doesn't exist.
    """
    if len(embedding) != EMBEDDING_DIMENSION:
        raise VectorStoreError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
            f"got {len(embedding)}"
        )

    collection = get_collection()
    doc_id = _make_document_id(image_id)

    # Check if embedding exists
    existing = collection.get(ids=[doc_id])
    if not existing["ids"]:
        raise EmbeddingNotFoundError(f"No embedding found for image {image_id}")

    # Prepare metadata
    chroma_metadata = {
        "original_path": str(metadata.get("original_path", "")),
        "has_gps": metadata.get("has_gps", False),
    }

    date_taken = metadata.get("date_taken")
    if date_taken:
        if isinstance(date_taken, datetime):
            chroma_metadata["date_taken"] = date_taken.isoformat()
        else:
            chroma_metadata["date_taken"] = str(date_taken)
    else:
        chroma_metadata["date_taken"] = ""

    collection.update(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[chroma_metadata],
    )
    logger.debug(f"Updated embedding for image {image_id}")


def upsert_embedding(
    image_id: int,
    embedding: list[float],
    description: str,
    metadata: dict[str, Any],
) -> None:
    """
    Add or update an embedding in the vector store.

    Args:
        image_id: The database image ID.
        embedding: 768-dimensional embedding vector.
        description: The image description text.
        metadata: Additional metadata dict.

    Raises:
        VectorStoreError: If embedding dimension is wrong.
    """
    if len(embedding) != EMBEDDING_DIMENSION:
        raise VectorStoreError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
            f"got {len(embedding)}"
        )

    collection = get_collection()
    doc_id = _make_document_id(image_id)

    # Prepare metadata
    chroma_metadata = {
        "original_path": str(metadata.get("original_path", "")),
        "has_gps": metadata.get("has_gps", False),
    }

    date_taken = metadata.get("date_taken")
    if date_taken:
        if isinstance(date_taken, datetime):
            chroma_metadata["date_taken"] = date_taken.isoformat()
        else:
            chroma_metadata["date_taken"] = str(date_taken)
    else:
        chroma_metadata["date_taken"] = ""

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[chroma_metadata],
    )
    logger.debug(f"Upserted embedding for image {image_id}")


def search(
    query_embedding: list[float],
    limit: int = 20,
    filters: Optional[dict[str, Any]] = None,
    min_score: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Search for similar images by embedding.

    Args:
        query_embedding: 768-dimensional query embedding vector.
        limit: Maximum number of results to return.
        filters: Optional filters dict with keys:
            - date_after: str - ISO date, return images taken after this date
            - date_before: str - ISO date, return images taken before this date
            - has_gps: bool - If True, only return images with GPS data
        min_score: Optional minimum similarity score threshold (0.0 to 1.0).
            Results with scores below this threshold are excluded.
            Note: When using min_score, fewer than `limit` results may be returned.

    Returns:
        List of search results, each containing:
            - id: int - Image database ID
            - score: float - Similarity score (higher is better)
            - description: str - Image description
            - metadata: dict - Image metadata

    Raises:
        VectorStoreError: If query embedding dimension is wrong.
    """
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise VectorStoreError(
            f"Query embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
            f"got {len(query_embedding)}"
        )

    collection = get_collection()

    # Build ChromaDB where clause from filters
    where_clause = None
    if filters:
        conditions = []

        if filters.get("has_gps") is True:
            conditions.append({"has_gps": {"$eq": True}})

        if filters.get("date_after"):
            conditions.append({"date_taken": {"$gte": filters["date_after"]}})

        if filters.get("date_before"):
            conditions.append({"date_taken": {"$lte": filters["date_before"]}})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

    # Perform search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    # Transform results into list of dicts
    search_results = []
    if results["ids"] and results["ids"][0]:
        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, doc_id in enumerate(ids):
            # Convert distance to similarity score
            # ChromaDB returns L2 distance for cosine, convert to similarity
            # For cosine distance, similarity = 1 - distance
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 - distance
            # Clamp to [-1, 1] to handle floating point precision issues
            score = max(-1.0, min(1.0, score))

            # Apply min_score filter if specified
            if min_score is not None and score < min_score:
                continue

            search_results.append({
                "id": _parse_document_id(doc_id),
                "score": score,
                "description": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })

    return search_results


def delete_embedding(image_id: int) -> None:
    """
    Delete an embedding from the vector store.

    Args:
        image_id: The database image ID.

    Raises:
        EmbeddingNotFoundError: If embedding doesn't exist.
    """
    collection = get_collection()
    doc_id = _make_document_id(image_id)

    # Check if embedding exists
    existing = collection.get(ids=[doc_id])
    if not existing["ids"]:
        raise EmbeddingNotFoundError(f"No embedding found for image {image_id}")

    collection.delete(ids=[doc_id])
    logger.debug(f"Deleted embedding for image {image_id}")


def delete_embeddings_batch(image_ids: list[int]) -> int:
    """
    Delete multiple embeddings from the vector store.

    Args:
        image_ids: List of database image IDs.

    Returns:
        Number of embeddings successfully deleted.
    """
    if not image_ids:
        return 0

    collection = get_collection()
    doc_ids = [_make_document_id(img_id) for img_id in image_ids]

    # Check which embeddings exist
    existing = collection.get(ids=doc_ids)
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    if not existing_ids:
        return 0

    # Delete only existing embeddings
    collection.delete(ids=list(existing_ids))
    count = len(existing_ids)
    logger.info(f"Deleted {count} embeddings from vector store")
    return count


def get_embedding(image_id: int) -> Optional[dict[str, Any]]:
    """
    Get an embedding by image ID.

    Args:
        image_id: The database image ID.

    Returns:
        Dict with embedding data or None if not found:
            - id: int - Image database ID
            - embedding: list[float] - The embedding vector
            - description: str - Image description
            - metadata: dict - Image metadata
    """
    collection = get_collection()
    doc_id = _make_document_id(image_id)

    result = collection.get(
        ids=[doc_id],
        include=["embeddings", "documents", "metadatas"],
    )

    if not result["ids"]:
        return None

    # Handle embeddings - ChromaDB may return numpy arrays or None
    embedding = []
    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
        emb = result["embeddings"][0]
        # Convert numpy array to list if needed
        embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)

    return {
        "id": image_id,
        "embedding": embedding,
        "description": result["documents"][0] if result["documents"] else "",
        "metadata": result["metadatas"][0] if result["metadatas"] else {},
    }


def get_collection_stats() -> dict[str, Any]:
    """
    Get statistics about the vector store collection.

    Returns:
        Dict with collection statistics:
            - total_embeddings: int - Total number of embeddings
            - collection_name: str - Name of the collection
            - persist_directory: str - Path to ChromaDB storage
    """
    collection = get_collection()

    return {
        "total_embeddings": collection.count(),
        "collection_name": COLLECTION_NAME,
        "persist_directory": str(Config.CHROMA_PATH),
    }


def reset_collection() -> None:
    """
    Reset (delete and recreate) the collection.

    WARNING: This deletes all embeddings!
    """
    global _collection
    client = _get_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.warning(f"Deleted collection: {COLLECTION_NAME}")
    except ValueError:
        # Collection doesn't exist
        pass

    _collection = None
    init_vectorstore()
    logger.info(f"Collection reset: {COLLECTION_NAME}")


def close() -> None:
    """
    Close the vector store connection and clear caches.
    """
    global _client, _collection
    _client = None
    _collection = None
    logger.debug("Vector store connection closed")


# CLI support
if __name__ == "__main__":
    import sys

    Config.configure_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m sims.vectorstore <command>")
        print("Commands:")
        print("  init   - Initialize the vector store")
        print("  stats  - Show collection statistics")
        print("  reset  - Reset (delete all) the collection")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "init":
        init_vectorstore()
        print(f"Vector store initialized at {Config.CHROMA_PATH}")
    elif command == "stats":
        try:
            stats = get_collection_stats()
            print(f"Collection: {stats['collection_name']}")
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Storage path: {stats['persist_directory']}")
        except Exception as e:
            print(f"Error getting stats: {e}")
            sys.exit(1)
    elif command == "reset":
        confirm = input("Are you sure you want to delete all embeddings? [y/N]: ")
        if confirm.lower() == "y":
            reset_collection()
            print("Collection reset successfully")
        else:
            print("Cancelled")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
