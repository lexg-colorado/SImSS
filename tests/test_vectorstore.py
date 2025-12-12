"""
Tests for the vectorstore module.

Tests ChromaDB vector storage and semantic search functionality.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from sims import vectorstore
from sims.vectorstore import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    init_vectorstore,
    get_collection,
    add_embedding,
    update_embedding,
    upsert_embedding,
    search,
    delete_embedding,
    get_embedding,
    get_collection_stats,
    reset_collection,
    close,
    VectorStoreError,
    CollectionNotInitializedError,
    EmbeddingExistsError,
    EmbeddingNotFoundError,
    _make_document_id,
    _parse_document_id,
)


def generate_mock_embedding(seed: int = 42) -> list[float]:
    """Generate a deterministic mock embedding for testing."""
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]


@pytest.fixture
def isolated_vectorstore(temp_chroma_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Fixture that provides an isolated vector store for testing."""
    # Close any existing connection
    close()

    # Clear module-level cache
    vectorstore._client = None
    vectorstore._collection = None

    # Set the chroma path to temporary directory
    monkeypatch.setenv("SIMS_CHROMA_PATH", str(temp_chroma_path))

    # Reload config to pick up new path
    from sims import config
    monkeypatch.setattr(config.Config, "CHROMA_PATH", temp_chroma_path)
    monkeypatch.setattr(config, "CHROMA_PATH", temp_chroma_path)

    yield

    # Cleanup
    close()
    vectorstore._client = None
    vectorstore._collection = None


class TestConstants:
    """Tests for module constants."""

    def test_collection_name(self) -> None:
        """Test that collection name is correct."""
        assert COLLECTION_NAME == "sims_embeddings"

    def test_embedding_dimension(self) -> None:
        """Test that embedding dimension is 768."""
        assert EMBEDDING_DIMENSION == 768


class TestDocumentIdHelpers:
    """Tests for document ID helper functions."""

    def test_make_document_id(self) -> None:
        """Test document ID creation."""
        assert _make_document_id(1) == "img_1"
        assert _make_document_id(123) == "img_123"
        assert _make_document_id(0) == "img_0"

    def test_parse_document_id(self) -> None:
        """Test document ID parsing."""
        assert _parse_document_id("img_1") == 1
        assert _parse_document_id("img_123") == 123
        assert _parse_document_id("img_0") == 0

    def test_parse_invalid_document_id(self) -> None:
        """Test parsing invalid document ID raises error."""
        with pytest.raises(ValueError, match="Invalid document ID format"):
            _parse_document_id("invalid_123")

        with pytest.raises(ValueError, match="Invalid document ID format"):
            _parse_document_id("123")

    def test_roundtrip(self) -> None:
        """Test that make and parse are inverses."""
        for image_id in [0, 1, 42, 999, 123456]:
            doc_id = _make_document_id(image_id)
            assert _parse_document_id(doc_id) == image_id


class TestInitialization:
    """Tests for vector store initialization."""

    def test_init_vectorstore(self, isolated_vectorstore: None) -> None:
        """Test vector store initialization creates collection."""
        collection = init_vectorstore()
        assert collection is not None
        assert collection.name == COLLECTION_NAME

    def test_get_collection_after_init(self, isolated_vectorstore: None) -> None:
        """Test getting collection after initialization."""
        init_vectorstore()
        collection = get_collection()
        assert collection is not None
        assert collection.name == COLLECTION_NAME

    def test_get_collection_auto_init(self, isolated_vectorstore: None) -> None:
        """Test that get_collection auto-initializes if needed."""
        collection = get_collection()
        assert collection is not None
        assert collection.name == COLLECTION_NAME

    def test_multiple_init_calls(self, isolated_vectorstore: None) -> None:
        """Test that multiple init calls don't cause issues."""
        collection1 = init_vectorstore()
        collection2 = init_vectorstore()
        assert collection1.name == collection2.name


class TestAddEmbedding:
    """Tests for add_embedding function."""

    def test_add_embedding_success(self, isolated_vectorstore: None) -> None:
        """Test successfully adding an embedding."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=1)
        metadata = {
            "original_path": "/photos/test.jpg",
            "date_taken": "2023-07-15",
            "has_gps": True,
        }

        add_embedding(
            image_id=1,
            embedding=embedding,
            description="A beautiful sunset",
            metadata=metadata,
        )

        # Verify it was added
        result = get_embedding(1)
        assert result is not None
        assert result["id"] == 1
        assert result["description"] == "A beautiful sunset"
        assert result["metadata"]["original_path"] == "/photos/test.jpg"
        assert result["metadata"]["has_gps"] is True

    def test_add_embedding_with_datetime(self, isolated_vectorstore: None) -> None:
        """Test adding embedding with datetime object."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=2)
        dt = datetime(2023, 7, 15, 14, 30, 0)
        metadata = {
            "original_path": "/photos/test.jpg",
            "date_taken": dt,
            "has_gps": False,
        }

        add_embedding(
            image_id=2,
            embedding=embedding,
            description="Test image",
            metadata=metadata,
        )

        result = get_embedding(2)
        assert result is not None
        assert "2023-07-15" in result["metadata"]["date_taken"]

    def test_add_embedding_wrong_dimension(self, isolated_vectorstore: None) -> None:
        """Test that wrong embedding dimension raises error."""
        init_vectorstore()

        wrong_embedding = [0.1] * 100  # Wrong dimension

        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            add_embedding(
                image_id=1,
                embedding=wrong_embedding,
                description="Test",
                metadata={"original_path": "/test.jpg", "has_gps": False},
            )

    def test_add_duplicate_embedding(self, isolated_vectorstore: None) -> None:
        """Test that adding duplicate embedding raises error."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=3)
        metadata = {"original_path": "/test.jpg", "has_gps": False}

        add_embedding(
            image_id=3,
            embedding=embedding,
            description="First",
            metadata=metadata,
        )

        with pytest.raises(EmbeddingExistsError, match="already exists"):
            add_embedding(
                image_id=3,
                embedding=embedding,
                description="Second",
                metadata=metadata,
            )

    def test_add_embedding_minimal_metadata(self, isolated_vectorstore: None) -> None:
        """Test adding embedding with minimal metadata."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=4)
        metadata = {}  # Empty metadata

        add_embedding(
            image_id=4,
            embedding=embedding,
            description="Minimal test",
            metadata=metadata,
        )

        result = get_embedding(4)
        assert result is not None
        assert result["description"] == "Minimal test"


class TestUpdateEmbedding:
    """Tests for update_embedding function."""

    def test_update_embedding_success(self, isolated_vectorstore: None) -> None:
        """Test successfully updating an embedding."""
        init_vectorstore()

        # Add initial embedding
        embedding1 = generate_mock_embedding(seed=10)
        add_embedding(
            image_id=10,
            embedding=embedding1,
            description="Original description",
            metadata={"original_path": "/test.jpg", "has_gps": False},
        )

        # Update it
        embedding2 = generate_mock_embedding(seed=11)
        update_embedding(
            image_id=10,
            embedding=embedding2,
            description="Updated description",
            metadata={"original_path": "/test.jpg", "has_gps": True},
        )

        result = get_embedding(10)
        assert result is not None
        assert result["description"] == "Updated description"
        assert result["metadata"]["has_gps"] is True

    def test_update_nonexistent_embedding(self, isolated_vectorstore: None) -> None:
        """Test updating non-existent embedding raises error."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=20)

        with pytest.raises(EmbeddingNotFoundError, match="No embedding found"):
            update_embedding(
                image_id=999,
                embedding=embedding,
                description="Test",
                metadata={"original_path": "/test.jpg", "has_gps": False},
            )


class TestUpsertEmbedding:
    """Tests for upsert_embedding function."""

    def test_upsert_new_embedding(self, isolated_vectorstore: None) -> None:
        """Test upsert creates new embedding."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=30)
        upsert_embedding(
            image_id=30,
            embedding=embedding,
            description="New via upsert",
            metadata={"original_path": "/test.jpg", "has_gps": False},
        )

        result = get_embedding(30)
        assert result is not None
        assert result["description"] == "New via upsert"

    def test_upsert_existing_embedding(self, isolated_vectorstore: None) -> None:
        """Test upsert updates existing embedding."""
        init_vectorstore()

        # Add initial
        embedding1 = generate_mock_embedding(seed=31)
        add_embedding(
            image_id=31,
            embedding=embedding1,
            description="Original",
            metadata={"original_path": "/test.jpg", "has_gps": False},
        )

        # Upsert same ID
        embedding2 = generate_mock_embedding(seed=32)
        upsert_embedding(
            image_id=31,
            embedding=embedding2,
            description="Updated via upsert",
            metadata={"original_path": "/test.jpg", "has_gps": True},
        )

        result = get_embedding(31)
        assert result is not None
        assert result["description"] == "Updated via upsert"
        assert result["metadata"]["has_gps"] is True


class TestSearch:
    """Tests for search function."""

    def test_search_returns_results(self, isolated_vectorstore: None) -> None:
        """Test that search returns matching results."""
        init_vectorstore()

        # Add some embeddings
        for i in range(5):
            embedding = generate_mock_embedding(seed=100 + i)
            add_embedding(
                image_id=100 + i,
                embedding=embedding,
                description=f"Test image {i}",
                metadata={
                    "original_path": f"/photos/test_{i}.jpg",
                    "date_taken": f"2023-0{i+1}-15",
                    "has_gps": i % 2 == 0,
                },
            )

        # Search with one of the embeddings (should find itself)
        query_embedding = generate_mock_embedding(seed=100)
        results = search(query_embedding, limit=3)

        assert len(results) <= 3
        assert len(results) > 0
        # First result should be the exact match
        assert results[0]["id"] == 100
        assert results[0]["score"] > 0.9  # Should be very similar

    def test_search_with_limit(self, isolated_vectorstore: None) -> None:
        """Test search respects limit parameter."""
        init_vectorstore()

        # Add 10 embeddings
        for i in range(10):
            embedding = generate_mock_embedding(seed=200 + i)
            add_embedding(
                image_id=200 + i,
                embedding=embedding,
                description=f"Image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        query_embedding = generate_mock_embedding(seed=200)
        results = search(query_embedding, limit=5)

        assert len(results) == 5

    def test_search_with_gps_filter(self, isolated_vectorstore: None) -> None:
        """Test search with has_gps filter."""
        init_vectorstore()

        # Add embeddings with and without GPS
        for i in range(6):
            embedding = generate_mock_embedding(seed=300 + i)
            add_embedding(
                image_id=300 + i,
                embedding=embedding,
                description=f"Image {i}",
                metadata={
                    "original_path": f"/test_{i}.jpg",
                    "has_gps": i < 3,  # First 3 have GPS
                },
            )

        query_embedding = generate_mock_embedding(seed=300)
        results = search(query_embedding, limit=10, filters={"has_gps": True})

        # All results should have GPS
        for result in results:
            assert result["metadata"]["has_gps"] is True

    def test_search_wrong_dimension(self, isolated_vectorstore: None) -> None:
        """Test search with wrong dimension raises error."""
        init_vectorstore()

        wrong_embedding = [0.1] * 100

        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            search(wrong_embedding, limit=10)

    def test_search_empty_collection(self, isolated_vectorstore: None) -> None:
        """Test search on empty collection returns empty list."""
        init_vectorstore()

        query_embedding = generate_mock_embedding(seed=400)
        results = search(query_embedding, limit=10)

        assert results == []

    def test_search_with_min_score_filters_results(self, isolated_vectorstore: None) -> None:
        """Test search with min_score filters out low-scoring results."""
        init_vectorstore()

        # Add embeddings with known seeds
        for i in range(5):
            embedding = generate_mock_embedding(seed=400 + i)
            add_embedding(
                image_id=400 + i,
                embedding=embedding,
                description=f"Test image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        # Search with the first embedding (should match itself perfectly)
        query_embedding = generate_mock_embedding(seed=400)

        # Without min_score, should get all results
        all_results = search(query_embedding, limit=10)
        assert len(all_results) == 5

        # With high min_score, should filter out dissimilar results
        # The exact match (seed 400) should have score ~1.0
        # Other seeds will have lower scores
        filtered_results = search(query_embedding, limit=10, min_score=0.95)

        # Should have fewer results (only the exact/near matches)
        assert len(filtered_results) <= len(all_results)
        # All results should meet the threshold
        for result in filtered_results:
            assert result["score"] >= 0.95

    def test_search_with_min_score_none_returns_all(self, isolated_vectorstore: None) -> None:
        """Test search with min_score=None returns all results (no filtering)."""
        init_vectorstore()

        for i in range(3):
            embedding = generate_mock_embedding(seed=450 + i)
            add_embedding(
                image_id=450 + i,
                embedding=embedding,
                description=f"Image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        query_embedding = generate_mock_embedding(seed=450)

        # With min_score=None (default), should return all results
        results = search(query_embedding, limit=10, min_score=None)
        assert len(results) == 3

    def test_search_min_score_boundary(self, isolated_vectorstore: None) -> None:
        """Test min_score exact boundary conditions."""
        init_vectorstore()

        # Add a single embedding
        embedding = generate_mock_embedding(seed=460)
        add_embedding(
            image_id=460,
            embedding=embedding,
            description="Exact match test",
            metadata={"original_path": "/test.jpg", "has_gps": False},
        )

        # Search for exact match
        query_embedding = generate_mock_embedding(seed=460)
        results = search(query_embedding, limit=10)

        # Should have one result with very high score
        assert len(results) == 1
        exact_score = results[0]["score"]
        assert exact_score > 0.99  # Should be ~1.0 for exact match

        # With min_score exactly at the score, should include the result
        results_at_threshold = search(query_embedding, limit=10, min_score=exact_score)
        assert len(results_at_threshold) == 1

        # With min_score just above, should exclude
        results_above_threshold = search(query_embedding, limit=10, min_score=exact_score + 0.01)
        assert len(results_above_threshold) == 0


class TestDeleteEmbedding:
    """Tests for delete_embedding function."""

    def test_delete_embedding_success(self, isolated_vectorstore: None) -> None:
        """Test successfully deleting an embedding."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=500)
        add_embedding(
            image_id=500,
            embedding=embedding,
            description="To be deleted",
            metadata={"original_path": "/test.jpg", "has_gps": False},
        )

        # Verify it exists
        assert get_embedding(500) is not None

        # Delete it
        delete_embedding(500)

        # Verify it's gone
        assert get_embedding(500) is None

    def test_delete_nonexistent_embedding(self, isolated_vectorstore: None) -> None:
        """Test deleting non-existent embedding raises error."""
        init_vectorstore()

        with pytest.raises(EmbeddingNotFoundError, match="No embedding found"):
            delete_embedding(999)


class TestGetEmbedding:
    """Tests for get_embedding function."""

    def test_get_existing_embedding(self, isolated_vectorstore: None) -> None:
        """Test getting an existing embedding."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=600)
        add_embedding(
            image_id=600,
            embedding=embedding,
            description="Test description",
            metadata={
                "original_path": "/photos/test.jpg",
                "date_taken": "2023-06-15",
                "has_gps": True,
            },
        )

        result = get_embedding(600)

        assert result is not None
        assert result["id"] == 600
        assert result["description"] == "Test description"
        assert len(result["embedding"]) == EMBEDDING_DIMENSION
        assert result["metadata"]["original_path"] == "/photos/test.jpg"

    def test_get_nonexistent_embedding(self, isolated_vectorstore: None) -> None:
        """Test getting non-existent embedding returns None."""
        init_vectorstore()

        result = get_embedding(999)
        assert result is None


class TestCollectionStats:
    """Tests for get_collection_stats function."""

    def test_stats_empty_collection(self, isolated_vectorstore: None) -> None:
        """Test stats on empty collection."""
        init_vectorstore()

        stats = get_collection_stats()

        assert stats["total_embeddings"] == 0
        assert stats["collection_name"] == COLLECTION_NAME
        assert "persist_directory" in stats

    def test_stats_with_embeddings(self, isolated_vectorstore: None) -> None:
        """Test stats after adding embeddings."""
        init_vectorstore()

        # Add some embeddings
        for i in range(5):
            embedding = generate_mock_embedding(seed=700 + i)
            add_embedding(
                image_id=700 + i,
                embedding=embedding,
                description=f"Image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        stats = get_collection_stats()
        assert stats["total_embeddings"] == 5


class TestResetCollection:
    """Tests for reset_collection function."""

    def test_reset_clears_all_data(self, isolated_vectorstore: None) -> None:
        """Test that reset clears all embeddings."""
        init_vectorstore()

        # Add some embeddings
        for i in range(3):
            embedding = generate_mock_embedding(seed=800 + i)
            add_embedding(
                image_id=800 + i,
                embedding=embedding,
                description=f"Image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        # Verify they exist
        stats = get_collection_stats()
        assert stats["total_embeddings"] == 3

        # Reset
        reset_collection()

        # Verify empty
        stats = get_collection_stats()
        assert stats["total_embeddings"] == 0

    def test_reset_on_empty_collection(self, isolated_vectorstore: None) -> None:
        """Test reset on empty collection doesn't error."""
        init_vectorstore()
        reset_collection()  # Should not raise

        stats = get_collection_stats()
        assert stats["total_embeddings"] == 0


class TestClose:
    """Tests for close function."""

    def test_close_clears_cache(self, isolated_vectorstore: None) -> None:
        """Test that close clears module-level cache."""
        init_vectorstore()

        assert vectorstore._collection is not None

        close()

        assert vectorstore._client is None
        assert vectorstore._collection is None


class TestSearchResultFormat:
    """Tests for search result format and scoring."""

    def test_result_contains_all_fields(self, isolated_vectorstore: None) -> None:
        """Test that search results contain all expected fields."""
        init_vectorstore()

        embedding = generate_mock_embedding(seed=900)
        add_embedding(
            image_id=900,
            embedding=embedding,
            description="Test image for field validation",
            metadata={
                "original_path": "/photos/field_test.jpg",
                "date_taken": "2023-09-15",
                "has_gps": True,
            },
        )

        results = search(embedding, limit=1)

        assert len(results) == 1
        result = results[0]

        assert "id" in result
        assert "score" in result
        assert "description" in result
        assert "metadata" in result

        assert isinstance(result["id"], int)
        assert isinstance(result["score"], float)
        assert isinstance(result["description"], str)
        assert isinstance(result["metadata"], dict)

    def test_scores_are_normalized(self, isolated_vectorstore: None) -> None:
        """Test that similarity scores are in valid range."""
        init_vectorstore()

        # Add diverse embeddings
        for i in range(5):
            embedding = generate_mock_embedding(seed=1000 + i * 100)
            add_embedding(
                image_id=1000 + i,
                embedding=embedding,
                description=f"Diverse image {i}",
                metadata={"original_path": f"/test_{i}.jpg", "has_gps": False},
            )

        query_embedding = generate_mock_embedding(seed=1000)
        results = search(query_embedding, limit=5)

        for result in results:
            # Cosine similarity should be between -1 and 1
            # After conversion from distance, score should be reasonable
            assert -1.0 <= result["score"] <= 1.0
