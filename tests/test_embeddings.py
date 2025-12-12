"""
Tests for the embeddings module.

Tests text embedding generation, batch processing, and validation.
Uses mocks for Ollama API calls to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from sims.embeddings import (
    EMBEDDING_DIMENSION,
    embed_text,
    embed_batch,
    embed_batch_concurrent,
    validate_embedding,
    cosine_similarity,
    check_embed_model_available,
    EmbeddingError,
    EmbeddingConnectionError,
    EmbeddingTimeoutError,
    EmbeddingModelError,
    EmbeddingDimensionError,
)


def generate_mock_embedding(seed: int = 42) -> list[float]:
    """Generate a deterministic mock embedding for testing."""
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]


class TestEmbeddingDimension:
    """Tests for embedding dimension constant."""

    def test_dimension_is_768(self) -> None:
        """Test that expected dimension is 768 (nomic-embed-text)."""
        assert EMBEDDING_DIMENSION == 768


class TestCheckEmbedModelAvailable:
    """Tests for check_embed_model_available function."""

    @pytest.mark.asyncio
    async def test_model_available(self) -> None:
        """Test when embedding model is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {
            "models": [
                {"name": "nomic-embed-text"},
                {"name": "other-model"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_embed_model_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_model_not_available(self) -> None:
        """Test when embedding model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {
            "models": [{"name": "other-model"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await check_embed_model_available()
            assert result is False


class TestEmbedText:
    """Tests for embed_text function."""

    @pytest.mark.asyncio
    async def test_embed_text_success(self) -> None:
        """Test successful text embedding."""
        mock_embedding = generate_mock_embedding()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"embedding": mock_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await embed_text("Test text for embedding")

            assert len(result) == EMBEDDING_DIMENSION
            assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_embed_text_empty_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await embed_text("")

    @pytest.mark.asyncio
    async def test_embed_text_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await embed_text("   \n\t  ")

    @pytest.mark.asyncio
    async def test_embed_text_model_not_found(self) -> None:
        """Test error when model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(EmbeddingModelError):
                await embed_text("Test text", max_retries=1)

    @pytest.mark.asyncio
    async def test_embed_text_connection_error_retries(self) -> None:
        """Test retry behavior on connection errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(EmbeddingConnectionError):
                await embed_text("Test text", max_retries=2)

            # Should have attempted twice
            assert mock_instance.post.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_text_timeout_retries(self) -> None:
        """Test retry behavior on timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(EmbeddingTimeoutError):
                await embed_text("Test text", max_retries=2)

    @pytest.mark.asyncio
    async def test_embed_text_wrong_dimension_raises_error(self) -> None:
        """Test error when embedding has wrong dimension."""
        wrong_embedding = [0.1] * 100  # Wrong size

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"embedding": wrong_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(EmbeddingDimensionError):
                await embed_text("Test text", max_retries=1)

    @pytest.mark.asyncio
    async def test_embed_text_skip_dimension_validation(self) -> None:
        """Test that dimension validation can be skipped."""
        wrong_embedding = [0.1] * 100

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"embedding": wrong_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            # Should not raise when validation is disabled
            result = await embed_text("Test text", validate_dimension=False)
            assert len(result) == 100


class TestEmbedBatch:
    """Tests for embed_batch function."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self) -> None:
        """Test successful batch embedding."""
        mock_embeddings = [generate_mock_embedding(i) for i in range(3)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        call_count = 0

        def get_response():
            nonlocal call_count
            result = {"embedding": mock_embeddings[call_count]}
            call_count += 1
            return result

        mock_response.json = get_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            texts = ["Text 1", "Text 2", "Text 3"]
            results = await embed_batch(texts)

            assert len(results) == 3
            for result in results:
                assert len(result) == EMBEDDING_DIMENSION

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await embed_batch([])


class TestEmbedBatchConcurrent:
    """Tests for embed_batch_concurrent function."""

    @pytest.mark.asyncio
    async def test_embed_batch_concurrent_success(self) -> None:
        """Test successful concurrent batch embedding."""
        mock_embeddings = [generate_mock_embedding(i) for i in range(3)]
        call_indices = []

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_post(*args, **kwargs):
            # Track call order
            text = kwargs.get("json", {}).get("prompt", "")
            index = int(text.split()[-1]) - 1 if text.split() else 0
            call_indices.append(index)

            response = MagicMock()
            response.status_code = 200
            response.json = lambda: {"embedding": mock_embeddings[index]}
            response.raise_for_status = MagicMock()
            return response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = mock_post
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            texts = ["Text 1", "Text 2", "Text 3"]
            results = await embed_batch_concurrent(texts, max_concurrent=2)

            assert len(results) == 3
            for result in results:
                assert len(result) == EMBEDDING_DIMENSION

    @pytest.mark.asyncio
    async def test_embed_batch_concurrent_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await embed_batch_concurrent([])


class TestValidateEmbedding:
    """Tests for validate_embedding function."""

    def test_valid_embedding(self) -> None:
        """Test validation of correct embedding."""
        embedding = generate_mock_embedding()
        assert validate_embedding(embedding) is True

    def test_wrong_dimension(self) -> None:
        """Test validation fails for wrong dimension."""
        embedding = [0.1] * 100
        assert validate_embedding(embedding) is False

    def test_empty_embedding(self) -> None:
        """Test validation fails for empty list."""
        assert validate_embedding([]) is False

    def test_not_a_list(self) -> None:
        """Test validation fails for non-list."""
        assert validate_embedding("not a list") is False
        assert validate_embedding(None) is False

    def test_non_numeric_values(self) -> None:
        """Test validation fails for non-numeric values."""
        embedding = ["a"] * EMBEDDING_DIMENSION
        assert validate_embedding(embedding) is False

    def test_mixed_types(self) -> None:
        """Test validation allows mixed int/float."""
        embedding = [1] * (EMBEDDING_DIMENSION // 2) + [0.5] * (EMBEDDING_DIMENSION // 2)
        assert validate_embedding(embedding) is True


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors is 1."""
        vec = generate_mock_embedding()
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors is 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors is -1."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.0001

    def test_different_lengths_raises_error(self) -> None:
        """Test that different length vectors raise ValueError."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity(vec1, vec2)

    def test_zero_vector(self) -> None:
        """Test similarity with zero vector is 0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 1.0, 1.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_similar_vectors(self) -> None:
        """Test similar vectors have high similarity."""
        vec1 = generate_mock_embedding(1)
        vec2 = generate_mock_embedding(1)  # Same seed = identical
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99

    def test_different_vectors(self) -> None:
        """Test different vectors have lower similarity."""
        vec1 = generate_mock_embedding(1)
        vec2 = generate_mock_embedding(2)
        similarity = cosine_similarity(vec1, vec2)
        # Random vectors should have some but not perfect similarity
        assert -1 <= similarity <= 1
