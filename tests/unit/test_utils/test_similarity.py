"""Tests for the semantic similarity module."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.similarity import (
    SemanticDuplicateChecker,
    cosine_similarity,
    get_semantic_checker,
    reset_global_checker,
)


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors_return_one(self):
        """Identical vectors have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors have similarity of 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors have similarity of -1.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_empty_vectors_return_zero(self):
        """Empty vectors return similarity of 0.0."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0, 2.0], []) == 0.0
        assert cosine_similarity([], [1.0, 2.0]) == 0.0

    def test_zero_magnitude_vectors_return_zero(self):
        """Zero magnitude vectors return similarity of 0.0."""
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_mismatched_dimensions_return_zero(self):
        """Mismatched dimensions return similarity of 0.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_similar_vectors_have_high_similarity(self):
        """Similar vectors have high similarity score."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99


class TestSemanticDuplicateChecker:
    """Tests for the SemanticDuplicateChecker class."""

    # Test constants to avoid hardcoded URLs that could hit real services
    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "test-embed-model"
    TEST_THRESHOLD = 0.85

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("src.utils.similarity.ollama.Client") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            yield mock_client

    def _create_checker(self, **kwargs: float | str) -> SemanticDuplicateChecker:
        """Create a checker with test defaults. Override any param via kwargs."""
        return SemanticDuplicateChecker(
            ollama_url=str(kwargs.get("ollama_url", self.TEST_OLLAMA_URL)),
            embedding_model=str(kwargs.get("embedding_model", self.TEST_MODEL)),
            similarity_threshold=float(kwargs.get("similarity_threshold", self.TEST_THRESHOLD)),
        )

    def test_get_embedding_caches_results(self, mock_ollama_client):
        """Embeddings are cached to avoid redundant API calls."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker()

        # First call - should hit API
        emb1 = checker.get_embedding("test text")
        assert emb1 == [1.0, 2.0, 3.0]
        assert mock_ollama_client.embeddings.call_count == 1

        # Second call with same text - should use cache
        emb2 = checker.get_embedding("test text")
        assert emb2 == [1.0, 2.0, 3.0]
        assert mock_ollama_client.embeddings.call_count == 1  # Still 1

    def test_get_embedding_normalizes_text(self, mock_ollama_client):
        """Text is normalized for caching (lowercase, stripped)."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker()

        # First call
        emb1 = checker.get_embedding("Test Text")
        # Second call with different case - should use cache
        emb2 = checker.get_embedding("test text")

        assert emb1 == emb2
        assert mock_ollama_client.embeddings.call_count == 1

    def test_get_embedding_returns_empty_for_empty_input(self, mock_ollama_client):
        """Empty input returns empty embedding without API call."""
        checker = self._create_checker()

        assert checker.get_embedding("") == []
        assert checker.get_embedding("   ") == []
        assert mock_ollama_client.embeddings.call_count == 0

    def test_get_embedding_handles_api_error(self, mock_ollama_client):
        """API errors return empty embedding without crashing."""
        mock_ollama_client.embeddings.side_effect = ConnectionError("API down")

        checker = self._create_checker()
        emb = checker.get_embedding("test")

        assert emb == []

    def test_get_embedding_handles_empty_response(self, mock_ollama_client):
        """Empty embedding in API response returns empty list."""
        mock_ollama_client.embeddings.return_value = {"embedding": []}

        checker = self._create_checker()
        emb = checker.get_embedding("test")

        assert emb == []

    def test_get_embedding_handles_unexpected_exception(self, mock_ollama_client):
        """Unexpected exceptions are logged and return empty embedding."""
        mock_ollama_client.embeddings.side_effect = RuntimeError("Unexpected error")

        checker = self._create_checker()
        emb = checker.get_embedding("test")

        assert emb == []

    def test_get_embedding_reinitializes_client_if_none(self, mock_ollama_client):
        """Client is reinitialized if set to None."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker()
        checker._client = None  # Simulate client being cleared

        emb = checker.get_embedding("test")
        assert emb == [1.0, 2.0, 3.0]

    def test_check_similarity_returns_score(self, mock_ollama_client):
        """check_similarity returns cosine similarity between names."""
        # Shadow Council and Council of Shadows should be similar
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.5, 0.3]},  # "Shadow Council"
            {"embedding": [0.95, 0.55, 0.35]},  # "Council of Shadows"
        ]

        checker = self._create_checker()
        similarity = checker.check_similarity("Shadow Council", "Council of Shadows")

        assert similarity > 0.9  # Should be high

    def test_check_similarity_returns_zero_on_embedding_failure(self, mock_ollama_client):
        """check_similarity returns 0.0 if either embedding fails."""
        # First embedding succeeds, second fails
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.5, 0.3]},
            ConnectionError("API down"),
        ]

        checker = self._create_checker()
        similarity = checker.check_similarity("Shadow Council", "Council of Shadows")

        assert similarity == 0.0

    def test_find_semantic_duplicate_detects_similar_names(self, mock_ollama_client):
        """find_semantic_duplicate detects semantically similar names."""
        # Set up embeddings that are similar
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.95, 0.1, 0.05]},  # Existing name (very similar)
        ]

        checker = self._create_checker(similarity_threshold=0.85)
        is_dup, match, score = checker.find_semantic_duplicate(
            "Shadow Council", ["Council of Shadows"]
        )

        assert is_dup is True
        assert match == "Council of Shadows"
        assert score > 0.85

    def test_find_semantic_duplicate_allows_different_names(self, mock_ollama_client):
        """find_semantic_duplicate allows clearly different names."""
        # Set up embeddings that are very different
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.0, 1.0, 0.0]},  # Existing name (orthogonal = different)
        ]

        checker = self._create_checker(similarity_threshold=0.85)
        is_dup, match, _score = checker.find_semantic_duplicate(
            "Shadow Council", ["The Happy Bakers"]
        )

        assert is_dup is False
        assert match is None

    def test_find_semantic_duplicate_empty_inputs(self, mock_ollama_client):
        """Empty inputs are handled gracefully."""
        checker = self._create_checker()

        is_dup, _match, _score = checker.find_semantic_duplicate("", ["test"])
        assert is_dup is False

        is_dup, _match, _score = checker.find_semantic_duplicate("test", [])
        assert is_dup is False

    def test_find_semantic_duplicate_no_new_embedding(self, mock_ollama_client):
        """Returns no duplicate if new name embedding fails."""
        mock_ollama_client.embeddings.side_effect = ConnectionError("API down")

        checker = self._create_checker()
        is_dup, match, score = checker.find_semantic_duplicate("test", ["existing"])

        assert is_dup is False
        assert match is None
        assert score == 0.0

    def test_find_semantic_duplicate_skips_empty_existing_names(self, mock_ollama_client):
        """Empty or whitespace existing names are skipped."""
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.9, 0.1, 0.0]},  # "valid" existing name
        ]

        checker = self._create_checker(similarity_threshold=0.85)
        # Call without assigning - we only care about the API call count
        checker.find_semantic_duplicate("test", ["", "   ", "valid"])

        # Should skip empty names and only check "valid"
        assert mock_ollama_client.embeddings.call_count == 2

    def test_find_semantic_duplicate_skips_failed_existing_embedding(self, mock_ollama_client):
        """Existing names with failed embeddings are skipped."""
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            ConnectionError("API down"),  # First existing - fails
            {"embedding": [0.9, 0.1, 0.0]},  # Second existing - succeeds
        ]

        checker = self._create_checker(similarity_threshold=0.85)
        is_dup, _match, score = checker.find_semantic_duplicate("test", ["failing", "similar"])

        # Should continue to second existing name
        assert is_dup is True or score > 0.0  # Will match or return best score

    def test_find_semantic_duplicate_returns_best_match_below_threshold(self, mock_ollama_client):
        """Returns best match info even when below threshold."""
        # Set up embeddings that are somewhat similar but below threshold
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.6, 0.6, 0.6]},  # First existing (lower similarity)
            {"embedding": [0.7, 0.5, 0.5]},  # Second existing (higher similarity)
        ]

        checker = self._create_checker(similarity_threshold=0.95)  # High threshold
        is_dup, match, score = checker.find_semantic_duplicate(
            "test", ["less similar", "more similar"]
        )

        # Not a duplicate (below threshold), but best_similarity is tracked
        assert is_dup is False
        assert match is None  # No match returned when below threshold
        assert score > 0.0  # But score is returned

    def test_clear_cache(self, mock_ollama_client):
        """clear_cache removes cached embeddings."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker()
        checker.get_embedding("test")
        assert len(checker._cache) == 1

        checker.clear_cache()
        assert len(checker._cache) == 0

    def test_get_cache_stats(self, mock_ollama_client):
        """get_cache_stats returns correct information."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker(embedding_model="test-model", similarity_threshold=0.9)
        checker.get_embedding("text1")
        checker.get_embedding("text2")

        stats = checker.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["model"] == "test-model"
        assert stats["threshold"] == 0.9


class TestGlobalChecker:
    """Tests for global checker functions."""

    # Test constants to avoid hardcoded URLs that could hit real services
    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "test-embed-model"

    def setup_method(self):
        """Reset global state before each test."""
        reset_global_checker()

    def teardown_method(self):
        """Clean up global state after each test."""
        reset_global_checker()

    def test_get_semantic_checker_caches_by_settings(self):
        """get_semantic_checker returns same instance for same settings."""
        with patch("src.utils.similarity.ollama.Client"):
            checker1 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.85,
            )
            checker2 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.85,
            )
            assert checker1 is checker2

    def test_get_semantic_checker_creates_new_for_different_settings(self):
        """get_semantic_checker creates new instance for different settings."""
        with patch("src.utils.similarity.ollama.Client"):
            checker1 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.85,
            )
            checker2 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.90,  # Different threshold
            )
            # Different settings should create different instances
            assert checker1 is not checker2
            assert checker1.similarity_threshold == 0.85
            assert checker2.similarity_threshold == 0.90

    def test_reset_global_checker(self):
        """reset_global_checker clears all cached checkers."""
        with patch("src.utils.similarity.ollama.Client"):
            checker1 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.9,
            )
            reset_global_checker()

            checker2 = get_semantic_checker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=0.9,
            )
            # After reset, should be a new instance even with same settings
            assert checker1 is not checker2
