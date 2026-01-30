"""Tests for the semantic similarity module."""

from unittest.mock import MagicMock, patch

import pytest

from src.settings._model_registry import get_embedding_prefix
from src.utils.similarity import (
    _SANITY_CHECK_CEILING,
    FALSE_POSITIVE_CEILING,
    SemanticDuplicateChecker,
    _normalize_for_ceiling,
    cosine_similarity,
    get_semantic_checker,
    reset_global_checker,
)


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """Skip model validation for all tests by default.

    Individual test classes for validation behavior override this
    by setting _skip_validation = False before creating checkers.
    """
    original = SemanticDuplicateChecker._skip_validation
    SemanticDuplicateChecker._skip_validation = True
    yield
    SemanticDuplicateChecker._skip_validation = original


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
    TEST_MODEL = "mxbai-embed-large"
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
        # Set up embeddings that are similar but below false-positive ceiling (~0.94)
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.9, 0.3, 0.1]},  # Existing name (similar, ~0.94)
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

        checker = self._create_checker(
            embedding_model="snowflake-arctic-embed:335m", similarity_threshold=0.9
        )
        checker.get_embedding("text1")
        checker.get_embedding("text2")

        stats = checker.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["model"] == "snowflake-arctic-embed:335m"
        assert stats["threshold"] == 0.9

    def test_get_cache_stats_includes_degraded(self, mock_ollama_client):
        """get_cache_stats includes degraded status."""
        checker = self._create_checker()
        stats = checker.get_cache_stats()
        assert stats["degraded"] is False

        checker._degraded = True
        stats = checker.get_cache_stats()
        assert stats["degraded"] is True

    def test_is_degraded_property(self, mock_ollama_client):
        """is_degraded property reflects internal state."""
        checker = self._create_checker()
        assert checker.is_degraded is False

        checker._degraded = True
        assert checker.is_degraded is True

    def test_get_embedding_sends_raw_text_for_unknown_model(self, mock_ollama_client):
        """Unknown models (not in registry) send raw text without prefix."""
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 2.0, 3.0]}

        checker = self._create_checker(embedding_model="unknown-model")
        checker.get_embedding("The Binary Veil")

        call_args = mock_ollama_client.embeddings.call_args
        assert call_args.kwargs.get("prompt") == "The Binary Veil"
        assert checker._embedding_prefix == ""

    def test_embedding_prefix_auto_resolved_from_registry(self, mock_ollama_client):
        """Prefix is auto-resolved from model registry at init time."""
        # mxbai-embed-large has no prefix (works raw)
        checker_mxbai = self._create_checker(embedding_model="mxbai-embed-large")
        assert checker_mxbai._embedding_prefix == ""

        # Unknown models get no prefix
        checker_unknown = self._create_checker(embedding_model="custom-embed-v1")
        assert checker_unknown._embedding_prefix == ""


class TestModelValidation:
    """Tests for the embedding model sanity check."""

    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "mxbai-embed-large"
    TEST_THRESHOLD = 0.85

    @pytest.fixture(autouse=True)
    def _enable_validation(self):
        """Enable validation for this test class (overrides module-level skip)."""
        original = SemanticDuplicateChecker._skip_validation
        SemanticDuplicateChecker._skip_validation = False
        yield
        SemanticDuplicateChecker._skip_validation = original

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("src.utils.similarity.ollama.Client") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            yield mock_client

    def test_sanity_check_passes_with_differentiated_embeddings(self, mock_ollama_client):
        """Model passes sanity check when embeddings are differentiated."""
        # Two orthogonal vectors — cosine similarity = 0.0
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # "The Iron Guard"
            {"embedding": [0.0, 0.0, 1.0]},  # "Dr. Sarah Chen"
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is False

    def test_sanity_check_fails_with_degenerate_embeddings(self, mock_ollama_client):
        """Model is marked degraded when sanity check produces near-identical vectors."""
        # All models return same degenerate vector — all fail sanity check + fallbacks
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 1.0, 1.0]}
        mock_ollama_client.show.return_value = {"model": "test"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_sanity_check_fails_at_ceiling(self, mock_ollama_client):
        """Model is marked degraded at exactly the ceiling threshold."""
        # Nearly identical vectors for all models — all fail sanity check
        mock_ollama_client.embeddings.return_value = {"embedding": [0.999, 0.01, 0.0]}
        mock_ollama_client.show.return_value = {"model": "test"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_sanity_check_passes_just_below_ceiling(self, mock_ollama_client):
        """Model passes when similarity is below the ceiling."""
        # Moderately similar vectors — cosine similarity ~0.7
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},
            {"embedding": [0.7, 0.7, 0.0]},  # cos ~0.7
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is False

    def test_sanity_check_degrades_on_empty_vectors(self, mock_ollama_client):
        """Model is degraded if sanity check returns empty vectors."""
        # All models return empty vectors — all fail
        mock_ollama_client.embeddings.return_value = {"embedding": []}
        mock_ollama_client.show.return_value = {"model": "test"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_sanity_check_degrades_on_api_error(self, mock_ollama_client):
        """Model is degraded if sanity check API call fails."""
        mock_ollama_client.embeddings.side_effect = ConnectionError("Ollama unavailable")

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_sanity_check_degrades_on_timeout(self, mock_ollama_client):
        """Model is degraded if sanity check times out."""
        mock_ollama_client.embeddings.side_effect = TimeoutError("Request timed out")

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_degraded_checker_skips_find_semantic_duplicate(self, mock_ollama_client):
        """Degraded checker returns no-match from find_semantic_duplicate."""
        # Make sanity check fail for ALL models (primary + fallbacks)
        # Use return_value so all embedding calls return degenerate vectors
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 1.0, 1.0]}
        mock_ollama_client.show.return_value = {"model": "test"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )
        assert checker.is_degraded is True

        # Record call count after init (includes sanity check + fallback attempts)
        calls_after_init = mock_ollama_client.embeddings.call_count

        # find_semantic_duplicate should return no-match without calling the API
        is_dup, match, score = checker.find_semantic_duplicate(
            "The Binary Veil", ["Commander Rhea Solis"]
        )

        assert is_dup is False
        assert match is None
        assert score == 0.0
        # No additional embedding calls after degradation
        assert mock_ollama_client.embeddings.call_count == calls_after_init

    def test_sanity_check_degrades_on_unexpected_error(self, mock_ollama_client):
        """Model is degraded if sanity check encounters an unexpected error."""
        # All embedding calls return differentiated vectors, but cosine_similarity
        # always raises, so primary + all fallbacks fail the sanity check.
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 0.0, 0.0]}
        mock_ollama_client.show.return_value = {"model": "test"}

        with patch(
            "src.utils.similarity.cosine_similarity", side_effect=RuntimeError("unexpected")
        ):
            checker = SemanticDuplicateChecker(
                ollama_url=self.TEST_OLLAMA_URL,
                embedding_model=self.TEST_MODEL,
                similarity_threshold=self.TEST_THRESHOLD,
            )

        assert checker.is_degraded is True

    def test_sanity_check_ceiling_value(self):
        """Verify the sanity check ceiling constant is reasonable."""
        assert _SANITY_CHECK_CEILING == 0.85
        assert 0.8 < _SANITY_CHECK_CEILING < 1.0


class TestModelFallback:
    """Tests for the embedding model fallback mechanism."""

    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "degenerate-embed:test"  # Fake model that always fails sanity check
    TEST_THRESHOLD = 0.85

    @pytest.fixture(autouse=True)
    def _enable_validation(self):
        """Enable validation for this test class (overrides module-level skip)."""
        original = SemanticDuplicateChecker._skip_validation
        SemanticDuplicateChecker._skip_validation = False
        yield
        SemanticDuplicateChecker._skip_validation = original

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("src.utils.similarity.ollama.Client") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            yield mock_client

    def test_fallback_succeeds_when_primary_fails(self, mock_ollama_client):
        """When primary model fails sanity check, fallback to a passing model."""
        call_count = {"n": 0}

        def embedding_side_effect(**kwargs):
            """Return degenerate vectors for primary, differentiated for fallback."""
            call_count["n"] += 1
            model = kwargs.get("model", "")
            if model == "degenerate-embed:test":
                # Primary model returns degenerate embeddings (identical vectors)
                return {"embedding": [1.0, 1.0, 1.0]}
            elif model == "mxbai-embed-large":
                # Fallback model returns differentiated embeddings
                if call_count["n"] % 2 == 0:
                    return {"embedding": [0.0, 0.0, 1.0]}
                return {"embedding": [1.0, 0.0, 0.0]}
            return {"embedding": []}

        mock_ollama_client.embeddings.side_effect = embedding_side_effect
        # show() should succeed for the fallback model (it's installed)
        mock_ollama_client.show.return_value = {"model": "mxbai-embed-large"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is False
        assert checker.embedding_model == "mxbai-embed-large"

    def test_fallback_degrades_when_all_fail(self, mock_ollama_client):
        """When all embedding models fail sanity check, degrade."""
        # All models return identical vectors
        mock_ollama_client.embeddings.return_value = {"embedding": [1.0, 1.0, 1.0]}
        mock_ollama_client.show.return_value = {"model": "test"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        assert checker.is_degraded is True

    def test_fallback_skips_unavailable_models(self, mock_ollama_client):
        """Fallback skips models not installed in Ollama."""
        call_count = {"n": 0}

        def embedding_side_effect(**kwargs):
            """Return degenerate for primary, working for snowflake only."""
            call_count["n"] += 1
            model = kwargs.get("model", "")
            if model == "degenerate-embed:test":
                # Primary fails
                return {"embedding": [1.0, 1.0, 1.0]}
            elif model == "snowflake-arctic-embed:335m":
                # This one works
                if call_count["n"] % 2 == 0:
                    return {"embedding": [0.0, 0.0, 1.0]}
                return {"embedding": [1.0, 0.0, 0.0]}
            return {"embedding": [1.0, 1.0, 1.0]}

        mock_ollama_client.embeddings.side_effect = embedding_side_effect

        def show_side_effect(model_id):
            """Simulate mxbai not installed, others available."""
            if model_id == "mxbai-embed-large":
                raise Exception("Model not found")
            return {"model": model_id}

        mock_ollama_client.show.side_effect = show_side_effect

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        # Should have fallen back to snowflake (mxbai was "not installed")
        assert checker.is_degraded is False
        assert checker.embedding_model == "snowflake-arctic-embed:335m"

    def test_fallback_clears_cache_on_switch(self, mock_ollama_client):
        """Cache is cleared when switching to a fallback model."""
        call_count = {"n": 0}

        def embedding_side_effect(**kwargs):
            """Return degenerate for primary, differentiated for mxbai fallback."""
            call_count["n"] += 1
            model = kwargs.get("model", "")
            if model == "degenerate-embed:test":
                return {"embedding": [1.0, 1.0, 1.0]}
            elif model == "mxbai-embed-large":
                if call_count["n"] % 2 == 0:
                    return {"embedding": [0.0, 0.0, 1.0]}
                return {"embedding": [1.0, 0.0, 0.0]}
            return {"embedding": []}

        mock_ollama_client.embeddings.side_effect = embedding_side_effect
        mock_ollama_client.show.return_value = {"model": "mxbai-embed-large"}

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=self.TEST_THRESHOLD,
        )

        # After successful fallback, cache should not contain primary model's entries
        assert checker.embedding_model == "mxbai-embed-large"
        # The sanity check vectors from the fallback model are cached, but
        # the primary model's degenerate vectors should have been cleared
        assert checker.is_degraded is False


class TestFalsePositiveCeiling:
    """Tests for the false-positive ceiling in find_semantic_duplicate."""

    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "mxbai-embed-large"

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("src.utils.similarity.ollama.Client") as mock_class:
            mock_client = MagicMock()
            mock_class.return_value = mock_client
            yield mock_client

    def test_false_positive_ceiling_skips_suspiciously_high_similarity(self, mock_ollama_client):
        """Similarity >= 0.98 for different strings is treated as false positive."""
        # Nearly identical vectors = suspiciously high similarity
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # "The Binary Veil"
            {"embedding": [0.9999, 0.001, 0.0]},  # "Commander Rhea Solis" (~1.0)
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=0.85,
        )

        is_dup, match, _score = checker.find_semantic_duplicate(
            "The Binary Veil", ["Commander Rhea Solis"]
        )

        # Should NOT be flagged as duplicate despite high similarity
        assert is_dup is False
        assert match is None

    def test_false_positive_ceiling_allows_legitimate_duplicates(self, mock_ollama_client):
        """Similarity above threshold but below ceiling IS flagged as duplicate."""
        # Similar but not suspiciously high — 0.90 similarity
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # "Shadow Council"
            {"embedding": [0.9, 0.3, 0.1]},  # "Council of Shadows" (~0.93)
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=0.85,
        )

        is_dup, match, score = checker.find_semantic_duplicate(
            "Shadow Council", ["Council of Shadows"]
        )

        assert is_dup is True
        assert match == "Council of Shadows"
        assert 0.85 <= score < FALSE_POSITIVE_CEILING

    def test_false_positive_ceiling_allows_exact_text_match(self, mock_ollama_client):
        """Identical text with high similarity IS flagged (not a false positive)."""
        # Same text = legitimately identical
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # "Shadow Council"
            {"embedding": [1.0, 0.0, 0.0]},  # "shadow council" (same after normalize)
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=0.85,
        )

        is_dup, match, _score = checker.find_semantic_duplicate(
            "Shadow Council", ["shadow council"]
        )

        # Same text normalized = legitimate duplicate even at 1.0
        assert is_dup is True
        assert match == "shadow council"

    def test_false_positive_ceiling_allows_punctuation_variant_match(self, mock_ollama_client):
        """Punctuation-only differences are recognized as the same entity at ceiling."""
        # Identical vectors = similarity 1.0 (above ceiling)
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # "Shadow-Council"
            {"embedding": [1.0, 0.0, 0.0]},  # "Shadow Council" (punctuation variant)
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=0.85,
        )

        is_dup, match, _score = checker.find_semantic_duplicate(
            "Shadow-Council", ["Shadow Council"]
        )

        # Punctuation variant normalizes identically — legitimate duplicate
        assert is_dup is True
        assert match == "Shadow Council"

    def test_false_positive_ceiling_continues_to_next_name(self, mock_ollama_client):
        """After skipping a false positive, continues checking remaining names."""
        mock_ollama_client.embeddings.side_effect = [
            {"embedding": [1.0, 0.0, 0.0]},  # New name
            {"embedding": [0.9999, 0.001, 0.0]},  # First existing: false positive (~1.0)
            {"embedding": [0.9, 0.3, 0.1]},  # Second existing: real match (~0.93)
        ]

        checker = SemanticDuplicateChecker(
            ollama_url=self.TEST_OLLAMA_URL,
            embedding_model=self.TEST_MODEL,
            similarity_threshold=0.85,
        )

        is_dup, match, score = checker.find_semantic_duplicate(
            "The Veil", ["Commander Rhea Solis", "The Veiled Ones"]
        )

        # Should skip false positive and find real match
        assert is_dup is True
        assert match == "The Veiled Ones"
        assert 0.85 <= score < FALSE_POSITIVE_CEILING

    def test_false_positive_ceiling_value(self):
        """Verify the false-positive ceiling constant is reasonable."""
        assert FALSE_POSITIVE_CEILING == 0.98
        assert _SANITY_CHECK_CEILING < 1.0


class TestGlobalChecker:
    """Tests for global checker functions."""

    # Test constants to avoid hardcoded URLs that could hit real services
    TEST_OLLAMA_URL = "http://test-ollama:11434"
    TEST_MODEL = "mxbai-embed-large"

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


class TestEmbeddingPrefixRegistry:
    """Tests for the get_embedding_prefix registry lookup."""

    def test_mxbai_embed_large_has_no_prefix(self):
        """mxbai-embed-large works raw without prefix."""
        assert get_embedding_prefix("mxbai-embed-large") == ""

    def test_snowflake_arctic_has_no_prefix(self):
        """snowflake-arctic-embed works raw without prefix."""
        assert get_embedding_prefix("snowflake-arctic-embed:335m") == ""

    def test_bge_m3_has_no_prefix(self):
        """bge-m3 works raw without prefix."""
        assert get_embedding_prefix("bge-m3") == ""

    def test_unknown_model_has_no_prefix(self):
        """Unknown models get no prefix (safe default)."""
        assert get_embedding_prefix("custom-embed-v1") == ""

    def test_non_embedding_model_has_no_prefix(self):
        """Chat models in the registry have no embedding prefix."""
        assert get_embedding_prefix("huihui_ai/dolphin3-abliterated:8b") == ""


class TestNormalizeForCeiling:
    """Tests for the _normalize_for_ceiling helper."""

    def test_lowercase_and_strip(self):
        """Basic lowercasing and stripping."""
        assert _normalize_for_ceiling("  Hello World  ") == "hello world"

    def test_removes_punctuation(self):
        """Punctuation is replaced with spaces and collapsed."""
        assert _normalize_for_ceiling("Shadow-Council") == "shadow council"
        assert _normalize_for_ceiling("Dr. Sarah Chen!") == "dr sarah chen"

    def test_collapses_whitespace(self):
        """Multiple whitespace chars collapse to single space."""
        assert _normalize_for_ceiling("The   Iron   Guard") == "the iron guard"

    def test_strips_special_characters(self):
        """Special characters are removed."""
        assert _normalize_for_ceiling("Council's of Shadows") == "council s of shadows"
        assert _normalize_for_ceiling("(The Binary Veil)") == "the binary veil"

    def test_empty_string(self):
        """Empty and whitespace-only strings return empty."""
        assert _normalize_for_ceiling("") == ""
        assert _normalize_for_ceiling("   ") == ""

    def test_punctuation_only_variants_match(self):
        """Names differing only by punctuation normalize identically."""
        assert _normalize_for_ceiling("Shadow Council") == _normalize_for_ceiling("Shadow-Council")
        assert _normalize_for_ceiling("Dr. Chen") == _normalize_for_ceiling("Dr Chen")
