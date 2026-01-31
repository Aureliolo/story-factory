"""Semantic similarity detection using embeddings.

Provides embedding-based duplicate detection to catch semantic duplicates
that string-based methods miss (e.g., "Shadow Council" vs "Council of Shadows").
"""

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

import ollama

from src.settings._model_registry import get_embedding_prefix

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.
        Returns 0.0 if either vector is empty or has zero magnitude.
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        logger.warning(
            "Embedding dimension mismatch: %d vs %d, returning 0.0", len(vec1), len(vec2)
        )
        return 0.0

    # Calculate dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# Sanity check reference strings — short entity names that are obviously different.
# These mimic the actual use case (faction names vs character names, 2-4 words).
# If a model returns high similarity for these, it can't differentiate short strings
# and will produce false positives in production.
_SANITY_CHECK_A = "The Iron Guard"
_SANITY_CHECK_B = "Dr. Sarah Chen"
_SANITY_CHECK_CEILING = 0.85  # Above this for obviously-different names = degenerate

# False-positive ceiling — if two different strings score above this,
# it's almost certainly a model artifact, not a real semantic duplicate.
# Real duplicates ("Shadow Council" vs "Council of Shadows") score 0.85-0.95.
FALSE_POSITIVE_CEILING = 0.98


def _normalize_for_ceiling(text: str) -> str:
    """Normalize text for false-positive ceiling identity checks.

    Strips punctuation, collapses whitespace, and lowercases so that
    near-identical strings like ``"Shadow-Council"`` and ``"Shadow Council"``
    are recognised as the same entity before the ceiling filter fires.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class SemanticDuplicateChecker:
    """Checks for semantic duplicates using Ollama embeddings.

    Caches embeddings to avoid redundant API calls and provides
    thread-safe operations for concurrent usage.

    Some embedding models require a specific prompt prefix for optimal
    results.  The prefix is auto-resolved from the model registry at init
    time and applied transparently in :meth:`get_embedding`.

    Includes a sanity check on initialization: two obviously-different
    entity names are embedded (through :meth:`get_embedding`, so the
    prefix is applied) and their cosine similarity is compared to a
    ceiling.  If the similarity exceeds the ceiling, the model is
    marked as degraded and all semantic checks are skipped.

    All parameters are required to avoid accidental use of hardcoded URLs
    that could hit real services in tests.

    Attributes:
        ollama_url: URL of the Ollama server (required).
        embedding_model: Model to use for generating embeddings (required).
        similarity_threshold: Cosine similarity threshold for duplicates (required).
        timeout: Timeout for Ollama API calls in seconds.
    """

    # Testing hook: skip model validation during unit tests to avoid needing
    # to mock the sanity check in every test. Production code never sets this.
    _skip_validation: ClassVar[bool] = False

    ollama_url: str
    embedding_model: str
    similarity_threshold: float
    timeout: float = 30.0

    # Embedding cache: maps text to embedding vector
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _client: ollama.Client | None = field(default=None, init=False)
    _degraded: bool = field(default=False, init=False)
    _embedding_prefix: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Initialize the Ollama client, resolve model prefix, and validate."""
        self._client = ollama.Client(host=self.ollama_url, timeout=self.timeout)
        self._embedding_prefix = get_embedding_prefix(self.embedding_model)
        logger.info(
            "SemanticDuplicateChecker initialized: model=%s, threshold=%.2f, prefix=%r",
            self.embedding_model,
            self.similarity_threshold,
            self._embedding_prefix,
        )
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate that the embedding model produces meaningful vectors.

        Embeds two obviously-different entity names through :meth:`get_embedding`
        (which applies the model's prompt prefix) and checks their cosine
        similarity. This tests the exact same code path used in production.

        If the similarity exceeds the ceiling, tries fallback embedding models
        from the registry before degrading.
        """
        if self._skip_validation:
            logger.debug("Skipping embedding model validation (test mode)")
            return

        try:
            vec_a = self.get_embedding(_SANITY_CHECK_A)
            vec_b = self.get_embedding(_SANITY_CHECK_B)

            if not vec_a or not vec_b:
                logger.warning(
                    "Embedding model '%s' returned empty vectors during sanity check — "
                    "attempting fallback models",
                    self.embedding_model,
                )
                if not self._try_fallback_model():
                    self._degraded = True
                return

            similarity = cosine_similarity(vec_a, vec_b)
            if similarity >= _SANITY_CHECK_CEILING:
                logger.error(
                    "Embedding model '%s' FAILED sanity check: "
                    "'%s' vs '%s' returned similarity %.3f (>= %.2f ceiling). "
                    "This model cannot differentiate short entity names. "
                    "Attempting fallback models.",
                    self.embedding_model,
                    _SANITY_CHECK_A,
                    _SANITY_CHECK_B,
                    similarity,
                    _SANITY_CHECK_CEILING,
                )
                if not self._try_fallback_model():
                    logger.error(
                        "No fallback embedding model passed sanity check. "
                        "Semantic duplicate detection is DISABLED."
                    )
                    self._degraded = True
            else:
                logger.info(
                    "Embedding model '%s' passed sanity check "
                    "(similarity=%.3f, ceiling=%.2f, prefix=%r)",
                    self.embedding_model,
                    similarity,
                    _SANITY_CHECK_CEILING,
                    self._embedding_prefix,
                )

        except Exception as e:
            logger.warning(
                "Embedding model sanity check failed for '%s': %s — attempting fallback models",
                self.embedding_model,
                e,
            )
            if not self._try_fallback_model():
                self._degraded = True

    def _revert_model(self, old_model: str, old_prefix: str) -> None:
        """Revert to a previous embedding model and clear the cache.

        Used during fallback iteration when a candidate model fails validation,
        restoring the prior model state before trying the next candidate.

        Args:
            old_model: Model ID to restore.
            old_prefix: Embedding prefix to restore.
        """
        self.embedding_model = old_model
        self._embedding_prefix = old_prefix
        self.clear_cache()

    def _try_fallback_model(self) -> bool:
        """Try alternative embedding models from the registry when primary fails.

        Iterates through RECOMMENDED_MODELS with the "embedding" tag, skips the
        current model, checks if each is installed via Ollama, and validates it
        with the sanity check. Switches to the first model that passes.

        Returns:
            True if a fallback model was found and activated, False otherwise.
        """
        from src.settings._model_registry import RECOMMENDED_MODELS

        for model_id, info in RECOMMENDED_MODELS.items():
            if "embedding" not in info.get("tags", []):
                continue
            if model_id == self.embedding_model:
                continue

            # Check if model is installed
            try:
                if self._client is None:  # pragma: no cover - defensive, set in __post_init__
                    self._client = ollama.Client(host=self.ollama_url, timeout=self.timeout)
                self._client.show(model_id)
            except Exception:
                logger.debug("Fallback model '%s' not installed, skipping", model_id)
                continue

            # Clear cache from previous model's embeddings
            self.clear_cache()

            # Try this model
            old_model = self.embedding_model
            old_prefix = self._embedding_prefix
            self.embedding_model = model_id
            self._embedding_prefix = get_embedding_prefix(model_id)

            try:
                vec_a = self.get_embedding(_SANITY_CHECK_A)
                vec_b = self.get_embedding(_SANITY_CHECK_B)

                if not vec_a or not vec_b:
                    logger.debug("Fallback model '%s' returned empty vectors", model_id)
                    self._revert_model(old_model, old_prefix)
                    continue

                similarity = cosine_similarity(vec_a, vec_b)
                if similarity < _SANITY_CHECK_CEILING:
                    logger.info(
                        "Fallback embedding model '%s' passed sanity check "
                        "(similarity=%.3f, ceiling=%.2f). Switching from '%s'.",
                        model_id,
                        similarity,
                        _SANITY_CHECK_CEILING,
                        old_model,
                    )
                    return True
                else:
                    logger.debug(
                        "Fallback model '%s' also failed sanity check (similarity=%.3f)",
                        model_id,
                        similarity,
                    )
                    self._revert_model(old_model, old_prefix)

            except Exception as e:
                logger.debug("Fallback model '%s' failed: %s", model_id, e)
                self._revert_model(old_model, old_prefix)

        return False

    @property
    def is_degraded(self) -> bool:
        """Whether the embedding model failed validation and semantic checks are disabled."""
        return self._degraded

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text, using cache if available.

        Applies the model-specific prompt prefix (if any) before calling the
        embedding API.  The prefix is resolved from the model registry at init
        time, so each model is prompted correctly without the caller needing to
        know about prefixes.

        Args:
            text: Text to embed (raw, without prefix).

        Returns:
            Embedding vector as list of floats.
            Returns empty list if embedding fails.
        """
        if not text or not text.strip():
            return []

        # Normalize text for caching (prefix-independent — same text always
        # maps to the same cache entry within a checker instance)
        normalized = text.lower().strip()

        with self._lock:
            if normalized in self._cache:
                logger.debug("Cache hit for embedding: '%s...'", text[:30])
                return self._cache[normalized]

        # Generate embedding
        try:
            if self._client is None:
                self._client = ollama.Client(host=self.ollama_url, timeout=self.timeout)

            # Apply model-specific prompt prefix from the registry (if configured).
            # Models without a prefix get raw text.
            prompt = f"{self._embedding_prefix}{text}"
            response = self._client.embeddings(model=self.embedding_model, prompt=prompt)

            embedding: list[float] = response.get("embedding", [])
            if not embedding:
                logger.warning("Empty embedding returned for text: '%s...'", text[:30])
                return []

            # Cache the result
            with self._lock:
                self._cache[normalized] = embedding

            logger.debug(
                "Generated embedding for '%s...' (%d dimensions)", text[:30], len(embedding)
            )
            return embedding

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning("Failed to generate embedding for '%s...': %s", text[:30], e)
            return []
        except Exception as e:
            logger.exception("Unexpected error generating embedding: %s", e)
            return []

    def check_similarity(self, name1: str, name2: str) -> float:
        """Check cosine similarity between two names.

        Args:
            name1: First name to compare.
            name2: Second name to compare.

        Returns:
            Cosine similarity score between 0 and 1.
            Returns 0.0 if either embedding fails.
        """
        emb1 = self.get_embedding(name1)
        emb2 = self.get_embedding(name2)

        if not emb1 or not emb2:
            return 0.0

        similarity = cosine_similarity(emb1, emb2)
        logger.debug("Similarity between '%s' and '%s': %.3f", name1, name2, similarity)
        return similarity

    def find_semantic_duplicate(
        self, name: str, existing_names: list[str]
    ) -> tuple[bool, str | None, float]:
        """Find if a name is semantically similar to any existing name.

        Args:
            name: Name to check.
            existing_names: List of existing names to compare against.

        Returns:
            Tuple of (is_duplicate, matching_name, best_similarity).
            If no duplicate found, returns (False, None, best_similarity)
            where best_similarity is the highest similarity score found
            (may be non-zero even when is_duplicate is False).
        """
        if not name or not existing_names:
            return False, None, 0.0

        # Skip if model failed sanity check
        if self._degraded:
            logger.debug("Skipping semantic check for '%s' — embedding model is degraded", name)
            return False, None, 0.0

        # Get embedding for the new name
        new_embedding = self.get_embedding(name)
        if not new_embedding:
            logger.debug("Could not get embedding for '%s', skipping semantic check", name)
            return False, None, 0.0

        best_match: str | None = None
        best_similarity: float = 0.0

        for existing in existing_names:
            if not existing or not existing.strip():
                continue

            existing_embedding = self.get_embedding(existing)
            if not existing_embedding:
                continue

            similarity = cosine_similarity(new_embedding, existing_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing

            # Early exit if we find a clear duplicate
            if similarity >= self.similarity_threshold:
                # False-positive ceiling: if similarity is suspiciously high
                # (>= 0.98) for two strings that aren't near-identical by text,
                # the model is producing degenerate embeddings for these inputs.
                # Real semantic duplicates ("Shadow Council" vs "Council of
                # Shadows") score 0.85-0.95, not 0.98+.
                if similarity >= FALSE_POSITIVE_CEILING:
                    normalized_new = _normalize_for_ceiling(name)
                    normalized_existing = _normalize_for_ceiling(existing)
                    if normalized_new != normalized_existing:
                        logger.warning(
                            "Ignoring likely false positive: '%s' vs '%s' "
                            "scored %.3f (>= %.2f ceiling). "
                            "The embedding model may not differentiate short strings well.",
                            name,
                            existing,
                            similarity,
                            FALSE_POSITIVE_CEILING,
                        )
                        continue

                logger.info(
                    "Semantic duplicate detected: '%s' similar to '%s' (%.3f >= %.3f)",
                    name,
                    existing,
                    similarity,
                    self.similarity_threshold,
                )
                return True, existing, similarity

        if best_match:
            logger.debug(
                "Best semantic match for '%s': '%s' (%.3f, below threshold %.3f)",
                name,
                best_match,
                best_similarity,
                self.similarity_threshold,
            )

        return False, None, best_similarity

    def clear_cache(self) -> None:
        """Clear the embedding cache.

        Useful when testing or after significant changes to the entity set.
        """
        with self._lock:
            self._cache.clear()
        logger.debug("Cleared embedding cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and other stats.
        """
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "model": self.embedding_model,
                "threshold": self.similarity_threshold,
                "degraded": self._degraded,
            }


# Global checker cache keyed by settings tuple (supports per-call settings)
_checker_cache: dict[tuple[str, str, float, float], SemanticDuplicateChecker] = {}
_checker_lock = threading.Lock()


def get_semantic_checker(
    ollama_url: str,
    embedding_model: str,
    similarity_threshold: float,
    timeout: float = 30.0,
) -> SemanticDuplicateChecker:
    """Get or create a semantic duplicate checker for the given settings.

    Checkers are cached by their settings tuple to avoid recreating
    instances with identical configurations while still supporting
    different settings when needed.

    Args:
        ollama_url: Ollama server URL (required).
        embedding_model: Embedding model to use (required).
        similarity_threshold: Similarity threshold for duplicates (required).
        timeout: API timeout in seconds.

    Returns:
        A SemanticDuplicateChecker instance configured with the given settings.
    """
    cache_key = (ollama_url, embedding_model, similarity_threshold, timeout)

    with _checker_lock:
        if cache_key not in _checker_cache:
            logger.debug(
                "Creating new SemanticDuplicateChecker: url=%s, model=%s, threshold=%.2f",
                ollama_url,
                embedding_model,
                similarity_threshold,
            )
            _checker_cache[cache_key] = SemanticDuplicateChecker(
                ollama_url=ollama_url,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                timeout=timeout,
            )
        return _checker_cache[cache_key]


def reset_global_checker() -> None:
    """Reset all cached semantic checkers.

    Useful for testing to ensure a clean state.
    """
    with _checker_lock:
        for checker in _checker_cache.values():
            checker.clear_cache()
        _checker_cache.clear()
    logger.debug("Reset all cached semantic checkers")
