"""Semantic similarity detection using embeddings.

Provides embedding-based duplicate detection to catch semantic duplicates
that string-based methods miss (e.g., "Shadow Council" vs "Council of Shadows").
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import ollama

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


@dataclass
class SemanticDuplicateChecker:
    """Checks for semantic duplicates using Ollama embeddings.

    Caches embeddings to avoid redundant API calls and provides
    thread-safe operations for concurrent usage.

    All parameters are required to avoid accidental use of hardcoded URLs
    that could hit real services in tests.

    Attributes:
        ollama_url: URL of the Ollama server (required).
        embedding_model: Model to use for generating embeddings (required).
        similarity_threshold: Cosine similarity threshold for duplicates (required).
        timeout: Timeout for Ollama API calls in seconds.
    """

    ollama_url: str
    embedding_model: str
    similarity_threshold: float
    timeout: float = 30.0

    # Embedding cache: maps text to embedding vector
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _client: ollama.Client | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the Ollama client."""
        self._client = ollama.Client(host=self.ollama_url, timeout=self.timeout)
        logger.info(
            "SemanticDuplicateChecker initialized: model=%s, threshold=%.2f",
            self.embedding_model,
            self.similarity_threshold,
        )

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text, using cache if available.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
            Returns empty list if embedding fails.
        """
        if not text or not text.strip():
            return []

        # Normalize text for caching
        normalized = text.lower().strip()

        with self._lock:
            if normalized in self._cache:
                logger.debug("Cache hit for embedding: '%s...'", text[:30])
                return self._cache[normalized]

        # Generate embedding
        try:
            if self._client is None:
                self._client = ollama.Client(host=self.ollama_url, timeout=self.timeout)

            response = self._client.embeddings(model=self.embedding_model, prompt=text)

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
