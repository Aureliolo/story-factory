"""Response caching for LLM calls to avoid redundant generation."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.validation import validate_not_empty, validate_not_none, validate_positive

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached LLM response with metadata."""

    prompt_hash: str
    response: str
    model: str
    temperature: float
    timestamp: float
    hit_count: int = 0


class ResponseCache:
    """LRU cache for LLM responses with TTL and disk persistence.

    Caches responses to avoid re-generating identical content, saving
    time and API costs. Cache entries expire after TTL seconds.
    """

    def __init__(self, cache_dir: Path, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the response cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        validate_not_none(cache_dir, "cache_dir")
        validate_positive(max_size, "max_size")
        validate_positive(ttl_seconds, "ttl_seconds")

        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CachedResponse] = {}
        self._access_times: dict[str, float] = {}  # For LRU eviction

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache from disk
        self._load_from_disk()

    def _hash_prompt(self, prompt: str, model: str, temperature: float) -> str:
        """Create deterministic hash of prompt + model + temperature.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting

        Returns:
            16-character hash string
        """
        content = f"{prompt}|{model}|{temperature:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str, temperature: float) -> str | None:
        """Get cached response if exists and not expired.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting

        Returns:
            Cached response text or None if not found/expired
        """
        validate_not_empty(prompt, "prompt")
        validate_not_empty(model, "model")

        key = self._hash_prompt(prompt, model, temperature)

        if key not in self._cache:
            logger.debug(f"Cache MISS for hash {key}")
            return None

        cached = self._cache[key]

        # Check expiration
        age = time.time() - cached.timestamp
        if age > self.ttl_seconds:
            logger.debug(f"Cache entry expired (age: {age:.0f}s > {self.ttl_seconds}s): {key}")
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return None

        # Update access time and hit count
        self._access_times[key] = time.time()
        cached.hit_count += 1
        logger.info(f"Cache HIT for hash {key} (hits: {cached.hit_count}, age: {age:.0f}s)")

        return cached.response

    def put(self, prompt: str, model: str, temperature: float, response: str):
        """Cache a response.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting
            response: The LLM response to cache
        """
        validate_not_empty(prompt, "prompt")
        validate_not_empty(model, "model")
        validate_not_empty(response, "response")

        key = self._hash_prompt(prompt, model, temperature)

        # Evict least recently used if cache full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        # Create cache entry
        self._cache[key] = CachedResponse(
            prompt_hash=key,
            response=response,
            model=model,
            temperature=temperature,
            timestamp=time.time(),
            hit_count=0,
        )
        self._access_times[key] = time.time()

        # Persist to disk asynchronously in background
        try:
            self._save_to_disk(key)
            logger.debug(f"Cached response with hash {key}")
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {key} to disk: {e}")

    def _evict_lru(self):
        """Evict the least recently used cache entry."""
        if not self._access_times:
            return

        # Find LRU entry
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        logger.debug(f"Evicting LRU cache entry: {lru_key}")
        del self._cache[lru_key]
        del self._access_times[lru_key]

        # Delete from disk
        cache_file = self.cache_dir / f"{lru_key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def _save_to_disk(self, key: str):
        """Save single cache entry to disk.

        Args:
            key: Cache key (hash)
        """
        if key not in self._cache:
            return

        cache_file = self.cache_dir / f"{key}.json"
        cached = self._cache[key]

        with open(cache_file, "w") as f:
            json.dump(asdict(cached), f)

    def _load_from_disk(self):
        """Load cache entries from disk on startup."""
        if not self.cache_dir.exists():
            return

        loaded_count = 0
        expired_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Skip expired entries
                age = time.time() - data["timestamp"]
                if age > self.ttl_seconds:
                    cache_file.unlink()
                    expired_count += 1
                    continue

                key = data["prompt_hash"]
                self._cache[key] = CachedResponse(**data)
                self._access_times[key] = data["timestamp"]
                loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                # Delete corrupted cache file
                try:
                    cache_file.unlink()
                except Exception:
                    pass

        if loaded_count > 0:
            logger.info(
                f"Loaded {loaded_count} cache entries from disk "
                f"({expired_count} expired entries removed)"
            )

    def clear(self):
        """Clear all cache entries from memory and disk."""
        self._cache.clear()
        self._access_times.clear()

        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        if not self._cache:
            return {
                "size": 0,
                "max_size": self.max_size,
                "total_hits": 0,
                "avg_age_seconds": 0,
                "oldest_age_seconds": 0,
            }

        current_time = time.time()
        ages = [current_time - c.timestamp for c in self._cache.values()]
        total_hits = sum(c.hit_count for c in self._cache.values())

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "total_hits": total_hits,
            "avg_age_seconds": sum(ages) / len(ages) if ages else 0,
            "oldest_age_seconds": max(ages) if ages else 0,
            "ttl_seconds": self.ttl_seconds,
        }
