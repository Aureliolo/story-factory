"""Tests for response cache system."""

import time

import pytest

from prompts.cache import CachedResponse, ResponseCache


class TestResponseCache:
    """Tests for ResponseCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        return tmp_path / "cache"

    def test_put_and_get_response(self, cache_dir):
        """Should cache and retrieve responses."""
        cache = ResponseCache(cache_dir, max_size=10, ttl_seconds=3600)

        prompt = "Write a story about a dragon"
        model = "test-model"
        temperature = 0.7
        response = "Once upon a time, there was a dragon..."

        # Cache response
        cache.put(prompt, model, temperature, response)

        # Retrieve response
        retrieved = cache.get(prompt, model, temperature)

        assert retrieved == response

    def test_get_returns_none_for_missing_entry(self, cache_dir):
        """Should return None for cache miss."""
        cache = ResponseCache(cache_dir)

        result = cache.get("nonexistent prompt", "model", 0.7)

        assert result is None

    def test_get_respects_ttl(self, cache_dir):
        """Should expire entries after TTL."""
        cache = ResponseCache(cache_dir, ttl_seconds=1)

        prompt = "Test prompt"
        response = "Test response"

        cache.put(prompt, "model", 0.7, response)

        # Should exist immediately
        assert cache.get(prompt, "model", 0.7) == response

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get(prompt, "model", 0.7) is None

    def test_cache_key_includes_model_and_temperature(self, cache_dir):
        """Should treat same prompt with different model/temp as different entries."""
        cache = ResponseCache(cache_dir)

        prompt = "Same prompt"
        cache.put(prompt, "model1", 0.7, "Response 1")
        cache.put(prompt, "model2", 0.7, "Response 2")
        cache.put(prompt, "model1", 0.9, "Response 3")

        assert cache.get(prompt, "model1", 0.7) == "Response 1"
        assert cache.get(prompt, "model2", 0.7) == "Response 2"
        assert cache.get(prompt, "model1", 0.9) == "Response 3"

    def test_lru_eviction_when_full(self, cache_dir):
        """Should evict least recently used entry when cache is full."""
        cache = ResponseCache(cache_dir, max_size=2, ttl_seconds=3600)

        # Add 2 entries (fills cache)
        cache.put("prompt1", "model", 0.7, "response1")
        cache.put("prompt2", "model", 0.7, "response2")

        # Access prompt1 (makes it recently used)
        cache.get("prompt1", "model", 0.7)

        # Add third entry (should evict prompt2, the LRU)
        cache.put("prompt3", "model", 0.7, "response3")

        assert cache.get("prompt1", "model", 0.7) == "response1"  # Still cached
        assert cache.get("prompt2", "model", 0.7) is None  # Evicted
        assert cache.get("prompt3", "model", 0.7) == "response3"  # New entry

    def test_hit_count_increments(self, cache_dir):
        """Should increment hit count on each access."""
        cache = ResponseCache(cache_dir)

        prompt = "Test prompt"
        cache.put(prompt, "model", 0.7, "response")

        # Access multiple times
        for _ in range(3):
            cache.get(prompt, "model", 0.7)

        # Check internal state
        key = cache._hash_prompt(prompt, "model", 0.7)
        assert cache._cache[key].hit_count == 3

    def test_clear_removes_all_entries(self, cache_dir):
        """Should clear all cache entries."""
        cache = ResponseCache(cache_dir)

        cache.put("prompt1", "model", 0.7, "response1")
        cache.put("prompt2", "model", 0.7, "response2")

        cache.clear()

        assert cache.get("prompt1", "model", 0.7) is None
        assert cache.get("prompt2", "model", 0.7) is None
        assert len(cache._cache) == 0

    def test_persist_and_load_from_disk(self, cache_dir):
        """Should persist cache to disk and reload on init."""
        # Create cache and add entries
        cache1 = ResponseCache(cache_dir, ttl_seconds=3600)
        cache1.put("prompt1", "model", 0.7, "response1")
        cache1.put("prompt2", "model", 0.8, "response2")

        # Create new cache instance (should load from disk)
        cache2 = ResponseCache(cache_dir, ttl_seconds=3600)

        assert cache2.get("prompt1", "model", 0.7) == "response1"
        assert cache2.get("prompt2", "model", 0.8) == "response2"

    def test_load_from_disk_skips_expired_entries(self, cache_dir):
        """Should not load expired entries from disk."""
        # Create cache with short TTL
        cache1 = ResponseCache(cache_dir, ttl_seconds=1)
        cache1.put("prompt1", "model", 0.7, "response1")

        # Wait for expiration
        time.sleep(1.1)

        # Create new cache (should skip expired entry)
        cache2 = ResponseCache(cache_dir, ttl_seconds=1)

        assert cache2.get("prompt1", "model", 0.7) is None

    def test_get_stats(self, cache_dir):
        """Should return cache statistics."""
        cache = ResponseCache(cache_dir, max_size=10)

        cache.put("prompt1", "model", 0.7, "response1")
        cache.put("prompt2", "model", 0.7, "response2")
        cache.get("prompt1", "model", 0.7)  # Increment hit count

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.2
        assert stats["total_hits"] == 1
        assert stats["avg_age_seconds"] >= 0
        assert stats["oldest_age_seconds"] >= 0

    def test_get_stats_empty_cache(self, cache_dir):
        """Should handle stats for empty cache."""
        cache = ResponseCache(cache_dir)

        stats = cache.get_stats()

        assert stats["size"] == 0
        assert stats["total_hits"] == 0
        assert stats["avg_age_seconds"] == 0

    def test_validation_errors(self, cache_dir):
        """Should validate inputs."""
        cache = ResponseCache(cache_dir)

        with pytest.raises(ValueError, match="cannot be empty"):
            cache.get("", "model", 0.7)

        with pytest.raises(ValueError, match="cannot be empty"):
            cache.get("prompt", "", 0.7)

        with pytest.raises(ValueError, match="cannot be empty"):
            cache.put("", "model", 0.7, "response")

        with pytest.raises(ValueError, match="cannot be empty"):
            cache.put("prompt", "model", 0.7, "")


class TestCachedResponse:
    """Tests for CachedResponse dataclass."""

    def test_create_cached_response(self):
        """Should create CachedResponse instance."""
        response = CachedResponse(
            prompt_hash="abc123",
            response="Test response",
            model="test-model",
            temperature=0.7,
            timestamp=time.time(),
            hit_count=0,
        )

        assert response.prompt_hash == "abc123"
        assert response.response == "Test response"
        assert response.model == "test-model"
        assert response.hit_count == 0
