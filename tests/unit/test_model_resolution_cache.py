"""Tests for ModelResolutionCache — env context TTL and log reduction."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.services.world_quality_service._model_cache import (
    _ENV_CONTEXT_TTL_SECONDS,
    ModelResolutionCache,
)


@pytest.fixture
def mock_settings():
    """Create mock Settings."""
    settings = MagicMock()
    settings.use_per_agent_models = False
    settings.default_model = "test-model:8b"
    settings.agent_models = {}
    settings.custom_model_tags = {}
    settings.vram_strategy = "auto"
    return settings


@pytest.fixture
def mock_mode_service():
    """Create mock ModelModeService."""
    svc = MagicMock()
    mode = MagicMock()
    mode.id = "balanced"
    svc.get_current_mode.return_value = mode
    return svc


@pytest.fixture
def cache(mock_settings, mock_mode_service):
    """Create a ModelResolutionCache for testing."""
    return ModelResolutionCache(mock_settings, mock_mode_service)


class TestEnvContextTTL:
    """Tests for environment context TTL configuration."""

    def test_ttl_is_300_seconds(self):
        """TTL should be 300 seconds (5 minutes) per #399 item 2."""
        assert _ENV_CONTEXT_TTL_SECONDS == 300.0

    def test_env_context_cached_within_ttl(self, cache):
        """Environment context should be cached within TTL period."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ) as mock_installed,
        ):
            # First call
            cache._get_env_context()
            assert mock_installed.call_count == 1

            # Second call within TTL should use cache
            cache._get_env_context()
            assert mock_installed.call_count == 1

    def test_env_context_refreshed_after_ttl(self, cache):
        """Environment context should refresh after TTL expires."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ) as mock_installed,
        ):
            # First call
            cache._get_env_context()
            assert mock_installed.call_count == 1

            # Simulate TTL expiry
            cache._env_context_timestamp = time.monotonic() - _ENV_CONTEXT_TTL_SECONDS - 1

            # Should refresh
            cache._get_env_context()
            assert mock_installed.call_count == 2


class TestCacheOperations:
    """Tests for basic cache get/store operations."""

    def test_store_and_retrieve_creator_model(self, cache):
        """Stored creator model should be retrievable after context is established."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            # First get establishes the resolution context
            assert cache.get_creator_model("writer") is None
            cache.store_creator_model("writer", "test-writer:8b")
            result = cache.get_creator_model("writer")
            assert result == "test-writer:8b"

    def test_store_and_retrieve_judge_model(self, cache):
        """Stored judge model should be retrievable after context is established."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            # First get establishes the resolution context
            assert cache.get_judge_model("judge") is None
            cache.store_judge_model("judge", "test-judge:12b")
            result = cache.get_judge_model("judge")
            assert result == "test-judge:12b"

    def test_invalidate_clears_all(self, cache):
        """invalidate() should clear all cached data."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            # Establish context, then populate
            cache.get_creator_model("writer")
            cache.store_creator_model("writer", "test-writer:8b")
            cache.store_judge_model("judge", "test-judge:12b")
            cache.mark_conflict_warned("test-key")

            cache.invalidate()

            assert cache.get_creator_model("writer") is None
            assert cache.get_judge_model("judge") is None
            assert not cache.has_warned_conflict("test-key")

    def test_conflict_warning_tracking(self, cache):
        """Conflict warnings should be tracked and queryable."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            assert not cache.has_warned_conflict("character:model-a")
            cache.mark_conflict_warned("character:model-a")
            assert cache.has_warned_conflict("character:model-a")

    def test_store_creator_model_returns_existing_on_race(self, cache):
        """Double-check: second store for same role returns the first-stored value."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            cache.get_creator_model("writer")  # establish context
            first = cache.store_creator_model("writer", "first-model:8b")
            second = cache.store_creator_model("writer", "second-model:8b")
            assert first == "first-model:8b"
            assert second == "first-model:8b"  # returns existing, not the new value

    def test_store_judge_model_returns_existing_on_race(self, cache):
        """Double-check: second store for same role returns the first-stored value."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            cache.get_judge_model("judge")  # establish context
            first = cache.store_judge_model("judge", "first-judge:12b")
            second = cache.store_judge_model("judge", "second-judge:12b")
            assert first == "first-judge:12b"
            assert second == "first-judge:12b"  # returns existing, not the new value

    def test_judge_model_with_creator_model_key(self, cache):
        """Judge model keyed by role+creator_model for per-entity-type caching."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            cache.get_judge_model("judge")  # establish context
            cache.store_judge_model("judge", "judge-a:8b", creator_model="creator-x:8b")
            result = cache.get_judge_model("judge", creator_model="creator-x:8b")
            assert result == "judge-a:8b"

    def test_different_creator_models_get_independent_entries(self, cache):
        """Different creator_model values produce independent cache entries."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            cache.get_judge_model("judge")  # establish context
            cache.store_judge_model("judge", "judge-a:8b", creator_model="creator-x:8b")
            cache.store_judge_model("judge", "judge-b:12b", creator_model="creator-y:12b")

            assert cache.get_judge_model("judge", "creator-x:8b") == "judge-a:8b"
            assert cache.get_judge_model("judge", "creator-y:12b") == "judge-b:12b"
            # Bare role key (no creator) is independent too
            assert cache.get_judge_model("judge") is None

    def test_store_judge_model_race_with_creator_model(self, cache):
        """Double-check with creator_model: second store returns first-stored value."""
        with (
            patch(
                "src.settings.get_available_vram",
                return_value=24000,
            ),
            patch(
                "src.settings.get_installed_models_with_sizes",
                return_value={"model-a": 8000},
            ),
        ):
            cache.get_judge_model("judge")  # establish context
            first = cache.store_judge_model("judge", "first-judge:12b", creator_model="creator:8b")
            second = cache.store_judge_model(
                "judge", "second-judge:12b", creator_model="creator:8b"
            )
            assert first == "first-judge:12b"
            assert second == "first-judge:12b"
