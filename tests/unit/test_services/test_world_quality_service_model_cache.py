"""Tests for WorldQualityService model resolution caching.

Verifies that model selections are stored to avoid redundant tier score calculations
during world building operations.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.mode_models import GenerationMode, SizePreference, VramStrategy
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings


@pytest.fixture
def settings():
    """Create settings with test values."""
    return Settings(
        ollama_url="http://localhost:11434",
        ollama_timeout=60,
        world_quality_enabled=True,
        world_quality_max_iterations=3,
        world_quality_threshold=7.0,
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        # Model settings
        use_per_agent_models=True,
        agent_models={
            "writer": "auto",
            "architect": "auto",
            "editor": "auto",
            "continuity": "auto",
            "validator": "auto",
            "judge": "auto",
        },
        vram_strategy="adaptive",
    )


@pytest.fixture
def mock_mode_service():
    """Create mock mode service with a test mode."""
    mode_service = MagicMock()
    mode = GenerationMode(
        id="test-mode",
        name="Test Mode",
        description="Test mode for caching tests",
        size_preference=SizePreference.MEDIUM,
        vram_strategy=VramStrategy.ADAPTIVE,
    )
    mode_service.get_current_mode.return_value = mode
    mode_service.get_model_for_agent.return_value = "test-model:8b"
    return mode_service


@pytest.fixture
def service(settings, mock_mode_service):
    """Create WorldQualityService with mocked dependencies."""
    svc = WorldQualityService(settings, mock_mode_service)
    # Mock analytics_db to prevent tests from writing to real database
    svc._analytics_db = MagicMock()
    return svc


def _patch_cache_deps():
    """Create patches for cache dependencies (VRAM and installed models)."""
    return [
        patch("src.settings.get_available_vram", return_value=16),
        patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0}),
    ]


class TestCreatorModelCaching:
    """Tests for _get_creator_model caching behavior."""

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_creator_model_stored_across_calls(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Second call for same role uses stored value instead of re-resolving."""
        # First call - should resolve and store
        model1 = service._get_creator_model("character")
        assert model1 == "test-model:8b"

        # Verify mode service was called once for resolution
        initial_call_count = mock_mode_service.get_model_for_agent.call_count

        # Second call - should use stored value
        model2 = service._get_creator_model("character")
        assert model2 == "test-model:8b"

        # Mode service should NOT be called again (storage hit)
        assert mock_mode_service.get_model_for_agent.call_count == initial_call_count

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_different_entity_types_same_role_use_same_storage(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Different entity types mapping to same role share storage."""
        # character and location both map to "writer" role
        model1 = service._get_creator_model("character")
        initial_call_count = mock_mode_service.get_model_for_agent.call_count

        model2 = service._get_creator_model("location")

        # Should use stored value for "writer" role
        assert mock_mode_service.get_model_for_agent.call_count == initial_call_count
        assert model1 == model2

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_different_roles_resolved_independently(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Writer vs architect roles are stored separately."""

        # Configure different models for different roles
        def model_for_agent(role):
            """Return model based on role."""
            if role == "writer":
                return "writer-model:8b"
            elif role == "architect":
                return "architect-model:8b"
            return "default-model:8b"

        mock_mode_service.get_model_for_agent.side_effect = model_for_agent

        # character uses writer, faction uses architect
        writer_model = service._get_creator_model("character")
        architect_model = service._get_creator_model("faction")

        assert writer_model == "writer-model:8b"
        assert architect_model == "architect-model:8b"

        # Both should be in storage
        assert "writer" in service._model_cache._resolved_creator_models
        assert "architect" in service._model_cache._resolved_creator_models


class TestJudgeModelCaching:
    """Tests for _get_judge_model caching behavior."""

    @patch("src.settings.get_installed_models_with_sizes", return_value={"judge-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_judge_model_stored_across_calls(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """Second call for same role uses stored value instead of re-resolving."""
        # Set up judge model different from creator to avoid swap logic
        mock_mode_service.get_model_for_agent.return_value = "judge-model:8b"
        settings.get_models_for_role = MagicMock(return_value=["judge-model:8b"])

        # First call - should resolve and store
        model1 = service._get_judge_model("character")
        assert model1 == "judge-model:8b"
        assert "judge" in service._model_cache._resolved_judge_models

        # Second call - should use stored value
        model2 = service._get_judge_model("character")
        assert model2 == "judge-model:8b"

        # Verify cache was used (judge model still in storage)
        assert service._model_cache._resolved_judge_models["judge"] == "judge-model:8b"

    @patch("src.settings.get_installed_models_with_sizes", return_value={"same-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_judge_model_swap_is_stored(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """When judge model is swapped due to conflict, the swapped model is stored."""
        # Set up judge model same as creator to trigger swap logic
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"
        settings.get_models_for_role = MagicMock(
            return_value=["same-model:8b", "alternate-judge:8b"]
        )

        model = service._get_judge_model("character")

        # Should have swapped to alternate
        assert model == "alternate-judge:8b"
        # Swapped model should be stored
        assert service._model_cache._resolved_judge_models.get("judge") == "alternate-judge:8b"


class TestCacheInvalidation:
    """Tests for cache invalidation behavior."""

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_mode_change(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Mode change clears storage to force re-resolution."""
        # Populate cache
        service._get_creator_model("character")
        assert len(service._model_cache._resolved_creator_models) > 0

        # Simulate mode change
        new_mode = GenerationMode(
            id="different-mode",
            name="Different Mode",
            size_preference=SizePreference.LARGEST,
            vram_strategy=VramStrategy.SEQUENTIAL,
        )
        mock_mode_service.get_current_mode.return_value = new_mode

        # Next access should detect context change and clear cache
        service._get_creator_model("character")

        # Cache should have been cleared and repopulated
        # The resolution context should now match the new mode
        assert service._model_cache._resolution_context is not None
        assert service._model_cache._resolution_context[0] == "different-mode"

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram")
    def test_cache_invalidated_on_vram_change(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """VRAM change clears storage to force re-resolution."""
        # Start with 16GB VRAM
        mock_vram.return_value = 16

        # Populate cache
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Simulate VRAM change (e.g., GPU load changed)
        mock_vram.return_value = 8

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context
        assert service._model_cache._resolution_context[2] == 8  # VRAM value

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_vram_strategy_change(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """VRAM strategy change clears storage to force re-resolution."""
        # Populate cache with initial strategy
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Change VRAM strategy in settings
        settings.vram_strategy = "sequential"

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context
        assert service._model_cache._resolution_context[1] == "sequential"

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_model_settings_change(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """Model settings change (agent_models, default_model) clears storage."""
        # Populate cache with initial settings
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Change model settings (user changed writer model in UI)
        settings.agent_models["writer"] = "different-model:8b"

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_default_model_change(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """default_model change clears storage."""
        # Populate cache
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Change default model
        settings.default_model = "new-default:8b"

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_per_agent_toggle(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """Toggling use_per_agent_models clears storage."""
        # Populate cache
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Toggle per-agent models setting
        settings.use_per_agent_models = not settings.use_per_agent_models

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context

    @patch("src.settings.get_installed_models_with_sizes", return_value={"same-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_explicit_invalidation(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """invalidate_model_cache() clears all storage including warned conflicts."""
        # Set up to populate warned conflicts
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"
        settings.get_models_for_role = MagicMock(return_value=["same-model:8b"])

        # Populate both caches and trigger conflict warning
        service._get_creator_model("character")
        service._get_judge_model("character")

        assert len(service._model_cache._resolved_creator_models) > 0
        assert len(service._model_cache._resolved_judge_models) > 0
        assert service._model_cache._resolution_context is not None
        assert len(service._model_cache._warned_conflicts) > 0

        # Explicitly invalidate
        service.invalidate_model_cache()

        # All storage should be cleared including warned conflicts
        assert len(service._model_cache._resolved_creator_models) == 0
        assert len(service._model_cache._resolved_judge_models) == 0
        assert service._model_cache._resolution_context is None
        assert len(service._model_cache._warned_conflicts) == 0

    @patch("src.settings.get_installed_models_with_sizes", return_value={"same-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_warned_conflicts_cleared_on_context_change(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """Warned conflicts are cleared when context changes."""
        # Set up to trigger a conflict warning
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"
        settings.get_models_for_role = MagicMock(return_value=["same-model:8b"])

        # Trigger the warning
        service._get_judge_model("character")
        assert len(service._model_cache._warned_conflicts) > 0

        # Change mode to trigger context invalidation
        new_mode = GenerationMode(
            id="new-mode",
            name="New Mode",
            size_preference=SizePreference.SMALLEST,
            vram_strategy=VramStrategy.PARALLEL,
        )
        mock_mode_service.get_current_mode.return_value = new_mode

        # Access again to trigger validation
        service._get_creator_model("character")

        # Warned conflicts should be cleared
        assert len(service._model_cache._warned_conflicts) == 0

    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_installed_models_change(
        self, mock_vram, service, mock_mode_service
    ):
        """Installed models change clears storage to force re-resolution."""
        # Initial models
        with patch(
            "src.settings.get_installed_models_with_sizes",
            return_value={"test-model:8b": 8.0},
        ):
            service._get_creator_model("character")
            original_context = service._model_cache._resolution_context

        # Simulate new model installed
        with patch(
            "src.settings.get_installed_models_with_sizes",
            return_value={"test-model:8b": 8.0, "new-model:16b": 16.0},
        ):
            service._get_creator_model("character")

            # Context should have been updated due to installed models change
            assert service._model_cache._resolution_context != original_context

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_invalidated_on_custom_tags_change(
        self, mock_vram, mock_models, service, mock_mode_service, settings
    ):
        """Custom model tags change clears storage to force re-resolution."""
        # Populate cache
        service._get_creator_model("character")
        original_context = service._model_cache._resolution_context

        # Add custom tags for a model
        settings.custom_model_tags["test-model:8b"] = ["writer", "creative"]

        # Next access should detect context change
        service._get_creator_model("character")

        # Context should have been updated
        assert service._model_cache._resolution_context != original_context


class TestCacheEfficiency:
    """Tests verifying cache reduces redundant calculations."""

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_multiple_entities_same_type_single_resolution(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Generating multiple entities of same type resolves model only once."""
        # Simulate generating 5 characters
        for _ in range(5):
            service._get_creator_model("character")

        # Mode service should be called only once for initial resolution
        # (subsequent calls use cache)
        # Note: First call establishes context, so get_current_mode is called,
        # but get_model_for_agent should only be called once for "writer" role
        writer_calls = [
            call
            for call in mock_mode_service.get_model_for_agent.call_args_list
            if call[0][0] == "writer"
        ]
        assert len(writer_calls) == 1

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_mixed_entity_types_efficient_resolution(
        self, mock_vram, mock_models, service, mock_mode_service
    ):
        """Mixed entity types resolve each unique role only once."""

        def model_for_agent(role):
            """Return model based on role."""
            return f"{role}-model:8b"

        mock_mode_service.get_model_for_agent.side_effect = model_for_agent

        # Simulate world building with various entity types
        entity_types = [
            "character",  # writer
            "character",  # writer (cached)
            "location",  # writer (cached)
            "faction",  # architect
            "faction",  # architect (cached)
            "concept",  # architect (cached)
            "relationship",  # editor
        ]

        for entity_type in entity_types:
            service._get_creator_model(entity_type)

        # Should resolve: writer, architect, editor (3 unique roles)
        # Not: 7 calls for 7 entity types
        assert mock_mode_service.get_model_for_agent.call_count == 3


class TestThreadSafety:
    """Tests for thread safety of cache operations."""

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_cache_has_lock(self, mock_vram, mock_models, service):
        """Cache has a lock attribute for thread safety."""
        assert hasattr(service._model_cache, "_lock")
        # Verify it's a reentrant lock
        import threading

        assert isinstance(service._model_cache._lock, type(threading.RLock()))

    @patch("src.settings.get_installed_models_with_sizes", return_value={"test-model:8b": 8.0})
    @patch("src.settings.get_available_vram", return_value=16)
    def test_concurrent_cache_access(self, mock_vram, mock_models, service, mock_mode_service):
        """Concurrent cache access doesn't cause race conditions."""
        import threading

        results = []
        errors = []

        def access_cache(entity_type):
            """Access cache from thread."""
            try:
                model = service._get_creator_model(entity_type)
                results.append((entity_type, model))
            except Exception as e:
                errors.append(e)

        # Create threads accessing cache concurrently
        threads = []
        entity_types = ["character", "location", "faction", "concept"] * 5
        for entity_type in entity_types:
            t = threading.Thread(target=access_cache, args=(entity_type,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0
        # All threads should have gotten results
        assert len(results) == len(entity_types)
