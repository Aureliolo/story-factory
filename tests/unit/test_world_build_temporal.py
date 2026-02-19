"""Tests for temporal validation step in world build pipeline."""

from unittest.mock import MagicMock

import pytest

from src.services.temporal_validation_service import (
    TemporalValidationResult,
)
from src.utils.exceptions import GenerationCancelledError


@pytest.fixture
def mock_svc():
    """Create a mock WorldService."""
    svc = MagicMock()
    svc.settings.validate_temporal_consistency = True
    return svc


@pytest.fixture
def mock_services():
    """Create a mock ServiceContainer."""
    services = MagicMock()
    services.temporal_validation.validate_world.return_value = TemporalValidationResult()
    services.embedding.embed_all_world_data.return_value = {}
    return services


@pytest.fixture
def mock_state():
    """Create a mock StoryState."""
    state = MagicMock()
    state.id = "test-story-1"
    state.brief = MagicMock()
    state.characters = []
    state.chapters = []
    state.plot_summary = ""
    state.plot_points = []
    return state


@pytest.fixture
def mock_world_db():
    """Create a mock WorldDatabase."""
    return MagicMock()


@pytest.fixture
def default_counts():
    """Default entity counts for build tests."""
    return {
        "characters": 0,
        "locations": 0,
        "factions": 0,
        "items": 0,
        "concepts": 0,
        "events": 0,
        "relationships": 0,
        "implicit_relationships": 0,
    }


@pytest.fixture
def disabled_options():
    """WorldBuildOptions with all generation flags disabled."""
    from src.services.world_service import WorldBuildOptions

    options = MagicMock(spec=WorldBuildOptions)
    options.is_cancelled.return_value = False
    options.clear_existing = False
    options.generate_structure = False
    options.generate_locations = False
    options.generate_factions = False
    options.generate_items = False
    options.generate_concepts = False
    options.generate_relationships = False
    options.generate_events = False
    return options


class TestBuildPipelineTemporalStep:
    """Tests for temporal validation in _build_world_entities."""

    def test_build_calls_temporal_validation_when_enabled(
        self, mock_svc, mock_state, mock_world_db, mock_services, disabled_options, default_counts
    ):
        """Build calls temporal validation when setting is enabled."""
        from src.services.world_service._build import _build_world_entities

        progress_calls = []

        def capture_progress(msg, entity_type=None, count=0):
            """Record progress messages for assertion."""
            progress_calls.append(msg)

        _build_world_entities(
            mock_svc,
            mock_state,
            mock_world_db,
            mock_services,
            disabled_options,
            default_counts,
            lambda: None,
            capture_progress,
        )

        mock_services.temporal_validation.validate_world.assert_called_once_with(mock_world_db)
        assert "Validating temporal consistency..." in progress_calls

    def test_build_skips_temporal_when_disabled(
        self, mock_svc, mock_state, mock_world_db, mock_services, disabled_options, default_counts
    ):
        """Build skips temporal validation when setting is disabled."""
        from src.services.world_service._build import _build_world_entities

        mock_svc.settings.validate_temporal_consistency = False

        _build_world_entities(
            mock_svc,
            mock_state,
            mock_world_db,
            mock_services,
            disabled_options,
            default_counts,
            lambda: None,
            lambda msg, entity_type=None, count=0: None,
        )

        mock_services.temporal_validation.validate_world.assert_not_called()

    def test_temporal_validation_failure_is_nonfatal(
        self, mock_svc, mock_state, mock_world_db, mock_services, disabled_options, default_counts
    ):
        """Temporal validation failure does not crash the build."""
        from src.services.world_service._build import _build_world_entities

        mock_services.temporal_validation.validate_world.side_effect = RuntimeError("Boom")

        # Should not raise
        _build_world_entities(
            mock_svc,
            mock_state,
            mock_world_db,
            mock_services,
            disabled_options,
            default_counts,
            lambda: None,
            lambda msg, entity_type=None, count=0: None,
        )

    def test_temporal_validation_cancelled_propagates(
        self, mock_svc, mock_state, mock_world_db, mock_services, disabled_options, default_counts
    ):
        """GenerationCancelledError during temporal validation propagates."""
        from src.services.world_service._build import _build_world_entities

        mock_services.temporal_validation.validate_world.side_effect = GenerationCancelledError(
            "Cancelled"
        )

        with pytest.raises(GenerationCancelledError):
            _build_world_entities(
                mock_svc,
                mock_state,
                mock_world_db,
                mock_services,
                disabled_options,
                default_counts,
                lambda: None,
                lambda msg, entity_type=None, count=0: None,
            )


class TestCalculateTotalSteps:
    """Tests for _calculate_total_steps with validate_temporal."""

    def test_total_steps_increments_with_validate_temporal(self, disabled_options):
        """Step count increases by 1 when validate_temporal is True."""
        from src.services.world_service._build import _calculate_total_steps

        steps_without = _calculate_total_steps(disabled_options, validate_temporal=False)
        steps_with = _calculate_total_steps(disabled_options, validate_temporal=True)

        assert steps_with == steps_without + 1

    def test_total_steps_unchanged_without_validate_temporal(self, disabled_options):
        """Step count unchanged when validate_temporal is False (default)."""
        from src.services.world_service._build import _calculate_total_steps

        steps_default = _calculate_total_steps(disabled_options)
        steps_explicit = _calculate_total_steps(disabled_options, validate_temporal=False)

        assert steps_default == steps_explicit
