"""Tests for temporal validation wiring in world health metrics."""

from unittest.mock import MagicMock

import pytest

from src.services.temporal_validation_service import (
    TemporalErrorSeverity,
    TemporalErrorType,
    TemporalValidationIssue,
    TemporalValidationResult,
)


@pytest.fixture
def mock_world_db():
    """Create a mock world database with basic entity data."""
    db = MagicMock()
    db.count_entities.return_value = 0
    db.list_relationships.return_value = []
    db.list_entities.return_value = []
    db.find_orphans.return_value = []
    db.find_circular_relationships.return_value = []
    return db


@pytest.fixture
def mock_svc():
    """Create a mock WorldService with settings."""
    svc = MagicMock()
    svc.settings.orphan_detection_enabled = False
    svc.settings.circular_detection_enabled = False
    svc.settings.validate_temporal_consistency = True
    return svc


@pytest.fixture
def mock_temporal_service():
    """Create a mock TemporalValidationService for DI injection."""
    return MagicMock()


class TestHealthMetricsTemporalEnabled:
    """Tests for temporal validation when enabled."""

    def test_health_metrics_include_temporal_errors(
        self, mock_svc, mock_world_db, mock_temporal_service
    ):
        """Temporal errors populate health metrics when validation enabled."""
        from src.services.world_service._health import get_world_health_metrics

        temporal_result = TemporalValidationResult(
            errors=[
                TemporalValidationIssue(
                    entity_id="char-1",
                    entity_name="Hero",
                    entity_type="character",
                    error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                    severity=TemporalErrorSeverity.ERROR,
                    message="Character born before faction founded",
                    suggestion="Adjust birth year",
                ),
            ],
            warnings=[
                TemporalValidationIssue(
                    entity_id="char-2",
                    entity_name="Villain",
                    entity_type="character",
                    error_type=TemporalErrorType.INVALID_DATE,
                    severity=TemporalErrorSeverity.WARNING,
                    message="Date validation issue",
                    suggestion="Adjust date",
                ),
            ],
            is_valid=False,
        )

        mock_temporal_service.validate_world.return_value = temporal_result
        mock_temporal_service.calculate_temporal_consistency_score.return_value = 6.0

        metrics = get_world_health_metrics(
            mock_svc, mock_world_db, temporal_validation=mock_temporal_service
        )

        assert metrics.temporal_error_count == 1
        assert metrics.temporal_warning_count == 1
        assert len(metrics.temporal_issues) == 2
        assert metrics.average_temporal_consistency == 6.0

        # Verify issue dict structure
        error_issue = metrics.temporal_issues[0]
        assert error_issue["entity_id"] == "char-1"
        assert error_issue["entity_name"] == "Hero"
        assert error_issue["severity"] == "error"
        assert error_issue["message"] == "Character born before faction founded"
        # Verify related_entity fields are present
        assert "related_entity_id" in error_issue
        assert "related_entity_name" in error_issue

    def test_temporal_errors_reduce_health_score(
        self, mock_svc, mock_world_db, mock_temporal_service
    ):
        """Temporal errors penalize the health score."""
        from src.services.world_service._health import get_world_health_metrics

        # Make at least 1 entity exist so health score isn't forced to 0
        mock_entity = MagicMock()
        mock_entity.id = "char-1"
        mock_entity.name = "Hero"
        mock_entity.type = "character"
        mock_entity.attributes = {}
        mock_entity.created_at = MagicMock()
        mock_world_db.count_entities.side_effect = lambda t: 1 if t == "character" else 0
        mock_world_db.list_entities.return_value = [mock_entity]

        temporal_result = TemporalValidationResult(
            errors=[
                TemporalValidationIssue(
                    entity_id="char-1",
                    entity_name="Hero",
                    entity_type="character",
                    error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                    severity=TemporalErrorSeverity.ERROR,
                    message="Error 1",
                    suggestion="Fix 1",
                ),
                TemporalValidationIssue(
                    entity_id="char-1",
                    entity_name="Hero",
                    entity_type="character",
                    error_type=TemporalErrorType.FOUNDING_ORDER,
                    severity=TemporalErrorSeverity.ERROR,
                    message="Error 2",
                    suggestion="Fix 2",
                ),
            ],
            is_valid=False,
        )

        mock_temporal_service.validate_world.return_value = temporal_result
        mock_temporal_service.calculate_temporal_consistency_score.return_value = 4.0

        metrics = get_world_health_metrics(
            mock_svc, mock_world_db, temporal_validation=mock_temporal_service
        )

        # 2 errors * 3 penalty = 6 points off structural score
        assert metrics.temporal_error_count == 2
        assert metrics.health_score <= 94.0

    def test_temporal_errors_produce_recommendations(
        self, mock_svc, mock_world_db, mock_temporal_service
    ):
        """Temporal errors generate recommendations."""
        from src.services.world_service._health import get_world_health_metrics

        temporal_result = TemporalValidationResult(
            errors=[
                TemporalValidationIssue(
                    entity_id="char-1",
                    entity_name="Hero",
                    entity_type="character",
                    error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                    severity=TemporalErrorSeverity.ERROR,
                    message="Error",
                    suggestion="Fix",
                ),
            ],
            is_valid=False,
        )

        mock_temporal_service.validate_world.return_value = temporal_result
        mock_temporal_service.calculate_temporal_consistency_score.return_value = 8.0

        metrics = get_world_health_metrics(
            mock_svc, mock_world_db, temporal_validation=mock_temporal_service
        )

        temporal_recs = [r for r in metrics.recommendations if "temporal" in r.lower()]
        assert len(temporal_recs) >= 1


class TestHealthMetricsTemporalDisabled:
    """Tests for temporal validation when disabled."""

    def test_health_metrics_skip_temporal_when_disabled(self, mock_svc, mock_world_db):
        """Temporal fields stay at defaults when validation is disabled."""
        from src.services.world_service._health import get_world_health_metrics

        mock_svc.settings.validate_temporal_consistency = False

        metrics = get_world_health_metrics(mock_svc, mock_world_db)

        assert metrics.temporal_error_count == 0
        assert metrics.temporal_warning_count == 0
        assert metrics.temporal_issues == []
        # When disabled, default is 10.0 ("not measured / no issues")
        assert metrics.average_temporal_consistency == 10.0


class TestHealthMetricsTemporalFailure:
    """Tests for temporal validation when it raises an exception."""

    def test_temporal_validation_failure_is_nonfatal(
        self, mock_svc, mock_world_db, mock_temporal_service
    ):
        """If temporal validation raises, health metrics still compute with failure flags."""
        from src.services.world_service._health import get_world_health_metrics

        mock_temporal_service.validate_world.side_effect = RuntimeError("Boom")

        metrics = get_world_health_metrics(
            mock_svc, mock_world_db, temporal_validation=mock_temporal_service
        )

        # Should still return valid metrics with distinct failure flags
        assert metrics.temporal_validation_failed is True
        assert metrics.temporal_validation_error == "Boom"
        # No fake issues injected — failure is tracked separately
        assert metrics.temporal_error_count == 0
        assert metrics.temporal_warning_count == 0
        assert metrics.temporal_issues == []
        # On failure, consistency stays at init default (10.0)
        assert metrics.average_temporal_consistency == 10.0
        assert metrics.health_score is not None
