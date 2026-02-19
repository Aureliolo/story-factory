"""Tests for temporal validation wiring in world health metrics."""

from unittest.mock import MagicMock, patch

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


class TestHealthMetricsTemporalEnabled:
    """Tests for temporal validation when enabled."""

    def test_health_metrics_include_temporal_errors(self, mock_svc, mock_world_db):
        """Temporal errors populate health metrics when validation enabled."""
        from src.services.world_service._health import get_world_health_metrics

        # Create temporal result with errors
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

        with patch(
            "src.services.temporal_validation_service.TemporalValidationService"
        ) as mock_tvs_cls:
            mock_tvs = MagicMock()
            mock_tvs.validate_world.return_value = temporal_result
            mock_tvs.calculate_temporal_consistency_score.return_value = 6.0
            mock_tvs_cls.return_value = mock_tvs

            metrics = get_world_health_metrics(mock_svc, mock_world_db)

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

    def test_temporal_errors_reduce_health_score(self, mock_svc, mock_world_db):
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

        with patch(
            "src.services.temporal_validation_service.TemporalValidationService"
        ) as mock_tvs_cls:
            mock_tvs = MagicMock()
            mock_tvs.validate_world.return_value = temporal_result
            mock_tvs.calculate_temporal_consistency_score.return_value = 4.0
            mock_tvs_cls.return_value = mock_tvs

            metrics = get_world_health_metrics(mock_svc, mock_world_db)

        # 2 errors * 3 penalty = 6 points off structural score
        assert metrics.temporal_error_count == 2
        assert metrics.health_score < 100.0

    def test_temporal_errors_produce_recommendations(self, mock_svc, mock_world_db):
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

        with patch(
            "src.services.temporal_validation_service.TemporalValidationService"
        ) as mock_tvs_cls:
            mock_tvs = MagicMock()
            mock_tvs.validate_world.return_value = temporal_result
            mock_tvs.calculate_temporal_consistency_score.return_value = 8.0
            mock_tvs_cls.return_value = mock_tvs

            metrics = get_world_health_metrics(mock_svc, mock_world_db)

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
        assert metrics.average_temporal_consistency == 10.0

    def test_temporal_validation_failure_is_nonfatal(self, mock_svc, mock_world_db):
        """If temporal validation raises, health metrics still compute."""
        from src.services.world_service._health import get_world_health_metrics

        with patch(
            "src.services.temporal_validation_service.TemporalValidationService"
        ) as mock_tvs_cls:
            mock_tvs = MagicMock()
            mock_tvs.validate_world.side_effect = RuntimeError("Boom")
            mock_tvs_cls.return_value = mock_tvs

            metrics = get_world_health_metrics(mock_svc, mock_world_db)

        # Should still return valid metrics with default temporal values
        assert metrics.temporal_error_count == 0
        assert metrics.temporal_warning_count == 0
        assert metrics.health_score is not None
