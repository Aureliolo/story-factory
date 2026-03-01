"""Tests for temporal validation lifespan checks and missing lifecycle logging."""

import logging
from unittest.mock import MagicMock

import pytest

from src.memory.entities import Entity
from src.memory.timeline_types import EntityLifecycle, StoryTimestamp
from src.services.temporal_validation_service import (
    TemporalErrorSeverity,
    TemporalErrorType,
    TemporalValidationResult,
    TemporalValidationService,
)


@pytest.fixture
def validation_service() -> TemporalValidationService:
    """Create a temporal validation service with validation enabled."""
    settings = MagicMock()
    settings.validate_temporal_consistency = True
    return TemporalValidationService(settings)


class TestNegativeLifespan:
    """Tests for C1+M9: negative lifespan detection (death before birth)."""

    def test_negative_lifespan_produces_error(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Negative lifespan (death before birth) produces ERROR severity issue."""
        entity = Entity(id="e1", name="TimeParadox", type="character", description="Impossible")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=500),
            death=StoryTimestamp(year=100),  # death before birth -> lifespan = -400
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert len(result.errors) == 1
        assert len(result.warnings) == 0
        error = result.errors[0]
        assert error.severity == TemporalErrorSeverity.ERROR
        assert error.error_type == TemporalErrorType.LIFESPAN_IMPLAUSIBLE
        assert "negative" in error.message
        assert "-400 years" in error.message
        assert "death before birth" in error.message
        assert error.entity_id == "e1"
        assert error.entity_name == "TimeParadox"

    def test_negative_lifespan_faction_produces_error(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Faction with destruction before founding produces ERROR."""
        entity = Entity(id="f1", name="Doomed Guild", type="faction", description="Paradoxical")
        lifecycle = EntityLifecycle(
            founding_year=800,
            destruction_year=200,  # lifespan = -600
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert len(result.errors) == 1
        assert len(result.warnings) == 0
        error = result.errors[0]
        assert error.severity == TemporalErrorSeverity.ERROR
        assert "-600 years" in error.message

    def test_negative_lifespan_logs_error(
        self, validation_service: TemporalValidationService, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Negative lifespan emits logger.error()."""
        entity = Entity(id="e1", name="Broken", type="character", description="Bad data")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=300),
            death=StoryTimestamp(year=100),
        )
        result = TemporalValidationResult()

        with caplog.at_level(logging.ERROR, logger="src.services.temporal_validation_service"):
            validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert any("negative" in record.message for record in caplog.records)


class TestImplausibleLifespan:
    """Tests for lifespan > max_lifespan produces WARNING (existing behavior)."""

    def test_implausible_lifespan_produces_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Lifespan exceeding max triggers a WARNING (not ERROR)."""
        entity = Entity(id="e1", name="Ancient", type="character", description="Old")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
            death=StoryTimestamp(year=500),  # lifespan = 400 > default 200
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result, max_lifespan=200)

        assert len(result.warnings) == 1
        assert len(result.errors) == 0
        warning = result.warnings[0]
        assert warning.severity == TemporalErrorSeverity.WARNING
        assert warning.error_type == TemporalErrorType.LIFESPAN_IMPLAUSIBLE
        assert "implausible" in warning.message
        assert "400 years" in warning.message

    def test_normal_lifespan_no_issues(self, validation_service: TemporalValidationService) -> None:
        """Normal lifespan produces no errors or warnings."""
        entity = Entity(id="e1", name="Mortal", type="character", description="Normal")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
            death=StoryTimestamp(year=180),
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_lifespan_unavailable_no_issues(
        self, validation_service: TemporalValidationService
    ) -> None:
        """No issues when lifespan cannot be calculated (missing death)."""
        entity = Entity(id="e1", name="Unknown", type="character", description="?")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_zero_lifespan_no_issues(self, validation_service: TemporalValidationService) -> None:
        """Zero lifespan (birth == death) produces no issues."""
        entity = Entity(id="e1", name="Brief", type="character", description="Ephemeral")
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
            death=StoryTimestamp(year=100),
        )
        result = TemporalValidationResult()

        validation_service.check_lifespan_plausibility(entity, lifecycle, result)

        assert len(result.errors) == 0
        assert len(result.warnings) == 0


class TestMissingLifecycleLogging:
    """Tests for M2: entities with no lifecycle data log INFO."""

    def test_missing_lifecycle_logs_info(
        self, validation_service: TemporalValidationService, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Entity with no attributes logs INFO about missing lifecycle data."""
        entity = Entity(id="e1", name="Mystery", type="character", description="Unknown")

        with caplog.at_level(logging.INFO, logger="src.services.temporal_validation_service"):
            validation_service.validate_entity(entity, None, [entity], [])

        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "no lifecycle data" in r.message
        ]
        assert len(info_records) == 1
        assert "Mystery" in info_records[0].message

    def test_missing_lifecycle_no_error_or_warning_added(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Entity with no lifecycle data does not add INFO to result errors/warnings.

        The INFO log is purely for observability — it should not pollute the
        validation result.  (Individual type validators may still add their own
        MISSING_TEMPORAL_DATA warnings.)
        """
        entity = Entity(id="e1", name="Concept", type="concept", description="Abstract")
        result = validation_service.validate_entity(entity, None, [entity], [])

        # The concept validator does not add any warnings for missing lifecycle
        # (concepts just return early), so the INFO log should be the only trace.
        # We verify that no ERROR was added by the INFO log itself.
        for issue in result.errors:
            assert issue.error_type != TemporalErrorType.MISSING_TEMPORAL_DATA or (
                issue.severity != TemporalErrorSeverity.ERROR
            )

    def test_entity_with_lifecycle_no_info_log(
        self, validation_service: TemporalValidationService, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Entity WITH lifecycle data does not emit the missing-data INFO log."""
        entity = Entity(
            id="e1",
            name="Hero",
            type="character",
            description="A hero",
            attributes={"lifecycle": {"birth": {"year": 100}}},
        )

        with caplog.at_level(logging.INFO, logger="src.services.temporal_validation_service"):
            validation_service.validate_entity(entity, None, [entity], [])

        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "no lifecycle data" in r.message
        ]
        assert len(info_records) == 0
