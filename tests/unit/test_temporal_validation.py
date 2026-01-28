"""Tests for temporal validation service."""

from unittest.mock import MagicMock

import pytest

from src.memory.entities import Entity
from src.memory.timeline_types import EntityLifecycle, StoryTimestamp
from src.memory.world_calendar import CalendarMonth, WorldCalendar
from src.services.temporal_validation_service import (
    TemporalErrorSeverity,
    TemporalErrorType,
    TemporalValidationResult,
    TemporalValidationService,
)


@pytest.fixture
def validation_service() -> TemporalValidationService:
    """Create a temporal validation service for testing."""
    settings = MagicMock()
    return TemporalValidationService(settings)


@pytest.fixture
def sample_calendar() -> WorldCalendar:
    """Create a sample calendar for testing."""
    return WorldCalendar(
        current_era_name="Third Age",
        era_abbreviation="TA",
        era_start_year=1,
        months=[
            CalendarMonth(name="Firstmoon", days=31),
            CalendarMonth(name="Midyear", days=30),
        ],
        days_per_week=7,
        day_names=["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"],
        current_story_year=500,
    )


class TestTemporalValidationResult:
    """Tests for TemporalValidationResult model."""

    def test_empty_result_is_valid(self) -> None:
        """Test that an empty result is considered valid."""
        result = TemporalValidationResult()
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.total_issues == 0

    def test_result_with_warnings_is_valid(self) -> None:
        """Test that a result with only warnings is still valid."""
        from src.services.temporal_validation_service import TemporalValidationError

        result = TemporalValidationResult()
        result.warnings.append(
            TemporalValidationError(
                entity_id="1",
                entity_name="Test",
                entity_type="character",
                error_type=TemporalErrorType.INVALID_DATE,
                severity=TemporalErrorSeverity.WARNING,
                message="Minor date issue",
            )
        )
        result.is_valid = len(result.errors) == 0

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.total_issues == 1

    def test_result_with_errors_is_invalid(self) -> None:
        """Test that a result with errors is invalid."""
        from src.services.temporal_validation_service import TemporalValidationError

        result = TemporalValidationResult()
        result.errors.append(
            TemporalValidationError(
                entity_id="1",
                entity_name="Test",
                entity_type="character",
                error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                severity=TemporalErrorSeverity.ERROR,
                message="Timeline conflict",
            )
        )
        result.is_valid = len(result.errors) == 0

        assert result.is_valid is False
        assert result.error_count == 1


class TestTemporalValidationService:
    """Tests for TemporalValidationService."""

    def test_validate_entity_without_lifecycle(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test validation of entity without lifecycle info."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Test Character",
            description="A test character",
            attributes={},
        )

        result = validation_service.validate_entity(
            entity=entity,
            calendar=None,
            all_entities=[entity],
            relationships=[],
        )

        # Should pass with no errors/warnings if no lifecycle data
        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_character_born_before_faction_founded(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test error when character born before faction they're a member of."""
        character = Entity(
            id="char-1",
            type="character",
            name="Young Knight",
            description="A young knight",
            attributes={
                "lifecycle": {
                    "birth": {"year": 400},  # Born year 400
                }
            },
        )

        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Order",
            description="A knightly order",
            attributes={
                "lifecycle": {
                    "founding_year": 450,  # Founded year 450 (after character birth)
                }
            },
        )

        # Character is member of faction
        relationships = [("char-1", "faction-1", "member_of")]

        result = validation_service.validate_entity(
            entity=character,
            calendar=None,
            all_entities=[character, faction],
            relationships=relationships,
        )

        # Should have an error - character born before faction founded
        assert result.is_valid is False
        assert result.error_count == 1
        assert result.errors[0].error_type == TemporalErrorType.PREDATES_DEPENDENCY
        assert "faction" in result.errors[0].message.lower()

    def test_validate_character_born_after_faction_founded(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test no error when character born after faction founded."""
        character = Entity(
            id="char-1",
            type="character",
            name="Young Knight",
            description="A young knight",
            attributes={
                "lifecycle": {
                    "birth": {"year": 500},  # Born year 500
                }
            },
        )

        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Order",
            description="A knightly order",
            attributes={
                "lifecycle": {
                    "founding_year": 450,  # Founded year 450 (before character birth)
                }
            },
        )

        relationships = [("char-1", "faction-1", "member_of")]

        result = validation_service.validate_entity(
            entity=character,
            calendar=None,
            all_entities=[character, faction],
            relationships=relationships,
        )

        # Should have no errors
        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_faction_founded_before_parent(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test error when faction founded before its parent faction."""
        child_faction = Entity(
            id="faction-child",
            type="faction",
            name="The Splinter Group",
            description="An offshoot",
            attributes={
                "lifecycle": {
                    "founding_year": 400,  # Founded 400
                }
            },
        )

        parent_faction = Entity(
            id="faction-parent",
            type="faction",
            name="The Original Order",
            description="The original order",
            attributes={
                "lifecycle": {
                    "founding_year": 500,  # Founded 500 (after child!)
                }
            },
        )

        relationships = [("faction-child", "faction-parent", "split_from")]

        result = validation_service.validate_entity(
            entity=child_faction,
            calendar=None,
            all_entities=[child_faction, parent_faction],
            relationships=relationships,
        )

        # Should have an error
        assert result.is_valid is False
        assert result.error_count == 1
        assert result.errors[0].error_type == TemporalErrorType.FOUNDING_ORDER

    def test_validate_dates_against_calendar(
        self,
        validation_service: TemporalValidationService,
        sample_calendar: WorldCalendar,
    ) -> None:
        """Test validation of dates against calendar rules."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Test Character",
            description="A test character",
            attributes={
                "lifecycle": {
                    "birth": {"year": 0},  # Before era start
                }
            },
        )

        result = validation_service.validate_entity(
            entity=entity,
            calendar=sample_calendar,
            all_entities=[entity],
            relationships=[],
        )

        # Should have a warning about invalid date
        assert result.warning_count >= 1
        invalid_date_warnings = [
            w for w in result.warnings if w.error_type == TemporalErrorType.INVALID_DATE
        ]
        assert len(invalid_date_warnings) >= 1

    def test_calculate_temporal_consistency_score_perfect(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test score calculation with no issues."""
        result = TemporalValidationResult(errors=[], warnings=[], is_valid=True)
        score = validation_service.calculate_temporal_consistency_score(result)
        assert score == 10.0

    def test_calculate_temporal_consistency_score_with_errors(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test score calculation with errors."""
        from src.services.temporal_validation_service import TemporalValidationError

        result = TemporalValidationResult()
        # Add 3 errors (3 * 2 = 6 penalty)
        for i in range(3):
            result.errors.append(
                TemporalValidationError(
                    entity_id=f"e{i}",
                    entity_name=f"Entity {i}",
                    entity_type="character",
                    error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                    severity=TemporalErrorSeverity.ERROR,
                    message="Error",
                )
            )

        score = validation_service.calculate_temporal_consistency_score(result)
        assert score == 4.0  # 10 - (3 * 2)

    def test_calculate_temporal_consistency_score_with_warnings(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test score calculation with warnings only."""
        from src.services.temporal_validation_service import TemporalValidationError

        result = TemporalValidationResult()
        # Add 4 warnings (4 * 0.5 = 2 penalty)
        for i in range(4):
            result.warnings.append(
                TemporalValidationError(
                    entity_id=f"e{i}",
                    entity_name=f"Entity {i}",
                    entity_type="character",
                    error_type=TemporalErrorType.INVALID_DATE,
                    severity=TemporalErrorSeverity.WARNING,
                    message="Warning",
                )
            )

        score = validation_service.calculate_temporal_consistency_score(result)
        assert score == 8.0  # 10 - (4 * 0.5)


class TestStoryTimestampFormatDisplay:
    """Tests for StoryTimestamp.format_display method."""

    def test_format_display_without_calendar(self) -> None:
        """Test format_display without a calendar."""
        ts = StoryTimestamp(year=500, month=3, day=15)
        result = ts.format_display(calendar=None)

        # Should fall back to display_text
        assert "500" in result

    def test_format_display_with_calendar(self, sample_calendar: WorldCalendar) -> None:
        """Test format_display with a calendar."""
        ts = StoryTimestamp(year=500, month=1, day=15)
        result = ts.format_display(calendar=sample_calendar)

        # Should use calendar formatting
        assert "500" in result
        assert "Firstmoon" in result


class TestEntityLifecycleProperties:
    """Tests for EntityLifecycle properties."""

    def test_lifespan_from_birth_death(self) -> None:
        """Test lifespan calculation from birth/death."""
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
            death=StoryTimestamp(year=180),
        )
        assert lifecycle.lifespan == 80

    def test_lifespan_from_founding_destruction(self) -> None:
        """Test lifespan calculation from founding/destruction."""
        lifecycle = EntityLifecycle(
            founding_year=100,
            destruction_year=250,
        )
        assert lifecycle.lifespan == 150

    def test_lifespan_ongoing(self) -> None:
        """Test lifespan is None for ongoing entities."""
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
        )
        assert lifecycle.lifespan is None

    def test_start_year_from_birth(self) -> None:
        """Test start_year from birth."""
        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=100),
        )
        assert lifecycle.start_year == 100

    def test_start_year_from_founding(self) -> None:
        """Test start_year from founding_year."""
        lifecycle = EntityLifecycle(
            founding_year=200,
        )
        assert lifecycle.start_year == 200

    def test_end_year_from_death(self) -> None:
        """Test end_year from death."""
        lifecycle = EntityLifecycle(
            death=StoryTimestamp(year=180),
        )
        assert lifecycle.end_year == 180

    def test_end_year_from_destruction(self) -> None:
        """Test end_year from destruction_year."""
        lifecycle = EntityLifecycle(
            destruction_year=300,
        )
        assert lifecycle.end_year == 300
