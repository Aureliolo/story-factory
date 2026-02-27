"""Tests for temporal validation service."""

import logging
from unittest.mock import MagicMock

import pytest

from src.memory.entities import Entity
from src.memory.timeline_types import (
    SENTINEL_YEARS,
    EntityLifecycle,
    StoryTimestamp,
    _parse_year,
    extract_lifecycle_from_attributes,
)
from src.memory.world_calendar import CalendarMonth, HistoricalEra, WorldCalendar
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
    settings.validate_temporal_consistency = True
    return TemporalValidationService(settings)


class TestTemporalValidationResultIsValid:
    """Tests for TemporalValidationResult.is_valid computed property."""

    def test_is_valid_true_when_no_errors(self):
        """is_valid returns True for a fresh result with no errors."""
        result = TemporalValidationResult()
        assert result.is_valid is True

    def test_is_valid_false_after_adding_error(self):
        """is_valid returns False after an error is appended."""
        from src.services.temporal_validation_service import (
            TemporalValidationIssue,
        )

        result = TemporalValidationResult()
        result.errors.append(
            TemporalValidationIssue(
                entity_id="e1",
                entity_name="Hero",
                entity_type="character",
                error_type=TemporalErrorType.INVALID_DATE,
                severity=TemporalErrorSeverity.ERROR,
                message="Test error",
            )
        )
        assert result.is_valid is False

    def test_is_valid_true_with_only_warnings(self):
        """is_valid returns True when only warnings are present (no errors)."""
        from src.services.temporal_validation_service import (
            TemporalValidationIssue,
        )

        result = TemporalValidationResult()
        result.warnings.append(
            TemporalValidationIssue(
                entity_id="e1",
                entity_name="Hero",
                entity_type="character",
                error_type=TemporalErrorType.INVALID_DATE,
                severity=TemporalErrorSeverity.WARNING,
                message="Test warning",
            )
        )
        assert result.is_valid is True

    def test_is_valid_reflects_error_list_mutations(self):
        """is_valid dynamically reflects the errors list — removing errors restores validity."""
        from src.services.temporal_validation_service import (
            TemporalValidationIssue,
        )

        result = TemporalValidationResult()
        issue = TemporalValidationIssue(
            entity_id="e1",
            entity_name="Hero",
            entity_type="character",
            error_type=TemporalErrorType.INVALID_DATE,
            severity=TemporalErrorSeverity.ERROR,
            message="Test error",
        )
        result.errors.append(issue)
        assert result.is_valid is False

        result.errors.remove(issue)
        assert result.is_valid is True


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


class TestTemporalValidationDisabled:
    """Tests for when temporal validation is disabled."""

    def test_validate_entity_returns_empty_when_disabled(self) -> None:
        """Test that validation returns empty result when disabled."""
        settings = MagicMock()
        settings.validate_temporal_consistency = False
        service = TemporalValidationService(settings)

        entity = Entity(
            id="char-1",
            type="character",
            name="Test",
            description="Test",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},
                }
            },
        )

        result = service.validate_entity(entity, None, [entity], [])

        # Should return empty result without checking
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_validate_world_returns_empty_when_disabled(self) -> None:
        """Test that world validation returns empty result when disabled."""
        settings = MagicMock()
        settings.validate_temporal_consistency = False
        service = TemporalValidationService(settings)

        mock_world_db = MagicMock()
        # These should not be called
        mock_world_db.list_entities.side_effect = Exception("Should not be called")

        result = service.validate_world(mock_world_db)

        assert result.is_valid is True
        assert result.error_count == 0


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
        from src.services.temporal_validation_service import TemporalValidationIssue

        result = TemporalValidationResult()
        result.warnings.append(
            TemporalValidationIssue(
                entity_id="1",
                entity_name="Test",
                entity_type="character",
                error_type=TemporalErrorType.INVALID_DATE,
                severity=TemporalErrorSeverity.WARNING,
                message="Minor date issue",
            )
        )
        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.total_issues == 1

    def test_result_with_errors_is_invalid(self) -> None:
        """Test that a result with errors is invalid."""
        from src.services.temporal_validation_service import TemporalValidationIssue

        result = TemporalValidationResult()
        result.errors.append(
            TemporalValidationIssue(
                entity_id="1",
                entity_name="Test",
                entity_type="character",
                error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                severity=TemporalErrorSeverity.ERROR,
                message="Timeline conflict",
            )
        )
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

        # No errors, but should have a MISSING_TEMPORAL_DATA warning
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA

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
                    "birth": {"year": -5},  # Before era start (year 1)
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
        result = TemporalValidationResult(errors=[], warnings=[])
        score = validation_service.calculate_temporal_consistency_score(result)
        assert score == 10.0

    def test_calculate_temporal_consistency_score_with_errors(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test score calculation with errors."""
        from src.services.temporal_validation_service import TemporalValidationIssue

        result = TemporalValidationResult()
        # Add 3 errors (3 * 2 = 6 penalty)
        for i in range(3):
            result.errors.append(
                TemporalValidationIssue(
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
        from src.services.temporal_validation_service import TemporalValidationIssue

        result = TemporalValidationResult()
        # Add 4 warnings (4 * 0.5 = 2 penalty)
        for i in range(4):
            result.warnings.append(
                TemporalValidationIssue(
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


class TestStoryTimestampDisplayText:
    """Tests for StoryTimestamp.display_text property."""

    def test_display_text_with_era_name(self) -> None:
        """Test display_text includes era_name when present."""
        ts = StoryTimestamp(year=500, era_name="Third Age")
        result = ts.display_text

        assert "500" in result
        assert "Third Age" in result

    def test_display_text_year_only(self) -> None:
        """Test display_text with year only."""
        ts = StoryTimestamp(year=300)
        result = ts.display_text

        assert result == "Year 300"

    def test_display_text_year_month_day(self) -> None:
        """Test display_text with full date."""
        ts = StoryTimestamp(year=300, month=6, day=15)
        result = ts.display_text

        assert "Year 300" in result
        assert "Month 6" in result
        assert "Day 15" in result


class TestExtractLifecycleFromAttributes:
    """Tests for extract_lifecycle_from_attributes function."""

    def test_extract_with_string_founding_year(self) -> None:
        """Test extracting lifecycle with string founding_year."""
        attributes = {
            "lifecycle": {
                "founding_year": "500",  # String instead of int
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.founding_year == 500

    def test_extract_with_string_destruction_year(self) -> None:
        """Test extracting lifecycle with string destruction_year."""
        attributes = {
            "lifecycle": {
                "destruction_year": "750",  # String instead of int
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.destruction_year == 750

    def test_extract_with_int_founding_year(self) -> None:
        """Test extracting lifecycle with int founding_year."""
        attributes = {
            "lifecycle": {
                "founding_year": 500,
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.founding_year == 500

    def test_extract_with_int_destruction_year(self) -> None:
        """Test extracting lifecycle with int destruction_year."""
        attributes = {
            "lifecycle": {
                "destruction_year": 750,
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.destruction_year == 750

    def test_extract_with_non_numeric_string_ignored(self) -> None:
        """Test that non-numeric string values are ignored."""
        attributes = {
            "lifecycle": {
                "founding_year": "unknown",
                "destruction_year": "ancient",
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.founding_year is None
        assert lifecycle.destruction_year is None


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


class TestValidateLocation:
    """Tests for location temporal validation."""

    def test_validate_location_without_destruction(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test location without destruction year passes validation."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Ancient Castle",
            description="A castle",
            attributes={
                "lifecycle": {
                    "founding_year": 100,
                }
            },
        )

        result = validation_service.validate_entity(
            entity=location,
            calendar=None,
            all_entities=[location],
            relationships=[],
        )

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_location_event_after_destruction(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test error when event occurs at location after its destruction."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Fallen Kingdom",
            description="A destroyed kingdom",
            attributes={
                "lifecycle": {
                    "founding_year": 100,
                    "destruction_year": 500,  # Destroyed in year 500
                }
            },
        )

        event = Entity(
            id="event-1",
            type="event",
            name="Battle After the Fall",
            description="A battle",
            attributes={
                "lifecycle": {
                    "birth": {"year": 600},  # Occurs in year 600 (after destruction)
                }
            },
        )

        # Event occurred at the destroyed location
        relationships = [("event-1", "loc-1", "occurred_at")]

        result = validation_service.validate_entity(
            entity=location,
            calendar=None,
            all_entities=[location, event],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1
        assert result.errors[0].error_type == TemporalErrorType.POST_DESTRUCTION

    def test_validate_location_event_before_destruction(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test no error when event occurs at location before its destruction."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Fallen Kingdom",
            description="A destroyed kingdom",
            attributes={
                "lifecycle": {
                    "founding_year": 100,
                    "destruction_year": 500,
                }
            },
        )

        event = Entity(
            id="event-1",
            type="event",
            name="Battle Before the Fall",
            description="A battle",
            attributes={
                "lifecycle": {
                    "birth": {"year": 400},  # Occurs in year 400 (before destruction)
                }
            },
        )

        relationships = [("event-1", "loc-1", "occurred_at")]

        result = validation_service.validate_entity(
            entity=location,
            calendar=None,
            all_entities=[location, event],
            relationships=relationships,
        )

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_location_character_located_after_destruction(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test error when character located at destroyed location."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Ruined City",
            description="A ruined city",
            attributes={
                "lifecycle": {
                    "destruction_year": 300,
                }
            },
        )

        character = Entity(
            id="char-1",
            type="character",
            name="Late Arrival",
            description="A character",
            attributes={
                "lifecycle": {
                    "birth": {"year": 400},  # Born after destruction
                }
            },
        )

        relationships = [("char-1", "loc-1", "located_in")]

        result = validation_service.validate_entity(
            entity=location,
            calendar=None,
            all_entities=[location, character],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1


class TestValidateItem:
    """Tests for item temporal validation."""

    def test_validate_item_without_lifecycle(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test item without lifecycle passes validation."""
        item = Entity(
            id="item-1",
            type="item",
            name="Ancient Sword",
            description="A sword",
            attributes={},
        )

        result = validation_service.validate_entity(
            entity=item,
            calendar=None,
            all_entities=[item],
            relationships=[],
        )

        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA

    def test_validate_item_created_before_creator_born(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test error when item created before its creator was born."""
        item = Entity(
            id="item-1",
            type="item",
            name="Magic Wand",
            description="A wand",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},  # Created in year 100
                }
            },
        )

        creator = Entity(
            id="char-1",
            type="character",
            name="Wizard",
            description="A wizard",
            attributes={
                "lifecycle": {
                    "birth": {"year": 200},  # Born in year 200 (after item creation!)
                }
            },
        )

        # Creator relationship
        relationships = [("char-1", "item-1", "created")]

        result = validation_service.validate_entity(
            entity=item,
            calendar=None,
            all_entities=[item, creator],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1
        assert result.errors[0].error_type == TemporalErrorType.PREDATES_DEPENDENCY
        assert "creator" in result.errors[0].message.lower()

    def test_validate_item_created_after_creator_born(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test no error when item created after its creator was born."""
        item = Entity(
            id="item-1",
            type="item",
            name="Magic Wand",
            description="A wand",
            attributes={
                "lifecycle": {
                    "birth": {"year": 300},  # Created in year 300
                }
            },
        )

        creator = Entity(
            id="char-1",
            type="character",
            name="Wizard",
            description="A wizard",
            attributes={
                "lifecycle": {
                    "birth": {"year": 200},  # Born in year 200
                }
            },
        )

        relationships = [("char-1", "item-1", "crafted")]

        result = validation_service.validate_entity(
            entity=item,
            calendar=None,
            all_entities=[item, creator],
            relationships=relationships,
        )

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_item_forged_relationship(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test item with 'forged' relationship type."""
        item = Entity(
            id="item-1",
            type="item",
            name="Legendary Blade",
            description="A blade",
            attributes={
                "lifecycle": {
                    "birth": {"year": 50},  # Forged in year 50
                }
            },
        )

        smith = Entity(
            id="char-1",
            type="character",
            name="Master Smith",
            description="A smith",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},  # Born year 100 (after item forged!)
                }
            },
        )

        relationships = [("char-1", "item-1", "forged")]

        result = validation_service.validate_entity(
            entity=item,
            calendar=None,
            all_entities=[item, smith],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1

    def test_validate_item_without_creation_year(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test item with lifecycle but no creation year."""
        item = Entity(
            id="item-1",
            type="item",
            name="Ancient Artifact",
            description="An artifact",
            attributes={
                "lifecycle": {
                    "temporal_notes": "From before recorded history",
                }
            },
        )

        result = validation_service.validate_entity(
            entity=item,
            calendar=None,
            all_entities=[item],
            relationships=[],
        )

        # No errors, but should warn about missing creation year
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA


class TestValidateWorld:
    """Tests for validate_world method."""

    def test_validate_world_empty(self, validation_service: TemporalValidationService) -> None:
        """Test validating an empty world."""
        mock_world_db = MagicMock()
        mock_world_db.list_entities.return_value = []
        mock_world_db.list_relationships.return_value = []

        result = validation_service.validate_world(mock_world_db)

        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_validate_world_with_valid_entities(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test validating a world with consistent entities."""
        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Order",
            description="A faction",
            attributes={
                "lifecycle": {
                    "founding_year": 100,
                }
            },
        )

        character = Entity(
            id="char-1",
            type="character",
            name="Knight",
            description="A knight",
            attributes={
                "lifecycle": {
                    "birth": {"year": 200},  # Born after faction founded
                }
            },
        )

        mock_rel = MagicMock()
        mock_rel.source_id = "char-1"
        mock_rel.target_id = "faction-1"
        mock_rel.relation_type = "member_of"

        mock_world_db = MagicMock()
        mock_world_db.list_entities.return_value = [faction, character]
        mock_world_db.list_relationships.return_value = [mock_rel]
        mock_world_db.get_world_settings.return_value = None  # No calendar

        result = validation_service.validate_world(mock_world_db)

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_world_with_errors(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test validating a world with temporal errors."""
        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Order",
            description="A faction",
            attributes={
                "lifecycle": {
                    "founding_year": 500,  # Founded late
                }
            },
        )

        character = Entity(
            id="char-1",
            type="character",
            name="Knight",
            description="A knight",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},  # Born before faction founded
                }
            },
        )

        mock_rel = MagicMock()
        mock_rel.source_id = "char-1"
        mock_rel.target_id = "faction-1"
        mock_rel.relation_type = "member_of"

        mock_world_db = MagicMock()
        mock_world_db.list_entities.return_value = [faction, character]
        mock_world_db.list_relationships.return_value = [mock_rel]
        mock_world_db.get_world_settings.return_value = None  # No calendar

        result = validation_service.validate_world(mock_world_db)

        assert result.is_valid is False
        assert result.error_count >= 1

    def test_validate_world_handles_settings_error(
        self, validation_service: TemporalValidationService, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that world validation handles errors when loading settings."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Test",
            description="Test",
            attributes={},
        )

        mock_world_db = MagicMock()
        mock_world_db.list_entities.return_value = [entity]
        mock_world_db.list_relationships.return_value = []
        # Simulate error loading world settings (e.g., table doesn't exist)
        import sqlite3

        mock_world_db.get_world_settings.side_effect = sqlite3.OperationalError("Table not found")

        # Should not raise - should handle gracefully
        with caplog.at_level(logging.WARNING):
            result = validation_service.validate_world(mock_world_db)

        assert result.is_valid is True
        assert result.error_count == 0
        # Calendar failure is logged, not added to result (would skew quality scoring).
        # But the entity has no lifecycle data, so it gets a MISSING_TEMPORAL_DATA warning.
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA
        assert any("Calendar-based validation skipped" in m for m in caplog.messages)


class TestDeathDateValidation:
    """Tests for death date validation against calendar."""

    def test_validate_death_date_invalid(
        self, validation_service: TemporalValidationService, sample_calendar: WorldCalendar
    ) -> None:
        """Test warning for invalid death date against calendar."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Test Character",
            description="A character",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},
                    "death": {"year": -5},  # Before era start (year 1)
                }
            },
        )

        result = validation_service.validate_entity(
            entity=entity,
            calendar=sample_calendar,
            all_entities=[entity],
            relationships=[],
        )

        # Should have warnings for invalid dates
        assert result.warning_count >= 1
        death_warnings = [w for w in result.warnings if "death" in w.message.lower()]
        assert len(death_warnings) >= 1

    def test_validate_death_date_invalid_month(
        self, validation_service: TemporalValidationService, sample_calendar: WorldCalendar
    ) -> None:
        """Test warning for invalid death month against calendar."""
        # Sample calendar only has 2 months, so month 10 is invalid
        entity = Entity(
            id="char-1",
            type="character",
            name="Test Character",
            description="A character",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},
                    "death": {"year": 200, "month": 10},  # Invalid for 2-month calendar
                }
            },
        )

        result = validation_service.validate_entity(
            entity=entity,
            calendar=sample_calendar,
            all_entities=[entity],
            relationships=[],
        )

        assert result.warning_count >= 1

    def test_validate_death_date_invalid_day(
        self, validation_service: TemporalValidationService, sample_calendar: WorldCalendar
    ) -> None:
        """Test warning for invalid death day against calendar."""
        # Month 2 (Midyear) has 30 days, so day 31 is invalid
        entity = Entity(
            id="char-1",
            type="character",
            name="Test Character",
            description="A character",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},
                    "death": {
                        "year": 200,
                        "month": 2,
                        "day": 31,
                    },  # Day 31 invalid for 30-day month
                }
            },
        )

        result = validation_service.validate_entity(
            entity=entity,
            calendar=sample_calendar,
            all_entities=[entity],
            relationships=[],
        )

        assert result.warning_count >= 1


class TestFactionValidationEdgeCases:
    """Tests for faction validation edge cases."""

    def test_validate_faction_without_founding_year(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test faction without founding year skips validation."""
        faction = Entity(
            id="faction-1",
            type="faction",
            name="Ancient Order",
            description="A faction",
            attributes={
                "lifecycle": {
                    "temporal_notes": "Founded in ancient times",
                }
            },
        )

        result = validation_service.validate_entity(
            entity=faction,
            calendar=None,
            all_entities=[faction],
            relationships=[],
        )

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_faction_with_child_of_relationship(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test faction with child_of relationship."""
        child = Entity(
            id="faction-child",
            type="faction",
            name="Child Faction",
            description="An offshoot",
            attributes={
                "lifecycle": {
                    "founding_year": 200,
                }
            },
        )

        parent = Entity(
            id="faction-parent",
            type="faction",
            name="Parent Faction",
            description="The parent",
            attributes={
                "lifecycle": {
                    "founding_year": 400,  # Founded after child!
                }
            },
        )

        relationships = [("faction-child", "faction-parent", "child_of")]

        result = validation_service.validate_entity(
            entity=child,
            calendar=None,
            all_entities=[child, parent],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1
        assert result.errors[0].error_type == TemporalErrorType.FOUNDING_ORDER

    def test_validate_faction_with_offshoot_of_relationship(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test faction with offshoot_of relationship."""
        offshoot = Entity(
            id="faction-offshoot",
            type="faction",
            name="Offshoot Faction",
            description="An offshoot",
            attributes={
                "lifecycle": {
                    "founding_year": 100,
                }
            },
        )

        parent = Entity(
            id="faction-parent",
            type="faction",
            name="Parent Faction",
            description="The parent",
            attributes={
                "lifecycle": {
                    "founding_year": 300,  # Founded after offshoot!
                }
            },
        )

        relationships = [("faction-offshoot", "faction-parent", "offshoot_of")]

        result = validation_service.validate_entity(
            entity=offshoot,
            calendar=None,
            all_entities=[offshoot, parent],
            relationships=relationships,
        )

        assert result.is_valid is False
        assert result.error_count == 1


class TestCharacterValidationEdgeCases:
    """Tests for character validation edge cases."""

    def test_validate_character_without_birth_year(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test character without birth year skips checks."""
        character = Entity(
            id="char-1",
            type="character",
            name="Ageless Being",
            description="A character",
            attributes={
                "lifecycle": {
                    "temporal_notes": "Age unknown",
                }
            },
        )

        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Order",
            description="A faction",
            attributes={
                "lifecycle": {
                    "founding_year": 500,
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

        # No errors, but should warn about missing temporal data
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA

    def test_validate_character_faction_without_founding_year(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Test character-faction when faction has no founding year."""
        character = Entity(
            id="char-1",
            type="character",
            name="Knight",
            description="A knight",
            attributes={
                "lifecycle": {
                    "birth": {"year": 100},
                }
            },
        )

        faction = Entity(
            id="faction-1",
            type="faction",
            name="Ancient Order",
            description="A faction",
            attributes={},  # No lifecycle
        )

        relationships = [("char-1", "faction-1", "member_of")]

        result = validation_service.validate_entity(
            entity=character,
            calendar=None,
            all_entities=[character, faction],
            relationships=relationships,
        )

        # Should pass because faction has no founding year to compare
        assert result.is_valid is True
        assert result.error_count == 0


class TestMissingTemporalDataWarning:
    """Tests for MISSING_TEMPORAL_DATA warning on entities without lifecycle data (#420 L2)."""

    def test_faction_without_lifecycle_gets_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Faction with empty attributes produces a MISSING_TEMPORAL_DATA warning."""
        faction = Entity(
            id="faction-1",
            type="faction",
            name="The Lost Order",
            description="A faction",
            attributes={},
        )

        result = validation_service.validate_entity(
            entity=faction, calendar=None, all_entities=[faction], relationships=[]
        )

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA
        assert "founding year" in result.warnings[0].message

    def test_location_without_lifecycle_gets_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Location with no lifecycle data produces a MISSING_TEMPORAL_DATA warning."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Unknown Ruins",
            description="A location",
            attributes={},
        )

        result = validation_service.validate_entity(
            entity=location, calendar=None, all_entities=[location], relationships=[]
        )

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA
        assert "lifecycle" in result.warnings[0].message

    def test_item_without_lifecycle_gets_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Item with no lifecycle data produces a MISSING_TEMPORAL_DATA warning."""
        item = Entity(
            id="item-1",
            type="item",
            name="Mystery Artifact",
            description="An item",
            attributes={},
        )

        result = validation_service.validate_entity(
            entity=item, calendar=None, all_entities=[item], relationships=[]
        )

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA
        assert "creation year" in result.warnings[0].message

    def test_character_with_birth_but_no_year_gets_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Character with lifecycle.birth present but birth.year=None gets warning."""
        character = Entity(
            id="char-1",
            type="character",
            name="Unknown Age",
            description="A character",
            attributes={"lifecycle": {"birth": {"era_name": "First Age"}}},
        )

        result = validation_service.validate_entity(
            entity=character, calendar=None, all_entities=[character], relationships=[]
        )

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA
        assert "birth year" in result.warnings[0].message

    def test_location_with_empty_lifecycle_dict_gets_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Location with empty lifecycle dict (truthy but no data) gets warning."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Empty Place",
            description="A location",
            attributes={"lifecycle": {}},
        )

        result = validation_service.validate_entity(
            entity=location, calendar=None, all_entities=[location], relationships=[]
        )

        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == TemporalErrorType.MISSING_TEMPORAL_DATA

    def test_location_with_lifecycle_no_destruction_no_warning(
        self, validation_service: TemporalValidationService
    ) -> None:
        """Location with lifecycle but no destruction year should NOT get missing data warning."""
        location = Entity(
            id="loc-1",
            type="location",
            name="Standing City",
            description="A location",
            attributes={"lifecycle": {"founding_year": 100}},
        )

        result = validation_service.validate_entity(
            entity=location, calendar=None, all_entities=[location], relationships=[]
        )

        # Has lifecycle data, just no destruction — that's normal
        assert result.is_valid is True
        assert result.warning_count == 0


class TestSentinelYearRejection:
    """Tests for sentinel year rejection in _parse_year (#395)."""

    def test_sentinel_minus_one_rejected(self) -> None:
        """Test that -1 is rejected as a sentinel value."""
        assert _parse_year(-1, "birth_year") is None

    def test_sentinel_zero_rejected(self) -> None:
        """Test that 0 is rejected as a sentinel value."""
        assert _parse_year(0, "birth_year") is None

    def test_sentinel_9999_rejected(self) -> None:
        """Test that 9999 is rejected as a sentinel value."""
        assert _parse_year(9999, "founding_year") is None

    def test_sentinel_minus_one_as_string_rejected(self) -> None:
        """Test that '-1' string is rejected as sentinel."""
        assert _parse_year("-1", "death_year") is None

    def test_sentinel_zero_as_string_rejected(self) -> None:
        """Test that '0' string is rejected as sentinel."""
        assert _parse_year("0", "birth_year") is None

    def test_sentinel_9999_as_string_rejected(self) -> None:
        """Test that '9999' string is rejected as sentinel."""
        assert _parse_year("9999", "destruction_year") is None

    def test_sentinel_minus_one_as_float_rejected(self) -> None:
        """Test that -1.0 float is rejected as sentinel."""
        assert _parse_year(-1.0, "birth_year") is None

    def test_sentinel_zero_as_float_rejected(self) -> None:
        """Test that 0.0 float is rejected as sentinel."""
        assert _parse_year(0.0, "birth_year") is None

    def test_sentinel_9999_as_float_rejected(self) -> None:
        """Test that 9999.0 float is rejected as sentinel."""
        assert _parse_year(9999.0, "destruction_year") is None

    def test_valid_negative_year_accepted(self) -> None:
        """Test that valid negative years (not -1) are accepted."""
        assert _parse_year(-500, "birth_year") == -500

    def test_valid_positive_year_accepted(self) -> None:
        """Test that normal positive years are accepted."""
        assert _parse_year(500, "birth_year") == 500

    def test_valid_year_one_accepted(self) -> None:
        """Test that year 1 is accepted (not a sentinel)."""
        assert _parse_year(1, "founding_year") == 1

    def test_sentinel_years_frozenset_contents(self) -> None:
        """Test SENTINEL_YEARS contains exactly the expected values."""
        assert SENTINEL_YEARS == frozenset({-1, 0, 9999})


class TestEraMismatchDetection:
    """Tests for _check_era_name_mismatch in temporal validation."""

    @pytest.fixture
    def calendar_with_eras(self) -> WorldCalendar:
        """Create a calendar with historical eras for era mismatch testing."""
        return WorldCalendar(
            current_era_name="Third Age",
            era_abbreviation="TA",
            era_start_year=1,
            months=[CalendarMonth(name="Firstmoon", days=31)],
            days_per_week=7,
            day_names=["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"],
            current_story_year=500,
            eras=[
                HistoricalEra(name="First Age", start_year=1, end_year=100),
                HistoricalEra(name="Second Age", start_year=101, end_year=300),
                HistoricalEra(name="Third Age", start_year=301, end_year=None),
            ],
        )

    def test_era_mismatch_detected(self, validation_service, calendar_with_eras):
        """Test that era name mismatch produces INVALID_ERA warning."""
        entity = Entity(id="ent-001", name="Hero", type="character")
        timestamp = StoryTimestamp(year=50, era_name="Wrong Era")
        result = TemporalValidationResult()

        validation_service._check_era_name_mismatch(
            entity, timestamp, calendar_with_eras, "birth", result
        )

        assert len(result.warnings) == 1
        assert result.warnings[0].error_type == TemporalErrorType.INVALID_ERA
        assert "Wrong Era" in result.warnings[0].message
        assert "First Age" in result.warnings[0].message

    def test_no_mismatch_when_era_matches(self, validation_service, calendar_with_eras):
        """Test no warning when entity era matches calendar era."""
        entity = Entity(id="ent-001", name="Hero", type="character")
        timestamp = StoryTimestamp(year=50, era_name="First Age")
        result = TemporalValidationResult()

        validation_service._check_era_name_mismatch(
            entity, timestamp, calendar_with_eras, "birth", result
        )

        assert len(result.warnings) == 0

    def test_skips_when_year_is_none(self, validation_service, calendar_with_eras):
        """Test early return when timestamp has no year."""
        entity = Entity(id="ent-001", name="Hero", type="character")
        timestamp = StoryTimestamp(year=None, era_name="First Age")
        result = TemporalValidationResult()

        validation_service._check_era_name_mismatch(
            entity, timestamp, calendar_with_eras, "birth", result
        )

        assert len(result.warnings) == 0

    def test_skips_when_era_name_is_empty(self, validation_service, calendar_with_eras):
        """Test early return when timestamp has no era_name."""
        entity = Entity(id="ent-001", name="Hero", type="character")
        timestamp = StoryTimestamp(year=50, era_name="")
        result = TemporalValidationResult()

        validation_service._check_era_name_mismatch(
            entity, timestamp, calendar_with_eras, "birth", result
        )

        assert len(result.warnings) == 0

    def test_skips_when_era_not_resolved(self, validation_service, calendar_with_eras):
        """Test early return when calendar can't resolve era for year."""
        entity = Entity(id="ent-001", name="Hero", type="character")
        # Year -500 is outside all era ranges
        timestamp = StoryTimestamp(year=-500, era_name="Ancient Era")
        result = TemporalValidationResult()

        validation_service._check_era_name_mismatch(
            entity, timestamp, calendar_with_eras, "birth", result
        )

        assert len(result.warnings) == 0
