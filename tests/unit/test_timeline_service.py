"""Tests for the timeline service."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.memory.entities import Entity, WorldEvent
from src.memory.timeline_types import (
    EntityLifecycle,
    StoryTimestamp,
    TimelineItem,
    extract_lifecycle_from_attributes,
    parse_timestamp,
)
from src.services.timeline_service import TimelineService
from src.settings import Settings
from src.utils.constants import ENTITY_COLORS, get_entity_color


class TestEntityColors:
    """Tests for entity color constants and functions."""

    def test_get_entity_color_known_types(self):
        """Test getting colors for known entity types."""
        assert get_entity_color("character") == ENTITY_COLORS["character"]
        assert get_entity_color("location") == ENTITY_COLORS["location"]
        assert get_entity_color("item") == ENTITY_COLORS["item"]
        assert get_entity_color("faction") == ENTITY_COLORS["faction"]
        assert get_entity_color("concept") == ENTITY_COLORS["concept"]
        assert get_entity_color("event") == ENTITY_COLORS["event"]

    def test_get_entity_color_case_insensitive(self):
        """Test getting colors is case insensitive."""
        assert get_entity_color("CHARACTER") == ENTITY_COLORS["character"]
        assert get_entity_color("Location") == ENTITY_COLORS["location"]

    def test_get_entity_color_unknown_type_returns_concept(self):
        """Test unknown entity types return concept color as default."""
        assert get_entity_color("unknown_type") == ENTITY_COLORS["concept"]
        assert get_entity_color("mystical_artifact") == ENTITY_COLORS["concept"]
        assert get_entity_color("") == ENTITY_COLORS["concept"]


class TestStoryTimestamp:
    """Tests for StoryTimestamp model."""

    def test_sort_key_with_full_date(self):
        """Test sort_key with complete calendar date."""
        ts = StoryTimestamp(year=1042, month=3, day=15)
        assert ts.sort_key == (0, 1042, 3, 15)

    def test_sort_key_with_year_only(self):
        """Test sort_key with year only."""
        ts = StoryTimestamp(year=1042)
        assert ts.sort_key == (0, 1042, 0, 0)

    def test_sort_key_with_relative_order(self):
        """Test sort_key with relative order (no calendar date)."""
        ts = StoryTimestamp(relative_order=5)
        assert ts.sort_key == (1, 5, 0, 0)

    def test_sort_key_with_no_data(self):
        """Test sort_key when no temporal data exists."""
        ts = StoryTimestamp()
        assert ts.sort_key == (2, 0, 0, 0)

    def test_display_text_uses_raw_text(self):
        """Test display_text returns raw_text when available."""
        ts = StoryTimestamp(year=1042, raw_text="Year 1042 of the Third Age")
        assert ts.display_text == "Year 1042 of the Third Age"

    def test_display_text_formats_date(self):
        """Test display_text formats date when no raw_text."""
        ts = StoryTimestamp(year=1042, month=3)
        assert "1042" in ts.display_text
        assert "3" in ts.display_text

    def test_display_text_relative_order(self):
        """Test display_text for relative order."""
        ts = StoryTimestamp(relative_order=5)
        assert "5" in ts.display_text

    def test_display_text_with_day(self):
        """Test display_text formats date with day."""
        ts = StoryTimestamp(year=1042, month=3, day=15)
        assert "1042" in ts.display_text
        assert "3" in ts.display_text
        assert "15" in ts.display_text

    def test_display_text_unknown_time(self):
        """Test display_text returns 'Unknown time' when no date info."""
        ts = StoryTimestamp()
        assert ts.display_text == "Unknown time"

    def test_has_date_true(self):
        """Test has_date returns True when date info exists."""
        assert StoryTimestamp(year=1042).has_date is True
        assert StoryTimestamp(relative_order=5).has_date is True

    def test_has_date_false(self):
        """Test has_date returns False when no date info."""
        assert StoryTimestamp().has_date is False


class TestTimelineItem:
    """Tests for TimelineItem model."""

    def test_is_range_with_end(self):
        """Test is_range returns True when end is present."""
        item = TimelineItem(
            id="test-1",
            entity_id="e1",
            label="Test",
            item_type="character",
            start=StoryTimestamp(year=1000),
            end=StoryTimestamp(year=1050),
            color="#000",
        )
        assert item.is_range is True

    def test_is_range_without_end(self):
        """Test is_range returns False when end is None."""
        item = TimelineItem(
            id="test-1",
            entity_id="e1",
            label="Test",
            item_type="event",
            start=StoryTimestamp(year=1000),
            end=None,
            color="#000",
        )
        assert item.is_range is False


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    def test_parse_year_basic(self):
        """Test parsing basic year format."""
        ts = parse_timestamp("Year 1042")
        assert ts.year == 1042
        assert ts.raw_text == "Year 1042"

    def test_parse_year_with_ad(self):
        """Test parsing year with AD suffix."""
        ts = parse_timestamp("1042 AD")
        assert ts.year == 1042

    def test_parse_year_month(self):
        """Test parsing year and month."""
        ts = parse_timestamp("Year 1042, Month 3")
        assert ts.year == 1042
        assert ts.month == 3

    def test_parse_day(self):
        """Test parsing day."""
        ts = parse_timestamp("Day 15 of Month 7, Year 1042")
        assert ts.year == 1042
        assert ts.month == 7
        assert ts.day == 15

    def test_parse_chapter_as_relative(self):
        """Test parsing chapter number as relative order."""
        ts = parse_timestamp("Chapter 5")
        assert ts.relative_order == 5

    def test_parse_event_as_relative(self):
        """Test parsing event number as relative order."""
        ts = parse_timestamp("Event 3")
        assert ts.relative_order == 3

    def test_parse_standalone_day_as_relative(self):
        """Test parsing standalone day as relative order."""
        ts = parse_timestamp("Day 15")
        assert ts.relative_order == 15
        assert ts.day is None  # Day is cleared when used as relative

    def test_parse_preserves_raw_text(self):
        """Test that raw_text is preserved."""
        ts = parse_timestamp("Before the great war")
        assert ts.raw_text == "Before the great war"


class TestExtractLifecycleFromAttributes:
    """Tests for extract_lifecycle_from_attributes function."""

    def test_extract_with_birth_death_dicts(self):
        """Test extracting lifecycle from dict format."""
        attributes = {
            "lifecycle": {
                "birth": {"year": 1000, "raw_text": "Year 1000"},
                "death": {"year": 1080, "raw_text": "Year 1080"},
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.birth is not None
        assert lifecycle.birth.year == 1000
        assert lifecycle.death is not None
        assert lifecycle.death.year == 1080

    def test_extract_with_string_dates(self):
        """Test extracting lifecycle from string format."""
        attributes = {
            "lifecycle": {
                "birth": "Year 1000",
                "death": "Year 1080",
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.birth is not None
        assert lifecycle.birth.year == 1000

    def test_extract_with_first_last_appearance_dict(self):
        """Test extracting first/last appearance from dict format."""
        attributes = {
            "lifecycle": {
                "first_appearance": {"year": 1000, "raw_text": "Year 1000"},
                "last_appearance": {"year": 1080, "raw_text": "Year 1080"},
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.first_appearance is not None
        assert lifecycle.first_appearance.year == 1000
        assert lifecycle.last_appearance is not None
        assert lifecycle.last_appearance.year == 1080

    def test_extract_with_first_last_appearance_string(self):
        """Test extracting first/last appearance from string format."""
        attributes = {
            "lifecycle": {
                "first_appearance": "Chapter 1",
                "last_appearance": "Chapter 10",
            }
        }
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.first_appearance is not None
        assert lifecycle.first_appearance.relative_order == 1
        assert lifecycle.last_appearance is not None
        assert lifecycle.last_appearance.relative_order == 10

    def test_extract_no_lifecycle(self):
        """Test when no lifecycle data exists."""
        attributes = {"other": "data"}
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is None

    def test_extract_empty_lifecycle(self):
        """Test when lifecycle dict is empty."""
        attributes: dict[str, dict] = {"lifecycle": {}}
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is not None
        assert lifecycle.birth is None
        assert lifecycle.death is None

    def test_extract_malformed_lifecycle_not_dict(self):
        """Test when lifecycle data is not a dict (malformed)."""
        from typing import Any

        # String instead of dict
        attributes: dict[str, Any] = {"lifecycle": "invalid string lifecycle"}
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is None

        # List instead of dict
        attributes = {"lifecycle": [1, 2, 3]}
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is None

        # Integer instead of dict
        attributes = {"lifecycle": 42}
        lifecycle = extract_lifecycle_from_attributes(attributes)
        assert lifecycle is None


class TestTimelineService:
    """Tests for TimelineService class."""

    @pytest.fixture
    def settings(self):
        """
        Create a default Settings instance used by tests.

        Returns:
            Settings: a new Settings object with default test configuration.
        """
        return Settings()

    @pytest.fixture
    def timeline_service(self, settings):
        """Create TimelineService instance."""
        return TimelineService(settings)

    @pytest.fixture
    def mock_world_db(self):
        """
        Create a MagicMock that mimics a WorldDatabase with no entities or events.

        Returns:
            mock_db (MagicMock): Mocked WorldDatabase where `list_entities()` and `list_events()` return empty lists.
        """
        mock_db = MagicMock()
        mock_db.list_entities.return_value = []
        mock_db.list_events.return_value = []
        return mock_db

    def test_init(self, timeline_service):
        """Test service initialization."""
        assert timeline_service.settings is not None

    def test_get_timeline_items_empty_world(self, timeline_service, mock_world_db):
        """Test getting timeline items from empty world."""
        items = timeline_service.get_timeline_items(mock_world_db)
        assert items == []

    def test_get_timeline_items_with_entities(self, timeline_service, mock_world_db):
        """Test getting timeline items with entities."""
        entity = Entity(
            id="test-entity-1",
            type="character",
            name="Test Character",
            description="A test character",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "birth": {"year": 1000},
                    "death": {"year": 1080},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].entity_id == "test-entity-1"
        assert items[0].label == "Test Character"
        assert items[0].start.year == 1000
        assert items[0].end is not None
        assert items[0].end.year == 1080

    def test_get_timeline_items_with_first_last_appearance(self, timeline_service, mock_world_db):
        """Test getting timeline items with first/last appearance instead of birth/death."""
        entity = Entity(
            id="test-entity-1",
            type="character",
            name="Mystery Character",
            description="A character",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "first_appearance": {"year": 1000},
                    "last_appearance": {"year": 1080},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].start.year == 1000
        assert items[0].end.year == 1080

    def test_get_timeline_items_no_lifecycle(self, timeline_service, mock_world_db):
        """Test getting timeline items with no lifecycle data."""
        entity = Entity(
            id="test-entity-1",
            type="character",
            name="New Character",
            description="A character",
            created_at=datetime.now(),
            attributes={},  # No lifecycle
        )
        mock_world_db.list_entities.return_value = [entity]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].start is not None
        assert items[0].start.relative_order is not None
        assert "Added:" in items[0].start.raw_text

    def test_get_timeline_items_with_events(self, timeline_service, mock_world_db):
        """Test getting timeline items with events."""
        event = WorldEvent(
            id="test-event-1",
            description="A great battle",
            timestamp_in_story="Year 1050",
            created_at=datetime.now(),
        )
        mock_world_db.list_events.return_value = [event]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].event_id == "test-event-1"
        assert items[0].start.year == 1050
        assert items[0].end is None  # Events are points

    def test_get_timeline_items_event_with_chapter_number(self, timeline_service, mock_world_db):
        """Test getting timeline items with event having chapter number."""
        event = WorldEvent(
            id="test-event-1",
            description="A chapter event",
            chapter_number=5,
            created_at=datetime.now(),
        )
        mock_world_db.list_events.return_value = [event]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].start.relative_order == 5
        assert "Chapter 5" in items[0].start.raw_text

    def test_get_timeline_items_event_no_timestamp(self, timeline_service, mock_world_db):
        """Test getting timeline items with event having no timestamp."""
        event = WorldEvent(
            id="test-event-1",
            description="An undated event",
            chapter_number=None,
            created_at=datetime.now(),
        )
        mock_world_db.list_events.return_value = [event]

        items = timeline_service.get_timeline_items(mock_world_db)

        assert len(items) == 1
        assert items[0].start.relative_order is not None
        assert "Added:" in items[0].start.raw_text

    def test_get_timeline_items_filters_by_type(self, timeline_service, mock_world_db):
        """Test filtering timeline items by entity type."""
        character = Entity(
            id="char-1",
            type="character",
            name="Character",
            description="",
            created_at=datetime.now(),
            attributes={"lifecycle": {"birth": {"year": 1000}}},
        )
        location = Entity(
            id="loc-1",
            type="location",
            name="Location",
            description="",
            created_at=datetime.now(),
            attributes={"lifecycle": {"birth": {"year": 1000}}},
        )
        mock_world_db.list_entities.return_value = [character, location]

        items = timeline_service.get_timeline_items(mock_world_db, entity_types=["character"])

        assert len(items) == 1
        assert items[0].entity_id == "char-1"

    def test_get_timeline_items_excludes_events(self, timeline_service, mock_world_db):
        """Test excluding events from timeline items."""
        event = WorldEvent(
            id="event-1",
            description="Event",
            timestamp_in_story="Year 1050",
            created_at=datetime.now(),
        )
        mock_world_db.list_events.return_value = [event]

        items = timeline_service.get_timeline_items(mock_world_db, include_events=False)

        assert len(items) == 0

    def test_get_entity_lifecycle(self, timeline_service):
        """Test extracting lifecycle from entity."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "birth": {"year": 1000},
                    "death": {"year": 1080},
                }
            },
        )

        lifecycle = timeline_service.get_entity_lifecycle(entity)

        assert lifecycle is not None
        assert lifecycle.birth is not None
        assert lifecycle.birth.year == 1000
        assert lifecycle.death is not None
        assert lifecycle.death.year == 1080

    def test_get_timeline_groups(self, timeline_service):
        """Test getting timeline groups."""
        items = [
            TimelineItem(
                id="1",
                entity_id="e1",
                label="Char",
                item_type="character",
                start=StoryTimestamp(year=1000),
                color="#000",
                group="character",
            ),
            TimelineItem(
                id="2",
                entity_id="e2",
                label="Loc",
                item_type="location",
                start=StoryTimestamp(year=1000),
                color="#000",
                group="location",
            ),
        ]

        groups = timeline_service.get_timeline_groups(items)

        assert len(groups) == 2
        group_ids = [g["id"] for g in groups]
        assert "character" in group_ids
        assert "location" in group_ids

    def test_get_timeline_data_for_visjs(self, timeline_service, mock_world_db):
        """Test generating vis.js formatted data."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={"lifecycle": {"birth": {"year": 1000}}},
        )
        mock_world_db.list_entities.return_value = [entity]

        data = timeline_service.get_timeline_data_for_visjs(mock_world_db)

        assert "items" in data
        assert "groups" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["start"] == "1000-01-01"

    def test_get_timeline_data_for_visjs_relative_order(self, timeline_service, mock_world_db):
        """Test generating vis.js formatted data with relative order."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "first_appearance": {"relative_order": 1},
                    "last_appearance": {"relative_order": 10},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        data = timeline_service.get_timeline_data_for_visjs(mock_world_db)

        assert len(data["items"]) == 1
        assert data["items"][0]["start"] == 1000  # relative_order * 1000
        assert data["items"][0]["end"] == 10000
        assert data["items"][0]["type"] == "range"

    def test_get_timeline_data_for_visjs_relative_order_point(
        self, timeline_service, mock_world_db
    ):
        """Test vis.js data with relative order as point (no end)."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "first_appearance": {"relative_order": 1},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        data = timeline_service.get_timeline_data_for_visjs(mock_world_db)

        assert len(data["items"]) == 1
        assert data["items"][0]["start"] == 1000
        assert data["items"][0]["type"] == "point"

    def test_update_entity_lifecycle(self, timeline_service, mock_world_db):
        """Test updating entity lifecycle."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={},
        )
        mock_world_db.get_entity.return_value = entity
        mock_world_db.update_entity.return_value = True

        lifecycle = EntityLifecycle(
            birth=StoryTimestamp(year=1000),
            death=StoryTimestamp(year=1080),
        )

        result = timeline_service.update_entity_lifecycle(mock_world_db, "test-1", lifecycle)

        assert result is True
        mock_world_db.update_entity.assert_called_once()

    def test_update_entity_lifecycle_with_appearances(self, timeline_service, mock_world_db):
        """Test updating entity lifecycle with first/last appearance."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={},
        )
        mock_world_db.get_entity.return_value = entity
        mock_world_db.update_entity.return_value = True

        lifecycle = EntityLifecycle(
            first_appearance=StoryTimestamp(year=1000),
            last_appearance=StoryTimestamp(year=1080),
        )

        result = timeline_service.update_entity_lifecycle(mock_world_db, "test-1", lifecycle)

        assert result is True
        mock_world_db.update_entity.assert_called_once()
        # Verify the attributes passed include first/last appearance
        call_args = mock_world_db.update_entity.call_args
        assert "first_appearance" in call_args.kwargs["attributes"]["lifecycle"]
        assert "last_appearance" in call_args.kwargs["attributes"]["lifecycle"]

    def test_update_entity_lifecycle_update_fails(self, timeline_service, mock_world_db):
        """Test updating lifecycle when database update fails."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={},
        )
        mock_world_db.get_entity.return_value = entity
        mock_world_db.update_entity.return_value = False

        lifecycle = EntityLifecycle(birth=StoryTimestamp(year=1000))

        result = timeline_service.update_entity_lifecycle(mock_world_db, "test-1", lifecycle)

        assert result is False

    def test_update_entity_lifecycle_not_found(self, timeline_service, mock_world_db):
        """Test updating lifecycle for non-existent entity."""
        mock_world_db.get_entity.return_value = None

        lifecycle = EntityLifecycle(birth=StoryTimestamp(year=1000))

        result = timeline_service.update_entity_lifecycle(mock_world_db, "nonexistent", lifecycle)

        assert result is False

    def test_get_timeline_data_for_visjs_year_range(self, timeline_service, mock_world_db):
        """Test vis.js data with year-based start and end dates (range)."""
        entity = Entity(
            id="test-1",
            type="character",
            name="Test",
            description="",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "birth": {"year": 1000},
                    "death": {"year": 1080},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        data = timeline_service.get_timeline_data_for_visjs(mock_world_db)

        assert len(data["items"]) == 1
        assert data["items"][0]["start"] == "1000-01-01"
        assert data["items"][0]["end"] == "1080-12-31"
        assert data["items"][0]["type"] == "range"

    def test_get_timeline_data_for_visjs_skips_no_temporal(self, timeline_service, mock_world_db):
        """Test vis.js data skips items with no temporal data."""
        from unittest.mock import patch

        # Create items - one with temporal data, one without
        item_with_data = TimelineItem(
            id="item-1",
            entity_id="e1",
            label="Has Data",
            item_type="character",
            start=StoryTimestamp(year=1000),
            color="#000",
            group="character",
        )
        item_without_data = TimelineItem(
            id="item-2",
            entity_id="e2",
            label="No Data",
            item_type="character",
            start=StoryTimestamp(),  # No year, no relative_order
            color="#000",
            group="character",
        )

        # Mock get_timeline_items to return our test items
        with patch.object(
            timeline_service, "get_timeline_items", return_value=[item_with_data, item_without_data]
        ):
            data = timeline_service.get_timeline_data_for_visjs(mock_world_db)

        # Should only have one item (the one with temporal data)
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "item-1"

    def test_build_temporal_context_empty_world(self, timeline_service, mock_world_db):
        """Return empty string when no timeline items exist."""
        result = timeline_service.build_temporal_context(mock_world_db)
        assert result == ""

    def test_build_temporal_context_with_entities(self, timeline_service, mock_world_db):
        """Return formatted sections grouped by entity type."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Hero",
            description="The protagonist",
            created_at=datetime.now(),
            attributes={
                "lifecycle": {
                    "birth": {"year": 100, "raw_text": "Year 100"},
                    "death": {"year": 180, "raw_text": "Year 180"},
                }
            },
        )
        mock_world_db.list_entities.return_value = [entity]

        result = timeline_service.build_temporal_context(mock_world_db)

        assert "CHARACTERS:" in result
        assert "Hero" in result
        assert "Year 100" in result
        assert "Year 180" in result

    def test_build_temporal_context_multiple_types(self, timeline_service, mock_world_db):
        """Return sections for each entity type present."""
        char = Entity(
            id="char-1",
            type="character",
            name="Hero",
            description="Protagonist",
            created_at=datetime.now(),
            attributes={"lifecycle": {"birth": {"year": 100}}},
        )
        faction = Entity(
            id="fac-1",
            type="faction",
            name="Knights",
            description="A faction",
            created_at=datetime.now(),
            attributes={"lifecycle": {"first_appearance": {"year": 50}}},
        )
        mock_world_db.list_entities.return_value = [char, faction]

        result = timeline_service.build_temporal_context(mock_world_db)

        assert "CHARACTERS:" in result
        assert "FACTIONS:" in result
        # Characters should come before factions (type_order)
        assert result.index("CHARACTERS:") < result.index("FACTIONS:")

    def test_build_temporal_context_point_event(self, timeline_service, mock_world_db):
        """Format items without end date as point events (no 'to')."""
        entity = Entity(
            id="char-1",
            type="character",
            name="Wanderer",
            description="A traveler",
            created_at=datetime.now(),
            attributes={"lifecycle": {"first_appearance": {"year": 200}}},
        )
        mock_world_db.list_entities.return_value = [entity]

        result = timeline_service.build_temporal_context(mock_world_db)

        assert "Wanderer: " in result
        assert " to " not in result

    def test_build_temporal_context_caps_at_twenty(self, timeline_service, mock_world_db):
        """Cap each type section at 20 items."""
        entities = [
            Entity(
                id=f"char-{i}",
                type="character",
                name=f"Char{i}",
                description=f"Character {i}",
                created_at=datetime.now(),
                attributes={"lifecycle": {"birth": {"year": 1000 + i}}},
            )
            for i in range(25)
        ]
        mock_world_db.list_entities.return_value = entities

        result = timeline_service.build_temporal_context(mock_world_db)

        assert "... and 5 more" in result
        # Should have exactly 20 "- Char" lines
        char_lines = [line for line in result.split("\n") if line.startswith("- Char")]
        assert len(char_lines) == 20
