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


class TestTimelineService:
    """Tests for TimelineService class."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def timeline_service(self, settings):
        """Create TimelineService instance."""
        return TimelineService(settings)

    @pytest.fixture
    def mock_world_db(self):
        """Create a mock WorldDatabase."""
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

    def test_update_entity_lifecycle_not_found(self, timeline_service, mock_world_db):
        """Test updating lifecycle for non-existent entity."""
        mock_world_db.get_entity.return_value = None

        lifecycle = EntityLifecycle(birth=StoryTimestamp(year=1000))

        result = timeline_service.update_entity_lifecycle(mock_world_db, "nonexistent", lifecycle)

        assert result is False
