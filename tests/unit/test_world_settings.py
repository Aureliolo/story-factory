"""Tests for world settings models."""

from src.memory.world_calendar import CalendarMonth, WorldCalendar
from src.memory.world_settings import WorldSettings, create_default_world_settings


class TestWorldSettings:
    """Tests for WorldSettings model."""

    def test_create_settings_with_defaults(self) -> None:
        """Test creating settings with default values."""
        settings = WorldSettings()
        assert settings.id is not None
        assert settings.calendar is None
        assert settings.timeline_start_year is None
        assert settings.timeline_end_year is None
        assert settings.validate_temporal_consistency is True
        assert settings.created_at is not None
        assert settings.updated_at is not None

    def test_create_settings_with_calendar(self) -> None:
        """Test creating settings with a calendar."""
        calendar = WorldCalendar(
            current_era_name="Third Age",
            era_abbreviation="TA",
            era_start_year=1,
            months=[CalendarMonth(name="Firstmoon", days=30)],
            days_per_week=7,
            day_names=["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"],
            current_story_year=500,
        )
        settings = WorldSettings(calendar=calendar)
        assert settings.calendar is not None
        assert settings.calendar.current_era_name == "Third Age"

    def test_has_calendar_property(self) -> None:
        """Test has_calendar property."""
        settings = WorldSettings()
        assert settings.has_calendar is False

        calendar = WorldCalendar(
            current_era_name="Test Era",
            era_abbreviation="TE",
            era_start_year=1,
            months=[CalendarMonth(name="Month1", days=30)],
            days_per_week=7,
            day_names=["D1", "D2", "D3", "D4", "D5", "D6", "D7"],
            current_story_year=100,
        )
        settings_with_calendar = WorldSettings(calendar=calendar)
        assert settings_with_calendar.has_calendar is True

    def test_timeline_span_with_both_bounds(self) -> None:
        """Test timeline_span when both bounds are set."""
        settings = WorldSettings(
            timeline_start_year=100,
            timeline_end_year=500,
        )
        assert settings.timeline_span == 400

    def test_timeline_span_with_missing_start(self) -> None:
        """Test timeline_span when start year is missing."""
        settings = WorldSettings(timeline_end_year=500)
        assert settings.timeline_span is None

    def test_timeline_span_with_missing_end(self) -> None:
        """Test timeline_span when end year is missing."""
        settings = WorldSettings(timeline_start_year=100)
        assert settings.timeline_span is None

    def test_get_era_abbreviation_without_calendar(self) -> None:
        """Test get_era_abbreviation without calendar."""
        settings = WorldSettings()
        assert settings.get_era_abbreviation() == "Y"

    def test_get_era_abbreviation_with_calendar(self) -> None:
        """Test get_era_abbreviation with calendar."""
        calendar = WorldCalendar(
            current_era_name="Third Age",
            era_abbreviation="TA",
            era_start_year=1,
            months=[CalendarMonth(name="Month1", days=30)],
            days_per_week=7,
            day_names=["D1"],
            current_story_year=500,
        )
        settings = WorldSettings(calendar=calendar)
        assert settings.get_era_abbreviation() == "TA"

    def test_format_year_without_calendar(self) -> None:
        """Test format_year without calendar."""
        settings = WorldSettings()
        result = settings.format_year(500)
        assert result == "Year 500 Y"

    def test_format_year_with_calendar(self) -> None:
        """Test format_year with calendar."""
        calendar = WorldCalendar(
            current_era_name="Third Age",
            era_abbreviation="TA",
            era_start_year=1,
            months=[CalendarMonth(name="Month1", days=30)],
            days_per_week=7,
            day_names=["D1"],
            current_story_year=500,
        )
        settings = WorldSettings(calendar=calendar)
        result = settings.format_year(342)
        assert "342" in result
        assert "TA" in result

    def test_to_dict_without_calendar(self) -> None:
        """Test to_dict without calendar."""
        settings = WorldSettings(
            id="test-id",
            timeline_start_year=100,
            timeline_end_year=500,
        )
        data = settings.to_dict()
        assert data["id"] == "test-id"
        assert data["timeline_start_year"] == 100
        assert data["timeline_end_year"] == 500
        assert data["calendar"] is None

    def test_to_dict_with_calendar(self) -> None:
        """Test to_dict with calendar."""
        calendar = WorldCalendar(
            current_era_name="Test Era",
            era_abbreviation="TE",
            era_start_year=1,
            months=[CalendarMonth(name="Month1", days=30)],
            days_per_week=7,
            day_names=["D1"],
            current_story_year=100,
        )
        settings = WorldSettings(calendar=calendar)
        data = settings.to_dict()
        assert data["calendar"] is not None
        assert data["calendar"]["current_era_name"] == "Test Era"

    def test_from_dict_without_calendar(self) -> None:
        """Test from_dict without calendar."""
        data = {
            "id": "test-id",
            "timeline_start_year": 100,
            "timeline_end_year": 500,
            "validate_temporal_consistency": False,
            "calendar": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        settings = WorldSettings.from_dict(data)
        assert settings.id == "test-id"
        assert settings.timeline_start_year == 100
        assert settings.calendar is None
        assert settings.validate_temporal_consistency is False

    def test_from_dict_with_calendar(self) -> None:
        """Test from_dict with calendar data."""
        calendar_data = {
            "current_era_name": "Test Era",
            "era_abbreviation": "TE",
            "era_start_year": 1,
            "months": [{"name": "Month1", "days": 30, "description": ""}],
            "days_per_week": 7,
            "day_names": ["D1"],
            "current_story_year": 100,
            "eras": [],
            "date_format": "{day} {month}, Year {year} {era}",
        }
        data = {
            "id": "test-id",
            "calendar": calendar_data,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        settings = WorldSettings.from_dict(data)
        assert settings.calendar is not None
        assert settings.calendar.current_era_name == "Test Era"


class TestCreateDefaultWorldSettings:
    """Tests for create_default_world_settings function."""

    def test_creates_settings_without_calendar(self) -> None:
        """Test that default settings have no calendar."""
        settings = create_default_world_settings()
        assert settings.calendar is None
        assert settings.has_calendar is False

    def test_creates_settings_with_validation_enabled(self) -> None:
        """Test that default settings have validation enabled."""
        settings = create_default_world_settings()
        assert settings.validate_temporal_consistency is True

    def test_creates_unique_ids(self) -> None:
        """Test that each call creates unique settings."""
        settings1 = create_default_world_settings()
        settings2 = create_default_world_settings()
        assert settings1.id != settings2.id
