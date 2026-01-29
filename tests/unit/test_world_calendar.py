"""Tests for world calendar models."""

import pytest

from src.memory.world_calendar import (
    CalendarMonth,
    HistoricalEra,
    WorldCalendar,
    create_default_calendar,
)


class TestCalendarMonth:
    """Tests for CalendarMonth model."""

    def test_create_month_with_defaults(self) -> None:
        """Test creating a month with default values."""
        month = CalendarMonth(name="Frostfall")
        assert month.name == "Frostfall"
        assert month.days == 30
        assert month.description == ""

    def test_create_month_with_custom_values(self) -> None:
        """Test creating a month with custom values."""
        month = CalendarMonth(
            name="Highsun",
            days=45,
            description="The longest days of summer",
        )
        assert month.name == "Highsun"
        assert month.days == 45
        assert month.description == "The longest days of summer"

    def test_month_days_validation(self) -> None:
        """Test that month days are validated within bounds."""
        month = CalendarMonth(name="Test", days=1)
        assert month.days == 1

        month = CalendarMonth(name="Test", days=100)
        assert month.days == 100

        with pytest.raises(ValueError):
            CalendarMonth(name="Test", days=0)

        with pytest.raises(ValueError):
            CalendarMonth(name="Test", days=101)


class TestHistoricalEra:
    """Tests for HistoricalEra model."""

    def test_create_ongoing_era(self) -> None:
        """Test creating an ongoing era without end year."""
        era = HistoricalEra(
            name="Age of Dragons",
            start_year=500,
            description="When dragons ruled the skies",
        )
        assert era.name == "Age of Dragons"
        assert era.start_year == 500
        assert era.end_year is None
        assert era.is_ongoing is True
        assert era.duration is None

    def test_create_completed_era(self) -> None:
        """Test creating a completed era with end year."""
        era = HistoricalEra(
            name="The Long Winter",
            start_year=100,
            end_year=250,
            description="150 years of endless cold",
        )
        assert era.name == "The Long Winter"
        assert era.start_year == 100
        assert era.end_year == 250
        assert era.is_ongoing is False
        assert era.duration == 151  # Inclusive


class TestWorldCalendar:
    """Tests for WorldCalendar model."""

    @pytest.fixture
    def sample_calendar(self) -> WorldCalendar:
        """Create a sample calendar for testing."""
        return WorldCalendar(
            current_era_name="Third Age",
            era_abbreviation="TA",
            era_start_year=1,
            months=[
                CalendarMonth(name="Firstmoon", days=31),
                CalendarMonth(name="Snowmelt", days=28),
                CalendarMonth(name="Springtide", days=30),
            ],
            days_per_week=7,
            day_names=[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ],
            current_story_year=342,
            eras=[
                HistoricalEra(name="First Age", start_year=1, end_year=500),
                HistoricalEra(name="Second Age", start_year=501, end_year=1000),
                HistoricalEra(name="Third Age", start_year=1001),
            ],
        )

    def test_total_days_per_year(self, sample_calendar: WorldCalendar) -> None:
        """Test calculating total days per year."""
        assert sample_calendar.total_days_per_year == 89  # 31 + 28 + 30

    def test_month_count(self, sample_calendar: WorldCalendar) -> None:
        """Test getting month count."""
        assert sample_calendar.month_count == 3

    def test_get_month_by_number(self, sample_calendar: WorldCalendar) -> None:
        """Test getting month by 1-based number."""
        month = sample_calendar.get_month_by_number(1)
        assert month is not None
        assert month.name == "Firstmoon"

        month = sample_calendar.get_month_by_number(3)
        assert month is not None
        assert month.name == "Springtide"

        # Invalid month numbers
        assert sample_calendar.get_month_by_number(0) is None
        assert sample_calendar.get_month_by_number(4) is None

    def test_get_month_name(self, sample_calendar: WorldCalendar) -> None:
        """Test getting month name by number."""
        assert sample_calendar.get_month_name(1) == "Firstmoon"
        assert sample_calendar.get_month_name(2) == "Snowmelt"
        assert sample_calendar.get_month_name(99) == "Month 99"  # Fallback

    def test_format_date_year_only(self, sample_calendar: WorldCalendar) -> None:
        """Test formatting a date with just year."""
        result = sample_calendar.format_date(342)
        assert "342" in result
        assert "TA" in result

    def test_format_date_with_month(self, sample_calendar: WorldCalendar) -> None:
        """Test formatting a date with year and month."""
        result = sample_calendar.format_date(342, month=2)
        assert "Snowmelt" in result
        assert "342" in result
        assert "TA" in result

    def test_format_date_full(self, sample_calendar: WorldCalendar) -> None:
        """Test formatting a full date."""
        result = sample_calendar.format_date(342, month=1, day=15)
        assert "15" in result
        assert "Firstmoon" in result
        assert "342" in result
        assert "TA" in result

    def test_validate_date_valid(self, sample_calendar: WorldCalendar) -> None:
        """Test validating a valid date."""
        is_valid, error = sample_calendar.validate_date(342, month=1, day=15)
        assert is_valid is True
        assert error == ""

    def test_validate_date_invalid_year(self, sample_calendar: WorldCalendar) -> None:
        """Test validating a date before era start."""
        is_valid, error = sample_calendar.validate_date(0)
        assert is_valid is False
        assert "before era start" in error

    def test_validate_date_invalid_month(self, sample_calendar: WorldCalendar) -> None:
        """Test validating an invalid month."""
        is_valid, error = sample_calendar.validate_date(342, month=99)
        assert is_valid is False
        assert "Month 99 is invalid" in error

    def test_validate_date_invalid_day(self, sample_calendar: WorldCalendar) -> None:
        """Test validating an invalid day."""
        is_valid, error = sample_calendar.validate_date(342, month=2, day=30)
        assert is_valid is False
        assert "Day 30 is invalid" in error
        assert "Snowmelt" in error

    def test_get_era_for_year(self, sample_calendar: WorldCalendar) -> None:
        """Test finding era for a given year."""
        era = sample_calendar.get_era_for_year(250)
        assert era is not None
        assert era.name == "First Age"

        era = sample_calendar.get_era_for_year(750)
        assert era is not None
        assert era.name == "Second Age"

        era = sample_calendar.get_era_for_year(2000)
        assert era is not None
        assert era.name == "Third Age"

    def test_to_dict_and_from_dict(self, sample_calendar: WorldCalendar) -> None:
        """Test serialization and deserialization."""
        data = sample_calendar.to_dict()
        restored = WorldCalendar.from_dict(data)

        assert restored.current_era_name == sample_calendar.current_era_name
        assert restored.era_abbreviation == sample_calendar.era_abbreviation
        assert restored.current_story_year == sample_calendar.current_story_year
        assert len(restored.months) == len(sample_calendar.months)
        assert len(restored.eras) == len(sample_calendar.eras)


class TestCreateDefaultCalendar:
    """Tests for create_default_calendar function."""

    def test_creates_calendar_with_defaults(self) -> None:
        """Test creating a default calendar."""
        calendar = create_default_calendar()

        assert calendar.current_era_name == "Common Era"
        assert calendar.era_abbreviation == "CE"
        assert calendar.current_story_year == 1000
        assert len(calendar.months) == 12
        assert len(calendar.day_names) == 7
        assert len(calendar.eras) == 1

    def test_creates_calendar_with_custom_values(self) -> None:
        """Test creating a calendar with custom era and year."""
        calendar = create_default_calendar(
            era_name="Age of Wonders",
            era_abbrev="AW",
            current_year=500,
        )

        assert calendar.current_era_name == "Age of Wonders"
        assert calendar.era_abbreviation == "AW"
        assert calendar.current_story_year == 500


class TestWorldCalendarEdgeCases:
    """Tests for edge cases in WorldCalendar."""

    def test_empty_months_list(self) -> None:
        """Test calendar with empty months list."""
        calendar = WorldCalendar(
            current_era_name="Test Era",
            era_abbreviation="TE",
            era_start_year=1,
            months=[],  # Empty list
            days_per_week=7,
            day_names=["D1"],
            current_story_year=100,
        )
        # Should use fallback for total_days_per_year
        assert calendar.total_days_per_year == 365
        assert calendar.month_count == 12

    def test_calendar_without_eras_list(self) -> None:
        """Test calendar without historical eras."""
        calendar = WorldCalendar(
            current_era_name="Single Era",
            era_abbreviation="SE",
            era_start_year=1,
            months=[CalendarMonth(name="Month1", days=30)],
            days_per_week=7,
            day_names=["D1"],
            current_story_year=100,
            eras=[],
        )
        assert len(calendar.eras) == 0
        assert calendar.get_era_for_year(50) is None

    def test_validate_date_day_without_month_metadata(self) -> None:
        """Test validate_date returns error when day provided but month metadata missing."""
        calendar = WorldCalendar(
            current_era_name="Test Era",
            era_abbreviation="TE",
            era_start_year=1,
            months=[],  # Empty list - no month metadata
            days_per_week=7,
            day_names=["D1"],
            current_story_year=100,
        )
        # Day validation should fail because month metadata is missing
        is_valid, error_msg = calendar.validate_date(50, month=5, day=15)
        assert not is_valid
        assert "month metadata missing" in error_msg
