"""World Calendar Models - fictional calendar system for story worlds.

Contains Pydantic models for:
- Calendar months with custom names and lengths
- Historical eras for timeline organization
- WorldCalendar with date formatting and validation
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class CalendarMonth(BaseModel):
    """A month in the fictional calendar.

    Months can have custom names and varying lengths to match
    the world's unique timekeeping system.
    """

    name: str = Field(description="Custom name for the month (e.g., 'Frostfall', 'Sunharvest')")
    days: int = Field(default=30, ge=1, le=100, description="Number of days in this month")
    description: str = Field(default="", description="Flavor text describing this month's nature")

    model_config = ConfigDict(use_enum_values=True)


class HistoricalEra(BaseModel):
    """A historical era in the world's timeline.

    Eras provide narrative context for timeline events and help
    organize history into meaningful periods.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique era identifier")
    name: str = Field(description="Name of the era (e.g., 'Age of Dragons', 'The Long Winter')")
    start_year: int = Field(description="First year of this era")
    end_year: int | None = Field(
        default=None, description="Last year of this era (None if ongoing)"
    )
    description: str = Field(default="", description="Historical context for this era")
    display_order: int = Field(default=0, description="Order for display in timelines")

    model_config = ConfigDict(use_enum_values=True)

    @property
    def is_ongoing(self) -> bool:
        """Check if this era is still ongoing (no end year)."""
        return self.end_year is None

    @property
    def duration(self) -> int | None:
        """Get the duration of this era in years, or None if ongoing."""
        if self.end_year is None:
            return None
        return self.end_year - self.start_year + 1


class WorldCalendar(BaseModel):
    """A fictional calendar system for a story world.

    Provides custom month names, era tracking, and date formatting
    for consistent timeline representation.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique calendar ID")
    current_era_name: str = Field(
        description="Name of the current era (e.g., 'Third Age', 'After the Fall')"
    )
    era_abbreviation: str = Field(
        default="AE", description="Short abbreviation for era (e.g., 'AD', 'TE', 'AE')"
    )
    era_start_year: int = Field(default=1, description="Year when the current era began")
    months: list[CalendarMonth] = Field(default_factory=list, description="List of months in order")
    days_per_week: int = Field(default=7, ge=1, le=20, description="Number of days in a week")
    day_names: list[str] = Field(default_factory=list, description="Custom names for weekdays")
    current_story_year: int = Field(description="Current year when the story takes place")
    story_start_year: int | None = Field(
        default=None, description="Year when the story narrative begins"
    )
    story_end_year: int | None = Field(
        default=None, description="Year when the story narrative ends (if known)"
    )
    eras: list[HistoricalEra] = Field(
        default_factory=list, description="Historical eras in chronological order"
    )
    date_format: str = Field(
        default="{day} {month}, Year {year} {era}",
        description="Format template for displaying dates",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("months")
    @classmethod
    def validate_months(cls, v: list[CalendarMonth]) -> list[CalendarMonth]:
        """Log a debug message if the months list is empty.

        Empty months are allowed; fallback values are used in properties.
        """
        if v is not None and len(v) == 0:
            logger.debug("Empty months list provided, fallback values will be used")
        return v

    @property
    def total_days_per_year(self) -> int:
        """Calculate total days in a year."""
        if not self.months:
            return 365  # Default fallback
        return sum(month.days for month in self.months)

    @property
    def month_count(self) -> int:
        """Get the number of months in a year."""
        return len(self.months) if self.months else 12

    def get_month_by_number(self, month_number: int) -> CalendarMonth | None:
        """Get a month by its 1-based number.

        Args:
            month_number: 1-based month number.

        Returns:
            CalendarMonth if found, None otherwise.
        """
        if not self.months or month_number < 1 or month_number > len(self.months):
            logger.debug(f"Invalid month number {month_number}, total months: {len(self.months)}")
            return None
        return self.months[month_number - 1]

    def get_month_name(self, month_number: int) -> str:
        """Get the name of a month by its 1-based number.

        Args:
            month_number: 1-based month number.

        Returns:
            Month name or "Month N" if not found.
        """
        month = self.get_month_by_number(month_number)
        if month:
            return month.name
        return f"Month {month_number}"

    def format_date(
        self,
        year: int,
        month: int | None = None,
        day: int | None = None,
        era_name: str | None = None,
    ) -> str:
        """Format a date according to the calendar's format template.

        Args:
            year: Year number.
            month: Optional month number (1-based).
            day: Optional day of month.
            era_name: Optional era name override.

        Returns:
            Formatted date string.
        """
        era = era_name if era_name else self.era_abbreviation
        month_name = self.get_month_name(month) if month else ""

        # Build the date string based on available components
        if day is not None and month is not None:
            result = self.date_format.format(
                day=day,
                month=month_name,
                year=year,
                era=era,
            )
        elif month is not None:
            # No day, just month and year
            result = f"{month_name}, Year {year} {era}"
        else:
            # Just year
            result = f"Year {year} {era}"

        logger.debug(f"Formatted date: year={year}, month={month}, day={day} -> {result}")
        return result

    def validate_date(
        self,
        year: int,
        month: int | None = None,
        day: int | None = None,
    ) -> tuple[bool, str]:
        """Validate a date against the calendar rules.

        Args:
            year: Year to validate.
            month: Optional month (1-based).
            day: Optional day of month.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Year validation — use era-aware check when eras are defined
        if self.eras:
            era = self.get_era_for_year(year)
            if era is None:
                # Year doesn't fall within ANY defined era
                earliest = min(e.start_year for e in self.eras)
                return (
                    False,
                    f"Year {year} is outside all defined eras (earliest era starts at {earliest})",
                )
        else:
            # No eras defined — fall back to current era start
            if year < self.era_start_year:
                return False, f"Year {year} is before era start ({self.era_start_year})"

        # Month validation - use month_count for consistency with empty months fallback
        if month is not None:
            max_month = self.month_count
            if month < 1 or month > max_month:
                return False, f"Month {month} is invalid (1-{max_month})"

            # Day validation
            if day is not None:
                month_obj = self.get_month_by_number(month)
                if month_obj is None:
                    # Month metadata missing (empty months list) - can't validate day
                    return (
                        False,
                        f"Cannot validate day {day} for month {month}: month metadata missing",
                    )
                if day < 1 or day > month_obj.days:
                    return False, f"Day {day} is invalid for {month_obj.name} (1-{month_obj.days})"

        logger.debug(f"Date validation passed: year={year}, month={month}, day={day}")
        return True, ""

    def get_era_for_year(self, year: int) -> HistoricalEra | None:
        """Find which historical era a given year falls within.

        Args:
            year: Year to check.

        Returns:
            HistoricalEra if found, None otherwise.
        """
        for era in self.eras:
            if era.start_year <= year:
                if era.end_year is None or year <= era.end_year:
                    logger.debug(f"Year {year} is in era '{era.name}'")
                    return era
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert calendar to dictionary for storage.

        Returns:
            Dictionary representation suitable for JSON storage.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldCalendar:
        """Create a WorldCalendar from dictionary data.

        Args:
            data: Dictionary with calendar data.

        Returns:
            WorldCalendar instance.
        """
        return cls.model_validate(data)


def create_default_calendar(
    era_name: str = "Common Era",
    era_abbrev: str = "CE",
    current_year: int = 1000,
) -> WorldCalendar:
    """Create a default calendar with standard months.

    Args:
        era_name: Name for the current era.
        era_abbrev: Abbreviation for the era.
        current_year: Current story year.

    Returns:
        WorldCalendar with 12 standard months.
    """
    months = [
        CalendarMonth(name="Firstmoon", days=31, description="The month of new beginnings"),
        CalendarMonth(name="Snowmelt", days=28, description="When winter's grip loosens"),
        CalendarMonth(name="Seedsow", days=31, description="Time of planting"),
        CalendarMonth(name="Rainbloom", days=30, description="Spring rains bring flowers"),
        CalendarMonth(name="Sunpeak", days=31, description="Days grow long"),
        CalendarMonth(name="Highsun", days=30, description="The warmest month"),
        CalendarMonth(name="Goldleaf", days=31, description="Harvest begins"),
        CalendarMonth(name="Reaping", days=31, description="Harvest festival time"),
        CalendarMonth(name="Mistfall", days=30, description="Autumn mists descend"),
        CalendarMonth(name="Dimlight", days=31, description="Days grow shorter"),
        CalendarMonth(name="Frostdeep", days=30, description="Winter's hold tightens"),
        CalendarMonth(name="Lastnight", days=31, description="The darkest month"),
    ]

    day_names = [
        "Sunrest",
        "Moonrise",
        "Starsday",
        "Earthsday",
        "Windsday",
        "Firesday",
        "Waterday",
    ]

    logger.info(f"Created default calendar: {era_name} ({era_abbrev}), year {current_year}")
    return WorldCalendar(
        current_era_name=era_name,
        era_abbreviation=era_abbrev,
        era_start_year=1,
        months=months,
        days_per_week=7,
        day_names=day_names,
        current_story_year=current_year,
        eras=[
            HistoricalEra(
                name=era_name,
                start_year=1,
                description="The current age of the world",
            )
        ],
    )
