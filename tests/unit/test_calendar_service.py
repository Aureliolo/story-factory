"""Tests for calendar service."""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import StoryBrief
from src.memory.world_calendar import WorldCalendar
from src.services.calendar_service import CalendarService, GeneratedCalendarData
from src.settings import Settings
from src.utils.exceptions import CalendarGenerationError


@pytest.fixture
def settings() -> Settings:
    """Create settings for testing."""
    return Settings.load()


@pytest.fixture
def calendar_service(settings: Settings) -> CalendarService:
    """Create a calendar service for testing."""
    return CalendarService(settings)


@pytest.fixture
def sample_brief() -> StoryBrief:
    """Create a sample story brief for testing."""
    return StoryBrief(
        premise="A young wizard discovers ancient powers",
        genre="Fantasy",
        subgenres=["Epic", "Adventure"],
        tone="Heroic",
        themes=["Coming of age", "Power and responsibility"],
        setting_time="Ancient",
        setting_place="Magical kingdom",
        target_length="novella",
        language="English",
        content_rating="none",
        content_preferences=[],
        content_avoid=[],
    )


class TestCalendarServiceInit:
    """Tests for CalendarService initialization."""

    def test_init_creates_service(self, settings: Settings) -> None:
        """Test that service initializes correctly."""
        service = CalendarService(settings)
        assert service.settings == settings
        assert service._agent is None

    def test_get_agent_creates_agent_lazily(self, calendar_service: CalendarService) -> None:
        """Test that agent is created on first use."""
        assert calendar_service._agent is None
        agent = calendar_service._get_agent()
        assert agent is not None
        assert calendar_service._agent is agent

    def test_get_agent_returns_same_instance(self, calendar_service: CalendarService) -> None:
        """Test that agent is cached."""
        agent1 = calendar_service._get_agent()
        agent2 = calendar_service._get_agent()
        assert agent1 is agent2


class TestGenerateCalendar:
    """Tests for generate_calendar method."""

    def test_generate_calendar_success(
        self, calendar_service: CalendarService, sample_brief: StoryBrief
    ) -> None:
        """Test successful calendar generation."""
        # Mock the generated calendar data
        mock_data = GeneratedCalendarData(
            era_name="Third Age",
            era_abbreviation="TA",
            current_year=500,
            months=[
                {"name": "Frostmoon", "days": 30, "description": "Cold month"},
                {"name": "Sunpeak", "days": 31, "description": "Warm month"},
            ],
            day_names=[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ],
            historical_eras=[
                {
                    "name": "First Age",
                    "start_year": 1,
                    "end_year": 200,
                    "description": "The beginning",
                },
                {
                    "name": "Second Age",
                    "start_year": 201,
                    "end_year": 400,
                    "description": "Middle times",
                },
                {
                    "name": "Third Age",
                    "start_year": 401,
                    "end_year": None,
                    "description": "Current era",
                },
            ],
            date_format="{day} {month}, Year {year} {era}",
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar(sample_brief)

        assert isinstance(result, WorldCalendar)
        assert result.current_era_name == "Third Age"
        assert result.era_abbreviation == "TA"
        assert result.current_story_year == 500
        assert len(result.months) == 2
        assert result.months[0].name == "Frostmoon"
        assert len(result.eras) == 3
        assert len(result.day_names) == 7

    def test_generate_calendar_with_world_template(
        self, calendar_service: CalendarService, sample_brief: StoryBrief
    ) -> None:
        """Test calendar generation with world template name."""
        mock_data = GeneratedCalendarData(
            era_name="Age of Dragons",
            era_abbreviation="AD",
            current_year=1000,
            months=[{"name": "Dragonmoon", "days": 30, "description": "Dragon season"}],
            day_names=["Day1", "Day2", "Day3", "Day4", "Day5"],
            historical_eras=[
                {"name": "Age of Dragons", "start_year": 1, "end_year": None, "description": "Now"}
            ],
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar(
                sample_brief, world_template_name="Dragon Realm"
            )

        assert result.current_era_name == "Age of Dragons"
        # Verify template was passed to prompt (agent was called)
        mock_agent.generate_structured.assert_called_once()

    def test_generate_calendar_handles_missing_fields(
        self, calendar_service: CalendarService, sample_brief: StoryBrief
    ) -> None:
        """Test calendar generation with missing optional fields in response."""
        mock_data = GeneratedCalendarData(
            era_name="Test Era",
            era_abbreviation="TE",
            current_year=100,
            months=[{}],  # Missing name, days, description
            day_names=["D1"],
            historical_eras=[{}],  # Missing fields
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar(sample_brief)

        # Should use fallback values
        assert result.months[0].name == "Month 1"
        assert result.months[0].days == 30
        assert result.eras[0].name == "Era 1"
        assert result.eras[0].start_year == 1

    def test_generate_calendar_raises_on_agent_error(
        self, calendar_service: CalendarService, sample_brief: StoryBrief
    ) -> None:
        """Test that errors from agent are wrapped in CalendarGenerationError."""
        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.side_effect = Exception("LLM failed")
            mock_get_agent.return_value = mock_agent

            with pytest.raises(CalendarGenerationError) as exc_info:
                calendar_service.generate_calendar(sample_brief)

        assert "Failed to generate calendar" in str(exc_info.value)
        assert "LLM failed" in str(exc_info.value)

    def test_generate_calendar_with_empty_subgenres_and_themes(
        self, calendar_service: CalendarService
    ) -> None:
        """Test calendar generation with empty optional lists."""
        brief = StoryBrief(
            premise="A simple story",
            genre="Drama",
            subgenres=[],
            tone="Serious",
            themes=[],
            setting_time="Modern",
            setting_place="City",
            target_length="short_story",
            language="English",
            content_rating="none",
            content_preferences=[],
            content_avoid=[],
        )

        mock_data = GeneratedCalendarData(
            era_name="Modern Era",
            era_abbreviation="ME",
            current_year=2024,
            months=[{"name": "January", "days": 31}],
            day_names=["Mon", "Tue", "Wed", "Thu", "Fri"],
            historical_eras=[{"name": "Modern Era", "start_year": 1900}],
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar(brief)

        assert result.current_era_name == "Modern Era"


class TestGenerateCalendarForGenre:
    """Tests for generate_calendar_for_genre method."""

    def test_generate_calendar_for_fantasy_genre(self, calendar_service: CalendarService) -> None:
        """Test generating calendar for Fantasy genre."""
        mock_data = GeneratedCalendarData(
            era_name="Age of Magic",
            era_abbreviation="AM",
            current_year=500,
            months=[{"name": "Moonrise", "days": 28}],
            day_names=["Starday", "Moonday"],
            historical_eras=[{"name": "Age of Magic", "start_year": 1}],
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar_for_genre("Fantasy")

        assert result.current_era_name == "Age of Magic"

    def test_generate_calendar_for_scifi_genre(self, calendar_service: CalendarService) -> None:
        """Test generating calendar for Sci-Fi genre."""
        mock_data = GeneratedCalendarData(
            era_name="Stellar Epoch",
            era_abbreviation="SE",
            current_year=3000,
            months=[{"name": "Cycle-1", "days": 30}],
            day_names=["Unit-1", "Unit-2"],
            historical_eras=[{"name": "Stellar Epoch", "start_year": 2500}],
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar_for_genre("Sci-Fi")

        assert result.current_era_name == "Stellar Epoch"

    def test_generate_calendar_for_genre_fallback_on_error(
        self, calendar_service: CalendarService
    ) -> None:
        """Test that genre calendar falls back to default on error."""
        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.side_effect = Exception("LLM failed")
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar_for_genre("Fantasy")

        # Should fall back to default calendar
        assert result.current_era_name == "Fantasy Era"
        assert result.era_abbreviation == "FA"
        assert result.current_story_year == 1000

    def test_generate_calendar_for_historical_genre(
        self, calendar_service: CalendarService
    ) -> None:
        """Test generating calendar for Historical genre (uses Medieval setting)."""
        mock_data = GeneratedCalendarData(
            era_name="Medieval Period",
            era_abbreviation="MP",
            current_year=1200,
            months=[{"name": "Harvest", "days": 30}],
            day_names=["Day1"],
            historical_eras=[{"name": "Medieval Period", "start_year": 500}],
        )

        with patch.object(calendar_service, "_get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate_structured.return_value = mock_data
            mock_get_agent.return_value = mock_agent

            result = calendar_service.generate_calendar_for_genre("Historical")

        assert result.current_era_name == "Medieval Period"


class TestGeneratedCalendarData:
    """Tests for GeneratedCalendarData model."""

    def test_create_with_all_fields(self) -> None:
        """Test creating model with all fields."""
        data = GeneratedCalendarData(
            era_name="Test Era",
            era_abbreviation="TE",
            current_year=100,
            months=[{"name": "Month1", "days": 30}],
            day_names=["Day1"],
            historical_eras=[{"name": "Era1", "start_year": 1}],
            date_format="{day}/{month}/{year}",
        )
        assert data.era_name == "Test Era"
        assert data.date_format == "{day}/{month}/{year}"

    def test_default_date_format(self) -> None:
        """Test that date_format has a sensible default."""
        data = GeneratedCalendarData(
            era_name="Test",
            era_abbreviation="T",
            current_year=1,
            months=[],
            day_names=[],
            historical_eras=[],
        )
        assert data.date_format == "{day} {month}, Year {year} {era}"
