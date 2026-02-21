"""Tests for calendar quality loop functions."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_quality import CalendarQualityScores, RefinementConfig
from src.services.calendar_service import GeneratedCalendarData
from src.services.world_quality_service import WorldQualityService
from src.services.world_quality_service._calendar import (
    _create_calendar,
    _generated_data_to_world_calendar,
    _judge_calendar_quality,
    _refine_calendar,
    generate_calendar_with_quality,
)
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def story_state():
    """Create a sample story state with brief."""
    state = StoryState(id="test-project")
    state.brief = StoryBrief(
        premise="A hero's journey",
        genre="fantasy",
        tone="epic",
        target_length="short_story",
        themes=["courage"],
        language="English",
        setting_time="Medieval era",
        setting_place="A fantasy kingdom",
        content_rating="none",
    )
    return state


@pytest.fixture
def mock_svc(settings):
    """Create a mock WorldQualityService."""
    svc = MagicMock()
    svc.settings = settings
    svc.get_config.return_value = RefinementConfig.from_settings(settings)
    svc.get_judge_config.return_value = MagicMock(
        enabled=False,
        multi_call_enabled=False,
    )
    svc._get_creator_model.return_value = "test-model:8b"
    svc._get_judge_model.return_value = "test-model:8b"
    svc._log_refinement_analytics = MagicMock()
    return svc


@pytest.fixture
def sample_calendar_dict():
    """Create a sample calendar dict as returned by WorldCalendar.to_dict()."""
    return {
        "id": "test-calendar-id",
        "current_era_name": "Age of Legends",
        "era_abbreviation": "AL",
        "era_start_year": 1,
        "months": [
            {"name": "Frostmoon", "days": 30, "description": "Cold month"},
            {"name": "Sunpeak", "days": 31, "description": "Warm month"},
        ],
        "days_per_week": 7,
        "day_names": ["Sunrest", "Moonrise", "Starsday"],
        "current_story_year": 342,
        "story_start_year": None,
        "story_end_year": None,
        "eras": [
            {
                "id": "era-1",
                "name": "Age of Legends",
                "start_year": 1,
                "end_year": None,
                "description": "The current era",
                "display_order": 0,
            }
        ],
        "date_format": "{day} {month}, Year {year} {era}",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


@pytest.fixture
def sample_generated_data():
    """Create a sample GeneratedCalendarData from LLM."""
    return GeneratedCalendarData(
        era_name="Age of Legends",
        era_abbreviation="AL",
        current_year=342,
        months=[
            {"name": "Frostmoon", "days": 30, "description": "Cold month"},
            {"name": "Sunpeak", "days": 31, "description": "Warm month"},
        ],
        day_names=["Sunrest", "Moonrise", "Starsday"],
        historical_eras=[
            {
                "name": "Age of Legends",
                "start_year": 1,
                "end_year": None,
                "description": "The current era",
            }
        ],
    )


class TestGenerateCalendarWithQuality:
    """Tests for the top-level generate_calendar_with_quality function."""

    def test_success_returns_dict_scores_iterations(self, mock_svc, story_state):
        """Test that successful generation returns (dict, scores, int)."""
        expected_scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=8.0,
            completeness=8.0,
            uniqueness=8.0,
            feedback="Great calendar.",
        )
        sample_data = GeneratedCalendarData(
            era_name="Test Era",
            era_abbreviation="TE",
            current_year=100,
            months=[{"name": "Month1", "days": 30, "description": "First"}],
            day_names=["Day1"],
            historical_eras=[
                {"name": "Test Era", "start_year": 1, "end_year": None, "description": "Current"}
            ],
        )

        with (
            patch("src.services.world_quality_service._calendar.generate_structured") as mock_gen,
        ):
            # First call creates, second call judges
            mock_gen.side_effect = [sample_data, expected_scores]
            result_dict, result_scores, iterations = generate_calendar_with_quality(
                mock_svc, story_state
            )

        assert isinstance(result_dict, dict)
        assert result_dict.get("current_era_name") == "Test Era"
        assert result_dict.get("months")  # Non-empty months list
        assert isinstance(result_scores, CalendarQualityScores)
        assert iterations >= 1

    def test_raises_without_brief(self, mock_svc):
        """Test ValueError when story has no brief."""
        state = StoryState(id="no-brief")
        with pytest.raises(ValueError, match="brief"):
            generate_calendar_with_quality(mock_svc, state)


class TestCreateCalendar:
    """Tests for _create_calendar function."""

    def test_returns_valid_dict(self, mock_svc, story_state, sample_generated_data):
        """Test create returns a dict with expected keys."""
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=sample_generated_data,
        ):
            result = _create_calendar(mock_svc, story_state, 0.9)

        assert isinstance(result, dict)
        assert result.get("current_era_name") == "Age of Legends"
        assert result.get("era_abbreviation") == "AL"
        assert len(result.get("months", [])) == 2
        assert result.get("current_story_year") == 342

    def test_raises_on_missing_brief(self, mock_svc):
        """Test WorldGenerationError when no brief available."""
        state = StoryState(id="no-brief")
        with pytest.raises(WorldGenerationError, match="no brief"):
            _create_calendar(mock_svc, state, 0.9)

    def test_raises_world_generation_error_on_failure(self, mock_svc, story_state):
        """Test WorldGenerationError is raised on LLM failure."""
        with (
            patch(
                "src.services.world_quality_service._calendar.generate_structured",
                side_effect=RuntimeError("LLM failed"),
            ),
            pytest.raises(WorldGenerationError, match="Calendar creation failed"),
        ):
            _create_calendar(mock_svc, story_state, 0.9)

    def test_defaults_missing_month_days_and_era_start_year(self, mock_svc, story_state):
        """Test fallback defaults for month without days and era without start_year."""
        data = GeneratedCalendarData(
            era_name="First Era",
            era_abbreviation="TE",
            current_year=100,
            months=[{"name": "Frostmoon", "description": "Cold"}],  # No "days" key
            day_names=["Day1"],
            historical_eras=[
                {"name": "First Era", "end_year": None, "description": "Ancient"},  # No start_year
            ],
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=data,
        ):
            result = _create_calendar(mock_svc, story_state, 0.9)

        # Months missing "days" get default 30
        assert result["months"][0]["days"] == 30
        assert result["months"][0]["name"] == "Frostmoon"
        # Eras missing "start_year" get default 1
        assert result["eras"][0]["start_year"] == 1
        assert result["eras"][0]["name"] == "First Era"


class TestJudgeCalendarQuality:
    """Tests for _judge_calendar_quality function."""

    def test_returns_scores(self, mock_svc, story_state, sample_calendar_dict):
        """Test judge returns CalendarQualityScores."""
        expected = CalendarQualityScores(
            internal_consistency=7.5,
            thematic_fit=8.0,
            completeness=7.0,
            uniqueness=6.5,
            feedback="Needs more uniqueness.",
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=expected,
        ):
            result = _judge_calendar_quality(mock_svc, sample_calendar_dict, story_state, 0.1)

        assert isinstance(result, CalendarQualityScores)
        assert result.internal_consistency == 7.5
        assert result.thematic_fit == 8.0
        assert result.feedback == "Needs more uniqueness."

    def test_raises_on_failure(self, mock_svc, story_state, sample_calendar_dict):
        """Test WorldGenerationError is raised on judge failure."""
        with (
            patch(
                "src.services.world_quality_service._calendar.generate_structured",
                side_effect=RuntimeError("Judge failed"),
            ),
            pytest.raises(WorldGenerationError, match="Calendar quality judgment failed"),
        ):
            _judge_calendar_quality(mock_svc, sample_calendar_dict, story_state, 0.1)


class TestRefineCalendar:
    """Tests for _refine_calendar function."""

    def test_preserves_era_name(self, mock_svc, story_state, sample_calendar_dict):
        """Test refinement preserves the original era name."""
        scores = CalendarQualityScores(
            internal_consistency=6.0,
            thematic_fit=6.0,
            completeness=6.0,
            uniqueness=5.0,
            feedback="Needs improvement.",
        )
        refined_data = GeneratedCalendarData(
            era_name="Different Era Name",  # LLM might change this
            era_abbreviation="AL",
            current_year=342,
            months=[
                {"name": "Deepwinter", "days": 30, "description": "Cold month"},
                {"name": "Highsun", "days": 31, "description": "Warm month"},
            ],
            day_names=["Sunrest", "Moonrise", "Starsday"],
            historical_eras=[
                {"name": "Age of Legends", "start_year": 1, "end_year": None, "description": "Era"}
            ],
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=refined_data,
        ):
            result = _refine_calendar(mock_svc, sample_calendar_dict, scores, story_state, 0.7)

        # Era name should be preserved from original, not the LLM's version
        assert result["current_era_name"] == "Age of Legends"

    def test_raises_on_failure(self, mock_svc, story_state, sample_calendar_dict):
        """Test WorldGenerationError is raised on refinement failure."""
        scores = CalendarQualityScores(
            internal_consistency=6.0,
            thematic_fit=6.0,
            completeness=6.0,
            uniqueness=5.0,
            feedback="Needs improvement.",
        )
        with (
            patch(
                "src.services.world_quality_service._calendar.generate_structured",
                side_effect=RuntimeError("Refine failed"),
            ),
            pytest.raises(WorldGenerationError, match="Calendar refinement failed"),
        ):
            _refine_calendar(mock_svc, sample_calendar_dict, scores, story_state, 0.7)

    def test_preserves_era_name_warns_when_missing(self, mock_svc, story_state, caplog):
        """Test refinement warns and falls back to 'Unknown' when original era name is missing."""
        scores = CalendarQualityScores(
            internal_consistency=6.0,
            thematic_fit=6.0,
            completeness=6.0,
            uniqueness=5.0,
            feedback="Needs improvement.",
        )
        # Calendar dict WITHOUT current_era_name
        calendar_no_era = {
            "era_abbreviation": "AL",
            "current_story_year": 342,
            "months": [{"name": "Frostmoon", "days": 30}],
            "eras": [{"name": "Age of Legends", "start_year": 1}],
        }
        refined_data = GeneratedCalendarData(
            era_name="Refined Era",
            era_abbreviation="RE",
            current_year=342,
            months=[{"name": "Deepwinter", "days": 30}],
            day_names=["Sunrest"],
            historical_eras=[{"name": "Age of Legends", "start_year": 1}],
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=refined_data,
        ):
            result = _refine_calendar(mock_svc, calendar_no_era, scores, story_state, 0.7)
        assert result["current_era_name"] == "Unknown"
        assert "Original calendar dict missing 'current_era_name'" in caplog.text


class TestGeneratedDataToWorldCalendar:
    """Tests for _generated_data_to_world_calendar helper."""

    def test_converts_to_world_calendar(self, sample_generated_data):
        """Test conversion from GeneratedCalendarData to WorldCalendar."""
        calendar = _generated_data_to_world_calendar(sample_generated_data)
        assert calendar.current_era_name == "Age of Legends"
        assert calendar.era_abbreviation == "AL"
        assert calendar.current_story_year == 342
        assert len(calendar.months) == 2
        assert calendar.months[0].name == "Frostmoon"
        assert len(calendar.eras) == 1

    def test_handles_fallback_names(self):
        """Test fallback names when LLM returns empty values."""
        data = GeneratedCalendarData(
            era_name="Test",
            era_abbreviation="T",
            current_year=100,
            months=[{"days": 30}],  # Missing name
            day_names=["Day1"],
            historical_eras=[{"start_year": 1}],  # Missing name
        )
        calendar = _generated_data_to_world_calendar(data)
        assert calendar.months[0].name == "Month 1"
        assert calendar.eras[0].name == "Era 1"

    def test_logs_warning_for_missing_days(self, caplog):
        """Test warning logged when month is missing 'days' field."""
        data = GeneratedCalendarData(
            era_name="Test",
            era_abbreviation="T",
            current_year=100,
            months=[{"name": "Frostmoon"}],  # Missing days
            day_names=["Day1"],
            historical_eras=[{"name": "Era", "start_year": 1}],
        )
        calendar = _generated_data_to_world_calendar(data)
        assert calendar.months[0].days == 30  # Default applied
        assert "missing 'days' field, defaulting to 30" in caplog.text

    def test_logs_warning_for_missing_start_year(self, caplog):
        """Test warning logged when era is missing 'start_year' field."""
        data = GeneratedCalendarData(
            era_name="Lost Age",
            era_abbreviation="T",
            current_year=100,
            months=[{"name": "Frostmoon", "days": 28}],
            day_names=["Day1"],
            historical_eras=[{"name": "Lost Age"}],  # Missing start_year
        )
        calendar = _generated_data_to_world_calendar(data)
        assert calendar.eras[0].start_year == 1  # Default applied
        assert "missing 'start_year' field, defaulting to 1" in caplog.text

    def test_empty_eras_uses_fallback_start_year(self):
        """Test that empty eras list results in fallback era_start_year=1."""
        data = GeneratedCalendarData(
            era_name="Lonely Era",
            era_abbreviation="LE",
            current_year=500,
            months=[{"name": "Onlymonth", "days": 30}],
            day_names=["Day1"],
            historical_eras=[],  # No eras
        )
        calendar = _generated_data_to_world_calendar(data)
        assert calendar.era_start_year == 1
        assert calendar.eras == []

    def test_era_lookup_case_insensitive(self):
        """Test that era lookup matches case-insensitively."""
        data = GeneratedCalendarData(
            era_name="dark age",  # Lowercase
            era_abbreviation="DA",
            current_year=500,
            months=[{"name": "Frostmoon", "days": 30}],
            day_names=["Day1"],
            historical_eras=[
                {"name": "Dark Age", "start_year": 100, "end_year": None, "description": ""},
            ],
        )
        calendar = _generated_data_to_world_calendar(data)
        assert calendar.era_start_year == 100

    def test_era_lookup_ongoing_fallback(self):
        """Test that ongoing era (end_year=None) is used when name doesn't match."""
        data = GeneratedCalendarData(
            era_name="Custom Era Name",  # Doesn't match any era
            era_abbreviation="CE",
            current_year=500,
            months=[{"name": "Frostmoon", "days": 30}],
            day_names=["Day1"],
            historical_eras=[
                {"name": "Era 1", "start_year": 1, "end_year": 99, "description": "Past"},
                {"name": "Era 2", "start_year": 100, "end_year": None, "description": "Current"},
            ],
        )
        calendar = _generated_data_to_world_calendar(data)
        # Should use the ongoing era (Era 2, start_year=100), not the last era
        assert calendar.era_start_year == 100

    def test_era_lookup_no_match_no_ongoing_falls_back_to_last(self):
        """Test fallback to last era when no match and no ongoing era."""
        data = GeneratedCalendarData(
            era_name="Nonexistent Era",
            era_abbreviation="NE",
            current_year=500,
            months=[{"name": "Frostmoon", "days": 30}],
            day_names=["Day1"],
            historical_eras=[
                {"name": "Era 1", "start_year": 1, "end_year": 99, "description": "Past"},
                {"name": "Era 2", "start_year": 100, "end_year": 299, "description": "Also past"},
            ],
        )
        calendar = _generated_data_to_world_calendar(data)
        # Falls back to last era's start_year
        assert calendar.era_start_year == 100


class TestCalendarContext:
    """Tests for calendar context methods on WorldQualityService."""

    def test_set_calendar_context_formats_era_and_year(self, sample_calendar_dict):
        """Test that set_calendar_context formats era name, abbreviation, and story year."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        # Call the real method
        WorldQualityService.set_calendar_context(svc, sample_calendar_dict)
        ctx = svc._calendar_context
        assert "Age of Legends" in ctx
        assert "AL" in ctx
        assert "342" in ctx

    def test_set_calendar_context_includes_months(self, sample_calendar_dict):
        """Test that set_calendar_context includes month names."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        WorldQualityService.set_calendar_context(svc, sample_calendar_dict)
        ctx = svc._calendar_context
        assert "Frostmoon" in ctx
        assert "Sunpeak" in ctx

    def test_set_calendar_context_includes_historical_eras(self, sample_calendar_dict):
        """Test that set_calendar_context includes historical era ranges."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        WorldQualityService.set_calendar_context(svc, sample_calendar_dict)
        ctx = svc._calendar_context
        assert "Age of Legends" in ctx
        assert "1" in ctx
        assert "present" in ctx

    def test_set_calendar_context_none_clears(self):
        """Test that set_calendar_context(None) clears the context."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = "some existing context"
        svc._calendar_context_lock = threading.RLock()
        WorldQualityService.set_calendar_context(svc, None)
        assert svc._calendar_context is None

    def test_get_calendar_context_returns_empty_when_none(self):
        """Test that get_calendar_context returns empty string when no context set."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        result = WorldQualityService.get_calendar_context(svc)
        assert result == ""

    def test_get_calendar_context_returns_formatted_block(self, sample_calendar_dict):
        """Test that get_calendar_context returns a formatted prompt block."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        WorldQualityService.set_calendar_context(svc, sample_calendar_dict)
        result = WorldQualityService.get_calendar_context(svc)
        assert "CALENDAR & TIMELINE:" in result
        assert "Age of Legends" in result

    def test_set_calendar_context_warns_on_empty_dict(self):
        """Test that set_calendar_context warns when dict produces no extractable context."""
        svc = MagicMock(spec=WorldQualityService)
        svc._calendar_context = None
        svc._calendar_context_lock = threading.RLock()
        # Empty dict with no usable fields
        WorldQualityService.set_calendar_context(svc, {})
        assert svc._calendar_context is None


class TestServiceWrapperMethods:
    """Tests that exercise the WorldQualityService wrapper methods for calendar."""

    def test_generate_calendar_with_quality_wrapper(self, settings, story_state):
        """Test the public generate_calendar_with_quality delegates correctly."""
        mode_service = MagicMock()
        svc = WorldQualityService(settings, mode_service)

        expected_scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=8.0,
            completeness=8.0,
            uniqueness=8.0,
            feedback="Good.",
        )
        sample_data = GeneratedCalendarData(
            era_name="Test",
            era_abbreviation="T",
            current_year=100,
            months=[{"name": "M1", "days": 30, "description": ""}],
            day_names=["D1"],
            historical_eras=[
                {"name": "Test", "start_year": 1, "end_year": None, "description": ""}
            ],
        )

        with patch("src.services.world_quality_service._calendar.generate_structured") as mock_gen:
            mock_gen.side_effect = [sample_data, expected_scores]
            result_dict, result_scores, _iterations = svc.generate_calendar_with_quality(
                story_state
            )

        assert isinstance(result_dict, dict)
        assert isinstance(result_scores, CalendarQualityScores)

    def test_create_calendar_wrapper(self, settings, story_state):
        """Test the _create_calendar private delegate on the service."""
        mode_service = MagicMock()
        svc = WorldQualityService(settings, mode_service)

        sample_data = GeneratedCalendarData(
            era_name="Test",
            era_abbreviation="T",
            current_year=100,
            months=[{"name": "M1", "days": 30, "description": ""}],
            day_names=["D1"],
            historical_eras=[
                {"name": "Test", "start_year": 1, "end_year": None, "description": ""}
            ],
        )

        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=sample_data,
        ):
            result = svc._create_calendar(story_state, 0.9)
        assert isinstance(result, dict)

    def test_judge_calendar_quality_wrapper(self, settings, story_state, sample_calendar_dict):
        """Test the _judge_calendar_quality private delegate on the service."""
        mode_service = MagicMock()
        svc = WorldQualityService(settings, mode_service)

        expected = CalendarQualityScores(
            internal_consistency=7.0,
            thematic_fit=8.0,
            completeness=7.0,
            uniqueness=6.0,
            feedback="OK.",
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=expected,
        ):
            result = svc._judge_calendar_quality(sample_calendar_dict, story_state, 0.1)
        assert isinstance(result, CalendarQualityScores)

    def test_refine_calendar_wrapper(self, settings, story_state, sample_calendar_dict):
        """Test the _refine_calendar private delegate on the service."""
        mode_service = MagicMock()
        svc = WorldQualityService(settings, mode_service)

        scores = CalendarQualityScores(
            internal_consistency=6.0,
            thematic_fit=6.0,
            completeness=6.0,
            uniqueness=5.0,
            feedback="Needs work.",
        )
        refined_data = GeneratedCalendarData(
            era_name="Refined",
            era_abbreviation="R",
            current_year=200,
            months=[{"name": "RM1", "days": 30, "description": ""}],
            day_names=["RD1"],
            historical_eras=[
                {"name": "Refined", "start_year": 1, "end_year": None, "description": ""}
            ],
        )
        with patch(
            "src.services.world_quality_service._calendar.generate_structured",
            return_value=refined_data,
        ):
            result = svc._refine_calendar(sample_calendar_dict, scores, story_state, 0.7)
        assert isinstance(result, dict)


class TestJudgeMultiCallBranch:
    """Tests for the multi-call judge failure warning branch."""

    def test_multi_call_judge_failure_raises_with_warning(
        self, story_state, sample_calendar_dict, settings
    ):
        """Test that judge failure in multi-call mode logs a warning and raises."""
        svc = MagicMock()
        svc.settings = settings
        svc._get_judge_model.return_value = "test-model:8b"
        # Enable multi-call judging
        svc.get_judge_config.return_value = MagicMock(
            enabled=True,
            multi_call_enabled=True,
            multi_call_count=3,
        )

        with (
            patch(
                "src.services.world_quality_service._calendar.generate_structured",
                side_effect=RuntimeError("LLM error"),
            ),
            pytest.raises(WorldGenerationError, match="Calendar quality judgment failed"),
        ):
            _judge_calendar_quality(svc, sample_calendar_dict, story_state, 0.1)
