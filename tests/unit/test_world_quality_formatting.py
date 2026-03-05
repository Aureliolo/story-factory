"""Tests for the world quality service formatting utilities."""

import logging
from typing import Any

from src.services.world_quality_service._formatting import (
    calculate_eta,
    check_name_completeness,
    format_existing_names_warning,
    format_properties,
    log_batch_summary,
)


class TestCalculateEta:
    """Tests for the ETA calculation function."""

    def test_returns_none_for_empty_times(self):
        """Test that None is returned when no completed times and no initial estimate."""
        assert calculate_eta([], 5) is None

    def test_returns_none_for_zero_remaining(self):
        """Test that None is returned when remaining count is zero."""
        assert calculate_eta([10.0, 20.0], 0) is None

    def test_returns_none_for_negative_remaining(self):
        """Test that None is returned when remaining count is negative."""
        assert calculate_eta([10.0, 20.0], -1) is None

    def test_single_time_returns_simple_multiple(self):
        """Test ETA calculation with a single completed time."""
        # With one time, ETA = time * remaining
        result = calculate_eta([10.0], 5)
        assert result == 50.0

    def test_multiple_times_uses_ema(self):
        """Test that EMA weighting is applied for multiple times."""
        # EMA with alpha=0.3 should weight recent times more heavily
        result = calculate_eta([10.0, 20.0, 30.0], 2)
        # EMA: start=10, then 0.3*20 + 0.7*10 = 13, then 0.3*30 + 0.7*13 = 18.1
        # Result: 18.1 * 2 = 36.2
        assert result is not None
        assert 36.0 < result < 37.0


class TestFormatProperties:
    """Tests for the property formatting function."""

    def test_returns_empty_string_for_none(self):
        """Test that empty string is returned for None input."""
        assert format_properties(None) == ""

    def test_returns_empty_string_for_empty_list(self):
        """Test that empty string is returned for empty list."""
        assert format_properties([]) == ""

    def test_formats_string_list(self):
        """Test formatting a list of strings."""
        result = format_properties(["brave", "wise", "kind"])
        assert result == "brave, wise, kind"

    def test_handles_single_string_not_in_list(self):
        """Test handling a single string (not in a list)."""
        result = format_properties("brave")
        assert result == "brave"

    def test_extracts_name_from_dict(self):
        """Test extracting name from dict properties."""
        result = format_properties([{"name": "Courage"}, {"name": "Wisdom"}])
        assert result == "Courage, Wisdom"

    def test_extracts_description_from_dict_without_name(self):
        """Test extracting description when no name present."""
        result = format_properties([{"description": "A brave trait"}])
        assert result == "A brave trait"

    def test_handles_dict_without_name_or_description(self):
        """Test handling dict without name or description."""
        result = format_properties([{"other": "value"}])
        assert "{'other': 'value'}" in result

    def test_handles_mixed_types(self):
        """Test handling mixed string and dict types."""
        result = format_properties(["brave", {"name": "wise"}, {"description": "kind"}])
        assert result == "brave, wise, kind"

    def test_handles_none_values_in_dict(self):
        """Test handling None values in dict name/description."""
        result = format_properties([{"name": None}])
        assert result == ""

    def test_handles_mixed_none_and_valid_values(self):
        """Test handling mix of None and valid values without leading comma."""
        result = format_properties([{"name": None}, {"name": "Test"}])
        assert result == "Test"

    def test_handles_non_string_values_in_dict(self):
        """Test handling non-string values like integers."""
        result = format_properties([{"name": 42}])
        assert result == "42"


class TestFormatExistingNamesWarning:
    """Tests for the existing names warning formatter."""

    def test_empty_names_returns_first_entity_message(self):
        """Test that empty names list returns a 'first entity' message."""
        result = format_existing_names_warning([], "character")
        assert "None yet" in result
        assert "creating the first character" in result

    def test_formats_single_name(self):
        """Test formatting a single existing name."""
        result = format_existing_names_warning(["Alice"], "character")
        assert "Alice" in result
        assert "DO NOT DUPLICATE" in result

    def test_formats_multiple_names(self):
        """Test formatting multiple existing names."""
        result = format_existing_names_warning(["Alice", "Bob", "Charlie"], "location")
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result
        assert "LOCATION" in result  # Entity type is uppercased

    def test_includes_do_not_examples(self):
        """Test that DO NOT examples are included."""
        result = format_existing_names_warning(["Castle"], "location")
        # Should include variations of the first name
        assert '"Castle" (exact match)' in result
        assert '"CASTLE" (case variation)' in result
        assert '"The Castle" (prefix variation)' in result

    def test_different_entity_types(self):
        """Test formatting for different entity types."""
        for entity_type in ["faction", "item", "concept"]:
            result = format_existing_names_warning(["Test"], entity_type)
            assert entity_type.upper() in result

    def test_xml_tags_wrap_content(self):
        """Test that XML tags are added for prompt injection protection."""
        result = format_existing_names_warning(["Alice"], "character")
        assert result.startswith("<existing-characters>")
        assert result.endswith("</existing-characters>")


class TestCalculateEtaInitialEstimate:
    """Tests for calculate_eta with initial_estimate_seconds parameter."""

    def test_initial_estimate_used_when_no_completed_times(self):
        """Initial estimate provides fallback when completed_times is empty."""
        result = calculate_eta([], 5, initial_estimate_seconds=10.0)
        assert result == 50.0

    def test_initial_estimate_ignored_when_completed_times_present(self):
        """Initial estimate is ignored when actual data is available."""
        result = calculate_eta([10.0], 5, initial_estimate_seconds=100.0)
        assert result == 50.0

    def test_initial_estimate_none_falls_back_to_none(self):
        """None initial_estimate still returns None when no data."""
        assert calculate_eta([], 5, initial_estimate_seconds=None) is None

    def test_initial_estimate_with_zero_remaining(self):
        """Zero remaining returns None even with initial estimate."""
        assert calculate_eta([], 0, initial_estimate_seconds=10.0) is None


class TestCheckNameCompleteness:
    """Tests for check_name_completeness."""

    def test_complete_name_returns_true(self):
        """Normal complete names are detected as complete."""
        assert check_name_completeness("Gandalf the Grey") is True

    def test_truncated_name_returns_false(self):
        """Name ending in a short non-word is flagged as truncated."""
        assert check_name_completeness("The Kingdom of Ar") is False

    def test_valid_short_ending_returns_true(self):
        """Names ending in common short words are not flagged."""
        assert check_name_completeness("City of the") is True

    def test_empty_name_returns_true(self):
        """Empty or very short names are too short to judge."""
        assert check_name_completeness("") is True
        assert check_name_completeness("Hi") is True

    def test_single_word_short_returns_true(self):
        """Single-word names are not flagged even if short."""
        assert check_name_completeness("Al") is True

    def test_name_with_trailing_punctuation(self):
        """Trailing punctuation is stripped before checking."""
        assert check_name_completeness("The Great Wall.") is True

    def test_whitespace_only_returns_true(self):
        """Whitespace-only name stripped to empty words returns True."""
        assert check_name_completeness("   ") is True


# ---------------------------------------------------------------------------
# M6: log_batch_summary with requested_count parameter
# ---------------------------------------------------------------------------


class _FakeScores:
    """Minimal mock scores for log_batch_summary tests."""

    def __init__(self, average: float) -> None:
        """Create fake scores with the given average."""
        self.average = average


class TestLogBatchSummary:
    """Tests for log_batch_summary including the requested_count parameter."""

    def test_empty_results(self, caplog):
        """Empty results logs 0 entities produced."""
        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._formatting"):
            log_batch_summary([], "character", 7.5, 10.0)
        assert any("0 entities produced" in r.message for r in caplog.records)

    def test_summary_without_requested_count(self, caplog):
        """Summary without requested_count shows passed/total."""
        results: list[tuple[Any, Any]] = [
            ({"name": "A"}, _FakeScores(8.0)),
            ({"name": "B"}, _FakeScores(6.0)),
        ]
        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._formatting"):
            log_batch_summary(results, "character", 7.5, 5.0)
        summary_msgs = [r.message for r in caplog.records if "Batch character" in r.message]
        assert len(summary_msgs) == 1
        assert "passed=1/2" in summary_msgs[0]
        assert "requested=" not in summary_msgs[0]

    def test_summary_with_requested_count_different_from_total(self, caplog):
        """When requested_count differs from total, it is included in summary."""
        results: list[tuple[Any, Any]] = [
            ({"name": "A"}, _FakeScores(8.0)),
        ]
        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._formatting"):
            log_batch_summary(results, "character", 7.5, 5.0, requested_count=3)
        summary_msgs = [r.message for r in caplog.records if "Batch character" in r.message]
        assert len(summary_msgs) == 1
        assert "passed=1/1" in summary_msgs[0]
        assert "requested=3" in summary_msgs[0]

    def test_summary_with_requested_count_equal_to_total(self, caplog):
        """When requested_count equals total, it is not redundantly shown."""
        results: list[tuple[Any, Any]] = [
            ({"name": "A"}, _FakeScores(8.0)),
            ({"name": "B"}, _FakeScores(9.0)),
        ]
        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._formatting"):
            log_batch_summary(results, "character", 7.5, 5.0, requested_count=2)
        summary_msgs = [r.message for r in caplog.records if "Batch character" in r.message]
        assert len(summary_msgs) == 1
        assert "passed=2/2" in summary_msgs[0]
        assert "requested=" not in summary_msgs[0]

    def test_summary_with_get_name_callable(self, caplog):
        """get_name callable is used for failed entity names."""
        results: list[tuple[Any, Any]] = [
            ({"label": "FailEntity"}, _FakeScores(5.0)),
        ]
        with caplog.at_level(
            logging.WARNING, logger="src.services.world_quality_service._formatting"
        ):
            log_batch_summary(
                results,
                "character",
                7.5,
                5.0,
                get_name=lambda e: e.get("label", "Unknown"),
            )
        summary_msgs = [r.message for r in caplog.records if "Batch character" in r.message]
        assert len(summary_msgs) == 1
        assert "FailEntity" in summary_msgs[0]
