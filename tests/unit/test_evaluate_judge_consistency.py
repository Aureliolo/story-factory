"""Tests for scripts/evaluate_judge_consistency.py temperature sweep functionality.

Tests cover:
- parse_temperatures() validation and edge cases
- run_judge_consistency_test() with pre-created entity data and temperature override
- print_temperature_sweep_summary() output formatting
- print_temperature_sweep_guidance() decision logic branches
- Backward compatibility (no --temperatures flag)
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate_judge_consistency import (
    compute_statistics,
    parse_temperatures,
    print_temperature_sweep_guidance,
    print_temperature_sweep_summary,
    run_judge_consistency_test,
)


class TestParseTemperatures:
    """Tests for the parse_temperatures() function."""

    def test_single_temperature(self):
        """Parse a single temperature value."""
        result = parse_temperatures("0.5")
        assert result == [0.5]

    def test_multiple_temperatures_sorted(self):
        """Multiple temperatures are returned sorted."""
        result = parse_temperatures("0.7,0.1,0.5,0.3")
        assert result == [0.1, 0.3, 0.5, 0.7]

    def test_temperatures_with_spaces(self):
        """Whitespace around values is stripped."""
        result = parse_temperatures(" 0.1 , 0.3 , 0.5 ")
        assert result == [0.1, 0.3, 0.5]

    def test_boundary_zero(self):
        """Temperature 0.0 is valid."""
        result = parse_temperatures("0.0")
        assert result == [0.0]

    def test_boundary_two(self):
        """Temperature 2.0 is valid."""
        result = parse_temperatures("2.0")
        assert result == [2.0]

    def test_out_of_range_high(self):
        """Temperature above 2.0 exits with error."""
        with pytest.raises(SystemExit):
            parse_temperatures("2.1")

    def test_out_of_range_negative(self):
        """Negative temperature exits with error."""
        with pytest.raises(SystemExit):
            parse_temperatures("-0.1")

    def test_invalid_non_numeric(self):
        """Non-numeric temperature exits with error."""
        with pytest.raises(SystemExit):
            parse_temperatures("abc")

    def test_mixed_valid_invalid(self):
        """If any value is invalid, exits with error."""
        with pytest.raises(SystemExit):
            parse_temperatures("0.1,bad,0.5")

    def test_duplicate_temperatures(self):
        """Duplicate temperatures are preserved (sorted)."""
        result = parse_temperatures("0.5,0.5,0.1")
        assert result == [0.1, 0.5, 0.5]

    def test_integer_temperatures(self):
        """Integer values parse as floats."""
        result = parse_temperatures("0,1,2")
        assert result == [0.0, 1.0, 2.0]


class TestRunJudgeConsistencyTestWithOverrides:
    """Tests for run_judge_consistency_test() with entity_data and temperature overrides."""

    def _make_mock_svc(self, judge_temp: float = 0.1):
        """Create a mock ServiceContainer with WorldQualityService."""
        svc = MagicMock()
        config = MagicMock()
        config.judge_temperature = judge_temp
        svc.world_quality.get_config.return_value = config
        return svc

    def _make_mock_scores(self, avg: float = 7.0):
        """Create a mock quality scores object."""
        scores = MagicMock()
        scores.average = avg
        scores.feedback = "Good entity"
        scores.to_dict.return_value = {
            "relevance": 7.0,
            "depth": 7.0,
            "consistency": 7.0,
            "average": avg,
            "feedback": "Good entity",
        }
        return scores

    def test_reuses_pre_created_entity(self):
        """When entity_data is provided, skips creation step."""
        svc = self._make_mock_svc()
        story_state = MagicMock()
        entity_data = {"name": "Test Faction", "description": "A test faction"}

        svc.world_quality._judge_faction_quality.return_value = self._make_mock_scores()

        result = run_judge_consistency_test(
            svc,
            story_state,
            "faction",
            judge_calls=2,
            entity_data=entity_data,
            entity_name="Test Faction",
        )

        assert result["entity_name"] == "Test Faction"
        assert result["entity_data"] == entity_data
        assert result["error"] is None
        assert len(result["individual_scores"]) == 2

    def test_temperature_override(self):
        """When temperature is provided, uses it instead of config."""
        svc = self._make_mock_svc(judge_temp=0.1)
        story_state = MagicMock()
        entity_data = {"name": "Test Faction", "description": "A test faction"}

        svc.world_quality._judge_faction_quality.return_value = self._make_mock_scores()

        result = run_judge_consistency_test(
            svc,
            story_state,
            "faction",
            judge_calls=1,
            temperature=0.5,
            entity_data=entity_data,
            entity_name="Test Faction",
        )

        assert result["judge_temperature"] == 0.5
        # Verify the judge was called with the override temperature
        svc.world_quality._judge_faction_quality.assert_called_once_with(
            entity_data, story_state, 0.5
        )

    def test_uses_config_temperature_when_no_override(self):
        """When temperature is None, falls back to config.judge_temperature."""
        svc = self._make_mock_svc(judge_temp=0.1)
        story_state = MagicMock()
        entity_data = {"name": "Test Faction", "description": "A test faction"}

        svc.world_quality._judge_faction_quality.return_value = self._make_mock_scores()

        result = run_judge_consistency_test(
            svc,
            story_state,
            "faction",
            judge_calls=1,
            entity_data=entity_data,
            entity_name="Test Faction",
        )

        assert result["judge_temperature"] == 0.1
        svc.world_quality._judge_faction_quality.assert_called_once_with(
            entity_data, story_state, 0.1
        )

    def test_creates_entity_when_no_entity_data(self):
        """When entity_data is None, creates entity via create_entity."""
        svc = self._make_mock_svc()
        story_state = MagicMock()

        svc.world_quality._judge_faction_quality.return_value = self._make_mock_scores()

        faction_data = {"name": "Created Faction", "description": "Created"}
        with patch(
            "scripts.evaluate_judge_consistency.create_entity",
            return_value=(faction_data, "Created Faction"),
        ):
            result = run_judge_consistency_test(
                svc,
                story_state,
                "faction",
                judge_calls=1,
            )

        assert result["entity_name"] == "Created Faction"
        assert result["entity_data"] == faction_data

    def test_temperature_zero_is_valid_override(self):
        """Temperature 0.0 is a valid override (not treated as falsy)."""
        svc = self._make_mock_svc(judge_temp=0.5)
        story_state = MagicMock()
        entity_data = {"name": "Test Faction", "description": "A test faction"}

        svc.world_quality._judge_faction_quality.return_value = self._make_mock_scores()

        result = run_judge_consistency_test(
            svc,
            story_state,
            "faction",
            judge_calls=1,
            temperature=0.0,
            entity_data=entity_data,
            entity_name="Test Faction",
        )

        assert result["judge_temperature"] == 0.0
        svc.world_quality._judge_faction_quality.assert_called_once_with(
            entity_data, story_state, 0.0
        )


class TestPrintTemperatureSweepSummary:
    """Tests for print_temperature_sweep_summary() output."""

    def test_prints_header_and_rows(self, capsys):
        """Summary table has header, separator, and one row per result."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"mean": 7.2, "std": 0.0, "min": 7.2, "max": 7.2},
                "feedback_similarity": 0.98,
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.5,
                "verdict": "borderline",
                "average_stats": {"mean": 7.1, "std": 0.25, "min": 6.6, "max": 7.6},
                "feedback_similarity": 0.74,
                "error": None,
            },
        ]

        print_temperature_sweep_summary(results)
        output = capsys.readouterr().out

        assert "TEMPERATURE SWEEP" in output
        assert "character" in output
        assert "CONSISTENT" in output
        assert "BORDERLINE" in output
        assert "0.1" in output
        assert "0.5" in output

    def test_error_rows_show_error_label(self, capsys):
        """Results with errors display ERROR instead of stats."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "",
                "average_stats": {},
                "feedback_similarity": 0.0,
                "error": "Creation failed",
            },
        ]

        print_temperature_sweep_summary(results)
        output = capsys.readouterr().out

        assert "ERROR" in output
        assert "character" in output

    def test_handles_empty_results(self, capsys):
        """Empty results list prints header only."""
        print_temperature_sweep_summary([])
        output = capsys.readouterr().out
        assert "TEMPERATURE SWEEP" in output


class TestPrintTemperatureSweepGuidance:
    """Tests for print_temperature_sweep_guidance() decision branches."""

    def test_all_low_variance_recommends_disable(self, capsys):
        """When all temps have std < 0.05, recommends disabling multi-call averaging."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.0},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.3,
                "verdict": "consistent",
                "average_stats": {"std": 0.02},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.5,
                "verdict": "consistent",
                "average_stats": {"std": 0.04},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1, 0.3, 0.5])
        output = capsys.readouterr().out

        assert "WASTEFUL" in output
        assert "Disable multi-call averaging" in output

    def test_all_noisy_recommends_lower_temp(self, capsys):
        """When all temps are noisy (std > 0.5), recommends lowering temperature."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.3,
                "verdict": "noisy",
                "average_stats": {"std": 0.6},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.5,
                "verdict": "noisy",
                "average_stats": {"std": 0.8},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.3, 0.5])
        output = capsys.readouterr().out

        assert "high variance" in output
        assert "Lower judge temperature" in output

    def test_variance_at_high_temp_recommends_remove(self, capsys):
        """When variance only appears at temp >= 0.7, recommends removing multi-call."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.0},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.3,
                "verdict": "consistent",
                "average_stats": {"std": 0.01},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.7,
                "verdict": "borderline",
                "average_stats": {"std": 0.3},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1, 0.3, 0.7])
        output = capsys.readouterr().out

        assert "Variance only appears at temp >= 0.7" in output
        assert "Remove multi-call averaging" in output

    def test_variance_at_moderate_temp_reports_sweet_spot(self, capsys):
        """When variance appears at moderate temp (< 0.7), reports sweet spot."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.0},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.3,
                "verdict": "borderline",
                "average_stats": {"std": 0.15},
                "error": None,
            },
            {
                "entity_type": "character",
                "judge_temperature": 0.5,
                "verdict": "noisy",
                "average_stats": {"std": 0.6},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1, 0.3, 0.5])
        output = capsys.readouterr().out

        assert "Meaningful variance appears at temp >= 0.3" in output
        assert "multi-call averaging adds value" in output

    def test_prints_per_temperature_breakdown(self, capsys):
        """Guidance always includes per-temperature std breakdown."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.0},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1])
        output = capsys.readouterr().out

        assert "Per-temperature average std:" in output
        assert "temp=0.1" in output

    def test_skips_insufficient_data_and_errored_results(self, capsys):
        """Results with insufficient_data/empty verdict/errors are excluded from std."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "insufficient_data",
                "average_stats": {"std": 999.0},
                "error": None,
            },
            {
                "entity_type": "faction",
                "judge_temperature": 0.1,
                "verdict": "",
                "average_stats": {},
                "error": "Creation failed",
            },
            {
                "entity_type": "relationship",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.01},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1])
        output = capsys.readouterr().out

        # Should only use relationship's 0.01 std, not 999.0 or 0 from failed results
        assert "WASTEFUL" in output

    def test_multiple_entity_types_averaged(self, capsys):
        """When multiple entity types exist per temp, their stds are averaged."""
        results = [
            {
                "entity_type": "character",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.02},
                "error": None,
            },
            {
                "entity_type": "faction",
                "judge_temperature": 0.1,
                "verdict": "consistent",
                "average_stats": {"std": 0.04},
                "error": None,
            },
        ]

        print_temperature_sweep_guidance(results, [0.1])
        output = capsys.readouterr().out

        # avg_std = (0.02 + 0.04) / 2 = 0.03 < 0.05, so WASTEFUL
        assert "WASTEFUL" in output


class TestComputeStatisticsEdgeCases:
    """Extra edge case tests for compute_statistics used by sweep."""

    def test_identical_values(self):
        """Identical values produce zero std."""
        result = compute_statistics([7.0, 7.0, 7.0, 7.0, 7.0])
        assert result["std"] == 0.0
        assert result["min"] == 7.0
        assert result["max"] == 7.0

    def test_varied_values(self):
        """Varied values produce non-zero std."""
        result = compute_statistics([5.0, 7.0, 9.0])
        assert result["std"] > 0
        assert result["min"] == 5.0
        assert result["max"] == 9.0
