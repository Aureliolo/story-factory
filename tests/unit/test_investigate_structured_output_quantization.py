"""Unit tests for scripts/investigate_structured_output_quantization.py.

Tests the score analysis, comparison, and LLM call functions without requiring
a running Ollama instance.
"""

import json
from unittest.mock import MagicMock

import ollama
import pytest

from scripts.investigate_structured_output_quantization import (
    SCORE_DIMS,
    analyze_scores,
    print_comparison,
    run_constrained,
    run_unconstrained,
)


# =====================================================================
# Helpers
# =====================================================================
def _make_result(
    round_num: int,
    coherence: float,
    influence: float,
    conflict_potential: float,
    distinctiveness: float,
    time_val: float = 1.0,
) -> dict:
    """Build a single result dict matching the structure produced by run_*."""
    return {
        "round": round_num,
        "coherence": coherence,
        "influence": influence,
        "conflict_potential": conflict_potential,
        "distinctiveness": distinctiveness,
        "time": time_val,
    }


def _make_chat_response(content: str) -> dict:
    """Build a dict mimicking an Ollama chat response."""
    return {"message": {"content": content}}


# =====================================================================
# analyze_scores tests
# =====================================================================
class TestAnalyzeScores:
    """Tests for the analyze_scores function."""

    def test_empty_input(self):
        """Empty results should return an error dict."""
        result = analyze_scores([], "test-label")
        assert result["label"] == "test-label"
        assert result["rounds"] == 0
        assert result["error"] == "no valid results"

    def test_single_round(self):
        """Single round should produce correct basic stats."""
        results = [_make_result(1, 7.0, 6.5, 8.0, 5.5)]
        analysis = analyze_scores(results, "Single")

        assert analysis["label"] == "Single"
        assert analysis["rounds"] == 1
        assert analysis["total_scores"] == 4  # 4 dimensions
        assert analysis["unique_values"] == 4  # 7.0, 6.5, 8.0, 5.5

    def test_multiple_rounds(self):
        """Multiple rounds should aggregate all scores correctly."""
        results = [
            _make_result(1, 7.0, 6.0, 8.0, 5.0),
            _make_result(2, 7.0, 6.0, 8.0, 5.0),
            _make_result(3, 7.5, 6.5, 8.5, 5.5),
        ]
        analysis = analyze_scores(results, "Multi")

        assert analysis["rounds"] == 3
        assert analysis["total_scores"] == 12  # 3 rounds * 4 dims

    def test_unique_value_counting(self):
        """Unique values should count distinct score values across all dimensions."""
        results = [
            _make_result(1, 7.0, 7.0, 7.0, 7.0),
            _make_result(2, 7.0, 7.0, 7.0, 7.0),
        ]
        analysis = analyze_scores(results, "AllSame")

        assert analysis["unique_values"] == 1  # All values are 7.0

    def test_unique_value_counting_varied(self):
        """Unique values should correctly count when scores vary."""
        results = [
            _make_result(1, 1.0, 2.0, 3.0, 4.0),
            _make_result(2, 5.0, 6.0, 7.0, 8.0),
        ]
        analysis = analyze_scores(results, "Varied")

        assert analysis["unique_values"] == 8  # 1-8, all different

    def test_decimal_precision_detection_integers(self):
        """Integer scores should report 0 decimal places."""
        results = [_make_result(1, 7.0, 6.0, 8.0, 5.0)]
        analysis = analyze_scores(results, "Integers")

        # 7.0 -> "7.0000000000" -> rstrip("0") -> "7." -> 0 chars after "."
        # Actually: "7.0000000000".rstrip("0") = "7." -> len("") = 0
        assert analysis["max_decimal_places"] == 0

    def test_decimal_precision_detection_floats(self):
        """Scores with decimal places should be detected."""
        results = [_make_result(1, 7.25, 6.125, 8.1, 5.0)]
        analysis = analyze_scores(results, "Floats")

        # 7.25 has 2 decimal places, 6.125 has 3, 8.1 has 1, 5.0 has 0
        assert analysis["max_decimal_places"] == 3

    def test_per_dimension_stats(self):
        """Per-dimension statistics should be computed for each SCORE_DIMS entry."""
        results = [
            _make_result(1, 7.0, 6.0, 8.0, 5.0),
            _make_result(2, 7.5, 6.5, 8.5, 5.5),
        ]
        analysis = analyze_scores(results, "PerDim")

        per_dim = analysis["per_dimension"]
        assert set(per_dim.keys()) == set(SCORE_DIMS)

        # coherence: [7.0, 7.5] -> mean=7.25
        assert per_dim["coherence"]["mean"] == 7.25
        assert per_dim["coherence"]["unique"] == 2
        assert per_dim["coherence"]["values"] == [7.0, 7.5]

        # influence: [6.0, 6.5] -> mean=6.25
        assert per_dim["influence"]["mean"] == 6.25

    def test_per_dimension_std_dev_single_value(self):
        """Single-round per-dimension std_dev should be 0.0."""
        results = [_make_result(1, 7.0, 6.0, 8.0, 5.0)]
        analysis = analyze_scores(results, "SingleStd")

        for dim in SCORE_DIMS:
            assert analysis["per_dimension"][dim]["std_dev"] == 0.0

    def test_mean_and_std_dev(self):
        """Mean and std_dev should be computed over all scores."""
        results = [_make_result(1, 5.0, 5.0, 5.0, 5.0)]
        analysis = analyze_scores(results, "Uniform")

        assert analysis["mean"] == 5.0
        assert analysis["std_dev"] == 0.0  # All same -> single value, but len > 1 check

    def test_min_max_range(self):
        """Min, max, and range should reflect score extremes."""
        results = [_make_result(1, 2.0, 4.0, 6.0, 8.0)]
        analysis = analyze_scores(results, "Range")

        assert analysis["min"] == 2.0
        assert analysis["max"] == 8.0
        assert analysis["range"] == 6.0

    def test_modal_scores(self):
        """Modal scores should list the most common values."""
        results = [
            _make_result(1, 7.0, 7.0, 7.0, 8.0),
            _make_result(2, 7.0, 7.0, 8.0, 8.0),
        ]
        analysis = analyze_scores(results, "Modal")

        # 7.0 appears 5 times, 8.0 appears 3 times
        modal = analysis["modal_scores"]
        assert modal[0] == (7.0, 5)
        assert modal[1] == (8.0, 3)

    def test_decimal_distribution(self):
        """Decimal distribution should count how many scores have each number of decimals."""
        results = [_make_result(1, 7.0, 6.5, 8.25, 5.125)]
        analysis = analyze_scores(results, "DecDist")

        dec_dist = analysis["decimal_distribution"]
        # 7.0 -> 0 decimals, 6.5 -> 1, 8.25 -> 2, 5.125 -> 3
        assert dec_dist[0] == 1
        assert dec_dist[1] == 1
        assert dec_dist[2] == 1
        assert dec_dist[3] == 1


# =====================================================================
# print_comparison tests
# =====================================================================
class TestPrintComparison:
    """Tests for the print_comparison verdict logic."""

    def test_quantization_detected(self, capsys):
        """Verdict should be QUANTIZATION DETECTED when constrained unique < 50% of unconstrained."""
        constrained = {
            "unique_values": 3,
            "std_dev": 0.5,
            "range": 1.0,
            "avg_decimal_places": 1.0,
            "max_decimal_places": 1,
            "mean": 7.0,
        }
        unconstrained = {
            "unique_values": 10,
            "std_dev": 1.5,
            "range": 4.0,
            "avg_decimal_places": 2.0,
            "max_decimal_places": 3,
            "mean": 7.2,
        }
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "QUANTIZATION DETECTED" in output

    def test_precision_loss(self, capsys):
        """Verdict should be PRECISION LOSS when constrained decimals are significantly lower."""
        # unique_values NOT below 50% threshold (8 >= 10 * 0.5)
        # but avg_decimal_places has > 0.5 gap
        constrained = {
            "unique_values": 8,
            "std_dev": 1.0,
            "range": 3.0,
            "avg_decimal_places": 0.5,
            "max_decimal_places": 1,
            "mean": 7.0,
        }
        unconstrained = {
            "unique_values": 10,
            "std_dev": 1.2,
            "range": 3.5,
            "avg_decimal_places": 2.0,
            "max_decimal_places": 3,
            "mean": 7.1,
        }
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "PRECISION LOSS" in output

    def test_no_significant_quantization(self, capsys):
        """Verdict should be 'No significant quantization' when values are similar."""
        constrained = {
            "unique_values": 9,
            "std_dev": 1.0,
            "range": 3.0,
            "avg_decimal_places": 1.8,
            "max_decimal_places": 3,
            "mean": 7.0,
        }
        unconstrained = {
            "unique_values": 10,
            "std_dev": 1.1,
            "range": 3.2,
            "avg_decimal_places": 2.0,
            "max_decimal_places": 3,
            "mean": 7.1,
        }
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "No significant quantization" in output

    def test_inconclusive(self, capsys):
        """Verdict should be INCONCLUSIVE when differences are minor but outside 'OK' thresholds."""
        # unique_values differ by > 2, but constrained is NOT < 50% of unconstrained
        # avg_decimal_places differ by > 0.3 but <= 0.5
        constrained = {
            "unique_values": 6,
            "std_dev": 0.8,
            "range": 2.5,
            "avg_decimal_places": 1.6,
            "max_decimal_places": 2,
            "mean": 7.0,
        }
        unconstrained = {
            "unique_values": 10,
            "std_dev": 1.2,
            "range": 3.5,
            "avg_decimal_places": 2.0,
            "max_decimal_places": 3,
            "mean": 7.2,
        }
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "INCONCLUSIVE" in output

    def test_error_in_constrained(self, capsys):
        """Should print error message when constrained has an error."""
        constrained = {"error": "no valid results"}
        unconstrained = {"unique_values": 10}
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "Cannot compare" in output

    def test_error_in_unconstrained(self, capsys):
        """Should print error message when unconstrained has an error."""
        constrained = {"unique_values": 10}
        unconstrained = {"error": "no valid results"}
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "Cannot compare" in output

    def test_comparison_table_printed(self, capsys):
        """The comparison should print a metric table with labels."""
        constrained = {
            "unique_values": 5,
            "std_dev": 1.0,
            "range": 3.0,
            "avg_decimal_places": 1.5,
            "max_decimal_places": 2,
            "mean": 7.0,
        }
        unconstrained = {
            "unique_values": 5,
            "std_dev": 1.0,
            "range": 3.0,
            "avg_decimal_places": 1.5,
            "max_decimal_places": 2,
            "mean": 7.0,
        }
        print_comparison(constrained, unconstrained)
        output = capsys.readouterr().out
        assert "Unique score values" in output
        assert "Score std dev" in output
        assert "Avg decimal places" in output
        assert "VERDICT" in output


# =====================================================================
# run_constrained tests
# =====================================================================
class TestRunConstrained:
    """Tests for run_constrained with mocked Ollama client."""

    @pytest.fixture()
    def mock_client(self):
        """Create a mock Ollama client."""
        return MagicMock(spec=ollama.Client)

    def test_valid_response_parsing(self, mock_client):
        """Valid JSON response should be parsed into a result dict."""
        content = json.dumps(
            {
                "coherence": 7.5,
                "influence": 6.0,
                "conflict_potential": 8.2,
                "distinctiveness": 5.8,
                "feedback": "Decent faction.",
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_constrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 1
        assert results[0]["round"] == 1
        assert results[0]["coherence"] == 7.5
        assert results[0]["influence"] == 6.0
        assert results[0]["conflict_potential"] == 8.2
        assert results[0]["distinctiveness"] == 5.8
        assert "time" in results[0]

    def test_multiple_rounds(self, mock_client):
        """Multiple rounds should accumulate results."""
        content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
                "feedback": "OK",
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_constrained(mock_client, "test-model:8b", rounds=3, timeout=30)

        assert len(results) == 3
        assert results[0]["round"] == 1
        assert results[1]["round"] == 2
        assert results[2]["round"] == 3

    def test_validation_error_handling(self, mock_client):
        """ValidationError should be caught and round skipped."""
        # Return invalid content that fails Pydantic validation (score > 10)
        invalid_content = json.dumps(
            {
                "coherence": 999.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
                "feedback": "Bad",
            }
        )
        mock_client.chat.return_value = _make_chat_response(invalid_content)

        results = run_constrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 0

    def test_response_error_handling(self, mock_client):
        """ollama.ResponseError should be caught and round skipped."""
        mock_client.chat.side_effect = ollama.ResponseError("model not found")

        results = run_constrained(mock_client, "test-model:8b", rounds=2, timeout=30)

        assert len(results) == 0

    def test_mixed_success_and_failure(self, mock_client):
        """Successful rounds should be kept even if some fail."""
        valid_content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
                "feedback": "Good",
            }
        )
        mock_client.chat.side_effect = [
            _make_chat_response(valid_content),
            ollama.ResponseError("timeout"),
            _make_chat_response(valid_content),
        ]

        results = run_constrained(mock_client, "test-model:8b", rounds=3, timeout=30)

        assert len(results) == 2
        assert results[0]["round"] == 1
        assert results[1]["round"] == 3

    def test_key_error_handling(self, mock_client):
        """KeyError (e.g., missing 'message' key) should be caught."""
        mock_client.chat.return_value = {"bad_key": "no message"}

        results = run_constrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 0

    def test_chat_called_with_correct_params(self, mock_client):
        """Verify chat is called with format=schema and correct options."""
        content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
                "feedback": "OK",
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        run_constrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["model"] == "test-model:8b"
        assert call_kwargs.kwargs["options"] == {"temperature": 0.1, "num_ctx": 4096}
        # format should be a JSON schema dict (from Pydantic)
        assert isinstance(call_kwargs.kwargs["format"], dict)


# =====================================================================
# run_unconstrained tests
# =====================================================================
class TestRunUnconstrained:
    """Tests for run_unconstrained with mocked Ollama client."""

    @pytest.fixture()
    def mock_client(self):
        """Create a mock Ollama client."""
        return MagicMock(spec=ollama.Client)

    def test_valid_json_parsing(self, mock_client):
        """Valid JSON with all dimensions should produce a result."""
        content = json.dumps(
            {
                "coherence": 7.5,
                "influence": 6.0,
                "conflict_potential": 8.2,
                "distinctiveness": 5.8,
                "feedback": "Decent faction.",
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 1
        assert results[0]["coherence"] == 7.5
        assert results[0]["influence"] == 6.0
        assert results[0]["conflict_potential"] == 8.2
        assert results[0]["distinctiveness"] == 5.8
        assert results[0]["round"] == 1
        assert "time" in results[0]

    def test_multiple_rounds(self, mock_client):
        """Multiple rounds should accumulate results."""
        content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=3, timeout=30)

        assert len(results) == 3

    def test_missing_dimensions_skipped(self, mock_client):
        """Response with missing dimensions should be skipped."""
        content = json.dumps(
            {
                "coherence": 7.5,
                "influence": 6.0,
                # missing conflict_potential and distinctiveness
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 0

    def test_json_decode_error_handling(self, mock_client):
        """Invalid JSON should be caught and round skipped."""
        mock_client.chat.return_value = _make_chat_response("not valid json {{{")

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 0

    def test_response_error_handling(self, mock_client):
        """ollama.ResponseError should be caught and round skipped."""
        mock_client.chat.side_effect = ollama.ResponseError("server error")

        results = run_unconstrained(mock_client, "test-model:8b", rounds=2, timeout=30)

        assert len(results) == 0

    def test_extra_keys_ignored(self, mock_client):
        """Extra keys in JSON response should not cause failures."""
        content = json.dumps(
            {
                "coherence": 7.5,
                "influence": 6.0,
                "conflict_potential": 8.2,
                "distinctiveness": 5.8,
                "feedback": "Extra stuff",
                "unknown_field": 42,
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 1
        assert results[0]["coherence"] == 7.5

    def test_string_values_converted_to_float(self, mock_client):
        """String numeric values should be converted to float."""
        content = json.dumps(
            {
                "coherence": "7.5",
                "influence": "6",
                "conflict_potential": "8.2",
                "distinctiveness": "5.8",
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 1
        assert results[0]["coherence"] == 7.5
        assert results[0]["influence"] == 6.0

    def test_mixed_success_and_failure(self, mock_client):
        """Successful rounds should be kept even if some fail."""
        valid_content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
            }
        )
        mock_client.chat.side_effect = [
            _make_chat_response(valid_content),
            _make_chat_response("not json"),
            _make_chat_response(valid_content),
        ]

        results = run_unconstrained(mock_client, "test-model:8b", rounds=3, timeout=30)

        assert len(results) == 2
        assert results[0]["round"] == 1
        assert results[1]["round"] == 3

    def test_chat_called_with_json_format(self, mock_client):
        """Verify chat is called with format='json'."""
        content = json.dumps(
            {
                "coherence": 7.0,
                "influence": 6.0,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["model"] == "test-model:8b"
        assert call_kwargs.kwargs["format"] == "json"

    def test_none_dimension_value_skipped(self, mock_client):
        """Dimension with None value should cause the round to be skipped."""
        content = json.dumps(
            {
                "coherence": 7.0,
                "influence": None,
                "conflict_potential": 8.0,
                "distinctiveness": 5.0,
            }
        )
        mock_client.chat.return_value = _make_chat_response(content)

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        # None is filtered by `if val is not None`, so only 3 dims found -> skipped
        assert len(results) == 0

    def test_key_error_handling(self, mock_client):
        """KeyError (e.g., missing 'message' key) should be caught."""
        mock_client.chat.return_value = {"bad_key": "no message"}

        results = run_unconstrained(mock_client, "test-model:8b", rounds=1, timeout=30)

        assert len(results) == 0
