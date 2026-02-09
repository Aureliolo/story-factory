"""Unit tests for scripts/investigate_score_distribution.py.

Tests the log parsing and analysis functions without requiring actual log files
or a running Ollama instance.
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.investigate_score_distribution import (
    EntityRecord,
    _parse_dimension_dict,
    compute_clustering_metrics,
    compute_dimension_variance,
    compute_per_type_breakdown,
    compute_score_histogram,
    compute_time_vs_score,
    parse_log_file,
)


# =====================================================================
# EntityRecord tests
# =====================================================================
class TestEntityRecord:
    """Tests for the EntityRecord data class."""

    def test_best_score_returns_max(self):
        """best_score should return the highest score."""
        record = EntityRecord("faction", "Test Faction")
        record.scores = [6.5, 7.2, 7.8, 7.0]
        assert record.best_score == 7.8

    def test_best_score_empty(self):
        """best_score should return None when no scores exist."""
        record = EntityRecord("faction", "Test Faction")
        assert record.best_score is None

    def test_total_generation_time_creation_only(self):
        """Total time with only creation time."""
        record = EntityRecord("faction", "Test Faction")
        record.creation_time = 3.5
        assert record.total_generation_time == 3.5

    def test_total_generation_time_with_refinements(self):
        """Total time includes creation + all refinement times."""
        record = EntityRecord("faction", "Test Faction")
        record.creation_time = 2.0
        record.refinement_times = [1.5, 2.3]
        assert record.total_generation_time == pytest.approx(5.8)

    def test_total_generation_time_no_creation(self):
        """Total time with refinements but no creation time."""
        record = EntityRecord("faction", "Test Faction")
        record.refinement_times = [1.5, 2.3]
        assert record.total_generation_time == pytest.approx(3.8)

    def test_entity_type_lowercase(self):
        """Entity type should be stored lowercase."""
        record = EntityRecord("Faction", "Test")
        assert record.entity_type == "faction"


# =====================================================================
# _parse_dimension_dict tests
# =====================================================================
class TestParseDimensionDict:
    """Tests for parsing dimension score dicts from log output."""

    def test_standard_format(self):
        """Parse standard single-quoted Python dict format."""
        result = _parse_dimension_dict("{'coherence': '8.5', 'influence': '7.0'}")
        assert result == {"coherence": 8.5, "influence": 7.0}

    def test_decimal_values(self):
        """Parse decimal values correctly."""
        result = _parse_dimension_dict("{'depth': '6.3', 'uniqueness': '9.1'}")
        assert result == {"depth": 6.3, "uniqueness": 9.1}

    def test_single_dimension(self):
        """Parse a single dimension."""
        result = _parse_dimension_dict("{'atmosphere': '7.2'}")
        assert result == {"atmosphere": 7.2}

    def test_invalid_json_raises(self):
        """Invalid format should raise an error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_dimension_dict("not a dict")


# =====================================================================
# parse_log_file tests
# =====================================================================
class TestParseLogFile:
    """Tests for parsing log files."""

    def _write_log(self, lines: list[str]) -> Path:
        """Write lines to a temporary log file and return its path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8")
        for line in lines:
            tmp.write(line + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_parse_creation_and_score(self):
        """Parse creation time followed by iteration score."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Faction creation took 2.34s (iteration 1)",
                "2024-01-01 INFO Faction 'The Guild' iteration 1: score 7.5 "
                "(best so far: 7.5 at iteration 1)",
            ]
        )
        records = parse_log_file(log_path)
        assert len(records) == 1
        assert records[0].entity_type == "faction"
        assert records[0].name == "The Guild"
        assert records[0].scores == [7.5]
        assert records[0].creation_time == pytest.approx(2.34)
        log_path.unlink()

    def test_parse_refinement_time(self):
        """Parse refinement times correctly."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Faction creation took 2.00s (iteration 1)",
                "2024-01-01 INFO Faction 'The Guild' iteration 1: score 6.5 "
                "(best so far: 6.5 at iteration 1)",
                "2024-01-01 INFO Faction refinement took 3.12s (iteration 2)",
                "2024-01-01 INFO Faction 'The Guild' iteration 2: score 7.8 "
                "(best so far: 7.8 at iteration 2)",
            ]
        )
        records = parse_log_file(log_path)
        assert len(records) == 1
        assert records[0].refinement_times == [pytest.approx(3.12)]
        assert records[0].scores == [6.5, 7.8]
        log_path.unlink()

    def test_parse_threshold_met(self):
        """Parse threshold met line."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Faction 'The Guild' iteration 1: score 7.5 "
                "(best so far: 7.5 at iteration 1)",
                "2024-01-01 INFO Faction 'The Guild' met quality threshold (7.5 >= 7.5)",
            ]
        )
        records = parse_log_file(log_path)
        assert len(records) == 1
        assert records[0].threshold_met is True
        assert records[0].final_score == 7.5
        assert records[0].threshold_value == 7.5
        log_path.unlink()

    def test_parse_dimension_scores(self):
        """Parse dimension score lines."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Faction 'The Guild' iteration 1: score 7.5 "
                "(best so far: 7.5 at iteration 1)",
                "2024-01-01 DEBUG Faction 'The Guild' iteration 1 dimension scores: "
                "{'coherence': '8.0', 'influence': '7.0'}",
            ]
        )
        records = parse_log_file(log_path)
        assert len(records) == 1
        assert len(records[0].dimension_scores) == 1
        assert records[0].dimension_scores[0] == {"coherence": 8.0, "influence": 7.0}
        log_path.unlink()

    def test_parse_multiple_entities(self):
        """Parse multiple different entities."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Faction 'Guild A' iteration 1: score 7.5 "
                "(best so far: 7.5 at iteration 1)",
                "2024-01-01 INFO Location 'Castle' iteration 1: score 8.0 "
                "(best so far: 8.0 at iteration 1)",
            ]
        )
        records = parse_log_file(log_path)
        assert len(records) == 2
        types = {r.entity_type for r in records}
        assert types == {"faction", "location"}
        log_path.unlink()

    def test_parse_empty_log(self):
        """Empty log file should return no records."""
        log_path = self._write_log([])
        records = parse_log_file(log_path)
        assert records == []
        log_path.unlink()

    def test_parse_irrelevant_lines(self):
        """Lines that don't match patterns should be ignored."""
        log_path = self._write_log(
            [
                "2024-01-01 INFO Starting application...",
                "2024-01-01 DEBUG Some random debug message",
                "2024-01-01 WARNING Connection timeout",
            ]
        )
        records = parse_log_file(log_path)
        assert records == []
        log_path.unlink()


# =====================================================================
# compute_score_histogram tests
# =====================================================================
class TestComputeScoreHistogram:
    """Tests for score histogram computation."""

    def test_single_score(self):
        """Single score should produce one bin."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5]
        hist = compute_score_histogram([r])
        assert "7.5-8.0" in hist
        assert hist["7.5-8.0"] == 1

    def test_multiple_scores_same_bin(self):
        """Multiple scores in same bin should be counted."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5, 7.6, 7.8]
        hist = compute_score_histogram([r])
        assert hist["7.5-8.0"] == 3

    def test_scores_across_bins(self):
        """Scores in different bins should be separated."""
        r = EntityRecord("faction", "A")
        r.scores = [6.0, 7.5, 9.0]
        hist = compute_score_histogram([r])
        assert hist["6.0-6.5"] == 1
        assert hist["7.5-8.0"] == 1
        assert hist["9.0-9.5"] == 1

    def test_empty_records(self):
        """No records should produce empty histogram."""
        hist = compute_score_histogram([])
        assert hist == {}

    def test_histogram_is_sorted(self):
        """Histogram bins should be sorted by lower bound."""
        r = EntityRecord("faction", "A")
        r.scores = [9.0, 3.0, 6.0]
        hist = compute_score_histogram([r])
        keys = list(hist.keys())
        lower_bounds = [float(k.split("-")[0]) for k in keys]
        assert lower_bounds == sorted(lower_bounds)


# =====================================================================
# compute_clustering_metrics tests
# =====================================================================
class TestComputeClusteringMetrics:
    """Tests for score clustering analysis."""

    def test_empty_records(self):
        """Empty records should return zero metrics."""
        metrics = compute_clustering_metrics([])
        assert metrics["unique_scores"] == 0
        assert metrics["total_scores"] == 0
        assert metrics["score_range"] == 0.0

    def test_single_score(self):
        """Single score — range and std_dev should be 0."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5]
        metrics = compute_clustering_metrics([r])
        assert metrics["unique_scores"] == 1
        assert metrics["total_scores"] == 1
        assert metrics["score_range"] == 0.0
        assert metrics["std_dev"] == 0.0

    def test_varied_scores(self):
        """Varied scores should produce non-zero spread metrics."""
        r = EntityRecord("faction", "A")
        r.scores = [3.0, 5.0, 7.0, 9.0]
        metrics = compute_clustering_metrics([r])
        assert metrics["unique_scores"] == 4
        assert metrics["total_scores"] == 4
        assert metrics["score_range"] == 6.0
        assert metrics["std_dev"] > 2.0

    def test_clustered_scores(self):
        """Tightly clustered scores should have low std_dev."""
        r = EntityRecord("faction", "A")
        r.scores = [7.4, 7.5, 7.5, 7.6, 7.5]
        metrics = compute_clustering_metrics([r])
        assert metrics["std_dev"] < 0.1
        assert metrics["iqr"] < 0.5

    def test_modal_scores_ordered(self):
        """Modal scores should be ordered by frequency (highest first)."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5, 7.5, 7.5, 8.0, 8.0, 6.0]
        metrics = compute_clustering_metrics([r])
        assert metrics["modal_scores"][0]["score"] == 7.5
        assert metrics["modal_scores"][0]["count"] == 3


# =====================================================================
# compute_per_type_breakdown tests
# =====================================================================
class TestComputePerTypeBreakdown:
    """Tests for per-entity-type breakdown."""

    def test_single_type(self):
        """Single entity type should produce one entry."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5]
        r.creation_time = 2.0
        result = compute_per_type_breakdown([r])
        assert "faction" in result
        assert result["faction"]["entity_count"] == 1
        assert result["faction"]["mean_final_score"] == 7.5

    def test_multiple_types(self):
        """Multiple types should produce separate entries."""
        r1 = EntityRecord("faction", "A")
        r1.scores = [7.5]
        r2 = EntityRecord("location", "B")
        r2.scores = [8.0]
        result = compute_per_type_breakdown([r1, r2])
        assert "faction" in result
        assert "location" in result

    def test_threshold_met_rate(self):
        """Threshold met rate should be correct."""
        r1 = EntityRecord("faction", "A")
        r1.scores = [7.5]
        r1.threshold_met = True
        r2 = EntityRecord("faction", "B")
        r2.scores = [6.0]
        r2.threshold_met = False
        result = compute_per_type_breakdown([r1, r2])
        assert result["faction"]["threshold_met_rate"] == 0.5

    def test_creation_time_averaging(self):
        """Average creation time should be computed correctly."""
        r1 = EntityRecord("faction", "A")
        r1.scores = [7.5]
        r1.creation_time = 2.0
        r2 = EntityRecord("faction", "B")
        r2.scores = [7.0]
        r2.creation_time = 4.0
        result = compute_per_type_breakdown([r1, r2])
        assert result["faction"]["avg_creation_time"] == 3.0


# =====================================================================
# compute_dimension_variance tests
# =====================================================================
class TestComputeDimensionVariance:
    """Tests for per-dimension variance analysis."""

    def test_empty_records(self):
        """No records should return empty result."""
        result = compute_dimension_variance([])
        assert result == {}

    def test_single_dimension_scores(self):
        """Single entity with dimension scores should compute stats."""
        r = EntityRecord("faction", "A")
        r.dimension_scores = [{"coherence": 8.0, "influence": 6.0}]
        result = compute_dimension_variance([r])
        assert "faction" in result
        assert "coherence" in result["faction"]["dimensions"]
        assert result["faction"]["dimensions"]["coherence"]["mean"] == 8.0

    def test_intra_entity_spread(self):
        """Intra-entity spread measures range within one judge call."""
        r = EntityRecord("faction", "A")
        r.dimension_scores = [{"coherence": 9.0, "influence": 5.0}]
        result = compute_dimension_variance([r])
        # 9.0 - 5.0 = 4.0 spread
        assert result["faction"]["avg_intra_entity_spread"] == 4.0

    def test_multiple_entries_averaged(self):
        """Multiple dimension score entries should be averaged."""
        r = EntityRecord("faction", "A")
        r.dimension_scores = [
            {"coherence": 8.0, "influence": 6.0},
            {"coherence": 9.0, "influence": 7.0},
        ]
        result = compute_dimension_variance([r])
        assert result["faction"]["dimensions"]["coherence"]["mean"] == 8.5
        assert result["faction"]["dimensions"]["influence"]["mean"] == 6.5


# =====================================================================
# compute_time_vs_score tests
# =====================================================================
class TestComputeTimeVsScore:
    """Tests for time vs score correlation."""

    def test_empty_records(self):
        """No records should return zero correlation."""
        result = compute_time_vs_score([])
        assert result["overall_correlation"] == 0.0
        assert result["total_pairs"] == 0

    def test_no_timing_data(self):
        """Records without timing data should be skipped."""
        r = EntityRecord("faction", "A")
        r.scores = [7.5]
        result = compute_time_vs_score([r])
        assert result["total_pairs"] == 0

    def test_positive_correlation(self):
        """Longer generation times with higher scores should show positive r."""
        records = []
        for time_val, score_val in [(1.0, 5.0), (2.0, 6.0), (3.0, 7.0), (4.0, 8.0)]:
            r = EntityRecord("faction", f"F{time_val}")
            r.scores = [score_val]
            r.creation_time = time_val
            records.append(r)
        result = compute_time_vs_score(records)
        assert result["overall_correlation"] > 0.9

    def test_no_correlation(self):
        """Unrelated times and scores should show low correlation."""
        records = []
        # Deliberately uncorrelated: no monotonic relationship between time and score
        for time_val, score_val in [
            (1.0, 7.0),
            (2.0, 5.0),
            (3.0, 8.0),
            (4.0, 6.0),
            (5.0, 7.5),
        ]:
            r = EntityRecord("faction", f"F{time_val}")
            r.scores = [score_val]
            r.creation_time = time_val
            records.append(r)
        result = compute_time_vs_score(records)
        assert abs(result["overall_correlation"]) < 0.5

    def test_per_type_breakdown(self):
        """Per-type stats should be computed separately."""
        r1 = EntityRecord("faction", "A")
        r1.scores = [7.5]
        r1.creation_time = 2.0
        r2 = EntityRecord("location", "B")
        r2.scores = [8.0]
        r2.creation_time = 12.0
        result = compute_time_vs_score([r1, r2])
        assert "faction" in result["per_type"]
        assert "location" in result["per_type"]
        assert result["per_type"]["faction"]["avg_time"] == 2.0
        assert result["per_type"]["location"]["avg_time"] == 12.0
