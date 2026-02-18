"""Tests for world quality models."""

import pytest

from src.memory.world_quality import (
    ChapterQualityScores,
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    JudgeConsistencyConfig,
    LocationQualityScores,
    PlotQualityScores,
    RefinementHistory,
    ScoreStatistics,
)


class TestRefinementHistory:
    """Tests for RefinementHistory model."""

    def test_get_best_entity_returns_none_when_best_iteration_is_zero(self):
        """Test get_best_entity returns None when best_iteration is 0."""
        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=0,  # No best iteration set
        )
        result = history.get_best_entity()
        assert result is None

    def test_get_best_entity_returns_none_when_no_iterations(self):
        """Test get_best_entity returns None with empty iterations."""
        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=1,  # Set to 1 but no iterations
            iterations=[],
        )
        result = history.get_best_entity()
        assert result is None

    def test_get_best_entity_returns_none_when_iteration_not_found(self):
        """Test get_best_entity returns None when best_iteration not in list."""
        from src.memory.world_quality import IterationRecord

        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=5,  # Points to iteration 5
            iterations=[
                IterationRecord(
                    iteration=1,  # Only iteration 1 exists
                    entity_data={"name": "Test"},
                    scores={"depth": 5.0},
                    average_score=5.0,
                ),
                IterationRecord(
                    iteration=2,
                    entity_data={"name": "Test2"},
                    scores={"depth": 6.0},
                    average_score=6.0,
                ),
            ],
        )
        # best_iteration=5 but no iteration with that number exists
        result = history.get_best_entity()
        assert result is None

    def test_add_iteration_auto_numbers_sequentially(self):
        """Test add_iteration auto-numbers iterations starting from 1."""
        history = RefinementHistory(entity_type="character", entity_name="Test")
        history.add_iteration(
            entity_data={"name": "First"},
            scores={"depth": 5.0},
            average_score=5.0,
        )
        history.add_iteration(
            entity_data={"name": "Second"},
            scores={"depth": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Third"},
            scores={"depth": 6.0},
            average_score=6.0,
        )
        assert len(history.iterations) == 3
        assert history.iterations[0].iteration == 1
        assert history.iterations[1].iteration == 2
        assert history.iterations[2].iteration == 3

    def test_add_iteration_tracks_best_and_degradation(self):
        """Test add_iteration correctly tracks peak score and consecutive degradations."""
        history = RefinementHistory(entity_type="faction", entity_name="Test Faction")
        # Iteration 1: score 5.0 → new peak
        history.add_iteration(entity_data={"name": "v1"}, scores={"s": 5.0}, average_score=5.0)
        assert history.best_iteration == 1
        assert history.peak_score == 5.0
        assert history.consecutive_degradations == 0

        # Iteration 2: score 7.0 → new peak
        history.add_iteration(entity_data={"name": "v2"}, scores={"s": 7.0}, average_score=7.0)
        assert history.best_iteration == 2
        assert history.peak_score == 7.0
        assert history.consecutive_degradations == 0

        # Iteration 3: score 6.0 → degradation
        history.add_iteration(entity_data={"name": "v3"}, scores={"s": 6.0}, average_score=6.0)
        assert history.best_iteration == 2  # Still iteration 2
        assert history.consecutive_degradations == 1

    def test_add_iteration_numbering_independent_of_loop_counter(self):
        """Test that iteration numbering is independent of caller's loop counter.

        This verifies the fix for Issue 1: when creation retries happen (empty name
        → no add_iteration call), the loop counter advances but iteration numbering
        stays sequential. Before the fix, best_iteration=3 could happen with only
        1 iteration recorded, causing 'Total iterations: 1, Best iteration: 3' in logs.
        """
        history = RefinementHistory(entity_type="location", entity_name="")

        # Simulate: loop counter 0 → creation fails (no add_iteration)
        # Simulate: loop counter 1 → creation fails (no add_iteration)
        # Simulate: loop counter 2 → creation succeeds, judge called
        history.add_iteration(
            entity_data={"name": "Place A"},
            scores={"atmosphere": 7.0},
            average_score=7.0,
        )
        # Should be iteration 1, not 3
        assert history.iterations[0].iteration == 1
        assert history.best_iteration == 1

        # Simulate: loop counter 3 → refinement, judge called
        history.add_iteration(
            entity_data={"name": "Place A v2"},
            scores={"atmosphere": 8.0},
            average_score=8.0,
        )
        assert history.iterations[1].iteration == 2
        assert history.best_iteration == 2
        assert len(history.iterations) == 2

    def test_get_best_entity_returns_correct_entity(self):
        """Test get_best_entity returns entity from best iteration."""
        from src.memory.world_quality import IterationRecord

        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=2,
            iterations=[
                IterationRecord(
                    iteration=1,
                    entity_data={"name": "First"},
                    scores={"depth": 5.0},
                    average_score=5.0,
                ),
                IterationRecord(
                    iteration=2,
                    entity_data={"name": "Second"},
                    scores={"depth": 8.0},
                    average_score=8.0,
                ),
            ],
        )
        result = history.get_best_entity()
        assert result == {"name": "Second"}


class TestAnalyzeImprovement:
    """Tests for analyze_improvement() method (#303)."""

    def test_returns_scoring_rounds_key(self):
        """analyze_improvement() uses 'scoring_rounds' (not 'total_iterations')."""
        history = RefinementHistory(entity_type="character", entity_name="Hero")
        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 7.0},
            average_score=7.0,
        )
        analysis = history.analyze_improvement()
        assert "scoring_rounds" in analysis
        assert "total_iterations" not in analysis
        assert analysis["scoring_rounds"] == 1

    def test_scoring_rounds_matches_iteration_count(self):
        """scoring_rounds equals the number of recorded iterations."""
        history = RefinementHistory(entity_type="faction", entity_name="Guild")
        history.add_iteration(
            entity_data={"name": "Guild v1"},
            scores={"s": 5.0},
            average_score=5.0,
        )
        history.add_iteration(
            entity_data={"name": "Guild v2"},
            scores={"s": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Guild v3"},
            scores={"s": 6.0},
            average_score=6.0,
        )
        analysis = history.analyze_improvement()
        assert analysis["scoring_rounds"] == 3

    def test_mid_loop_regression_true_when_score_drops(self):
        """mid_loop_regression is True when any iteration scores lower than its predecessor."""
        history = RefinementHistory(entity_type="location", entity_name="Tavern")
        history.add_iteration(
            entity_data={"name": "Tavern v1"},
            scores={"s": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern v2"},
            scores={"s": 8.0},
            average_score=8.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern v3"},
            scores={"s": 6.0},
            average_score=6.0,  # Drop from 8.0 to 6.0
        )
        analysis = history.analyze_improvement()
        assert analysis["mid_loop_regression"] is True

    def test_mid_loop_regression_false_when_monotonic_increase(self):
        """mid_loop_regression is False when scores only increase."""
        history = RefinementHistory(entity_type="character", entity_name="Hero")
        history.add_iteration(
            entity_data={"name": "Hero v1"},
            scores={"s": 5.0},
            average_score=5.0,
        )
        history.add_iteration(
            entity_data={"name": "Hero v2"},
            scores={"s": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Hero v3"},
            scores={"s": 9.0},
            average_score=9.0,
        )
        analysis = history.analyze_improvement()
        assert analysis["mid_loop_regression"] is False

    def test_mid_loop_regression_false_when_flat(self):
        """mid_loop_regression is False when scores are flat."""
        history = RefinementHistory(entity_type="item", entity_name="Sword")
        history.add_iteration(
            entity_data={"name": "Sword v1"},
            scores={"s": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Sword v2"},
            scores={"s": 6.0},
            average_score=6.0,
        )
        analysis = history.analyze_improvement()
        assert analysis["mid_loop_regression"] is False

    def test_mid_loop_regression_false_single_iteration(self):
        """mid_loop_regression is False with only one iteration."""
        history = RefinementHistory(entity_type="concept", entity_name="Magic")
        history.add_iteration(
            entity_data={"name": "Magic"},
            scores={"s": 8.0},
            average_score=8.0,
        )
        analysis = history.analyze_improvement()
        assert analysis["mid_loop_regression"] is False

    def test_mid_loop_regression_false_no_iterations(self):
        """mid_loop_regression is False with no iterations."""
        history = RefinementHistory(entity_type="concept", entity_name="Theme")
        analysis = history.analyze_improvement()
        assert analysis["mid_loop_regression"] is False


class TestConsecutivePlateaus:
    """Test consecutive_plateaus tracking in RefinementHistory."""

    def test_plateau_increments_on_equal_scores(self):
        """consecutive_plateaus increments when score equals peak."""
        history = RefinementHistory(entity_type="location", entity_name="Test Place")

        # First iteration sets peak
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 0
        assert history.peak_score == 6.0

        # Second iteration matches peak — plateau
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 1

        # Third iteration matches peak again — plateau
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 2

    def test_plateau_resets_on_new_peak(self):
        """consecutive_plateaus resets to 0 when a new peak is reached."""
        history = RefinementHistory(entity_type="character", entity_name="Hero")

        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 1

        # New peak resets plateaus
        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 7.0},
            average_score=7.0,
        )
        assert history.consecutive_plateaus == 0
        assert history.peak_score == 7.0

    def test_degradation_resets_plateau_streak(self):
        """Degradation resets consecutive_plateaus (counters are mutually exclusive)."""
        history = RefinementHistory(entity_type="item", entity_name="Sword")

        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 7.0},
            average_score=7.0,
        )
        assert history.consecutive_plateaus == 1

        # Degradation — resets plateau streak
        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 5.0},
            average_score=5.0,
        )
        assert history.consecutive_plateaus == 0
        assert history.consecutive_degradations == 1


class TestShouldStopEarlyPlateau:
    """Test should_stop_early() plateau detection."""

    def test_flat_scores_trigger_early_stop(self):
        """Flat scores [6.0, 6.0, 6.0] with patience=2 triggers early stop."""
        history = RefinementHistory(entity_type="location", entity_name="Tavern")

        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )

        assert history.consecutive_plateaus == 2
        assert history.should_stop_early(patience=2, min_iterations=2) is True

    def test_plateau_does_not_trigger_before_min_iterations(self):
        """Plateau should not trigger early stop before min_iterations."""
        history = RefinementHistory(entity_type="faction", entity_name="Guild")

        history.add_iteration(
            entity_data={"name": "Guild"},
            scores={"coherence": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Guild"},
            scores={"coherence": 6.0},
            average_score=6.0,
        )

        # 2 iterations, min_iterations=3 — should NOT trigger
        assert history.consecutive_plateaus == 1
        assert history.should_stop_early(patience=1, min_iterations=3) is False

    def test_plateau_below_patience_does_not_trigger(self):
        """Plateau count below patience should not trigger early stop."""
        history = RefinementHistory(entity_type="concept", entity_name="Theme")

        history.add_iteration(
            entity_data={"name": "Theme"},
            scores={"relevance": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Theme"},
            scores={"relevance": 6.0},
            average_score=6.0,
        )

        # 1 plateau, patience=2 — should NOT trigger
        assert history.consecutive_plateaus == 1
        assert history.should_stop_early(patience=2, min_iterations=2) is False

    def test_mixed_plateau_then_improvement(self):
        """Plateau followed by improvement should NOT trigger early stop."""
        history = RefinementHistory(entity_type="location", entity_name="Castle")

        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 7.0},
            average_score=7.0,
        )

        assert history.consecutive_plateaus == 0
        assert history.should_stop_early(patience=2, min_iterations=2) is False

    def test_degradation_still_triggers_early_stop(self):
        """Degradation-based early stop still works alongside plateau detection."""
        history = RefinementHistory(entity_type="item", entity_name="Ring")

        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 8.0},
            average_score=8.0,
        )
        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 5.0},
            average_score=5.0,
        )

        assert history.consecutive_degradations == 2
        assert history.consecutive_plateaus == 0
        # Score drop is 3.0, well above variance tolerance of 0.3
        assert history.should_stop_early(patience=2, min_iterations=2) is True


class TestShouldStopEarlyNoPeak:
    """Test should_stop_early() when best_iteration is 0 (no peak established)."""

    def test_returns_false_when_no_peak_even_with_enough_iterations(self):
        """should_stop_early returns False when best_iteration is 0 (all zero scores)."""
        history = RefinementHistory(entity_type="character", entity_name="Test")
        # All zero scores: best_iteration stays 0 because 0.0 is not > 0.0 (peak_score)
        history.add_iteration(entity_data={"v": 1}, scores={"s": 0.0}, average_score=0.0)
        history.add_iteration(entity_data={"v": 2}, scores={"s": 0.0}, average_score=0.0)
        history.add_iteration(entity_data={"v": 3}, scores={"s": 0.0}, average_score=0.0)

        assert history.best_iteration == 0
        assert len(history.iterations) == 3
        # Even with enough iterations and plateaus, returns False when no peak
        assert history.should_stop_early(patience=2, min_iterations=2) is False


class TestFactionQualityScores:
    """Tests for FactionQualityScores model."""

    def test_weak_dimensions_identifies_conflict_potential(self):
        """Test weak_dimensions identifies low conflict_potential."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=5.0,  # Below threshold
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "conflict_potential" in weak
        assert "coherence" not in weak
        assert "influence" not in weak
        assert "distinctiveness" not in weak

    def test_weak_dimensions_identifies_distinctiveness(self):
        """Test weak_dimensions identifies low distinctiveness."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=5.0,  # Below threshold
            temporal_plausibility=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "distinctiveness" in weak
        assert "coherence" not in weak
        assert "influence" not in weak
        assert "conflict_potential" not in weak

    def test_weak_dimensions_identifies_multiple(self):
        """Test weak_dimensions identifies multiple low dimensions."""
        scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert len(weak) == 5
        assert "coherence" in weak
        assert "influence" in weak
        assert "conflict_potential" in weak
        assert "distinctiveness" in weak

    def test_weak_dimensions_returns_empty_when_all_above_threshold(self):
        """Test weak_dimensions returns empty list when all scores high."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert weak == []


class TestScoreStatistics:
    """Tests for ScoreStatistics model."""

    def test_calculate_empty_list(self) -> None:
        """Test calculate with empty score list."""
        stats = ScoreStatistics.calculate([])
        assert stats.mean == 0.0
        assert stats.std == 0.0
        assert stats.confidence == 0.0
        assert stats.scores == []

    def test_calculate_single_score(self) -> None:
        """Test calculate with single score."""
        stats = ScoreStatistics.calculate([8.0])
        assert stats.mean == 8.0
        assert stats.std == 0.0
        assert stats.confidence == 1.0  # Perfect confidence with single score

    def test_calculate_multiple_scores(self) -> None:
        """Test calculate with multiple scores."""
        stats = ScoreStatistics.calculate([7.0, 8.0, 9.0])
        assert stats.mean == 8.0
        assert stats.std == pytest.approx(1.0, rel=0.01)
        # Confidence = 1.0 - (1.0 / 8.0) = 0.875
        assert stats.confidence == pytest.approx(0.875, rel=0.01)

    def test_calculate_identical_scores(self) -> None:
        """Test calculate with identical scores has perfect confidence."""
        stats = ScoreStatistics.calculate([8.0, 8.0, 8.0])
        assert stats.mean == 8.0
        assert stats.std == 0.0
        assert stats.confidence == 1.0

    def test_calculate_high_variance_low_confidence(self) -> None:
        """Test that high variance leads to low confidence."""
        stats = ScoreStatistics.calculate([2.0, 5.0, 8.0])
        assert stats.mean == 5.0
        assert stats.std == pytest.approx(3.0, rel=0.01)
        # CV = 3.0 / 5.0 = 0.6, confidence = 1.0 - 0.6 = 0.4
        assert stats.confidence == pytest.approx(0.4, rel=0.01)

    def test_detect_outliers_no_outliers(self) -> None:
        """Test detect_outliers with no outliers."""
        stats = ScoreStatistics.calculate([7.5, 8.0, 8.5])
        outliers = stats.detect_outliers(std_threshold=2.0)
        assert outliers == []

    def test_detect_outliers_with_outlier(self) -> None:
        """Test detect_outliers detects extreme values."""
        # Use more extreme outlier: 0.0 among 8.0s
        # Mean = 6.4, std ~= 3.57, z-score for 0.0 = 6.4/3.57 ~= 1.79 - still not enough
        # Need even more extreme: use 8.0s with -5.0
        # Mean = 5.4, variance = (4*2.6^2 + 10.4^2)/4 = (27.04 + 108.16)/4 = 33.8, std = 5.81
        # z-score for -5.0 = 10.4/5.81 ~= 1.79 - still not 2.0
        # Let's use threshold 1.5 instead
        stats = ScoreStatistics.calculate([8.0, 8.0, 8.0, 8.0, 2.0])
        outliers = stats.detect_outliers(std_threshold=1.5)
        assert 4 in outliers  # Index 4 (value 2.0) is outlier at 1.5 threshold

    def test_detect_outliers_too_few_samples(self) -> None:
        """Test detect_outliers returns empty for fewer than 3 samples."""
        stats = ScoreStatistics.calculate([5.0, 10.0])
        outliers = stats.detect_outliers()
        assert outliers == []

    def test_detect_outliers_zero_std(self) -> None:
        """Test detect_outliers returns empty when std is zero."""
        stats = ScoreStatistics.calculate([8.0, 8.0, 8.0])
        outliers = stats.detect_outliers()
        assert outliers == []

    def test_get_filtered_mean_no_exclusions(self) -> None:
        """Test get_filtered_mean with no exclusions."""
        stats = ScoreStatistics.calculate([6.0, 7.0, 8.0, 9.0])
        filtered_mean = stats.get_filtered_mean()
        assert filtered_mean == 7.5  # Same as regular mean

    def test_get_filtered_mean_with_exclusions(self) -> None:
        """Test get_filtered_mean excludes specified indices."""
        stats = ScoreStatistics.calculate([6.0, 7.0, 8.0, 2.0])  # 2.0 is outlier
        stats.outliers = [3]  # Mark index 3 as outlier
        filtered_mean = stats.get_filtered_mean()
        assert filtered_mean == 7.0  # (6 + 7 + 8) / 3

    def test_get_filtered_mean_custom_exclusions(self) -> None:
        """Test get_filtered_mean with custom exclusion list."""
        stats = ScoreStatistics.calculate([6.0, 7.0, 8.0, 9.0])
        filtered_mean = stats.get_filtered_mean(exclude_indices=[0, 3])
        assert filtered_mean == 7.5  # (7 + 8) / 2

    def test_get_median_empty(self) -> None:
        """Test get_median with empty scores."""
        stats = ScoreStatistics(scores=[])
        assert stats.get_median() == 0.0

    def test_get_median_odd_count(self) -> None:
        """Test get_median with odd number of scores."""
        stats = ScoreStatistics.calculate([7.0, 8.0, 9.0])
        assert stats.get_median() == 8.0

    def test_get_median_even_count(self) -> None:
        """Test get_median with even number of scores."""
        stats = ScoreStatistics.calculate([7.0, 8.0, 9.0, 10.0])
        assert stats.get_median() == 8.5  # (8 + 9) / 2

    def test_should_refine_below_threshold(self) -> None:
        """Test should_refine returns True when mean is below threshold."""
        stats = ScoreStatistics.calculate([6.0, 7.0, 6.5])
        should_refine, threshold = stats.should_refine(threshold=7.5)
        assert should_refine is True
        assert threshold == 7.5

    def test_should_refine_above_threshold(self) -> None:
        """Test should_refine returns False when mean is above threshold."""
        stats = ScoreStatistics.calculate([8.0, 8.5, 9.0])
        should_refine, threshold = stats.should_refine(threshold=7.5)
        assert should_refine is False
        assert threshold == 7.5

    def test_should_refine_low_confidence_adjusts_threshold(self) -> None:
        """Test should_refine increases threshold with low confidence."""
        # High variance scores = low confidence
        stats = ScoreStatistics.calculate([2.0, 5.0, 8.0])
        # Mean is 5.0, confidence is low (~0.4)
        should_refine, adjusted = stats.should_refine(
            threshold=4.5, confidence_threshold=0.7, min_samples=3
        )
        # Threshold should be adjusted upward
        assert adjusted > 4.5
        # With adjusted threshold, should still refine
        assert should_refine is True

    def test_should_refine_min_samples_uses_standard_threshold(self) -> None:
        """Test should_refine uses standard threshold below min_samples."""
        stats = ScoreStatistics.calculate([7.0, 7.5])  # Only 2 samples
        _should_refine, threshold = stats.should_refine(threshold=7.0, min_samples=3)
        # Not enough samples, uses standard threshold
        assert threshold == 7.0


class TestJudgeConsistencyConfig:
    """Tests for JudgeConsistencyConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = JudgeConsistencyConfig()
        assert config.enabled is False  # Opt-in: disabled by default
        assert config.multi_call_enabled is False
        assert config.multi_call_count == 2
        assert config.confidence_threshold == 0.7
        assert config.outlier_detection is True
        assert config.outlier_std_threshold == 2.0
        assert config.outlier_strategy == "median"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = JudgeConsistencyConfig(
            enabled=False,
            multi_call_enabled=True,
            multi_call_count=5,
            confidence_threshold=0.8,
            outlier_detection=False,
            outlier_std_threshold=3.0,
            outlier_strategy="mean",
        )
        assert config.enabled is False
        assert config.multi_call_enabled is True
        assert config.multi_call_count == 5
        assert config.confidence_threshold == 0.8
        assert config.outlier_detection is False
        assert config.outlier_std_threshold == 3.0
        assert config.outlier_strategy == "mean"

    def test_multi_call_count_validation(self) -> None:
        """Test that multi_call_count has valid bounds."""
        # Below minimum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(multi_call_count=1)

        # Above maximum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(multi_call_count=6)

    def test_confidence_threshold_validation(self) -> None:
        """Test that confidence_threshold has valid bounds."""
        # Below minimum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(confidence_threshold=-0.1)

        # Above maximum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(confidence_threshold=1.5)

    def test_outlier_std_threshold_validation(self) -> None:
        """Test that outlier_std_threshold has valid bounds."""
        # Below minimum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(outlier_std_threshold=0.5)

        # Above maximum
        with pytest.raises(ValueError):
            JudgeConsistencyConfig(outlier_std_threshold=5.0)

    def test_from_settings(self) -> None:
        """Test from_settings creates config from settings object."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.judge_consistency_enabled = False
        mock_settings.judge_multi_call_enabled = True
        mock_settings.judge_multi_call_count = 4
        mock_settings.judge_confidence_threshold = 0.8
        mock_settings.judge_outlier_detection = False
        mock_settings.judge_outlier_std_threshold = 2.5
        mock_settings.judge_outlier_strategy = "mean"

        config = JudgeConsistencyConfig.from_settings(mock_settings)

        assert config.enabled is False
        assert config.multi_call_enabled is True
        assert config.multi_call_count == 4
        assert config.confidence_threshold == 0.8
        assert config.outlier_detection is False
        assert config.outlier_std_threshold == 2.5
        assert config.outlier_strategy == "mean"


class TestScoreStatisticsEdgeCases:
    """Additional edge case tests for ScoreStatistics."""

    def test_calculate_zero_mean_with_variance(self) -> None:
        """Test calculate with zero mean but positive variance returns confidence 0.0."""
        # Scores that average to 0 but have variance
        stats = ScoreStatistics.calculate([-5.0, 0.0, 5.0])
        assert stats.mean == 0.0
        assert stats.std > 0.0
        assert stats.confidence == 0.0  # Zero mean with variance = 0 confidence

    def test_get_filtered_mean_all_excluded(self) -> None:
        """Test get_filtered_mean returns mean when all scores excluded."""
        stats = ScoreStatistics.calculate([6.0, 7.0, 8.0])
        # Exclude all indices
        filtered_mean = stats.get_filtered_mean(exclude_indices=[0, 1, 2])
        assert filtered_mean == stats.mean  # Falls back to regular mean


class TestPlotQualityScores:
    """Tests for PlotQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation for plot quality."""
        scores = PlotQualityScores(
            coherence=8.0,
            tension_arc=6.0,
            character_integration=7.0,
            originality=9.0,
        )
        assert scores.average == 7.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = PlotQualityScores(
            coherence=8.0,
            tension_arc=6.0,
            character_integration=7.0,
            originality=9.0,
            feedback="Good plot",
        )
        result = scores.to_dict()
        assert result["coherence"] == 8.0
        assert result["tension_arc"] == 6.0
        assert result["character_integration"] == 7.0
        assert result["originality"] == 9.0
        assert result["average"] == 7.5
        assert result["feedback"] == "Good plot"

    def test_weak_dimensions_identifies_low_scores(self):
        """Test weak_dimensions identifies dimensions below threshold."""
        scores = PlotQualityScores(
            coherence=8.0,
            tension_arc=5.0,
            character_integration=8.0,
            originality=4.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "tension_arc" in weak
        assert "originality" in weak
        assert "coherence" not in weak
        assert "character_integration" not in weak

    def test_weak_dimensions_returns_empty_when_all_above(self):
        """Test weak_dimensions returns empty list when all scores high."""
        scores = PlotQualityScores(
            coherence=8.0,
            tension_arc=8.0,
            character_integration=8.0,
            originality=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert weak == []

    def test_score_validation_bounds(self):
        """Test that scores are bounded 0-10."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlotQualityScores(
                coherence=11.0,
                tension_arc=5.0,
                character_integration=5.0,
                originality=5.0,
            )

        with pytest.raises(ValidationError):
            PlotQualityScores(
                coherence=5.0,
                tension_arc=-1.0,
                character_integration=5.0,
                originality=5.0,
            )


class TestChapterQualityScores:
    """Tests for ChapterQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation for chapter quality."""
        scores = ChapterQualityScores(
            purpose=8.0,
            pacing=6.0,
            hook=7.0,
            coherence=9.0,
        )
        assert scores.average == 7.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = ChapterQualityScores(
            purpose=8.0,
            pacing=6.0,
            hook=7.0,
            coherence=9.0,
            feedback="Good chapter",
        )
        result = scores.to_dict()
        assert result["purpose"] == 8.0
        assert result["pacing"] == 6.0
        assert result["hook"] == 7.0
        assert result["coherence"] == 9.0
        assert result["average"] == 7.5
        assert result["feedback"] == "Good chapter"

    def test_weak_dimensions_identifies_low_scores(self):
        """Test weak_dimensions identifies dimensions below threshold."""
        scores = ChapterQualityScores(
            purpose=8.0,
            pacing=5.0,
            hook=4.0,
            coherence=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "pacing" in weak
        assert "hook" in weak
        assert "purpose" not in weak
        assert "coherence" not in weak

    def test_weak_dimensions_returns_empty_when_all_above(self):
        """Test weak_dimensions returns empty list when all scores high."""
        scores = ChapterQualityScores(
            purpose=8.0,
            pacing=8.0,
            hook=8.0,
            coherence=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert weak == []

    def test_score_validation_bounds(self):
        """Test that scores are bounded 0-10."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChapterQualityScores(
                purpose=11.0,
                pacing=5.0,
                hook=5.0,
                coherence=5.0,
            )

        with pytest.raises(ValidationError):
            ChapterQualityScores(
                purpose=5.0,
                pacing=-1.0,
                hook=5.0,
                coherence=5.0,
            )


class TestCharacterTemporalPlausibility:
    """Tests for temporal_plausibility in CharacterQualityScores."""

    def test_temporal_plausibility_in_average(self):
        """Average includes temporal_plausibility (sum of 6 dimensions / 6.0)."""
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )
        assert scores.average == 6.0

    def test_temporal_plausibility_affects_average(self):
        """Changing temporal_plausibility changes the average."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=2.0,
        )
        # (8*5 + 2) / 6 = 42/6 = 7.0
        assert scores.average == 7.0

    def test_temporal_plausibility_in_to_dict(self):
        """to_dict includes 'temporal_plausibility' key."""
        scores = CharacterQualityScores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=7.0,
            arc_potential=7.0,
            temporal_plausibility=5.0,
        )
        result = scores.to_dict()
        assert "temporal_plausibility" in result
        assert result["temporal_plausibility"] == 5.0

    def test_temporal_plausibility_weak_dimension(self):
        """weak_dimensions includes temporal_plausibility when below threshold."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" in weak
        assert "depth" not in weak

    def test_temporal_plausibility_not_weak_when_above(self):
        """weak_dimensions excludes temporal_plausibility when above threshold."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" not in weak


class TestLocationTemporalPlausibility:
    """Tests for temporal_plausibility in LocationQualityScores."""

    def test_temporal_plausibility_in_average(self):
        """Average includes temporal_plausibility (sum of 5 dimensions / 5.0)."""
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )
        assert scores.average == 6.0

    def test_temporal_plausibility_affects_average(self):
        """Changing temporal_plausibility changes the average."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=3.0,
        )
        # (8*4 + 3) / 5 = 35/5 = 7.0
        assert scores.average == 7.0

    def test_temporal_plausibility_in_to_dict(self):
        """to_dict includes 'temporal_plausibility' key."""
        scores = LocationQualityScores(
            atmosphere=7.0,
            significance=7.0,
            story_relevance=7.0,
            distinctiveness=7.0,
            temporal_plausibility=4.0,
        )
        result = scores.to_dict()
        assert "temporal_plausibility" in result
        assert result["temporal_plausibility"] == 4.0

    def test_temporal_plausibility_weak_dimension(self):
        """weak_dimensions includes temporal_plausibility when below threshold."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" in weak
        assert "atmosphere" not in weak

    def test_temporal_plausibility_not_weak_when_above(self):
        """weak_dimensions excludes temporal_plausibility when above threshold."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" not in weak


class TestFactionTemporalPlausibility:
    """Tests for temporal_plausibility in FactionQualityScores."""

    def test_temporal_plausibility_in_average(self):
        """Average includes temporal_plausibility (sum of 5 dimensions / 5.0)."""
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )
        assert scores.average == 6.0

    def test_temporal_plausibility_affects_average(self):
        """Changing temporal_plausibility changes the average."""
        scores = FactionQualityScores(
            coherence=10.0,
            influence=10.0,
            conflict_potential=10.0,
            distinctiveness=10.0,
            temporal_plausibility=0.0,
        )
        # (10*4 + 0) / 5 = 40/5 = 8.0
        assert scores.average == 8.0

    def test_temporal_plausibility_in_to_dict(self):
        """to_dict includes 'temporal_plausibility' key."""
        scores = FactionQualityScores(
            coherence=7.0,
            influence=7.0,
            conflict_potential=7.0,
            distinctiveness=7.0,
            temporal_plausibility=3.0,
        )
        result = scores.to_dict()
        assert "temporal_plausibility" in result
        assert result["temporal_plausibility"] == 3.0

    def test_temporal_plausibility_weak_dimension(self):
        """weak_dimensions includes temporal_plausibility when below threshold."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" in weak
        assert "coherence" not in weak

    def test_temporal_plausibility_not_weak_when_above(self):
        """weak_dimensions excludes temporal_plausibility when above threshold."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" not in weak


class TestItemTemporalPlausibility:
    """Tests for temporal_plausibility in ItemQualityScores."""

    def test_temporal_plausibility_in_average(self):
        """Average includes temporal_plausibility (sum of 5 dimensions / 5.0)."""
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=6.0,
        )
        assert scores.average == 6.0

    def test_temporal_plausibility_affects_average(self):
        """Changing temporal_plausibility changes the average."""
        scores = ItemQualityScores(
            significance=10.0,
            uniqueness=10.0,
            narrative_potential=10.0,
            integration=10.0,
            temporal_plausibility=5.0,
        )
        # (10*4 + 5) / 5 = 45/5 = 9.0
        assert scores.average == 9.0

    def test_temporal_plausibility_in_to_dict(self):
        """to_dict includes 'temporal_plausibility' key."""
        scores = ItemQualityScores(
            significance=7.0,
            uniqueness=7.0,
            narrative_potential=7.0,
            integration=7.0,
            temporal_plausibility=2.0,
        )
        result = scores.to_dict()
        assert "temporal_plausibility" in result
        assert result["temporal_plausibility"] == 2.0

    def test_temporal_plausibility_weak_dimension(self):
        """weak_dimensions includes temporal_plausibility when below threshold."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" in weak
        assert "uniqueness" not in weak

    def test_temporal_plausibility_not_weak_when_above(self):
        """weak_dimensions excludes temporal_plausibility when above threshold."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" not in weak


class TestConceptTemporalPlausibility:
    """Tests for temporal_plausibility in ConceptQualityScores."""

    def test_temporal_plausibility_in_average(self):
        """Average includes temporal_plausibility (sum of 5 dimensions / 5.0)."""
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=6.0,
        )
        assert scores.average == 6.0

    def test_temporal_plausibility_affects_average(self):
        """Changing temporal_plausibility changes the average."""
        scores = ConceptQualityScores(
            relevance=10.0,
            depth=10.0,
            manifestation=10.0,
            resonance=10.0,
            temporal_plausibility=0.0,
        )
        # (10*4 + 0) / 5 = 40/5 = 8.0
        assert scores.average == 8.0

    def test_temporal_plausibility_in_to_dict(self):
        """to_dict includes 'temporal_plausibility' key."""
        scores = ConceptQualityScores(
            relevance=7.0,
            depth=7.0,
            manifestation=7.0,
            resonance=7.0,
            temporal_plausibility=1.0,
        )
        result = scores.to_dict()
        assert "temporal_plausibility" in result
        assert result["temporal_plausibility"] == 1.0

    def test_temporal_plausibility_weak_dimension(self):
        """weak_dimensions includes temporal_plausibility when below threshold."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" in weak
        assert "relevance" not in weak

    def test_temporal_plausibility_not_weak_when_above(self):
        """weak_dimensions excludes temporal_plausibility when above threshold."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "temporal_plausibility" not in weak
