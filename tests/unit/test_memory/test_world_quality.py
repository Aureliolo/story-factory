"""Tests for world quality models."""

import pytest

from src.memory.world_quality import (
    FactionQualityScores,
    JudgeConsistencyConfig,
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


class TestFactionQualityScores:
    """Tests for FactionQualityScores model."""

    def test_weak_dimensions_identifies_conflict_potential(self):
        """Test weak_dimensions identifies low conflict_potential."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=5.0,  # Below threshold
            distinctiveness=8.0,
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
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert len(weak) == 4
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
        assert config.enabled is True
        assert config.multi_call_enabled is False
        assert config.multi_call_count == 3
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
            outlier_strategy="retry",
        )
        assert config.enabled is False
        assert config.multi_call_enabled is True
        assert config.multi_call_count == 5
        assert config.confidence_threshold == 0.8
        assert config.outlier_detection is False
        assert config.outlier_std_threshold == 3.0
        assert config.outlier_strategy == "retry"

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
        mock_settings.judge_outlier_strategy = "retry"

        config = JudgeConsistencyConfig.from_settings(mock_settings)

        assert config.enabled is False
        assert config.multi_call_enabled is True
        assert config.multi_call_count == 4
        assert config.confidence_threshold == 0.8
        assert config.outlier_detection is False
        assert config.outlier_std_threshold == 2.5
        assert config.outlier_strategy == "retry"


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
