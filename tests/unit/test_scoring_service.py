"""Tests for the scoring service."""

from unittest.mock import MagicMock

import pytest

from services.scoring_service import ScoringService


class TestScoringServiceExceptionHandling:
    """Tests for exception handling in ScoringService."""

    @pytest.fixture
    def mock_mode_service(self):
        """Create a mock ModelModeService."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.user_rating_min = 1
        mock.settings.user_rating_max = 5
        return mock

    @pytest.fixture
    def scoring_service(self, mock_mode_service):
        """Create a ScoringService with mocked dependencies."""
        return ScoringService(mock_mode_service)

    def test_start_tracking_raises_on_exception(self, scoring_service, mock_mode_service):
        """Test start_tracking re-raises exceptions from mode_service."""
        # Lines 98-102: Exception handling in start_tracking
        mock_mode_service.record_generation.side_effect = RuntimeError("Database error")

        with pytest.raises(RuntimeError, match="Database error"):
            scoring_service.start_tracking(
                project_id="test-project",
                agent_role="writer",
                model_id="test-model",
            )

    def test_finish_tracking_raises_on_exception(self, scoring_service, mock_mode_service):
        """Test finish_tracking re-raises exceptions from mode_service."""
        # Lines 145-147: Exception handling in finish_tracking
        mock_mode_service.update_performance_metrics.side_effect = RuntimeError("Metrics error")

        with pytest.raises(RuntimeError, match="Metrics error"):
            scoring_service.finish_tracking(
                score_id=1,
                content="Test content",
                tokens_generated=100,
                time_seconds=5.0,
            )

    def test_judge_chapter_quality_raises_on_exception(self, scoring_service, mock_mode_service):
        """Test judge_chapter_quality re-raises exceptions from mode_service."""
        # Lines 287-289: Exception handling in judge_chapter_quality
        mock_mode_service.judge_quality.side_effect = RuntimeError("Quality judgment error")

        with pytest.raises(RuntimeError, match="Quality judgment error"):
            scoring_service.judge_chapter_quality(
                content="Test chapter content",
                genre="Fantasy",
                tone="Epic",
                themes=["Adventure"],
            )

    def test_calculate_consistency_score_raises_on_exception(
        self, scoring_service, mock_mode_service
    ):
        """Test calculate_consistency_score re-raises exceptions from mode_service."""
        # Lines 319-321: Exception handling in calculate_consistency_score
        mock_mode_service.calculate_consistency_score.side_effect = RuntimeError(
            "Consistency error"
        )

        with pytest.raises(RuntimeError, match="Consistency error"):
            scoring_service.calculate_consistency_score(issues=[{"severity": "minor"}])

    def test_start_tracking_logs_error_on_exception(
        self, scoring_service, mock_mode_service, caplog
    ):
        """Test start_tracking logs error with exc_info on exception."""
        import logging

        mock_mode_service.record_generation.side_effect = ValueError("Invalid input")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                scoring_service.start_tracking(
                    project_id="test-project",
                    agent_role="writer",
                    model_id="test-model",
                )

        assert "Failed to start tracking" in caplog.text

    def test_finish_tracking_logs_error_on_exception(
        self, scoring_service, mock_mode_service, caplog
    ):
        """Test finish_tracking logs error with exc_info on exception."""
        import logging

        mock_mode_service.update_performance_metrics.side_effect = ValueError("Invalid metrics")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                scoring_service.finish_tracking(
                    score_id=1,
                    content="Test content",
                    tokens_generated=100,
                    time_seconds=5.0,
                )

        assert "Failed to finish tracking" in caplog.text

    def test_judge_chapter_quality_logs_error_on_exception(
        self, scoring_service, mock_mode_service, caplog
    ):
        """Test judge_chapter_quality logs error with exc_info on exception."""
        import logging

        mock_mode_service.judge_quality.side_effect = ValueError("Invalid content")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                scoring_service.judge_chapter_quality(
                    content="Test content",
                    genre="Fantasy",
                    tone="Epic",
                    themes=["Adventure"],
                )

        assert "Failed to judge quality" in caplog.text

    def test_calculate_consistency_score_logs_error_on_exception(
        self, scoring_service, mock_mode_service, caplog
    ):
        """Test calculate_consistency_score logs error with exc_info on exception."""
        import logging

        mock_mode_service.calculate_consistency_score.side_effect = ValueError("Invalid issues")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                scoring_service.calculate_consistency_score(issues=[])

        assert "Failed to calculate consistency score" in caplog.text
