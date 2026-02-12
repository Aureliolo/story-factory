"""Tests for refinement analytics logging (#328).

Tests cover:
- Condensed single-line log for first-try passes
- Full multi-line log for multi-iteration, regression, or failure cases
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.memory.world_quality import RefinementHistory
from src.services.world_quality_service._analytics import log_refinement_analytics


@pytest.fixture
def mock_svc():
    """Create a mock WorldQualityService with analytics DB."""
    svc = MagicMock()
    svc.analytics_db = MagicMock()
    svc._get_creator_model = MagicMock(return_value="test-model:8b")
    return svc


def _make_history(
    entity_type: str = "character",
    entity_name: str = "Hero",
    iterations: int = 1,
    scores: list[float] | None = None,
    final_score: float = 8.5,
) -> RefinementHistory:
    """Create a RefinementHistory with the given number of iterations."""
    history = RefinementHistory(entity_type=entity_type, entity_name=entity_name)
    score_values = scores or [final_score] * iterations
    for i, score in enumerate(score_values):
        history.add_iteration(
            entity_data={"name": entity_name, "version": i + 1},
            scores={"depth": score, "goals": score, "average": score},
            average_score=score,
            feedback=f"Iteration {i + 1} feedback",
        )
    history.final_iteration = len(score_values)
    history.final_score = final_score
    return history


class TestCondensedAnalytics:
    """Tests for condensed analytics on first-try passes (#328 B3)."""

    def test_single_iteration_pass_logs_condensed(self, mock_svc, caplog):
        """Single iteration + threshold met → condensed single-line log."""
        history = _make_history(iterations=1, final_score=8.5)

        with caplog.at_level(logging.INFO):
            log_refinement_analytics(
                mock_svc,
                history,
                "test-project",
                threshold_met=True,
            )

        analytics_msgs = [msg for msg in caplog.messages if "REFINEMENT ANALYTICS" in msg]
        assert len(analytics_msgs) == 1
        msg = analytics_msgs[0]
        # Should be a single line, not multi-line
        assert "\n" not in msg
        assert "score=8.5" in msg
        assert "iterations=1" in msg
        assert "passed=True" in msg

    def test_single_iteration_fail_logs_full(self, mock_svc, caplog):
        """Single iteration + threshold NOT met → full multi-line log."""
        history = _make_history(iterations=1, final_score=5.0)

        with caplog.at_level(logging.INFO):
            log_refinement_analytics(
                mock_svc,
                history,
                "test-project",
                threshold_met=False,
            )

        analytics_msgs = [msg for msg in caplog.messages if "REFINEMENT ANALYTICS" in msg]
        assert len(analytics_msgs) == 1
        msg = analytics_msgs[0]
        # Full block has multiple lines
        assert "\n" in msg
        assert "Scoring rounds:" in msg

    def test_multi_iteration_pass_logs_full(self, mock_svc, caplog):
        """Multiple iterations + threshold met → full multi-line log."""
        history = _make_history(
            iterations=3,
            scores=[5.0, 6.5, 8.5],
            final_score=8.5,
        )

        with caplog.at_level(logging.INFO):
            log_refinement_analytics(
                mock_svc,
                history,
                "test-project",
                threshold_met=True,
            )

        analytics_msgs = [msg for msg in caplog.messages if "REFINEMENT ANALYTICS" in msg]
        assert len(analytics_msgs) == 1
        msg = analytics_msgs[0]
        # Multi-iteration always gets full block
        assert "\n" in msg
        assert "Scoring rounds:" in msg

    def test_multi_iteration_early_stop_logs_full(self, mock_svc, caplog):
        """Multiple iterations + early stop → full multi-line log."""
        history = _make_history(
            iterations=3,
            scores=[6.0, 6.0, 6.0],
            final_score=6.0,
        )

        with caplog.at_level(logging.INFO):
            log_refinement_analytics(
                mock_svc,
                history,
                "test-project",
                threshold_met=False,
                early_stop_triggered=True,
            )

        analytics_msgs = [msg for msg in caplog.messages if "REFINEMENT ANALYTICS" in msg]
        assert len(analytics_msgs) == 1
        msg = analytics_msgs[0]
        assert "\n" in msg
        assert "Early stop triggered: True" in msg

    def test_condensed_log_still_records_to_db(self, mock_svc, caplog):
        """Condensed log path still calls record_entity_quality."""
        history = _make_history(iterations=1, final_score=8.5)

        with (
            caplog.at_level(logging.INFO),
            patch(
                "src.services.world_quality_service._analytics.record_entity_quality"
            ) as mock_record,
        ):
            log_refinement_analytics(
                mock_svc,
                history,
                "test-project",
                threshold_met=True,
            )

        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args
        assert call_kwargs.kwargs["threshold_met"] is True
        assert call_kwargs.kwargs["entity_type"] == "character"
