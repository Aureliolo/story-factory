"""Tests for batch generation/review summary logging (#303)."""

import logging
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import CharacterQualityScores
from src.services.world_quality_service._batch import (
    _generate_batch,
    _log_batch_summary,
    _review_batch,
)


def _make_scores(avg: float, feedback: str = "Test") -> CharacterQualityScores:
    """Create CharacterQualityScores where all dimensions equal avg."""
    return CharacterQualityScores(
        depth=avg, goals=avg, flaws=avg, uniqueness=avg, arc_potential=avg, feedback=feedback
    )


@pytest.fixture
def mock_svc():
    """Create a mock WorldQualityService."""
    svc = MagicMock()
    svc._calculate_eta = MagicMock(return_value=0.0)
    config = MagicMock()
    config.quality_threshold = 7.5
    svc.get_config = MagicMock(return_value=config)
    return svc


class TestLogBatchSummary:
    """Tests for _log_batch_summary helper."""

    def test_empty_results_logs_zero(self, caplog):
        """Empty results list logs '0 entities produced'."""
        with caplog.at_level(logging.INFO):
            _log_batch_summary([], "character", 7.5, 1.0)

        assert any("0 entities produced" in msg for msg in caplog.messages)

    def test_all_pass_threshold(self, caplog):
        """All entities above threshold → passed=N/N, no below threshold names."""
        results = [
            ({"name": "Hero"}, _make_scores(8.0)),
            ({"name": "Villain"}, _make_scores(9.0)),
        ]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "character", 7.5, 5.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch character summary" in msg]
        assert len(summary_msgs) == 1
        assert "passed=2/2" in summary_msgs[0]
        assert "below threshold" not in summary_msgs[0]

    def test_some_fail_threshold(self, caplog):
        """Entities below threshold → names listed in summary."""
        results = [
            ({"name": "Hero"}, _make_scores(8.0)),
            ({"name": "Weak NPC"}, _make_scores(6.0)),
        ]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "character", 7.5, 3.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch character summary" in msg]
        assert len(summary_msgs) == 1
        assert "passed=1/2" in summary_msgs[0]
        assert "Weak NPC" in summary_msgs[0]

    def test_includes_score_stats(self, caplog):
        """Summary includes min, max, avg scores."""
        results = [
            ({"name": "A"}, _make_scores(6.0)),
            ({"name": "B"}, _make_scores(8.0)),
            ({"name": "C"}, _make_scores(10.0)),
        ]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "faction", 7.0, 10.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch faction summary" in msg]
        assert len(summary_msgs) == 1
        assert "min=6.0" in summary_msgs[0]
        assert "max=10.0" in summary_msgs[0]
        assert "avg=8.0" in summary_msgs[0]

    def test_includes_threshold(self, caplog):
        """Summary includes the quality threshold value."""
        results = [({"name": "Item"}, _make_scores(7.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "item", 7.5, 2.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch item summary" in msg]
        assert len(summary_msgs) == 1
        assert "threshold=7.5" in summary_msgs[0]

    def test_pydantic_entity_name_extraction(self, caplog):
        """Entities with .name attribute (Pydantic models) are handled correctly."""

        class FakeEntity:
            name = "Pydantic Hero"

        results = [(FakeEntity(), _make_scores(5.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "character", 7.5, 1.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch character summary" in msg]
        assert "Pydantic Hero" in summary_msgs[0]


class TestGenerateBatchSummary:
    """Tests for batch summary being called from _generate_batch."""

    def test_generate_batch_logs_summary(self, mock_svc, caplog):
        """_generate_batch logs batch summary after completing."""
        entity = {"name": "Test Entity"}
        scores = _make_scores(8.0)

        with caplog.at_level(logging.INFO):
            _generate_batch(
                svc=mock_svc,
                count=1,
                entity_type="faction",
                generate_fn=lambda _i: (entity, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
            )

        assert any("Batch faction summary" in msg for msg in caplog.messages)

    def test_review_batch_logs_summary(self, mock_svc, caplog):
        """_review_batch logs batch summary after completing."""
        entity = {"name": "Test Entity"}
        scores = _make_scores(8.0)

        with caplog.at_level(logging.INFO):
            _review_batch(
                svc=mock_svc,
                entities=[entity],
                entity_type="character",
                review_fn=lambda e: (e, scores, 1),
                get_name=lambda e: e["name"],
                zero_scores_fn=lambda reason: _make_scores(0.0),
                quality_threshold=7.5,
            )

        assert any("Batch character summary" in msg for msg in caplog.messages)
