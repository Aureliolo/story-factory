"""Tests for batch generation/review summary logging (#303, #328)."""

import logging
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import CharacterQualityScores
from src.services.world_quality_service._batch import (
    _aggregate_errors,
    _generate_batch,
    _log_batch_summary,
    _review_batch,
)
from src.utils.exceptions import WorldGenerationError


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
            """Stub entity with a .name attribute."""

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


class TestLogBatchSummaryGetName:
    """Tests for get_name parameter in _log_batch_summary (#328)."""

    def test_get_name_used_for_failed_entities(self, caplog):
        """When get_name is provided, it is used instead of hardcoded name extraction."""

        # Simulate a chapter entity (has .number + .title, no .name)
        class FakeChapter:
            """Stub chapter entity without a .name attribute."""

            number = 3
            title = "The Betrayal"

        results = [(FakeChapter(), _make_scores(5.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(
                results,
                "chapter",
                7.5,
                1.0,
                get_name=lambda ch: f"Ch{ch.number}: {ch.title}",
            )

        summary_msgs = [msg for msg in caplog.messages if "Batch chapter summary" in msg]
        assert len(summary_msgs) == 1
        assert "Ch3: The Betrayal" in summary_msgs[0]
        assert "Unknown" not in summary_msgs[0]

    def test_relationship_get_name(self, caplog):
        """Relationship entities use source->target via get_name."""
        rel = {"source": "Alice", "target": "Bob", "type": "ally"}
        results = [(rel, _make_scores(4.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(
                results,
                "relationship",
                7.5,
                1.0,
                get_name=lambda r: f"{r['source']} -> {r['target']}",
            )

        summary_msgs = [msg for msg in caplog.messages if "Batch relationship summary" in msg]
        assert "Alice -> Bob" in summary_msgs[0]

    def test_fallback_without_get_name(self, caplog):
        """Without get_name, dict entities fall back to entity.get('name')."""
        results = [({"name": "Fallback Hero"}, _make_scores(5.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "character", 7.5, 1.0)

        summary_msgs = [msg for msg in caplog.messages if "Batch character summary" in msg]
        assert "Fallback Hero" in summary_msgs[0]

    def test_get_name_not_called_for_passing_entities(self, caplog):
        """get_name is only called for entities below threshold."""
        call_count = 0

        def tracking_get_name(entity):
            """Track get_name calls."""
            nonlocal call_count
            call_count += 1
            return entity["name"]

        results = [({"name": "Good"}, _make_scores(9.0))]
        with caplog.at_level(logging.INFO):
            _log_batch_summary(results, "character", 7.5, 1.0, get_name=tracking_get_name)

        assert call_count == 0  # Not called for passing entities


class TestAggregateErrors:
    """Tests for _aggregate_errors deduplication (#328 B2)."""

    def test_no_duplicates_unchanged(self):
        """Unique errors are joined normally."""
        errors = ["Error A", "Error B", "Error C"]
        result = _aggregate_errors(errors)
        assert result == "Error A; Error B; Error C"

    def test_duplicates_aggregated(self):
        """Repeated errors get (xN) suffix."""
        errors = ["Same error"] * 5
        result = _aggregate_errors(errors)
        assert result == "Same error (x5)"

    def test_mixed_duplicates(self):
        """Mix of unique and repeated errors."""
        errors = ["Error A", "Error B", "Error A", "Error B", "Error A"]
        result = _aggregate_errors(errors)
        assert "Error A (x3)" in result
        assert "Error B (x2)" in result

    def test_single_error(self):
        """Single error has no count suffix."""
        result = _aggregate_errors(["Only one"])
        assert result == "Only one"

    def test_empty_list(self):
        """Empty error list returns empty string."""
        result = _aggregate_errors([])
        assert result == ""


class TestGenerateBatchAggregatedErrors:
    """Tests for aggregated error logging in _generate_batch (#328 B2)."""

    def test_generate_batch_aggregates_repeated_errors(self, mock_svc, caplog):
        """_generate_batch aggregates identical error messages."""
        call_count = 0

        def failing_generate_fn(_i):
            """Fail with same message each time."""
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise WorldGenerationError("Failed to generate relationship after 3 attempts")
            return ({"name": "Valid"}, _make_scores(8.0), 1)

        with caplog.at_level(logging.WARNING):
            _generate_batch(
                svc=mock_svc,
                count=3,
                entity_type="relationship",
                generate_fn=failing_generate_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
            )

        warning_msgs = [msg for msg in caplog.messages if "failed:" in msg.lower()]
        assert len(warning_msgs) == 1
        assert "(x2)" in warning_msgs[0]

    def test_review_batch_aggregates_repeated_errors(self, mock_svc, caplog):
        """_review_batch aggregates identical error messages."""
        entities = [{"name": f"Entity {i}"} for i in range(3)]
        call_count = 0

        def failing_review_fn(entity):
            """Fail with same message for first two entities."""
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise WorldGenerationError("LLM timeout after 60s")
            return (entity, _make_scores(8.0), 1)

        with caplog.at_level(logging.WARNING):
            _review_batch(
                svc=mock_svc,
                entities=entities,
                entity_type="character",
                review_fn=failing_review_fn,
                get_name=lambda e: e["name"],
                zero_scores_fn=lambda msg: _make_scores(0.0),
                quality_threshold=7.5,
            )

        warning_msgs = [msg for msg in caplog.messages if "failed:" in msg.lower()]
        assert len(warning_msgs) == 1
        assert "(x2)" in warning_msgs[0]
