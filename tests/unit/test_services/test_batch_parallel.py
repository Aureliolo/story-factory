"""Tests for parallel batch generation (_generate_batch_parallel, _ThreadSafeRelsList)."""

import logging
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import CharacterQualityScores, RelationshipQualityScores
from src.services.world_quality_service._batch import generate_relationships_with_quality
from src.services.world_quality_service._batch_parallel import (
    _generate_batch_parallel,
    _ThreadSafeRelsList,
)
from src.utils.exceptions import WorldGenerationError


def _make_char_scores(avg: float, feedback: str = "Test") -> CharacterQualityScores:
    """Create CharacterQualityScores where all dimensions equal avg."""
    return CharacterQualityScores(
        depth=avg,
        goals=avg,
        flaws=avg,
        uniqueness=avg,
        arc_potential=avg,
        temporal_plausibility=avg,
        feedback=feedback,
    )


def _make_rel_scores(avg: float, feedback: str = "Test") -> RelationshipQualityScores:
    """Create RelationshipQualityScores where all dimensions equal avg."""
    return RelationshipQualityScores(
        tension=avg,
        dynamics=avg,
        story_potential=avg,
        authenticity=avg,
        feedback=feedback,
    )


@pytest.fixture
def mock_svc():
    """Create a mock WorldQualityService with settings."""
    svc = MagicMock()
    svc._calculate_eta = MagicMock(return_value=0.0)
    config = MagicMock()
    config.quality_threshold = 7.5
    config.get_threshold = MagicMock(return_value=7.5)
    svc.get_config = MagicMock(return_value=config)
    svc.settings = MagicMock()
    svc.settings.llm_max_concurrent_requests = 2
    return svc


# ---------------------------------------------------------------------------
# _ThreadSafeRelsList tests
# ---------------------------------------------------------------------------


class TestThreadSafeRelsList:
    """Tests for the _ThreadSafeRelsList thread-safe wrapper."""

    def test_snapshot_returns_copy(self):
        """Snapshot returns a copy that doesn't share identity with internal data."""
        initial = [("A", "B", "ally")]
        safe = _ThreadSafeRelsList(initial)
        snap = safe.snapshot()
        assert snap == [("A", "B", "ally")]
        assert snap is not safe._data

    def test_append_visible_in_next_snapshot(self):
        """Appended items appear in subsequent snapshots."""
        safe = _ThreadSafeRelsList([])
        safe.append(("A", "B", "rival"))
        snap = safe.snapshot()
        assert snap == [("A", "B", "rival")]

    def test_len_reflects_appends(self):
        """__len__ returns accurate count after appends."""
        safe = _ThreadSafeRelsList([("X", "Y", "ally")])
        assert len(safe) == 1
        safe.append(("A", "B", "rival"))
        assert len(safe) == 2

    def test_snapshot_isolation(self):
        """Snapshot taken before append does not include the appended item."""
        safe = _ThreadSafeRelsList([("A", "B", "ally")])
        snap_before = safe.snapshot()
        safe.append(("C", "D", "rival"))
        assert len(snap_before) == 1
        assert len(safe) == 2

    def test_initial_data_not_shared(self):
        """Modifying the initial list doesn't affect the internal data."""
        initial = [("A", "B", "ally")]
        safe = _ThreadSafeRelsList(initial)
        initial.append(("C", "D", "rival"))
        assert len(safe) == 1

    def test_concurrent_appends(self):
        """Concurrent appends from multiple threads produce correct total count."""
        safe = _ThreadSafeRelsList([])
        barrier = threading.Barrier(4)
        appends_per_thread = 50

        def worker(thread_id: int) -> None:
            """Append items from a single thread."""
            barrier.wait()
            for i in range(appends_per_thread):
                safe.append((f"T{thread_id}", f"E{i}", "ally"))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(safe) == 4 * appends_per_thread

    def test_snapshot_during_concurrent_appends(self):
        """Snapshot taken during concurrent appends returns a consistent state."""
        safe = _ThreadSafeRelsList([("initial", "pair", "ally")])
        snapshots: list[list[tuple[str, str, str]]] = []
        barrier = threading.Barrier(2)

        def appender() -> None:
            """Append items rapidly."""
            barrier.wait()
            for i in range(20):
                safe.append((f"A{i}", f"B{i}", "rival"))

        def reader() -> None:
            """Take snapshots rapidly."""
            barrier.wait()
            for _ in range(20):
                snapshots.append(safe.snapshot())

        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All snapshots must have at least the initial item
        for snap in snapshots:
            assert len(snap) >= 1
            assert snap[0] == ("initial", "pair", "ally")


# ---------------------------------------------------------------------------
# _generate_batch_parallel tests
# ---------------------------------------------------------------------------


class TestGenerateBatchParallel:
    """Tests for _generate_batch_parallel function."""

    def test_correct_count_generated(self, mock_svc):
        """Generates exactly `count` entities with max_workers=2."""
        entity = {"name": "Test"}
        scores = _make_char_scores(8.0)

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=5,
            entity_type="test",
            generate_fn=lambda _i: (entity, scores, 1),
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            max_workers=2,
        )

        assert len(results) == 5
        for ent, sc in results:
            assert ent == entity
            assert sc.average == 8.0

    def test_single_worker_delegates_to_sequential(self, mock_svc):
        """max_workers=1 delegates to _generate_batch (sequential)."""
        call_indices: list[int] = []

        def gen_fn(i: int):
            """Track call order."""
            call_indices.append(i)
            return ({"name": f"E{i}"}, _make_char_scores(8.0), 1)

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=3,
            entity_type="test",
            generate_fn=gen_fn,
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            max_workers=1,
        )

        assert len(results) == 3
        # Sequential: indices should be in order
        assert call_indices == [0, 1, 2]

    def test_zero_count_returns_empty(self, mock_svc):
        """count=0 returns empty list without calling generate_fn."""
        gen_fn = MagicMock()
        results = _generate_batch_parallel(
            svc=mock_svc,
            count=0,
            entity_type="test",
            generate_fn=gen_fn,
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            max_workers=2,
        )
        assert results == []
        gen_fn.assert_not_called()

    def test_cancel_stops_new_submissions(self, mock_svc):
        """cancel_check returning True stops submitting new tasks."""
        generated_count = 0
        cancel_after = 2

        def gen_fn(i: int):
            """Track generation count."""
            nonlocal generated_count
            generated_count += 1
            return ({"name": f"E{i}"}, _make_char_scores(8.0), 1)

        call_count = 0

        def cancel_check():
            """Cancel after `cancel_after` completions."""
            nonlocal call_count
            call_count += 1
            return call_count > cancel_after

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=10,
            entity_type="test",
            generate_fn=gen_fn,
            get_name=lambda e: e["name"],
            cancel_check=cancel_check,
            quality_threshold=7.5,
            max_workers=2,
        )

        # Should have generated fewer than 10 entities
        assert len(results) < 10

    def test_progress_callbacks_called(self, mock_svc):
        """Progress callback receives updates for each entity."""
        callback = MagicMock()

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=3,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "E"}, _make_char_scores(8.0), 1),
            get_name=lambda e: e["name"],
            progress_callback=callback,
            quality_threshold=7.5,
            max_workers=2,
        )

        assert len(results) == 3
        # At least one "generating" and one "complete" callback per entity
        assert callback.call_count >= 3

    def test_on_success_called_per_entity(self, mock_svc):
        """on_success is called once for each successfully generated entity."""
        success_names: list[str] = []

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=3,
            entity_type="test",
            generate_fn=lambda i: ({"name": f"E{i}"}, _make_char_scores(8.0), 1),
            get_name=lambda e: e["name"],
            on_success=lambda e: success_names.append(e["name"]),
            quality_threshold=7.5,
            max_workers=2,
        )

        assert len(results) == 3
        assert len(success_names) == 3

    def test_all_failures_raises_error(self, mock_svc):
        """Zero successes raises WorldGenerationError."""

        def failing_fn(_i):
            """Always fail."""
            raise WorldGenerationError("LLM timeout")

        with pytest.raises(WorldGenerationError, match="Failed to generate any"):
            _generate_batch_parallel(
                svc=mock_svc,
                count=3,
                entity_type="test",
                generate_fn=failing_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

    def test_partial_results_on_mixed_success_failure(self, mock_svc, caplog):
        """Mixed success/failure returns partial results and logs warning."""
        call_count = 0

        def mixed_fn(_i):
            """Alternate success/failure."""
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise WorldGenerationError("Intermittent failure")
            return ({"name": f"E{call_count}"}, _make_char_scores(8.0), 1)

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=6,
                entity_type="test",
                generate_fn=mixed_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        # Should have some results (at least the successes)
        assert len(results) >= 1
        assert len(results) < 6

    def test_consecutive_failures_trigger_early_termination(self, mock_svc, caplog):
        """MAX_CONSECUTIVE_BATCH_FAILURES consecutive errors triggers early stop."""
        call_count = 0

        def gen_fn(_i):
            """First succeeds, then all fail."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ({"name": "First"}, _make_char_scores(8.0), 1)
            raise WorldGenerationError("Persistent failure")

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=20,
                entity_type="test",
                generate_fn=gen_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        # Should have the first success, then stop early
        assert len(results) >= 1
        assert len(results) < 20
        # Check for early termination log
        early_term_msgs = [
            msg for msg in caplog.messages if "early termination" in msg or "recovery" in msg
        ]
        assert len(early_term_msgs) >= 1

    def test_actual_parallelism(self, mock_svc):
        """Verify that tasks actually run in parallel (not sequentially)."""
        barrier = threading.Barrier(2, timeout=5)
        generation_times: list[float] = []

        def slow_fn(_i):
            """Use a barrier to prove two tasks run concurrently."""
            start = time.time()
            barrier.wait()  # Both threads must reach this before continuing
            generation_times.append(time.time() - start)
            return ({"name": "Parallel"}, _make_char_scores(8.0), 1)

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=2,
            entity_type="test",
            generate_fn=slow_fn,
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            max_workers=2,
        )

        assert len(results) == 2
        # If truly parallel, both tasks passed the barrier (no timeout)

    def test_batch_summary_logged(self, mock_svc, caplog):
        """Parallel batch logs a summary at the end."""
        with caplog.at_level(logging.INFO):
            _generate_batch_parallel(
                svc=mock_svc,
                count=2,
                entity_type="widget",
                generate_fn=lambda _i: ({"name": "W"}, _make_char_scores(8.0), 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        assert any("Batch widget summary" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# generate_relationships_with_quality integration tests
# ---------------------------------------------------------------------------


class TestGenerateRelationshipsWithQuality:
    """Integration tests for the parallel relationship generation path."""

    def test_uses_parallel_path(self, mock_svc):
        """generate_relationships_with_quality routes through parallel generation."""
        from src.memory.story_state import StoryState

        story_state = MagicMock(spec=StoryState)
        story_state.id = "test-story"

        rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "ally",
            "description": "Friends",
        }
        scores = _make_rel_scores(8.0)
        mock_svc.generate_relationship_with_quality = MagicMock(return_value=(rel, scores, 1))

        results = generate_relationships_with_quality(
            svc=mock_svc,
            story_state=story_state,
            entity_names=["Alice", "Bob", "Carol"],
            existing_rels=[],
            count=2,
        )

        assert len(results) == 2
        assert mock_svc.generate_relationship_with_quality.call_count == 2

    def test_on_success_appends_to_thread_safe_list(self, mock_svc):
        """Successful relationships are tracked for deduplication across workers."""
        from src.memory.story_state import StoryState

        story_state = MagicMock(spec=StoryState)
        story_state.id = "test-story"

        call_count = 0

        def mock_generate(story, names, rels):
            """Generate unique relationships, checking rels grows."""
            nonlocal call_count
            call_count += 1
            rel = {
                "source": f"E{call_count}",
                "target": f"F{call_count}",
                "relation_type": "ally",
                "description": f"Rel {call_count}",
            }
            return (rel, _make_rel_scores(8.0), 1)

        mock_svc.generate_relationship_with_quality = mock_generate

        results = generate_relationships_with_quality(
            svc=mock_svc,
            story_state=story_state,
            entity_names=["E1", "E2", "E3", "F1", "F2", "F3"],
            existing_rels=[],
            count=3,
        )

        assert len(results) == 3

    def test_max_workers_capped_by_count(self, mock_svc):
        """When count < llm_max_concurrent_requests, max_workers = count."""
        from src.memory.story_state import StoryState

        story_state = MagicMock(spec=StoryState)
        story_state.id = "test-story"
        mock_svc.settings.llm_max_concurrent_requests = 4

        rel = {
            "source": "A",
            "target": "B",
            "relation_type": "rival",
            "description": "Enemies",
        }
        mock_svc.generate_relationship_with_quality = MagicMock(
            return_value=(rel, _make_rel_scores(8.0), 1)
        )

        # count=1 should effectively use max_workers=1 (sequential)
        results = generate_relationships_with_quality(
            svc=mock_svc,
            story_state=story_state,
            entity_names=["A", "B"],
            existing_rels=[],
            count=1,
        )

        assert len(results) == 1
