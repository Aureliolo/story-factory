"""Tests for parallel batch generation (_generate_batch_parallel, _ThreadSafeRelsList)."""

import concurrent.futures as cf
import logging
import threading
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import CharacterQualityScores, RelationshipQualityScores
from src.services.world_quality_service._batch import generate_relationships_with_quality
from src.services.world_quality_service._batch_parallel import (
    _collect_late_results,
    _generate_batch_parallel,
    _ThreadSafeRelsList,
)
from src.utils.exceptions import WorldGenerationError


def _make_char_scores(avg: float, feedback: str = "Test") -> CharacterQualityScores:
    """Create CharacterQualityScores where all dimensions equal avg.

    Used as a generic BaseQualityScores stand-in for testing the type-agnostic
    ``_generate_batch_parallel`` function.
    """
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

        # Snapshot lengths should be monotonically non-decreasing
        # (the underlying list only grows, and the reader is single-threaded)
        snapshot_lengths = [len(s) for s in snapshots]
        for i in range(1, len(snapshot_lengths)):
            assert snapshot_lengths[i] >= snapshot_lengths[i - 1], (
                f"Snapshot lengths not monotonic: {snapshot_lengths}"
            )

    # --- append_if_new_pair tests ---

    def test_append_if_new_pair_adds_new(self):
        """New pair is appended and returns True."""
        safe = _ThreadSafeRelsList([("A", "B", "ally")])
        assert safe.append_if_new_pair(("C", "D", "rival")) is True
        assert len(safe) == 2

    def test_append_if_new_pair_rejects_exact_duplicate(self):
        """Exact same pair (same direction) returns False and does not append."""
        safe = _ThreadSafeRelsList([("A", "B", "ally")])
        assert safe.append_if_new_pair(("A", "B", "rival")) is False
        assert len(safe) == 1

    def test_append_if_new_pair_rejects_reversed_pair(self):
        """Reversed pair (B, A) is treated as duplicate of (A, B)."""
        safe = _ThreadSafeRelsList([("A", "B", "ally")])
        assert safe.append_if_new_pair(("B", "A", "enemy")) is False
        assert len(safe) == 1

    def test_append_if_new_pair_concurrent_dedup(self):
        """Concurrent append_if_new_pair for same pair: exactly one wins."""
        safe = _ThreadSafeRelsList([])
        barrier = threading.Barrier(2)
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            """Try to atomically add the same pair."""
            barrier.wait()
            accepted = safe.append_if_new_pair(("X", "Y", "ally"))
            with lock:
                results.append(accepted)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1
        assert results.count(False) == 1
        assert len(safe) == 1


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

    def test_sequential_on_success_rejection_excludes_entity(self, mock_svc):
        """Sequential path (max_workers=1) excludes entities rejected by on_success."""
        call_count = 0

        def gen_fn(_i):
            """Generate entities."""
            nonlocal call_count
            call_count += 1
            return ({"name": f"E{call_count}"}, _make_char_scores(8.0), 1)

        def reject_second(entity):
            """Reject the second entity."""
            if entity["name"] == "E2":
                raise WorldGenerationError("Duplicate rejected")

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=3,
            entity_type="test",
            generate_fn=gen_fn,
            get_name=lambda e: e["name"],
            on_success=reject_second,
            quality_threshold=7.5,
            max_workers=1,
        )

        # E2 should be excluded; E1 and E3 should be present
        result_names = [r[0]["name"] for r in results]
        assert "E2" not in result_names
        assert "E1" in result_names
        assert "E3" in result_names

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
        cancel_after = 2

        def gen_fn(i: int):
            """Generate an entity."""
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

        # With max_workers=2 and cancel after 2 cancel_check calls:
        # 2 seeded tasks complete, then cancel fires on next _submit_next.
        assert len(results) == 2

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
        lock = threading.Lock()
        call_count = 0

        def mixed_fn(_i):
            """Alternate success/failure with thread-safe counter."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current % 2 == 0:
                raise WorldGenerationError("Intermittent failure")
            return ({"name": f"E{current}"}, _make_char_scores(8.0), 1)

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
        lock = threading.Lock()
        call_count = 0

        def gen_fn(_i):
            """First succeeds, then all fail."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current == 1:
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

        def slow_fn(_i):
            """Use a barrier to prove two tasks run concurrently."""
            barrier.wait()  # Both threads must reach this before continuing
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

    # --- New tests for exception handling (items 1, 3) ---

    def test_cancelled_error_handled_gracefully(self, mock_svc, caplog):
        """CancelledError from a future is handled without crashing the batch."""
        lock = threading.Lock()
        call_count = 0

        def gen_fn(_i):
            """Second call raises CancelledError to simulate a cancelled future."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current == 2:
                raise cf.CancelledError()
            return ({"name": f"E{current}"}, _make_char_scores(8.0), 1)

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=3,
                entity_type="test",
                generate_fn=gen_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        # Should have results from the non-cancelled calls
        assert len(results) >= 1
        # CancelledError should be logged as a warning
        assert any("cancelled" in msg.lower() for msg in caplog.messages)

    def test_unexpected_error_handled_as_failure(self, mock_svc, caplog):
        """Non-WorldGenerationError from generate_fn is caught and logged."""
        lock = threading.Lock()
        call_count = 0

        def gen_fn(_i):
            """Second call raises RuntimeError (unexpected exception)."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current == 2:
                raise RuntimeError("Unexpected internal error")
            return ({"name": f"E{current}"}, _make_char_scores(8.0), 1)

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=4,
                entity_type="test",
                generate_fn=gen_fn,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        # Should have results from the non-failing calls
        assert len(results) >= 1
        # Should log the unexpected error
        assert any("Unexpected error" in msg for msg in caplog.messages)

    def test_on_success_failure_rejects_entity(self, mock_svc, caplog):
        """If on_success raises WorldGenerationError, entity is not in results."""
        lock = threading.Lock()
        call_count = 0

        def gen_fn(_i):
            """Generate entities with unique names."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            return ({"name": f"E{current}"}, _make_char_scores(8.0), 1)

        def rejecting_on_success(entity):
            """Reject entity E2."""
            if entity["name"] == "E2":
                raise WorldGenerationError("Duplicate detected")

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=4,
                entity_type="test",
                generate_fn=gen_fn,
                get_name=lambda e: e["name"],
                on_success=rejecting_on_success,
                quality_threshold=7.5,
                max_workers=2,
            )

        # E2 should NOT be in results (on_success rejected it before results.append)
        result_names = [e["name"] for e, _ in results]
        assert "E2" not in result_names
        # Should still have the other entities
        assert len(results) >= 1

    # --- New test for quality_threshold=None (item 18) ---

    def test_quality_threshold_none_resolves_from_config(self, mock_svc):
        """quality_threshold=None resolves via svc.get_config().get_threshold()."""
        mock_svc.get_config.return_value.get_threshold.return_value = 6.0

        results = _generate_batch_parallel(
            svc=mock_svc,
            count=2,
            entity_type="widget",
            generate_fn=lambda _i: ({"name": "W"}, _make_char_scores(8.0), 1),
            get_name=lambda e: e["name"],
            quality_threshold=None,
            max_workers=2,
        )

        assert len(results) == 2
        mock_svc.get_config.return_value.get_threshold.assert_called_with("widget")

    # --- New test for shuffle/recovery full cycle (item 19) ---

    def test_recovery_then_termination(self, mock_svc, caplog):
        """Recovery resets consecutive failures; second round of failures terminates."""
        from src.services.world_quality_service._batch import MAX_BATCH_SHUFFLE_RETRIES

        def always_fail(_i):
            """Every call fails."""
            raise WorldGenerationError("Persistent failure")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Failed to generate any"):
                _generate_batch_parallel(
                    svc=mock_svc,
                    count=30,
                    entity_type="test",
                    generate_fn=always_fail,
                    get_name=lambda e: e["name"],
                    quality_threshold=7.5,
                    max_workers=2,
                )

        # Should see both recovery and early termination messages
        recovery_msgs = [m for m in caplog.messages if "Attempting recovery" in m]
        termination_msgs = [
            m for m in caplog.messages if "early termination" in m and "consecutive failures" in m
        ]
        assert len(recovery_msgs) == MAX_BATCH_SHUFFLE_RETRIES
        assert len(termination_msgs) == 1

    def test_duplicate_name_error_not_counted_as_failure(self, mock_svc, caplog):
        """DuplicateNameError should be logged as warning, not counted as consecutive failure."""
        from src.utils.exceptions import DuplicateNameError

        call_count = 0
        lock = threading.Lock()

        def fail_then_succeed(i):
            """First call raises DuplicateNameError, rest succeed."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current == 1:
                raise DuplicateNameError("Name already exists")
            return {"name": f"Entity-{i}"}, _make_char_scores(8.5), 1

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=3,
                entity_type="test",
                generate_fn=fail_then_succeed,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
            )

        # Should produce all requested results (replacement tasks submitted for duplicates)
        assert len(results) == 3
        # DuplicateNameError should be logged as a warning, not an error
        duplicate_msgs = [m for m in caplog.messages if "Duplicate" in m]
        assert len(duplicate_msgs) >= 1


# ---------------------------------------------------------------------------
# _collect_late_results tests
# ---------------------------------------------------------------------------


class TestCollectLateResults:
    """Tests for _collect_late_results function."""

    def test_collects_successful_late_results(self):
        """Successful futures are collected into results during early termination."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: ({"name": "Late"}, _make_char_scores(8.0), 1))
            # Wait for the future to complete before testing
            future.result()

            pending: dict = {future: (0, 0.0)}
            results: list = []
            completed_times: list[float] = []
            errors: list[str] = []

            _collect_late_results(
                pending,
                results,
                completed_times,
                errors,
                "test",
                lambda e: e["name"],
                None,
            )

        assert len(results) == 1
        assert results[0][0] == {"name": "Late"}
        assert len(errors) == 0

    def test_skips_cancelled_futures(self):
        """Cancelled futures are skipped without error."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit and immediately cancel (may or may not succeed)
            future = executor.submit(lambda: None)
            future.result()  # Wait for completion first
            # Create a mock cancelled future
            cancelled_future = MagicMock()
            cancelled_future.cancelled.return_value = True

            pending: dict = {cancelled_future: (0, 0.0)}
            results: list = []
            completed_times: list[float] = []
            errors: list[str] = []

            _collect_late_results(
                pending,
                results,
                completed_times,
                errors,
                "test",
                lambda e: e["name"],
                None,
            )

        assert len(results) == 0
        assert len(errors) == 0

    def test_handles_failed_late_futures(self, caplog):
        """Failed late futures are logged and added to errors."""
        from concurrent.futures import ThreadPoolExecutor

        def fail():
            """Raise an error."""
            raise WorldGenerationError("Late failure")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fail)
            # Wait for it to finish (it will store the exception)
            try:
                future.result()
            except WorldGenerationError:
                pass

            pending: dict = {future: (0, 0.0)}
            results: list = []
            completed_times: list[float] = []
            errors: list[str] = []

            with caplog.at_level(logging.DEBUG):
                _collect_late_results(
                    pending,
                    results,
                    completed_times,
                    errors,
                    "test",
                    lambda e: e["name"],
                    None,
                )

        assert len(results) == 0
        assert len(errors) == 1

    def test_on_success_called_for_late_results(self):
        """on_success hook is invoked for each successful late result."""
        from concurrent.futures import ThreadPoolExecutor

        success_calls: list[dict] = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: ({"name": "Late"}, _make_char_scores(8.0), 1))
            future.result()

            pending: dict = {future: (0, 0.0)}
            results: list = []
            completed_times: list[float] = []
            errors: list[str] = []

            _collect_late_results(
                pending,
                results,
                completed_times,
                errors,
                "test",
                lambda e: e["name"],
                lambda e: success_calls.append(e),
            )

        assert len(results) == 1
        assert len(success_calls) == 1
        assert success_calls[0] == {"name": "Late"}


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

        rels = [
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "ally",
                "description": "Friends",
            },
            {
                "source": "Alice",
                "target": "Carol",
                "relation_type": "rival",
                "description": "Rivals",
            },
        ]
        scores = _make_rel_scores(8.0)
        mock_svc.generate_relationship_with_quality = MagicMock(
            side_effect=[(rels[0], scores, 1), (rels[1], scores, 1)]
        )

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

        lock = threading.Lock()
        call_count = 0

        def mock_generate(story, names, rels):
            """Generate unique relationships with thread-safe counter."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            rel = {
                "source": f"E{current}",
                "target": f"F{current}",
                "relation_type": "ally",
                "description": f"Rel {current}",
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

    def test_duplicate_from_parallel_worker_rejected(self, mock_svc, caplog):
        """Duplicate relationship from concurrent worker is rejected by atomic dedup."""
        from src.memory.story_state import StoryState

        story_state = MagicMock(spec=StoryState)
        story_state.id = "test-story"

        # All calls return the same pair — second call should be rejected
        # by append_if_new_pair, third call returns a different pair
        lock = threading.Lock()
        call_count = 0

        def mock_generate(story, names, rels):
            """Return same pair for first two calls, different pair for third."""
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current <= 2:
                return (
                    {
                        "source": "Alice",
                        "target": "Bob",
                        "relation_type": "ally",
                        "description": "Same pair",
                    },
                    _make_rel_scores(8.0),
                    1,
                )
            return (
                {
                    "source": "Alice",
                    "target": "Carol",
                    "relation_type": "rival",
                    "description": "Different pair",
                },
                _make_rel_scores(8.0),
                1,
            )

        mock_svc.generate_relationship_with_quality = mock_generate

        with caplog.at_level(logging.ERROR):
            results = generate_relationships_with_quality(
                svc=mock_svc,
                story_state=story_state,
                entity_names=["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=3,  # count > 2 so a replacement task is submitted after duplicate
            )

        # One duplicate was rejected, but the third call returned a unique pair
        assert len(results) == 2
        # Should have unique pairs in results
        pairs = {(r["source"], r["target"]) for r, _ in results}
        assert len(pairs) == 2
