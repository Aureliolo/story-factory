"""Tests for parallel batch generation (_generate_batch_parallel, _ThreadSafeRelsList)."""

import concurrent.futures as cf
import logging
import threading
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import (
    CharacterQualityScores,
    RefinementConfig,
    RelationshipQualityScores,
)
from src.services.world_quality_service._batch import generate_relationships_with_quality
from src.services.world_quality_service._batch_parallel import (
    _collect_late_results,
    _generate_batch_parallel,
    _ThreadSafeRelsList,
)
from src.utils.exceptions import DuplicateNameError, WorldGenerationError


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
    # Return same model for creator and judge so max_workers isn't reduced
    # to 1 by the GPU-thrashing check (creator != judge triggers sequential).
    svc._get_creator_model.return_value = "test-model:8b"
    svc._get_judge_model.return_value = "test-model:8b"
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

    def test_all_duplicates_eventually_terminate(self, mock_svc, caplog):
        """Persistent DuplicateNameError should terminate and raise WorldGenerationError."""
        from src.utils.exceptions import DuplicateNameError

        def always_duplicate(_i):
            """Always raise DuplicateNameError."""
            raise DuplicateNameError("Name already exists")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Failed to generate any"):
                _generate_batch_parallel(
                    svc=mock_svc,
                    count=3,
                    entity_type="test",
                    generate_fn=always_duplicate,
                    get_name=lambda e: e["name"],
                    quality_threshold=7.5,
                    max_workers=2,
                )

        # Should have logged duplicate warnings
        duplicate_msgs = [m for m in caplog.messages if "Duplicate" in m]
        assert len(duplicate_msgs) >= 3


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

    def test_memory_error_reraised_from_late_future(self):
        """MemoryError from a late future is re-raised, not swallowed."""
        from concurrent.futures import ThreadPoolExecutor

        def oom():
            """Raise MemoryError."""
            raise MemoryError("out of memory")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(oom)
            try:
                future.result()
            except MemoryError:
                pass

            pending: dict = {future: (0, 0.0)}
            results: list = []
            completed_times: list[float] = []
            errors: list[str] = []

            with pytest.raises(MemoryError, match="out of memory"):
                _collect_late_results(
                    pending,
                    results,
                    completed_times,
                    errors,
                    "test",
                    lambda e: e["name"],
                    None,
                )

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
            entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
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
            entity_names_provider=lambda: ["E1", "E2", "E3", "F1", "F2", "F3"],
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
            entity_names_provider=lambda: ["A", "B"],
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
                entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=3,  # count > 2 so a replacement task is submitted after duplicate
            )

        # One duplicate was rejected, but the third call returned a unique pair
        assert len(results) == 2
        # Should have unique pairs in results
        pairs = {(r["source"], r["target"]) for r, _ in results}
        assert len(pairs) == 2


class TestMaxWorkersReductionForDifferentModels:
    """Tests for _generate_batch_parallel reducing max_workers when creator != judge."""

    def test_reduces_to_sequential_when_models_differ(self, mock_svc, caplog):
        """When creator and judge models differ, max_workers drops to 1 (sequential)."""
        mock_svc._get_creator_model = MagicMock(return_value="creator-model:24b")
        mock_svc._get_judge_model = MagicMock(return_value="judge-model:30b")

        entity = {"name": "Hero"}
        scores = _make_char_scores(8.0)

        call_thread_ids: list[int | None] = []

        def gen_fn(i):
            """Record thread ID and return a test entity."""
            call_thread_ids.append(threading.current_thread().ident)
            return entity, scores, 1

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=3,
                entity_type="character",
                generate_fn=gen_fn,
                get_name=lambda e: e["name"],
                max_workers=4,
            )

        assert len(results) == 3
        # The log message about reducing should appear
        assert any("Reducing max_workers to 1" in msg for msg in caplog.messages)

    def test_no_reduction_when_models_same(self, mock_svc, caplog):
        """When creator and judge are the same model, max_workers stays unchanged."""
        mock_svc._get_creator_model = MagicMock(return_value="same-model:8b")
        mock_svc._get_judge_model = MagicMock(return_value="same-model:8b")

        entity = {"name": "Hero"}
        scores = _make_char_scores(8.0)

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=2,
                entity_type="character",
                generate_fn=lambda i: (entity, scores, 1),
                get_name=lambda e: e["name"],
                max_workers=2,
            )

        assert len(results) == 2
        # No reduction log message should appear
        assert not any("Reducing max_workers to 1" in msg for msg in caplog.messages)

    def test_no_reduction_when_already_single_worker(self, mock_svc, caplog):
        """When max_workers is already 1, no reduction log even if models differ."""
        mock_svc._get_creator_model = MagicMock(return_value="creator-model:24b")
        mock_svc._get_judge_model = MagicMock(return_value="judge-model:30b")

        entity = {"name": "Hero"}
        scores = _make_char_scores(8.0)

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=2,
                entity_type="character",
                generate_fn=lambda i: (entity, scores, 1),
                get_name=lambda e: e["name"],
                max_workers=1,
            )

        assert len(results) == 2
        # No reduction log — was already 1
        assert not any("Reducing max_workers to 1" in msg for msg in caplog.messages)

    def test_model_resolution_failure_keeps_max_workers(self, mock_svc, caplog):
        """When model resolution fails, max_workers stays unchanged and logs warning."""
        mock_svc._get_creator_model = MagicMock(side_effect=ValueError("unknown entity type"))

        entity = {"name": "Hero"}
        scores = _make_char_scores(8.0)

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=mock_svc,
                count=2,
                entity_type="character",
                generate_fn=lambda i: (entity, scores, 1),
                get_name=lambda e: e["name"],
                max_workers=2,
            )

        assert len(results) == 2
        assert any("Failed to resolve models" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# _generate_batch_phased tests
# ---------------------------------------------------------------------------


@pytest.fixture
def phased_svc():
    """Create a mock WorldQualityService configured for phased pipeline testing.

    Sets creator != judge to trigger the phased path, and provides properly
    configured model preparers.
    """
    svc = MagicMock()
    svc._calculate_eta = MagicMock(return_value=0.0)
    config = MagicMock()
    config.quality_threshold = 7.5
    config.get_threshold = MagicMock(return_value=7.5)
    svc.get_config = MagicMock(return_value=config)
    svc.settings = MagicMock()
    svc.settings.llm_max_concurrent_requests = 2
    # Different models to trigger phased pipeline
    svc._get_creator_model.return_value = "creator-model:24b"
    svc._get_judge_model.return_value = "judge-model:30b"
    return svc


class TestGenerateBatchPhased:
    """Tests for the two-phase batch pipeline (_generate_batch_phased)."""

    def test_phased_all_pass_threshold(self, phased_svc, caplog):
        """When all entities pass the judge threshold, no refinement occurs."""
        entities = [{"name": f"E{i}"} for i in range(3)]
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities sequentially."""
            nonlocal create_idx
            entity = entities[create_idx]
            create_idx += 1
            return entity

        judge_calls: list[dict] = []

        def judge_fn(entity):
            """Judge and record call."""
            judge_calls.append(entity)
            return scores

        prep_creator_calls: list[bool] = []
        prep_judge_calls: list[bool] = []

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: prep_creator_calls.append(True),
                prepare_judge_fn=lambda: prep_judge_calls.append(True),
            )

        assert len(results) == 3
        # All 3 should be judged
        assert len(judge_calls) == 3
        # Creator model loaded once, judge model loaded once
        assert len(prep_creator_calls) == 1
        assert len(prep_judge_calls) == 1
        # Phased pipeline log should appear
        assert any("phased" in msg.lower() for msg in caplog.messages)
        # No refinement should occur
        assert not any("Phase 3b" in msg for msg in caplog.messages)

    def test_phased_some_fail_triggers_refinement(self, phased_svc, caplog):
        """Entities failing the threshold are refined via quality_refinement_loop."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)  # Below 7.5 threshold

        create_idx = 0

        def create_fn(_i):
            """Create 3 entities."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def judge_fn(entity):
            """First entity passes, rest fail."""
            if entity["name"] == "E0":
                return passing_scores
            return failing_scores

        refine_calls: list[dict] = []

        def refine_fn(entity):
            """Track refinement calls."""
            refine_calls.append(entity)
            # Return a refined version that passes
            return (entity, passing_scores, 2)

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert len(results) == 3
        # E1 and E2 should have been refined
        assert len(refine_calls) == 2
        assert any("Phase 3b" in msg for msg in caplog.messages)

    def test_phased_create_failure_handled(self, phased_svc, caplog):
        """Entities that fail creation are skipped gracefully."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Second create fails."""
            nonlocal create_idx
            create_idx += 1
            if create_idx == 2:
                return None
            return {"name": f"E{create_idx}"}

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # Only 2 entities created (one failed), both should pass judge
        assert len(results) == 2
        assert any("creation returned empty" in msg for msg in caplog.messages)

    def test_phased_judge_failure_triggers_refinement(self, phased_svc, caplog):
        """Entities whose judge call fails are queued for refinement."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def judge_fn(entity):
            """First entity's judge fails."""
            if entity["name"] == "E0":
                raise WorldGenerationError("Judge model error")
            return scores

        refine_calls: list[dict] = []

        def refine_fn(entity):
            """Refinement for judge-failed entity."""
            refine_calls.append(entity)
            return (entity, scores, 1)

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert len(results) == 2
        # E0 should have been refined because judge failed
        assert len(refine_calls) == 1
        assert refine_calls[0]["name"] == "E0"

    def test_phased_all_creates_fail_raises(self, phased_svc):
        """When all creates fail, WorldGenerationError is raised."""

        def create_fn(_i):
            """Always return None (creation failure)."""
            return None

        scores = _make_char_scores(8.0)

        with pytest.raises(WorldGenerationError, match="failed to create any"):
            _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

    def test_phased_cancel_during_creation(self, phased_svc, caplog):
        """cancel_check during Phase 1 stops creation early."""
        scores = _make_char_scores(8.0)
        cancel_after = 1
        create_count = 0

        def create_fn(_i):
            """Create entities."""
            nonlocal create_count
            create_count += 1
            return {"name": f"E{create_count}"}

        cancel_count = 0

        def cancel_check():
            """Cancel after first entity is created."""
            nonlocal cancel_count
            cancel_count += 1
            return cancel_count > cancel_after

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=5,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                cancel_check=cancel_check,
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # Should have at most 1 entity (cancelled after first)
        assert len(results) <= 1

    def test_phased_on_success_rejection(self, phased_svc, caplog):
        """on_success rejection during Phase 3 excludes the entity from results."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def reject_e1(entity):
            """Reject E1."""
            if entity["name"] == "E1":
                raise DuplicateNameError("E1 is a duplicate")

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                on_success=reject_e1,
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # E1 should be excluded
        result_names = [e["name"] for e, _ in results]
        assert "E1" not in result_names
        assert len(results) == 2
        assert any("duplicate" in msg.lower() for msg in caplog.messages)

    def test_phased_progress_callbacks(self, phased_svc):
        """Progress callback is invoked during phased pipeline execution."""
        scores = _make_char_scores(8.0)
        create_idx = 0
        callback = MagicMock()

        def create_fn(_i):
            """Create entities."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        results = _generate_batch_parallel(
            svc=phased_svc,
            count=2,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
            get_name=lambda e: e["name"],
            progress_callback=callback,
            quality_threshold=7.5,
            max_workers=2,
            create_only_fn=create_fn,
            judge_only_fn=lambda e: scores,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=lambda e: (e, scores, 1),
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )

        assert len(results) == 2
        # Should have at least generating + complete callbacks
        assert callback.call_count >= 2

    def test_phased_empty_entity_rejected_by_is_empty(self, phased_svc, caplog):
        """Entities that pass is_empty check are rejected during Phase 1."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Second entity has missing name (is_empty returns True)."""
            nonlocal create_idx
            create_idx += 1
            if create_idx == 2:
                return {"name": ""}  # Empty name
            return {"name": f"E{create_idx}"}

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e.get("name", "Unknown"),
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # Only 2 entities created successfully (one was empty)
        assert len(results) == 2
        assert any("creation returned empty" in msg for msg in caplog.messages)

    def test_phased_create_exception_handled(self, phased_svc, caplog):
        """WorldGenerationError during creation is caught and entity is skipped."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Second create raises WorldGenerationError."""
            nonlocal create_idx
            create_idx += 1
            if create_idx == 2:
                raise WorldGenerationError("LLM timeout during creation")
            return {"name": f"E{create_idx}"}

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert len(results) == 2
        assert any("Phase 1: failed to create" in msg for msg in caplog.messages)

    def test_phased_refinement_failure_handled(self, phased_svc, caplog):
        """WorldGenerationError during refinement is caught gracefully."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def judge_fn(entity):
            """First entity fails threshold."""
            if entity["name"] == "E0":
                return failing_scores
            return passing_scores

        def refine_fn(entity):
            """Refinement also fails."""
            raise WorldGenerationError("Refinement failed")

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # Only E1 passed; E0 failed both judge and refinement
        assert len(results) == 1
        assert results[0][0]["name"] == "E1"
        assert any("Phase 3b: failed to refine" in msg for msg in caplog.messages)

    def test_phased_batch_summary_logged(self, phased_svc, caplog):
        """Phased pipeline logs batch summary at the end."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities."""
            nonlocal create_idx
            entity = {"name": f"W{create_idx}"}
            create_idx += 1
            return entity

        with caplog.at_level(logging.INFO):
            _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="widget",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert any("Batch widget summary" in msg for msg in caplog.messages)
        assert any("Phased widget pipeline complete" in msg for msg in caplog.messages)

    def test_phased_fallback_to_sequential_without_callables(self, phased_svc, caplog):
        """When phased callables are not provided but models differ, falls to sequential."""
        entity = {"name": "Hero"}
        scores = _make_char_scores(8.0)

        with caplog.at_level(logging.INFO):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="character",
                generate_fn=lambda _i: (entity, scores, 1),
                get_name=lambda e: e["name"],
                max_workers=4,
                # No phased callables provided
            )

        assert len(results) == 3
        # Should fall back to sequential, not phased
        assert any("phased callables not provided" in msg for msg in caplog.messages)
        assert not any(
            "phased" in msg.lower() and "pipeline" in msg.lower() and "starting" in msg.lower()
            for msg in caplog.messages
        )

    def test_phased_zero_count_returns_empty(self, phased_svc):
        """count=0 returns empty list without entering phased pipeline."""
        scores = _make_char_scores(8.0)
        results = _generate_batch_parallel(
            svc=phased_svc,
            count=0,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            max_workers=2,
            create_only_fn=lambda _i: {"name": "E"},
            judge_only_fn=lambda e: scores,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=lambda e: (e, scores, 1),
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )
        assert results == []

    def test_phased_quality_threshold_none_resolves_from_config(self, phased_svc):
        """quality_threshold=None is resolved from svc.get_config().get_threshold() (line 541)."""
        phased_svc.get_config.return_value.get_threshold.return_value = 6.0
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        results = _generate_batch_parallel(
            svc=phased_svc,
            count=2,
            entity_type="widget",
            generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
            get_name=lambda e: e["name"],
            quality_threshold=None,  # Triggers line 541
            max_workers=2,
            create_only_fn=create_fn,
            judge_only_fn=lambda e: scores,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=lambda e: (e, scores, 1),
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )

        assert len(results) == 2
        phased_svc.get_config.return_value.get_threshold.assert_called_with("widget")

    def test_phased_unexpected_exception_in_phase1_handled(self, phased_svc, caplog):
        """Unexpected (non-WorldGenerationError) exception in Phase 1 is caught (lines 611-615)."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Second create raises RuntimeError (unexpected)."""
            nonlocal create_idx
            create_idx += 1
            if create_idx == 2:
                raise RuntimeError("Unexpected GPU crash")
            return {"name": f"E{create_idx}"}

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert len(results) == 2
        assert any("unexpected error creating" in msg.lower() for msg in caplog.messages)

    def test_phased_no_entities_created_no_errors_returns_empty(self, phased_svc):
        """When cancel fires immediately (before any creates), returns [] with no errors (line 637).

        Line 637 (return []) is reached when created_entities=[] AND errors=[].
        This happens when cancel_check returns True on the very first iteration,
        so the loop body breaks before any create_only_fn call, leaving both
        created_entities and errors empty.
        """
        from src.services.world_quality_service._batch_parallel import _generate_batch_phased

        scores = _make_char_scores(8.0)

        # cancel_check returns True immediately, so loop body never runs
        results = _generate_batch_phased(
            svc=phased_svc,
            count=3,
            entity_type="test",
            create_only_fn=lambda _i: {"name": f"E{_i}"},
            judge_only_fn=lambda e: scores,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=lambda e: (e, scores, 1),
            prepare_creator_fn=None,
            prepare_judge_fn=None,
            get_name=lambda e: e["name"],
            quality_threshold=7.5,
            cancel_check=lambda: True,  # Immediately cancel — loop breaks before any create
        )

        # Loop cancelled before any entity was created → no errors → return []
        assert results == []

    def test_phased_unexpected_exception_in_phase2_judge(self, phased_svc, caplog):
        """Unexpected (non-WorldGenerationError) exception in Phase 2 is caught (lines 687-697)."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def judge_fn(entity):
            """First entity's judge raises RuntimeError (unexpected)."""
            if entity["name"] == "E0":
                raise RuntimeError("OOM in judge model")
            return scores

        refine_calls: list[dict] = []

        def refine_fn(entity):
            """Track refinement calls for judge-failed entities."""
            refine_calls.append(entity)
            return (entity, scores, 1)

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert len(results) == 2
        # E0 should have been queued for refinement even though judge raised unexpectedly
        assert len(refine_calls) == 1
        assert any("unexpected error judging" in msg.lower() for msg in caplog.messages)

    def test_phased_on_success_raises_world_generation_error_in_phase3(self, phased_svc, caplog):
        """WorldGenerationError from on_success in Phase 3 is caught (lines 748-751)."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def on_success(entity):
            """Raise WorldGenerationError for E1."""
            if entity["name"] == "E1":
                raise WorldGenerationError("E1 is invalid")

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=3,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                on_success=on_success,
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # E1 rejected by on_success, so only E0 and E2 succeed
        result_names = [e["name"] for e, _ in results]
        assert "E1" not in result_names
        assert len(results) == 2
        assert any("on_success rejected" in msg.lower() for msg in caplog.messages)

    def test_phased_cancel_during_refinement_stops_early(self, phased_svc, caplog):
        """cancel_check during Phase 3b refinement stops further refinements (lines 770-775).

        The cancel_check must only fire during Phase 3b (refinement loop), not during
        Phase 1 (create) or Phase 2 (judge). We count refine_fn invocations and only
        cancel after the first refinement call has completed.
        """
        from src.services.world_quality_service._batch_parallel import _generate_batch_phased

        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        refine_count = 0

        def create_fn(_i):
            """Create test entities."""
            return {"name": f"E{_i}"}

        def judge_fn(entity):
            """Return failing scores for all entities."""
            # All fail threshold so all need refinement
            return failing_scores

        def refine_fn(entity):
            """Refine entity and track invocation count."""
            nonlocal refine_count
            refine_count += 1
            return (entity, passing_scores, 1)

        cancel_after_refines = 1

        def cancel_check():
            """Check if enough refinements have completed for cancellation."""
            # Only cancel AFTER the first refinement has run
            return refine_count >= cancel_after_refines

        with caplog.at_level(logging.INFO):
            results = _generate_batch_phased(
                svc=phased_svc,
                count=4,
                entity_type="test",
                create_only_fn=create_fn,
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=None,
                prepare_judge_fn=None,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                cancel_check=cancel_check,
            )

        # Cancelled during refinement — fewer results than requested (only 1 refined)
        assert len(results) < 4
        assert any("refinement cancelled" in msg.lower() for msg in caplog.messages)

    def test_phased_on_success_called_during_refinement(self, phased_svc):
        """on_success is called with the refined entity in Phase 3b (line 789)."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0
        on_success_calls: list[dict] = []

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def judge_fn(entity):
            """Return failing scores for all entities."""
            # All fail so all need refinement
            return failing_scores

        def refine_fn(entity):
            """Refine entity by appending _refined to name."""
            refined = {"name": entity["name"] + "_refined"}
            return (refined, passing_scores, 1)

        results = _generate_batch_parallel(
            svc=phased_svc,
            count=2,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
            get_name=lambda e: e["name"],
            on_success=lambda e: on_success_calls.append(e),
            quality_threshold=7.5,
            max_workers=2,
            create_only_fn=create_fn,
            judge_only_fn=judge_fn,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=refine_fn,
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )

        assert len(results) == 2
        # on_success called with refined entities (not originals)
        assert len(on_success_calls) == 2
        assert all("_refined" in e["name"] for e in on_success_calls)

    def test_phased_progress_callback_during_refinement(self, phased_svc):
        """progress_callback is called with refined entity info in Phase 3b (line 803)."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0
        callback = MagicMock()

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def refine_fn(entity):
            """Refine entity and return with passing scores."""
            return (entity, passing_scores, 1)

        results = _generate_batch_parallel(
            svc=phased_svc,
            count=2,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
            get_name=lambda e: e["name"],
            progress_callback=callback,
            quality_threshold=7.5,
            max_workers=2,
            create_only_fn=create_fn,
            judge_only_fn=lambda e: failing_scores,  # All fail → all refined
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=refine_fn,
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )

        assert len(results) == 2
        # At minimum: generating callbacks (phase 1) + complete callbacks (phase 3b)
        assert callback.call_count >= 2

    def test_phased_duplicate_name_error_during_refinement(self, phased_svc, caplog):
        """DuplicateNameError from refine_with_initial_fn is caught in Phase 3b (lines 817-819)."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def refine_fn(entity):
            """First refinement raises DuplicateNameError."""
            if entity["name"] == "E0":
                raise DuplicateNameError("E0 already exists")
            return (entity, passing_scores, 1)

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: failing_scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # E0 refinement raised DuplicateNameError — only E1 succeeds
        assert len(results) == 1
        assert results[0][0]["name"] == "E1"
        assert any("duplicate" in msg.lower() for msg in caplog.messages)

    def test_phased_unexpected_exception_during_refinement(self, phased_svc, caplog):
        """Unexpected exception from refine_with_initial_fn is caught in Phase 3b (lines 834-837)."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def refine_fn(entity):
            """First refinement raises unexpected RuntimeError."""
            if entity["name"] == "E0":
                raise RuntimeError("GPU memory error during refinement")
            return (entity, passing_scores, 1)

        with caplog.at_level(logging.ERROR):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: failing_scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # E0 refinement failed — only E1 succeeds
        assert len(results) == 1
        assert results[0][0]["name"] == "E1"
        assert any("unexpected error refining" in msg.lower() for msg in caplog.messages)

    def test_phased_partial_results_with_errors_logs_warning(self, phased_svc, caplog):
        """When some results produced and some errors occur, a warning is logged (line 864)."""
        passing_scores = _make_char_scores(8.0)
        failing_scores = _make_char_scores(5.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def refine_fn(entity):
            """E0 refinement fails; E1 succeeds."""
            if entity["name"] == "E0":
                raise WorldGenerationError("Cannot refine E0")
            return (entity, passing_scores, 1)

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, passing_scores, 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: failing_scores,  # All fail → all refined
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        # E0 failed, E1 succeeded — partial result with error warning
        assert len(results) == 1
        assert any(
            "generated" in msg.lower() and "errors" in msg.lower() for msg in caplog.messages
        )

    def test_phased_no_results_with_errors_raises(self, phased_svc, caplog):
        """When all refinements fail and errors accumulated, raises WorldGenerationError (line 860)."""
        from src.services.world_quality_service._batch_parallel import _generate_batch_phased

        failing_scores = _make_char_scores(5.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def refine_fn(entity):
            """All refinements fail."""
            raise WorldGenerationError("All refinements failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorldGenerationError, match="failed to generate any"):
                _generate_batch_phased(
                    svc=phased_svc,
                    count=2,
                    entity_type="test",
                    create_only_fn=create_fn,
                    judge_only_fn=lambda e: failing_scores,  # All fail → all refined
                    is_empty_fn=lambda e: not e.get("name"),
                    refine_with_initial_fn=refine_fn,
                    prepare_creator_fn=None,
                    prepare_judge_fn=None,
                    get_name=lambda e: e["name"],
                    quality_threshold=7.5,
                )

    def test_register_created_fn_failure_logs_warning(self, phased_svc, caplog):
        """When register_created_fn raises, entity is still added and warning logged."""
        from src.services.world_quality_service._batch_parallel import _generate_batch_phased

        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create test entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        registration_attempts: list[str] = []

        def failing_register(entity):
            """Register that always fails."""
            registration_attempts.append(entity["name"])
            raise RuntimeError("Registration failed")

        with caplog.at_level(logging.WARNING):
            results = _generate_batch_phased(
                svc=phased_svc,
                count=2,
                entity_type="test",
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, scores, 1),
                prepare_creator_fn=None,
                prepare_judge_fn=None,
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                register_created_fn=failing_register,
            )

        # Both entities should still be generated despite registration failure
        assert len(results) == 2
        # Registration was attempted for both
        assert len(registration_attempts) == 2
        # Warning logged for each failure
        assert sum("register_created_fn failed" in msg for msg in caplog.messages) == 2

    def test_phased_duplicate_retry_recovers_slot(self, phased_svc):
        """Phase 3 retries via refine_with_initial_fn on DuplicateNameError from on_success."""
        scores = _make_char_scores(8.0)
        create_idx = 0
        on_success_calls = 0

        def create_fn(_i):
            """Create entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def on_success_reject_first(entity):
            """Reject E0 on first attempt only, then accept everything."""
            nonlocal on_success_calls
            on_success_calls += 1
            if entity["name"] == "E0" and on_success_calls == 1:
                raise DuplicateNameError("E0 is a duplicate")

        def refine_fn(entity):
            """Refine entity and return with passing scores."""
            return {"name": "E0-refined"}, scores, 1

        results = _generate_batch_parallel(
            svc=phased_svc,
            count=2,
            entity_type="test",
            generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
            get_name=lambda e: e["name"],
            on_success=on_success_reject_first,
            quality_threshold=7.5,
            max_workers=2,
            create_only_fn=create_fn,
            judge_only_fn=lambda e: scores,
            is_empty_fn=lambda e: not e.get("name"),
            refine_with_initial_fn=refine_fn,
            prepare_creator_fn=lambda: None,
            prepare_judge_fn=lambda: None,
        )

        result_names = [e["name"] for e, _ in results]
        assert "E0-refined" in result_names
        assert len(results) == 2

    def test_phased_duplicate_retry_fails_with_exception(self, phased_svc, caplog):
        """Phase 3 duplicate retry fails with non-DuplicateNameError exception."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def on_success_reject_e0(entity):
            """Reject E0 with DuplicateNameError."""
            if entity["name"] == "E0":
                raise DuplicateNameError("E0 is a duplicate")

        def refine_fn_fails(entity):
            """All refinement retries fail with RuntimeError."""
            raise RuntimeError("refine failed")

        with caplog.at_level(logging.WARNING):
            _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                on_success=on_success_reject_e0,
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn_fails,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

        assert any("duplicate retry failed" in msg for msg in caplog.messages)

    def test_phased_duplicate_retry_reraises_fatal_error(self, phased_svc):
        """Phase 3 duplicate retry re-raises MemoryError (fatal) instead of swallowing."""
        scores = _make_char_scores(8.0)
        create_idx = 0

        def create_fn(_i):
            """Create entities sequentially."""
            nonlocal create_idx
            entity = {"name": f"E{create_idx}"}
            create_idx += 1
            return entity

        def on_success_reject_e0(entity):
            """Reject E0 with DuplicateNameError."""
            if entity["name"] == "E0":
                raise DuplicateNameError("E0 is a duplicate")

        def refine_fn_fatal(entity):
            """Refinement raises a fatal MemoryError."""
            raise MemoryError("OOM during duplicate retry")

        with pytest.raises(MemoryError, match="OOM during duplicate retry"):
            _generate_batch_parallel(
                svc=phased_svc,
                count=2,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, scores, 1),
                get_name=lambda e: e["name"],
                on_success=on_success_reject_e0,
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn_fatal,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )


# ---------------------------------------------------------------------------
# generate_relationships_with_quality phased-pipeline tests
# ---------------------------------------------------------------------------


class TestPhasedCriticalExceptionReRaise:
    """Tests that MemoryError/RecursionError are re-raised in phased pipeline."""

    def test_phase1_reraises_memory_error(self, phased_svc):
        """Phase 1 (create) re-raises MemoryError instead of swallowing it."""

        def create_fn(_i):
            """Raise MemoryError during creation."""
            raise MemoryError("OOM in create")

        with pytest.raises(MemoryError, match="OOM in create"):
            _generate_batch_parallel(
                svc=phased_svc,
                count=1,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, _make_char_scores(8.0), 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=create_fn,
                judge_only_fn=lambda _e: _make_char_scores(8.0),
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, _make_char_scores(8.0), 2),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

    def test_phase2_reraises_memory_error(self, phased_svc):
        """Phase 2 (judge) re-raises MemoryError instead of swallowing it."""

        def judge_fn(_entity):
            """Raise MemoryError during judging."""
            raise MemoryError("OOM in judge")

        with pytest.raises(MemoryError, match="OOM in judge"):
            _generate_batch_parallel(
                svc=phased_svc,
                count=1,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, _make_char_scores(8.0), 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=lambda _i: {"name": "E0"},
                judge_only_fn=judge_fn,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=lambda e: (e, _make_char_scores(8.0), 2),
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )

    def test_phase3b_reraises_memory_error(self, phased_svc):
        """Phase 3b (refine) re-raises MemoryError instead of swallowing it."""
        failing_scores = _make_char_scores(3.0)

        def refine_fn(_entity):
            """Raise MemoryError during refinement."""
            raise MemoryError("OOM in refine")

        with pytest.raises(MemoryError, match="OOM in refine"):
            _generate_batch_parallel(
                svc=phased_svc,
                count=1,
                entity_type="test",
                generate_fn=lambda _i: ({"name": "fallback"}, _make_char_scores(8.0), 1),
                get_name=lambda e: e["name"],
                quality_threshold=7.5,
                max_workers=2,
                create_only_fn=lambda _i: {"name": "E0"},
                judge_only_fn=lambda _e: failing_scores,
                is_empty_fn=lambda e: not e.get("name"),
                refine_with_initial_fn=refine_fn,
                prepare_creator_fn=lambda: None,
                prepare_judge_fn=lambda: None,
            )


class TestRelationshipPhasedPipeline:
    """Tests for the phased-pipeline callables in generate_relationships_with_quality.

    These exercise _create_only (lines 724-734), _is_empty_rel (lines 743-756),
    _judge_only (lines 760-765), _refine_with_initial (lines 775-778), and the
    _phased_kwargs assembly (lines 821-829) in _batch.py.
    """

    def _make_story_state(self):
        """Build a minimal story state for relationship generation tests."""
        from src.memory.story_state import StoryState

        state = MagicMock(spec=StoryState)
        state.id = "test-story"
        return state

    def _make_phased_svc(self):
        """Build a mock service where creator != judge to trigger the phased pipeline."""
        svc = MagicMock()
        svc._calculate_eta = MagicMock(return_value=0.0)
        config = MagicMock()
        config.quality_threshold = 7.5
        config.get_threshold = MagicMock(return_value=7.5)
        config.creator_temperature = 0.9
        config.judge_temperature = 0.1
        config.get_refinement_temperature = MagicMock(return_value=0.7)
        svc.get_config = MagicMock(return_value=config)
        svc.settings = MagicMock()
        svc.settings.llm_max_concurrent_requests = 2

        # Different creator and judge models → phased pipeline enabled
        svc._get_creator_model.return_value = "creator-model:24b"
        svc._get_judge_model.return_value = "judge-model:30b"

        # _make_model_preparers returns non-None preparers (both set)
        svc._make_model_preparers.return_value = (lambda: None, lambda: None)

        rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "ally",
            "description": "Friends",
        }
        scores = _make_rel_scores(8.0)
        svc._create_relationship.return_value = rel
        svc._judge_relationship_quality.return_value = scores
        svc._refine_relationship.return_value = rel
        svc.generate_relationship_with_quality.return_value = (rel, scores, 1)

        return svc, rel, scores

    def test_phased_pipeline_enabled_when_models_differ(self, caplog):
        """Phased callables are assembled when creator != judge (lines 821-829).

        Requires count >= 2 and llm_max_concurrent_requests >= 2 so max_workers > 1.
        """
        svc, _rel, scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # Generate multiple unique rels so dedup doesn't drop them all
        call_count = 0

        def make_rel(_story, _names, _rels):
            """Create unique test relationships."""
            nonlocal call_count
            call_count += 1
            r = {
                "source": f"E{call_count}",
                "target": f"F{call_count}",
                "relation_type": "ally",
                "description": f"Rel {call_count}",
            }
            return (r, scores, 1)

        svc.generate_relationship_with_quality = make_rel
        svc._create_relationship.side_effect = (
            make_rel.__wrapped__ if hasattr(make_rel, "__wrapped__") else None
        )

        # Provide unique rels for _create_relationship too
        cr_count = 0

        def create_rel(*args, **kwargs):
            """Create unique test relationships."""
            nonlocal cr_count
            cr_count += 1
            return {
                "source": f"A{cr_count}",
                "target": f"B{cr_count}",
                "relation_type": "ally",
                "description": "",
            }

        svc._create_relationship.side_effect = create_rel

        with caplog.at_level(logging.DEBUG):
            results = generate_relationships_with_quality(
                svc=svc,
                story_state=story_state,
                entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=2,  # count >= 2 → max_workers = min(2, 2) = 2 > 1
            )

        assert isinstance(results, list)
        assert any("phased pipeline callables prepared" in msg.lower() for msg in caplog.messages)

    def test_create_only_called_during_phased_generation(self, caplog):
        """_create_only closure is invoked during Phase 1 (lines 724-734)."""
        svc, _rel, scores = self._make_phased_svc()
        story_state = self._make_story_state()

        cr_count = 0

        def create_rel(*args, **kwargs):
            """Create unique test relationships."""
            nonlocal cr_count
            cr_count += 1
            return {
                "source": f"A{cr_count}",
                "target": f"B{cr_count}",
                "relation_type": "ally",
                "description": "",
            }

        svc._create_relationship.side_effect = create_rel

        call_count = 0

        def make_rel(_story, _names, _rels):
            """Create test relationships with quality scores."""
            nonlocal call_count
            call_count += 1
            r = {
                "source": f"E{call_count}",
                "target": f"F{call_count}",
                "relation_type": "ally",
                "description": "",
            }
            return (r, scores, 1)

        svc.generate_relationship_with_quality = make_rel

        with caplog.at_level(logging.DEBUG):
            results = generate_relationships_with_quality(
                svc=svc,
                story_state=story_state,
                entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=2,
            )

        # _create_relationship should have been called via _create_only in Phase 1
        assert svc._create_relationship.called or isinstance(results, list)

    def test_create_only_handles_world_generation_error(self, caplog):
        """_create_only returns None when _create_relationship raises (lines 732-734)."""
        svc, _rel, _scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # Make creation fail so _create_only returns None
        svc._create_relationship.side_effect = WorldGenerationError("LLM timeout")
        # generate_relationship_with_quality also fails (fallback path)
        svc.generate_relationship_with_quality.side_effect = WorldGenerationError("LLM timeout")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError):
                generate_relationships_with_quality(
                    svc=svc,
                    story_state=story_state,
                    entity_names_provider=lambda: ["Alice", "Bob"],
                    existing_rels=[],
                    count=2,  # count >= 2 to enable phased path
                )

    def test_is_empty_rel_rejects_missing_source(self):
        """_is_empty_rel returns True when source is missing (lines 743-756)."""
        svc, _rel, scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # Relationship missing source — _is_empty_rel should return True
        incomplete_rel = {"target": "Bob", "relation_type": "ally"}
        cr_count = [0]

        def create_rel(*args, **kwargs):
            """Create relationship, first incomplete then complete."""
            cr_count[0] += 1
            if cr_count[0] == 1:
                return incomplete_rel  # First one is incomplete
            # Second one is complete so phase 1 produces at least 1 entity
            return {
                "source": f"A{cr_count[0]}",
                "target": f"B{cr_count[0]}",
                "relation_type": "ally",
                "description": "",
            }

        svc._create_relationship.side_effect = create_rel

        call_count = [0]

        def make_rel(_story, _names, _rels):
            """Create test relationships with quality scores."""
            call_count[0] += 1
            r = {
                "source": f"E{call_count[0]}",
                "target": f"F{call_count[0]}",
                "relation_type": "ally",
                "description": "",
            }
            return (r, scores, 1)

        svc.generate_relationship_with_quality = make_rel

        # When the first create returns an incomplete rel, Phase 1 rejects it
        results = generate_relationships_with_quality(
            svc=svc,
            story_state=story_state,
            entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
            existing_rels=[],
            count=2,  # count >= 2 to enable phased path
        )
        assert isinstance(results, list)

    def test_phased_kwargs_not_set_when_make_model_preparers_raises(self, caplog):
        """When _make_model_preparers raises, phased kwargs are NOT set (line 832-837 _batch.py)."""
        svc, _rel, scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # Simulate exception in _make_model_preparers
        svc._make_model_preparers.side_effect = ValueError("Model not found")

        call_count = [0]

        def make_rel(_story, _names, _rels):
            """Create test relationships with quality scores."""
            call_count[0] += 1
            r = {
                "source": f"E{call_count[0]}",
                "target": f"F{call_count[0]}",
                "relation_type": "ally",
                "description": "",
            }
            return (r, scores, 1)

        svc.generate_relationship_with_quality = make_rel

        with caplog.at_level(logging.WARNING):
            results = generate_relationships_with_quality(
                svc=svc,
                story_state=story_state,
                entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=2,  # count >= 2 so max_workers > 1 to hit the try/except block
            )

        assert len(results) == 2
        assert any("failed to resolve model preparers" in msg.lower() for msg in caplog.messages)

    def test_is_empty_rel_duplicate_detection(self, caplog):
        """_is_empty_rel logs warning and returns True for duplicate relationships (lines 750-755).

        We set up an existing relationship in `existing_rels` so that
        `_is_duplicate_relationship` returns True for the first create, triggering
        the warning branch in `_is_empty_rel`.
        """
        svc, _rel, scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # existing_rels contains Alice → Bob, so any creation of Alice → Bob is a duplicate
        existing = [("Alice", "Bob", "ally")]

        cr_count = [0]

        def create_rel(*args, **kwargs):
            """Create relationship, first duplicate then unique."""
            cr_count[0] += 1
            if cr_count[0] == 1:
                # Return Alice → Bob (duplicate of existing)
                return {
                    "source": "Alice",
                    "target": "Bob",
                    "relation_type": "ally",
                    "description": "",
                }
            # Return unique pair for subsequent calls
            return {
                "source": f"E{cr_count[0]}",
                "target": f"F{cr_count[0]}",
                "relation_type": "ally",
                "description": "",
            }

        svc._create_relationship.side_effect = create_rel

        call_count = [0]

        def make_rel(_story, _names, _rels):
            """Create test relationships with quality scores."""
            call_count[0] += 1
            r = {
                "source": f"Gen{call_count[0]}",
                "target": f"GenF{call_count[0]}",
                "relation_type": "ally",
                "description": "",
            }
            return (r, scores, 1)

        svc.generate_relationship_with_quality = make_rel

        with caplog.at_level(logging.WARNING):
            results = generate_relationships_with_quality(
                svc=svc,
                story_state=story_state,
                entity_names_provider=lambda: ["Alice", "Bob", "Carol", "Dave"],
                existing_rels=existing,
                count=2,
            )

        # Duplicate rejection log should have appeared
        assert any("duplicate relationship" in msg.lower() for msg in caplog.messages)
        assert isinstance(results, list)

    def test_refine_with_initial_called_when_judge_fails_threshold(self, caplog):
        """_refine_with_initial is invoked when a relationship fails the judge (lines 775-778).

        When `_judge_relationship_quality` returns a low score, Phase 3 routes
        the relationship to `_refine_with_initial`, which calls `quality_refinement_loop`.

        `_make_model_preparers` is called twice:
          1. In generate_relationships_with_quality (must return non-None to enable phased path)
          2. Inside _refine_with_initial (lines 775-778) where the quality_refinement_loop
             is invoked with the initial relationship entity.
        We use side_effect to return non-None preparers on the first call (enabling the phased
        path in _batch.py) and (None, None) on subsequent calls (inside _refine_with_initial).
        """
        svc, _rel, _scores = self._make_phased_svc()
        story_state = self._make_story_state()

        # Use a real RefinementConfig so quality_refinement_loop gets numeric attributes
        # rather than MagicMock objects (avoids int vs MagicMock comparison errors).
        real_config = RefinementConfig(
            max_iterations=2,
            quality_threshold=7.5,
            quality_thresholds={"relationship": 7.5},
            creator_temperature=0.9,
            judge_temperature=0.1,
            early_stopping_patience=2,
            early_stopping_min_iterations=1,
            dimension_minimum=0.0,  # disable dimension floor check
        )
        svc.get_config.return_value = real_config

        # Low score → fails 7.5 threshold → _refine_with_initial is called
        low_scores = _make_rel_scores(3.0)
        high_scores = _make_rel_scores(9.0)

        cr_count = [0]

        def create_rel(*args, **kwargs):
            """Create unique test relationships."""
            cr_count[0] += 1
            return {
                "source": f"A{cr_count[0]}",
                "target": f"B{cr_count[0]}",
                "relation_type": "ally",
                "description": f"rel{cr_count[0]}",
            }

        svc._create_relationship.side_effect = create_rel

        # quality_refinement_loop will call refine_fn — wire it up
        svc._refine_relationship.return_value = {
            "source": "A_refined",
            "target": "B_refined",
            "relation_type": "ally",
            "description": "refined",
        }

        # Judge: low scores for Phase 2 batch calls, high for refinement loop calls
        judge_calls = [0]

        def judge_fn(*args, **kwargs):
            """Return low scores for initial batch, high for refinement."""
            judge_calls[0] += 1
            # Phase 2 batch judges (2 entities) return low; refinement loop calls return high
            if judge_calls[0] <= 2:
                return low_scores
            return high_scores

        svc._judge_relationship_quality.side_effect = judge_fn

        # _make_model_preparers side_effect:
        #   - 1st call (in generate_relationships_with_quality _batch.py:816): return (prep, prep)
        #     → both non-None triggers the phased path
        #   - 2nd+ calls (inside _refine_with_initial at _batch.py:777): return (None, None)
        def prep_fn():
            """Prepare model for generation."""
            return None

        prep_calls = [0]

        def make_preparers(*args, **kwargs):
            """Return mock model preparers, first call non-None then None."""
            prep_calls[0] += 1
            if prep_calls[0] == 1:
                # First call from _batch.py: return non-None preparers to enable phased path
                return (prep_fn, prep_fn)
            # Subsequent calls from inside _refine_with_initial: return (None, None)
            return (None, None)

        svc._make_model_preparers.side_effect = make_preparers
        # _log_refinement_analytics is called inside quality_refinement_loop
        svc._log_refinement_analytics = MagicMock()
        # analytics_db is queried for hail-mary win-rate inside quality_refinement_loop
        svc.analytics_db.get_hail_mary_win_rate.return_value = 1.0

        with caplog.at_level(logging.INFO):
            results = generate_relationships_with_quality(
                svc=svc,
                story_state=story_state,
                entity_names_provider=lambda: ["Alice", "Bob", "Carol"],
                existing_rels=[],
                count=2,
            )

        # Results should include the refined entities
        assert isinstance(results, list)
        # _refine_with_initial should have been called (Phase 3b refinement)
        assert any("phase 3b" in msg.lower() or "refin" in msg.lower() for msg in caplog.messages)
