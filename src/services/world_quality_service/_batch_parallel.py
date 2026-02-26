"""Parallel batch generation for entity pipelines.

Extracted from ``_batch.py`` to keep both modules under the 1000-line limit.
Provides :class:`_ThreadSafeRelsList` and :func:`_generate_batch_parallel`,
which are used exclusively by ``generate_relationships_with_quality``.
"""

import concurrent.futures as cf
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.memory.world_quality import BaseQualityScores
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

if TYPE_CHECKING:
    from src.services.world_quality_service import EntityGenerationProgress, WorldQualityService

logger = logging.getLogger(__name__)


class _ThreadSafeRelsList:
    """Thread-safe wrapper for the shared relationship deduplication list.

    Workers get point-in-time snapshots of the list, accepting that snapshots
    may be stale by at most ``max_workers - 1`` items (because completions are
    processed one at a time in the main thread, and each triggers at most one
    new submission).  The ``_is_duplicate_relationship()`` check in the quality
    refinement loop rejects duplicates that exist in the snapshot; as an
    additional safety net, :meth:`append_if_new_pair` performs an atomic
    check-and-insert to catch duplicates that concurrent workers may produce
    from identical snapshots.
    """

    def __init__(self, initial: list[tuple[str, str, str]]) -> None:
        """Initialize with an existing relationship list to deduplicate against."""
        self._lock = threading.Lock()
        self._data: list[tuple[str, str, str]] = list(initial)

    def snapshot(self) -> list[tuple[str, str, str]]:
        """Return a copy of the list under lock."""
        with self._lock:
            return list(self._data)

    def append(self, rel: tuple[str, str, str]) -> None:
        """Atomically add a new relationship."""
        with self._lock:
            self._data.append(rel)

    def append_if_new_pair(self, rel: tuple[str, str, str]) -> bool:
        """Atomically append only if the source/target pair is not already present.

        The check is bidirectional: ``(A, B)`` and ``(B, A)`` are treated as
        the same pair regardless of relation type.

        Returns:
            True if the pair was new and was appended, False if it already existed.
        """
        source, target, _rel_type = rel
        with self._lock:
            for existing_source, existing_target, _ in self._data:
                if (source == existing_source and target == existing_target) or (
                    source == existing_target and target == existing_source
                ):
                    return False
            self._data.append(rel)
            return True

    def __len__(self) -> int:
        """Return the number of relationships in the list."""
        with self._lock:
            return len(self._data)


def _generate_batch_parallel[T, S: BaseQualityScores](
    svc: WorldQualityService,
    count: int,
    entity_type: str,
    generate_fn: Callable[[int], tuple[T, S, int]],
    get_name: Callable[[T], str],
    on_success: Callable[[T], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    quality_threshold: float | None = None,
    max_workers: int = 2,
) -> list[tuple[T, S]]:
    """Parallel batch generation using a rolling-window thread pool.

    Same contract as ``_generate_batch`` but submits up to ``max_workers``
    tasks concurrently.  Progress callbacks, ``on_success`` hooks, and
    consecutive-failure tracking all run in the main thread (processing
    ``as_completed`` results), avoiding UI thread-safety issues.

    **Consecutive-failure semantics differ from the sequential version.**
    Because futures complete in non-deterministic order, "consecutive failures"
    really means "recently completed failures without an intervening success".
    This is more conservative (terminates earlier) than the sequential version,
    which is the safer direction for a parallel system.

    Args:
        svc: WorldQualityService instance (used for ``_calculate_eta``).
        count: Number of entities to generate.
        entity_type: Human-readable type name for logging/progress.
        generate_fn: Called with the task index ``i``; must return
            ``(entity, scores, iterations)``.  Must be thread-safe.
        get_name: Extract a display name from the generated entity.
        on_success: Optional hook called after a successful generation.
            Called from the main thread.  May raise ``WorldGenerationError``
            to reject the entity (e.g. duplicate detection via
            :meth:`_ThreadSafeRelsList.append_if_new_pair`).
        cancel_check: Optional callable that returns ``True`` to cancel.
        progress_callback: Optional callback for progress updates.
            Called from the main thread.
        quality_threshold: Quality threshold for pass/fail in batch summary.
        max_workers: Maximum concurrent generation tasks.

    Returns:
        List of ``(entity, scores)`` tuples.

    Raises:
        WorldGenerationError: If **no** entities could be generated.
    """
    from src.services.world_quality_service._batch import (
        MAX_BATCH_SHUFFLE_RETRIES,
        MAX_CONSECUTIVE_BATCH_FAILURES,
        _aggregate_errors,
        _generate_batch,
        _log_batch_summary,
    )

    if count == 0:
        return []

    # Degenerate case: single worker → delegate to sequential
    if max_workers <= 1:
        return _generate_batch(
            svc=svc,
            count=count,
            entity_type=entity_type,
            generate_fn=generate_fn,
            get_name=get_name,
            on_success=on_success,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            quality_threshold=quality_threshold,
        )

    from src.services.world_quality_service import EntityGenerationProgress

    if quality_threshold is None:
        quality_threshold = svc.get_config().get_threshold(entity_type)

    results: list[tuple[T, S]] = []
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []
    consecutive_failures = 0
    shuffles_remaining = MAX_BATCH_SHUFFLE_RETRIES
    completed_count = 0

    logger.info(
        "Starting parallel %s generation: %d entities with max_workers=%d",
        entity_type,
        count,
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Track futures → (task_index, submit_time)
        pending: dict[Future[tuple[T, S, int]], tuple[int, float]] = {}
        next_index = 0

        def _submit_next() -> bool:
            """Submit the next task if available and not cancelled.

            Must be called from the main thread only (accesses ``pending``
            and ``next_index`` without synchronization).

            Returns:
                True if a task was submitted, False otherwise.
            """
            nonlocal next_index
            if next_index >= count:
                return False
            if cancel_check and cancel_check():
                return False

            idx = next_index
            next_index += 1

            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=idx + 1,
                        total=count,
                        entity_type=entity_type,
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - completed_count
                        ),
                    )
                )

            future: Future[tuple[T, S, int]] = executor.submit(generate_fn, idx)
            pending[future] = (idx, time.time())
            return True

        # Seed the pool with initial tasks
        for _ in range(min(max_workers, count)):
            _submit_next()

        # Process completions one at a time
        while pending:
            done_iter = as_completed(pending, timeout=None)
            for future in done_iter:
                task_idx, task_start = pending.pop(future)
                completed_count += 1
                entity_elapsed = time.time() - task_start

                try:
                    entity, scores, iterations = future.result()
                    entity_name = get_name(entity)

                    if on_success:
                        on_success(entity)

                    # Only count as success after get_name and on_success pass
                    completed_times.append(entity_elapsed)
                    results.append((entity, scores))
                    consecutive_failures = 0

                    logger.info(
                        "%s '%s' complete after %d iteration(s), "
                        "quality: %.1f, generation_time: %.2fs",
                        entity_type.capitalize(),
                        entity_name,
                        iterations,
                        scores.average,
                        entity_elapsed,
                    )

                    if progress_callback:
                        progress_callback(
                            EntityGenerationProgress(
                                current=completed_count,
                                total=count,
                                entity_type=entity_type,
                                entity_name=entity_name,
                                phase="complete",
                                elapsed_seconds=time.time() - batch_start_time,
                                estimated_remaining_seconds=svc._calculate_eta(
                                    completed_times, count - completed_count
                                ),
                            )
                        )

                except cf.CancelledError:
                    logger.warning(
                        "%s task %d was cancelled before completion",
                        entity_type.capitalize(),
                        task_idx + 1,
                    )

                except WorldGenerationError as e:
                    error_msg = summarize_llm_error(e, max_length=200)
                    errors.append(error_msg)
                    consecutive_failures += 1
                    logger.error(
                        "Failed to generate %s (task %d): %s",
                        entity_type,
                        task_idx + 1,
                        error_msg,
                    )

                except Exception as e:
                    error_msg = summarize_llm_error(e, max_length=200)
                    errors.append(error_msg)
                    consecutive_failures += 1
                    logger.error(
                        "Unexpected error in parallel %s generation (task %d): %s",
                        entity_type,
                        task_idx + 1,
                        error_msg,
                    )

                # Check for early termination after any failure
                if consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    if shuffles_remaining > 0:
                        shuffles_remaining -= 1
                        logger.warning(
                            "%s parallel batch: %d consecutive failures. "
                            "Attempting recovery (generated %d/%d so far).",
                            entity_type.capitalize(),
                            consecutive_failures,
                            len(results),
                            count,
                        )
                        consecutive_failures = 0
                    else:
                        logger.warning(
                            "%s parallel batch early termination: %d consecutive "
                            "failures after recovery attempt. "
                            "Generated %d/%d successfully before stopping.",
                            entity_type.capitalize(),
                            consecutive_failures,
                            len(results),
                            count,
                        )
                        # Cancel futures that haven't started
                        for remaining_future in pending:
                            remaining_future.cancel()
                        # Collect results from already-running futures
                        _collect_late_results(
                            pending,
                            results,
                            completed_times,
                            errors,
                            entity_type,
                            get_name,
                            on_success,
                        )
                        pending.clear()
                        break

                # Submit next task after processing this completion
                _submit_next()

                # Re-enter as_completed so newly submitted futures are
                # included in the wait set (the current iterator only
                # tracks futures from when it was created).
                break

    if not results and errors:
        raise WorldGenerationError(
            f"Failed to generate any {entity_type}s. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            "Generated %d/%d %ss (parallel). %d failed: %s",
            len(results),
            count,
            entity_type,
            len(errors),
            _aggregate_errors(errors),
        )

    _log_batch_summary(
        results, entity_type, quality_threshold, time.time() - batch_start_time, get_name=get_name
    )

    return results


def _collect_late_results[T, S: BaseQualityScores](
    pending: dict[Future[tuple[T, S, int]], tuple[int, float]],
    results: list[tuple[T, S]],
    completed_times: list[float],
    errors: list[str],
    entity_type: str,
    get_name: Callable[[T], str],
    on_success: Callable[[T], None] | None,
) -> None:
    """Collect results from futures that were already running during early termination.

    Futures that have already started cannot be cancelled by ``Future.cancel()``.
    Rather than silently discarding their results when the executor shuts down,
    this function waits for them and appends any successes to ``results``.

    Args:
        pending: The pending futures dict (read-only; not modified here).
        results: Mutable results list to append late successes to.
        completed_times: Mutable timing list for late successes.
        errors: Mutable error list for late failures.
        entity_type: For logging.
        get_name: Entity name extractor.
        on_success: Optional success hook.
    """
    for remaining_future, (_remaining_idx, remaining_start) in list(pending.items()):
        if remaining_future.cancelled():
            continue
        try:
            entity, scores, _iters = remaining_future.result()
            entity_name = get_name(entity)
            if on_success:
                on_success(entity)
            results.append((entity, scores))
            completed_times.append(time.time() - remaining_start)
            logger.info(
                "Collected late %s '%s' during early termination",
                entity_type,
                entity_name,
            )
        except Exception as late_err:
            errors.append(str(late_err)[:200])
            logger.debug(
                "Discarded late %s during early termination: %s",
                entity_type,
                late_err,
            )
