"""Helper utilities for parallel batch generation.

Contains :class:`_ThreadSafeRelsList` for thread-safe relationship deduplication
and :func:`_collect_late_results` for gathering results from in-flight futures
during early termination.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

from src.memory.world_quality import BaseQualityScores
from src.utils.exceptions import VRAMAllocationError, summarize_llm_error

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
            if isinstance(late_err, (MemoryError, RecursionError)):
                raise
            if isinstance(getattr(late_err, "__cause__", None), VRAMAllocationError):
                raise
            late_msg = summarize_llm_error(late_err, max_length=200)
            errors.append(late_msg)
            logger.warning(
                "Discarded late %s during early termination: %s",
                entity_type,
                late_msg,
                exc_info=True,
            )
