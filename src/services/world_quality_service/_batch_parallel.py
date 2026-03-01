"""Parallel batch generation for entity pipelines.

Provides :class:`_ThreadSafeRelsList`, :func:`_generate_batch_parallel`,
and :func:`_generate_batch_phased`.
"""

import concurrent.futures as cf
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.memory.world_quality import BaseQualityScores
from src.utils.exceptions import (
    DuplicateNameError,
    VRAMAllocationError,
    WorldGenerationError,
    summarize_llm_error,
)

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
    *,
    create_only_fn: Callable[[int], T | None] | None = None,
    judge_only_fn: Callable[[T], S] | None = None,
    is_empty_fn: Callable[[T], bool] | None = None,
    refine_with_initial_fn: Callable[[T], tuple[T, S, int]] | None = None,
    prepare_creator_fn: Callable[[], None] | None = None,
    prepare_judge_fn: Callable[[], None] | None = None,
    register_created_fn: Callable[[T], None] | None = None,
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

    **Two-phase pipeline:** When ``creator != judge`` and all phased callables
    are provided (``create_only_fn``, ``judge_only_fn``, ``is_empty_fn``,
    ``refine_with_initial_fn``, ``prepare_creator_fn``, ``prepare_judge_fn``),
    a phased pipeline is used instead of the sequential fallback.  Phase 1
    loads the creator model once and batches all creates; Phase 2 loads the
    judge model once and batches all judges.  Entities that pass the quality
    threshold are emitted directly; failures fall back to per-entity
    ``quality_refinement_loop`` with ``initial_entity``.  This reduces GPU
    model swaps from ~2N to ~4 (two bulk swaps plus refinement swaps).

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
        create_only_fn: Creates a single entity without judging.  Returns
            the entity or ``None`` on failure.  Required for phased pipeline.
        judge_only_fn: Judges a single entity.  Returns quality scores.
            Required for phased pipeline.
        is_empty_fn: Returns ``True`` if the entity is invalid/empty.
            Required for phased pipeline.
        refine_with_initial_fn: Runs the full quality refinement loop with
            ``initial_entity`` for entities that fail the judge threshold.
            Returns ``(entity, scores, iterations)``.  Required for phased
            pipeline.
        prepare_creator_fn: Callback to load the creator model into VRAM.
            Required for phased pipeline (may be ``None`` when models are
            the same, but phased pipeline is not used in that case).
        register_created_fn: Optional callback invoked after an entity is
            created in Phase 1.  Used for dedup registration (e.g. adding
            the entity to the build pipeline's tracking).  Errors are
            non-fatal — the entity proceeds to judging regardless.
        prepare_judge_fn: Callback to load the judge model into VRAM.
            Required for phased pipeline.

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

    # When creator and judge models differ, check whether the phased pipeline
    # can be used.  The phased approach batches all creates under one model load
    # and all judges under another, reducing GPU swaps from ~2N to ~4.
    _use_phased = False
    if max_workers > 1:
        try:
            creator = svc._get_creator_model(entity_type)
            judge = svc._get_judge_model(entity_type)
            if creator != judge:
                _has_phased_callables = all(
                    (
                        create_only_fn is not None,
                        judge_only_fn is not None,
                        is_empty_fn is not None,
                        refine_with_initial_fn is not None,
                    )
                )
                if _has_phased_callables:
                    _use_phased = True
                    logger.info(
                        "Using phased pipeline: creator (%s) != judge (%s), "
                        "batching creates then judges to reduce GPU swaps",
                        creator,
                        judge,
                    )
                else:
                    logger.info(
                        "Reducing max_workers to 1: creator (%s) != judge (%s), "
                        "avoiding GPU thrashing (phased callables not provided)",
                        creator,
                        judge,
                    )
                    max_workers = 1
        except (ValueError, KeyError, LookupError) as e:
            logger.warning(
                "Failed to resolve models for max_workers decision on %s: %s. "
                "Keeping max_workers=%d",
                entity_type,
                e,
                max_workers,
            )

    if _use_phased:
        # All phased callables are guaranteed non-None by the check above.
        assert create_only_fn is not None  # nosec — invariant
        assert judge_only_fn is not None  # nosec — invariant
        assert is_empty_fn is not None  # nosec — invariant
        assert refine_with_initial_fn is not None  # nosec — invariant
        return _generate_batch_phased(
            svc=svc,
            count=count,
            entity_type=entity_type,
            create_only_fn=create_only_fn,
            judge_only_fn=judge_only_fn,
            is_empty_fn=is_empty_fn,
            refine_with_initial_fn=refine_with_initial_fn,
            prepare_creator_fn=prepare_creator_fn,
            prepare_judge_fn=prepare_judge_fn,
            get_name=get_name,
            on_success=on_success,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            quality_threshold=quality_threshold,
            register_created_fn=register_created_fn,
        )

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
    duplicates_found = 0
    # Cap duplicate retries to prevent unbounded submission when workers
    # persistently produce duplicates (e.g. all entity-pair slots are taken).
    max_duplicate_retries = count * (MAX_BATCH_SHUFFLE_RETRIES + 1)

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
            if next_index >= count + min(duplicates_found, max_duplicate_retries):
                return False
            if cancel_check and cancel_check():
                return False

            idx = next_index
            next_index += 1

            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=min(idx + 1, count),
                        total=count,
                        entity_type=entity_type,
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, max(count - completed_count, 0)
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
                                current=min(len(results), count),
                                total=count,
                                entity_type=entity_type,
                                entity_name=entity_name,
                                phase="complete",
                                elapsed_seconds=time.time() - batch_start_time,
                                estimated_remaining_seconds=svc._calculate_eta(
                                    completed_times, max(count - len(results), 0)
                                ),
                            )
                        )

                except cf.CancelledError:
                    logger.warning(
                        "%s task %d was cancelled before completion",
                        entity_type.capitalize(),
                        task_idx + 1,
                    )

                except DuplicateNameError as e:
                    # Expected race outcome — extend submission bound for replacement;
                    # record in errors so all-duplicate runs raise instead of returning [].
                    duplicate_msg = summarize_llm_error(e, max_length=200)
                    duplicates_found += 1
                    errors.append(duplicate_msg)
                    logger.warning(
                        "Duplicate %s from parallel worker (task %d): %s",
                        entity_type,
                        task_idx + 1,
                        duplicate_msg,
                    )

                except WorldGenerationError as e:
                    # VRAMAllocationError is non-retryable — stop the entire batch
                    if isinstance(e.__cause__, VRAMAllocationError):
                        raise
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


def _generate_batch_phased[T, S: BaseQualityScores](
    svc: WorldQualityService,
    count: int,
    entity_type: str,
    create_only_fn: Callable[[int], T | None],
    judge_only_fn: Callable[[T], S],
    is_empty_fn: Callable[[T], bool],
    refine_with_initial_fn: Callable[[T], tuple[T, S, int]],
    prepare_creator_fn: Callable[[], None] | None,
    prepare_judge_fn: Callable[[], None] | None,
    get_name: Callable[[T], str],
    on_success: Callable[[T], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    quality_threshold: float | None = None,
    register_created_fn: Callable[[T], None] | None = None,
) -> list[tuple[T, S]]:
    """Two-phase batch pipeline: batch creates, then batch judges.

    Reduces GPU model swaps from ~2N (per-entity create+judge interleaving)
    to ~4 (one bulk create swap, one bulk judge swap, plus refinement swaps
    for entities that fail the threshold).

    Phase 1 — Batch creates: Load creator model once, run all N create calls
    sequentially.  Collect valid entities.

    Phase 2 — Batch judges: Load judge model once, run all N judge calls
    sequentially.  Collect scores.

    Phase 3 — Handle results: Entities that pass the quality threshold are
    emitted directly.  Entities that fail are refined via the existing
    per-entity ``quality_refinement_loop`` (with ``initial_entity``), which
    handles all early-stopping, hail-mary, and analytics logic.

    Args:
        svc: WorldQualityService instance.
        count: Number of entities to generate.
        entity_type: Human-readable type name for logging/progress.
        create_only_fn: Creates a single entity without judging.  Returns
            the entity or ``None`` on failure.
        judge_only_fn: Judges a single entity, returns quality scores.
        is_empty_fn: Returns ``True`` if the entity is invalid/empty.
        refine_with_initial_fn: Runs the full quality refinement loop with
            ``initial_entity`` for entities needing refinement.
        prepare_creator_fn: Callback to load creator model into VRAM.
        prepare_judge_fn: Callback to load judge model into VRAM.
        get_name: Extract display name from an entity.
        on_success: Optional hook called after a successful generation.
        cancel_check: Optional callable returning ``True`` to cancel.
        progress_callback: Optional progress update callback.
        quality_threshold: Quality threshold for pass/fail.
        register_created_fn: Optional callback invoked after an entity is
            created in Phase 1.  Used for dedup registration.  Errors are
            non-fatal — the entity proceeds to judging regardless.

    Returns:
        List of ``(entity, scores)`` tuples.

    Raises:
        WorldGenerationError: If no entities could be generated.
    """
    from src.services.world_quality_service import EntityGenerationProgress
    from src.services.world_quality_service._batch import (
        _aggregate_errors,
        _log_batch_summary,
    )

    if quality_threshold is None:
        quality_threshold = svc.get_config().get_threshold(entity_type)

    results: list[tuple[T, S]] = []
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    logger.info(
        "Starting phased %s pipeline: %d entities (batch creates, then batch judges)",
        entity_type,
        count,
    )

    # ── Phase 1: Batch creates ──────────────────────────────────────────
    if prepare_creator_fn:
        logger.info("Phase 1: loading creator model for batch creation")
        prepare_creator_fn()

    created_entities: list[tuple[int, T]] = []  # (index, entity) pairs
    create_errors = 0

    for i in range(count):
        if cancel_check and cancel_check():
            logger.info(
                "Phased %s creation cancelled after %d/%d",
                entity_type,
                len(created_entities),
                count,
            )
            break

        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type=entity_type,
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(
                        completed_times, max(count - i, 0)
                    ),
                )
            )

        entity_start = time.time()
        try:
            entity = create_only_fn(i)
            if entity is not None and not is_empty_fn(entity):
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                if register_created_fn:
                    try:
                        register_created_fn(entity)
                    except Exception as reg_err:
                        logger.warning(
                            "Phase 1: register_created_fn failed for %s %d/%d '%s': %s",
                            entity_type,
                            i + 1,
                            count,
                            get_name(entity),
                            reg_err,
                        )
                created_entities.append((i, entity))
                logger.debug(
                    "Phase 1: created %s %d/%d '%s' in %.2fs",
                    entity_type,
                    i + 1,
                    count,
                    get_name(entity),
                    entity_elapsed,
                )
            else:
                create_errors += 1
                error_msg = f"{entity_type} creation returned empty on index {i}"
                errors.append(error_msg)
                logger.warning("Phase 1: %s", error_msg)
        except (WorldGenerationError, ValueError) as e:
            # VRAMAllocationError is non-retryable — stop the entire batch
            if isinstance(getattr(e, "__cause__", None), VRAMAllocationError):
                raise
            create_errors += 1
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 1: failed to create %s %d/%d: %s", entity_type, i + 1, count, error_msg
            )
        except MemoryError, RecursionError, KeyboardInterrupt, SystemExit:
            raise
        except Exception as e:
            create_errors += 1
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 1: unexpected error creating %s %d/%d: %s",
                entity_type,
                i + 1,
                count,
                error_msg,
            )

    logger.info(
        "Phase 1 complete: %d/%d %ss created (%d failed) in %.1fs",
        len(created_entities),
        count,
        entity_type,
        create_errors,
        time.time() - batch_start_time,
    )

    if not created_entities:
        if errors:
            raise WorldGenerationError(
                f"Phased pipeline: failed to create any {entity_type}s. Errors: {'; '.join(errors)}"
            )
        return []

    # ── Phase 2: Batch judges ───────────────────────────────────────────
    if prepare_judge_fn:
        logger.info("Phase 2: loading judge model for batch judging")
        prepare_judge_fn()

    # Track: (index, entity, scores, passed_threshold)
    judged: list[tuple[int, T, S, bool]] = []
    judge_errors = 0

    for idx, entity in created_entities:
        if cancel_check and cancel_check():
            logger.info(
                "Phased %s judging cancelled after %d/%d",
                entity_type,
                len(judged),
                len(created_entities),
            )
            break

        entity_name = get_name(entity)
        judge_start = time.time()
        try:
            scores = judge_only_fn(entity)
            judge_elapsed = time.time() - judge_start
            rounded_score = round(scores.average, 1)
            passed = rounded_score >= quality_threshold
            judged.append((idx, entity, scores, passed))
            logger.info(
                "Phase 2: judged %s '%s': score=%.1f %s (%.2fs)",
                entity_type,
                entity_name,
                scores.average,
                "PASS" if passed else "FAIL",
                judge_elapsed,
            )
        except (WorldGenerationError, ValueError) as e:
            # VRAMAllocationError is non-retryable — stop the entire batch
            if isinstance(getattr(e, "__cause__", None), VRAMAllocationError):
                raise
            judge_errors += 1
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 2: failed to judge %s '%s': %s",
                entity_type,
                entity_name,
                error_msg,
            )
            # Judge failed — still worth attempting refinement (includes its own judge).
            judged.append((idx, entity, None, False))  # type: ignore[arg-type]
        except MemoryError, RecursionError, KeyboardInterrupt, SystemExit:
            raise
        except Exception as e:
            judge_errors += 1
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 2: unexpected error judging %s '%s': %s",
                entity_type,
                entity_name,
                error_msg,
            )
            judged.append((idx, entity, None, False))  # type: ignore[arg-type]

    logger.info(
        "Phase 2 complete: %d/%d %ss judged (%d errors) in %.1fs",
        len(judged),
        len(created_entities),
        entity_type,
        judge_errors,
        time.time() - batch_start_time,
    )

    # ── Phase 3: Handle results ─────────────────────────────────────────
    passed_count = 0
    needs_refinement: list[tuple[int, T]] = []

    for idx, entity, scores, passed in judged:
        if passed and scores is not None:
            # Entity passed threshold — emit directly
            entity_start = time.time()
            try:
                if on_success:
                    on_success(entity)
                entity_name = get_name(entity)
                results.append((entity, scores))
                completed_times.append(time.time() - entity_start)
                passed_count += 1
                logger.info(
                    "Phase 3: %s '%s' passed threshold (%.1f >= %.1f), emitting directly",
                    entity_type,
                    entity_name,
                    scores.average,
                    quality_threshold,
                )
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=min(len(results), count),
                            total=count,
                            entity_type=entity_type,
                            entity_name=entity_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=svc._calculate_eta(
                                completed_times, max(count - len(results), 0)
                            ),
                        )
                    )
            except DuplicateNameError as e:
                # H4: retry via refine to recover the slot (up to 2 attempts)
                dup_msg = summarize_llm_error(e, max_length=200)
                _dup_recovered = False
                if refine_with_initial_fn is not None:
                    for _retry in range(2):
                        try:
                            retry_entity, retry_scores, _iters = refine_with_initial_fn(entity)
                            if on_success:
                                on_success(retry_entity)
                            results.append((retry_entity, retry_scores))
                            completed_times.append(time.time() - entity_start)
                            logger.info(
                                "Phase 3: recovered duplicate %s via retry %d",
                                entity_type,
                                _retry + 1,
                            )
                            _dup_recovered = True
                            break
                        except DuplicateNameError:
                            continue
                        except MemoryError, RecursionError, KeyboardInterrupt, SystemExit:
                            raise
                        except Exception as retry_err:
                            logger.warning(
                                "Phase 3: duplicate retry failed for %s: %s", entity_type, retry_err
                            )
                            break
                if not _dup_recovered:
                    errors.append(dup_msg)
                    logger.warning("Phase 3: duplicate %s rejected: %s", entity_type, dup_msg)
            except WorldGenerationError as e:
                # VRAMAllocationError is non-retryable — stop the entire batch
                if isinstance(e.__cause__, VRAMAllocationError):
                    raise
                error_msg = summarize_llm_error(e, max_length=200)
                errors.append(error_msg)
                logger.error("Phase 3: on_success rejected %s: %s", entity_type, error_msg)
        else:
            # Entity failed threshold or judge errored — queue for refinement
            needs_refinement.append((idx, entity))

    logger.info(
        "Phase 3: %d/%d %ss passed threshold, %d need refinement",
        passed_count,
        len(judged),
        entity_type,
        len(needs_refinement),
    )

    # ── Phase 3b: Refine entities that failed threshold ─────────────────
    # Each refinement call invokes the full quality_refinement_loop with
    # initial_entity, which handles its own model swapping, early-stopping,
    # hail-mary, and analytics.
    for _idx, entity in needs_refinement:
        if cancel_check and cancel_check():
            logger.info(
                "Phased %s refinement cancelled after %d results",
                entity_type,
                len(results),
            )
            break

        entity_name = get_name(entity)
        refine_start = time.time()
        try:
            logger.info(
                "Phase 3b: refining %s '%s' via quality_refinement_loop",
                entity_type,
                entity_name,
            )
            refined_entity, refined_scores, iterations = refine_with_initial_fn(entity)
            refine_elapsed = time.time() - refine_start

            if on_success:
                on_success(refined_entity)

            results.append((refined_entity, refined_scores))
            completed_times.append(refine_elapsed)
            refined_name = get_name(refined_entity)
            logger.info(
                "Phase 3b: %s '%s' refined in %d iteration(s), score=%.1f (%.2fs)",
                entity_type,
                refined_name,
                iterations,
                refined_scores.average,
                refine_elapsed,
            )
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=min(len(results), count),
                        total=count,
                        entity_type=entity_type,
                        entity_name=refined_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, max(count - len(results), 0)
                        ),
                    )
                )
        except DuplicateNameError as e:
            dup_msg = summarize_llm_error(e, max_length=200)
            errors.append(dup_msg)
            logger.warning(
                "Phase 3b: duplicate %s '%s' during refinement: %s",
                entity_type,
                entity_name,
                dup_msg,
            )
        except WorldGenerationError as e:
            # VRAMAllocationError is non-retryable — stop the entire batch
            if isinstance(e.__cause__, VRAMAllocationError):
                raise
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 3b: failed to refine %s '%s': %s",
                entity_type,
                entity_name,
                error_msg,
            )
        except MemoryError, RecursionError, KeyboardInterrupt, SystemExit:
            raise
        except Exception as e:
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error(
                "Phase 3b: unexpected error refining %s '%s': %s",
                entity_type,
                entity_name,
                error_msg,
            )

    # ── Final summary ───────────────────────────────────────────────────
    total_elapsed = time.time() - batch_start_time
    logger.info(
        "Phased %s pipeline complete: %d/%d entities in %.1fs "
        "(phase1: %d created, phase2: %d judged, phase3: %d passed + %d/%d refined)",
        entity_type,
        len(results),
        count,
        total_elapsed,
        len(created_entities),
        len(judged),
        passed_count,
        len(results) - passed_count,
        len(needs_refinement),
    )

    if not results and errors:
        raise WorldGenerationError(
            f"Phased pipeline: failed to generate any {entity_type}s. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            "Phased pipeline: generated %d/%d %ss. %d errors: %s",
            len(results),
            count,
            entity_type,
            len(errors),
            _aggregate_errors(errors),
        )

    _log_batch_summary(results, entity_type, quality_threshold, total_elapsed, get_name=get_name)

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
            if isinstance(late_err, (MemoryError, RecursionError)):
                raise
            late_msg = summarize_llm_error(late_err, max_length=200)
            errors.append(late_msg)
            logger.warning(
                "Discarded late %s during early termination: %s",
                entity_type,
                late_msg,
                exc_info=True,
            )
