"""Generic quality refinement loop for all entity types.

Extracts the common create-judge-refine loop that was previously copy-pasted
across 6 entity modules (~60 lines each). Each entity module now provides
entity-specific callables (create, judge, refine, serialize, get_name)
and this module orchestrates the loop.
"""

import logging
from collections.abc import Callable
from typing import TypeVar, cast

from src.memory.world_quality import BaseQualityScores, RefinementConfig, RefinementHistory
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S", bound=BaseQualityScores)


def quality_refinement_loop(
    *,
    entity_type: str,
    create_fn: Callable[[int], T],
    judge_fn: Callable[[T], S],
    refine_fn: Callable[[T, S, int], T],
    get_name: Callable[[T], str],
    serialize: Callable[[T], dict],
    is_empty: Callable[[T], bool],
    score_cls: type[S],
    config: RefinementConfig,
    svc,
    story_id: str,
    initial_entity: T | None = None,
) -> tuple[T, S, int]:
    """Run the quality refinement loop for any entity type.

    Handles creation, judging, early stopping, best-iteration tracking, and
    analytics logging. Supports both generation mode (create from scratch)
    and review mode (judge/refine an existing entity).

    Args:
        entity_type: Entity type label for logging and analytics.
        create_fn: Creates a new entity. Takes creation_retries count for temperature escalation.
            Called on iteration 0 unless initial_entity is set.
        judge_fn: Judges an entity, returns a score model.
        refine_fn: Refines an entity using scores and iteration number.
        get_name: Extracts display name from an entity.
        serialize: Serializes an entity to dict for RefinementHistory tracking.
        is_empty: Returns True if the entity is invalid/empty (e.g. duplicate name cleared).
        score_cls: The Pydantic score model class for score reconstruction.
        config: RefinementConfig with thresholds and temperatures.
        svc: WorldQualityService instance (for analytics logging).
        story_id: Project/story ID for analytics.
        initial_entity: If provided, skip creation and start with this entity (review mode).

    Returns:
        Tuple of (best_entity, best_scores, total_iterations).

    Raises:
        WorldGenerationError: If no valid entity could be produced after all attempts.
    """
    logger.info(
        "Starting quality refinement loop for %s (threshold=%.1f, max_iterations=%d, "
        "review_mode=%s)",
        entity_type,
        config.quality_threshold,
        config.max_iterations,
        initial_entity is not None,
    )

    history = RefinementHistory(entity_type=entity_type, entity_name="")
    iteration = 0
    entity: T | None = initial_entity
    scores: S | None = None
    last_error: str = ""
    creation_retries = 0
    early_stopped = False

    while iteration < config.max_iterations:
        try:
            # Creation or refinement
            if iteration == 0 and initial_entity is not None:
                # Review mode: use the provided entity directly
                entity = initial_entity
            elif entity is None or (iteration == 0) or is_empty(entity):
                # Need fresh creation
                entity = create_fn(creation_retries)
                if entity is None or is_empty(entity):
                    creation_retries += 1
                    last_error = (
                        f"{entity_type} creation returned empty on iteration {iteration + 1}"
                    )
                    logger.warning(
                        "%s (retry %d)",
                        last_error,
                        creation_retries,
                    )
                    iteration += 1
                    continue
            else:
                # Refinement based on previous feedback
                if scores is not None:
                    dynamic_temp_iter = iteration + 1
                    entity = refine_fn(entity, scores, dynamic_temp_iter)
                    if entity is None or is_empty(entity):
                        creation_retries += 1
                        last_error = (
                            f"{entity_type} refinement returned empty on iteration {iteration + 1}"
                        )
                        logger.warning(last_error)
                        iteration += 1
                        continue

                    # Detect unchanged refinement output (#246 RC5)
                    if history.iterations:
                        prev_data = history.iterations[-1].entity_data
                        curr_data = serialize(entity)
                        if prev_data == curr_data:
                            logger.info(
                                "%s '%s' refinement produced unchanged output on "
                                "iteration %d, skipping further iterations",
                                entity_type.capitalize(),
                                get_name(entity),
                                iteration + 1,
                            )
                            early_stopped = True
                            break

            # Update history entity name on first successful entity
            if not history.entity_name:
                history.entity_name = get_name(entity)

            # Judge
            scores = judge_fn(entity)

            # Track in history
            history.add_iteration(
                entity_data=serialize(entity),
                scores=scores.to_dict(),
                average_score=scores.average,
                feedback=scores.feedback,
            )

            current_iter = history.iterations[-1].iteration
            logger.info(
                "%s '%s' iteration %d: score %.1f (best so far: %.1f at iteration %d)",
                entity_type.capitalize(),
                get_name(entity),
                current_iter,
                scores.average,
                history.peak_score,
                history.best_iteration,
            )
            logger.debug(
                "%s '%s' iteration %d dimension scores: %s",
                entity_type.capitalize(),
                get_name(entity),
                current_iter,
                {k: f"{v:.1f}" for k, v in scores.to_dict().items() if isinstance(v, float)},
            )

            # Threshold check
            if scores.average >= config.quality_threshold:
                logger.info(
                    "%s '%s' met quality threshold (%.1f >= %.1f)",
                    entity_type.capitalize(),
                    get_name(entity),
                    scores.average,
                    config.quality_threshold,
                )
                history.final_iteration = current_iter
                history.final_score = scores.average
                svc._log_refinement_analytics(
                    history,
                    story_id,
                    threshold_met=True,
                    early_stop_triggered=False,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return entity, scores, current_iter

            # Early stopping check
            if history.should_stop_early(
                config.early_stopping_patience,
                min_iterations=config.early_stopping_min_iterations,
                variance_tolerance=config.early_stopping_variance_tolerance,
            ):
                early_stopped = True
                reason = (
                    f"plateaued for {history.consecutive_plateaus} consecutive iterations"
                    if history.consecutive_plateaus >= config.early_stopping_patience
                    else (f"degraded for {history.consecutive_degradations} consecutive iterations")
                )
                logger.info(
                    "Early stopping: %s '%s' quality %s (patience: %d). Stopping at iteration %d.",
                    entity_type.capitalize(),
                    get_name(entity),
                    reason,
                    config.early_stopping_patience,
                    iteration + 1,
                )
                break

        except WorldGenerationError as e:
            last_error = str(e)
            logger.warning(
                "%s generation error on iteration %d (already logged upstream): %s",
                entity_type.capitalize(),
                iteration + 1,
                str(e)[:200],
            )
            # Reset scores to prevent refining with stale feedback from a
            # previous iteration.  With scores=None the next iteration will
            # re-judge the current entity instead of blindly refining.
            scores = None

        iteration += 1

    # Post-loop: return best iteration
    if not history.iterations:
        raise WorldGenerationError(
            f"Failed to generate {entity_type} after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    best_entity_data = history.get_best_entity()

    if best_entity_data and history.iterations[-1].average_score < history.peak_score:
        logger.warning(
            "%s '%s' iterations got WORSE after peak. "
            "Best: iteration %d (%.1f), Final: iteration %d (%.1f). "
            "Returning best iteration.",
            entity_type.capitalize(),
            history.entity_name,
            history.best_iteration,
            history.peak_score,
            len(history.iterations),
            history.iterations[-1].average_score,
        )

    if best_entity_data:
        # Reconstruct scores from best iteration
        best_scores: S | None = None
        for record in history.iterations:
            if record.iteration == history.best_iteration:
                best_scores = score_cls(**record.scores)
                break
        if best_scores:
            history.final_iteration = history.best_iteration
            history.final_score = history.peak_score
            svc._log_refinement_analytics(
                history,
                story_id,
                threshold_met=history.peak_score >= config.quality_threshold,
                early_stop_triggered=early_stopped,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            # Reconstruct the entity from stored data
            # entity is guaranteed non-None here: we only reach this point
            # after at least one successful iteration recorded in history
            if entity is None:  # pragma: no cover — defensive
                raise WorldGenerationError(
                    f"Internal error: best entity data exists but entity is None for {entity_type}"
                )
            # Return total iteration count (not best_iteration index) so
            # callers can log "after N iteration(s)" accurately.
            return (
                _reconstruct_entity(best_entity_data, entity, entity_type),
                best_scores,
                len(history.iterations),
            )

    # Fallback to last iteration
    if entity is not None and not is_empty(entity) and scores is not None:
        logger.warning(
            "%s '%s' did not meet quality threshold (%.1f < %.1f), returning anyway",
            entity_type.capitalize(),
            get_name(entity),
            scores.average,
            config.quality_threshold,
        )
        history.final_iteration = len(history.iterations)
        history.final_score = scores.average
        svc._log_refinement_analytics(
            history,
            story_id,
            threshold_met=False,
            early_stop_triggered=early_stopped,
            quality_threshold=config.quality_threshold,
            max_iterations=config.max_iterations,
        )
        return entity, scores, len(history.iterations)

    raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
        f"Failed to generate {entity_type} after {config.max_iterations} attempts. "
        f"Last error: {last_error}"
    )


def _reconstruct_entity[T](entity_data: dict, current_entity: T, entity_type: str) -> T:
    """Reconstruct an entity from stored dict data.

    For Pydantic model entities (Character, PlotOutline, Chapter), reconstruct
    from the stored dict. For dict-based entities, return the dict directly.

    Args:
        entity_data: The serialized entity data from history.
        current_entity: The current entity instance (used for type detection).
        entity_type: The entity type string for logging.

    Returns:
        The reconstructed entity.
    """
    if isinstance(current_entity, dict):
        logger.debug("Reconstructing %s from dict data", entity_type)
        return cast(T, entity_data)
    # Pydantic model — reconstruct from dict
    entity_cls = type(current_entity)
    logger.debug("Reconstructing %s as %s from stored data", entity_type, entity_cls.__name__)
    return entity_cls(**entity_data)
