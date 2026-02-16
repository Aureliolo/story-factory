"""Generic quality refinement loop for all entity types.

Extracts the common create-judge-refine loop that was previously copy-pasted
across 6 entity modules (~60 lines each). Each entity module now provides
entity-specific callables (create, judge, refine, serialize, get_name)
and this module orchestrates the loop.
"""

import logging
import time
from collections.abc import Callable
from typing import cast

from src.memory.world_quality import BaseQualityScores, RefinementConfig, RefinementHistory
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


def quality_refinement_loop[T, S: BaseQualityScores](
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
        Tuple of (best_entity, best_scores, scoring_rounds) where scoring_rounds
        is the number of successful judge calls.

    Raises:
        WorldGenerationError: If no valid entity could be produced after all attempts.
    """
    # Resolve per-entity threshold
    entity_threshold = config.get_threshold(entity_type)

    logger.info(
        "Starting quality refinement loop for %s (threshold=%.1f, max_iterations=%d, "
        "review_mode=%s)",
        entity_type,
        entity_threshold,
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
    needs_judging = False
    scoring_rounds = 0
    stage = ""  # Tracks current phase: "create", "judge", or "refine"

    while iteration < config.max_iterations:
        try:
            # Creation, refinement, or re-judge after previous judge error
            if needs_judging:
                # Entity already exists and needs judging (e.g. judge failed
                # on previous iteration). Skip creation/refinement and go
                # straight to judging.
                stage = "judge"
                logger.debug(
                    "%s '%s' re-judging after previous judge error on iteration %d",
                    entity_type.capitalize(),
                    get_name(entity) if entity is not None else "unknown",
                    iteration + 1,
                )
            elif iteration == 0 and initial_entity is not None:
                # Review mode: use the provided entity directly
                stage = "create"
                entity = initial_entity
                needs_judging = True
            elif entity is None or (iteration == 0) or is_empty(entity):
                # Need fresh creation
                stage = "create"
                t_create = time.perf_counter()
                entity = create_fn(creation_retries)
                logger.info(
                    "%s creation took %.2fs (iteration %d)",
                    entity_type.capitalize(),
                    time.perf_counter() - t_create,
                    iteration + 1,
                )
                if entity is None or is_empty(entity):
                    creation_retries += 1
                    last_error = (
                        f"{entity_type} creation returned empty on iteration {iteration + 1}"
                    )
                    logger.debug(
                        "%s (retry %d)",
                        last_error,
                        creation_retries,
                    )
                    iteration += 1
                    continue
                needs_judging = True
            else:
                # Refinement based on previous feedback
                if scores is not None:
                    stage = "refine"
                    dynamic_temp_iter = iteration + 1
                    t_refine = time.perf_counter()
                    entity = refine_fn(entity, scores, dynamic_temp_iter)
                    logger.info(
                        "%s refinement took %.2fs (iteration %d)",
                        entity_type.capitalize(),
                        time.perf_counter() - t_refine,
                        iteration + 1,
                    )
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

                    needs_judging = True
                else:  # pragma: no cover — defensive, unreachable with current error handler
                    # scores is None and not first iteration: entity is
                    # unchanged since the last error. Skip this iteration
                    # to avoid redundant judging (#266 B10). With the
                    # current error handler this path is unreachable:
                    # judge errors keep needs_judging=True (handled at
                    # top of loop), refinement errors preserve scores.
                    logger.debug(
                        "%s '%s' unchanged after error on iteration %d, "
                        "skipping redundant judge call",
                        entity_type.capitalize(),
                        get_name(entity) if entity is not None else "unknown",
                        iteration + 1,
                    )
                    iteration += 1
                    continue

            # Update history entity name on first successful entity
            if not history.entity_name and entity is not None:
                history.entity_name = get_name(entity)

            # At this point entity is guaranteed non-None and needs_judging is
            # True: all code paths above either set needs_judging=True, or
            # continue/break out of this iteration.
            assert entity is not None  # nosec — invariant, not runtime check

            stage = "judge"
            t_judge = time.perf_counter()
            scores = judge_fn(entity)
            judge_duration = time.perf_counter() - t_judge
            scoring_rounds += 1
            needs_judging = False
            logger.info(
                "%s judge call took %.2fs (scoring round %d)",
                entity_type.capitalize(),
                judge_duration,
                scoring_rounds,
            )

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

            # Threshold check — round to 1 decimal so the comparison matches
            # the %.1f log display (e.g. 7.46 rounds to 7.5, passes >= 7.5).
            rounded_score = round(scores.average, 1)
            if rounded_score >= entity_threshold:
                logger.info(
                    "%s '%s' met quality threshold (%.1f >= %.1f)",
                    entity_type.capitalize(),
                    get_name(entity),
                    scores.average,
                    entity_threshold,
                )
                history.final_iteration = current_iter
                history.final_score = scores.average
                svc._log_refinement_analytics(
                    history,
                    story_id,
                    threshold_met=True,
                    early_stop_triggered=False,
                    quality_threshold=entity_threshold,
                    max_iterations=config.max_iterations,
                )
                return entity, scores, scoring_rounds

            # Score-plateau early-stop: consecutive identical scores (#328)
            if (
                len(history.iterations) >= max(2, config.early_stopping_min_iterations)
                and abs(scores.average - history.iterations[-2].average_score)
                <= config.score_plateau_tolerance
            ):
                early_stopped = True
                logger.info(
                    "Early stop: %s '%s' score plateaued at %.1f "
                    "(no improvement from scoring round %d to %d)",
                    entity_type.capitalize(),
                    get_name(entity),
                    scores.average,
                    scoring_rounds - 1,
                    scoring_rounds,
                )
                break

            # Early stopping check — use scoring_rounds for min_iterations (#266 C3)
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
                    "Early stopping: %s '%s' quality %s (patience: %d). "
                    "Stopping at iteration %d (scoring round %d).",
                    entity_type.capitalize(),
                    get_name(entity),
                    reason,
                    config.early_stopping_patience,
                    iteration + 1,
                    scoring_rounds,
                )
                break

        except WorldGenerationError as e:
            last_error = str(e)[:200]
            logger.warning(
                "%s %s error on iteration %d (already logged upstream): %s",
                entity_type.capitalize(),
                stage,
                iteration + 1,
                last_error,
            )
            if stage == "judge":
                # Error occurred during judging — entity was successfully
                # created/refined but the judge call failed. Keep
                # needs_judging=True so the next iteration re-judges the
                # same entity rather than trying to refine without scores.
                scores = None
                logger.debug(
                    "%s '%s' judge failed, will re-judge on next iteration",
                    entity_type.capitalize(),
                    get_name(entity) if entity is not None else "unknown",
                )
            elif stage == "refine":
                # Error occurred during refinement — entity is unchanged
                # from before this iteration. Don't re-judge an unchanged
                # entity (#266 B10). Keep existing scores (still valid for
                # the current entity) so refinement can be retried.
                history.failed_refinements += 1
                logger.debug(
                    "%s '%s' refinement failed, entity unchanged, skipping redundant re-judge",
                    entity_type.capitalize(),
                    get_name(entity) if entity is not None else "unknown",
                )
            else:
                # Error occurred during creation — no entity exists yet,
                # increment creation_retries so next iteration retries
                # with escalated temperature.
                creation_retries += 1
                logger.debug(
                    "%s creation failed (retry %d), will retry",
                    entity_type.capitalize(),
                    creation_retries,
                )

        iteration += 1

    # Post-loop: return best iteration using scoring_rounds (#266 C3)
    if not history.iterations:
        raise WorldGenerationError(
            f"Failed to generate {entity_type} after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    # Hail-mary: when threshold not met and config allows multiple iterations,
    # try one fresh creation to see if it beats the best refinement result.
    best_entity_data = history.get_best_entity()
    threshold_met_pre_hail_mary = round(history.peak_score, 1) >= entity_threshold

    if not threshold_met_pre_hail_mary and config.max_iterations > 1:
        logger.info(
            "%s '%s': threshold not met (best=%.1f < %.1f), attempting hail-mary fresh creation",
            entity_type.capitalize(),
            history.entity_name,
            history.peak_score,
            entity_threshold,
        )
        try:
            fresh_entity = create_fn(creation_retries + 1)
            if fresh_entity is not None and not is_empty(fresh_entity):
                fresh_scores = judge_fn(fresh_entity)
                scoring_rounds += 1
                logger.info(
                    "%s hail-mary scored %.1f (previous best: %.1f)",
                    entity_type.capitalize(),
                    fresh_scores.average,
                    history.peak_score,
                )
                if fresh_scores.average > history.peak_score:
                    logger.info(
                        "%s hail-mary beats previous best! Using fresh entity.",
                        entity_type.capitalize(),
                    )
                    # Add to history and use as best
                    history.add_iteration(
                        entity_data=serialize(fresh_entity),
                        scores=fresh_scores.to_dict(),
                        average_score=fresh_scores.average,
                        feedback=fresh_scores.feedback,
                    )
                    # Update best entity data
                    best_entity_data = history.get_best_entity()
                    entity = fresh_entity
                    scores = fresh_scores
                else:
                    logger.info(
                        "%s hail-mary did not beat best score, keeping original",
                        entity_type.capitalize(),
                    )
            else:
                logger.info(
                    "%s hail-mary returned empty entity, keeping original",
                    entity_type.capitalize(),
                )
        except (WorldGenerationError, ValueError, KeyError) as e:
            logger.warning(
                "%s hail-mary failed: %s. Keeping original best.",
                entity_type.capitalize(),
                str(e)[:200],
            )

    if best_entity_data and history.iterations[-1].average_score < history.peak_score:
        logger.warning(
            "%s '%s' iterations got WORSE after peak. "
            "Best: iteration %d (%.1f), Final: iteration %d (%.1f). "
            "Returning best iteration.",
            entity_type.capitalize(),
            history.entity_name,
            history.best_iteration,
            history.peak_score,
            scoring_rounds,
            history.iterations[-1].average_score,
        )

    if best_entity_data:
        threshold_met = round(history.peak_score, 1) >= entity_threshold
        if not threshold_met:
            logger.warning(
                "%s '%s' did not meet quality threshold after %d scoring round(s) "
                "(best: %.1f, threshold: %.1f). Returning best iteration %d.",
                entity_type.capitalize(),
                history.entity_name,
                scoring_rounds,
                history.peak_score,
                entity_threshold,
                history.best_iteration,
            )

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
                threshold_met=threshold_met,
                early_stop_triggered=early_stopped,
                quality_threshold=entity_threshold,
                max_iterations=config.max_iterations,
            )
            # Reconstruct the entity from stored data
            # entity is guaranteed non-None here: we only reach this point
            # after at least one successful iteration recorded in history
            if entity is None:  # pragma: no cover — defensive
                raise WorldGenerationError(
                    f"Internal error: best entity data exists but entity is None for {entity_type}"
                )
            # Return scoring_rounds (not loop iteration count) so callers
            # get an accurate count of actual judge calls (#266 C3).
            return (
                _reconstruct_entity(best_entity_data, entity, entity_type),
                best_scores,
                scoring_rounds,
            )

    # Fallback to last iteration
    if entity is not None and not is_empty(entity) and scores is not None:
        logger.warning(
            "%s '%s' did not meet quality threshold (%.1f < %.1f), returning anyway",
            entity_type.capitalize(),
            get_name(entity),
            scores.average,
            entity_threshold,
        )
        history.final_iteration = scoring_rounds
        history.final_score = scores.average
        svc._log_refinement_analytics(
            history,
            story_id,
            threshold_met=False,
            early_stop_triggered=early_stopped,
            quality_threshold=entity_threshold,
            max_iterations=config.max_iterations,
        )
        return entity, scores, scoring_rounds

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
