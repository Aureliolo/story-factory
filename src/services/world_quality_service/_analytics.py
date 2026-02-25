"""Analytics recording logic for WorldQualityService.

Handles the recording of entity quality scores and refinement analytics
to the analytics database.
"""

import logging
from typing import TYPE_CHECKING, Any

from src.memory.world_quality import RefinementHistory
from src.utils.validation import validate_not_empty

if TYPE_CHECKING:
    from src.services.world_quality_service import WorldQualityService

logger = logging.getLogger(__name__)


def record_entity_quality(
    service: WorldQualityService,
    project_id: str,
    entity_type: str,
    entity_name: str,
    scores: dict[str, Any],
    iterations: int,
    generation_time: float,
    model_id: str | None = None,
    *,
    early_stop_triggered: bool = False,
    threshold_met: bool = False,
    peak_score: float | None = None,
    final_score: float | None = None,
    score_progression: list[float] | None = None,
    consecutive_degradations: int = 0,
    best_iteration: int = 0,
    quality_threshold: float | None = None,
    max_iterations: int | None = None,
    below_threshold_admitted: bool = False,
) -> None:
    """Persist entity quality scores and refinement metrics to the analytics database.

    Parameters:
        service: The WorldQualityService instance.
        project_id: Project identifier.
        entity_type: Entity category (e.g., "character", "location", "faction").
        entity_name: Name of the entity being recorded.
        scores: Dictionary of quality metrics.
        iterations: Number of refinement iterations performed.
        generation_time: Total generation time in seconds.
        model_id: Identifier of the model used.
        early_stop_triggered: Whether the refinement loop stopped early.
        threshold_met: Whether the configured quality threshold was reached.
        peak_score: Highest score observed across iterations.
        final_score: Score of the final returned entity.
        score_progression: Sequence of scores per iteration.
        consecutive_degradations: Count of consecutive iterations with decreasing scores.
        best_iteration: Iteration index that produced the best observed score.
        quality_threshold: Quality threshold used to judge success.
        max_iterations: Maximum allowed refinement iterations.
        below_threshold_admitted: Whether entity was admitted below quality threshold.
    """
    validate_not_empty(project_id, "project_id")
    validate_not_empty(entity_type, "entity_type")
    validate_not_empty(entity_name, "entity_name")

    # Determine model_id if not provided
    if model_id is None:
        model_id = service._get_creator_model(entity_type)

    try:
        service.analytics_db.record_world_entity_score(
            project_id=project_id,
            entity_type=entity_type,
            entity_name=entity_name,
            model_id=model_id,
            scores=scores,
            iterations_used=iterations,
            generation_time_seconds=generation_time,
            feedback=scores.get("feedback", ""),
            early_stop_triggered=early_stop_triggered,
            threshold_met=threshold_met,
            peak_score=peak_score,
            final_score=final_score,
            score_progression=score_progression,
            consecutive_degradations=consecutive_degradations,
            best_iteration=best_iteration,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
            below_threshold_admitted=below_threshold_admitted,
        )
        logger.debug(
            f"Recorded {entity_type} '{entity_name}' quality to analytics "
            f"(model={model_id}, avg: {scores.get('average', 0):.1f}, "
            f"threshold_met={threshold_met}, early_stop={early_stop_triggered})"
        )
    except Exception as e:
        # Don't fail generation if analytics recording fails
        logger.warning(
            "Failed to record %s '%s' quality to analytics (project=%s, model=%s): %s",
            entity_type,
            entity_name,
            project_id,
            model_id,
            e,
            exc_info=True,
        )


def log_refinement_analytics(
    service: WorldQualityService,
    history: RefinementHistory,
    project_id: str,
    *,
    early_stop_triggered: bool = False,
    threshold_met: bool = False,
    quality_threshold: float | None = None,
    max_iterations: int | None = None,
) -> None:
    """
    Record refinement iteration analytics for a completed RefinementHistory, log an informational summary, and persist extended scores and refinement metrics to the analytics database.

    Parameters:
        history: The RefinementHistory for the entity being reported.
        project_id: Project identifier under which analytics should be recorded.
        early_stop_triggered: True if the refinement loop stopped early.
        threshold_met: True if the configured quality threshold was reached.
        quality_threshold: The numeric quality threshold used, if any.
        max_iterations: The configured maximum number of refinement iterations, if any.
    """
    analysis = history.analyze_improvement()

    # Condensed single-line log for entities that passed on the first try
    if len(history.iterations) == 1 and threshold_met:
        logger.info(
            "REFINEMENT ANALYTICS [%s] '%s': score=%.1f, iterations=1, passed=True",
            history.entity_type,
            history.entity_name,
            history.final_score,
        )
    else:
        logger.info(
            f"REFINEMENT ANALYTICS [{history.entity_type}] '{history.entity_name}':\n"
            f"  - Scoring rounds: {analysis['scoring_rounds']}\n"
            f"  - Score progression: "
            f"{' -> '.join(f'{s:.1f}' for s in analysis['score_progression'])}\n"
            f"  - Best iteration: {analysis['best_iteration']} ({history.peak_score:.1f})\n"
            f"  - Final returned: iteration {history.final_iteration} "
            f"({history.final_score:.1f})\n"
            f"  - Improved over first: {analysis['improved']}\n"
            f"  - Worsened after peak: {analysis['worsened_after_peak']}\n"
            f"  - Mid-loop regression: {analysis['mid_loop_regression']}\n"
            f"  - Threshold met: {threshold_met}\n"
            f"  - Below threshold admitted: {history.below_threshold_admitted}\n"
            f"  - Early stop triggered: {early_stop_triggered}"
        )

    # Record to analytics database with extended scores info
    extended_scores = {
        "final_score": history.final_score,
        "peak_score": history.peak_score,
        "best_iteration": analysis["best_iteration"],
        "improved": analysis["improved"],
        "worsened_after_peak": analysis["worsened_after_peak"],
        "below_threshold_admitted": history.below_threshold_admitted,
        "average": history.final_score,  # For backwards compatibility
    }
    record_entity_quality(
        service,
        project_id=project_id,
        entity_type=history.entity_type,
        entity_name=history.entity_name,
        scores=extended_scores,
        iterations=analysis["scoring_rounds"],
        generation_time=0.0,  # Not tracked at this level
        early_stop_triggered=early_stop_triggered,
        threshold_met=threshold_met,
        peak_score=history.peak_score,
        final_score=history.final_score,
        score_progression=analysis["score_progression"],
        consecutive_degradations=history.consecutive_degradations,
        best_iteration=analysis["best_iteration"],
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
        below_threshold_admitted=history.below_threshold_admitted,
    )
