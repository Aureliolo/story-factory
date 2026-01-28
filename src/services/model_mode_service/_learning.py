"""Learning and tuning recommendation functions for ModelModeService."""

import logging
from typing import TYPE_CHECKING

from src.memory.mode_models import (
    AutonomyLevel,
    LearningSettings,
    LearningTrigger,
    RecommendationType,
    TuningRecommendation,
)

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger(__name__)


def set_learning_settings(svc: ModelModeService, settings: LearningSettings) -> None:
    """Update learning settings.

    Args:
        svc: The ModelModeService instance.
        settings: New learning settings.
    """
    logger.debug("set_learning_settings called: settings=%s", settings)
    svc._learning_settings = settings


def get_learning_settings(svc: ModelModeService) -> LearningSettings:
    """Get current learning settings.

    Args:
        svc: The ModelModeService instance.

    Returns:
        Current LearningSettings.
    """
    logger.debug("get_learning_settings called")
    return svc._learning_settings


def should_tune(svc: ModelModeService) -> bool:
    """Check if tuning analysis should run based on triggers.

    Args:
        svc: The ModelModeService instance.

    Returns:
        True if tuning analysis should run.
    """
    logger.debug("should_tune called")
    triggers = svc._learning_settings.triggers

    if LearningTrigger.OFF in triggers:
        return False

    if LearningTrigger.CONTINUOUS in triggers:
        return True

    if LearningTrigger.PERIODIC in triggers:
        if svc._chapters_since_analysis >= svc._learning_settings.periodic_interval:
            return True

    return False


def on_chapter_complete(svc: ModelModeService) -> None:
    """Called when a chapter is completed.

    Args:
        svc: The ModelModeService instance.
    """
    logger.debug("on_chapter_complete called")
    svc._chapters_since_analysis += 1


def on_project_complete(svc: ModelModeService) -> list[TuningRecommendation]:
    """Called when a project is completed.

    Returns recommendations if after_project trigger is enabled.

    Args:
        svc: The ModelModeService instance.

    Returns:
        List of tuning recommendations, may be empty.
    """
    logger.debug("on_project_complete called")
    svc._chapters_since_analysis = 0

    if LearningTrigger.AFTER_PROJECT in svc._learning_settings.triggers:
        return get_recommendations(svc)

    return []


def get_recommendations(svc: ModelModeService) -> list[TuningRecommendation]:
    """Generate tuning recommendations based on historical data.

    Args:
        svc: The ModelModeService instance.

    Returns:
        List of recommendations, may be empty if insufficient data.
    """
    logger.debug("get_recommendations called")
    recommendations: list[TuningRecommendation] = []
    min_samples = svc._learning_settings.min_samples_for_recommendation

    # Check if we have enough data
    total_scores = svc._db.get_score_count()
    if total_scores < min_samples:
        logger.debug(f"Not enough samples for recommendations: {total_scores}")
        return []

    # Get current mode
    mode = svc.get_current_mode()

    # Analyze each agent role
    for role in ["writer", "architect", "editor", "continuity"]:
        current_model = mode.agent_models.get(role)
        if not current_model:
            continue

        # Get top performers for this role
        top_models = svc._db.get_top_models_for_role(role, limit=3, min_samples=min_samples)

        if not top_models:
            continue

        # Check if there's a better option
        best = top_models[0]
        if best["model_id"] != current_model:
            # Get current model's performance
            current_perf = svc._db.get_model_performance(model_id=current_model, agent_role=role)

            if current_perf:
                current_quality = current_perf[0].get("avg_prose_quality", 0) or 0
                best_quality = best.get("avg_prose_quality", 0) or 0

                if best_quality > current_quality:
                    improvement = (
                        (best_quality - current_quality) / current_quality * 100
                        if current_quality > 0
                        else 0
                    )
                    confidence = min(
                        0.95,
                        best["sample_count"] / (min_samples * 3),
                    )

                    rec = TuningRecommendation(
                        recommendation_type=RecommendationType.MODEL_SWAP,
                        current_value=current_model,
                        suggested_value=best["model_id"],
                        affected_role=role,
                        reason=(
                            f"{best['model_id']} scores {best_quality:.1f} avg "
                            f"vs {current_quality:.1f} for {role}"
                        ),
                        confidence=confidence,
                        evidence={
                            "current_quality": current_quality,
                            "suggested_quality": best_quality,
                            "sample_count": best["sample_count"],
                        },
                        expected_improvement=f"+{improvement:.0f}% quality",
                    )
                    recommendations.append(rec)

    svc._chapters_since_analysis = 0
    return recommendations


def apply_recommendation(svc: ModelModeService, recommendation: TuningRecommendation) -> bool:
    """Apply a tuning recommendation to the current mode.

    Persists its outcome when applicable.

    Args:
        svc: The ModelModeService instance.
        recommendation: The recommendation to apply.

    Returns:
        True if the recommendation was applied, False otherwise.
    """
    try:
        if recommendation.recommendation_type == RecommendationType.MODEL_SWAP:
            if recommendation.affected_role and svc._current_mode:
                svc._current_mode.agent_models[recommendation.affected_role] = (
                    recommendation.suggested_value
                )
                logger.info(
                    f"Applied recommendation: {recommendation.affected_role} "
                    f"now uses {recommendation.suggested_value}"
                )

                # Record outcome
                if recommendation.id:
                    svc._db.update_recommendation_outcome(
                        recommendation.id,
                        was_applied=True,
                        user_feedback="accepted",
                    )
                return True

        elif recommendation.recommendation_type == RecommendationType.TEMP_ADJUST:
            if recommendation.affected_role and svc._current_mode:
                svc._current_mode.agent_temperatures[recommendation.affected_role] = float(
                    recommendation.suggested_value
                )
                logger.info(
                    f"Applied recommendation: {recommendation.affected_role} "
                    f"temperature now {recommendation.suggested_value}"
                )

                # Record outcome (same as model_swap to prevent resurface)
                if recommendation.id:
                    svc._db.update_recommendation_outcome(
                        recommendation.id,
                        was_applied=True,
                        user_feedback="accepted",
                    )
                return True

    except Exception as e:
        logger.error(f"Failed to apply recommendation: {e}")

    return False


def handle_recommendations(
    svc: ModelModeService,
    recommendations: list[TuningRecommendation],
) -> list[TuningRecommendation]:
    """Handle recommendations based on autonomy level.

    Args:
        svc: The ModelModeService instance.
        recommendations: List of recommendations to handle.

    Returns:
        Recommendations that were not auto-applied (need user approval).
    """
    logger.debug("handle_recommendations called: recommendations_count=%s", len(recommendations))
    autonomy = svc._learning_settings.autonomy
    pending = []

    for rec in recommendations:
        # Save to database
        rec_id = svc._db.record_recommendation(
            recommendation_type=rec.recommendation_type.value
            if hasattr(rec.recommendation_type, "value")
            else rec.recommendation_type,
            current_value=rec.current_value,
            suggested_value=rec.suggested_value,
            reason=rec.reason,
            confidence=rec.confidence,
            evidence=rec.evidence,
            affected_role=rec.affected_role,
            expected_improvement=rec.expected_improvement,
        )
        rec.id = rec_id

        # Decide whether to auto-apply
        should_apply = False

        if autonomy == AutonomyLevel.MANUAL:
            should_apply = False
        elif autonomy == AutonomyLevel.CAUTIOUS:
            should_apply = rec.recommendation_type == RecommendationType.TEMP_ADJUST
        elif autonomy == AutonomyLevel.BALANCED:
            should_apply = rec.confidence >= svc._learning_settings.confidence_threshold
        elif autonomy in (AutonomyLevel.AGGRESSIVE, AutonomyLevel.EXPERIMENTAL):
            should_apply = True

        if should_apply:
            apply_recommendation(svc, rec)
        else:
            pending.append(rec)

    return pending
