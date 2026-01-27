"""Learning and tuning mixin for ModelModeService."""

import logging

from src.memory.mode_models import (
    AutonomyLevel,
    LearningSettings,
    LearningTrigger,
    RecommendationType,
    TuningRecommendation,
)

from ._base import ModelModeServiceBase

logger = logging.getLogger(__name__)


class LearningMixin(ModelModeServiceBase):
    """Mixin providing learning and tuning functionality."""

    def set_learning_settings(self, settings: LearningSettings) -> None:
        """Update learning settings."""
        self._learning_settings = settings

    def get_learning_settings(self) -> LearningSettings:
        """Get current learning settings."""
        return self._learning_settings

    def should_tune(self) -> bool:
        """Check if tuning analysis should run based on triggers."""
        triggers = self._learning_settings.triggers

        if LearningTrigger.OFF in triggers:
            return False

        if LearningTrigger.CONTINUOUS in triggers:
            return True

        if LearningTrigger.PERIODIC in triggers:
            if self._chapters_since_analysis >= self._learning_settings.periodic_interval:
                return True

        return False

    def on_chapter_complete(self) -> None:
        """Called when a chapter is completed."""
        self._chapters_since_analysis += 1

    def on_project_complete(self) -> list[TuningRecommendation]:
        """Called when a project is completed.

        Returns recommendations if after_project trigger is enabled.
        """
        self._chapters_since_analysis = 0

        if LearningTrigger.AFTER_PROJECT in self._learning_settings.triggers:
            return self.get_recommendations()

        return []

    def get_recommendations(self) -> list[TuningRecommendation]:
        """Generate tuning recommendations based on historical data.

        Returns:
            List of recommendations, may be empty if insufficient data.
        """
        recommendations: list[TuningRecommendation] = []
        min_samples = self._learning_settings.min_samples_for_recommendation

        # Check if we have enough data
        total_scores = self._db.get_score_count()
        if total_scores < min_samples:
            logger.debug(f"Not enough samples for recommendations: {total_scores}")
            return []

        # Get current mode
        mode = self.get_current_mode()

        # Analyze each agent role
        for role in ["writer", "architect", "editor", "continuity"]:
            current_model = mode.agent_models.get(role)
            if not current_model:
                continue

            # Get top performers for this role
            top_models = self._db.get_top_models_for_role(role, limit=3, min_samples=min_samples)

            if not top_models:
                continue

            # Check if there's a better option
            best = top_models[0]
            if best["model_id"] != current_model:
                # Get current model's performance
                current_perf = self._db.get_model_performance(
                    model_id=current_model, agent_role=role
                )

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

        self._chapters_since_analysis = 0
        return recommendations

    def apply_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """
        Apply a tuning recommendation to the current mode and persist its outcome when applicable.

        Parameters:
            recommendation (TuningRecommendation): Recommendation describing the action (e.g., model swap or temperature adjustment),
                the affected role, and the suggested value; if `recommendation.id` is present the outcome will be recorded.

        Returns:
            bool: `True` if the recommendation was applied to the in-memory current mode and (when present) recorded in the database,
            `False` otherwise.
        """
        try:
            if recommendation.recommendation_type == "model_swap":
                if recommendation.affected_role and self._current_mode:
                    self._current_mode.agent_models[recommendation.affected_role] = (
                        recommendation.suggested_value
                    )
                    logger.info(
                        f"Applied recommendation: {recommendation.affected_role} "
                        f"now uses {recommendation.suggested_value}"
                    )

                    # Record outcome
                    if recommendation.id:
                        self._db.update_recommendation_outcome(
                            recommendation.id,
                            was_applied=True,
                            user_feedback="accepted",
                        )
                    return True

            elif recommendation.recommendation_type == "temp_adjust":
                if recommendation.affected_role and self._current_mode:
                    self._current_mode.agent_temperatures[recommendation.affected_role] = float(
                        recommendation.suggested_value
                    )
                    logger.info(
                        f"Applied recommendation: {recommendation.affected_role} "
                        f"temperature now {recommendation.suggested_value}"
                    )

                    # Record outcome (same as model_swap to prevent resurface)
                    if recommendation.id:
                        self._db.update_recommendation_outcome(
                            recommendation.id,
                            was_applied=True,
                            user_feedback="accepted",
                        )
                    return True

        except Exception as e:
            logger.error(f"Failed to apply recommendation: {e}")

        return False

    def handle_recommendations(
        self,
        recommendations: list[TuningRecommendation],
    ) -> list[TuningRecommendation]:
        """Handle recommendations based on autonomy level.

        Returns:
            Recommendations that were not auto-applied (need user approval).
        """
        autonomy = self._learning_settings.autonomy
        pending = []

        for rec in recommendations:
            # Save to database
            rec_id = self._db.record_recommendation(
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
                should_apply = rec.recommendation_type == "temp_adjust"
            elif autonomy == AutonomyLevel.BALANCED:
                should_apply = rec.confidence >= self._learning_settings.confidence_threshold
            elif autonomy in (AutonomyLevel.AGGRESSIVE, AutonomyLevel.EXPERIMENTAL):
                should_apply = True

            if should_apply:
                self.apply_recommendation(rec)
            else:
                pending.append(rec)

        return pending
