"""Score recording mixin for ModelModeService."""

import hashlib
import logging

from src.memory.mode_models import QualityScores
from src.utils.validation import validate_not_empty, validate_not_none, validate_positive

from ._base import ModelModeServiceBase

logger = logging.getLogger(__name__)


class ScoringMixin(ModelModeServiceBase):
    """Mixin providing score recording functionality."""

    def record_generation(
        self,
        project_id: str,
        agent_role: str,
        model_id: str,
        *,
        chapter_id: str | None = None,
        genre: str | None = None,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        prompt_text: str | None = None,
    ) -> int:
        """Record a generation event.

        Returns:
            The score ID for later updates.
        """
        validate_not_empty(project_id, "project_id")
        validate_not_empty(agent_role, "agent_role")
        validate_not_empty(model_id, "model_id")
        mode = self.get_current_mode()

        # Calculate tokens/second
        tokens_per_second = None
        if tokens_generated and time_seconds and time_seconds > 0:
            tokens_per_second = tokens_generated / time_seconds

        # Generate prompt hash for A/B comparisons
        prompt_hash = None
        if prompt_text:
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]

        try:
            score_id = self._db.record_score(
                project_id=project_id,
                agent_role=agent_role,
                model_id=model_id,
                mode_name=mode.id,
                chapter_id=chapter_id,
                genre=genre,
                tokens_generated=tokens_generated,
                time_seconds=time_seconds,
                tokens_per_second=tokens_per_second,
                prompt_hash=prompt_hash,
            )

            speed_display = f"{tokens_per_second:.1f}" if tokens_per_second else "N/A"
            time_display = f"{time_seconds:.1f}" if time_seconds is not None else "N/A"

            logger.info(
                f"Recorded generation score {score_id}: {agent_role}/{model_id} "
                f"(mode={mode.id}, tokens={tokens_generated}, time={time_display}s, "
                f"speed={speed_display} t/s)"
            )
            return score_id

        except Exception as e:
            logger.error(
                f"Failed to record generation for {agent_role}/{model_id}: {e}",
                exc_info=True,
            )
            raise

    def update_quality_scores(
        self,
        score_id: int,
        quality: QualityScores,
    ) -> None:
        """Update a score record with quality scores."""
        validate_positive(score_id, "score_id")
        validate_not_none(quality, "quality")
        try:
            self._db.update_score(
                score_id,
                prose_quality=quality.prose_quality,
                instruction_following=quality.instruction_following,
                consistency_score=quality.consistency_score,
            )
            logger.debug(
                f"Updated quality scores for {score_id}: "
                f"prose={quality.prose_quality}, instruction={quality.instruction_following}, "
                f"consistency={quality.consistency_score}"
            )
        except Exception as e:
            logger.error(f"Failed to update quality scores for {score_id}: {e}", exc_info=True)
            raise

    def record_implicit_signal(
        self,
        score_id: int,
        *,
        was_regenerated: bool | None = None,
        edit_distance: int | None = None,
        user_rating: int | None = None,
    ) -> None:
        """Record an implicit quality signal."""
        validate_positive(score_id, "score_id")
        try:
            self._db.update_score(
                score_id,
                was_regenerated=was_regenerated,
                edit_distance=edit_distance,
                user_rating=user_rating,
            )
            signals = []
            if was_regenerated:
                signals.append("regenerated")
            if edit_distance is not None:
                signals.append(f"edited({edit_distance} chars)")
            if user_rating is not None:
                signals.append(f"rated({user_rating}/5)")

            logger.debug(f"Recorded signals for score {score_id}: {', '.join(signals)}")
        except Exception as e:
            logger.error(f"Failed to record implicit signal for {score_id}: {e}", exc_info=True)
            raise

    def update_performance_metrics(
        self,
        score_id: int,
        *,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        tokens_per_second: float | None = None,
        vram_used_gb: float | None = None,
    ) -> None:
        """Update a score record with performance metrics.

        Args:
            score_id: The score record ID.
            tokens_generated: Number of tokens generated.
            time_seconds: Generation time in seconds.
            tokens_per_second: Generation speed (calculated if not provided).
            vram_used_gb: VRAM used during generation.
        """
        validate_positive(score_id, "score_id")
        # Calculate tokens_per_second if not provided
        if tokens_per_second is None and tokens_generated and time_seconds and time_seconds > 0:
            tokens_per_second = tokens_generated / time_seconds

        try:
            self._db.update_performance_metrics(
                score_id,
                tokens_generated=tokens_generated,
                time_seconds=time_seconds,
                tokens_per_second=tokens_per_second,
                vram_used_gb=vram_used_gb,
            )
            time_display = f"{time_seconds:.1f}" if time_seconds is not None else "N/A"
            speed_display = f"{tokens_per_second:.1f}" if tokens_per_second else "N/A"
            logger.debug(
                f"Updated performance metrics for {score_id}: "
                f"tokens={tokens_generated}, time={time_display}s, "
                f"speed={speed_display} t/s"
            )
        except Exception as e:
            logger.error(f"Failed to update performance metrics for {score_id}: {e}", exc_info=True)
            raise
