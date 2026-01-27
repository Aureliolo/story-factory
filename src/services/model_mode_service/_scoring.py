"""Score recording and quality judgment functions for ModelModeService."""

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from src.memory.mode_models import QualityScores
from src.services.llm_client import generate_structured
from src.utils.validation import validate_not_empty, validate_not_none, validate_positive

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger("src.services.model_mode_service._scoring")


def record_generation(
    svc: ModelModeService,
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

    Args:
        svc: The ModelModeService instance.
        project_id: The project ID.
        agent_role: The agent role.
        model_id: The model used.
        chapter_id: Optional chapter ID.
        genre: Optional genre.
        tokens_generated: Optional token count.
        time_seconds: Optional generation time.
        prompt_text: Optional prompt text for hashing.

    Returns:
        The score ID for later updates.
    """
    validate_not_empty(project_id, "project_id")
    validate_not_empty(agent_role, "agent_role")
    validate_not_empty(model_id, "model_id")
    mode = svc.get_current_mode()

    # Calculate tokens/second
    tokens_per_second = None
    if tokens_generated and time_seconds and time_seconds > 0:
        tokens_per_second = tokens_generated / time_seconds

    # Generate prompt hash for A/B comparisons
    prompt_hash = None
    if prompt_text:
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]

    try:
        score_id = svc._db.record_score(
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
    svc: ModelModeService,
    score_id: int,
    quality: QualityScores,
) -> None:
    """Update a score record with quality scores.

    Args:
        svc: The ModelModeService instance.
        score_id: The score record ID.
        quality: The quality scores to record.
    """
    validate_positive(score_id, "score_id")
    validate_not_none(quality, "quality")
    try:
        svc._db.update_score(
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
    svc: ModelModeService,
    score_id: int,
    *,
    was_regenerated: bool | None = None,
    edit_distance: int | None = None,
    user_rating: int | None = None,
) -> None:
    """Record an implicit quality signal.

    Args:
        svc: The ModelModeService instance.
        score_id: The score record ID.
        was_regenerated: Whether the content was regenerated.
        edit_distance: Edit distance from original.
        user_rating: User rating (1-5).
    """
    validate_positive(score_id, "score_id")
    try:
        svc._db.update_score(
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
    svc: ModelModeService,
    score_id: int,
    *,
    tokens_generated: int | None = None,
    time_seconds: float | None = None,
    tokens_per_second: float | None = None,
    vram_used_gb: float | None = None,
) -> None:
    """Update a score record with performance metrics.

    Args:
        svc: The ModelModeService instance.
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
        svc._db.update_performance_metrics(
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


def judge_quality(
    svc: ModelModeService,
    content: str,
    genre: str,
    tone: str,
    themes: list[str],
) -> QualityScores:
    """Use LLM to judge content quality.

    Args:
        svc: The ModelModeService instance.
        content: The generated content to evaluate.
        genre: Story genre.
        tone: Story tone.
        themes: Story themes.

    Returns:
        QualityScores with prose_quality and instruction_following.
    """
    validate_not_empty(content, "content")
    validate_not_empty(genre, "genre")
    validate_not_empty(tone, "tone")
    # Use validator model or smallest available
    judge_model = svc.get_model_for_agent("validator")
    logger.debug(f"Using {judge_model} to judge quality for {genre}/{tone}")

    # Limit content size for faster judging
    truncated_content = content[: svc.settings.content_truncation_for_judgment]

    prompt = f"""You are evaluating the quality of AI-generated story content.

**Story Brief:**
Genre: {genre}
Tone: {tone}
Themes: {", ".join(themes)}

**Content to evaluate:**
{truncated_content}

Rate each dimension from 0-10:

1. prose_quality: Creativity, flow, engagement, vocabulary variety
2. instruction_following: Adherence to genre, tone, themes"""

    try:
        scores = generate_structured(
            settings=svc.settings,
            model=judge_model,
            prompt=prompt,
            response_model=QualityScores,
            temperature=svc.settings.temp_capability_check,
        )
        logger.info(
            f"Quality judged: prose={scores.prose_quality:.1f}, "
            f"instruction={scores.instruction_following:.1f}"
        )
        return scores
    except Exception as e:
        logger.error(f"Quality judgment failed: {e}", exc_info=True)

    # Return neutral scores on failure
    logger.warning("Returning neutral quality scores (5.0) due to judgment failure")
    return QualityScores(prose_quality=5.0, instruction_following=5.0)


def calculate_consistency_score(issues: list[dict[str, Any]]) -> float:
    """Calculate consistency score from continuity issues.

    Args:
        issues: List of ContinuityIssue-like dicts with 'severity'.

    Returns:
        Score from 0-10 (10 = no issues).
    """
    if not issues:
        return 10.0

    # Weight by severity
    penalty = 0.0
    for issue in issues:
        severity = issue.get("severity", "minor")
        if severity == "critical":
            penalty += 3.0
        elif severity == "moderate":
            penalty += 1.5
        else:  # minor
            penalty += 0.5

    return max(0.0, 10.0 - penalty)
