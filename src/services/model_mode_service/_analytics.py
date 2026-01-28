"""Analytics and recommendation retrieval functions for ModelModeService."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.memory.mode_models import RecommendationType, TuningRecommendation
from src.utils.validation import validate_not_empty, validate_positive

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger(__name__)


def get_quality_vs_speed_data(
    svc: ModelModeService,
    agent_role: str | None = None,
) -> list[dict[str, Any]]:
    """Get data for quality vs speed scatter plot.

    Args:
        svc: The ModelModeService instance.
        agent_role: Optional filter by agent role.

    Returns:
        List of data points with quality and speed metrics.
    """
    logger.debug("get_quality_vs_speed_data called: agent_role=%s", agent_role)
    return svc._db.get_quality_vs_speed_data(agent_role)


def get_model_performance(
    svc: ModelModeService,
    model_id: str | None = None,
    agent_role: str | None = None,
) -> list[dict[str, Any]]:
    """Get aggregated model performance.

    Args:
        svc: The ModelModeService instance.
        model_id: Optional filter by model ID.
        agent_role: Optional filter by agent role.

    Returns:
        List of performance data dicts.
    """
    logger.debug("get_model_performance called: model_id=%s, agent_role=%s", model_id, agent_role)
    return svc._db.get_model_performance(model_id, agent_role)


def get_recommendation_history(svc: ModelModeService, limit: int = 50) -> list[dict[str, Any]]:
    """Get recommendation history.

    Args:
        svc: The ModelModeService instance.
        limit: Maximum number of recommendations to return.

    Returns:
        List of recommendation history records.
    """
    validate_positive(limit, "limit")
    return svc._db.get_recommendation_history(limit)


def export_scores_csv(svc: ModelModeService, output_path: Path | str) -> int:
    """Export all recorded scores to a CSV file at the given path.

    Args:
        svc: The ModelModeService instance.
        output_path: Destination file path for the exported CSV.

    Returns:
        Number of score records written to the CSV.
    """
    return svc._db.export_scores_csv(output_path)


def get_pending_recommendations(svc: ModelModeService) -> list[TuningRecommendation]:
    """Return pending tuning recommendations retrieved from the database.

    Each database row is converted into a TuningRecommendation object:
    - Timestamps stored as ISO strings are parsed to datetimes (falls back to now if missing).
    - Recommendation types are converted to the RecommendationType enum.
    - Evidence JSON is parsed when present; malformed JSON is logged and evidence is left as None.
    - Rows that fail to parse are skipped with a warning.

    Args:
        svc: The ModelModeService instance.

    Returns:
        Parsed pending recommendations (may be empty).
    """
    rows = svc._db.get_pending_recommendations(limit=20)
    recommendations = []
    for row in rows:
        row_id = row.get("id")
        try:
            # Validate required fields
            required = ["current_value", "suggested_value", "reason", "confidence"]
            missing = [k for k in required if row.get(k) is None]
            if missing:
                logger.warning(f"Skipping recommendation {row_id}: missing {', '.join(missing)}")
                continue

            # Parse timestamp from string (SQLite stores as ISO format)
            timestamp_raw = row.get("timestamp")
            if timestamp_raw:
                timestamp = datetime.fromisoformat(str(timestamp_raw))
            else:
                timestamp = datetime.now()

            # Parse recommendation_type from string
            rec_type = RecommendationType(str(row.get("recommendation_type")))

            # Parse evidence from JSON (DB column is evidence_json)
            evidence = None
            evidence_json = row.get("evidence_json")
            if evidence_json:
                try:
                    evidence = json.loads(evidence_json)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse evidence JSON for recommendation {row_id}")

            rec = TuningRecommendation(
                id=row_id,
                timestamp=timestamp,
                recommendation_type=rec_type,
                current_value=str(row["current_value"]),
                suggested_value=str(row["suggested_value"]),
                affected_role=row.get("affected_role"),
                reason=str(row["reason"]),
                confidence=float(row["confidence"]),
                evidence=evidence,
                expected_improvement=row.get("expected_improvement"),
            )
            recommendations.append(rec)
        except Exception as e:
            logger.warning(f"Failed to parse recommendation {row_id}: {e}")
    return recommendations


def dismiss_recommendation(svc: ModelModeService, recommendation: TuningRecommendation) -> None:
    """Mark a tuning recommendation as dismissed.

    Records it as ignored and not re-surfaced. If the recommendation has no `id`,
    no database change is made and a warning is logged.

    Args:
        svc: The ModelModeService instance.
        recommendation: The recommendation to dismiss; must include `id` to persist.
    """
    if recommendation.id is None:
        logger.warning("Cannot dismiss recommendation without ID")
        return
    svc._db.update_recommendation_outcome(
        recommendation_id=recommendation.id,
        was_applied=False,
        user_feedback="ignored",
    )
    logger.debug(f"Dismissed recommendation {recommendation.id}")


def on_regenerate(svc: ModelModeService, project_id: str, chapter_id: str) -> None:
    """Record regeneration as a negative implicit signal.

    When a user regenerates a chapter, it indicates dissatisfaction with
    the previous output. This updates the most recent score for that
    chapter to mark it as regenerated.

    Args:
        svc: The ModelModeService instance.
        project_id: The project ID.
        chapter_id: The chapter being regenerated.
    """
    validate_not_empty(project_id, "project_id")
    validate_not_empty(chapter_id, "chapter_id")
    try:
        # Find the most recent score for this project/chapter using efficient LIMIT 1 query
        score = svc._db.get_latest_score_for_chapter(project_id, chapter_id)
        if score:
            score_id = score.get("id")
            if score_id:
                svc._db.update_score(score_id, was_regenerated=True)
                logger.debug(
                    f"Marked score {score_id} as regenerated for "
                    f"project {project_id}, chapter {chapter_id}"
                )
                return
        logger.debug(
            f"No score found to mark as regenerated for project {project_id}, chapter {chapter_id}"
        )
    except Exception as e:
        logger.warning(f"Failed to record regeneration signal: {e}")
