"""Recommendation functions for ModeDatabase.

Handles recording, updating, and querying tuning recommendations.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from src.memory.mode_models import RecommendationType, TuningRecommendation

logger = logging.getLogger(__name__)


def record_recommendation(
    db,
    recommendation_type: str,
    current_value: str,
    suggested_value: str,
    reason: str,
    confidence: float,
    evidence: dict[str, Any] | None = None,
    affected_role: str | None = None,
    expected_improvement: str | None = None,
) -> int:
    """Record a tuning recommendation.

    Returns:
        The ID of the inserted recommendation.

    Raises:
        sqlite3.Error: If database operation fails.
    """
    try:
        with db._lock:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO recommendations (
                        recommendation_type, current_value, suggested_value,
                        affected_role, reason, confidence, evidence_json,
                        expected_improvement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recommendation_type,
                        current_value,
                        suggested_value,
                        affected_role,
                        reason,
                        confidence,
                        json.dumps(evidence) if evidence is not None else None,
                        expected_improvement,
                    ),
                )
                conn.commit()
                return cursor.lastrowid or 0
    except sqlite3.Error as e:
        logger.error(
            "Failed to record recommendation type=%s role=%s: %s",
            recommendation_type,
            affected_role,
            e,
            exc_info=True,
        )
        raise


def update_recommendation_outcome(
    db,
    recommendation_id: int,
    was_applied: bool,
    user_feedback: str | None = None,
) -> None:
    """Record the user's outcome and optional feedback for a tuning recommendation.

    Parameters:
        db: ModeDatabase instance.
        recommendation_id (int): ID of the recommendation to update.
        was_applied (bool): True if the recommendation was applied, False otherwise.
        user_feedback (str | None): Optional free-text feedback.
    """
    logger.debug(
        "update_recommendation_outcome called: recommendation_id=%s, was_applied=%s",
        recommendation_id,
        was_applied,
    )
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                """
                UPDATE recommendations
                SET was_applied = ?, user_feedback = ?
                WHERE id = ?
                """,
                (1 if was_applied else 0, user_feedback, recommendation_id),
            )
            conn.commit()


def get_pending_recommendations(db, limit: int = 50) -> list[dict[str, Any]]:
    """Return recommendations that have not been applied and lack user feedback.

    Parameters:
        db: ModeDatabase instance.
        limit (int): Maximum number of recommendations to return.

    Returns:
        list[dict[str, Any]]: Recommendation records ordered by timestamp descending.
    """
    logger.debug("get_pending_recommendations called: limit=%s", limit)
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM recommendations
                WHERE was_applied = 0 AND user_feedback IS NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_recommendation_history(db, limit: int = 50) -> list[dict[str, Any]]:
    """Get recent recommendation history."""
    logger.debug("get_recommendation_history called: limit=%s", limit)
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM recommendations
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_recent_recommendations(db, limit: int = 10) -> list[TuningRecommendation]:
    """Get recent recommendations as TuningRecommendation objects."""
    logger.debug("get_recent_recommendations called: limit=%s", limit)
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM recommendations
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else None
                try:
                    rec_type = RecommendationType(row["recommendation_type"])
                except ValueError:
                    logger.warning(
                        "Invalid recommendation type: %s, falling back to MODEL_SWAP",
                        row["recommendation_type"],
                    )
                    rec_type = RecommendationType.MODEL_SWAP

                results.append(
                    TuningRecommendation(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        recommendation_type=rec_type,
                        current_value=row["current_value"],
                        suggested_value=row["suggested_value"],
                        affected_role=row["affected_role"],
                        reason=row["reason"],
                        confidence=row["confidence"],
                        evidence=evidence,
                        expected_improvement=row["expected_improvement"],
                        was_applied=bool(row["was_applied"]),
                        user_feedback=row["user_feedback"],
                    )
                )
            return results
