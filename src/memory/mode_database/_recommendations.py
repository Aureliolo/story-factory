"""Recommendations mixin for ModeDatabase.

Handles tuning recommendations recording and retrieval.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from src.memory.mode_database._base import ModeDatabaseBase
from src.memory.mode_models import RecommendationType, TuningRecommendation

logger = logging.getLogger(__name__)


class RecommendationsMixin(ModeDatabaseBase):
    """Mixin providing recommendation operations."""

    def record_recommendation(
        self,
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
            with sqlite3.connect(self.db_path) as conn:
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
                        json.dumps(evidence) if evidence else None,
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
        self,
        recommendation_id: int,
        was_applied: bool,
        user_feedback: str | None = None,
    ) -> None:
        """
        Record the user's outcome and optional feedback for a tuning recommendation in the database.

        Parameters:
            recommendation_id (int): ID of the recommendation to update.
            was_applied (bool): `True` if the recommendation was applied, `False` otherwise.
            user_feedback (str | None): Optional free-text feedback; pass `None` to clear any existing feedback.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE recommendations
                SET was_applied = ?, user_feedback = ?
                WHERE id = ?
                """,
                (1 if was_applied else 0, user_feedback, recommendation_id),
            )
            conn.commit()

    def get_pending_recommendations(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Return recommendations that have not been applied and lack user feedback, ordered by newest first.

        Parameters:
            limit (int): Maximum number of recommendations to return.

        Returns:
            list[dict[str, Any]]: Recommendation records as dictionaries, ordered by timestamp descending and limited to `limit`.
        """
        with sqlite3.connect(self.db_path) as conn:
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

    def get_recommendation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent recommendation history."""
        with sqlite3.connect(self.db_path) as conn:
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

    def get_recent_recommendations(self, limit: int = 10) -> list[TuningRecommendation]:
        """Get recent recommendations as TuningRecommendation objects."""
        with sqlite3.connect(self.db_path) as conn:
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
                try:
                    rec_type = RecommendationType(row["recommendation_type"])
                except ValueError:
                    rec_type = RecommendationType.MODEL_SWAP

                results.append(
                    TuningRecommendation(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"])
                        if row["timestamp"]
                        else datetime.now(),
                        recommendation_type=rec_type,
                        current_value=row["current_value"],
                        suggested_value=row["suggested_value"],
                        affected_role=row["affected_role"],
                        reason=row["reason"] or "",
                        confidence=row["confidence"] or 0.0,
                        evidence=json.loads(row["evidence_json"]) if row["evidence_json"] else None,
                        expected_improvement=row["expected_improvement"],
                        was_applied=bool(row["was_applied"]),
                        user_feedback=row["user_feedback"],
                    )
                )
            return results
