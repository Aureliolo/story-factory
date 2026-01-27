"""Refinement analytics mixin for ModeDatabase.

Handles refinement effectiveness tracking and analysis.
"""

import json
import logging
import sqlite3
from typing import Any

from src.memory.mode_database._base import ModeDatabaseBase
from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class RefinementMixin(ModeDatabaseBase):
    """Mixin providing refinement analytics operations."""

    def get_refinement_effectiveness_summary(
        self,
        entity_type: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Summarizes refinement-loop effectiveness for world entities over a recent time window.

        Parameters:
            entity_type: Optional filter for the entity type to include (e.g., "character", "faction").
            days: Number of days to include in the analysis; must be a non-negative integer.

        Returns:
            A dictionary containing refinement effectiveness metrics:
            - total_entities: Number of entities analyzed.
            - threshold_met_rate: Percentage of entities that met their quality threshold.
            - early_stop_rate: Percentage of entities that triggered early stopping.
            - avg_iterations: Average number of refinement iterations used.
            - avg_score_loss: Average difference between peak and final score.
            - best_is_final_rate: Percentage of cases where best iteration equals returned iteration.
            - by_entity_type: List of per-entity-type breakdowns.
        """
        logger.debug(
            "Getting refinement effectiveness summary: entity_type=%s, days=%d",
            entity_type,
            days,
        )
        validated_days = int(days)
        if validated_days < 0:
            raise ValidationError("days must be a non-negative integer")

        where_clauses = ["DATE(timestamp) >= DATE('now', ?)"]
        params: list[Any] = [f"-{validated_days} days"]

        if entity_type:
            where_clauses.append("entity_type = ?")
            params.append(entity_type)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Overall statistics
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_entities,
                    SUM(threshold_met) as threshold_met_count,
                    SUM(early_stop_triggered) as early_stop_count,
                    AVG(iterations_used) as avg_iterations,
                    AVG(CASE WHEN peak_score IS NOT NULL AND final_score IS NOT NULL
                        THEN peak_score - final_score ELSE 0 END) as avg_score_loss,
                    SUM(CASE WHEN best_iteration = iterations_used THEN 1 ELSE 0 END)
                        as best_is_final_count
                FROM world_entity_scores
                WHERE {where_sql}
                """,
                params,
            )
            row = cursor.fetchone()

            total = row["total_entities"] or 0
            summary: dict[str, Any] = {
                "total_entities": total,
                "threshold_met_rate": (
                    round((row["threshold_met_count"] or 0) / total * 100, 1) if total > 0 else 0.0
                ),
                "early_stop_rate": (
                    round((row["early_stop_count"] or 0) / total * 100, 1) if total > 0 else 0.0
                ),
                "avg_iterations": (
                    round(row["avg_iterations"], 2) if row["avg_iterations"] else None
                ),
                "avg_score_loss": round(row["avg_score_loss"], 3) if row["avg_score_loss"] else 0.0,
                "best_is_final_rate": (
                    round((row["best_is_final_count"] or 0) / total * 100, 1) if total > 0 else 0.0
                ),
                "by_entity_type": [],
            }

            # Breakdown by entity type
            cursor = conn.execute(
                f"""
                SELECT
                    entity_type,
                    COUNT(*) as count,
                    SUM(threshold_met) as threshold_met,
                    SUM(early_stop_triggered) as early_stopped,
                    AVG(iterations_used) as avg_iterations,
                    AVG(CASE WHEN peak_score IS NOT NULL AND final_score IS NOT NULL
                        THEN peak_score - final_score ELSE 0 END) as avg_score_loss
                FROM world_entity_scores
                WHERE {where_sql}
                GROUP BY entity_type
                ORDER BY count DESC
                """,
                params,
            )
            for r in cursor.fetchall():
                count = r["count"] or 0
                summary["by_entity_type"].append(
                    {
                        "entity_type": r["entity_type"],
                        "count": count,
                        "threshold_met_rate": (
                            round((r["threshold_met"] or 0) / count * 100, 1) if count > 0 else 0.0
                        ),
                        "early_stop_rate": (
                            round((r["early_stopped"] or 0) / count * 100, 1) if count > 0 else 0.0
                        ),
                        "avg_iterations": round(r["avg_iterations"], 2)
                        if r["avg_iterations"]
                        else None,
                        "avg_score_loss": round(r["avg_score_loss"], 3)
                        if r["avg_score_loss"]
                        else 0.0,
                    }
                )

            logger.debug(
                "Refinement effectiveness summary: total=%d, threshold_met=%.1f%%, early_stop=%.1f%%",
                summary["total_entities"],
                summary["threshold_met_rate"],
                summary["early_stop_rate"],
            )
            return summary

    def get_refinement_progression_data(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent world-entity refinement progression records with parsed progression data.

        Only records that have a stored score progression are returned; each result includes parsed
        `score_progression` as a list of numeric scores.

        Parameters:
            entity_type (str | None): If provided, restrict results to this entity type.
            limit (int): Maximum number of records to return, ordered by newest first.

        Returns:
            list[dict[str, Any]]: A list of records containing keys:
                - id, timestamp, project_id, entity_type, entity_name
                - iterations_used, peak_score, final_score, best_iteration
                - threshold_met, early_stop_triggered, consecutive_degradations, quality_threshold
                - score_progression (list[float])
        """
        logger.debug(
            "Getting refinement progression data: entity_type=%s, limit=%d",
            entity_type,
            limit,
        )
        query = """
            SELECT
                id, timestamp, project_id, entity_type, entity_name,
                iterations_used, peak_score, final_score, best_iteration,
                threshold_met, early_stop_triggered, consecutive_degradations,
                quality_threshold, score_progression_json
            FROM world_entity_scores
            WHERE score_progression_json IS NOT NULL
        """
        params: list[Any] = []

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                record = dict(row)
                # Parse JSON progression
                if record.get("score_progression_json"):
                    record["score_progression"] = json.loads(record["score_progression_json"])
                    del record["score_progression_json"]
                else:
                    record["score_progression"] = []
                results.append(record)
            logger.debug("Retrieved %d refinement progression records", len(results))
            return results
