"""Cost tracking mixin for ModeDatabase.

Handles generation run tracking and cost analysis.
"""

import json
import logging
import sqlite3
from typing import Any

from src.memory.mode_database._base import ModeDatabaseBase
from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class CostTrackingMixin(ModeDatabaseBase):
    """Mixin providing cost tracking operations."""

    def start_generation_run(
        self,
        run_id: str,
        project_id: str,
        run_type: str,
    ) -> int:
        """Start tracking a new generation run.

        Args:
            run_id: Unique identifier for this run.
            project_id: Project ID.
            run_type: Type of run ('story_generation' or 'world_build').

        Returns:
            The ID of the inserted record.

        Raises:
            sqlite3.Error: If database operation fails.
        """
        logger.debug(
            "Starting generation run: run_id=%s, project=%s, type=%s",
            run_id,
            project_id,
            run_type,
        )
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO generation_runs (run_id, project_id, run_type)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, project_id, run_type),
                )
                conn.commit()
                logger.info(f"Started generation run: {run_id}")
                return cursor.lastrowid or 0
        except sqlite3.Error as e:
            logger.error(
                "Failed to start generation run: run_id=%s, project=%s: %s",
                run_id,
                project_id,
                e,
                exc_info=True,
            )
            raise

    def update_generation_run(
        self,
        run_id: str,
        *,
        total_tokens: int | None = None,
        total_time_seconds: float | None = None,
        total_calls: int | None = None,
        by_entity_type: dict[str, dict] | None = None,
        by_model: dict[str, dict] | None = None,
        total_iterations: int | None = None,
        wasted_iterations: int | None = None,
        completed: bool = False,
    ) -> None:
        """
        Update accumulated metrics for an existing generation run and optionally mark it completed.

        Parameters:
            run_id (str): Identifier of the generation run to update.
            total_tokens (int | None): Cumulative tokens used for the run.
            total_time_seconds (float | None): Cumulative generation time in seconds.
            total_calls (int | None): Cumulative number of generation calls.
            by_entity_type (dict[str, dict] | None): Per-entity-type breakdown; stored as JSON.
            by_model (dict[str, dict] | None): Per-model breakdown; stored as JSON.
            total_iterations (int | None): Total refinement iterations performed.
            wasted_iterations (int | None): Refinement iterations considered wasted.
            completed (bool): If True, sets the run's completion timestamp to now.

        Raises:
            sqlite3.Error: If the database update fails.
        """
        updates: list[str] = []
        values: list[Any] = []

        if total_tokens is not None:
            updates.append("total_tokens = ?")
            values.append(total_tokens)
        if total_time_seconds is not None:
            updates.append("total_time_seconds = ?")
            values.append(total_time_seconds)
        if total_calls is not None:
            updates.append("total_calls = ?")
            values.append(total_calls)
        if by_entity_type is not None:
            updates.append("by_entity_type_json = ?")
            values.append(json.dumps(by_entity_type))
        if by_model is not None:
            updates.append("by_model_json = ?")
            values.append(json.dumps(by_model))
        if total_iterations is not None:
            updates.append("total_iterations = ?")
            values.append(total_iterations)
        if wasted_iterations is not None:
            updates.append("wasted_iterations = ?")
            values.append(wasted_iterations)
        if completed:
            updates.append("completed_at = datetime('now')")

        if not updates:
            return

        values.append(run_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE generation_runs SET {', '.join(updates)} WHERE run_id = ?",
                    values,
                )
                conn.commit()
                logger.debug(f"Updated generation run: {run_id}")
        except sqlite3.Error as e:
            logger.error(
                "Failed to update generation run: run_id=%s: %s",
                run_id,
                e,
                exc_info=True,
            )
            raise

    def complete_generation_run(self, run_id: str) -> None:
        """
        Mark a generation run as completed by setting its completed_at timestamp to the current time.

        Parameters:
            run_id (str): Identifier of the generation run to mark completed.

        Raises:
            sqlite3.Error: If a database error occurs while updating the run.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE generation_runs SET completed_at = datetime('now') WHERE run_id = ?",
                    (run_id,),
                )
                conn.commit()
                logger.info(f"Completed generation run: {run_id}")
        except sqlite3.Error as e:
            logger.error(
                "Failed to complete generation run: run_id=%s: %s",
                run_id,
                e,
                exc_info=True,
            )
            raise

    def get_generation_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Retrieve a generation run record by run_id.

        Returns:
            dict: Run record with JSON fields `by_entity_type` and `by_model` deserialized into dictionaries, or `None` if no matching run is found.
        """
        logger.debug(f"get_generation_run: run_id={run_id}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM generation_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get("by_entity_type_json"):
                    result["by_entity_type"] = json.loads(result.pop("by_entity_type_json"))
                else:
                    result["by_entity_type"] = {}
                    result.pop("by_entity_type_json", None)
                if result.get("by_model_json"):
                    result["by_model"] = json.loads(result.pop("by_model_json"))
                else:
                    result["by_model"] = {}
                    result.pop("by_model_json", None)
                logger.debug(f"get_generation_run: found run_id={run_id}")
                return result
            logger.debug(f"get_generation_run: run_id={run_id} not found")
            return None

    def get_generation_runs(
        self,
        project_id: str | None = None,
        run_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent generation runs, optionally filtered by project or run type.

        Parameters:
            project_id (str | None): If provided, only runs for this project are returned.
            run_type (str | None): If provided, only runs of this type are returned.
            limit (int): Maximum number of runs to return.

        Returns:
            list[dict[str, Any]]: A list of run records.
        """
        logger.debug(
            f"get_generation_runs: project_id={project_id}, run_type={run_type}, limit={limit}"
        )
        query = "SELECT * FROM generation_runs WHERE 1=1"
        params: list[Any] = []

        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        if run_type:
            query += " AND run_type = ?"
            params.append(run_type)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                record = dict(row)
                # Parse JSON fields
                if record.get("by_entity_type_json"):
                    record["by_entity_type"] = json.loads(record.pop("by_entity_type_json"))
                else:
                    record["by_entity_type"] = {}
                    record.pop("by_entity_type_json", None)
                if record.get("by_model_json"):
                    record["by_model"] = json.loads(record.pop("by_model_json"))
                else:
                    record["by_model"] = {}
                    record.pop("by_model_json", None)
                results.append(record)
            logger.debug(f"get_generation_runs: returning {len(results)} runs")
            return results

    def get_cost_summary(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Summarizes generation-run cost and usage metrics for a recent time window.

        Parameters:
            project_id (str | None): Optional project identifier to filter results.
            days (int): Number of days to include (must be >= 0).

        Returns:
            dict: Summary containing total_runs, total_tokens, total_time_seconds, etc.

        Raises:
            ValueError: If `days` is negative.
        """
        logger.debug(f"get_cost_summary: project_id={project_id}, days={days}")
        validated_days = int(days)
        if validated_days < 0:
            raise ValidationError("days must be a non-negative integer")

        where_clauses = ["DATE(started_at) >= DATE('now', ?)"]
        params: list[Any] = [f"-{validated_days} days"]

        if project_id:
            where_clauses.append("project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_time_seconds) as total_time_seconds,
                    SUM(total_calls) as total_calls,
                    SUM(total_iterations) as total_iterations,
                    SUM(wasted_iterations) as wasted_iterations,
                    AVG(total_tokens) as avg_tokens_per_run,
                    AVG(total_time_seconds) as avg_time_per_run
                FROM generation_runs
                WHERE {where_sql}
                """,
                params,
            )
            row = cursor.fetchone()
            summary = {
                "total_runs": row[0] or 0,
                "total_tokens": row[1] or 0,
                "total_time_seconds": row[2] or 0.0,
                "total_calls": row[3] or 0,
                "total_iterations": row[4] or 0,
                "wasted_iterations": row[5] or 0,
                "avg_tokens_per_run": row[6] or 0.0,
                "avg_time_per_run": row[7] or 0.0,
            }

            # Breakdown by run type
            cursor = conn.execute(
                f"""
                SELECT
                    run_type,
                    COUNT(*) as count,
                    SUM(total_tokens) as tokens,
                    SUM(total_time_seconds) as time_seconds
                FROM generation_runs
                WHERE {where_sql}
                GROUP BY run_type
                ORDER BY tokens DESC
                """,
                params,
            )
            summary["by_run_type"] = [
                {
                    "run_type": r[0],
                    "count": r[1],
                    "tokens": r[2] or 0,
                    "time_seconds": r[3] or 0.0,
                }
                for r in cursor.fetchall()
            ]

            # Calculate efficiency
            total_iters = summary["total_iterations"]
            wasted_iters = summary["wasted_iterations"]
            if total_iters > 0:
                summary["efficiency_ratio"] = round((total_iters - wasted_iters) / total_iters, 3)
            else:
                summary["efficiency_ratio"] = 1.0

            logger.debug(f"get_cost_summary: total_runs={summary['total_runs']}")
            return summary

    def get_model_cost_breakdown(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Produce a per-model cost and usage breakdown for generation activity.

        Parameters:
            project_id (str | None): Optional project identifier to filter results.
            days (int): Lookback window in days; must be zero or positive.

        Returns:
            list[dict[str, Any]]: A list of mappings, one per model.

        Raises:
            ValidationError: If `days` is negative.
        """
        logger.debug(f"get_model_cost_breakdown: project_id={project_id}, days={days}")
        validated_days = int(days)
        if validated_days < 0:
            raise ValidationError("days must be a non-negative integer")

        where_clauses = [
            "DATE(timestamp) >= DATE('now', ?)",
            "tokens_generated IS NOT NULL",
        ]
        params: list[Any] = [f"-{validated_days} days"]

        if project_id:
            where_clauses.append("project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    model_id,
                    COUNT(*) as call_count,
                    SUM(tokens_generated) as total_tokens,
                    SUM(time_seconds) as total_time,
                    AVG(tokens_per_second) as avg_speed,
                    AVG(prose_quality) as avg_quality
                FROM generation_scores
                WHERE {where_sql}
                GROUP BY model_id
                ORDER BY total_tokens DESC
                """,
                params,
            )
            results = [
                {
                    "model_id": r[0],
                    "call_count": r[1],
                    "total_tokens": r[2] or 0,
                    "total_time_seconds": r[3] or 0.0,
                    "avg_tokens_per_second": r[4],
                    "avg_quality": r[5],
                }
                for r in cursor.fetchall()
            ]
            logger.debug(f"get_model_cost_breakdown: returning {len(results)} models")
            return results

    def get_entity_type_cost_breakdown(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Produce cost breakdowns aggregated by world entity type over the past `days`.

        Parameters:
            project_id (str | None): Optional project identifier to filter results.
            days (int): Number of days to include in the window; must be zero or positive.

        Returns:
            list[dict[str, Any]]: Each dictionary contains entity_type, count, etc.

        Raises:
            ValidationError: If `days` is negative.
        """
        logger.debug(f"get_entity_type_cost_breakdown: project_id={project_id}, days={days}")
        validated_days = int(days)
        if validated_days < 0:
            raise ValidationError("days must be a non-negative integer")

        where_clauses = ["DATE(timestamp) >= DATE('now', ?)"]
        params: list[Any] = [f"-{validated_days} days"]

        if project_id:
            where_clauses.append("project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    entity_type,
                    COUNT(*) as count,
                    SUM(generation_time_seconds) as total_time,
                    AVG(iterations_used) as avg_iterations,
                    SUM(CASE WHEN NOT threshold_met THEN iterations_used ELSE 0 END) as wasted_iterations,
                    AVG(average_score) as avg_quality
                FROM world_entity_scores
                WHERE {where_sql}
                GROUP BY entity_type
                ORDER BY count DESC
                """,
                params,
            )
            results = [
                {
                    "entity_type": r[0],
                    "count": r[1],
                    "total_time_seconds": r[2] or 0.0,
                    "avg_iterations": r[3] or 0.0,
                    "wasted_iterations": int(r[4] or 0),
                    "avg_quality": r[5],
                }
                for r in cursor.fetchall()
            ]
            logger.debug(f"get_entity_type_cost_breakdown: returning {len(results)} entity types")
            return results
