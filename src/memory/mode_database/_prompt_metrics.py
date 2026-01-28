"""Prompt metrics functions for ModeDatabase.

Handles recording and querying prompt template usage analytics.
"""

import logging
import sqlite3
from typing import Any

from src.settings import Settings
from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


def record_prompt_metrics(
    db,
    prompt_hash: str,
    agent_role: str,
    task: str,
    template_version: str,
    model_id: str,
    tokens_generated: int | None = None,
    generation_time_seconds: float | None = None,
    success: bool = True,
    project_id: str | None = None,
    error_message: str | None = None,
) -> int:
    """Record prompt template usage for analytics.

    Args:
        db: ModeDatabase instance.
        prompt_hash: MD5 hash of the template content.
        agent_role: Agent role (writer, editor, etc.).
        task: Task name (write_chapter, edit_passage, etc.).
        template_version: Version of the template used.
        model_id: Model used for generation.
        tokens_generated: Number of tokens generated.
        generation_time_seconds: Time taken for generation.
        success: Whether generation succeeded.
        project_id: Optional project ID for context.
        error_message: Error message if generation failed.

    Returns:
        The ID of the inserted record.

    Raises:
        sqlite3.Error: If database operation fails.
    """
    # Check if prompt metrics are enabled
    settings = Settings.load()
    if not settings.prompt_metrics_enabled:
        logger.debug("Prompt metrics disabled, skipping recording")
        return 0

    try:
        with db._lock:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO prompt_metrics (
                        prompt_hash, agent_role, task, template_version, model_id,
                        tokens_generated, generation_time_seconds, success,
                        project_id, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt_hash,
                        agent_role,
                        task,
                        template_version,
                        model_id,
                        tokens_generated,
                        generation_time_seconds,
                        1 if success else 0,
                        project_id,
                        error_message,
                    ),
                )
                conn.commit()
                return cursor.lastrowid or 0
    except sqlite3.Error as e:
        logger.error(
            "Failed to record prompt metrics for hash=%s role=%s task=%s model=%s: %s",
            prompt_hash,
            agent_role,
            task,
            model_id,
            e,
            exc_info=True,
        )
        raise


def get_prompt_analytics(
    db,
    agent_role: str | None = None,
    task: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Get prompt performance statistics.

    Args:
        db: ModeDatabase instance.
        agent_role: Filter by agent role.
        task: Filter by task name.
        days: Number of days to include.

    Returns:
        List of analytics records with aggregated stats per template.

    Raises:
        ValueError: If days is negative.
    """
    logger.debug(
        "get_prompt_analytics called: agent_role=%s, task=%s, days=%s",
        agent_role,
        task,
        days,
    )
    # Validate days to prevent SQL injection
    try:
        validated_days = int(days)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"days must be a non-negative integer, got {days!r}") from e
    if validated_days < 0:
        raise ValidationError("days must be a non-negative integer")

    where_clauses = ["DATE(timestamp) >= DATE('now', ?)"]
    params: list[Any] = [f"-{validated_days} days"]

    if agent_role:
        where_clauses.append("agent_role = ?")
        params.append(agent_role)
    if task:
        where_clauses.append("task = ?")
        params.append(task)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            agent_role,
            task,
            template_version,
            prompt_hash,
            COUNT(*) as total_calls,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
            ROUND(AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as success_rate,
            AVG(tokens_generated) as avg_tokens,
            AVG(generation_time_seconds) as avg_time,
            MIN(timestamp) as first_used,
            MAX(timestamp) as last_used
        FROM prompt_metrics
        WHERE {where_sql}
        GROUP BY agent_role, task, template_version, prompt_hash
        ORDER BY total_calls DESC
    """

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def get_prompt_metrics_summary(db) -> dict[str, Any]:
    """Get overall summary of prompt metrics.

    Returns:
        Dictionary with summary statistics.
    """
    logger.debug("get_prompt_metrics_summary called")
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            # Overall statistics
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_generations,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(tokens_generated) as total_tokens,
                    AVG(generation_time_seconds) as avg_time,
                    COUNT(DISTINCT agent_role) as unique_agents,
                    COUNT(DISTINCT task) as unique_tasks,
                    COUNT(DISTINCT prompt_hash) as unique_templates
                FROM prompt_metrics
                """
            )
            row = cursor.fetchone()
            summary = {
                "total_generations": row[0] or 0,
                "successful_generations": row[1] or 0,
                "total_tokens": row[2] or 0,
                "avg_generation_time": row[3],
                "unique_agents": row[4] or 0,
                "unique_tasks": row[5] or 0,
                "unique_templates": row[6] or 0,
            }

            # Calculate success rate
            if summary["total_generations"] > 0:
                summary["success_rate"] = round(
                    (summary["successful_generations"] / summary["total_generations"]) * 100, 1
                )
            else:
                summary["success_rate"] = 0.0

            # By agent role
            cursor = conn.execute(
                """
                SELECT
                    agent_role,
                    COUNT(*) as count,
                    AVG(tokens_generated) as avg_tokens,
                    ROUND(AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as success_rate
                FROM prompt_metrics
                GROUP BY agent_role
                ORDER BY count DESC
                """
            )
            summary["by_agent"] = [
                {
                    "agent_role": r[0],
                    "count": r[1],
                    "avg_tokens": r[2],
                    "success_rate": r[3],
                }
                for r in cursor.fetchall()
            ]

            return summary


def get_prompt_metrics_by_hash(
    db,
    prompt_hash: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get metrics for a specific template hash.

    Args:
        db: ModeDatabase instance.
        prompt_hash: Template hash to query.
        limit: Maximum number of records.

    Returns:
        List of metric records for the template.
    """
    logger.debug("get_prompt_metrics_by_hash called: prompt_hash=%s, limit=%s", prompt_hash, limit)
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM prompt_metrics
                WHERE prompt_hash = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (prompt_hash, limit),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_prompt_error_summary(db, days: int = 7) -> list[dict[str, Any]]:
    """Get summary of prompt errors for debugging.

    Args:
        db: ModeDatabase instance.
        days: Number of days to include.

    Returns:
        List of error summaries grouped by agent/task.

    Raises:
        ValueError: If days is negative.
    """
    logger.debug("get_prompt_error_summary called: days=%s", days)
    # Validate days to prevent SQL injection
    try:
        validated_days = int(days)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"days must be a non-negative integer, got {days!r}") from e
    if validated_days < 0:
        raise ValidationError("days must be a non-negative integer")

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    agent_role,
                    task,
                    template_version,
                    prompt_hash,
                    COUNT(*) as error_count,
                    GROUP_CONCAT(DISTINCT error_message) as error_messages
                FROM prompt_metrics
                WHERE success = 0
                AND DATE(timestamp) >= DATE('now', ?)
                GROUP BY agent_role, task, template_version, prompt_hash
                ORDER BY error_count DESC
                """,
                (f"-{validated_days} days",),
            )
            return [dict(row) for row in cursor.fetchall()]
