"""Scoring functions for ModeDatabase.

Handles recording, updating, querying, and exporting generation scores.
"""

import csv
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.memory.mode_models import (
    GenerationScore,
    ImplicitSignals,
    PerformanceMetrics,
    QualityScores,
)

logger = logging.getLogger("src.memory.mode_database._scoring")


def record_score(
    db,
    project_id: str,
    agent_role: str,
    model_id: str,
    mode_name: str,
    *,
    chapter_id: str | None = None,
    genre: str | None = None,
    tokens_generated: int | None = None,
    time_seconds: float | None = None,
    tokens_per_second: float | None = None,
    vram_used_gb: float | None = None,
    prose_quality: float | None = None,
    instruction_following: float | None = None,
    consistency_score: float | None = None,
    was_regenerated: bool = False,
    edit_distance: int | None = None,
    user_rating: int | None = None,
    prompt_hash: str | None = None,
) -> int:
    """Record a single generation score row for a project and model.

    Parameters:
        db: ModeDatabase instance.
        project_id (str): Project identifier.
        agent_role (str): Role or persona that generated the content.
        model_id (str): Identifier of the model used.
        mode_name (str): Name of the generation mode used.
        chapter_id (str | None): Optional chapter identifier associated with the generation.
        genre (str | None): Optional genre label.
        tokens_generated (int | None): Number of tokens produced by the generation.
        time_seconds (float | None): Time taken to generate in seconds.
        tokens_per_second (float | None): Token throughput measured as tokens per second.
        vram_used_gb (float | None): VRAM consumed in gigabytes.
        prose_quality (float | None): Prose quality score (higher is better).
        instruction_following (float | None): Score for how well the output followed instructions.
        consistency_score (float | None): Score for internal consistency of the generated content.
        was_regenerated (bool): True if this output was a regeneration of a previous attempt.
        edit_distance (int | None): Edit distance from a reference or previous version, if available.
        user_rating (int | None): Optional user-provided rating.
        prompt_hash (str | None): Optional hash of the prompt/template used.

    Returns:
        int: The row ID of the inserted generation_scores record (0 if unavailable).

    Raises:
        sqlite3.Error: If the database operation fails.
    """
    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO generation_scores (
                    project_id, chapter_id, agent_role, model_id, mode_name, genre,
                    tokens_generated, time_seconds, tokens_per_second, vram_used_gb,
                    prose_quality, instruction_following, consistency_score,
                    was_regenerated, edit_distance, user_rating, prompt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    chapter_id,
                    agent_role,
                    model_id,
                    mode_name,
                    genre,
                    tokens_generated,
                    time_seconds,
                    tokens_per_second,
                    vram_used_gb,
                    prose_quality,
                    instruction_following,
                    consistency_score,
                    1 if was_regenerated else 0,
                    edit_distance,
                    user_rating,
                    prompt_hash,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0
    except sqlite3.Error as e:
        logger.error(
            "Failed to record score for project=%s model=%s agent=%s chapter=%s: %s",
            project_id,
            model_id,
            agent_role,
            chapter_id,
            e,
            exc_info=True,
        )
        raise


def update_score(
    db,
    score_id: int,
    *,
    prose_quality: float | None = None,
    instruction_following: float | None = None,
    consistency_score: float | None = None,
    was_regenerated: bool | None = None,
    edit_distance: int | None = None,
    user_rating: int | None = None,
) -> None:
    """Update an existing score with additional metrics.

    Raises:
        sqlite3.Error: If database operation fails.
    """
    set_expressions = []
    values = []

    if prose_quality is not None:
        set_expressions.append("prose_quality = ?")
        values.append(prose_quality)
    if instruction_following is not None:
        set_expressions.append("instruction_following = ?")
        values.append(instruction_following)
    if consistency_score is not None:
        set_expressions.append("consistency_score = ?")
        values.append(consistency_score)
    if was_regenerated is not None:
        set_expressions.append("was_regenerated = ?")
        values.append(1 if was_regenerated else 0)
    if edit_distance is not None:
        set_expressions.append("edit_distance = ?")
        values.append(edit_distance)
    if user_rating is not None:
        set_expressions.append("user_rating = ?")
        values.append(user_rating)

    if not set_expressions:
        return

    set_clause = ", ".join(set_expressions)
    values.append(score_id)
    try:
        with sqlite3.connect(db.db_path) as conn:
            sql = f"UPDATE generation_scores SET {set_clause} WHERE id = ?"
            conn.execute(sql, values)
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to update score {score_id}: {e}", exc_info=True)
        raise


def update_performance_metrics(
    db,
    score_id: int,
    *,
    tokens_generated: int | None = None,
    time_seconds: float | None = None,
    tokens_per_second: float | None = None,
    vram_used_gb: float | None = None,
) -> None:
    """Update an existing score with performance metrics.

    Args:
        db: ModeDatabase instance.
        score_id: The score record ID.
        tokens_generated: Number of tokens generated.
        time_seconds: Generation time in seconds.
        tokens_per_second: Generation speed.
        vram_used_gb: VRAM used during generation.

    Raises:
        sqlite3.Error: If database operation fails.
    """
    updates: list[str] = []
    values: list[int | float] = []

    if tokens_generated is not None:
        updates.append("tokens_generated = ?")
        values.append(tokens_generated)
    if time_seconds is not None:
        updates.append("time_seconds = ?")
        values.append(time_seconds)
    if tokens_per_second is not None:
        updates.append("tokens_per_second = ?")
        values.append(tokens_per_second)
    if vram_used_gb is not None:
        updates.append("vram_used_gb = ?")
        values.append(vram_used_gb)

    if not updates:
        return

    values.append(score_id)
    try:
        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                f"UPDATE generation_scores SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(
            "Failed to update performance metrics for score_id=%s: %s",
            score_id,
            e,
            exc_info=True,
        )
        raise


def get_scores_for_model(
    db,
    model_id: str,
    agent_role: str | None = None,
    genre: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get recent scores for a model."""
    query = "SELECT * FROM generation_scores WHERE model_id = ?"
    params: list[Any] = [model_id]

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_scores_for_project(db, project_id: str) -> list[dict[str, Any]]:
    """Retrieve all generation scores for a project ordered by most recent first.

    Returns:
        list[dict[str, Any]]: List of rows from `generation_scores` as dictionaries,
        ordered by `timestamp` descending.
    """
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT * FROM generation_scores
            WHERE project_id = ?
            ORDER BY timestamp DESC
            """,
            (project_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_latest_score_for_chapter(db, project_id: str, chapter_id: str) -> dict[str, Any] | None:
    """Get the most recent score for a specific project chapter.

    Args:
        db: ModeDatabase instance.
        project_id: The project ID.
        chapter_id: The chapter ID.

    Returns:
        The latest score dict or None if not found.
    """
    logger.debug(f"Fetching latest score for project {project_id} chapter {chapter_id}")
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT * FROM generation_scores
            WHERE project_id = ? AND chapter_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (project_id, chapter_id),
        )
        row = cursor.fetchone()
        result = dict(row) if row else None
        if result:
            logger.debug(f"Found score id={result.get('id')} for chapter {chapter_id}")
        else:
            logger.debug(f"No score found for project {project_id} chapter {chapter_id}")
        return result


def get_score_count(
    db,
    model_id: str | None = None,
    agent_role: str | None = None,
    genre: str | None = None,
) -> int:
    """Return count of generation score records, optionally filtered by model, agent role, or genre.

    Parameters:
        db: ModeDatabase instance.
        model_id (str | None): If provided, only count scores for this model_id.
        agent_role (str | None): If provided, only count scores for this agent role.
        genre (str | None): If provided, only count scores for this genre.

    Returns:
        count (int): Number of matching rows in the generation_scores table.
    """
    query = "SELECT COUNT(*) FROM generation_scores WHERE 1=1"
    params: list[Any] = []

    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)
    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute(query, params)
        result = cursor.fetchone()[0]
        return int(result) if result is not None else 0


def get_all_scores(
    db,
    agent_role: str | None = None,
    genre: str | None = None,
    limit: int = 1000,
) -> list[GenerationScore]:
    """Get all scores as GenerationScore objects."""
    query = "SELECT * FROM generation_scores WHERE 1=1"
    params: list[Any] = []

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(
                GenerationScore(
                    project_id=row["project_id"],
                    chapter_id=row["chapter_id"],
                    agent_role=row["agent_role"],
                    model_id=row["model_id"],
                    mode_name=row["mode_name"],
                    genre=row["genre"],
                    quality=QualityScores(
                        prose_quality=row["prose_quality"],
                        instruction_following=row["instruction_following"],
                        consistency_score=row["consistency_score"],
                    ),
                    performance=PerformanceMetrics(
                        tokens_generated=row["tokens_generated"],
                        time_seconds=row["time_seconds"],
                        tokens_per_second=row["tokens_per_second"],
                        vram_used_gb=row["vram_used_gb"],
                    ),
                    signals=ImplicitSignals(
                        was_regenerated=bool(row["was_regenerated"]),
                        edit_distance=row["edit_distance"],
                        user_rating=row["user_rating"],
                    ),
                    prompt_hash=row["prompt_hash"],
                    timestamp=datetime.fromisoformat(row["timestamp"])
                    if row["timestamp"]
                    else datetime.now(),
                )
            )
        return results


def export_scores_csv(db, output_path: Path | str) -> int:
    """Export all scores to CSV.

    Returns:
        Number of rows exported.
    """
    output_path = Path(output_path)
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM generation_scores ORDER BY timestamp")
        rows = cursor.fetchall()

        if not rows:
            return 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)


def get_average_score(
    db,
    metric: str,
    agent_role: str | None = None,
    genre: str | None = None,
) -> float | None:
    """Get average of a specific metric.

    Args:
        db: ModeDatabase instance.
        metric: Column name (prose_quality, instruction_following, etc.)
        agent_role: Optional filter.
        genre: Optional filter.

    Returns:
        Average value or None if no data.
    """
    valid_metrics = {
        "prose_quality",
        "instruction_following",
        "consistency_score",
        "tokens_per_second",
    }
    if metric not in valid_metrics:
        return None

    query = f"SELECT AVG({metric}) FROM generation_scores WHERE 1=1"
    params: list[Any] = []

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute(query, params)
        result = cursor.fetchone()[0]
        return float(result) if result is not None else None


def get_content_statistics(
    db,
    project_id: str | None = None,
    agent_role: str | None = None,
) -> dict:
    """Get content statistics from generation scores.

    Args:
        db: ModeDatabase instance.
        project_id: Optional filter by project.
        agent_role: Optional filter by agent role.

    Returns:
        Dictionary with content statistics.
    """
    where_clauses = ["1=1", "tokens_generated IS NOT NULL"]
    params: list = []

    if project_id:
        where_clauses.append("project_id = ?")
        params.append(project_id)
    if agent_role:
        where_clauses.append("agent_role = ?")
        params.append(agent_role)

    where_sql = " AND ".join(where_clauses)

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute(
            f"""
            SELECT
                COUNT(*) as generation_count,
                SUM(tokens_generated) as total_tokens,
                AVG(tokens_generated) as avg_tokens,
                MIN(tokens_generated) as min_tokens,
                MAX(tokens_generated) as max_tokens,
                AVG(time_seconds) as avg_time
            FROM generation_scores
            WHERE {where_sql}
            """,
            params,
        )
        row = cursor.fetchone()
        return {
            "generation_count": row[0] or 0,
            "total_tokens": row[1] or 0,
            "avg_tokens": row[2],
            "min_tokens": row[3],
            "max_tokens": row[4],
            "avg_generation_time": row[5],
        }


def get_quality_time_series(
    db,
    metric: str = "prose_quality",
    agent_role: str | None = None,
    genre: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get time series data for quality metrics.

    Args:
        db: ModeDatabase instance.
        metric: Metric to track (prose_quality, instruction_following, consistency_score).
        agent_role: Optional filter by agent role.
        genre: Optional filter by genre.
        limit: Maximum number of data points.

    Returns:
        List of time series data points with timestamp and value.
    """
    valid_metrics = {
        "prose_quality",
        "instruction_following",
        "consistency_score",
        "tokens_per_second",
    }
    if metric not in valid_metrics:
        logger.warning(f"Invalid metric {metric}, defaulting to prose_quality")
        metric = "prose_quality"

    query = f"SELECT timestamp, {metric} FROM generation_scores WHERE {metric} IS NOT NULL"
    params: list = []

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [{"timestamp": row[0], "value": row[1]} for row in cursor.fetchall()]


def get_daily_quality_averages(
    db,
    metric: str = "prose_quality",
    days: int = 30,
    agent_role: str | None = None,
) -> list[dict]:
    """Get daily average quality metrics.

    Args:
        db: ModeDatabase instance.
        metric: Metric to track.
        days: Number of days to include.
        agent_role: Optional filter by agent role.

    Returns:
        List of daily averages with date and average value.
    """
    valid_metrics = {
        "prose_quality",
        "instruction_following",
        "consistency_score",
        "tokens_per_second",
    }
    if metric not in valid_metrics:
        logger.warning(f"Invalid metric {metric}, defaulting to prose_quality")
        metric = "prose_quality"

    query = f"""
        SELECT
            DATE(timestamp) as date,
            AVG({metric}) as avg_value,
            COUNT(*) as sample_count
        FROM generation_scores
        WHERE {metric} IS NOT NULL
        AND DATE(timestamp) >= DATE('now', '-{days} days')
    """
    params: list = []

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)

    query += " GROUP BY DATE(timestamp) ORDER BY date DESC"

    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [
            {"date": row[0], "avg_value": row[1], "sample_count": row[2]}
            for row in cursor.fetchall()
        ]
