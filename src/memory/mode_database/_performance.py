"""Performance aggregate functions for ModeDatabase.

Handles model performance aggregation, top models, summaries, quality vs speed data,
and genre breakdowns.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any

from src.memory.mode_models import ModelPerformanceSummary

logger = logging.getLogger(__name__)


def update_model_performance(
    db,
    model_id: str,
    agent_role: str,
    genre: str | None = None,
) -> None:
    """Recalculate aggregated performance for a model/role/genre combo."""
    logger.debug(
        "update_model_performance called: model_id=%s, agent_role=%s, genre=%s",
        model_id,
        agent_role,
        genre,
    )
    genre_value = genre or ""
    genre_condition = "genre = ?" if genre else "(genre IS NULL OR genre = '')"

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            # Calculate aggregates
            cursor = conn.execute(
                f"""
                SELECT
                    AVG(prose_quality) as avg_prose,
                    AVG(instruction_following) as avg_instruction,
                    AVG(consistency_score) as avg_consistency,
                    AVG(tokens_per_second) as avg_speed,
                    COUNT(*) as sample_count
                FROM generation_scores
                WHERE model_id = ? AND agent_role = ? AND {genre_condition}
                """,
                (model_id, agent_role, genre_value) if genre else (model_id, agent_role),
            )
            row = cursor.fetchone()

            if row and row[4] > 0:  # sample_count > 0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO model_performance (
                        model_id, agent_role, genre,
                        avg_prose_quality, avg_instruction_following,
                        avg_consistency, avg_tokens_per_second,
                        sample_count, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        agent_role,
                        genre_value,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()


def get_model_performance(
    db,
    model_id: str | None = None,
    agent_role: str | None = None,
    genre: str | None = None,
) -> list[dict[str, Any]]:
    """Get aggregated model performance."""
    logger.debug(
        "get_model_performance called: model_id=%s, agent_role=%s, genre=%s",
        model_id,
        agent_role,
        genre,
    )
    query = "SELECT * FROM model_performance WHERE 1=1"
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

    query += " ORDER BY avg_prose_quality DESC"

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def get_top_models_for_role(
    db,
    agent_role: str,
    limit: int = 5,
    min_samples: int = 3,
) -> list[dict[str, Any]]:
    """Get top performing models for a role."""
    logger.debug(
        "get_top_models_for_role called: agent_role=%s, limit=%s, min_samples=%s",
        agent_role,
        limit,
        min_samples,
    )
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM model_performance
                WHERE agent_role = ? AND sample_count >= ?
                ORDER BY avg_prose_quality DESC
                LIMIT ?
                """,
                (agent_role, min_samples, limit),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_model_summaries(
    db,
    agent_role: str | None = None,
    genre: str | None = None,
) -> list[ModelPerformanceSummary]:
    """Get model performance summaries.

    Returns list of ModelPerformanceSummary objects.
    """
    logger.debug("get_model_summaries called: agent_role=%s, genre=%s", agent_role, genre)
    query = "SELECT * FROM model_performance WHERE 1=1"
    params: list[Any] = []

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)
    if genre:
        query += " AND genre = ?"
        params.append(genre)

    query += " ORDER BY avg_prose_quality DESC"

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append(
                    ModelPerformanceSummary(
                        model_id=row["model_id"],
                        agent_role=row["agent_role"],
                        genre=row["genre"] if row["genre"] else None,
                        avg_prose_quality=row["avg_prose_quality"],
                        avg_instruction_following=row["avg_instruction_following"],
                        avg_consistency=row["avg_consistency"],
                        avg_tokens_per_second=row["avg_tokens_per_second"],
                        sample_count=row["sample_count"],
                        last_updated=datetime.fromisoformat(row["last_updated"])
                        if row["last_updated"]
                        else None,
                    )
                )
            return results


def get_quality_vs_speed_data(
    db,
    agent_role: str | None = None,
    min_samples: int = 3,
) -> list[dict[str, Any]]:
    """Get data for quality vs speed scatter plot."""
    logger.debug(
        "get_quality_vs_speed_data called: agent_role=%s, min_samples=%s",
        agent_role,
        min_samples,
    )
    query = """
        SELECT
            model_id,
            agent_role,
            avg_prose_quality,
            avg_tokens_per_second,
            sample_count
        FROM model_performance
        WHERE sample_count >= ?
    """
    params: list[Any] = [min_samples]

    if agent_role:
        query += " AND agent_role = ?"
        params.append(agent_role)

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def get_genre_breakdown(db, model_id: str) -> list[dict[str, Any]]:
    """Get performance breakdown by genre for a model."""
    logger.debug("get_genre_breakdown called: model_id=%s", model_id)
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    genre,
                    AVG(prose_quality) as avg_quality,
                    AVG(tokens_per_second) as avg_speed,
                    COUNT(*) as sample_count
                FROM generation_scores
                WHERE model_id = ? AND genre IS NOT NULL
                GROUP BY genre
                ORDER BY avg_quality DESC
                """,
                (model_id,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_unique_genres(db) -> list[str]:
    """Get list of unique genres from scores."""
    logger.debug("get_unique_genres called")
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT genre FROM generation_scores WHERE genre IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]
