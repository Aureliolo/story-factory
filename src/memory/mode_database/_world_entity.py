"""World entity score functions for ModeDatabase.

Handles recording and querying world entity quality scores.
"""

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


def record_world_entity_score(
    db,
    project_id: str,
    entity_type: str,
    entity_name: str,
    model_id: str,
    scores: dict[str, float],
    *,
    entity_id: str | None = None,
    iterations_used: int | None = None,
    generation_time_seconds: float | None = None,
    feedback: str | None = None,
    early_stop_triggered: bool = False,
    threshold_met: bool = False,
    peak_score: float | None = None,
    final_score: float | None = None,
    score_progression: list[float] | None = None,
    consecutive_degradations: int = 0,
    best_iteration: int = 0,
    quality_threshold: float | None = None,
    max_iterations: int | None = None,
    temporal_consistency_score: float | None = None,
    temporal_validation_errors: list[dict] | None = None,
    below_threshold_admitted: bool = False,
) -> int:
    """Record a world entity quality score with refinement effectiveness metrics.

    Args:
        db: ModeDatabase instance.
        project_id: The project ID.
        entity_type: Type of entity (character, location, faction, item, concept).
        entity_name: Name of the entity.
        model_id: The model used for generation.
        scores: Dictionary of scores (keys depend on entity type).
        entity_id: Optional entity ID from world database.
        iterations_used: Number of refinement iterations used.
        generation_time_seconds: Time taken to generate.
        feedback: Quality feedback from judge.
        early_stop_triggered: Whether early stopping was triggered due to score degradation.
        threshold_met: Whether the quality threshold was met.
        peak_score: Highest score achieved during refinement.
        final_score: Final score of the returned entity.
        score_progression: List of scores from each iteration.
        consecutive_degradations: Number of consecutive score decreases before stopping.
        best_iteration: Iteration number that produced the best score.
        quality_threshold: The quality threshold used for this entity.
        max_iterations: Maximum iterations configured for this entity.
        temporal_consistency_score: Score for temporal consistency validation (0-10).
        temporal_validation_errors: List of temporal validation error dicts.
        below_threshold_admitted: Whether entity was admitted below quality threshold.

    Returns:
        The ID of the inserted record.

    Raises:
        sqlite3.Error: If database operation fails.
    """
    # Extract up to 4 score values, excluding "average" which is stored separately
    score_values = [v for k, v in scores.items() if k != "average"]
    score_1 = score_values[0] if len(score_values) > 0 else None
    score_2 = score_values[1] if len(score_values) > 1 else None
    score_3 = score_values[2] if len(score_values) > 2 else None
    score_4 = score_values[3] if len(score_values) > 3 else None
    average_score = scores.get("average")

    # Convert score progression to JSON
    score_progression_json = json.dumps(score_progression) if score_progression else None
    temporal_errors_json = (
        json.dumps(temporal_validation_errors) if temporal_validation_errors else None
    )

    try:
        with db._lock:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO world_entity_scores (
                        project_id, entity_id, entity_type, entity_name, model_id,
                        score_1, score_2, score_3, score_4, average_score,
                        iterations_used, generation_time_seconds, feedback,
                        early_stop_triggered, threshold_met, peak_score, final_score,
                        score_progression_json, consecutive_degradations, best_iteration,
                        quality_threshold, max_iterations,
                        temporal_consistency_score, temporal_validation_errors,
                        below_threshold_admitted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        entity_id,
                        entity_type,
                        entity_name,
                        model_id,
                        score_1,
                        score_2,
                        score_3,
                        score_4,
                        average_score,
                        iterations_used,
                        generation_time_seconds,
                        feedback,
                        1 if early_stop_triggered else 0,
                        1 if threshold_met else 0,
                        peak_score,
                        final_score,
                        score_progression_json,
                        consecutive_degradations,
                        best_iteration,
                        quality_threshold,
                        max_iterations,
                        temporal_consistency_score,
                        temporal_errors_json,
                        1 if below_threshold_admitted else 0,
                    ),
                )
                conn.commit()
                return cursor.lastrowid or 0
    except sqlite3.Error as e:
        logger.error(
            "Failed to record world entity score for project=%s entity=%s type=%s model=%s: %s",
            project_id,
            entity_name,
            entity_type,
            model_id,
            e,
            exc_info=True,
        )
        raise


def get_world_entity_scores(
    db,
    project_id: str | None = None,
    entity_type: str | None = None,
    model_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get world entity scores with optional filters.

    Args:
        db: ModeDatabase instance.
        project_id: Filter by project.
        entity_type: Filter by entity type.
        model_id: Filter by model.
        limit: Maximum number of results.

    Returns:
        List of score records as dictionaries.
    """
    logger.debug(
        "get_world_entity_scores called: project_id=%s, entity_type=%s, model_id=%s, limit=%s",
        project_id,
        entity_type,
        model_id,
        limit,
    )
    query = "SELECT * FROM world_entity_scores WHERE 1=1"
    params: list[Any] = []

    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)
    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def get_world_quality_summary(
    db,
    entity_type: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Get summary statistics for world entity quality scores.

    Args:
        db: ModeDatabase instance.
        entity_type: Optional filter by entity type.
        model_id: Optional filter by model.

    Returns:
        Dictionary with summary statistics.
    """
    logger.debug(
        "get_world_quality_summary called: entity_type=%s, model_id=%s",
        entity_type,
        model_id,
    )
    where_clauses = ["1=1"]
    params: list[Any] = []

    if entity_type:
        where_clauses.append("entity_type = ?")
        params.append(entity_type)
    if model_id:
        where_clauses.append("model_id = ?")
        params.append(model_id)

    where_sql = " AND ".join(where_clauses)

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            # Overall statistics
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_entities,
                    AVG(average_score) as avg_quality,
                    MIN(average_score) as min_quality,
                    MAX(average_score) as max_quality,
                    AVG(iterations_used) as avg_iterations,
                    AVG(generation_time_seconds) as avg_generation_time
                FROM world_entity_scores
                WHERE {where_sql}
                """,
                params,
            )
            row = cursor.fetchone()
            summary = {
                "total_entities": row[0] or 0,
                "avg_quality": row[1],
                "min_quality": row[2],
                "max_quality": row[3],
                "avg_iterations": row[4],
                "avg_generation_time": row[5],
            }

            # Breakdown by entity type
            cursor = conn.execute(
                f"""
                SELECT
                    entity_type,
                    COUNT(*) as count,
                    AVG(average_score) as avg_quality
                FROM world_entity_scores
                WHERE {where_sql}
                GROUP BY entity_type
                ORDER BY count DESC
                """,
                params,
            )
            summary["by_entity_type"] = [
                {"entity_type": r[0], "count": r[1], "avg_quality": r[2]} for r in cursor.fetchall()
            ]

            # Breakdown by model
            cursor = conn.execute(
                f"""
                SELECT
                    model_id,
                    COUNT(*) as count,
                    AVG(average_score) as avg_quality
                FROM world_entity_scores
                WHERE {where_sql}
                GROUP BY model_id
                ORDER BY avg_quality DESC
                """,
                params,
            )
            summary["by_model"] = [
                {"model_id": r[0], "count": r[1], "avg_quality": r[2]} for r in cursor.fetchall()
            ]

            return summary


def get_world_entity_count(
    db,
    entity_type: str | None = None,
    model_id: str | None = None,
) -> int:
    """Get count of world entity scores.

    Args:
        db: ModeDatabase instance.
        entity_type: Optional filter by entity type.
        model_id: Optional filter by model.

    Returns:
        Number of matching records.
    """
    logger.debug(
        "get_world_entity_count called: entity_type=%s, model_id=%s", entity_type, model_id
    )
    query = "SELECT COUNT(*) FROM world_entity_scores WHERE 1=1"
    params: list[Any] = []

    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return int(result) if result is not None else 0


def record_hail_mary_attempt(
    db,
    entity_type: str,
    won: bool,
    best_score: float | None = None,
    hail_mary_score: float | None = None,
) -> int:
    """Record a hail-mary fresh-creation attempt outcome.

    Args:
        db: ModeDatabase instance.
        entity_type: Type of entity (e.g. "faction", "relationship").
        won: True if the hail-mary entity beat the previous best.
        best_score: Best score before the hail-mary attempt.
        hail_mary_score: Score of the hail-mary entity.

    Returns:
        The ID of the inserted record.
    """
    logger.debug(
        "Recording hail-mary attempt: entity_type=%s, won=%s, best=%.1f, hail_mary=%.1f",
        entity_type,
        won,
        best_score or 0.0,
        hail_mary_score or 0.0,
    )
    try:
        with db._lock:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO hail_mary_stats (entity_type, won, best_score, hail_mary_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entity_type, 1 if won else 0, best_score, hail_mary_score),
                )
                conn.commit()
                return cursor.lastrowid or 0
    except sqlite3.Error as e:
        logger.error(
            "Failed to record hail-mary attempt for entity_type=%s won=%s: %s",
            entity_type,
            won,
            e,
            exc_info=True,
        )
        raise


def get_hail_mary_win_rate(
    db,
    entity_type: str | None = None,
    min_attempts: int = 10,
) -> float | None:
    """Get the hail-mary win rate for an entity type.

    Args:
        db: ModeDatabase instance.
        entity_type: Filter by entity type. None for overall rate.
        min_attempts: Minimum attempts required to return a rate.

    Returns:
        Win rate as a float (0.0 to 1.0), or None if insufficient data.
    """
    if min_attempts < 1:
        raise ValueError("min_attempts must be >= 1")
    query = "SELECT COUNT(*), SUM(won) FROM hail_mary_stats WHERE 1=1"
    params: list[Any] = []
    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)

    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            total = row[0] or 0
            wins = row[1] or 0
            if total == 0 or total < min_attempts:
                logger.debug(
                    "Hail-mary win rate: insufficient data (%d/%d attempts for %s)",
                    total,
                    min_attempts,
                    entity_type or "all",
                )
                return None
            rate = wins / total
            logger.debug(
                "Hail-mary win rate for %s: %.1f%% (%d/%d)",
                entity_type or "all",
                rate * 100,
                wins,
                total,
            )
            return rate


def get_first_pass_rate(
    db,
    entity_type: str,
    min_records: int = 20,
) -> float | None:
    """Get the fraction of entities that passed on the first iteration.

    Args:
        db: ModeDatabase instance.
        entity_type: Entity type to query.
        min_records: Minimum records required to return a rate.

    Returns:
        First-pass rate as a float (0.0 to 1.0), or None if insufficient data.
    """
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    query = """
        SELECT COUNT(*),
               SUM(CASE WHEN iterations_used = 1 AND threshold_met = 1 THEN 1 ELSE 0 END)
        FROM world_entity_scores
        WHERE entity_type = ?
    """
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(query, (entity_type,))
            row = cursor.fetchone()
            total = row[0] or 0
            first_pass = row[1] or 0
            if total == 0 or total < min_records:
                logger.debug(
                    "First-pass rate: insufficient data (%d/%d records for %s)",
                    total,
                    min_records,
                    entity_type,
                )
                return None
            rate = first_pass / total
            logger.debug(
                "First-pass rate for %s: %.1f%% (%d/%d)",
                entity_type,
                rate * 100,
                first_pass,
                total,
            )
            return rate
