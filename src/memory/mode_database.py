"""Database for model scoring and mode management.

Stores generation scores, model performance metrics, and tuning recommendations
for the adaptive learning system.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.memory.mode_models import (
    GenerationScore,
    ImplicitSignals,
    ModelPerformanceSummary,
    PerformanceMetrics,
    QualityScores,
    RecommendationType,
    TuningRecommendation,
)
from src.settings import Settings

logger = logging.getLogger(__name__)


# Go up from memory/ to src/ to project root, then into output/
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "output" / "model_scores.db"


class ModeDatabase:
    """SQLite database for model scoring and learning data.

    Tables:
    - generation_scores: Per-generation metrics (quality, speed, implicit signals)
    - model_performance: Aggregated model performance by role and genre
    - recommendations: Tuning recommendation history
    - custom_modes: User-defined generation modes
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. Defaults to output/model_scores.db.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """
        Create and initialize the SQLite schema required by the ModeDatabase.

        Creates tables for generation scores, aggregated model performance, recommendations,
        world entity scores, prompt metrics, and custom modes, along with relevant indexes,
        then applies any pending schema migrations.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Per-generation scores (granular tracking)
                CREATE TABLE IF NOT EXISTS generation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),

                    -- Context
                    project_id TEXT NOT NULL,
                    chapter_id TEXT,
                    agent_role TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    mode_name TEXT NOT NULL,
                    genre TEXT,

                    -- Performance metrics
                    tokens_generated INTEGER,
                    time_seconds REAL,
                    tokens_per_second REAL,
                    vram_used_gb REAL,

                    -- Quality scores (0-10 scale)
                    prose_quality REAL,
                    instruction_following REAL,
                    consistency_score REAL,

                    -- Implicit signals
                    was_regenerated INTEGER DEFAULT 0,
                    edit_distance INTEGER,
                    user_rating INTEGER,

                    -- For A/B comparisons
                    prompt_hash TEXT
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_scores_model ON generation_scores(model_id);
                CREATE INDEX IF NOT EXISTS idx_scores_role ON generation_scores(agent_role);
                CREATE INDEX IF NOT EXISTS idx_scores_genre ON generation_scores(genre);
                CREATE INDEX IF NOT EXISTS idx_scores_project ON generation_scores(project_id);
                CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON generation_scores(timestamp);

                -- Aggregated model performance (materialized view-like)
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_id TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    genre TEXT NOT NULL DEFAULT '',

                    avg_prose_quality REAL,
                    avg_instruction_following REAL,
                    avg_consistency REAL,
                    avg_tokens_per_second REAL,

                    sample_count INTEGER DEFAULT 0,
                    last_updated TEXT,

                    PRIMARY KEY (model_id, agent_role, genre)
                );

                -- Tuning recommendations history
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),

                    recommendation_type TEXT NOT NULL,
                    current_value TEXT,
                    suggested_value TEXT,
                    affected_role TEXT,
                    reason TEXT,
                    confidence REAL,
                    evidence_json TEXT,
                    expected_improvement TEXT,

                    -- Outcome tracking
                    was_applied INTEGER DEFAULT 0,
                    user_feedback TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_recommendations_type
                    ON recommendations(recommendation_type);

                -- World entity quality scores
                CREATE TABLE IF NOT EXISTS world_entity_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    project_id TEXT NOT NULL,
                    entity_id TEXT,
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    score_1 REAL,
                    score_2 REAL,
                    score_3 REAL,
                    score_4 REAL,
                    average_score REAL,
                    iterations_used INTEGER,
                    generation_time_seconds REAL,
                    feedback TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_world_entity_project
                    ON world_entity_scores(project_id);
                CREATE INDEX IF NOT EXISTS idx_world_entity_type
                    ON world_entity_scores(entity_type);
                CREATE INDEX IF NOT EXISTS idx_world_model
                    ON world_entity_scores(model_id);
                CREATE INDEX IF NOT EXISTS idx_world_entity_timestamp
                    ON world_entity_scores(timestamp);

                -- Prompt template metrics
                CREATE TABLE IF NOT EXISTS prompt_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    prompt_hash TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    task TEXT NOT NULL,
                    template_version TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    tokens_generated INTEGER,
                    generation_time_seconds REAL,
                    success INTEGER NOT NULL DEFAULT 1,
                    project_id TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_prompt_metrics_hash
                    ON prompt_metrics(prompt_hash);
                CREATE INDEX IF NOT EXISTS idx_prompt_metrics_agent
                    ON prompt_metrics(agent_role);
                CREATE INDEX IF NOT EXISTS idx_prompt_metrics_task
                    ON prompt_metrics(task);
                CREATE INDEX IF NOT EXISTS idx_prompt_metrics_timestamp
                    ON prompt_metrics(timestamp);

                -- Custom generation modes
                CREATE TABLE IF NOT EXISTS custom_modes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    agent_models_json TEXT NOT NULL,
                    agent_temperatures_json TEXT NOT NULL,
                    size_preference TEXT NOT NULL DEFAULT 'medium',
                    vram_strategy TEXT NOT NULL DEFAULT 'adaptive',
                    is_experimental INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                -- Generation cost tracking
                CREATE TABLE IF NOT EXISTS generation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    project_id TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT (datetime('now')),
                    completed_at TEXT,
                    run_type TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0,
                    total_time_seconds REAL DEFAULT 0,
                    total_calls INTEGER DEFAULT 0,
                    by_entity_type_json TEXT,
                    by_model_json TEXT,
                    total_iterations INTEGER DEFAULT 0,
                    wasted_iterations INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_generation_runs_project
                    ON generation_runs(project_id);
                CREATE INDEX IF NOT EXISTS idx_generation_runs_type
                    ON generation_runs(run_type);
                CREATE INDEX IF NOT EXISTS idx_generation_runs_started
                    ON generation_runs(started_at);
            """)
            conn.commit()

            # Run migrations for schema updates
            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """
        Apply pending schema migrations to the connected SQLite database.

        Ensures the custom_modes table contains the `size_preference` column and adds it with a default value of "medium" if missing.
        Also adds refinement effectiveness tracking columns to world_entity_scores.

        Parameters:
            conn (sqlite3.Connection): Active SQLite connection whose schema may be modified.
        """
        # Migration: custom_modes size_preference
        cursor = conn.execute("PRAGMA table_info(custom_modes)")
        custom_modes_columns = {row[1] for row in cursor.fetchall()}

        if "size_preference" not in custom_modes_columns:
            logger.info("Migrating custom_modes: adding size_preference column")
            conn.execute(
                "ALTER TABLE custom_modes ADD COLUMN size_preference TEXT NOT NULL DEFAULT 'medium'"
            )
            conn.commit()

        # Migration: world_entity_scores refinement effectiveness tracking
        cursor = conn.execute("PRAGMA table_info(world_entity_scores)")
        world_entity_columns = {row[1] for row in cursor.fetchall()}

        # New columns for refinement effectiveness tracking
        refinement_columns = [
            ("early_stop_triggered", "INTEGER DEFAULT 0"),
            ("threshold_met", "INTEGER DEFAULT 0"),
            ("peak_score", "REAL"),
            ("final_score", "REAL"),
            ("score_progression_json", "TEXT"),
            ("consecutive_degradations", "INTEGER DEFAULT 0"),
            ("best_iteration", "INTEGER DEFAULT 0"),
            ("quality_threshold", "REAL"),
            ("max_iterations", "INTEGER"),
        ]

        for col_name, col_def in refinement_columns:
            if col_name not in world_entity_columns:
                logger.info(f"Migrating world_entity_scores: adding {col_name} column")
                conn.execute(f"ALTER TABLE world_entity_scores ADD COLUMN {col_name} {col_def}")

        conn.commit()

    # === Generation Scores ===

    def record_score(
        self,
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
        """
        Record a single generation score row for a project and model.

        Parameters:
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
            with sqlite3.connect(self.db_path) as conn:
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
        self,
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
            with sqlite3.connect(self.db_path) as conn:
                sql = f"UPDATE generation_scores SET {set_clause} WHERE id = ?"
                conn.execute(sql, values)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update score {score_id}: {e}", exc_info=True)
            raise

    def update_performance_metrics(
        self,
        score_id: int,
        *,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        tokens_per_second: float | None = None,
        vram_used_gb: float | None = None,
    ) -> None:
        """Update an existing score with performance metrics.

        Args:
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
            with sqlite3.connect(self.db_path) as conn:
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
        self,
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_scores_for_project(self, project_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all generation scores for a project ordered by most recent first.

        Returns:
            list[dict[str, Any]]: List of rows from `generation_scores` as dictionaries, ordered by `timestamp` descending.
        """
        with sqlite3.connect(self.db_path) as conn:
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

    def get_latest_score_for_chapter(
        self, project_id: str, chapter_id: str
    ) -> dict[str, Any] | None:
        """Get the most recent score for a specific project chapter.

        Args:
            project_id: The project ID.
            chapter_id: The chapter ID.

        Returns:
            The latest score dict or None if not found.
        """
        logger.debug(f"Fetching latest score for project {project_id} chapter {chapter_id}")
        with sqlite3.connect(self.db_path) as conn:
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
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> int:
        """
        Return count of generation score records, optionally filtered by model, agent role, or genre.

        Parameters:
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

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return int(result) if result is not None else 0

    # === Model Performance Aggregates ===

    def update_model_performance(
        self,
        model_id: str,
        agent_role: str,
        genre: str | None = None,
    ) -> None:
        """Recalculate aggregated performance for a model/role/genre combo."""
        genre_value = genre or ""
        genre_condition = "genre = ?" if genre else "(genre IS NULL OR genre = '')"

        with sqlite3.connect(self.db_path) as conn:
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
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated model performance."""
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_top_models_for_role(
        self,
        agent_role: str,
        limit: int = 5,
        min_samples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get top performing models for a role."""
        with sqlite3.connect(self.db_path) as conn:
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

    # === Recommendations ===

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

    # === Custom Modes ===

    def save_custom_mode(
        self,
        mode_id: str,
        name: str,
        agent_models: dict[str, str],
        agent_temperatures: dict[str, float],
        size_preference: str = "medium",
        vram_strategy: str = "adaptive",
        description: str = "",
        is_experimental: bool = False,
    ) -> None:
        """Save or update a custom generation mode.

        Uses INSERT ... ON CONFLICT to preserve created_at on updates.

        Args:
            mode_id: Unique identifier for the mode.
            name: Display name for the mode.
            agent_models: Mapping of agent_role to model_id.
            agent_temperatures: Mapping of agent_role to temperature.
            size_preference: Model size preference (largest, medium, smallest).
            vram_strategy: VRAM management strategy.
            description: User-facing description.
            is_experimental: Whether this mode tries variations.

        Raises:
            sqlite3.Error: If database operation fails.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO custom_modes (
                        id, name, description, agent_models_json,
                        agent_temperatures_json, size_preference, vram_strategy,
                        is_experimental, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        agent_models_json = excluded.agent_models_json,
                        agent_temperatures_json = excluded.agent_temperatures_json,
                        size_preference = excluded.size_preference,
                        vram_strategy = excluded.vram_strategy,
                        is_experimental = excluded.is_experimental,
                        updated_at = datetime('now')
                    """,
                    (
                        mode_id,
                        name,
                        description,
                        json.dumps(agent_models),
                        json.dumps(agent_temperatures),
                        size_preference,
                        vram_strategy,
                        1 if is_experimental else 0,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(
                "Failed to save custom mode mode_id=%s name=%s: %s",
                mode_id,
                name,
                e,
                exc_info=True,
            )
            raise

    def get_custom_mode(self, mode_id: str) -> dict[str, Any] | None:
        """Get a custom mode by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM custom_modes WHERE id = ?",
                (mode_id,),
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["agent_models"] = json.loads(result.pop("agent_models_json"))
                result["agent_temperatures"] = json.loads(result.pop("agent_temperatures_json"))
                return result
            return None

    def list_custom_modes(self) -> list[dict[str, Any]]:
        """List all custom modes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM custom_modes ORDER BY name")
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result["agent_models"] = json.loads(result.pop("agent_models_json"))
                result["agent_temperatures"] = json.loads(result.pop("agent_temperatures_json"))
                results.append(result)
            return results

    def delete_custom_mode(self, mode_id: str) -> bool:
        """Delete a custom mode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM custom_modes WHERE id = ?",
                (mode_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    # === Analytics Queries ===

    def get_quality_vs_speed_data(
        self,
        agent_role: str | None = None,
        min_samples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get data for quality vs speed scatter plot."""
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_genre_breakdown(self, model_id: str) -> list[dict[str, Any]]:
        """Get performance breakdown by genre for a model."""
        with sqlite3.connect(self.db_path) as conn:
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

    def get_unique_genres(self) -> list[str]:
        """Get list of unique genres from scores."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT genre FROM generation_scores WHERE genre IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_average_score(
        self,
        metric: str,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> float | None:
        """Get average of a specific metric.

        Args:
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

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return float(result) if result is not None else None

    def get_model_summaries(
        self,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> list[ModelPerformanceSummary]:
        """Get model performance summaries.

        Returns list of ModelPerformanceSummary objects.
        """
        query = "SELECT * FROM model_performance WHERE 1=1"
        params: list[Any] = []

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)
        if genre:
            query += " AND genre = ?"
            params.append(genre)

        query += " ORDER BY avg_prose_quality DESC"

        with sqlite3.connect(self.db_path) as conn:
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
                evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else None
                try:
                    rec_type = RecommendationType(row["recommendation_type"])
                except ValueError:
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

    def get_all_scores(
        self,
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

        with sqlite3.connect(self.db_path) as conn:
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

    def export_scores_csv(self, output_path: Path | str) -> int:
        """Export all scores to CSV.

        Returns:
            Number of rows exported.
        """
        import csv

        output_path = Path(output_path)
        with sqlite3.connect(self.db_path) as conn:
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

    # === World Entity Scores ===

    def record_world_entity_score(
        self,
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
    ) -> int:
        """Record a world entity quality score with refinement effectiveness metrics.

        Args:
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

        Returns:
            The ID of the inserted record.

        Raises:
            sqlite3.Error: If database operation fails.
        """
        # Extract up to 4 score values
        score_values = list(scores.values())
        score_1 = score_values[0] if len(score_values) > 0 else None
        score_2 = score_values[1] if len(score_values) > 1 else None
        score_3 = score_values[2] if len(score_values) > 2 else None
        score_4 = score_values[3] if len(score_values) > 3 else None
        average_score = scores.get("average")

        # Convert score progression to JSON
        score_progression_json = json.dumps(score_progression) if score_progression else None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO world_entity_scores (
                        project_id, entity_id, entity_type, entity_name, model_id,
                        score_1, score_2, score_3, score_4, average_score,
                        iterations_used, generation_time_seconds, feedback,
                        early_stop_triggered, threshold_met, peak_score, final_score,
                        score_progression_json, consecutive_degradations, best_iteration,
                        quality_threshold, max_iterations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        self,
        project_id: str | None = None,
        entity_type: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get world entity scores with optional filters.

        Args:
            project_id: Filter by project.
            entity_type: Filter by entity type.
            model_id: Filter by model.
            limit: Maximum number of results.

        Returns:
            List of score records as dictionaries.
        """
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_world_quality_summary(
        self,
        entity_type: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Get summary statistics for world entity quality scores.

        Args:
            entity_type: Optional filter by entity type.
            model_id: Optional filter by model.

        Returns:
            Dictionary with summary statistics.
        """
        where_clauses = ["1=1"]
        params: list[Any] = []

        if entity_type:
            where_clauses.append("entity_type = ?")
            params.append(entity_type)
        if model_id:
            where_clauses.append("model_id = ?")
            params.append(model_id)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
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
        self,
        entity_type: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Get count of world entity scores.

        Args:
            entity_type: Optional filter by entity type.
            model_id: Optional filter by model.

        Returns:
            Number of matching records.
        """
        query = "SELECT COUNT(*) FROM world_entity_scores WHERE 1=1"
        params: list[Any] = []

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return int(result) if result is not None else 0

    # === Content Statistics ===

    def get_content_statistics(
        self,
        project_id: str | None = None,
        agent_role: str | None = None,
    ) -> dict:
        """Get content statistics from generation scores.

        Args:
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

        with sqlite3.connect(self.db_path) as conn:
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

    # === Time Series Data ===

    def get_quality_time_series(
        self,
        metric: str = "prose_quality",
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get time series data for quality metrics.

        Args:
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [{"timestamp": row[0], "value": row[1]} for row in cursor.fetchall()]

    def get_daily_quality_averages(
        self,
        metric: str = "prose_quality",
        days: int = 30,
        agent_role: str | None = None,
    ) -> list[dict]:
        """Get daily average quality metrics.

        Args:
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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [
                {"date": row[0], "avg_value": row[1], "sample_count": row[2]}
                for row in cursor.fetchall()
            ]

    # === Prompt Template Metrics ===

    def record_prompt_metrics(
        self,
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
            with sqlite3.connect(self.db_path) as conn:
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
        self,
        agent_role: str | None = None,
        task: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get prompt performance statistics.

        Args:
            agent_role: Filter by agent role.
            task: Filter by task name.
            days: Number of days to include.

        Returns:
            List of analytics records with aggregated stats per template.

        Raises:
            ValueError: If days is negative.
        """
        # Validate days to prevent SQL injection
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_prompt_metrics_summary(self) -> dict[str, Any]:
        """Get overall summary of prompt metrics.

        Returns:
            Dictionary with summary statistics.
        """
        with sqlite3.connect(self.db_path) as conn:
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
        self,
        prompt_hash: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get metrics for a specific template hash.

        Args:
            prompt_hash: Template hash to query.
            limit: Maximum number of records.

        Returns:
            List of metric records for the template.
        """
        with sqlite3.connect(self.db_path) as conn:
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

    def get_prompt_error_summary(self, days: int = 7) -> list[dict[str, Any]]:
        """Get summary of prompt errors for debugging.

        Args:
            days: Number of days to include.

        Returns:
            List of error summaries grouped by agent/task.

        Raises:
            ValueError: If days is negative.
        """
        # Validate days to prevent SQL injection
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

        with sqlite3.connect(self.db_path) as conn:
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

    # === Refinement Effectiveness Analytics ===

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
            - threshold_met_rate: Percentage of entities that met their quality threshold (rounded to 0.1%).
            - early_stop_rate: Percentage of entities that triggered early stopping due to degradation (rounded to 0.1%).
            - avg_iterations: Average number of refinement iterations used (rounded to 2 decimals) or None if not available.
            - avg_score_loss: Average difference between peak and final score (rounded to 3 decimals).
            - best_is_final_rate: Percentage of cases where the best iteration equals the returned iteration (rounded to 0.1%).
            - by_entity_type: List of per-entity-type breakdowns; each item contains:
                - entity_type: The entity type name.
                - count: Number of entities of that type.
                - threshold_met_rate: Percentage for that type (rounded to 0.1%).
                - early_stop_rate: Percentage for that type (rounded to 0.1%).
                - avg_iterations: Average iterations for that type (rounded to 2 decimals) or None.
                - avg_score_loss: Average score loss for that type (rounded to 3 decimals).
        """
        logger.debug(
            "Getting refinement effectiveness summary: entity_type=%s, days=%d",
            entity_type,
            days,
        )
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

        where_clauses = ["DATE(timestamp) >= DATE('now', ?)"]
        params: list[Any] = [f"-{validated_days} days"]

        if entity_type:
            where_clauses.append("entity_type = ?")
            params.append(entity_type)

        where_sql = " AND ".join(where_clauses)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Access columns by name for robustness
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

    # === Generation Cost Tracking ===

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
                return result
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
            limit (int): Maximum number of runs to return; results are ordered by `started_at` descending.
        
        Returns:
            list[dict[str, Any]]: A list of run records. Each record includes parsed `by_entity_type` and `by_model` dicts (deserialized from stored JSON) and other columns from the `generation_runs` table.
        """
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
            days (int): Number of days to include (must be greater than or equal to 0).
        
        Returns:
            dict: Summary containing:
                - total_runs (int): Number of runs in the window.
                - total_tokens (int): Sum of tokens across runs.
                - total_time_seconds (float): Sum of run times in seconds.
                - total_calls (int): Sum of API/model call counts.
                - total_iterations (int): Sum of iterations across runs.
                - wasted_iterations (int): Sum of wasted iterations.
                - avg_tokens_per_run (float): Average tokens per run.
                - avg_time_per_run (float): Average run time in seconds.
                - by_run_type (list[dict]): Breakdown per run_type with keys
                    "run_type", "count", "tokens", and "time_seconds".
                - efficiency_ratio (float): Ratio of useful iterations to total iterations (0.0–1.0).
        
        Raises:
            ValueError: If `days` is negative.
        """
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

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

            return summary

    def get_model_cost_breakdown(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Produce a per-model cost and usage breakdown for generation activity within a recent time window.
        
        Parameters:
            project_id (str | None): Optional project identifier to filter results; include all projects if None.
            days (int): Lookback window in days; must be zero or a positive integer.
        
        Returns:
            list[dict[str, Any]]: A list of mappings, one per model, with keys:
                - model_id (str): Model identifier.
                - call_count (int): Number of generation calls recorded for the model.
                - total_tokens (int): Sum of tokens generated (0 if no tokens recorded).
                - total_time_seconds (float): Sum of generation time in seconds (0.0 if not recorded).
                - avg_tokens_per_second (float | None): Average tokens per second for the model, or None if unavailable.
                - avg_quality (float | None): Average prose quality score for the model, or None if unavailable.
        
        Raises:
            ValueError: If `days` is negative.
        """
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

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
            return [
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
            list[dict[str, Any]]: Each dictionary contains:
                - entity_type: The world entity type.
                - count: Number of entities sampled.
                - total_time_seconds: Sum of generation_time_seconds for the entity type.
                - avg_iterations: Average iterations_used for the entity type.
                - wasted_iterations: Sum of iterations considered wasted (where threshold_met is false).
                - avg_quality: Average of average_score for the entity type.
        
        Raises:
            ValueError: If `days` is negative.
        """
        validated_days = int(days)
        if validated_days < 0:
            raise ValueError("days must be a non-negative integer")

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
            return [
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