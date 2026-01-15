"""Database for model scoring and mode management.

Stores generation scores, model performance metrics, and tuning recommendations
for the adaptive learning system.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModeDatabase:
    """SQLite database for model scoring and learning data.

    Tables:
    - generation_scores: Per-generation metrics (quality, speed, implicit signals)
    - model_performance: Aggregated model performance by role and genre
    - recommendations: Tuning recommendation history
    - custom_modes: User-defined generation modes
    """

    def __init__(self, db_path: Path | str):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
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
                    genre TEXT,

                    avg_prose_quality REAL,
                    avg_instruction_following REAL,
                    avg_consistency REAL,
                    avg_tokens_per_second REAL,

                    sample_count INTEGER DEFAULT 0,
                    last_updated TEXT,

                    PRIMARY KEY (model_id, agent_role, COALESCE(genre, ''))
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

                -- Custom generation modes
                CREATE TABLE IF NOT EXISTS custom_modes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    agent_models_json TEXT NOT NULL,
                    agent_temperatures_json TEXT NOT NULL,
                    vram_strategy TEXT NOT NULL DEFAULT 'adaptive',
                    is_experimental INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)
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
        """Record a generation score.

        Returns:
            The ID of the inserted record.
        """
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
        """Update an existing score with additional metrics."""
        updates = []
        values = []

        if prose_quality is not None:
            updates.append("prose_quality = ?")
            values.append(prose_quality)
        if instruction_following is not None:
            updates.append("instruction_following = ?")
            values.append(instruction_following)
        if consistency_score is not None:
            updates.append("consistency_score = ?")
            values.append(consistency_score)
        if was_regenerated is not None:
            updates.append("was_regenerated = ?")
            values.append(1 if was_regenerated else 0)
        if edit_distance is not None:
            updates.append("edit_distance = ?")
            values.append(edit_distance)
        if user_rating is not None:
            updates.append("user_rating = ?")
            values.append(user_rating)

        if not updates:
            return

        values.append(score_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE generation_scores SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()

    def get_scores_for_model(
        self,
        model_id: str,
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent scores for a model."""
        query = "SELECT * FROM generation_scores WHERE model_id = ?"
        params: list = [model_id]

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

    def get_scores_for_project(self, project_id: str) -> list[dict]:
        """Get all scores for a project."""
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

    def get_score_count(self, model_id: str | None = None) -> int:
        """Get total number of scores, optionally filtered by model."""
        query = "SELECT COUNT(*) FROM generation_scores"
        params: list = []

        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)

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
        genre_condition = "genre = ?" if genre else "genre IS NULL"
        genre_value = genre or ""

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
                (model_id, agent_role, genre) if genre else (model_id, agent_role),
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
    ) -> list[dict]:
        """Get aggregated model performance."""
        query = "SELECT * FROM model_performance WHERE 1=1"
        params: list = []

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
    ) -> list[dict]:
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
        evidence: dict | None = None,
        affected_role: str | None = None,
        expected_improvement: str | None = None,
    ) -> int:
        """Record a tuning recommendation."""
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

    def update_recommendation_outcome(
        self,
        recommendation_id: int,
        was_applied: bool,
        user_feedback: str | None = None,
    ) -> None:
        """Update the outcome of a recommendation."""
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

    def get_pending_recommendations(self) -> list[dict]:
        """Get recommendations that haven't been actioned."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM recommendations
                WHERE was_applied = 0 AND user_feedback IS NULL
                ORDER BY timestamp DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recommendation_history(self, limit: int = 50) -> list[dict]:
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
        vram_strategy: str = "adaptive",
        description: str = "",
        is_experimental: bool = False,
    ) -> None:
        """Save or update a custom generation mode."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO custom_modes (
                    id, name, description, agent_models_json,
                    agent_temperatures_json, vram_strategy, is_experimental,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    mode_id,
                    name,
                    description,
                    json.dumps(agent_models),
                    json.dumps(agent_temperatures),
                    vram_strategy,
                    1 if is_experimental else 0,
                ),
            )
            conn.commit()

    def get_custom_mode(self, mode_id: str) -> dict | None:
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

    def list_custom_modes(self) -> list[dict]:
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
    ) -> list[dict]:
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
        params: list = [min_samples]

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_genre_breakdown(self, model_id: str) -> list[dict]:
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
