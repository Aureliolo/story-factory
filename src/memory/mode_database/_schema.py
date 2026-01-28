"""Schema initialization and migration functions for ModeDatabase."""

import logging
import sqlite3

logger = logging.getLogger(__name__)


def init_db(db) -> None:
    """Create and initialize the SQLite schema required by the ModeDatabase.

    Creates tables for generation scores, aggregated model performance, recommendations,
    world entity scores, prompt metrics, and custom modes, along with relevant indexes,
    then applies any pending schema migrations.

    Args:
        db: ModeDatabase instance.
    """
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
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
            run_migrations(db, conn)


def run_migrations(db, conn: sqlite3.Connection) -> None:
    """Apply pending schema migrations to the connected SQLite database.

    Ensures the custom_modes table contains the `size_preference` column and adds it
    with a default value of "medium" if missing. Also adds refinement effectiveness
    tracking columns to world_entity_scores.

    Args:
        db: ModeDatabase instance.
        conn: Active SQLite connection whose schema may be modified.
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
