"""Custom mode functions for ModeDatabase.

Handles saving, retrieving, listing, and deleting custom generation modes.
"""

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger("src.memory.mode_database._custom_modes")


def save_custom_mode(
    db,
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
        db: ModeDatabase instance.
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
        with db._lock:
            with sqlite3.connect(db.db_path) as conn:
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


def get_custom_mode(db, mode_id: str) -> dict[str, Any] | None:
    """Get a custom mode by ID."""
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM custom_modes WHERE id = ?",
                (mode_id,),
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                try:
                    result["agent_models"] = json.loads(result.pop("agent_models_json"))
                    result["agent_temperatures"] = json.loads(result.pop("agent_temperatures_json"))
                except json.JSONDecodeError:
                    logger.warning(
                        "Corrupt JSON in custom mode id=%s, skipping record",
                        mode_id,
                    )
                    return None
                return result
            return None


def list_custom_modes(db) -> list[dict[str, Any]]:
    """List all custom modes."""
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM custom_modes ORDER BY name")
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                try:
                    result["agent_models"] = json.loads(result.pop("agent_models_json"))
                    result["agent_temperatures"] = json.loads(result.pop("agent_temperatures_json"))
                except json.JSONDecodeError:
                    logger.warning(
                        "Corrupt JSON in custom mode id=%s, skipping record",
                        result.get("id", "unknown"),
                    )
                    continue
                results.append(result)
            return results


def delete_custom_mode(db, mode_id: str) -> bool:
    """Delete a custom mode."""
    with db._lock:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM custom_modes WHERE id = ?",
                (mode_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
