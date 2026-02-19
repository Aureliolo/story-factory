"""World settings persistence for WorldDatabase."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.world_settings import WorldSettings

logger = logging.getLogger(__name__)


def get_world_settings(db) -> WorldSettings | None:
    """Get world settings including calendar configuration.

    Args:
        db: WorldDatabase instance.

    Returns:
        WorldSettings instance or None if not configured.
    """
    # Import here to avoid circular imports
    from src.memory.world_settings import WorldSettings

    logger.debug("Loading world settings from database")
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT id, calendar_json, timeline_start_year, timeline_end_year,
                   validate_temporal_consistency, created_at, updated_at
            FROM world_settings
            LIMIT 1
            """
        )
        row = cursor.fetchone()

    if not row:
        logger.debug("No world settings found in database")
        return None

    calendar_data = json.loads(row[1]) if row[1] else None
    settings_data = {
        "id": row[0],
        "calendar": calendar_data,
        "timeline_start_year": row[2],
        "timeline_end_year": row[3],
        "validate_temporal_consistency": bool(row[4]),
        "created_at": row[5],
        "updated_at": row[6],
    }
    logger.debug("Loaded world settings: id=%s, has_calendar=%s", row[0], calendar_data is not None)
    return WorldSettings.from_dict(settings_data)


def save_world_settings(db, settings) -> None:
    """Save or update world settings.

    Args:
        db: WorldDatabase instance.
        settings: WorldSettings instance to save.
    """
    logger.debug("Saving world settings: id=%s", settings.id)
    calendar_json = json.dumps(settings.calendar.to_dict()) if settings.calendar else None
    now = datetime.now().isoformat()

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO world_settings
            (id, calendar_json, timeline_start_year, timeline_end_year,
             validate_temporal_consistency, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                settings.id,
                calendar_json,
                settings.timeline_start_year,
                settings.timeline_end_year,
                int(settings.validate_temporal_consistency),
                settings.created_at.isoformat()
                if hasattr(settings.created_at, "isoformat")
                else now,
                now,
            ),
        )
        db.conn.commit()
    logger.info("Saved world settings: id=%s", settings.id)
