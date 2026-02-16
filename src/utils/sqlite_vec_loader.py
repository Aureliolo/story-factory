"""Utility for loading the sqlite-vec extension into SQLite connections.

sqlite-vec adds vector search capabilities to SQLite, enabling KNN queries
for semantic similarity search in the world database.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)


def load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into an existing SQLite connection.

    Handles ImportError (sqlite-vec not installed) and extension load failures
    gracefully, logging warnings instead of raising exceptions. This allows
    the application to function without vector search when sqlite-vec is
    unavailable.

    Args:
        conn: An open sqlite3 connection to load the extension into.

    Returns:
        True if the extension was loaded successfully, False otherwise.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "sqlite-vec package not installed. "
            "Vector search features will be unavailable. "
            "Install with: pip install sqlite-vec"
        )
        return False

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        logger.info("sqlite-vec extension loaded successfully")
        return True
    except Exception as e:
        logger.warning("Failed to load sqlite-vec extension: %s", e)
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False
