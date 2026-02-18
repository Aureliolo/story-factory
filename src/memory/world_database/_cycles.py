"""Accepted cycle operations for WorldDatabase."""

import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_cycle_hash(cycle_edges: list[tuple[str, str, str]]) -> str:
    """Compute a deterministic hash for a cycle independent of traversal start edge.

    Edges are sorted lexicographically, so rotations of the same directed cycle
    produce the same hash. A cycle traversed in reverse direction is treated as a
    distinct cycle (different edge directions = different hash).

    Args:
        cycle_edges: List of (source_id, relation_type, target_id) tuples.

    Returns:
        16-character hex hash string.
    """
    logger.debug("Computing cycle hash for %d edges", len(cycle_edges))
    sorted_edges = sorted(cycle_edges)
    edge_str = "|".join(f"{src},{rel},{tgt}" for src, rel, tgt in sorted_edges)
    digest = hashlib.sha256(edge_str.encode()).hexdigest()[:16]
    logger.debug("Computed cycle hash: %s", digest)
    return digest


def validate_cycle_hash(cycle_hash: str) -> None:
    """Validate that a cycle hash is exactly 16 hex characters.

    Args:
        cycle_hash: Hash string to validate.

    Raises:
        ValueError: If cycle_hash is not exactly 16 hex characters.
    """
    if not cycle_hash or len(cycle_hash) != 16:
        raise ValueError(f"Invalid cycle hash (expected 16 hex chars): {cycle_hash!r}")
    try:
        int(cycle_hash, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid cycle hash (expected 16 hex chars): {cycle_hash!r}") from exc


def accept_cycle(db, cycle_hash: str) -> None:
    """Mark a circular chain as accepted/intentional.

    Args:
        db: WorldDatabase instance.
        cycle_hash: Hash of the cycle (from compute_cycle_hash).

    Raises:
        ValueError: If cycle_hash is not exactly 16 hex characters.
    """
    validate_cycle_hash(cycle_hash)
    logger.info("Accepting cycle with hash: %s", cycle_hash)
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO accepted_cycles (cycle_hash, accepted_at) VALUES (?, ?)",
            (cycle_hash, datetime.now().isoformat()),
        )
        db.conn.commit()


def remove_accepted_cycle(db, cycle_hash: str) -> bool:
    """Remove a previously accepted cycle.

    Args:
        db: WorldDatabase instance.
        cycle_hash: Hash of the cycle to un-accept.

    Returns:
        True if the cycle was removed, False if it wasn't found.

    Raises:
        ValueError: If cycle_hash is not exactly 16 hex characters.
    """
    validate_cycle_hash(cycle_hash)
    logger.info("Removing accepted cycle with hash: %s", cycle_hash)
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        cursor.execute("DELETE FROM accepted_cycles WHERE cycle_hash = ?", (cycle_hash,))
        db.conn.commit()
        removed: bool = cursor.rowcount > 0
    logger.debug("Cycle %s removal result: %s", cycle_hash, removed)
    return removed


def get_accepted_cycles(db) -> set[str]:
    """Get all accepted cycle hashes.

    Args:
        db: WorldDatabase instance.

    Returns:
        Set of accepted cycle hash strings.
    """
    logger.debug("Loading accepted cycles from database")
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        cursor.execute("SELECT cycle_hash FROM accepted_cycles")
        rows = cursor.fetchall()
    accepted = {row[0] for row in rows}
    logger.debug("Loaded %d accepted cycles", len(accepted))
    return accepted
