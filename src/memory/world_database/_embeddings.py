"""Embedding storage and vector search operations for WorldDatabase.

Provides functions to store, search, and manage vector embeddings in the
vec_embeddings virtual table (sqlite-vec) alongside the embedding_metadata
tracking table.
"""

import hashlib
import logging
import struct
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import WorldDatabase

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Compute a SHA-256 hash of text content for change detection.

    Args:
        text: Content text to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_embedding(
    db: WorldDatabase,
    source_id: str,
    content_type: str,
    text: str,
    embedding: list[float],
    model: str,
    entity_type: str = "",
    chapter_number: int | None = None,
) -> bool:
    """Insert or replace an embedding in the vec0 table and update metadata.

    Skips the operation if the content hash hasn't changed since the last embedding
    with the same model, avoiding redundant embedding API calls.

    Args:
        db: WorldDatabase instance.
        source_id: Unique identifier for the content (entity_id, rel_id, etc.).
        content_type: Type of content (entity, relationship, event, fact, rule, etc.).
        text: The display text that was embedded.
        embedding: The embedding vector as a list of floats.
        model: The embedding model used to generate the vector.
        entity_type: Optional entity type for filtered queries.
        chapter_number: Optional chapter number for filtered queries.

    Returns:
        True if the embedding was upserted, False if skipped (unchanged).
    """
    content_hash_val = _content_hash(text)
    now = datetime.now().isoformat()

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Check if content is unchanged
        cursor.execute(
            "SELECT content_hash, embedding_model FROM embedding_metadata WHERE source_id = ?",
            (source_id,),
        )
        existing = cursor.fetchone()
        if existing and existing[0] == content_hash_val and existing[1] == model:
            logger.debug("Embedding unchanged for %s, skipping upsert", source_id)
            return False

        try:
            # Delete existing embedding if present (vec0 doesn't support UPDATE)
            if existing:
                cursor.execute(
                    "DELETE FROM vec_embeddings WHERE source_id = ? AND content_type = ?",
                    (source_id, content_type),
                )

            # Insert new embedding into vec0
            embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
            cursor.execute(
                """
                INSERT INTO vec_embeddings (
                    embedding, content_type, source_id, entity_type,
                    chapter_number, display_text, embedding_model, embedded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding_blob,
                    content_type,
                    source_id,
                    entity_type,
                    chapter_number,
                    text[:500],  # Truncate display text for storage efficiency
                    model,
                    now,
                ),
            )

            # Update metadata tracking table
            cursor.execute(
                """
                INSERT OR REPLACE INTO embedding_metadata
                (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_id, content_type, content_hash_val, model, now, len(embedding)),
            )

            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise

    logger.debug("Upserted embedding for %s (%s, %d dims)", source_id, content_type, len(embedding))
    return True


def delete_embedding(db: WorldDatabase, source_id: str) -> bool:
    """Remove an embedding and its metadata when the source content is deleted.

    Args:
        db: WorldDatabase instance.
        source_id: The source content identifier.

    Returns:
        True if an embedding was deleted, False if not found or unavailable.
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Query content_type BEFORE deleting metadata (needed for vec0 partition key)
        content_type = None
        cursor.execute(
            "SELECT content_type FROM embedding_metadata WHERE source_id = ?", (source_id,)
        )
        row = cursor.fetchone()
        if row:
            content_type = row[0]

        try:
            # Clean up metadata (regular table)
            cursor.execute("DELETE FROM embedding_metadata WHERE source_id = ?", (source_id,))
            metadata_deleted = cursor.rowcount > 0

            # Clean up vec0 table
            vec_deleted = False
            if content_type:
                cursor.execute(
                    "DELETE FROM vec_embeddings WHERE source_id = ? AND content_type = ?",
                    (source_id, content_type),
                )
                vec_deleted = cursor.rowcount > 0
            else:
                # Fallback: try without partition key if metadata was missing
                cursor.execute(
                    "DELETE FROM vec_embeddings WHERE source_id = ?",
                    (source_id,),
                )
                vec_deleted = cursor.rowcount > 0

            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise

    deleted = metadata_deleted or vec_deleted
    if deleted:
        logger.debug("Deleted embedding for %s", source_id)
    return deleted


def search_similar(
    db: WorldDatabase,
    query_embedding: list[float],
    k: int = 10,
    content_type: str | None = None,
    entity_type: str | None = None,
    chapter_number: int | None = None,
) -> list[dict[str, Any]]:
    """Perform KNN vector similarity search in the embeddings table.

    Args:
        db: WorldDatabase instance.
        query_embedding: The query vector to find neighbors for.
        k: Number of nearest neighbors to return.
        content_type: Optional filter by content type (uses partition key).
        entity_type: Optional filter by entity type.
        chapter_number: Optional filter by chapter number.

    Returns:
        List of dicts with keys: source_id, content_type, entity_type,
        display_text, distance. Sorted by distance ascending (most similar first).
    """
    if not query_embedding:
        logger.debug("search_similar skipped: empty query embedding")
        return []

    query_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    # Over-fetch when post-query filters will reduce results below requested k
    fetch_k = k * 3 if (entity_type or chapter_number is not None) else k

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Build query based on filters
        if content_type:
            # Use partition key for efficient filtered search
            cursor.execute(
                """
                SELECT source_id, content_type, entity_type, chapter_number,
                       display_text, distance
                FROM vec_embeddings
                WHERE embedding MATCH ? AND k = ? AND content_type = ?
                ORDER BY distance
                """,
                (query_blob, fetch_k, content_type),
            )
        else:
            cursor.execute(
                """
                SELECT source_id, content_type, entity_type, chapter_number,
                       display_text, distance
                FROM vec_embeddings
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (query_blob, fetch_k),
            )

        results = []
        for row in cursor.fetchall():
            # Apply post-query filters (entity_type, chapter_number)
            if entity_type and row[2] != entity_type:
                continue
            if chapter_number is not None and row[3] != chapter_number:
                continue

            results.append(
                {
                    "source_id": row[0],
                    "content_type": row[1],
                    "entity_type": row[2],
                    "chapter_number": row[3],
                    "display_text": row[4],
                    "distance": row[5],
                }
            )
            # Stop once we have enough results after filtering
            if len(results) >= k:
                break

    logger.debug(
        "search_similar: found %d results (k=%d, content_type=%s)",
        len(results),
        k,
        content_type,
    )
    return results


def get_embedding_stats(db: WorldDatabase) -> dict[str, Any]:
    """Get statistics about stored embeddings.

    Args:
        db: WorldDatabase instance.

    Returns:
        Dict with counts per content_type, total count, models used, and
        staleness info. Returns empty stats if no embeddings exist.
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Count by content_type
        cursor.execute(
            "SELECT content_type, COUNT(*) FROM embedding_metadata GROUP BY content_type"
        )
        counts_by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Total count
        cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
        total = cursor.fetchone()[0]

        # Models used
        cursor.execute("SELECT DISTINCT embedding_model FROM embedding_metadata")
        models = [row[0] for row in cursor.fetchall()]

        # Dimensions used
        cursor.execute("SELECT DISTINCT embedding_dim FROM embedding_metadata")
        dimensions = [row[0] for row in cursor.fetchall()]

    logger.debug(
        "Embedding stats: %d total, %d models",
        total,
        len(models),
    )
    return {
        "total": total,
        "by_content_type": counts_by_type,
        "models": models,
        "dimensions": dimensions,
        "vec_available": True,  # Always True — sqlite-vec is mandatory
    }


def needs_reembedding(db: WorldDatabase, current_model: str) -> bool:
    """Check if any embeddings use a different model than the current one.

    Args:
        db: WorldDatabase instance.
        current_model: The currently configured embedding model.

    Returns:
        True if any embeddings exist with a different model, False otherwise.
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM embedding_metadata WHERE embedding_model != ?",
            (current_model,),
        )
        result = cursor.fetchone()
        stale_count = result[0] if result else 0

    if stale_count > 0:
        logger.info(
            "%d embeddings use a different model than '%s' and need re-embedding",
            stale_count,
            current_model,
        )
    return stale_count > 0


def clear_all_embeddings(db: WorldDatabase) -> None:
    """Remove all embeddings from both vec0 and metadata tables.

    Used when the embedding model changes and all vectors must be regenerated.

    Args:
        db: WorldDatabase instance.
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Query actual dimensions BEFORE deleting metadata
        cursor.execute("SELECT DISTINCT embedding_dim FROM embedding_metadata LIMIT 1")
        dim_row = cursor.fetchone()
        dimensions = dim_row[0] if dim_row else 1024

        cursor.execute("DELETE FROM embedding_metadata")
        metadata_count = cursor.rowcount

        # Drop and recreate vec0 table (most reliable way to clear it)
        cursor.execute("DROP TABLE IF EXISTS vec_embeddings")
        cursor.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{int(dimensions)}],
                content_type text partition key,
                +source_id text,
                +entity_type text,
                +chapter_number integer,
                +display_text text,
                +embedding_model text,
                +embedded_at text
            )
            """
        )

        db.conn.commit()

    logger.info("Cleared all embeddings (%d metadata records removed)", metadata_count)


def recreate_vec_table(db: WorldDatabase, dimensions: int) -> None:
    """Recreate the vec0 table with new dimensions.

    Used when the embedding model changes and produces vectors of a different
    dimensionality.

    Args:
        db: WorldDatabase instance.
        dimensions: New embedding dimension size.
    """
    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS vec_embeddings")
        cursor.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{int(dimensions)}],
                content_type text partition key,
                +source_id text,
                +entity_type text,
                +chapter_number integer,
                +display_text text,
                +embedding_model text,
                +embedded_at text
            )
            """
        )

        # Also clear metadata since old embeddings are incompatible
        cursor.execute("DELETE FROM embedding_metadata")

        db.conn.commit()

    logger.info("Recreated vec_embeddings table with %d dimensions", dimensions)
