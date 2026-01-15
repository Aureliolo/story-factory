"""SQLite-backed worldbuilding database with NetworkX integration."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from memory.entities import Entity, EventParticipant, Relationship, WorldEvent

logger = logging.getLogger(__name__)


class WorldDatabase:
    """SQLite-backed worldbuilding database with NetworkX integration."""

    def __init__(self, db_path: Path | str):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._graph: nx.DiGraph | None = None

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Core entity table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                attributes TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Relationships table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                strength REAL DEFAULT 1.0,
                bidirectional INTEGER DEFAULT 0,
                attributes TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            )
        """
        )

        # Events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                chapter_number INTEGER,
                timestamp_in_story TEXT DEFAULT '',
                consequences TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """
        )

        # Event participants (many-to-many)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_participants (
                event_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (event_id, entity_id),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
        """
        )

        # Indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")

        self.conn.commit()
        logger.debug(f"Database schema initialized: {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Entity CRUD Operations
    # =========================================================================

    def add_entity(
        self,
        entity_type: str,
        name: str,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a new entity to the database.

        Args:
            entity_type: Type of entity (character, location, item, faction, concept)
            name: Entity name
            description: Entity description
            attributes: Additional attributes as key-value pairs

        Returns:
            Entity ID

        Raises:
            ValueError: If name or entity_type is invalid
        """
        # Validate inputs
        name = name.strip()
        if not name:
            raise ValueError("Entity name cannot be empty")
        if len(name) > 200:
            raise ValueError("Entity name cannot exceed 200 characters")

        entity_type = entity_type.strip()
        if not entity_type:
            raise ValueError("Entity type cannot be empty")

        # Strip whitespace from description for consistency
        description = description.strip() if description else ""
        if len(description) > 5000:
            raise ValueError("Entity description cannot exceed 5000 characters")

        entity_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        attrs_json = json.dumps(attributes or {})

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO entities (id, type, name, description, attributes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_id, entity_type, name, description, attrs_json, now, now),
        )
        self.conn.commit()
        self._invalidate_graph()
        logger.debug(f"Added entity: {name} ({entity_type}) id={entity_id}")
        return entity_id

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def get_entity_by_name(self, name: str, entity_type: str | None = None) -> Entity | None:
        """Get an entity by name (case-insensitive).

        Args:
            name: Entity name
            entity_type: Optional type filter

        Returns:
            Entity or None if not found
        """
        cursor = self.conn.cursor()
        if entity_type:
            cursor.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(?) AND type = ?",
                (name, entity_type),
            )
        else:
            cursor.execute("SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def update_entity(self, entity_id: str, **updates: Any) -> bool:
        """Update an entity.

        Args:
            entity_id: Entity ID
            **updates: Fields to update (name, description, attributes, type)

        Returns:
            True if updated, False if entity not found

        Raises:
            ValueError: If validation fails
        """
        allowed_fields = {"name", "description", "attributes", "type"}
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if not update_fields:
            return False

        # Validate name if being updated
        if "name" in update_fields:
            name = update_fields["name"].strip()
            if not name:
                raise ValueError("Entity name cannot be empty")
            if len(name) > 200:
                raise ValueError("Entity name cannot exceed 200 characters")
            update_fields["name"] = name

        # Validate description if being updated
        if "description" in update_fields:
            description = update_fields["description"]
            # Strip whitespace from description for consistency
            if description is not None:
                description = str(description).strip()
                if len(description) > 5000:
                    raise ValueError("Entity description cannot exceed 5000 characters")
                update_fields["description"] = description

        # Validate type if being updated
        if "type" in update_fields:
            entity_type = update_fields["type"].strip()
            if not entity_type:
                raise ValueError("Entity type cannot be empty")
            update_fields["type"] = entity_type

        # Handle attributes specially
        if "attributes" in update_fields:
            update_fields["attributes"] = json.dumps(update_fields["attributes"])

        update_fields["updated_at"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in update_fields.keys())
        values = list(update_fields.values()) + [entity_id]

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE entities SET {set_clause} WHERE id = ?",
            values,
        )
        self.conn.commit()
        self._invalidate_graph()
        return cursor.rowcount > 0

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            entity_id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self.conn.commit()
        self._invalidate_graph()
        return cursor.rowcount > 0

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        """List all entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter

        Returns:
            List of entities
        """
        cursor = self.conn.cursor()
        if entity_type:
            cursor.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY name",
                (entity_type,),
            )
        else:
            cursor.execute("SELECT * FROM entities ORDER BY type, name")
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def count_entities(self, entity_type: str | None = None) -> int:
        """Count entities, optionally by type.

        Args:
            entity_type: Optional type filter

        Returns:
            Count of entities
        """
        cursor = self.conn.cursor()
        if entity_type:
            cursor.execute("SELECT COUNT(*) FROM entities WHERE type = ?", (entity_type,))
        else:
            cursor.execute("SELECT COUNT(*) FROM entities")
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def search_entities(self, query: str, entity_type: str | None = None) -> list[Entity]:
        """Search entities by name or description.

        Args:
            query: Search query
            entity_type: Optional type filter

        Returns:
            List of matching entities
        """
        cursor = self.conn.cursor()
        search_pattern = f"%{query}%"
        if entity_type:
            cursor.execute(
                """
                SELECT * FROM entities
                WHERE (name LIKE ? OR description LIKE ?) AND type = ?
                ORDER BY name
                """,
                (search_pattern, search_pattern, entity_type),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM entities
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY type, name
                """,
                (search_pattern, search_pattern),
            )
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert a database row to an Entity."""
        return Entity(
            id=row["id"],
            type=row["type"],
            name=row["name"],
            description=row["description"],
            attributes=json.loads(row["attributes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # =========================================================================
    # Relationship CRUD Operations
    # =========================================================================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        strength: float = 1.0,
        bidirectional: bool = False,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            description: Relationship description
            strength: Relationship strength (0.0-1.0)
            bidirectional: Whether relationship goes both ways
            attributes: Additional attributes

        Returns:
            Relationship ID
        """
        rel_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        attrs_json = json.dumps(attributes or {})

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO relationships
            (id, source_id, target_id, relation_type, description, strength, bidirectional, attributes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel_id,
                source_id,
                target_id,
                relation_type,
                description,
                strength,
                1 if bidirectional else 0,
                attrs_json,
                now,
            ),
        )
        self.conn.commit()
        self._invalidate_graph()
        logger.debug(f"Added relationship: {source_id} --{relation_type}--> {target_id}")
        return rel_id

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        cursor = self.conn.cursor()

        if direction == "outgoing":
            cursor.execute("SELECT * FROM relationships WHERE source_id = ?", (entity_id,))
        elif direction == "incoming":
            cursor.execute("SELECT * FROM relationships WHERE target_id = ?", (entity_id,))
        else:
            cursor.execute(
                "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )

        return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def get_relationship_between(self, source_id: str, target_id: str) -> Relationship | None:
        """Get relationship between two specific entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            Relationship or None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM relationships
            WHERE (source_id = ? AND target_id = ?)
               OR (bidirectional = 1 AND source_id = ? AND target_id = ?)
            """,
            (source_id, target_id, target_id, source_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relationship(row)

    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship.

        Args:
            rel_id: Relationship ID

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM relationships WHERE id = ?", (rel_id,))
        self.conn.commit()
        self._invalidate_graph()
        return cursor.rowcount > 0

    def update_relationship(
        self,
        relationship_id: str,
        relation_type: str | None = None,
        description: str | None = None,
        strength: float | None = None,
        bidirectional: bool | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing relationship.

        Args:
            relationship_id: Relationship ID to update.
            relation_type: New relationship type.
            description: New description.
            strength: New strength value.
            bidirectional: New bidirectional flag.
            attributes: New attributes (merged with existing).

        Returns:
            True if updated, False if not found.
        """
        cursor = self.conn.cursor()

        # Get current relationship
        cursor.execute("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
        row = cursor.fetchone()
        if row is None:
            return False

        # Prepare updated values
        new_type = relation_type if relation_type is not None else row["relation_type"]
        new_desc = description if description is not None else row["description"]
        new_strength = strength if strength is not None else row["strength"]
        new_bidir = bidirectional if bidirectional is not None else row["bidirectional"]

        # Handle attributes merging
        current_attrs = json.loads(row["attributes"]) if row["attributes"] else {}
        if attributes is not None:
            current_attrs.update(attributes)

        cursor.execute(
            """
            UPDATE relationships
            SET relation_type = ?,
                description = ?,
                strength = ?,
                bidirectional = ?,
                attributes = ?
            WHERE id = ?
            """,
            (
                new_type,
                new_desc,
                new_strength,
                1 if new_bidir else 0,
                json.dumps(current_attrs),
                relationship_id,
            ),
        )
        self.conn.commit()
        self._invalidate_graph()
        return cursor.rowcount > 0

    def list_relationships(self) -> list[Relationship]:
        """List all relationships.

        Returns:
            List of all relationships
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM relationships ORDER BY created_at")
        return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert a database row to a Relationship."""
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            description=row["description"],
            strength=row["strength"],
            bidirectional=bool(row["bidirectional"]),
            attributes=json.loads(row["attributes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # =========================================================================
    # Event Management
    # =========================================================================

    def add_event(
        self,
        description: str,
        participants: list[tuple[str, str]] | None = None,
        chapter_number: int | None = None,
        timestamp_in_story: str = "",
        consequences: list[str] | None = None,
    ) -> str:
        """Add an event.

        Args:
            description: Event description
            participants: List of (entity_id, role) tuples
            chapter_number: Chapter where event occurs
            timestamp_in_story: In-world timing
            consequences: List of consequences

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        consequences_json = json.dumps(consequences or [])

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (id, description, chapter_number, timestamp_in_story, consequences, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, description, chapter_number, timestamp_in_story, consequences_json, now),
        )

        # Add participants
        if participants:
            for entity_id, role in participants:
                cursor.execute(
                    "INSERT INTO event_participants (event_id, entity_id, role) VALUES (?, ?, ?)",
                    (event_id, entity_id, role),
                )

        self.conn.commit()
        logger.debug(f"Added event: {description[:50]}...")
        return event_id

    def get_events_for_entity(self, entity_id: str) -> list[WorldEvent]:
        """Get all events involving an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of events
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT e.* FROM events e
            JOIN event_participants ep ON e.id = ep.event_id
            WHERE ep.entity_id = ?
            ORDER BY e.chapter_number, e.created_at
            """,
            (entity_id,),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_events_for_chapter(self, chapter_number: int) -> list[WorldEvent]:
        """Get all events for a chapter.

        Args:
            chapter_number: Chapter number

        Returns:
            List of events
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE chapter_number = ? ORDER BY created_at",
            (chapter_number,),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_event_participants(self, event_id: str) -> list[EventParticipant]:
        """Get participants for an event.

        Args:
            event_id: Event ID

        Returns:
            List of event participants
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM event_participants WHERE event_id = ?", (event_id,))
        return [
            EventParticipant(
                event_id=row["event_id"],
                entity_id=row["entity_id"],
                role=row["role"],
            )
            for row in cursor.fetchall()
        ]

    def list_events(self, limit: int | None = None) -> list[WorldEvent]:
        """List all events.

        Args:
            limit: Optional limit on number of events

        Returns:
            List of events
        """
        cursor = self.conn.cursor()
        if limit:
            cursor.execute(
                "SELECT * FROM events ORDER BY chapter_number, created_at LIMIT ?",
                (limit,),
            )
        else:
            cursor.execute("SELECT * FROM events ORDER BY chapter_number, created_at")
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row: sqlite3.Row) -> WorldEvent:
        """Convert a database row to a WorldEvent."""
        return WorldEvent(
            id=row["id"],
            description=row["description"],
            chapter_number=row["chapter_number"],
            timestamp_in_story=row["timestamp_in_story"],
            consequences=json.loads(row["consequences"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # =========================================================================
    # NetworkX Graph Operations
    # =========================================================================

    def _invalidate_graph(self) -> None:
        """Invalidate cached graph."""
        self._graph = None

    def _rebuild_graph(self) -> None:
        """Rebuild NetworkX graph from database."""
        self._graph = nx.DiGraph()

        # Add nodes (entities)
        for entity in self.list_entities():
            self._graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                attributes=entity.attributes,
            )

        # Add edges (relationships)
        for rel in self.list_relationships():
            self._graph.add_edge(
                rel.source_id,
                rel.target_id,
                id=rel.id,
                relation_type=rel.relation_type,
                description=rel.description,
                strength=rel.strength,
            )
            # Add reverse edge for bidirectional relationships
            if rel.bidirectional:
                self._graph.add_edge(
                    rel.target_id,
                    rel.source_id,
                    id=rel.id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    strength=rel.strength,
                    is_reverse=True,
                )

        logger.debug(
            f"Graph rebuilt: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def get_graph(self) -> nx.DiGraph:
        """Get NetworkX graph (lazy-loaded).

        Returns:
            NetworkX directed graph
        """
        if self._graph is None:
            self._rebuild_graph()
        assert self._graph is not None  # Guaranteed by _rebuild_graph
        return self._graph

    def find_path(self, source_id: str, target_id: str) -> list[str]:
        """Find shortest path between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            List of entity IDs forming the path, empty if no path exists
        """
        graph = self.get_graph()
        try:
            path: list[str] = nx.shortest_path(graph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_all_paths(
        self, source_id: str, target_id: str, max_length: int = 5
    ) -> list[list[str]]:
        """Find all paths between two entities up to a maximum length.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of entity IDs)
        """
        graph = self.get_graph()
        try:
            return list(nx.all_simple_paths(graph, source_id, target_id, cutoff=max_length))
        except nx.NodeNotFound:
            return []

    def get_connected_entities(self, entity_id: str, max_depth: int = 2) -> list[Entity]:
        """Get all entities connected to a given entity within a depth.

        Args:
            entity_id: Starting entity ID
            max_depth: Maximum depth to traverse

        Returns:
            List of connected entities
        """
        graph = self.get_graph()
        if entity_id not in graph:
            return []

        # Use BFS to find connected nodes
        connected_ids: set[str] = set()
        current_level: set[str] = {entity_id}

        for _ in range(max_depth):
            next_level: set[str] = set()
            for node in current_level:
                # Add both successors and predecessors
                next_level.update(graph.successors(node))
                next_level.update(graph.predecessors(node))
            next_level -= connected_ids
            next_level.discard(entity_id)
            connected_ids.update(next_level)
            current_level = next_level
            if not current_level:
                break

        # Filter out None results from get_entity
        entities: list[Entity] = []
        for eid in connected_ids:
            entity = self.get_entity(eid)
            if entity is not None:
                entities.append(entity)
        return entities

    def get_communities(self) -> list[list[str]]:
        """Find communities/clusters of entities.

        Returns:
            List of communities (each community is a list of entity IDs)
        """
        graph = self.get_graph()
        if graph.number_of_nodes() == 0:
            return []

        # Convert to undirected for community detection
        undirected = graph.to_undirected()

        # Use connected components as basic communities
        communities = list(nx.connected_components(undirected))
        return [list(c) for c in communities]

    def get_entity_centrality(self) -> dict[str, float]:
        """Calculate importance/centrality of each entity.

        Returns:
            Dict mapping entity ID to centrality score
        """
        graph = self.get_graph()
        if graph.number_of_nodes() == 0:
            return {}

        # Use degree centrality (normalized by number of nodes)
        centrality: dict[str, float] = nx.degree_centrality(graph)
        return centrality

    def get_most_connected(self, limit: int = 10) -> list[tuple[Entity, int]]:
        """Get most connected entities.

        Args:
            limit: Number of entities to return

        Returns:
            List of (entity, connection_count) tuples
        """
        graph = self.get_graph()
        if graph.number_of_nodes() == 0:
            return []

        # Count both incoming and outgoing edges
        degrees = [(node, graph.degree(node)) for node in graph.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)

        result = []
        for node_id, degree in degrees[:limit]:
            entity = self.get_entity(node_id)
            if entity:
                result.append((entity, degree))
        return result

    # =========================================================================
    # Context for Agents
    # =========================================================================

    def get_context_for_agents(self, max_entities: int = 50) -> dict[str, Any]:
        """Get compressed world context for AI agents.

        Args:
            max_entities: Maximum entities to include per type

        Returns:
            Dict with world context
        """
        return {
            "characters": [
                {"name": e.name, "description": e.description, "attributes": e.attributes}
                for e in self.list_entities("character")[:20]
            ],
            "locations": [
                {"name": e.name, "description": e.description, "attributes": e.attributes}
                for e in self.list_entities("location")[:15]
            ],
            "items": [
                {"name": e.name, "description": e.description}
                for e in self.list_entities("item")[:10]
            ],
            "factions": [
                {"name": e.name, "description": e.description}
                for e in self.list_entities("faction")[:10]
            ],
            "key_relationships": self._get_important_relationships(limit=30),
            "recent_events": [
                {"description": e.description, "chapter": e.chapter_number}
                for e in self.list_events(limit=20)
            ],
            "entity_counts": {
                "characters": self.count_entities("character"),
                "locations": self.count_entities("location"),
                "items": self.count_entities("item"),
                "factions": self.count_entities("faction"),
                "concepts": self.count_entities("concept"),
            },
        }

    def _get_important_relationships(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get most important relationships for context.

        Args:
            limit: Maximum relationships to return

        Returns:
            List of relationship dicts
        """
        relationships = self.list_relationships()
        # Sort by strength descending
        relationships.sort(key=lambda r: r.strength, reverse=True)

        result = []
        for rel in relationships[:limit]:
            source = self.get_entity(rel.source_id)
            target = self.get_entity(rel.target_id)
            if source and target:
                result.append(
                    {
                        "from": source.name,
                        "to": target.name,
                        "type": rel.relation_type,
                        "description": rel.description,
                    }
                )
        return result

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_to_json(self) -> dict[str, Any]:
        """Export entire database to JSON.

        Returns:
            Dict with all data
        """
        return {
            "entities": [e.model_dump(mode="json") for e in self.list_entities()],
            "relationships": [r.model_dump(mode="json") for r in self.list_relationships()],
            "events": [
                {
                    **e.model_dump(mode="json"),
                    "participants": [
                        {"entity_id": p.entity_id, "role": p.role}
                        for p in self.get_event_participants(e.id)
                    ],
                }
                for e in self.list_events()
            ],
        }

    def import_from_json(self, data: dict[str, Any]) -> None:
        """Import data from JSON export.

        Args:
            data: Previously exported data
        """
        cursor = self.conn.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM event_participants")
        cursor.execute("DELETE FROM events")
        cursor.execute("DELETE FROM relationships")
        cursor.execute("DELETE FROM entities")

        # Import entities
        for entity_data in data.get("entities", []):
            cursor.execute(
                """
                INSERT INTO entities (id, type, name, description, attributes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_data["id"],
                    entity_data["type"],
                    entity_data["name"],
                    entity_data.get("description", ""),
                    json.dumps(entity_data.get("attributes", {})),
                    entity_data.get("created_at", datetime.now().isoformat()),
                    entity_data.get("updated_at", datetime.now().isoformat()),
                ),
            )

        # Import relationships
        for rel_data in data.get("relationships", []):
            cursor.execute(
                """
                INSERT INTO relationships
                (id, source_id, target_id, relation_type, description, strength, bidirectional, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_data["id"],
                    rel_data["source_id"],
                    rel_data["target_id"],
                    rel_data["relation_type"],
                    rel_data.get("description", ""),
                    rel_data.get("strength", 1.0),
                    1 if rel_data.get("bidirectional", False) else 0,
                    json.dumps(rel_data.get("attributes", {})),
                    rel_data.get("created_at", datetime.now().isoformat()),
                ),
            )

        # Import events
        for event_data in data.get("events", []):
            cursor.execute(
                """
                INSERT INTO events (id, description, chapter_number, timestamp_in_story, consequences, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_data["id"],
                    event_data["description"],
                    event_data.get("chapter_number"),
                    event_data.get("timestamp_in_story", ""),
                    json.dumps(event_data.get("consequences", [])),
                    event_data.get("created_at", datetime.now().isoformat()),
                ),
            )

            # Import participants
            for participant in event_data.get("participants", []):
                cursor.execute(
                    "INSERT INTO event_participants (event_id, entity_id, role) VALUES (?, ?, ?)",
                    (event_data["id"], participant["entity_id"], participant["role"]),
                )

        self.conn.commit()
        self._invalidate_graph()
        logger.info(
            f"Imported {len(data.get('entities', []))} entities, "
            f"{len(data.get('relationships', []))} relationships, "
            f"{len(data.get('events', []))} events"
        )
