"""NetworkX graph operations mixin for WorldDatabase."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx import DiGraph

from src.memory.entities import Entity

if TYPE_CHECKING:
    from ._base import WorldDatabaseBase

    _GraphMixinBase = WorldDatabaseBase
else:
    _GraphMixinBase = object

logger = logging.getLogger(__name__)


class GraphMixin(_GraphMixinBase):
    """Mixin providing NetworkX graph operations."""

    def _invalidate_graph(self) -> None:
        """Invalidate cached graph."""
        self._graph = None

    def invalidate_graph_cache(self) -> None:
        """Force graph rebuild on next access.

        Call this after updating entity attributes (like mini_description)
        to ensure the graph reflects the latest data for tooltips.
        """
        self._graph = None
        logger.debug("Graph cache invalidated - will rebuild on next access")

    def _add_entity_to_graph(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        attributes: dict[str, Any],
    ) -> None:
        """Add entity to graph incrementally (no full rebuild).

        Args:
            entity_id: Entity ID
            name: Entity name
            entity_type: Entity type
            description: Entity description
            attributes: Entity attributes
        """
        if self._graph is None:
            return  # Graph not built yet, will be built on next get_graph()

        self._graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            description=description,
            attributes=attributes,
        )
        logger.debug(f"Added entity to graph: {name} ({entity_id})")

    def _remove_entity_from_graph(self, entity_id: str) -> None:
        """Remove entity from graph incrementally (no full rebuild).

        Args:
            entity_id: Entity ID to remove
        """
        if self._graph is None:
            return  # Graph not built yet

        if entity_id in self._graph:
            self._graph.remove_node(entity_id)
            logger.debug(f"Removed entity from graph: {entity_id}")

    def _update_entity_in_graph(
        self,
        entity_id: str,
        name: str | None = None,
        entity_type: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Update entity in graph incrementally (no full rebuild).

        Args:
            entity_id: Entity ID
            name: New name (if changed)
            entity_type: New type (if changed)
            description: New description (if changed)
            attributes: New attributes (if changed)
        """
        if self._graph is None:
            return  # Graph not built yet

        if entity_id not in self._graph:
            return  # Entity not in graph

        # Update only provided fields
        if name is not None:
            self._graph.nodes[entity_id]["name"] = name
        if entity_type is not None:
            self._graph.nodes[entity_id]["type"] = entity_type
        if description is not None:
            self._graph.nodes[entity_id]["description"] = description
        if attributes is not None:
            self._graph.nodes[entity_id]["attributes"] = attributes

        logger.debug(f"Updated entity in graph: {entity_id}")

    def _add_relationship_to_graph(
        self,
        rel_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str,
        strength: float,
        bidirectional: bool,
    ) -> None:
        """Add relationship to graph incrementally (no full rebuild).

        Args:
            rel_id: Relationship ID
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Relationship type
            description: Relationship description
            strength: Relationship strength
            bidirectional: Whether relationship is bidirectional
        """
        if self._graph is None:
            return  # Graph not built yet

        self._graph.add_edge(
            source_id,
            target_id,
            id=rel_id,
            relation_type=relation_type,
            description=description,
            strength=strength,
        )

        if bidirectional:
            self._graph.add_edge(
                target_id,
                source_id,
                id=rel_id,
                relation_type=relation_type,
                description=description,
                strength=strength,
                is_reverse=True,
            )
        logger.debug(f"Added relationship to graph: {source_id} -> {target_id}")

    def _remove_relationship_from_graph(
        self, source_id: str, target_id: str, bidirectional: bool = False
    ) -> None:
        """Remove relationship from graph incrementally (no full rebuild).

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            bidirectional: Whether to also remove reverse edge
        """
        if self._graph is None:
            return  # Graph not built yet

        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)

        if bidirectional and self._graph.has_edge(target_id, source_id):
            self._graph.remove_edge(target_id, source_id)

        logger.debug(f"Removed relationship from graph: {source_id} -> {target_id}")

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

    def get_graph(self) -> DiGraph[Any]:
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
        logger.debug(
            "find_all_paths called: source=%s, target=%s, max_length=%d",
            source_id,
            target_id,
            max_length,
        )
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
        logger.debug(
            "get_connected_entities called: entity_id=%s, max_depth=%d", entity_id, max_depth
        )
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

    def find_orphans(self, entity_type: str | None = None) -> list[Entity]:
        """Find entities with no relationships (orphans).

        An orphan entity has no incoming or outgoing relationships.

        Args:
            entity_type: Optional type filter (e.g., "character", "location").

        Returns:
            List of orphan entities.
        """
        logger.debug(f"find_orphans called: entity_type={entity_type}")

        with self._lock:
            cursor = self.conn.cursor()

            # Query for entities that are not in any relationship
            if entity_type:
                cursor.execute(
                    """
                    SELECT * FROM entities
                    WHERE type = ?
                    AND id NOT IN (
                        SELECT source_id FROM relationships
                        UNION
                        SELECT target_id FROM relationships
                    )
                    ORDER BY name
                    """,
                    (entity_type,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM entities
                    WHERE id NOT IN (
                        SELECT source_id FROM relationships
                        UNION
                        SELECT target_id FROM relationships
                    )
                    ORDER BY type, name
                    """
                )

            orphans = [self._row_to_entity(row) for row in cursor.fetchall()]
            logger.debug(f"Found {len(orphans)} orphan entities")
            return orphans

    def find_circular_relationships(
        self,
        relation_types: list[str] | None = None,
        max_cycle_length: int = 10,
    ) -> list[list[tuple[str, str, str]]]:
        """Find circular relationships (cycles) in the entity graph.

        Uses NetworkX's simple_cycles algorithm to detect cycles.

        Args:
            relation_types: Optional list of relationship types to check.
                If None, checks all relationships.
            max_cycle_length: Maximum cycle length to detect (default 10).
                Longer cycles are ignored for performance.

        Returns:
            List of cycles. Each cycle is a list of (source_id, relation_type, target_id)
            tuples representing the edges forming the cycle.
        """
        logger.debug(
            f"find_circular_relationships called: relation_types={relation_types}, "
            f"max_cycle_length={max_cycle_length}"
        )

        graph = self.get_graph()
        if graph.number_of_nodes() == 0:
            return []

        # If relation_types specified, create a filtered subgraph
        subgraph: nx.DiGraph
        if relation_types:
            # Create subgraph with only edges of specified types
            edges_to_keep = [
                (u, v, data)
                for u, v, data in graph.edges(data=True)
                if data.get("relation_type") in relation_types
            ]
            if not edges_to_keep:
                logger.debug("No edges match specified relation_types")
                return []

            subgraph = nx.DiGraph()
            subgraph.add_edges_from(edges_to_keep)
        else:
            subgraph = graph

        # Find simple cycles using NetworkX
        try:
            cycles_raw = list(nx.simple_cycles(subgraph, length_bound=max_cycle_length))
        except Exception as e:
            logger.warning(f"Error finding cycles: {e}")
            return []

        # Convert cycles from node lists to edge tuples with relation types
        cycles: list[list[tuple[str, str, str]]] = []
        for cycle_nodes in cycles_raw:
            if len(cycle_nodes) < 2:
                continue  # Skip degenerate cycles

            cycle_edges: list[tuple[str, str, str]] = []
            # Iterate through pairs of adjacent nodes in the cycle
            for i in range(len(cycle_nodes)):
                source = cycle_nodes[i]
                target = cycle_nodes[(i + 1) % len(cycle_nodes)]

                # Get edge data
                if subgraph.has_edge(source, target):
                    edge_data = subgraph.get_edge_data(source, target)
                    rel_type = edge_data.get("relation_type", "unknown") if edge_data else "unknown"
                    cycle_edges.append((source, rel_type, target))

            if cycle_edges:
                cycles.append(cycle_edges)

        logger.debug(f"Found {len(cycles)} circular relationship chains")
        return cycles
