"""Conflict analysis service - analyzes relationships for tensions and alliances.

This service handles:
- Relationship classification into conflict categories
- Tension and alliance metrics calculation
- Faction clustering based on alliances
- Conflict graph data generation for visualization
"""

import logging
from typing import TYPE_CHECKING

from src.memory.conflict_types import (
    ConflictCategory,
    ConflictGraphData,
    ConflictGraphEdge,
    ConflictGraphNode,
    ConflictMetrics,
    FactionCluster,
    TensionPair,
    classify_relationship,
    get_conflict_color,
)
from src.memory.entities import Entity, Relationship
from src.settings import Settings
from src.utils.constants import get_entity_color
from src.utils.validation import validate_not_none

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase

logger = logging.getLogger(__name__)


class ConflictAnalysisService:
    """Service for conflict and tension analysis.

    This service coordinates:
    - Relationship classification by conflict category
    - Tension metrics calculation
    - Faction/alliance clustering
    - Graph data generation for visualization
    """

    def __init__(self, settings: Settings):
        """Initialize conflict analysis service.

        Args:
            settings: Application settings.
        """
        logger.debug("Initializing ConflictAnalysisService")
        self.settings = settings
        logger.debug("ConflictAnalysisService initialized successfully")

    def analyze_conflicts(self, world_db: WorldDatabase) -> ConflictMetrics:
        """Analyze all relationships and compute conflict metrics.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            ConflictMetrics with all analysis results.
        """
        validate_not_none(world_db, "world_db")
        logger.debug("Starting conflict analysis")

        entities = world_db.list_entities()
        relationships = world_db.list_relationships()

        # Build entity lookup
        entity_lookup = {e.id: e for e in entities}

        # Classify all relationships
        classified: dict[str, list[Relationship]] = {
            ConflictCategory.ALLIANCE.value: [],
            ConflictCategory.RIVALRY.value: [],
            ConflictCategory.TENSION.value: [],
            ConflictCategory.NEUTRAL.value: [],
        }

        for rel in relationships:
            category = classify_relationship(rel.relation_type)
            classified[category.value].append(rel)

        # Calculate counts
        alliance_count = len(classified[ConflictCategory.ALLIANCE.value])
        rivalry_count = len(classified[ConflictCategory.RIVALRY.value])
        tension_count = len(classified[ConflictCategory.TENSION.value])
        neutral_count = len(classified[ConflictCategory.NEUTRAL.value])
        total = len(relationships)

        # Calculate conflict density
        conflict_density = 0.0
        if total > 0:
            conflict_density = (rivalry_count + tension_count) / total

        # Find highest tension pairs
        highest_tension = self._find_tension_pairs(
            classified[ConflictCategory.RIVALRY.value] + classified[ConflictCategory.TENSION.value],
            entity_lookup,
            limit=5,
        )

        # Find strongest alliances
        strongest_alliances = self._find_tension_pairs(
            classified[ConflictCategory.ALLIANCE.value],
            entity_lookup,
            limit=5,
        )

        # Find isolated entities (no alliance connections)
        isolated = self._find_isolated_entities(
            entities,
            classified[ConflictCategory.ALLIANCE.value],
        )

        # Find most connected entities
        most_connected = self._find_most_connected(entities, relationships, limit=5)

        # Detect faction clusters (based on alliance relationships)
        faction_clusters = self._detect_faction_clusters(
            entities,
            classified[ConflictCategory.ALLIANCE.value],
            entity_lookup,
        )

        metrics = ConflictMetrics(
            highest_tension_pairs=highest_tension,
            strongest_alliances=strongest_alliances,
            isolated_entities=isolated,
            most_connected_entities=most_connected,
            faction_clusters=faction_clusters,
            total_relationships=total,
            alliance_count=alliance_count,
            rivalry_count=rivalry_count,
            tension_count=tension_count,
            neutral_count=neutral_count,
            conflict_density=conflict_density,
        )

        logger.info(
            f"Conflict analysis complete: {total} relationships, "
            f"density={conflict_density:.2f}, {len(faction_clusters)} factions"
        )
        return metrics

    def _find_tension_pairs(
        self,
        relationships: list[Relationship],
        entity_lookup: dict[str, Entity],
        limit: int = 5,
    ) -> list[TensionPair]:
        """Find pairs of entities with the strongest tension/alliance.

        Args:
            relationships: List of relationships to analyze.
            entity_lookup: Entity ID to Entity mapping.
            limit: Maximum number of pairs to return.

        Returns:
            List of TensionPair objects sorted by score.
        """
        # Group relationships by entity pair
        pair_rels: dict[tuple[str, str], list[Relationship]] = {}
        for rel in relationships:
            # Normalize pair order for grouping
            sorted_ids = sorted([rel.source_id, rel.target_id])
            pair: tuple[str, str] = (sorted_ids[0], sorted_ids[1])
            if pair not in pair_rels:
                pair_rels[pair] = []
            pair_rels[pair].append(rel)

        # Calculate scores for each pair
        pairs: list[TensionPair] = []
        for (id_a, id_b), rels in pair_rels.items():
            entity_a = entity_lookup.get(id_a)
            entity_b = entity_lookup.get(id_b)

            if not entity_a or not entity_b:
                continue

            # Score based on number and strength of relationships
            total_strength = sum(rel.strength for rel in rels)
            score = min(1.0, total_strength / len(rels)) if rels else 0.0

            pairs.append(
                TensionPair(
                    entity_a_id=id_a,
                    entity_a_name=entity_a.name,
                    entity_b_id=id_b,
                    entity_b_name=entity_b.name,
                    score=score,
                    relationship_types=[rel.relation_type for rel in rels],
                )
            )

        # Sort by score and return top N
        pairs.sort(key=lambda p: p.score, reverse=True)
        return pairs[:limit]

    def _find_isolated_entities(
        self,
        entities: list[Entity],
        alliance_rels: list[Relationship],
    ) -> list[str]:
        """Find entities with no alliance connections.

        Args:
            entities: All entities.
            alliance_rels: Alliance relationships.

        Returns:
            List of isolated entity IDs.
        """
        # Get all entities involved in alliances
        allied_ids = set()
        for rel in alliance_rels:
            allied_ids.add(rel.source_id)
            allied_ids.add(rel.target_id)

        # Find entities not in any alliance
        isolated = []
        for entity in entities:
            # Only consider characters and factions for isolation
            if entity.type in ["character", "faction"] and entity.id not in allied_ids:
                isolated.append(entity.id)

        return isolated

    def _find_most_connected(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        limit: int = 5,
    ) -> list[tuple[str, int]]:
        """Find entities with the most relationship connections.

        Args:
            entities: All entities.
            relationships: All relationships.
            limit: Maximum number to return.

        Returns:
            List of (entity_id, connection_count) tuples.
        """
        # Count connections per entity
        connection_counts: dict[str, int] = {}
        for rel in relationships:
            connection_counts[rel.source_id] = connection_counts.get(rel.source_id, 0) + 1
            connection_counts[rel.target_id] = connection_counts.get(rel.target_id, 0) + 1

        # Sort and return top N
        sorted_counts = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:limit]

    def _detect_faction_clusters(
        self,
        entities: list[Entity],
        alliance_rels: list[Relationship],
        entity_lookup: dict[str, Entity],
    ) -> list[FactionCluster]:
        """Detect clusters of allied entities using union-find.

        Args:
            entities: All entities.
            alliance_rels: Alliance relationships.
            entity_lookup: Entity ID to Entity mapping.

        Returns:
            List of FactionCluster objects.
        """
        if not alliance_rels:
            return []

        # Union-Find implementation
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union entities connected by alliances
        for rel in alliance_rels:
            union(rel.source_id, rel.target_id)

        # Group entities by their root
        clusters_dict: dict[str, list[str]] = {}
        for rel in alliance_rels:
            for entity_id in [rel.source_id, rel.target_id]:
                root = find(entity_id)
                if root not in clusters_dict:
                    clusters_dict[root] = []
                if entity_id not in clusters_dict[root]:
                    clusters_dict[root].append(entity_id)

        # Convert to FactionCluster objects
        clusters: list[FactionCluster] = []
        for i, (_, entity_ids) in enumerate(clusters_dict.items()):
            if len(entity_ids) < 2:
                continue  # Skip single-entity "clusters"

            entity_names = [entity_lookup[eid].name for eid in entity_ids if eid in entity_lookup]

            # Calculate internal alliance strength
            internal_strength = 0.0
            internal_count = 0
            for rel in alliance_rels:
                if rel.source_id in entity_ids and rel.target_id in entity_ids:
                    internal_strength += rel.strength
                    internal_count += 1

            avg_strength = internal_strength / internal_count if internal_count > 0 else 0.0

            clusters.append(
                FactionCluster(
                    id=f"cluster-{i + 1}",
                    entity_ids=entity_ids,
                    entity_names=entity_names,
                    internal_alliance_strength=avg_strength,
                )
            )

        # Sort by size
        clusters.sort(key=lambda c: len(c.entity_ids), reverse=True)
        return clusters

    def get_conflict_graph_data(
        self,
        world_db: WorldDatabase,
        categories: list[ConflictCategory] | None = None,
        entity_types: list[str] | None = None,
    ) -> ConflictGraphData:
        """Get graph data for conflict visualization.

        Args:
            world_db: WorldDatabase instance.
            categories: Optional filter for conflict categories.
            entity_types: Optional filter for entity types.

        Returns:
            ConflictGraphData with nodes, edges, and metrics.
        """
        validate_not_none(world_db, "world_db")
        logger.debug(f"get_conflict_graph_data: categories={categories}, types={entity_types}")

        entities = world_db.list_entities()
        relationships = world_db.list_relationships()

        # Filter entities by type if specified
        if entity_types:
            entities = [e for e in entities if e.type in entity_types]

        entity_ids = {e.id for e in entities}

        # Build nodes
        nodes: list[ConflictGraphNode] = []
        connection_counts: dict[str, int] = {}

        for rel in relationships:
            if rel.source_id in entity_ids:
                connection_counts[rel.source_id] = connection_counts.get(rel.source_id, 0) + 1
            if rel.target_id in entity_ids:
                connection_counts[rel.target_id] = connection_counts.get(rel.target_id, 0) + 1

        for entity in entities:
            node = ConflictGraphNode(
                id=entity.id,
                label=entity.name,
                entity_type=entity.type,
                color=get_entity_color(entity.type),
                size=10 + min(20, connection_counts.get(entity.id, 0) * 2),
                title=f"{entity.name}\n{entity.description[:100]}..."
                if entity.description
                else entity.name,
            )
            nodes.append(node)

        # Build edges (filter by category if specified)
        edges: list[ConflictGraphEdge] = []
        for rel in relationships:
            # Only include edges where both endpoints are in our entity set
            if rel.source_id not in entity_ids or rel.target_id not in entity_ids:
                continue

            category = classify_relationship(rel.relation_type)

            # Filter by category if specified
            if categories and category not in categories:
                continue

            color = get_conflict_color(category)

            # Width based on strength
            width = max(1, int(rel.strength * 3))

            # Dashed line for tension (potential conflict)
            dashes = category == ConflictCategory.TENSION

            edge = ConflictGraphEdge(
                from_id=rel.source_id,
                to_id=rel.target_id,
                relation_type=rel.relation_type,
                category=category,
                color=color,
                width=width,
                title=f"{rel.relation_type}: {rel.description}"
                if rel.description
                else rel.relation_type,
                dashes=dashes,
            )
            edges.append(edge)

        # Get metrics
        metrics = self.analyze_conflicts(world_db)

        graph_data = ConflictGraphData(
            nodes=nodes,
            edges=edges,
            metrics=metrics,
        )

        logger.info(f"Generated conflict graph: {len(nodes)} nodes, {len(edges)} edges")
        return graph_data

    def get_category_summary(self, world_db: WorldDatabase) -> dict[str, int]:
        """Get a summary count of relationships by category.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            Dictionary mapping category name to count.
        """
        validate_not_none(world_db, "world_db")

        relationships = world_db.list_relationships()

        summary = {
            ConflictCategory.ALLIANCE.value: 0,
            ConflictCategory.RIVALRY.value: 0,
            ConflictCategory.TENSION.value: 0,
            ConflictCategory.NEUTRAL.value: 0,
        }

        for rel in relationships:
            category = classify_relationship(rel.relation_type)
            summary[category.value] += 1

        logger.debug(f"Category summary: {summary}")
        return summary

    def suggest_missing_conflicts(
        self,
        world_db: WorldDatabase,
        limit: int = 5,
    ) -> list[dict]:
        """Suggest potential conflicts that might be missing.

        Identifies pairs of entities that have opposing goals/values
        but no conflict relationship.

        Args:
            world_db: WorldDatabase instance.
            limit: Maximum suggestions to return.

        Returns:
            List of suggestion dictionaries with entity pairs and reasons.
        """
        validate_not_none(world_db, "world_db")
        logger.debug(f"suggest_missing_conflicts: limit={limit}")

        entities = world_db.list_entities()
        relationships = world_db.list_relationships()

        # Get existing relationship pairs
        existing_pairs = set()
        for rel in relationships:
            pair = tuple(sorted([rel.source_id, rel.target_id]))
            existing_pairs.add(pair)

        suggestions: list[dict] = []

        # Look at factions - opposing goals suggest conflict
        factions = [e for e in entities if e.type == "faction"]
        for i, faction_a in enumerate(factions):
            for faction_b in factions[i + 1 :]:
                pair = tuple(sorted([faction_a.id, faction_b.id]))
                if pair in existing_pairs:
                    continue

                # Check for opposing goals
                goals_a = faction_a.attributes.get("goals", [])
                goals_b = faction_b.attributes.get("goals", [])

                # Simple heuristic: different goals might indicate conflict
                if goals_a and goals_b:
                    suggestions.append(
                        {
                            "entity_a_id": faction_a.id,
                            "entity_a_name": faction_a.name,
                            "entity_b_id": faction_b.id,
                            "entity_b_name": faction_b.name,
                            "reason": "Both are factions with defined goals - consider their relationship",
                            "suggested_types": ["ally_of", "enemy_of", "competes_with"],
                        }
                    )

                if len(suggestions) >= limit:
                    break
            if len(suggestions) >= limit:
                break

        logger.info(f"Generated {len(suggestions)} conflict suggestions")
        return suggestions
