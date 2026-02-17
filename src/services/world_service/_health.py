"""World health detection functions for WorldService."""

import logging
import math
from typing import TYPE_CHECKING

from src.memory.entities import Entity
from src.memory.world_database import WorldDatabase

if TYPE_CHECKING:
    from src.memory.world_health import WorldHealthMetrics
    from src.services.world_service import WorldService

logger = logging.getLogger(__name__)


def find_orphan_entities(
    svc: WorldService,
    world_db: WorldDatabase,
    entity_type: str | None = None,
) -> list[Entity]:
    """Find entities with no relationships (orphans).

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        entity_type: Optional type filter.

    Returns:
        List of orphan entities.
    """
    logger.debug(f"find_orphan_entities called: entity_type={entity_type}")
    orphans = world_db.find_orphans(entity_type=entity_type)
    logger.info(f"Found {len(orphans)} orphan entities")
    return orphans


def find_circular_relationships(
    svc: WorldService,
    world_db: WorldDatabase,
    relation_types: list[str] | None = None,
    max_cycle_length: int = 10,
) -> list[list[tuple[str, str, str]]]:
    """Find circular relationships (cycles) in the world.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        relation_types: Optional list of relationship types to check.
        max_cycle_length: Maximum cycle length to detect.

    Returns:
        List of cycles. Each cycle is a list of (source_id, relation_type, target_id)
        tuples.
    """
    logger.debug(
        f"find_circular_relationships called: relation_types={relation_types}, "
        f"max_length={max_cycle_length}"
    )
    cycles = world_db.find_circular_relationships(
        relation_types=relation_types,
        max_cycle_length=max_cycle_length,
    )
    logger.info(f"Found {len(cycles)} circular relationship chains")
    return cycles


def get_world_health_metrics(
    svc: WorldService,
    world_db: WorldDatabase,
    quality_threshold: float = 6.0,
) -> WorldHealthMetrics:
    """Get comprehensive health metrics for a story world.

    Aggregates entity counts, orphan detection, circular relationships,
    quality scores, and computes overall health score.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        quality_threshold: Minimum quality score for "healthy" entities.

    Returns:
        WorldHealthMetrics object with all computed metrics.
    """
    from src.memory.world_health import WorldHealthMetrics

    logger.info("Computing world health metrics")

    # Entity counts
    entity_counts = {
        "character": world_db.count_entities("character"),
        "location": world_db.count_entities("location"),
        "faction": world_db.count_entities("faction"),
        "item": world_db.count_entities("item"),
        "concept": world_db.count_entities("concept"),
    }
    total_entities = sum(entity_counts.values())
    relationships = world_db.list_relationships()
    total_relationships = len(relationships)

    # Orphan detection (respects settings toggle)
    orphan_entities: list[dict] = []
    if svc.settings.orphan_detection_enabled:
        orphans = world_db.find_orphans()
        orphan_entities = [{"id": e.id, "name": e.name, "type": e.type} for e in orphans]
        logger.debug(f"Orphan detection enabled: found {len(orphan_entities)} orphans")
    else:
        logger.debug("Orphan detection disabled by settings")

    # Quality metrics (moved before circular detection to enable name lookups)
    all_entities = world_db.list_entities()
    entity_name_lookup = {e.id: e.name for e in all_entities}

    # Circular relationship detection (respects settings toggle)
    circular_relationships: list[dict] = []
    hierarchical_circular_count = 0
    mutual_circular_count = 0
    if svc.settings.circular_detection_enabled:
        circular = world_db.find_circular_relationships(
            relation_types=None
            if svc.settings.circular_check_all_types
            else svc.settings.circular_relationship_types,
        )
        # Filter out accepted cycles
        accepted_hashes = world_db.get_accepted_cycles()
        if accepted_hashes:
            original_count = len(circular)
            circular = [
                c for c in circular if world_db.compute_cycle_hash(c) not in accepted_hashes
            ]
            filtered_count = original_count - len(circular)
            if filtered_count > 0:
                logger.debug(
                    "Filtered %d accepted cycles from %d total",
                    filtered_count,
                    original_count,
                )
        for cycle in circular:
            # Include entity names for human-readable display
            edges_with_names = []
            for e in cycle:
                source_id, relation_type, target_id = e[0], e[1], e[2]
                edges_with_names.append(
                    {
                        "source": source_id,
                        "source_name": entity_name_lookup.get(source_id, source_id),
                        "type": relation_type,
                        "target": target_id,
                        "target_name": entity_name_lookup.get(target_id, target_id),
                    }
                )
            cycle_info = {
                "edges": edges_with_names,
                "length": len(cycle),
            }
            circular_relationships.append(cycle_info)
        logger.debug(f"Circular detection enabled: found {len(circular_relationships)} cycles")

        # Classify cycles as hierarchical vs mutual
        from src.memory.conflict_types import HIERARCHICAL_RELATIONSHIP_TYPES

        for cycle_info in circular_relationships:
            edges: list[dict] = cycle_info.get("edges", [])  # type: ignore[assignment]
            is_hierarchical = any(
                edge.get("type", "") in HIERARCHICAL_RELATIONSHIP_TYPES for edge in edges
            )
            if is_hierarchical:
                hierarchical_circular_count += 1
            else:
                mutual_circular_count += 1
        logger.debug(
            "Circular classification: %d hierarchical, %d mutual",
            hierarchical_circular_count,
            mutual_circular_count,
        )
    else:
        logger.debug("Circular detection disabled by settings")
    quality_scores = []
    low_quality_entities = []

    for entity in all_entities:
        # Get quality score from attributes if available
        # Quality scores are stored as a dict with 'average' key (e.g., quality_scores.average)
        attrs = entity.attributes or {}
        quality_score = 0.0

        if "quality_scores" in attrs and isinstance(attrs["quality_scores"], dict):
            quality_score = attrs["quality_scores"].get("average", 0.0)

        if isinstance(quality_score, (int, float)) and not isinstance(quality_score, bool):
            score_float = float(quality_score)
            if not math.isfinite(score_float):
                logger.warning(
                    f"Entity {entity.name} has non-finite quality score: {score_float}, "
                    "treating as 0.0"
                )
                score_float = 0.0
            score_float = max(0.0, min(10.0, score_float))
            quality_scores.append(score_float)
            if score_float < quality_threshold:
                low_quality_entities.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "quality_score": score_float,
                    }
                )
        elif isinstance(quality_score, bool):
            logger.debug(f"Entity {entity.name} has bool quality score: {quality_score}, skipping")

    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Quality distribution (buckets of 2)
    quality_distribution: dict[str, int] = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
    for score in quality_scores:
        if score < 2:
            quality_distribution["0-2"] += 1
        elif score < 4:
            quality_distribution["2-4"] += 1
        elif score < 6:
            quality_distribution["4-6"] += 1
        elif score < 8:
            quality_distribution["6-8"] += 1
        else:
            quality_distribution["8-10"] += 1

    # Relationship density
    relationship_density = total_relationships / total_entities if total_entities > 0 else 0.0

    # Build metrics object
    metrics = WorldHealthMetrics(
        total_entities=total_entities,
        entity_counts=entity_counts,
        total_relationships=total_relationships,
        orphan_count=len(orphan_entities),
        orphan_entities=orphan_entities,
        circular_count=len(circular_relationships),
        circular_relationships=circular_relationships,
        hierarchical_circular_count=hierarchical_circular_count,
        mutual_circular_count=mutual_circular_count,
        average_quality=average_quality,
        quality_distribution=quality_distribution,
        low_quality_entities=low_quality_entities,
        relationship_density=relationship_density,
    )

    # Calculate health score and generate recommendations
    metrics.calculate_health_score()
    metrics.generate_recommendations()

    logger.info(
        f"World health metrics computed: score={metrics.health_score:.1f}, "
        f"entities={total_entities}, relationships={total_relationships}, "
        f"orphans={len(orphan_entities)}, circular={len(circular_relationships)}"
    )

    return metrics
