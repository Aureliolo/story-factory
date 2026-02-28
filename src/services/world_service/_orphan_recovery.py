"""Orphan entity recovery for the world build pipeline.

Connects entities that have no relationships by generating targeted
relationships for each orphan until all entities are integrated into
the world graph.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.memory.entities import Entity
from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._name_matching import _find_entity_by_name, _normalize_name
from src.utils.exceptions import GenerationCancelledError, WorldGenerationError

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.services.world_service import WorldService

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_ORPHAN = 2  # Up to 3 total attempts per orphan (1 + 2 retries)


def _recover_orphans(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Connect orphan entities by generating relationships (up to MAX_RETRIES_PER_ORPHAN+1 attempts each)."""
    threshold = svc.settings.fuzzy_match_threshold

    orphans = world_db.find_orphans()
    if not orphans:
        logger.debug("No orphan entities found, skipping recovery")
        return 0

    logger.info(
        "Orphan recovery: found %d orphan entities: %s",
        len(orphans),
        [o.name for o in orphans],
    )

    all_entities = world_db.list_entities()
    entity_by_id = {e.id: e for e in all_entities}

    # Pre-build normalized name cache to avoid O(N^2) re-normalization
    normalized_name_cache: dict[str, str] = {e.name: _normalize_name(e.name) for e in all_entities}

    # Build existing relationships list
    existing_rels: list[tuple[str, str, str]] = []
    for r in world_db.list_relationships():
        source_entity_obj = entity_by_id.get(r.source_id)
        target_entity_obj = entity_by_id.get(r.target_id)
        if not source_entity_obj or not target_entity_obj:
            logger.warning(
                "Skipping relationship %s -> %s (%s): missing entity reference",
                r.source_id,
                r.target_id,
                r.relation_type,
            )
            continue
        existing_rels.append((source_entity_obj.name, target_entity_obj.name, r.relation_type))

    # Track orphan names still needing connections (mutable set for fast lookup).
    # Orphans are a subset of all_entities, so normalized_name_cache is guaranteed
    # to contain every orphan's name from the comprehension at line 52.
    orphan_names: set[str] = {normalized_name_cache[o.name] for o in orphans}

    added_count = 0

    for orphan in orphans:
        if cancel_check and cancel_check():
            logger.info("Orphan recovery cancelled")
            break

        # Skip if this orphan was already connected by a previous orphan's relationship
        orphan_norm = normalized_name_cache.get(orphan.name, _normalize_name(orphan.name))
        if orphan_norm not in orphan_names:
            logger.debug(
                "Orphan '%s' already connected by a previous relationship, skipping",
                orphan.name,
            )
            continue

        # Build entity list with orphan first (primacy bias) + all potential partners
        partner_names = [e.name for e in all_entities if e.id != orphan.id]
        if not partner_names:
            logger.warning("Orphan recovery: only one entity '%s' available; skipping", orphan.name)
            continue
        constrained_names = [orphan.name, *partner_names]

        for attempt in range(MAX_RETRIES_PER_ORPHAN + 1):
            if cancel_check and cancel_check():
                logger.info("Orphan recovery cancelled during retries for '%s'", orphan.name)
                break

            logger.debug(
                "Orphan recovery: orphan '%s' attempt %d/%d",
                orphan.name,
                attempt + 1,
                MAX_RETRIES_PER_ORPHAN + 1,
            )

            try:
                rel, scores, _iterations = (
                    services.world_quality.generate_relationship_with_quality(
                        state,
                        constrained_names,
                        existing_rels,
                        required_entity=orphan.name,
                    )
                )

                if not rel or not rel.get("source") or not rel.get("target"):
                    logger.warning(
                        "Orphan recovery: empty relationship for '%s' (attempt %d)",
                        orphan.name,
                        attempt + 1,
                    )
                    continue

                # Find entities — when the orphan's normalized name matches a
                # relationship endpoint, use the orphan object directly (bypassing
                # fuzzy lookup) to avoid cross-type name collisions where multiple
                # entities share the same name
                source_name_norm = normalized_name_cache.get(
                    rel["source"], _normalize_name(rel["source"])
                )
                target_name_norm = normalized_name_cache.get(
                    rel["target"], _normalize_name(rel["target"])
                )
                orphan_name_norm = orphan_norm

                source_entity: Entity | None
                target_entity: Entity | None
                if source_name_norm == orphan_name_norm:
                    source_entity = orphan
                    target_entity = _find_entity_by_name(
                        all_entities, rel["target"], threshold=threshold
                    )
                    logger.debug(
                        "Orphan '%s' matched source '%s' (normalized: '%s'),"
                        " using direct reference; target '%s' via fuzzy lookup",
                        orphan.name,
                        rel["source"],
                        source_name_norm,
                        rel["target"],
                    )
                elif target_name_norm == orphan_name_norm:
                    source_entity = _find_entity_by_name(
                        all_entities, rel["source"], threshold=threshold
                    )
                    target_entity = orphan
                    logger.debug(
                        "Orphan '%s' matched target '%s' (normalized: '%s'),"
                        " using direct reference; source '%s' via fuzzy lookup",
                        orphan.name,
                        rel["target"],
                        target_name_norm,
                        rel["source"],
                    )
                else:
                    source_entity = _find_entity_by_name(
                        all_entities, rel["source"], threshold=threshold
                    )
                    target_entity = _find_entity_by_name(
                        all_entities, rel["target"], threshold=threshold
                    )
                    logger.debug(
                        "Orphan '%s' (normalized: '%s') matched neither source '%s'"
                        " (normalized: '%s') nor target '%s' (normalized: '%s'),"
                        " using fuzzy lookup for both",
                        orphan.name,
                        orphan_name_norm,
                        rel["source"],
                        source_name_norm,
                        rel["target"],
                        target_name_norm,
                    )

                if source_entity and target_entity:
                    # Safety check: verify at least one endpoint is an orphan
                    # (should always pass due to required_entity constraint in quality loop)
                    source_is_orphan = (
                        normalized_name_cache.get(
                            source_entity.name, _normalize_name(source_entity.name)
                        )
                        in orphan_names
                    )
                    target_is_orphan = (
                        normalized_name_cache.get(
                            target_entity.name, _normalize_name(target_entity.name)
                        )
                        in orphan_names
                    )
                    if not source_is_orphan and not target_is_orphan:
                        logger.warning(
                            "Orphan recovery: skipping relationship %s -> %s"
                            " (neither is an orphan)",
                            rel["source"],
                            rel["target"],
                        )
                        continue

                    relation_type = rel.get("relation_type")
                    if not relation_type:
                        logger.debug(
                            "Orphan recovery: no relation_type in generated relationship,"
                            " defaulting to 'related_to'"
                        )
                        relation_type = "related_to"
                    world_db.add_relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation_type=relation_type,
                        description=rel.get("description", ""),
                    )
                    existing_rels.append((source_entity.name, target_entity.name, relation_type))
                    added_count += 1
                    logger.info(
                        "Orphan recovery: added %s -> %s (%s), quality: %.1f",
                        rel["source"],
                        rel["target"],
                        relation_type,
                        scores.average,
                    )

                    # Remove connected orphan(s) from tracking set
                    if source_is_orphan:
                        orphan_names.discard(
                            normalized_name_cache.get(
                                source_entity.name, _normalize_name(source_entity.name)
                            )
                        )
                    if target_is_orphan:
                        orphan_names.discard(
                            normalized_name_cache.get(
                                target_entity.name, _normalize_name(target_entity.name)
                            )
                        )
                    break  # Success — move to the next orphan
                else:
                    logger.warning(
                        "Orphan recovery: could not resolve entities for %s -> %s",
                        rel.get("source"),
                        rel.get("target"),
                    )

            except GenerationCancelledError:
                raise
            except WorldGenerationError as e:
                logger.warning(
                    "Orphan recovery: attempt %d for '%s' failed: %s",
                    attempt + 1,
                    orphan.name,
                    e,
                )
                continue

    logger.info(
        "Orphan recovery complete: generated %d relationships for %d orphan entities",
        added_count,
        len(orphans),
    )
    return added_count
