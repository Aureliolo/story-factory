"""World building functions for WorldService."""

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.utils.exceptions import GenerationCancelledError
from src.utils.validation import validate_not_none, validate_type

if TYPE_CHECKING:
    from src.memory.entities import Entity
    from src.services import ServiceContainer
    from src.services.world_service import WorldBuildOptions, WorldBuildProgress, WorldService

logger = logging.getLogger(__name__)


def build_world(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    options: WorldBuildOptions | None = None,
    progress_callback: Callable[[WorldBuildProgress], None] | None = None,
) -> dict[str, int]:
    """Build/rebuild world with unified logic.

    This is the single entry point for all world building operations.
    Both "Build Story Structure" and "Rebuild World" use this method
    with different options.

    Args:
        svc: WorldService instance.
        state: Story state with completed brief.
        world_db: WorldDatabase to populate.
        services: ServiceContainer for accessing other services.
        options: World build options. Defaults to minimal build.
        progress_callback: Optional callback for progress updates.

    Returns:
        Dictionary with counts of generated entities by type.

    Raises:
        ValueError: If no brief exists.
        WorldGenerationError: If generation fails.
    """
    from src.services.world_service import WorldBuildOptions, WorldBuildProgress

    validate_not_none(state, "state")
    validate_type(state, "state", StoryState)
    validate_not_none(world_db, "world_db")
    validate_type(world_db, "world_db", WorldDatabase)

    if not state.brief:
        raise ValueError("Cannot build world - no brief exists.")

    options = options or WorldBuildOptions.full()
    counts: dict[str, int] = {
        "characters": 0,
        "locations": 0,
        "factions": 0,
        "items": 0,
        "concepts": 0,
        "relationships": 0,
    }

    # Calculate total steps for progress reporting
    total_steps = _calculate_total_steps(options)
    current_step = 0

    def report_progress(message: str, entity_type: str | None = None, count: int = 0) -> None:
        """Report world build progress to callback."""
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(
                WorldBuildProgress(
                    step=current_step,
                    total_steps=total_steps,
                    message=message,
                    entity_type=entity_type,
                    count=count,
                )
            )

    logger.info(f"Starting world build for project {state.id} with options: {options}")

    # Persist world template ID to state if provided
    if options.world_template:
        state.world_template_id = options.world_template.id
        logger.debug(f"Set world_template_id on state: {options.world_template.id}")

    def check_cancelled() -> None:
        """Raise if cancellation requested."""
        if options.is_cancelled():
            logger.info("World build cancelled by user")
            raise GenerationCancelledError("Generation cancelled by user")

    # Step 1: Clear existing data if requested
    if options.clear_existing:
        check_cancelled()
        report_progress("Clearing existing world data...")
        _clear_world_db(world_db)
        logger.info("Cleared existing world data")

    # Step 2: Generate story structure (characters, chapters) if requested
    if options.generate_structure:
        check_cancelled()
        report_progress("Generating story structure...")
        if options.clear_existing:
            # Full rebuild - clear story state and regenerate
            services.story.rebuild_world(state)
        else:
            # Initial build - build on existing state (doesn't clear)
            # Note: build_structure also extracts to world_db, but we handle that below
            orchestrator = services.story._get_orchestrator(state)
            orchestrator.build_story_structure()
            services.story._sync_state(orchestrator, state)
        counts["characters"] = len(state.characters)
        logger.info(
            f"Story structure built: {len(state.characters)} characters, "
            f"{len(state.chapters)} chapters"
        )

    # Step 3: Extract characters to world database
    check_cancelled()
    report_progress("Adding characters to world...", "character")
    char_count = _extract_characters_to_world(state, world_db)
    counts["characters"] = char_count
    logger.info(f"Extracted {char_count} characters to world database")

    # Step 4: Generate locations if requested
    if options.generate_locations:
        check_cancelled()
        report_progress("Generating locations...", "location")
        loc_count = _generate_locations(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["locations"] = loc_count
        logger.info(f"Generated {loc_count} locations")

    # Step 5: Generate factions if requested
    if options.generate_factions:
        check_cancelled()
        report_progress("Generating factions...", "faction")
        faction_count = _generate_factions(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["factions"] = faction_count
        logger.info(f"Generated {faction_count} factions")

    # Step 6: Generate items if requested
    if options.generate_items:
        check_cancelled()
        report_progress("Generating items...", "item")
        item_count = _generate_items(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["items"] = item_count
        logger.info(f"Generated {item_count} items")

    # Step 7: Generate concepts if requested
    if options.generate_concepts:
        check_cancelled()
        report_progress("Generating concepts...", "concept")
        concept_count = _generate_concepts(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["concepts"] = concept_count
        logger.info(f"Generated {concept_count} concepts")

    # Step 8: Generate relationships if requested
    if options.generate_relationships:
        check_cancelled()
        report_progress("Generating relationships...", "relationship")
        rel_count = _generate_relationships(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["relationships"] = rel_count
        logger.info(f"Generated {rel_count} relationships")

    report_progress("World build complete!")
    logger.info(f"World build complete for project {state.id}: {counts}")
    return counts


def _calculate_total_steps(options: WorldBuildOptions) -> int:
    """Calculate total number of steps for progress reporting."""
    steps = 2  # Character extraction + completion
    if options.clear_existing:
        steps += 1
    if options.generate_structure:
        steps += 1
    if options.generate_locations:
        steps += 1
    if options.generate_factions:
        steps += 1
    if options.generate_items:
        steps += 1
    if options.generate_concepts:
        steps += 1
    if options.generate_relationships:
        steps += 1
    return steps


def _clear_world_db(world_db: WorldDatabase) -> None:
    """Clear all entities and relationships from world database."""
    # Delete relationships first (they reference entities)
    relationships = world_db.list_relationships()
    logger.info(f"Deleting {len(relationships)} existing relationships...")
    for rel in relationships:
        world_db.delete_relationship(rel.id)

    # Delete all entities
    entities = world_db.list_entities()
    logger.info(f"Deleting {len(entities)} existing entities...")
    for entity in entities:
        world_db.delete_entity(entity.id)


def _extract_characters_to_world(state: StoryState, world_db: WorldDatabase) -> int:
    """Extract characters from story state to world database."""
    added_count = 0
    for char in state.characters:
        # Check if already exists
        existing = world_db.search_entities(char.name, entity_type="character")
        if existing:
            logger.debug(f"Character already exists: {char.name}")
            continue

        world_db.add_entity(
            entity_type="character",
            name=char.name,
            description=char.description,
            attributes={
                "role": char.role,
                "personality_traits": char.personality_traits,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
            },
        )
        added_count += 1

    return added_count


def _generate_locations(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add locations to world database using quality refinement."""
    # Use project-level settings if available, otherwise fall back to global
    loc_min = state.target_locations_min or svc.settings.world_gen_locations_min
    loc_max = state.target_locations_max or svc.settings.world_gen_locations_max
    location_count = random.randint(loc_min, loc_max)

    location_names = [e.name for e in world_db.list_entities() if e.type == "location"]

    location_results = services.world_quality.generate_locations_with_quality(
        state,
        location_names,
        location_count,
        cancel_check=cancel_check,
    )

    added_count = 0
    for loc, scores in location_results:
        if cancel_check and cancel_check():
            logger.info(f"Location processing cancelled after {added_count} locations")
            break
        name = loc.get("name", "")
        if name:
            world_db.add_entity(
                entity_type="location",
                name=name,
                description=loc.get("description", ""),
                attributes={
                    "significance": loc.get("significance", ""),
                    "quality_scores": scores.to_dict(),
                },
            )
            added_count += 1
        else:
            logger.warning(f"Skipping invalid location: {loc}")

    return added_count


def _generate_factions(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add factions to world database."""
    all_entities = world_db.list_entities()
    faction_names = [e.name for e in all_entities if e.type == "faction"]
    existing_locations = [e.name for e in all_entities if e.type == "location"]

    # Use project-level settings if available, otherwise fall back to global
    fac_min = state.target_factions_min or svc.settings.world_gen_factions_min
    fac_max = state.target_factions_max or svc.settings.world_gen_factions_max
    faction_count = random.randint(fac_min, fac_max)

    faction_results = services.world_quality.generate_factions_with_quality(
        state,
        faction_names,
        faction_count,
        existing_locations,
        cancel_check=cancel_check,
    )
    added_count = 0

    for faction, faction_scores in faction_results:
        if isinstance(faction, dict) and "name" in faction:
            faction_entity_id = world_db.add_entity(
                entity_type="faction",
                name=faction["name"],
                description=faction.get("description", ""),
                attributes={
                    "leader": faction.get("leader", ""),
                    "goals": faction.get("goals", []),
                    "values": faction.get("values", []),
                    "base_location": faction.get("base_location", ""),
                    "quality_scores": faction_scores.to_dict(),
                },
            )
            added_count += 1

            # Create relationship to base location if it exists
            base_loc = faction.get("base_location", "")
            if base_loc:
                location_entity = next(
                    (e for e in all_entities if e.name == base_loc and e.type == "location"),
                    None,
                )
                if location_entity:
                    world_db.add_relationship(
                        source_id=faction_entity_id,
                        target_id=location_entity.id,
                        relation_type="based_in",
                        description=f"{faction['name']} is headquartered in {base_loc}",
                    )
        else:
            logger.warning(f"Skipping invalid faction: {faction}")

    return added_count


def _generate_items(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add items to world database."""
    item_names = [e.name for e in world_db.list_entities() if e.type == "item"]

    # Use project-level settings if available, otherwise fall back to global
    item_min = state.target_items_min or svc.settings.world_gen_items_min
    item_max = state.target_items_max or svc.settings.world_gen_items_max
    item_count = random.randint(item_min, item_max)

    item_results = services.world_quality.generate_items_with_quality(
        state,
        item_names,
        item_count,
        cancel_check=cancel_check,
    )
    added_count = 0

    for item, item_scores in item_results:
        if isinstance(item, dict) and "name" in item:
            world_db.add_entity(
                entity_type="item",
                name=item["name"],
                description=item.get("description", ""),
                attributes={
                    "significance": item.get("significance", ""),
                    "owner": item.get("owner", ""),
                    "location": item.get("location", ""),
                    "quality_scores": item_scores.to_dict(),
                },
            )
            added_count += 1
        else:
            logger.warning(f"Skipping invalid item: {item}")

    return added_count


def _generate_concepts(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add concepts to world database."""
    concept_names = [e.name for e in world_db.list_entities() if e.type == "concept"]

    # Use project-level settings if available, otherwise fall back to global
    concept_min = state.target_concepts_min or svc.settings.world_gen_concepts_min
    concept_max = state.target_concepts_max or svc.settings.world_gen_concepts_max
    concept_count = random.randint(concept_min, concept_max)

    concept_results = services.world_quality.generate_concepts_with_quality(
        state,
        concept_names,
        concept_count,
        cancel_check=cancel_check,
    )
    added_count = 0

    for concept, concept_scores in concept_results:
        if isinstance(concept, dict) and "name" in concept:
            world_db.add_entity(
                entity_type="concept",
                name=concept["name"],
                description=concept.get("description", ""),
                attributes={
                    "type": concept.get("type", ""),
                    "importance": concept.get("importance", ""),
                    "quality_scores": concept_scores.to_dict(),
                },
            )
            added_count += 1
        else:
            logger.warning(f"Skipping invalid concept: {concept}")

    return added_count


def _generate_relationships(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add relationships between entities."""
    all_entities = world_db.list_entities()
    entity_names = [e.name for e in all_entities]
    existing_rels = [(r.source_id, r.target_id) for r in world_db.list_relationships()]

    rel_count = random.randint(
        svc.settings.world_gen_relationships_min,
        svc.settings.world_gen_relationships_max,
    )

    relationships = services.story.generate_relationships(
        state, entity_names, existing_rels, rel_count
    )
    added_count = 0

    for rel in relationships:
        if cancel_check and cancel_check():
            logger.info(f"Relationship processing cancelled after {added_count} relationships")
            break
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            # Find source and target entities (fuzzy match for LLM name variations)
            source_entity = _find_entity_by_name(all_entities, rel["source"])
            target_entity = _find_entity_by_name(all_entities, rel["target"])

            if source_entity and target_entity:
                world_db.add_relationship(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=rel.get("relation_type", "related_to"),
                    description=rel.get("description", ""),
                )
                added_count += 1
            else:
                logger.warning(
                    f"Could not find entities for relationship: "
                    f"{rel.get('source')} -> {rel.get('target')}"
                )
        else:
            logger.warning(f"Skipping invalid relationship: {rel}")

    return added_count


_LEADING_ARTICLES = ("the ", "a ", "an ")


def _normalize_name(name: str) -> str:
    """Normalize an entity name for fuzzy comparison.

    Collapses whitespace, lowercases, and strips common English articles
    ("The", "A", "An") that LLMs frequently prepend, causing mismatches
    (e.g., "The Echoes of the Network" vs "Echoes of the Network").
    """
    normalized = " ".join(name.split()).lower()
    for article in _LEADING_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article) :]
            break
    return normalized


def _find_entity_by_name(entities: list[Entity], name: str) -> Entity | None:
    """Find an entity by name with fuzzy matching.

    Tries exact match first, then falls back to normalized comparison
    to handle LLM name variations like added "The" prefixes or
    case differences. If multiple entities match via fuzzy matching,
    logs a warning and returns None to avoid ambiguous assignment.

    Args:
        entities: List of entity objects with .name attribute.
        name: Name to search for.

    Returns:
        Matching entity, or None if not found or ambiguous.
    """
    # Exact match first (fast path)
    for e in entities:
        if e.name == name:
            return e

    # Fuzzy match: normalize both sides and collect all matches
    normalized_target = _normalize_name(name)
    matches = [e for e in entities if _normalize_name(e.name) == normalized_target]

    if len(matches) == 1:
        logger.debug(f"Fuzzy matched relationship entity: '{name}' -> '{matches[0].name}'")
        return matches[0]

    if len(matches) > 1:
        match_names = [e.name for e in matches]
        logger.warning(f"Ambiguous fuzzy match for '{name}': {match_names}. Skipping assignment.")
        return None

    return None
