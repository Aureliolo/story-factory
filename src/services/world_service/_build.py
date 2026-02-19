"""World building functions for WorldService."""

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.memory.story_state import PlotOutline, StoryState
from src.memory.world_calendar import WorldCalendar
from src.memory.world_database import WorldDatabase
from src.memory.world_settings import WorldSettings
from src.services.world_service._event_helpers import (
    build_event_entity_context,
    build_event_timestamp,
    resolve_event_participants,
)
from src.services.world_service._lifecycle_helpers import (
    build_character_lifecycle,
    build_entity_lifecycle,
)
from src.services.world_service._name_matching import _find_entity_by_name
from src.services.world_service._orphan_recovery import _recover_orphans
from src.utils.exceptions import GenerationCancelledError, WorldGenerationError
from src.utils.validation import validate_not_none, validate_type

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.services.world_service import WorldBuildOptions, WorldBuildProgress, WorldService

logger = logging.getLogger(__name__)

# Cap relationship count at this ratio of entity count to avoid excessive duplicates
RELATIONSHIP_TO_ENTITY_RATIO_CAP = 1.5


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
        "calendar": 0,
        "characters": 0,
        "locations": 0,
        "factions": 0,
        "items": 0,
        "concepts": 0,
        "events": 0,
        "relationships": 0,
        "implicit_relationships": 0,
    }

    # Reconcile calendar option with settings gate before computing steps
    effective_calendar = options.generate_calendar and svc.settings.generate_calendar_on_world_build
    # Calculate total steps for progress reporting
    total_steps = _calculate_total_steps(options, generate_calendar=effective_calendar)
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

    logger.info("Starting world build for project %s with options: %s", state.id, options)

    # Persist world template ID to state if provided
    if options.world_template:
        state.world_template_id = options.world_template.id
        logger.debug("Set world_template_id on state: %s", options.world_template.id)

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

    try:
        # Step 1a: Generate calendar if requested and enabled in settings
        if options.generate_calendar and svc.settings.generate_calendar_on_world_build:
            check_cancelled()
            report_progress("Generating calendar system...", "calendar")
            try:
                calendar_dict, calendar_scores, calendar_iterations = (
                    services.world_quality.generate_calendar_with_quality(state)
                )
                # Convert dict back to WorldCalendar and save to world settings
                calendar = WorldCalendar.from_dict(calendar_dict)
                world_settings = world_db.get_world_settings()
                if world_settings:
                    world_settings.calendar = calendar
                else:
                    world_settings = WorldSettings(calendar=calendar)
                world_db.save_world_settings(world_settings)
                counts["calendar"] = 1
                # Set calendar context for downstream entity generation prompts
                services.world_quality.set_calendar_context(calendar_dict)
                logger.info(
                    "Generated calendar '%s' after %d iteration(s), quality: %.1f",
                    calendar.current_era_name,
                    calendar_iterations,
                    calendar_scores.average,
                )
            except GenerationCancelledError:
                raise
            except (WorldGenerationError, ValueError, RuntimeError) as e:
                logger.warning("Calendar generation failed (non-fatal), continuing without: %s", e)

        # Steps 2-9 (with sub-steps 2a-2c, 8a-8b): entity generation (inside try/finally for calendar context cleanup)
        _build_world_entities(
            svc,
            state,
            world_db,
            services,
            options,
            counts,
            check_cancelled,
            report_progress,
        )
    finally:
        services.world_quality.set_calendar_context(None)

    report_progress("World build complete!")
    total_rels = counts["relationships"] + counts["implicit_relationships"]
    logger.info(
        "World build complete for project %s: %s (total relationships: %d = %d explicit + %d implicit)",
        state.id,
        counts,
        total_rels,
        counts["relationships"],
        counts["implicit_relationships"],
    )
    return counts


def _build_world_entities(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    options: WorldBuildOptions,
    counts: dict[str, int],
    check_cancelled: Callable[[], None],
    report_progress: Callable[..., None],
) -> None:
    """Execute entity generation steps 2-9 of the world build."""
    # Step 2: Generate story structure (characters, chapters) if requested
    if options.generate_structure:
        check_cancelled()
        report_progress("Generating story structure...")
        if options.clear_existing:
            # Full rebuild - clear story state and regenerate
            services.story.rebuild_world(state)
        else:
            # Initial build - build on existing state (doesn't clear)
            orchestrator = services.story._get_orchestrator(state)
            orchestrator.build_story_structure()
            services.story._sync_state(orchestrator, state)
        counts["characters"] = len(state.characters)
        logger.info(
            "Story structure built: %d characters, %d chapters",
            len(state.characters),
            len(state.chapters),
        )

        # Step 2a: Review character quality from Architect output
        if state.characters:
            check_cancelled()
            report_progress("Reviewing character quality...", "character")
            reviewed_chars = services.world_quality.review_characters_batch(
                state.characters,
                state,
                cancel_check=options.is_cancelled,
            )
            # Update state with refined characters
            state.characters = [char for char, _scores in reviewed_chars]
            logger.info(
                "Character quality review complete: %d characters reviewed",
                len(reviewed_chars),
            )

        # Step 2b: Review plot quality from Architect output
        if state.plot_summary:
            check_cancelled()
            report_progress("Reviewing plot quality...", "plot")
            plot_outline = PlotOutline(
                plot_summary=state.plot_summary,
                plot_points=state.plot_points,
            )
            reviewed_plot, plot_scores, plot_iterations = (
                services.world_quality.review_plot_quality(plot_outline, state)
            )
            state.plot_summary = reviewed_plot.plot_summary
            state.plot_points = reviewed_plot.plot_points
            logger.info(
                "Plot quality review complete after %d iteration(s), quality: %.1f",
                plot_iterations,
                plot_scores.average,
            )

        # Step 2c: Review chapter quality from Architect output
        if state.chapters:
            check_cancelled()
            report_progress("Reviewing chapter quality...", "chapter")
            reviewed_chapters = services.world_quality.review_chapters_batch(
                state.chapters,
                state,
                cancel_check=options.is_cancelled,
            )
            # Update state with refined chapters
            state.chapters = [ch for ch, _scores in reviewed_chapters]
            logger.info(
                "Chapter quality review complete: %d chapters reviewed",
                len(reviewed_chapters),
            )

    # Step 3: Extract characters to world database
    check_cancelled()
    report_progress("Adding characters to world...", "character")
    char_count, char_implicit_rels = _extract_characters_to_world(state, world_db)
    counts["characters"] = char_count
    counts["implicit_relationships"] += char_implicit_rels
    logger.info("Extracted %d characters to world database", char_count)

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
        logger.info("Generated %d locations", loc_count)

    # Step 5: Generate factions if requested
    if options.generate_factions:
        check_cancelled()
        report_progress("Generating factions...", "faction")
        faction_count, implicit_rel_count = _generate_factions(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["factions"] = faction_count
        counts["implicit_relationships"] += implicit_rel_count
        logger.info("Generated %d factions", faction_count)

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
        logger.info("Generated %d items", item_count)

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
        logger.info("Generated %d concepts", concept_count)

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
        logger.info("Generated %d relationships", rel_count)

    # Step 8a: Recover orphan entities by generating additional relationships
    if options.generate_relationships:
        check_cancelled()
        report_progress("Recovering orphan entities...")
        orphan_count = _recover_orphans(
            svc, state, world_db, services, cancel_check=options.is_cancelled
        )
        if orphan_count > 0:
            counts["relationships"] += orphan_count
            logger.info("Orphan recovery added %d relationships", orphan_count)

    # Step 8b: Generate world events if requested
    if options.generate_events:
        check_cancelled()
        report_progress("Generating world events...", "event")
        event_count = _generate_events(
            svc, state, world_db, services, cancel_check=options.is_cancelled
        )
        counts["events"] = event_count
        logger.info("Generated %d world events", event_count)

    # Step 9: Batch-embed all world content for RAG context retrieval
    check_cancelled()
    report_progress("Embedding world content for RAG...")
    try:
        embed_counts = services.embedding.embed_all_world_data(world_db, state)
        logger.info("World embedding complete: %s", embed_counts)
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("World embedding failed (non-fatal), RAG context unavailable: %s", e)


def _calculate_total_steps(options: WorldBuildOptions, *, generate_calendar: bool = False) -> int:
    """Calculate total number of steps for progress reporting."""
    steps = 3  # Character extraction + embedding + completion
    if options.clear_existing:
        steps += 1
    if generate_calendar:
        steps += 1
    if options.generate_structure:
        steps += 1
        # Quality review steps for Architect output (characters, plot, chapters)
        steps += 3
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
        # +1 for orphan recovery step after relationship generation
        steps += 1
    if options.generate_events:
        steps += 1
    return steps


def _clear_world_db(world_db: WorldDatabase) -> None:
    """Clear all entities and relationships from world database."""
    # Delete relationships first (they reference entities)
    relationships = world_db.list_relationships()
    logger.info("Deleting %d existing relationships...", len(relationships))
    for rel in relationships:
        world_db.delete_relationship(rel.id)

    # Delete all entities
    entities = world_db.list_entities()
    logger.info("Deleting %d existing entities...", len(entities))
    for entity in entities:
        world_db.delete_entity(entity.id)


def _extract_characters_to_world(state: StoryState, world_db: WorldDatabase) -> tuple[int, int]:
    """Extract characters and their pre-defined relationships to world database.

    Uses a two-pass approach: first adds all character entities, then creates
    implicit relationships from Character.relationships (set by ArchitectAgent).

    Returns:
        Tuple of (characters_added, implicit_relationships_added).
    """
    added_count = 0
    char_id_map: dict[str, str] = {}
    newly_added: set[str] = set()

    # Pass 1: add all characters, building a name→ID map
    for char in state.characters:
        existing = world_db.get_entity_by_name(char.name, entity_type="character")
        if existing:
            logger.debug("Character already exists: %s", char.name)
            char_id_map[char.name] = existing.id
            continue

        entity_id = world_db.add_entity(
            entity_type="character",
            name=char.name,
            description=char.description,
            attributes={
                "role": char.role,
                "personality_traits": char.trait_names,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
                **build_character_lifecycle(char),
            },
        )
        char_id_map[char.name] = entity_id
        newly_added.add(char.name)
        added_count += 1

    # Pass 2: create implicit relationships only for newly added characters
    # (skip pre-existing characters to avoid duplicates on incremental builds)
    implicit_rel_count = 0
    for char in state.characters:
        if char.name not in newly_added:
            continue
        source_id = char_id_map[char.name]  # guaranteed by pass 1

        for related_name, relationship in char.relationships.items():
            target_id = char_id_map.get(related_name)
            if not target_id:
                logger.debug(
                    "Skipping relationship %s -[%s]-> %s: target not in character list",
                    char.name,
                    relationship,
                    related_name,
                )
                continue

            world_db.add_relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relationship,
            )
            implicit_rel_count += 1
            logger.debug(
                "Created implicit character relationship: %s -[%s]-> %s",
                char.name,
                relationship,
                related_name,
            )

    if implicit_rel_count:
        logger.info(
            "Character extraction created %d implicit relationship(s)",
            implicit_rel_count,
        )
    return added_count, implicit_rel_count


def _generate_locations(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add locations to world database using quality refinement."""
    # Use project-level settings if available, otherwise fall back to global
    loc_min = (
        state.target_locations_min
        if state.target_locations_min is not None
        else svc.settings.world_gen_locations_min
    )
    loc_max = (
        state.target_locations_max
        if state.target_locations_max is not None
        else svc.settings.world_gen_locations_max
    )
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
            logger.info("Location processing cancelled after %d locations", added_count)
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
                    **build_entity_lifecycle(loc, "location"),
                },
            )
            added_count += 1
        else:
            logger.warning("Skipping invalid location: %s", loc)

    return added_count


def _generate_factions(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Generate and add factions to world database.

    Returns:
        Tuple of (factions_added, implicit_relationships_added).
    """
    all_entities = world_db.list_entities()
    faction_names = [e.name for e in all_entities if e.type == "faction"]
    existing_locations = [e.name for e in all_entities if e.type == "location"]

    # Use project-level settings if available, otherwise fall back to global
    fac_min = (
        state.target_factions_min
        if state.target_factions_min is not None
        else svc.settings.world_gen_factions_min
    )
    fac_max = (
        state.target_factions_max
        if state.target_factions_max is not None
        else svc.settings.world_gen_factions_max
    )
    faction_count = random.randint(fac_min, fac_max)

    faction_results = services.world_quality.generate_factions_with_quality(
        state,
        faction_names,
        faction_count,
        existing_locations,
        cancel_check=cancel_check,
    )
    added_count = 0
    implicit_rel_count = 0

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
                    **build_entity_lifecycle(faction, "faction"),
                },
            )
            added_count += 1

            # Create implicit relationship to base location if it exists
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
                    implicit_rel_count += 1
                    logger.info(
                        "Created implicit 'based_in' relationship: %s -> %s",
                        faction["name"],
                        base_loc,
                    )
        else:
            logger.warning("Skipping invalid faction: %s", faction)

    if implicit_rel_count:
        logger.info(
            "Faction generation created %d implicit based_in relationship(s)",
            implicit_rel_count,
        )
    return added_count, implicit_rel_count


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
    item_min = (
        state.target_items_min
        if state.target_items_min is not None
        else svc.settings.world_gen_items_min
    )
    item_max = (
        state.target_items_max
        if state.target_items_max is not None
        else svc.settings.world_gen_items_max
    )
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
                    **build_entity_lifecycle(item, "item"),
                },
            )
            added_count += 1
        else:
            logger.warning("Skipping invalid item: %s", item)

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
    concept_min = (
        state.target_concepts_min
        if state.target_concepts_min is not None
        else svc.settings.world_gen_concepts_min
    )
    concept_max = (
        state.target_concepts_max
        if state.target_concepts_max is not None
        else svc.settings.world_gen_concepts_max
    )
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
                    **build_entity_lifecycle(concept, "concept"),
                },
            )
            added_count += 1
        else:
            logger.warning("Skipping invalid concept: %s", concept)

    return added_count


def _generate_relationships(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add relationships between entities using quality refinement.

    Args:
        svc: WorldService instance.
        state: Current story state with brief.
        world_db: World database to read entities from and persist relationships to.
        services: Service container providing the world quality service.
        cancel_check: Optional callable that returns True to stop generation.

    Returns:
        Number of relationships successfully added to the world database.
    """
    all_entities = world_db.list_entities()
    entity_names = [e.name for e in all_entities]

    # Map IDs to names so the quality service gets name pairs for duplicate detection
    # 3-tuples: (source_name, target_name, relation_type) for diversity analysis
    entity_by_id = {e.id: e.name for e in all_entities}
    existing_rels: list[tuple[str, str, str]] = []
    for r in world_db.list_relationships():
        source_name = entity_by_id.get(r.source_id)
        target_name = entity_by_id.get(r.target_id)
        if not source_name or not target_name:
            logger.warning(
                "Skipping relationship %s -> %s: missing entity reference",
                r.source_id,
                r.target_id,
            )
            continue
        existing_rels.append((source_name, target_name, r.relation_type))

    rel_count = random.randint(
        svc.settings.world_gen_relationships_min,
        svc.settings.world_gen_relationships_max,
    )

    # Cap relationship count based on available entities to avoid excessive duplicates
    max_by_entities = int(len(entity_names) * RELATIONSHIP_TO_ENTITY_RATIO_CAP)
    if rel_count > max_by_entities:
        logger.info(
            "Capping relationship count from %d to %d (based on %d entities x %.1f)",
            rel_count,
            max_by_entities,
            len(entity_names),
            RELATIONSHIP_TO_ENTITY_RATIO_CAP,
        )
        rel_count = max_by_entities

    relationship_results = services.world_quality.generate_relationships_with_quality(
        state, entity_names, existing_rels, rel_count, cancel_check=cancel_check
    )
    added_count = 0

    for rel, scores in relationship_results:
        if cancel_check and cancel_check():
            logger.info("Relationship processing cancelled after %d relationships", added_count)
            break
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            # Find source and target entities (fuzzy match for LLM name variations)
            source_entity = _find_entity_by_name(all_entities, rel["source"])
            target_entity = _find_entity_by_name(all_entities, rel["target"])

            if source_entity and target_entity:
                # relation_type may be absent if the LLM omits it;
                # default to "related_to" as a fallback.
                relation_type = rel.get("relation_type")
                if not relation_type:
                    logger.debug(
                        "Relationship %s -> %s missing relation_type, defaulting to 'related_to'",
                        rel["source"],
                        rel["target"],
                    )
                    relation_type = "related_to"
                world_db.add_relationship(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=relation_type,
                    description=rel.get("description", ""),
                )
                added_count += 1
                logger.debug(
                    "Added relationship %s -> %s (quality: %.1f)",
                    rel["source"],
                    rel["target"],
                    scores.average,
                )
            else:
                logger.warning(
                    "Could not find entities for relationship: %s -> %s",
                    rel.get("source"),
                    rel.get("target"),
                )
        else:
            logger.warning("Skipping invalid relationship: %s", rel)

    return added_count


def _generate_events(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add world events to the database using quality refinement.

    Collects all entities, relationships, and lifecycle data to build context,
    then generates events that reference existing entities as participants.

    Args:
        svc: WorldService instance.
        state: Current story state with brief.
        world_db: World database to read entities from and persist events to.
        services: Service container providing the world quality service.
        cancel_check: Optional callable that returns True to stop generation.

    Returns:
        Number of events successfully added to the world database.
    """
    entity_context = build_event_entity_context(world_db)

    # Get existing event descriptions for dedup
    existing_events = world_db.list_events()
    existing_descriptions = [e.description for e in existing_events]

    # Determine event count
    event_min = (
        state.target_events_min
        if state.target_events_min is not None
        else svc.settings.world_gen_events_min
    )
    event_max = (
        state.target_events_max
        if state.target_events_max is not None
        else svc.settings.world_gen_events_max
    )
    event_count = random.randint(event_min, event_max)

    event_results = services.world_quality.generate_events_with_quality(
        state,
        existing_descriptions,
        entity_context,
        event_count,
        cancel_check=cancel_check,
    )

    all_entities = world_db.list_entities()
    added_count = 0
    for event, event_scores in event_results:
        if cancel_check and cancel_check():
            logger.info("Event processing cancelled after %d events", added_count)
            break

        description = event.get("description", "")
        if not description:
            logger.warning("Skipping event with empty description: %s", event)
            continue

        timestamp_in_story = build_event_timestamp(event)
        participants = resolve_event_participants(event, all_entities)

        consequences = event.get("consequences", [])

        world_db.add_event(
            description=description,
            participants=participants if participants else None,
            timestamp_in_story=timestamp_in_story,
            consequences=consequences if consequences else None,
        )
        added_count += 1
        logger.debug(
            "Added event '%s' (quality: %.1f, participants: %d)",
            description[:60],
            event_scores.average,
            len(participants),
        )

    return added_count
