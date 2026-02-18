"""World building functions for WorldService."""

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.memory.story_state import PlotOutline, StoryState
from src.memory.world_calendar import WorldCalendar
from src.memory.world_database import WorldDatabase
from src.memory.world_settings import WorldSettings
from src.utils.exceptions import GenerationCancelledError, WorldGenerationError
from src.utils.validation import validate_not_none, validate_type

if TYPE_CHECKING:
    from src.memory.entities import Entity
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
            except Exception as e:
                logger.warning("Calendar generation failed (non-fatal), continuing without: %s", e)

        # Steps 2-9: entity generation (inside try/finally for calendar context cleanup)
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
        faction_count, implicit_rel_count = _generate_factions(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
        )
        counts["factions"] = faction_count
        counts["implicit_relationships"] += implicit_rel_count
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

    # Step 8a: Recover orphan entities by generating additional relationships
    if options.generate_relationships:
        check_cancelled()
        report_progress("Recovering orphan entities...")
        orphan_count = _recover_orphans(
            svc, state, world_db, services, cancel_check=options.is_cancelled
        )
        if orphan_count > 0:
            counts["relationships"] += orphan_count
            logger.info(f"Orphan recovery added {orphan_count} relationships")

    # Step 9: Batch-embed all world content for RAG context retrieval
    check_cancelled()
    report_progress("Embedding world content for RAG...")
    try:
        embed_counts = services.embedding.embed_all_world_data(world_db, state)
        logger.info("World embedding complete: %s", embed_counts)
    except Exception as e:
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
            logger.debug(f"Character already exists: {char.name}")
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
) -> tuple[int, int]:
    """Generate and add factions to world database.

    Returns:
        Tuple of (factions_added, implicit_relationships_added).
    """
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
            logger.warning(f"Skipping invalid faction: {faction}")

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
            logger.info(f"Relationship processing cancelled after {added_count} relationships")
            break
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            # Find source and target entities (fuzzy match for LLM name variations)
            source_entity = _find_entity_by_name(all_entities, rel["source"])
            target_entity = _find_entity_by_name(all_entities, rel["target"])

            if source_entity and target_entity:
                # relation_type is already normalized by world_quality_service;
                # just use it directly with a safety default.
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
                    f"Could not find entities for relationship: "
                    f"{rel.get('source')} -> {rel.get('target')}"
                )
        else:
            logger.warning(f"Skipping invalid relationship: {rel}")

    return added_count


MAX_RETRIES_PER_ORPHAN = 2  # Up to 3 total attempts per orphan (1 + 2 retries)


def _recover_orphans(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Connect orphan entities by generating relationships (up to MAX_RETRIES_PER_ORPHAN+1 attempts each)."""
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
    entity_by_id = {e.id: e.name for e in all_entities}

    # Build existing relationships list
    existing_rels: list[tuple[str, str, str]] = []
    for r in world_db.list_relationships():
        source_name = entity_by_id.get(r.source_id)
        target_name = entity_by_id.get(r.target_id)
        if source_name and target_name:
            existing_rels.append((source_name, target_name, r.relation_type))

    # Track orphan names still needing connections (mutable set for fast lookup)
    orphan_names = {o.name.lower() for o in orphans}

    added_count = 0

    for orphan in orphans:
        if cancel_check and cancel_check():
            logger.info("Orphan recovery cancelled")
            break

        # Skip if this orphan was already connected by a previous orphan's relationship
        if orphan.name.lower() not in orphan_names:
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

                # Find entities and add to database
                source_entity = _find_entity_by_name(all_entities, rel["source"])
                target_entity = _find_entity_by_name(all_entities, rel["target"])

                if source_entity and target_entity:
                    # Safety check: verify at least one endpoint is an orphan
                    # (should always pass due to required_entity constraint in quality loop)
                    source_is_orphan = source_entity.name.lower() in orphan_names
                    target_is_orphan = target_entity.name.lower() in orphan_names
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
                        orphan_names.discard(source_entity.name.lower())
                    if target_is_orphan:
                        orphan_names.discard(target_entity.name.lower())
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
