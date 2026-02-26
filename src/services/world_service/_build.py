"""World building functions for WorldService."""

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.memory.story_state import PlotOutline, StoryState
from src.memory.world_calendar import WorldCalendar
from src.memory.world_database import WorldDatabase
from src.memory.world_settings import WorldSettings
from src.services.world_service._event_helpers import _generate_events
from src.services.world_service._lifecycle_helpers import (
    build_character_lifecycle,
    build_entity_lifecycle,
)
from src.services.world_service._name_matching import _find_entity_by_name
from src.services.world_service._orphan_recovery import _recover_orphans
from src.services.world_service._warmup import _warm_models
from src.utils.exceptions import (
    DatabaseClosedError,
    GenerationCancelledError,
    WorldGenerationError,
)
from src.utils.validation import validate_not_none, validate_type

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.services.world_quality_service import EntityGenerationProgress
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
    total_steps = _calculate_total_steps(
        options,
        generate_calendar=effective_calendar,
        validate_temporal=svc.settings.validate_temporal_consistency,
    )
    current_step = 0

    def report_progress(message: str, entity_type: str | None = None, count: int = 0) -> None:
        """Report world build progress to callback."""
        nonlocal current_step
        current_step += 1
        prefixed = f"[{current_step}/{total_steps}] {message}"
        logger.info("Build progress: %s", prefixed)
        if progress_callback:
            progress_callback(
                WorldBuildProgress(
                    step=current_step,
                    total_steps=total_steps,
                    message=prefixed,
                    entity_type=entity_type,
                    count=count,
                )
            )

    logger.info(f"Starting world build for project {state.id} with options: {options}")

    # Check cancellation before doing warm-up work
    if options.is_cancelled():
        logger.info("World build cancelled by user before warm-up")
        raise GenerationCancelledError("Generation cancelled by user")

    # Pre-load creator and judge models to avoid ~4.5s cold-start penalty (~2s net savings)
    _warm_models(services)

    # Log quality thresholds once at build start (not per-entity)
    config = services.world_quality.get_config()
    logger.info(
        "Quality loop config: max_iterations=%d, default_threshold=%.1f, "
        "creator_temp=%.1f, judge_temp=%.1f, early_stop_patience=%d",
        config.max_iterations,
        config.quality_threshold,
        config.creator_temperature,
        config.judge_temperature,
        config.early_stopping_patience,
    )
    if config.quality_thresholds:
        logger.info("Per-entity thresholds: %s", config.quality_thresholds)

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
            except GenerationCancelledError, DatabaseClosedError:
                raise
            except Exception as e:
                logger.warning("Calendar generation failed (non-fatal), continuing without: %s", e)

        # Steps 2-10: entity generation (inside try/finally for calendar context cleanup)
        _build_world_entities(
            svc,
            state,
            world_db,
            services,
            options,
            counts,
            check_cancelled,
            report_progress,
            raw_progress_callback=progress_callback,
            step_context=lambda: (current_step, total_steps),
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
    *,
    raw_progress_callback: Callable[[WorldBuildProgress], None] | None = None,
    step_context: Callable[[], tuple[int, int]] | None = None,
) -> None:
    """Execute all world-building steps after initial setup.

    Steps: structure generation, character extraction, quality review (characters,
    plot, chapters), locations, factions, items, concepts, relationships, orphan
    recovery, events, temporal validation, and embeddings.
    """
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

        # Wire sub-step progress: adapter maps EntityGenerationProgress
        # into WorldBuildProgress so the build dialog shows per-relationship updates.
        rel_progress_cb = None
        if raw_progress_callback and step_context:
            from src.services.world_service import WorldBuildProgress as WBP

            def _rel_progress_adapter(p: EntityGenerationProgress) -> None:
                """Map EntityGenerationProgress to WorldBuildProgress with sub-step fields."""
                try:
                    cur_step, tot_steps = step_context()
                    raw_progress_callback(
                        WBP(
                            step=cur_step,
                            total_steps=tot_steps,
                            message=(
                                f"[{cur_step}/{tot_steps}] Generating relationship "
                                f"{p.current}/{p.total}..."
                            ),
                            entity_type="relationship",
                            count=p.current,
                            sub_current=p.current,
                            sub_total=p.total,
                            sub_entity_name=p.entity_name,
                        )
                    )
                except Exception:
                    logger.warning("Sub-step progress callback failed", exc_info=True)

            rel_progress_cb = _rel_progress_adapter

        rel_count = _generate_relationships(
            svc,
            state,
            world_db,
            services,
            cancel_check=options.is_cancelled,
            progress_callback=rel_progress_cb,
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

    # Step 8b: Generate world events if requested (non-fatal — build continues if this fails)
    if options.generate_events:
        check_cancelled()
        report_progress("Generating world events...", "event")
        try:
            event_count = _generate_events(
                svc, state, world_db, services, cancel_check=options.is_cancelled
            )
            counts["events"] = event_count
            logger.info("Generated %d world events", event_count)
        except GenerationCancelledError:
            raise
        except (WorldGenerationError, ValueError) as e:
            logger.warning("Event generation failed (non-fatal), continuing without events: %s", e)

    # Step 9: Validate temporal consistency (non-fatal, before embedding)
    if svc.settings.validate_temporal_consistency:
        check_cancelled()
        report_progress("Validating temporal consistency...")
        try:
            result = services.temporal_validation.validate_world(world_db)
            for issue in result.errors:
                logger.warning("Temporal error: %s", issue.message)
            for issue in result.warnings:
                logger.debug("Temporal warning: %s", issue.message)
            logger.info(
                "Temporal validation complete: %d errors, %d warnings",
                result.error_count,
                result.warning_count,
            )
        except GenerationCancelledError, DatabaseClosedError:
            raise
        except Exception as e:
            logger.warning("Temporal validation failed (non-fatal): %s", e, exc_info=True)

    # Step 10: Batch-embed all world content for RAG context retrieval
    check_cancelled()
    report_progress("Embedding world content for RAG...")
    try:
        embed_counts = services.embedding.embed_all_world_data(world_db, state)
        logger.info("World embedding complete: %s", embed_counts)
    except GenerationCancelledError, DatabaseClosedError:
        raise
    except Exception as e:
        logger.warning(
            "World embedding failed (non-fatal), RAG context unavailable: %s", e, exc_info=True
        )


def _calculate_total_steps(
    options: WorldBuildOptions,
    *,
    generate_calendar: bool = False,
    validate_temporal: bool = False,
) -> int:
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
    if validate_temporal:
        steps += 1
    logger.debug(
        "_calculate_total_steps: %d steps (calendar=%s, temporal=%s)",
        steps,
        generate_calendar,
        validate_temporal,
    )
    return steps


def _clear_world_db(world_db: WorldDatabase) -> None:
    """Clear all entities, relationships, and events from world database."""
    # Delete events first (they reference entities via participants)
    world_db.clear_events()

    # Delete relationships (they reference entities)
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
    if loc_min > loc_max:
        logger.warning("Invalid location count range: min=%d > max=%d, swapping", loc_min, loc_max)
        loc_min, loc_max = loc_max, loc_min
    location_count = random.randint(loc_min, loc_max)

    existing_names = [e.name for e in world_db.list_entities()]

    location_results = services.world_quality.generate_locations_with_quality(
        state,
        existing_names,
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
                    **build_entity_lifecycle(loc, "location"),
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
    existing_names = [e.name for e in all_entities]
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
    if fac_min > fac_max:
        logger.warning("Invalid faction count range: min=%d > max=%d, swapping", fac_min, fac_max)
        fac_min, fac_max = fac_max, fac_min
    faction_count = random.randint(fac_min, fac_max)

    faction_results = services.world_quality.generate_factions_with_quality(
        state,
        existing_names,
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
    existing_names = [e.name for e in world_db.list_entities()]

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
    if item_min > item_max:
        logger.warning("Invalid item count range: min=%d > max=%d, swapping", item_min, item_max)
        item_min, item_max = item_max, item_min
    item_count = random.randint(item_min, item_max)

    item_results = services.world_quality.generate_items_with_quality(
        state,
        existing_names,
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
    existing_names = [e.name for e in world_db.list_entities()]

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
    if concept_min > concept_max:
        logger.warning(
            "Invalid concept count range: min=%d > max=%d, swapping", concept_min, concept_max
        )
        concept_min, concept_max = concept_max, concept_min
    concept_count = random.randint(concept_min, concept_max)

    concept_results = services.world_quality.generate_concepts_with_quality(
        state,
        existing_names,
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
            logger.warning(f"Skipping invalid concept: {concept}")

    return added_count


def _generate_relationships(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
) -> int:
    """Generate and add relationships between entities using quality refinement.

    Args:
        svc: WorldService instance.
        state: Current story state with brief.
        world_db: World database to read entities from and persist relationships to.
        services: Service container providing the world quality service.
        cancel_check: Optional callable that returns True to stop generation.
        progress_callback: Optional callback receiving EntityGenerationProgress updates.

    Returns:
        Number of relationships successfully added to the world database.
    """
    all_entities = world_db.list_entities()
    entity_names = [e.name for e in all_entities]

    # Map IDs to entity objects so the quality service gets name pairs for duplicate detection
    # 3-tuples: (source_name, target_name, relation_type) for diversity analysis
    entity_by_id = {e.id: e for e in all_entities}
    existing_rels: list[tuple[str, str, str]] = []
    for r in world_db.list_relationships():
        source_entity_obj = entity_by_id.get(r.source_id)
        target_entity_obj = entity_by_id.get(r.target_id)
        if not source_entity_obj or not target_entity_obj:
            logger.warning(
                "Skipping relationship %s -> %s: missing entity reference",
                r.source_id,
                r.target_id,
            )
            continue
        existing_rels.append((source_entity_obj.name, target_entity_obj.name, r.relation_type))

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
        state,
        entity_names,
        existing_rels,
        rel_count,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    added_count = 0
    threshold = svc.settings.fuzzy_match_threshold

    for rel, scores in relationship_results:
        if cancel_check and cancel_check():
            logger.info(f"Relationship processing cancelled after {added_count} relationships")
            break
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            # Find source and target entities (fuzzy match for LLM name variations)
            source_entity = _find_entity_by_name(all_entities, rel["source"], threshold=threshold)
            target_entity = _find_entity_by_name(all_entities, rel["target"], threshold=threshold)

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
                    f"Could not find entities for relationship: "
                    f"{rel.get('source')} -> {rel.get('target')}"
                )
        else:
            logger.warning(f"Skipping invalid relationship: {rel}")

    return added_count
