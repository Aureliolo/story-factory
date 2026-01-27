"""Entity generation operations for the World page.

Contains the main generate_more dispatcher and generators for characters
and locations.  Generators for factions, items, concepts, relationships,
and generate_relationships_for_entities live in _gen_entity_types.py.
"""

import logging
import threading
from typing import Any

from nicegui import ui

from src.ui.pages.world._gen_dialogs import (
    create_progress_dialog,
    get_all_entity_names,
    get_random_count,
    make_update_progress,
    notify_partial_failure,
    prompt_for_relationships_after_add,
    show_entity_preview_dialog,
)
from src.ui.pages.world._gen_entity_types import (
    _generate_concepts,
    _generate_factions,
    _generate_items,
    _generate_relationships,
    generate_relationships_for_entities,  # noqa: F401 — re-exported for __init__
)
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger("src.ui.pages.world._gen_operations")


async def generate_more(
    page, entity_type: str, count: int | None = None, custom_instructions: str | None = None
) -> None:
    """Generate more entities of a specific type.

    Args:
        page: WorldPage instance.
        entity_type: Type of entities to generate.
        count: Number of entities to generate (defaults to random from settings).
        custom_instructions: Optional custom instructions to refine generation.
    """
    logger.info(
        f"Generate more: entity_type={entity_type}, count={count}, "
        f"custom_instructions={custom_instructions[:50] if custom_instructions else None}"
    )

    if not page.state.project or not page.state.world_db:
        logger.warning("Generate more failed: no project or world_db")
        ui.notify("No project loaded", type="negative")
        return

    logger.info(f"Starting generation of {entity_type} for project {page.state.project.id}")

    # Check if quality refinement is enabled
    use_quality = (
        page.state.quality_refinement_enabled and page.services.settings.world_quality_enabled
    )
    logger.info(f"Quality refinement enabled: {use_quality}")

    # Use provided count or get random from settings
    if count is None:
        count = get_random_count(page, entity_type)
    logger.info(f"Will generate {count} {entity_type}")

    # Get ALL existing entity names to avoid duplicates
    all_existing_names = get_all_entity_names(page)
    logger.info(f"Found {len(all_existing_names)} existing entities to avoid duplicates")

    # Create cancellation infrastructure for quality generation
    page._generation_cancel_event = threading.Event()

    def should_cancel() -> bool:
        """Check if generation should be cancelled."""
        return page._generation_cancel_event is not None and page._generation_cancel_event.is_set()

    # Create progress dialog for quality generation, or simple notification for non-quality
    quality_msg = " with quality refinement" if use_quality else ""

    if use_quality:
        progress_label, progress_bar, eta_label = create_progress_dialog(page, entity_type, count)
        update_progress = make_update_progress(progress_label, progress_bar, eta_label)
        notification = None  # No notification when using dialog
    else:
        # Use simple notification for non-quality generation
        notification = ui.notification(
            message=f"Generating {count} {entity_type}{quality_msg}...",
            spinner=True,
            timeout=None,
        )
        update_progress = None
        progress_label = None

    try:
        if entity_type == "characters":
            await _generate_characters(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                progress_label,
                notification,
                custom_instructions,
            )
        elif entity_type == "locations":
            await _generate_locations(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                progress_label,
                notification,
            )
        elif entity_type == "factions":
            await _generate_factions(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                progress_label,
                notification,
            )
        elif entity_type == "items":
            await _generate_items(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                progress_label,
                notification,
            )
        elif entity_type == "concepts":
            await _generate_concepts(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                progress_label,
                notification,
            )
        elif entity_type == "relationships":
            await _generate_relationships(
                page,
                count,
                use_quality,
                all_existing_names,
                should_cancel,
                update_progress,
                notification,
            )

        # Invalidate graph cache to ensure fresh tooltips
        page.state.world_db.invalidate_graph_cache()

        # Refresh the UI
        logger.info("Refreshing UI after generation...")
        page._refresh_entity_list()
        if page._graph:
            page._graph.refresh()

        # Save the project
        if page.state.project:
            logger.info(f"Saving project {page.state.project.id}...")
            page.services.project.save_project(page.state.project)
            logger.info("Project saved successfully")

        logger.info(f"Generation of {entity_type} completed successfully")

    except WorldGenerationError as e:
        if page._generation_dialog:
            page._generation_dialog.close()
        elif notification:
            notification.dismiss()
        logger.error(f"World generation failed for {entity_type}: {e}")
        ui.notify(f"Generation failed: {e}", type="negative", close_button=True, timeout=10)
    except Exception as e:
        if page._generation_dialog:
            page._generation_dialog.close()
        elif notification:
            notification.dismiss()
        logger.exception(f"Unexpected error generating {entity_type}: {e}")
        ui.notify(f"Error: {e}", type="negative")


async def _generate_characters(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    progress_label,
    notification,
    custom_instructions,
) -> None:
    """Generate characters (quality or standard).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        progress_label: Progress label widget.
        notification: Notification widget.
        custom_instructions: Optional custom instructions.
    """
    from nicegui import run

    if use_quality:
        logger.info("Calling world quality service to generate characters...")
        results = await run.io_bound(
            page.services.world_quality.generate_characters_with_quality,
            page.state.project,
            all_existing_names,
            count,
            custom_instructions,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(results)} characters with quality refinement")

        notify_partial_failure(len(results), count, "characters", should_cancel)
        if len(results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any characters", type="negative")
            return

        # Generate mini descriptions for hover tooltips
        if progress_label:
            progress_label.text = "Generating hover summaries..."
        entity_data = [
            {"name": c.name, "type": "character", "description": c.description} for c, _ in results
        ]
        mini_descs = await run.io_bound(
            page.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        if page._generation_dialog:
            page._generation_dialog.close()

        # Define callback to add selected characters
        def add_selected_characters(selected: list[tuple[Any, Any]]) -> None:
            """Add selected characters to the world database and project."""
            if not selected:
                ui.notify("No characters selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
                ui.notify("No project loaded", type="negative")
                return
            added_names = []
            for char, scores in selected:
                attrs = {
                    "role": char.role,
                    "traits": char.personality_traits,
                    "goals": char.goals,
                    "arc": char.arc_notes,
                    "quality_scores": scores.to_dict(),
                }
                if char.name in mini_descs:
                    attrs["mini_description"] = mini_descs[char.name]
                page.services.world.add_entity(
                    page.state.world_db,
                    name=char.name,
                    entity_type="character",
                    description=char.description,
                    attributes=attrs,
                )
                # Also add to story state
                page.state.project.characters.append(char)
                added_names.append(char.name)
            # Refresh UI and save
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {len(selected)} characters (avg quality: {avg_quality:.1f})",
                type="positive",
            )
            prompt_for_relationships_after_add(page, added_names)

        show_entity_preview_dialog(page, "character", results, add_selected_characters)
        return  # Early return - callback handles the rest
    else:
        logger.info("Calling story service to generate characters...")
        new_chars = await run.io_bound(
            page.services.story.generate_more_characters, page.state.project, count
        )
        logger.info(f"Generated {len(new_chars)} characters from LLM")
        for char in new_chars:
            page.services.world.add_entity(
                page.state.world_db,
                name=char.name,
                entity_type="character",
                description=char.description,
                attributes={
                    "role": char.role,
                    "traits": char.personality_traits,
                    "goals": char.goals,
                    "arc": char.arc_notes,
                },
            )
        logger.info(f"Added {len(new_chars)} characters to world database")
        if notification:
            notification.dismiss()
        ui.notify(f"Added {len(new_chars)} new characters!", type="positive")


async def _generate_locations(
    page,
    count,
    use_quality,
    all_existing_names,
    should_cancel,
    update_progress,
    progress_label,
    notification,
) -> None:
    """Generate locations (quality or standard).

    Args:
        page: WorldPage instance.
        count: Number to generate.
        use_quality: Whether quality refinement is enabled.
        all_existing_names: Existing entity names to avoid duplicates.
        should_cancel: Cancel check callable.
        update_progress: Progress update callback.
        progress_label: Progress label widget.
        notification: Notification widget.
    """
    from nicegui import run

    if use_quality:
        logger.info("Calling world quality service to generate locations...")
        loc_results = await run.io_bound(
            page.services.world_quality.generate_locations_with_quality,
            page.state.project,
            all_existing_names,
            count,
            should_cancel,
            update_progress,
        )
        logger.info(f"Generated {len(loc_results)} locations with quality refinement")

        notify_partial_failure(len(loc_results), count, "locations", should_cancel)
        if len(loc_results) == 0:
            if page._generation_dialog:
                page._generation_dialog.close()
            ui.notify("Failed to generate any locations", type="negative")
            return

        if progress_label:
            progress_label.text = "Generating hover summaries..."
        entity_data = [
            {
                "name": loc.get("name", ""),
                "type": "location",
                "description": loc.get("description", ""),
            }
            for loc, _ in loc_results
            if isinstance(loc, dict) and loc.get("name")
        ]
        mini_descs = await run.io_bound(
            page.services.world_quality.generate_mini_descriptions_batch,
            entity_data,
        )
        if page._generation_dialog:
            page._generation_dialog.close()

        def add_selected_locations(selected: list[tuple[Any, Any]]) -> None:
            """Add selected locations to the world database."""
            if not selected:
                ui.notify("No locations selected", type="info")
                return
            if not page.state.world_db or not page.state.project:
                ui.notify("No project loaded", type="negative")
                return
            added_names = []
            for loc, scores in selected:
                if isinstance(loc, dict) and "name" in loc:
                    attrs = {
                        "significance": loc.get("significance", ""),
                        "quality_scores": scores.to_dict(),
                    }
                    if loc["name"] in mini_descs:
                        attrs["mini_description"] = mini_descs[loc["name"]]
                    page.services.world.add_entity(
                        page.state.world_db,
                        name=loc["name"],
                        entity_type="location",
                        description=loc.get("description", ""),
                        attributes=attrs,
                    )
                    added_names.append(loc["name"])
            page.state.world_db.invalidate_graph_cache()
            page._refresh_entity_list()
            if page._graph:
                page._graph.refresh()
            page.services.project.save_project(page.state.project)
            avg_quality = sum(s.average for _, s in selected) / len(selected) if selected else 0
            ui.notify(
                f"Added {len(selected)} locations (avg quality: {avg_quality:.1f})",
                type="positive",
            )
            prompt_for_relationships_after_add(page, added_names)

        show_entity_preview_dialog(page, "location", loc_results, add_selected_locations)
        return
    else:
        logger.info("Calling story service to generate locations...")
        locations = await run.io_bound(
            page.services.story.generate_locations, page.state.project, count
        )
        logger.info(f"Generated {len(locations)} locations from LLM")
        added_count = 0
        for loc in locations:
            if isinstance(loc, dict) and "name" in loc:
                page.services.world.add_entity(
                    page.state.world_db,
                    name=loc["name"],
                    entity_type="location",
                    description=loc.get("description", ""),
                    attributes={"significance": loc.get("significance", "")},
                )
                added_count += 1
            else:
                logger.warning(f"Skipping invalid location: {loc}")
        logger.info(f"Added {added_count} locations to world database")
        if notification:
            notification.dismiss()
        ui.notify(f"Added {added_count} new locations!", type="positive")
