"""Generation dialog UI functions for the World page."""

import logging
import random
from collections.abc import Callable
from typing import Any

from nicegui import ui

from src.memory.world_quality import RefinementConfig
from src.services.world_quality_service import EntityGenerationProgress

logger = logging.getLogger(__name__)


def notify_partial_failure(
    results_count: int,
    requested_count: int,
    entity_type: str,
    should_cancel: Callable[[], bool],
) -> None:
    """Notify user of partial generation failure with cancel awareness.

    Args:
        results_count: Number of entities successfully generated.
        requested_count: Number of entities originally requested.
        entity_type: Type of entity (e.g., "characters", "locations").
        should_cancel: Callable that returns True if generation was cancelled.
    """
    if results_count >= requested_count:
        return
    failed_count = requested_count - results_count
    if should_cancel():
        ui.notify(
            f"Generation cancelled. Generated {results_count} of {requested_count} {entity_type}.",
            type="info",
            timeout=5000,
        )
    else:
        ui.notify(
            f"ERROR: {failed_count} of {requested_count} {entity_type} FAILED to generate! "
            "Check logs for details.",
            type="negative",
            timeout=10000,
            close_button=True,
        )


def get_all_entity_names(page) -> list[str]:
    """Get all entity names from the world database.

    Args:
        page: WorldPage instance.

    Returns:
        List of all entity names across all types.
    """
    if not page.state.world_db:
        return []
    return [e.name for e in page.state.world_db.list_entities()]


def get_entity_names_by_type(page, entity_type: str) -> list[str]:
    """Get entity names filtered by type.

    Args:
        page: WorldPage instance.
        entity_type: Type to filter by.

    Returns:
        List of entity names of the specified type.
    """
    if not page.state.world_db:
        return []
    return [e.name for e in page.state.world_db.list_entities() if e.type == entity_type]


def get_random_count(page, entity_type: str) -> int:
    """Get a random count for entity generation based on settings.

    Args:
        page: WorldPage instance.
        entity_type: Type of entity.

    Returns:
        Random integer between min and max from settings.
    """
    settings = page.services.settings
    ranges = {
        "characters": (settings.world_gen_characters_min, settings.world_gen_characters_max),
        "locations": (settings.world_gen_locations_min, settings.world_gen_locations_max),
        "factions": (settings.world_gen_factions_min, settings.world_gen_factions_max),
        "items": (settings.world_gen_items_min, settings.world_gen_items_max),
        "concepts": (settings.world_gen_concepts_min, settings.world_gen_concepts_max),
        "events": (settings.world_gen_events_min, settings.world_gen_events_max),
        "relationships": (
            settings.world_gen_relationships_min,
            settings.world_gen_relationships_max,
        ),
    }
    if entity_type not in ranges:
        logger.error("Unknown entity_type '%s' in get_random_count", entity_type)
        raise ValueError(f"Unknown entity_type: {entity_type}")
    min_val, max_val = ranges[entity_type]
    return random.randint(min_val, max_val)


def show_generate_dialog(page, entity_type: str) -> None:
    """Show dialog for generating entities with count and custom prompt options.

    Args:
        page: WorldPage instance.
        entity_type: Type of entities to generate.
    """
    logger.info(f"Showing generate dialog for {entity_type}")

    # Get default count range from settings
    settings = page.services.settings
    ranges = {
        "characters": (settings.world_gen_characters_min, settings.world_gen_characters_max),
        "locations": (settings.world_gen_locations_min, settings.world_gen_locations_max),
        "factions": (settings.world_gen_factions_min, settings.world_gen_factions_max),
        "items": (settings.world_gen_items_min, settings.world_gen_items_max),
        "concepts": (settings.world_gen_concepts_min, settings.world_gen_concepts_max),
        "events": (settings.world_gen_events_min, settings.world_gen_events_max),
        "relationships": (
            settings.world_gen_relationships_min,
            settings.world_gen_relationships_max,
        ),
    }
    if entity_type not in ranges:
        logger.error("Unknown entity_type '%s' in show_generate_dialog", entity_type)
        raise ValueError(f"Unknown entity_type: {entity_type}")
    min_val, max_val = ranges[entity_type]
    default_count = (min_val + max_val) // 2

    # Pretty names for display
    type_names = {
        "characters": "Characters",
        "locations": "Locations",
        "factions": "Factions",
        "items": "Items",
        "concepts": "Concepts",
        "events": "Events",
        "relationships": "Relationships",
    }
    type_name = type_names.get(entity_type, entity_type.title())

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[450px] bg-gray-800"),
    ):
        ui.label(f"Generate {type_name}").classes("text-xl font-bold mb-4")

        # Count input
        with ui.card().classes("w-full mb-4 p-3 bg-gray-700"):
            ui.label("How many to generate?").classes("font-medium mb-2")
            count_input = (
                ui.number(
                    value=default_count,
                    min=1,
                    max=20,
                    step=1,
                )
                .props("outlined dense")
                .classes("w-24")
            )
            ui.label(f"Default range: {min_val}-{max_val}").classes("text-xs text-gray-500 mt-1")

        # Custom instructions textarea
        with ui.card().classes("w-full mb-4 p-3 bg-gray-700"):
            ui.label("Custom Instructions (optional)").classes("font-medium mb-2")
            custom_prompt = (
                ui.textarea(
                    placeholder=f"Describe specific {entity_type} you want...\n"
                    f"e.g., 'A mysterious mentor character' or 'A haunted location'",
                )
                .props("outlined")
                .classes("w-full")
            )
            ui.label("The AI will use these instructions to refine the generation").classes(
                "text-xs text-gray-500 mt-1"
            )

        # Buttons
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            async def do_generate() -> None:
                """Close the dialog and trigger entity generation."""
                count = int(count_input.value) if count_input.value else default_count
                custom = custom_prompt.value.strip() if custom_prompt.value else None
                dialog.close()
                await page._generate_more(entity_type, count=count, custom_instructions=custom)

            ui.button("Generate", on_click=do_generate).props("color=primary")

    dialog.open()


def show_quality_settings_dialog(page) -> None:
    """Show dialog to configure quality refinement settings.

    Args:
        page: WorldPage instance.
    """
    settings = page.services.settings
    config = RefinementConfig.from_settings(settings)

    with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
        ui.label("Quality Refinement Settings").classes("text-lg font-bold mb-4")

        # Quality Threshold with reactive value display
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ui.label("Quality Threshold").classes("text-sm font-medium flex-grow")
            threshold_value_label = ui.label(f"{config.quality_threshold:.1f}").classes(
                "text-sm font-bold text-primary"
            )
        threshold_slider = ui.slider(
            min=0.0,
            max=10.0,
            step=0.5,
            value=config.quality_threshold,
            on_change=lambda e: threshold_value_label.set_text(f"{e.value:.1f}"),
        ).classes("w-full")
        with ui.row().classes("w-full justify-between text-xs text-gray-500"):
            ui.label("0 (Accept all)")
            ui.label("10 (Very strict)")

        # Max Iterations with reactive value display
        with ui.row().classes("w-full items-center gap-2 mt-4"):
            ui.label("Max Iterations").classes("text-sm font-medium flex-grow")
            iterations_value_label = ui.label(f"{config.max_iterations}").classes(
                "text-sm font-bold text-primary"
            )
        iterations_slider = ui.slider(
            min=1,
            max=10,
            step=1,
            value=config.max_iterations,
            on_change=lambda e: iterations_value_label.set_text(f"{int(e.value)}"),
        ).classes("w-full")
        with ui.row().classes("w-full justify-between text-xs text-gray-500"):
            ui.label("1 (No refinement)")
            ui.label("10 (Max)")

        with ui.expansion("Advanced", icon="tune").classes("w-full mt-4"):
            # Creator Temperature with reactive display
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.label("Creator Temperature").classes("text-sm font-medium flex-grow")
                creator_value_label = ui.label(f"{config.creator_temperature:.1f}").classes(
                    "text-sm font-bold text-primary"
                )
            creator_temp = ui.slider(
                min=0.0,
                max=2.0,
                step=0.1,
                value=config.creator_temperature,
                on_change=lambda e: creator_value_label.set_text(f"{e.value:.1f}"),
            ).classes("w-full")
            ui.label("Higher = more creative").classes("text-xs text-gray-500")

            # Judge Temperature with reactive display
            with ui.row().classes("w-full items-center gap-2 mt-4"):
                ui.label("Judge Temperature").classes("text-sm font-medium flex-grow")
                judge_value_label = ui.label(f"{config.judge_temperature:.1f}").classes(
                    "text-sm font-bold text-primary"
                )
            judge_temp = ui.slider(
                min=0.0,
                max=2.0,
                step=0.1,
                value=config.judge_temperature,
                on_change=lambda e: judge_value_label.set_text(f"{e.value:.1f}"),
            ).classes("w-full")
            ui.label("Lower = more consistent").classes("text-xs text-gray-500")

            # Refinement Temperature with reactive display
            with ui.row().classes("w-full items-center gap-2 mt-4"):
                ui.label("Refinement Temperature").classes("text-sm font-medium flex-grow")
                refine_value_label = ui.label(f"{config.refinement_temperature:.1f}").classes(
                    "text-sm font-bold text-primary"
                )
            refine_temp = ui.slider(
                min=0.0,
                max=2.0,
                step=0.1,
                value=config.refinement_temperature,
                on_change=lambda e: refine_value_label.set_text(f"{e.value:.1f}"),
            ).classes("w-full")
            ui.label("Balanced creativity and consistency").classes("text-xs text-gray-500")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def save_settings() -> None:
                """Save quality settings to application settings."""
                settings.world_quality_threshold = threshold_slider.value
                settings.world_quality_max_iterations = int(iterations_slider.value)
                settings.world_quality_creator_temp = creator_temp.value
                settings.world_quality_judge_temp = judge_temp.value
                settings.world_quality_refinement_temp = refine_temp.value
                settings.save()
                dialog.close()
                ui.notify("Quality settings saved", type="positive")

            ui.button("Save", on_click=save_settings).props("color=primary")

    dialog.open()


def show_entity_preview_dialog(
    page,
    entity_type: str,
    entities: list[tuple[Any, Any]],
    on_confirm: Any,
) -> None:
    """Show a preview dialog for generated entities before adding them.

    Args:
        page: WorldPage instance.
        entity_type: Type of entity (character, location, faction, item, concept)
        entities: List of (entity_data, quality_scores) tuples
        on_confirm: Callback function that receives selected entities list
    """
    if not entities:
        ui.notify("No entities to preview", type="warning")
        return

    logger.info(f"Showing preview dialog for {len(entities)} {entity_type}(s)")

    # Track selected entities
    selected = dict.fromkeys(range(len(entities)), True)  # All selected by default

    def toggle_selection(idx: int) -> None:
        """Toggle the selection state of an entity at the given index."""
        selected[idx] = not selected[idx]
        logger.debug(f"Toggled entity {idx}: {selected[idx]}")

    def confirm_selection() -> None:
        """Confirm the selection and invoke the callback with selected entities."""
        selected_entities = [entities[i] for i in range(len(entities)) if selected[i]]
        logger.info(f"User confirmed {len(selected_entities)} of {len(entities)} {entity_type}(s)")
        dialog.close()
        on_confirm(selected_entities)

    def cancel_selection() -> None:
        """Cancel the selection and close the preview dialog."""
        logger.info(f"User cancelled {entity_type} preview")
        dialog.close()
        ui.notify(f"Cancelled adding {entity_type}s", type="info")

    with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[500px] max-w-[700px]"):
        ui.label(f"Preview Generated {entity_type.title()}s").classes("text-lg font-bold mb-2")
        ui.label(
            f"Select which {entity_type}s to add to your world. Uncheck any you don't want."
        ).classes("text-gray-400 mb-4")

        # Scrollable container for entities
        with ui.scroll_area().classes("w-full max-h-[400px]"):
            for idx, (entity_data, scores) in enumerate(entities):
                # Get entity name and description based on type
                if entity_type == "character":
                    name = entity_data.name if hasattr(entity_data, "name") else str(entity_data)
                    desc = (
                        entity_data.description[:150] + "..."
                        if hasattr(entity_data, "description")
                        and len(entity_data.description) > 150
                        else getattr(entity_data, "description", "")
                    )
                    role = getattr(entity_data, "role", "")
                    extra = f" ({role})" if role else ""
                elif entity_type == "event":
                    # Events use description instead of name
                    desc_full = (
                        entity_data.get("description", "Unknown")
                        if isinstance(entity_data, dict)
                        else str(entity_data)
                    )
                    name = desc_full[:80] + ("..." if len(desc_full) > 80 else "")
                    desc = desc_full[:150] + ("..." if len(desc_full) > 150 else "")
                    year = entity_data.get("year") if isinstance(entity_data, dict) else None
                    era = entity_data.get("era_name", "") if isinstance(entity_data, dict) else ""
                    extra_parts: list[str] = []
                    if year is not None:
                        extra_parts.append(f"Year {year}")
                    if era:
                        extra_parts.append(era)
                    extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
                elif entity_type == "relationship":
                    # Relationships show source -> relation_type -> target
                    source = (
                        entity_data.get("source", "?") if isinstance(entity_data, dict) else "?"
                    )
                    target = (
                        entity_data.get("target", "?") if isinstance(entity_data, dict) else "?"
                    )
                    rel_type = (
                        entity_data.get("relation_type", "related_to")
                        if isinstance(entity_data, dict)
                        else "related_to"
                    )
                    name = f"{source} → {rel_type} → {target}"
                    desc = (
                        entity_data.get("description", "")[:150]
                        if isinstance(entity_data, dict)
                        else ""
                    )
                    if (
                        len(entity_data.get("description", "")) > 150
                        if isinstance(entity_data, dict)
                        else False
                    ):
                        desc += "..."
                    extra = ""
                else:
                    name = (
                        entity_data.get("name", "Unknown")
                        if isinstance(entity_data, dict)
                        else str(entity_data)
                    )
                    desc = (
                        entity_data.get("description", "")[:150]
                        if isinstance(entity_data, dict)
                        else ""
                    )
                    if (
                        len(entity_data.get("description", "")) > 150
                        if isinstance(entity_data, dict)
                        else False
                    ):
                        desc += "..."
                    extra = ""
                    if entity_type == "faction":
                        base = (
                            entity_data.get("base_location", "")
                            if isinstance(entity_data, dict)
                            else ""
                        )
                        if base:
                            extra = f" (based in: {base})"

                quality = (
                    f" - Quality: {scores.average:.1f}"
                    if scores and hasattr(scores, "average")
                    else ""
                )

                with ui.row().classes("w-full items-start gap-2 py-2 border-b border-gray-700"):
                    ui.checkbox(value=True, on_change=lambda _, i=idx: toggle_selection(i)).classes(
                        "mt-1"
                    )
                    with ui.column().classes("flex-1"):
                        ui.label(f"{name}{extra}").classes("font-semibold")
                        if desc:
                            ui.label(desc).classes("text-sm text-gray-400")
                        if quality:
                            ui.label(quality).classes("text-xs text-blue-400")

        # Helper functions for select/deselect all
        def select_all() -> None:
            """Select all entities and refresh the preview dialog."""
            selected.update(dict.fromkeys(range(len(entities)), True))
            dialog.close()
            show_entity_preview_dialog(page, entity_type, entities, on_confirm)

        def deselect_all() -> None:
            """Deselect all entities and refresh the preview dialog."""
            selected.update(dict.fromkeys(range(len(entities)), False))
            dialog.close()
            show_entity_preview_dialog(page, entity_type, entities, on_confirm)

        # Action buttons
        with ui.row().classes("w-full justify-between mt-4"):
            with ui.row().classes("gap-2"):
                ui.button("Select All", on_click=select_all).props("flat dense")
                ui.button("Deselect All", on_click=deselect_all).props("flat dense")
            with ui.row().classes("gap-2"):
                ui.button("Cancel", on_click=cancel_selection).props("flat")
                ui.button(
                    "Add Selected",
                    on_click=confirm_selection,
                    icon="add",
                ).props("color=primary")

    dialog.open()


def prompt_for_relationships_after_add(page, entity_names: list[str]) -> None:
    """Prompt user to generate relationships for newly added entities.

    Args:
        page: WorldPage instance.
        entity_names: Names of the newly added entities.
    """
    if not entity_names or not page.state.project or not page.state.world_db:
        return

    logger.info(f"Prompting for relationships for {len(entity_names)} new entities")

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[450px] bg-gray-800"),
    ):
        ui.label("Generate Relationships?").classes("text-xl font-bold mb-2")
        ui.label(
            f"Would you like to generate relationships for the {len(entity_names)} "
            "newly added entities?"
        ).classes("text-gray-400 mb-4")

        with ui.card().classes("w-full mb-4 p-3 bg-gray-700"):
            ui.label("New entities:").classes("font-medium mb-2")
            for name in entity_names[:5]:  # Show first 5
                ui.label(f"• {name}").classes("text-sm")
            if len(entity_names) > 5:
                ui.label(f"... and {len(entity_names) - 5} more").classes("text-sm text-gray-500")

        # Relationships per entity input
        ui.label("Relationships per entity:").classes("text-sm mb-1")
        rel_count = (
            ui.number(value=2, min=1, max=5, step=1).props("dense outlined").classes("w-20 mb-4")
        )

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Skip", on_click=dialog.close).props("flat")

            async def do_generate_relationships() -> None:
                """Close the dialog and generate relationships for the new entities."""
                count = int(rel_count.value) if rel_count.value else 2
                dialog.close()
                await page._generate_relationships_for_entities(entity_names, count)

            ui.button("Generate Relationships", on_click=do_generate_relationships).props(
                "color=primary"
            )

    dialog.open()


def create_progress_dialog(
    page, entity_type: str, count: int
) -> tuple[ui.label | None, ui.linear_progress | None, ui.label | None]:
    """Create a progress dialog for quality generation.

    Args:
        page: WorldPage instance.
        entity_type: Type of entity being generated.
        count: Number of entities to generate.

    Returns:
        Tuple of (progress_label, progress_bar, eta_label).
    """
    page._generation_dialog = ui.dialog().props("persistent")
    progress_label: ui.label | None = None
    progress_bar: ui.linear_progress | None = None
    eta_label: ui.label | None = None

    with page._generation_dialog, ui.card().classes("w-96 p-4"):
        ui.label(f"Generating {entity_type.title()}").classes("text-lg font-bold")
        progress_label = ui.label(f"Starting generation of {count} {entity_type}...")
        progress_bar = ui.linear_progress(value=0).classes("w-full my-2")
        eta_label = ui.label("Calculating...").classes("text-sm text-gray-500")

        def do_cancel() -> None:
            """Handle cancel button click."""
            logger.info(f"User requested cancellation of {entity_type} generation")
            if page._generation_cancel_event:
                page._generation_cancel_event.set()

        cancel_btn = ui.button("Cancel", on_click=do_cancel).props("flat color=negative")

        # Attach cancel button disable to cancel event
        def _on_cancel_click() -> None:
            """Disable cancel button and update label on click."""
            cancel_btn.disable()
            if progress_label:
                progress_label.text = f"Cancelling after current {entity_type[:-1]}..."

        cancel_btn.on("click", _on_cancel_click)

    page._generation_dialog.open()

    return progress_label, progress_bar, eta_label


def make_update_progress(
    progress_label: ui.label | None,
    progress_bar: ui.linear_progress | None,
    eta_label: ui.label | None,
) -> Callable[[EntityGenerationProgress], None]:
    """Create a progress update callback.

    Args:
        progress_label: Label to update with progress text.
        progress_bar: Progress bar to update.
        eta_label: Label for estimated time remaining.

    Returns:
        Callback function for progress updates.
    """

    def update_progress(progress: EntityGenerationProgress) -> None:
        """Update dialog with generation progress."""
        try:
            if progress_label:
                if progress.entity_name:
                    progress_label.text = f"Generated: {progress.entity_name}"
                else:
                    progress_label.text = (
                        f"Generating {progress.entity_type} {progress.current}/{progress.total}..."
                    )

            if progress_bar:
                progress_bar.value = progress.progress_fraction

            if eta_label:
                if progress.estimated_remaining_seconds is not None:
                    total_secs = int(progress.estimated_remaining_seconds)
                    if total_secs >= 3600:
                        hours, remainder = divmod(total_secs, 3600)
                        mins, secs = divmod(remainder, 60)
                        eta_label.text = f"~{hours}:{mins:02d}:{secs:02d} remaining"
                    else:
                        mins, secs = divmod(total_secs, 60)
                        eta_label.text = f"~{mins}:{secs:02d} remaining"
                elif progress.current > 1:
                    eta_label.text = "Calculating..."
        except RuntimeError:
            logger.debug("Progress update skipped: UI element destroyed")

    return update_progress
