"""Entity browser functions for the World page."""

import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity
from src.ui.components.entity_card import entity_list_item
from src.ui.local_prefs import load_prefs_deferred, save_pref
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger(__name__)

_PAGE_KEY = "entity_browser"


def build_entity_browser(page) -> None:
    """Build the entity browser panel.

    Args:
        page: WorldPage instance.
    """

    with ui.card().classes("w-full h-full"):
        ui.label("Entity Browser").classes("text-lg font-semibold")

        # Search with Ctrl+F hint
        page._search_input = (
            ui.input(
                placeholder="Search entities... (Ctrl+F)",
                value=page.state.entity_search_query,
                on_change=lambda e: on_search(page, e),
            )
            .classes("w-full")
            .props("outlined dense clearable")
        )

        # Search scope checkboxes
        with ui.row().classes("w-full gap-4 text-xs items-center"):
            page._search_names_cb = ui.checkbox(
                "Names",
                value=page.state.entity_search_names,
                on_change=lambda e: update_search_scope(page, "names", e.value),
            ).props("dense")
            page._search_desc_cb = ui.checkbox(
                "Descriptions",
                value=page.state.entity_search_descriptions,
                on_change=lambda e: update_search_scope(page, "descriptions", e.value),
            ).props("dense")

        # Filter and sort row
        with ui.row().classes("w-full gap-2 items-center mt-1"):
            # Quality filter
            page._quality_select = (
                ui.select(
                    label="Quality",
                    options={"all": "All", "high": "8+", "medium": "6-8", "low": "<6"},
                    value=page.state.entity_quality_filter,
                    on_change=lambda e: on_quality_filter_change(page, e),
                )
                .classes("w-28")
                .props("dense outlined")
            )

            # Sort dropdown
            page._sort_select = (
                ui.select(
                    label="Sort",
                    options={
                        "name": "Name",
                        "type": "Type",
                        "quality": "Quality",
                        "relationships": "Relationships",
                    },
                    value=page.state.entity_sort_by,
                    on_change=lambda e: on_sort_change(page, e),
                )
                .classes("w-28")
                .props("dense outlined")
            )

            # Sort direction toggle
            page._sort_direction_btn = (
                ui.button(
                    icon="arrow_upward"
                    if not page.state.entity_sort_descending
                    else "arrow_downward",
                    on_click=lambda: toggle_sort_direction(page),
                )
                .props("flat dense")
                .tooltip("Toggle sort direction")
            )

        # Entity list with flexible height to match editor
        page._entity_list = (
            ui.column()
            .classes(
                "w-full gap-1 overflow-auto flex-grow p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
            )
            .style("max-height: calc(100vh - 520px); min-height: 200px")
        )
        refresh_entity_list(page)

        # Add button
        ui.button(
            "+ Add Entity",
            on_click=lambda: show_add_dialog(page),
            icon="add",
        ).props("color=primary").classes("w-full mt-2")

        # Undo/Redo buttons
        with ui.row().classes("w-full gap-2 mt-2"):
            page._undo_btn = (
                ui.button(
                    icon="undo",
                    on_click=page._do_undo,
                )
                .props("flat dense")
                .tooltip("Undo (Ctrl+Z)")
            )
            page._redo_btn = (
                ui.button(
                    icon="redo",
                    on_click=page._do_redo,
                )
                .props("flat dense")
                .tooltip("Redo (Ctrl+Y)")
            )
            page._update_undo_redo_buttons()

        # Register Ctrl+F keyboard shortcut
        register_keyboard_shortcuts(page)

    # Restore persisted preferences from localStorage
    load_prefs_deferred(_PAGE_KEY, lambda prefs: _apply_prefs(page, prefs))


def refresh_entity_list(page) -> None:
    """Refresh the entity list display.

    Args:
        page: WorldPage instance.
    """
    if not page._entity_list or not page.state.world_db:
        return

    page._entity_list.clear()

    entities = page.services.world.list_entities(page.state.world_db)

    # Filter by type
    if page.state.entity_filter_types:
        entities = [e for e in entities if e.type in page.state.entity_filter_types]

    # Filter by search (with scope) - concise list comprehension
    if page.state.entity_search_query:
        query = page.state.entity_search_query.lower()
        entities = [
            e
            for e in entities
            if (page.state.entity_search_names and query in e.name.lower())
            or (page.state.entity_search_descriptions and query in e.description.lower())
        ]

    # Filter by quality
    if page.state.entity_quality_filter != "all":
        entities = filter_by_quality(page, entities)

    # Sort entities
    entities = sort_entities(page, entities)

    # Clear selection if selected entity is filtered out
    if page.state.selected_entity_id:
        visible_ids = {e.id for e in entities}
        if page.state.selected_entity_id not in visible_ids:
            page.state.select_entity(None)
            page._refresh_entity_editor()

    with page._entity_list:
        if not entities:
            # Check if there are entities at all or just filtered out
            all_entities = page.services.world.list_entities(page.state.world_db)
            if not all_entities:
                # No entities exist at all - show guidance
                with ui.column().classes("items-center gap-2 py-4"):
                    ui.icon("group_add", size="md").classes("text-gray-400 dark:text-gray-500")
                    ui.label("No entities yet").classes(
                        "text-gray-500 dark:text-gray-400 font-medium"
                    )
                    ui.label("Add characters, locations, and more").classes(
                        "text-xs text-gray-400 dark:text-gray-500 text-center"
                    )
                    ui.label("using the button below.").classes(
                        "text-xs text-gray-400 dark:text-gray-500 text-center"
                    )
            else:
                # Entities exist but are filtered out
                ui.label("No matching entities").classes("text-gray-500 dark:text-gray-400 text-sm")
                ui.label("Try adjusting filters or search").classes(
                    "text-xs text-gray-400 dark:text-gray-500"
                )
        else:
            for entity in entities:
                entity_list_item(
                    entity=entity,
                    on_select=lambda e: select_entity(page, e),
                    selected=entity.id == page.state.selected_entity_id,
                )


def filter_by_quality(page, entities: list[Entity]) -> list[Entity]:
    """Filter entities by quality score.

    Args:
        page: WorldPage instance.
        entities: List of entities to filter.

    Returns:
        Filtered list based on quality filter setting.
    """
    logger.debug(
        "Filtering %d entities by quality: %s", len(entities), page.state.entity_quality_filter
    )
    result = []
    for entity in entities:
        scores = entity.attributes.get("quality_scores") if entity.attributes else None
        avg = scores.get("average", 0) if scores else 0

        if page.state.entity_quality_filter == "high" and avg >= 8:
            result.append(entity)
        elif page.state.entity_quality_filter == "medium" and 6 <= avg < 8:
            result.append(entity)
        elif page.state.entity_quality_filter == "low" and avg < 6:
            result.append(entity)
    logger.debug("Quality filter returned %d entities", len(result))
    return result


def sort_entities(page, entities: list[Entity]) -> list[Entity]:
    """Sort entities by current sort setting.

    Args:
        page: WorldPage instance.
        entities: List of entities to sort.

    Returns:
        Sorted list based on sort setting.
    """
    sort_key = page.state.entity_sort_by
    descending = page.state.entity_sort_descending
    logger.debug("Sorting %d entities by %s (desc=%s)", len(entities), sort_key, descending)

    if sort_key == "name":
        return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)
    elif sort_key == "type":
        return sorted(entities, key=lambda e: e.type, reverse=descending)
    elif sort_key == "quality":
        return sorted(
            entities,
            key=lambda e: (e.attributes.get("quality_scores") or {}).get("average", 0),
            reverse=descending,
        )
    elif sort_key == "relationships":

        def count_rels(entity: Entity) -> int:
            """Count relationships for sorting."""
            if page.state.world_db:
                count = len(page.state.world_db.get_relationships(entity.id))
                logger.debug("Entity %s has %d relationships", entity.id, count)
                return count
            return 0

        return sorted(entities, key=count_rels, reverse=descending)
    else:
        return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)


def toggle_type_filter(page, entity_type: str, enabled: bool) -> None:
    """Toggle entity type filter.

    Args:
        page: WorldPage instance.
        entity_type: Entity type to toggle.
        enabled: Whether to enable or disable the filter.
    """
    if enabled and entity_type not in page.state.entity_filter_types:
        page.state.entity_filter_types.append(entity_type)
    elif not enabled and entity_type in page.state.entity_filter_types:
        page.state.entity_filter_types.remove(entity_type)

    refresh_entity_list(page)
    if page._graph:
        page._graph.set_filter(page.state.entity_filter_types)


def on_search(page, e: Any) -> None:
    """Handle search input change.

    Args:
        page: WorldPage instance.
        e: Change event.
    """
    page.state.entity_search_query = e.value
    refresh_entity_list(page)

    # Highlight matching nodes in the graph
    if page._graph:
        page._graph.highlight_search(e.value)


def update_search_scope(page, scope: str, enabled: bool) -> None:
    """Update search scope settings.

    Args:
        page: WorldPage instance.
        scope: 'names' or 'descriptions'.
        enabled: Whether this scope is enabled.
    """
    if scope == "names":
        page.state.entity_search_names = enabled
        save_pref(_PAGE_KEY, "entity_search_names", enabled)
    elif scope == "descriptions":
        page.state.entity_search_descriptions = enabled
        save_pref(_PAGE_KEY, "entity_search_descriptions", enabled)
    logger.debug(f"Search scope updated: {scope}={enabled}")
    refresh_entity_list(page)


def on_quality_filter_change(page, e: Any) -> None:
    """Handle quality filter dropdown change.

    Args:
        page: WorldPage instance.
        e: Change event.
    """
    page.state.entity_quality_filter = e.value
    logger.debug(f"Quality filter changed to: {e.value}")
    save_pref(_PAGE_KEY, "entity_quality_filter", e.value)
    refresh_entity_list(page)


def on_sort_change(page, e: Any) -> None:
    """Handle sort dropdown change.

    Args:
        page: WorldPage instance.
        e: Change event.
    """
    page.state.entity_sort_by = e.value
    logger.debug(f"Sort changed to: {e.value}")
    save_pref(_PAGE_KEY, "entity_sort_by", e.value)
    refresh_entity_list(page)


def toggle_sort_direction(page) -> None:
    """Toggle sort direction between ascending and descending.

    Args:
        page: WorldPage instance.
    """
    page.state.entity_sort_descending = not page.state.entity_sort_descending
    logger.debug(
        f"Sort direction: {'descending' if page.state.entity_sort_descending else 'ascending'}"
    )

    # Update button icon
    if page._sort_direction_btn:
        icon = "arrow_downward" if page.state.entity_sort_descending else "arrow_upward"
        page._sort_direction_btn.props(f"icon={icon}")

    save_pref(_PAGE_KEY, "entity_sort_descending", page.state.entity_sort_descending)
    refresh_entity_list(page)


def register_keyboard_shortcuts(page) -> None:
    """Register keyboard shortcuts for the world page.

    Args:
        page: WorldPage instance.
    """
    ui.keyboard(on_key=lambda e: handle_keyboard(page, e))


async def handle_keyboard(page, e: Any) -> None:
    """Handle keyboard events.

    Args:
        page: WorldPage instance.
        e: Keyboard event with key and modifiers.
    """
    # Ctrl+F or Cmd+F focuses the search input (cross-platform)
    ctrl_pressed = getattr(e.modifiers, "ctrl", False)
    meta_pressed = getattr(e.modifiers, "meta", False)
    # e.key is a KeyboardKey object, convert to string for comparison
    key_str = str(e.key).lower() if e.key else ""
    if (ctrl_pressed or meta_pressed) and key_str == "f":
        if page._search_input:
            await page._search_input.run_method("focus")
            await page._search_input.run_method("select")
        return

    # Esc clears the search
    if key_str == "escape" and page._search_input:
        page._search_input.value = ""
        page.state.entity_search_query = ""
        refresh_entity_list(page)
        if page._graph:
            page._graph.highlight_search("")


def select_entity(page, entity: Entity) -> None:
    """Select an entity for editing.

    Args:
        page: WorldPage instance.
        entity: Entity to select.
    """
    page.state.select_entity(entity.id)
    refresh_entity_list(page)
    page._refresh_entity_editor()
    if page._graph:
        page._graph.set_selected(entity.id)


async def show_add_dialog(page) -> None:
    """Show dialog to add new entity.

    Args:
        page: WorldPage instance.
    """
    with ui.dialog() as dialog, ui.card():
        ui.label("Add New Entity").classes("text-lg font-semibold")

        name_input = ui.input(label="Name").classes("w-full")
        type_select = ui.select(
            label="Type",
            options=["character", "location", "item", "faction", "concept"],
            value="character",
        ).classes("w-full")
        desc_input = ui.textarea(label="Description").props("filled").classes("w-full")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Add",
                on_click=lambda: add_entity(
                    page, dialog, name_input.value, type_select.value, desc_input.value
                ),
            ).props("color=primary")

    dialog.open()


def add_entity(page, dialog: ui.dialog, name: str, entity_type: str, description: str) -> None:
    """Add a new entity.

    Args:
        page: WorldPage instance.
        dialog: The dialog to close.
        name: Entity name.
        entity_type: Entity type.
        description: Entity description.
    """
    if not name or not page.state.world_db:
        ui.notify("Name is required", type="warning")
        return

    try:
        entity_id = page.services.world.add_entity(
            page.state.world_db,
            entity_type=entity_type,
            name=name,
            description=description,
        )

        # Record action for undo
        page.state.record_action(
            UndoAction(
                action_type=ActionType.ADD_ENTITY,
                data={
                    "entity_id": entity_id,
                    "type": entity_type,
                    "name": name,
                    "description": description,
                },
                inverse_data={
                    "type": entity_type,
                    "name": name,
                    "description": description,
                },
            )
        )

        dialog.close()
        refresh_entity_list(page)
        if page._graph:
            page._graph.refresh()
        page._update_undo_redo_buttons()
        ui.notify(f"Added {name}", type="positive")
    except Exception as e:
        logger.exception(f"Failed to add entity {name}")
        ui.notify(f"Error: {e}", type="negative")


def _apply_prefs(page, prefs: dict) -> None:
    """Apply loaded preferences to state and UI widgets.

    Validates every field against its allowed values/types before applying.
    Invalid or stale localStorage entries are silently ignored.

    Args:
        page: WorldPage instance.
        prefs: Dict of field→value from localStorage.
    """
    if not prefs:
        return

    _VALID_QUALITY = {"all", "high", "medium", "low"}
    _VALID_SORT = {"name", "type", "quality", "relationships"}

    changed = False

    if "entity_quality_filter" in prefs:
        val = prefs["entity_quality_filter"]
        if (
            isinstance(val, str)
            and val in _VALID_QUALITY
            and val != page.state.entity_quality_filter
        ):
            page.state.entity_quality_filter = val
            changed = True

    if "entity_sort_by" in prefs:
        val = prefs["entity_sort_by"]
        if isinstance(val, str) and val in _VALID_SORT and val != page.state.entity_sort_by:
            page.state.entity_sort_by = val
            changed = True

    if "entity_sort_descending" in prefs:
        val = prefs["entity_sort_descending"]
        if isinstance(val, bool) and val != page.state.entity_sort_descending:
            page.state.entity_sort_descending = val
            changed = True

    if "entity_search_names" in prefs:
        val = prefs["entity_search_names"]
        if isinstance(val, bool) and val != page.state.entity_search_names:
            page.state.entity_search_names = val
            changed = True

    if "entity_search_descriptions" in prefs:
        val = prefs["entity_search_descriptions"]
        if isinstance(val, bool) and val != page.state.entity_search_descriptions:
            page.state.entity_search_descriptions = val
            changed = True

    if changed:
        logger.info("Restored entity browser preferences from localStorage")
        if hasattr(page, "_quality_select") and page._quality_select:
            page._quality_select.value = page.state.entity_quality_filter
        if hasattr(page, "_sort_select") and page._sort_select:
            page._sort_select.value = page.state.entity_sort_by
        if hasattr(page, "_sort_direction_btn") and page._sort_direction_btn:
            icon = "arrow_downward" if page.state.entity_sort_descending else "arrow_upward"
            page._sort_direction_btn.props(f"icon={icon}")
        if hasattr(page, "_search_names_cb") and page._search_names_cb:
            page._search_names_cb.value = page.state.entity_search_names
        if hasattr(page, "_search_desc_cb") and page._search_desc_cb:
            page._search_desc_cb.value = page.state.entity_search_descriptions
        refresh_entity_list(page)
