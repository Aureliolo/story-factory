"""Entity editor functions for the World page.

Contains display and form-building functions for the entity editor panel.
Regeneration, mutation, and deletion operations live in _editor_ops.py.
"""

import json
import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity, EntityVersion
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger("src.ui.pages.world._editor")


def build_entity_editor(page) -> None:
    """Build the entity editor panel.

    Args:
        page: WorldPage instance.
    """
    # Import here to avoid circular imports at module level
    from src.ui.pages.world._editor_ops import confirm_delete_entity, show_regenerate_dialog

    with ui.card().classes("w-full h-full"):
        ui.label("Entity Editor").classes("text-lg font-semibold")

        if not page.state.selected_entity_id or not page.state.world_db:
            ui.label("Select an entity to edit").classes(
                "text-gray-500 dark:text-gray-400 text-sm mt-4"
            )
            return

        entity = page.services.world.get_entity(page.state.world_db, page.state.selected_entity_id)
        if not entity:
            ui.label("Entity not found").classes("text-red-500 text-sm")
            return

        # Initialize attrs from entity for saving later
        page._entity_attrs = entity.attributes.copy() if entity.attributes else {}

        # Entity form - common fields
        page._entity_name_input = ui.input(
            label="Name",
            value=entity.name,
        ).classes("w-full")

        page._entity_type_select = ui.select(
            label="Type",
            options=["character", "location", "item", "faction", "concept"],
            value=entity.type,
            on_change=lambda e: refresh_entity_editor(page),
        ).classes("w-full")

        page._entity_desc_input = (
            ui.textarea(
                label="Description",
                value=entity.description,
            )
            .classes("w-full")
            .props("rows=4")
        )

        # Type-specific attribute fields
        build_type_specific_fields(page, entity)

        # Action buttons
        with ui.row().classes("w-full gap-2 mt-4"):
            ui.button(
                "Save",
                on_click=lambda: save_entity(page),
                icon="save",
            ).props("color=primary")

            ui.button(
                "Regenerate",
                on_click=lambda: show_regenerate_dialog(page),
                icon="refresh",
            ).props("outline color=secondary").tooltip("Regenerate this entity with AI")

            ui.button(
                "Delete",
                on_click=lambda: confirm_delete_entity(page),
                icon="delete",
            ).props("color=negative outline")


def build_type_specific_fields(page, entity: Entity) -> None:
    """Build attribute fields dynamically from entity attributes.

    Args:
        page: WorldPage instance.
        entity: Entity to build fields for.
    """
    attrs = entity.attributes or {}

    # Store dynamic attribute inputs for saving
    page._dynamic_attr_inputs: dict[str, Any] = {}

    with ui.expansion("Attributes", icon="list", value=True).classes("w-full"):
        # Skip quality_scores - handled separately below
        skip_keys = {"quality_scores"}

        for key, value in sorted(attrs.items()):
            if key in skip_keys:
                continue

            # Format label nicely
            label = key.replace("_", " ").title()

            # Handle different value types
            if isinstance(value, list):
                # Lists become comma-separated inputs
                value_str = ", ".join(str(v) for v in value)
                input_widget = ui.input(
                    label=f"{label} (comma-separated)",
                    value=value_str,
                ).classes("w-full")
                page._dynamic_attr_inputs[key] = ("list", input_widget)

            elif isinstance(value, dict):
                # Dicts shown as JSON (read-only for complex nested data)
                json_str = json.dumps(value, indent=2)
                ui.label(label).classes("text-sm font-medium mt-2")
                ui.code(json_str, language="json").classes("w-full text-xs")

            elif isinstance(value, bool):
                # Booleans as checkboxes
                checkbox = ui.checkbox(label, value=value).classes("w-full")
                page._dynamic_attr_inputs[key] = ("bool", checkbox)

            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Numbers as number inputs
                number_widget = ui.number(
                    label=label,
                    value=value,
                ).classes("w-full")
                page._dynamic_attr_inputs[key] = ("number", number_widget)

            elif value is None or value == "":
                # Empty values as text inputs
                input_widget = ui.input(
                    label=label,
                    value="",
                ).classes("w-full")
                page._dynamic_attr_inputs[key] = ("str", input_widget)

            else:
                # Strings - use textarea if long, input if short
                str_value = str(value)
                if len(str_value) > 100 or "\n" in str_value:
                    input_widget = (
                        ui.textarea(label=label, value=str_value).classes("w-full").props("rows=3")
                    )
                else:
                    input_widget = ui.input(label=label, value=str_value).classes("w-full")
                page._dynamic_attr_inputs[key] = ("str", input_widget)

        # Add button to add new attribute
        with ui.row().classes("w-full mt-2 gap-2"):
            page._new_attr_key = ui.input(placeholder="New attribute name").classes("flex-1")
            ui.button(
                icon="add",
                on_click=lambda: add_new_attribute(page),
            ).props("flat dense")

        # Show quality scores if present (read-only display)
        quality_scores = attrs.get("quality_scores")
        if quality_scores and isinstance(quality_scores, dict):
            with ui.expansion("Quality Scores", icon="star", value=False).classes("w-full mt-2"):
                avg = quality_scores.get("average", 0)
                ui.label(f"Average: {avg:.1f}/10").classes("font-semibold text-primary")
                for key, value in quality_scores.items():
                    if key not in ("average", "feedback") and isinstance(value, (int, float)):
                        ui.label(f"{key.replace('_', ' ').title()}: {value:.1f}").classes("text-sm")
                feedback = quality_scores.get("feedback", "")
                if feedback:
                    ui.label(f"Feedback: {feedback}").classes("text-xs text-gray-500 mt-2")

        # Version history panel
        build_version_history_panel(page, entity.id)


def add_new_attribute(page) -> None:
    """Add a new attribute to the current entity.

    Args:
        page: WorldPage instance.
    """
    if not hasattr(page, "_new_attr_key") or not page._new_attr_key.value:
        ui.notify("Enter an attribute name", type="warning")
        return

    key = page._new_attr_key.value.strip().lower().replace(" ", "_")
    if not key:
        return

    # Add to entity_attrs for saving
    if key not in page._entity_attrs:
        page._entity_attrs[key] = ""
        ui.notify(f"Added attribute: {key}. Click Save to persist.", type="info")
        refresh_entity_editor(page)
    else:
        ui.notify(f"Attribute '{key}' already exists", type="warning")


def build_version_history_panel(page, entity_id: str) -> None:
    """Build the version history panel for an entity.

    Args:
        page: WorldPage instance.
        entity_id: The entity ID to show version history for.
    """
    if not page.state.world_db:
        return

    # Use configured retention limit instead of hardcoded value
    retention_limit = page.services.settings.entity_version_retention
    versions = page.services.world.get_entity_versions(
        page.state.world_db, entity_id, limit=retention_limit
    )

    if not versions:
        return  # No versions yet, skip showing the panel

    with ui.expansion("Version History", icon="history", value=False).classes("w-full mt-2"):
        ui.label(f"{len(versions)} versions").classes(
            "text-xs text-gray-500 dark:text-gray-400 mb-2"
        )

        for version in versions:
            change_type_icons = {
                "created": "add_circle",
                "refined": "auto_fix_high",
                "edited": "edit",
                "regenerated": "refresh",
            }
            icon = change_type_icons.get(version.change_type, "history")

            with ui.row().classes(
                "w-full items-center gap-2 p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            ):
                ui.icon(icon, size="xs").classes("text-gray-500")

                with ui.column().classes("flex-grow gap-0"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"v{version.version_number}").classes(
                            "text-sm font-mono font-bold"
                        )
                        ui.label(version.change_type).classes(
                            "text-xs px-1 py-0.5 rounded bg-gray-200 dark:bg-gray-600"
                        )
                        if version.quality_score is not None:
                            ui.label(f"{version.quality_score:.1f}").classes(
                                "text-xs text-blue-600 dark:text-blue-400"
                            )

                    # Format timestamp
                    time_str = version.created_at.strftime("%Y-%m-%d %H:%M")
                    ui.label(time_str).classes("text-xs text-gray-500")

                    if version.change_reason:
                        ui.label(version.change_reason).classes("text-xs text-gray-400 italic")

                # Action buttons
                with ui.row().classes("gap-1"):
                    ui.button(
                        icon="visibility",
                        on_click=lambda v=version: show_version_diff(page, v),
                    ).props("flat dense size=xs").tooltip("View changes")

                    # Don't show revert for the latest version
                    if version.version_number < versions[0].version_number:
                        ui.button(
                            icon="restore",
                            on_click=lambda v=version: confirm_revert_version(page, v),
                        ).props("flat dense size=xs color=warning").tooltip(
                            "Revert to this version"
                        )


def confirm_revert_version(page, version: EntityVersion) -> None:
    """Show confirmation dialog for reverting to a version.

    Args:
        page: WorldPage instance.
        version: The version to revert to.
    """
    with ui.dialog() as dialog, ui.card().classes("p-4"):
        ui.label("Revert Entity?").classes("text-lg font-bold mb-2")
        ui.label(
            f"Revert to version {version.version_number} "
            f"({version.change_type} from {version.created_at.strftime('%Y-%m-%d %H:%M')})?"
        ).classes("text-sm text-gray-600 dark:text-gray-400 mb-4")

        ui.label(
            "This will restore the entity's name, type, description, and attributes "
            "to the state captured in this version."
        ).classes("text-xs text-gray-500 mb-4")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Revert",
                on_click=lambda: revert_to_version(page, version, dialog),
            ).props("color=warning")

    dialog.open()


def revert_to_version(page, version: EntityVersion, dialog: ui.dialog) -> None:
    """Execute the revert to a specific version.

    Args:
        page: WorldPage instance.
        version: The version to revert to.
        dialog: The dialog to close after completion.
    """
    if not page.state.world_db:
        return

    try:
        page.services.world.revert_entity_to_version(
            page.state.world_db, version.entity_id, version.version_number
        )
        dialog.close()
        ui.notify(
            f"Reverted to version {version.version_number}",
            type="positive",
        )
        # Refresh list, editor, and graph to show updated data
        page._refresh_entity_list()
        refresh_entity_editor(page)
        if page._graph:
            page._graph.refresh()
    except ValueError as e:
        logger.error(f"Failed to revert: {e}")
        ui.notify(f"Revert failed: {e}", type="negative")


def show_version_diff(page, version: EntityVersion) -> None:
    """Show a dialog comparing a version with current entity state.

    Args:
        page: WorldPage instance.
        version: The version to compare.
    """
    if not page.state.world_db:
        return

    # Get current entity
    entity = page.services.world.get_entity(page.state.world_db, version.entity_id)
    if not entity:
        ui.notify("Entity not found", type="warning")
        return

    version_data = version.data_json
    current_data = {
        "type": entity.type,
        "name": entity.name,
        "description": entity.description,
        "attributes": entity.attributes or {},
    }

    with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[500px] max-w-[800px]"):
        with ui.row().classes("w-full items-center justify-between mb-4"):
            ui.label(f"Version {version.version_number} vs Current").classes("text-lg font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat dense")

        # Show diff for each field
        fields = ["name", "type", "description"]

        for field in fields:
            old_val = version_data.get(field, "")
            new_val = current_data.get(field, "")

            if old_val != new_val:
                ui.label(field.title()).classes("text-sm font-semibold mt-2")
                with ui.row().classes("w-full gap-4"):
                    with ui.column().classes("flex-1 p-2 bg-red-50 dark:bg-red-900/20 rounded"):
                        ui.label("Version").classes("text-xs text-gray-500")
                        ui.label(str(old_val) or "(empty)").classes("text-sm")
                    with ui.column().classes("flex-1 p-2 bg-green-50 dark:bg-green-900/20 rounded"):
                        ui.label("Current").classes("text-xs text-gray-500")
                        ui.label(str(new_val) or "(empty)").classes("text-sm")

        # Compare attributes - cast to dict for mypy
        old_attrs_raw = version_data.get("attributes", {})
        old_attrs = dict(old_attrs_raw) if isinstance(old_attrs_raw, dict) else {}
        new_attrs_raw = current_data["attributes"]
        new_attrs = dict(new_attrs_raw) if isinstance(new_attrs_raw, dict) else {}
        all_attr_keys = set(old_attrs.keys()) | set(new_attrs.keys())

        if old_attrs != new_attrs:
            ui.label("Attributes").classes("text-sm font-semibold mt-4")

            for key in sorted(all_attr_keys):
                attr_old_val: Any = old_attrs.get(key)
                attr_new_val: Any = new_attrs.get(key)

                if attr_old_val != attr_new_val:
                    ui.label(key.replace("_", " ").title()).classes("text-xs text-gray-600 mt-2")
                    with ui.row().classes("w-full gap-4"):
                        with ui.column().classes("flex-1 p-2 bg-red-50 dark:bg-red-900/20 rounded"):
                            old_str = (
                                "(not set)" if attr_old_val is None else str(attr_old_val)[:200]
                            )
                            ui.label(old_str).classes("text-xs")
                        with ui.column().classes(
                            "flex-1 p-2 bg-green-50 dark:bg-green-900/20 rounded"
                        ):
                            new_str = (
                                "(not set)" if attr_new_val is None else str(attr_new_val)[:200]
                            )
                            ui.label(new_str).classes("text-xs")

        # If no differences
        if old_attrs == new_attrs and all(
            version_data.get(f) == current_data.get(f) for f in fields
        ):
            ui.label("No differences from current state").classes(
                "text-sm text-gray-500 italic mt-4"
            )

        with ui.row().classes("w-full justify-end mt-4"):
            ui.button("Close", on_click=dialog.close).props("flat")

    dialog.open()


def refresh_entity_editor(page) -> None:
    """Refresh the entity editor panel with current selection.

    Args:
        page: WorldPage instance.
    """
    if not page._editor_container:
        return

    page._editor_container.clear()
    with page._editor_container:
        build_entity_editor(page)


def collect_attrs_from_form(page, entity_type: str) -> dict[str, Any]:
    """Collect attributes from dynamic form fields.

    Args:
        page: WorldPage instance.
        entity_type: The type of entity being edited (unused, kept for compatibility).

    Returns:
        Dictionary of attributes collected from form fields.
    """
    # Start with existing attrs to preserve fields not shown in form
    attrs = page._entity_attrs.copy() if page._entity_attrs else {}

    # Collect from dynamic attribute inputs
    if hasattr(page, "_dynamic_attr_inputs"):
        for key, (value_type, widget) in page._dynamic_attr_inputs.items():
            if value_type == "list":
                # Parse comma-separated values back to list
                if widget.value:
                    attrs[key] = [v.strip() for v in widget.value.split(",") if v.strip()]
                else:
                    attrs[key] = []
            elif value_type == "bool":
                attrs[key] = widget.value
            elif value_type == "number":
                attrs[key] = widget.value
            else:  # str
                attrs[key] = widget.value

    return attrs


def save_entity(page) -> None:
    """Save current entity changes.

    Args:
        page: WorldPage instance.
    """
    if not page.state.selected_entity_id or not page.state.world_db:
        return

    try:
        # Get current state for inverse data
        old_entity = page.services.world.get_entity(
            page.state.world_db, page.state.selected_entity_id
        )
        if not old_entity:
            ui.notify("Entity not found", type="negative")
            return

        new_name = page._entity_name_input.value if page._entity_name_input else None
        new_desc = page._entity_desc_input.value if page._entity_desc_input else None
        new_type = page._entity_type_select.value if page._entity_type_select else old_entity.type

        # Collect attributes from form fields
        new_attrs = collect_attrs_from_form(page, new_type)

        page.services.world.update_entity(
            page.state.world_db,
            entity_id=page.state.selected_entity_id,
            name=new_name,
            description=new_desc,
            attributes=new_attrs,
        )

        # Record action for undo
        page.state.record_action(
            UndoAction(
                action_type=ActionType.UPDATE_ENTITY,
                data={
                    "entity_id": page.state.selected_entity_id,
                    "name": new_name,
                    "description": new_desc,
                    "attributes": new_attrs,
                },
                inverse_data={
                    "name": old_entity.name,
                    "description": old_entity.description,
                    "attributes": old_entity.attributes,
                },
            )
        )

        page._refresh_entity_list()
        if page._graph:
            page._graph.refresh()
        page._update_undo_redo_buttons()
        ui.notify("Entity saved", type="positive")
    except Exception as e:
        logger.exception(f"Failed to save entity {page.state.selected_entity_id}")
        ui.notify(f"Error: {e}", type="negative")
