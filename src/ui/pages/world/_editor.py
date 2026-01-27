"""World Builder page - editor mixin with entity editor methods."""

import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity, EntityVersion
from src.ui.pages.world._editor_ops import EditorOpsMixin
from src.ui.pages.world._page import WorldPageBase

logger = logging.getLogger(__name__)


class EditorMixin(EditorOpsMixin, WorldPageBase):
    """Mixin providing entity editor methods for WorldPage."""

    def _build_entity_editor(self) -> None:
        """Build the entity editor panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Editor").classes("text-lg font-semibold")

            if not self.state.selected_entity_id or not self.state.world_db:
                ui.label("Select an entity to edit").classes(
                    "text-gray-500 dark:text-gray-400 text-sm mt-4"
                )
                return

            entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not entity:
                ui.label("Entity not found").classes("text-red-500 text-sm")
                return

            # Initialize attrs from entity for saving later
            self._entity_attrs = entity.attributes.copy() if entity.attributes else {}

            # Entity form - common fields
            self._entity_name_input = ui.input(
                label="Name",
                value=entity.name,
            ).classes("w-full")

            self._entity_type_select = ui.select(
                label="Type",
                options=["character", "location", "item", "faction", "concept"],
                value=entity.type,
                on_change=lambda e: self._refresh_entity_editor(),
            ).classes("w-full")

            self._entity_desc_input = (
                ui.textarea(
                    label="Description",
                    value=entity.description,
                )
                .classes("w-full")
                .props("rows=4")
            )

            # Type-specific attribute fields
            self._build_type_specific_fields(entity)

            # Action buttons
            with ui.row().classes("w-full gap-2 mt-4"):
                ui.button(
                    "Save",
                    on_click=self._save_entity,
                    icon="save",
                ).props("color=primary")

                ui.button(
                    "Regenerate",
                    on_click=self._show_regenerate_dialog,
                    icon="refresh",
                ).props("outline color=secondary").tooltip("Regenerate this entity with AI")

                ui.button(
                    "Delete",
                    on_click=self._confirm_delete_entity,
                    icon="delete",
                ).props("color=negative outline")

    def _build_type_specific_fields(self, entity: Entity) -> None:
        """Build attribute fields dynamically from entity attributes."""
        attrs = entity.attributes or {}

        # Store dynamic attribute inputs for saving
        self._dynamic_attr_inputs: dict[str, Any] = {}

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
                    self._dynamic_attr_inputs[key] = ("list", input_widget)

                elif isinstance(value, dict):
                    # Dicts shown as JSON (read-only for complex nested data)
                    import json

                    json_str = json.dumps(value, indent=2)
                    ui.label(label).classes("text-sm font-medium mt-2")
                    ui.code(json_str, language="json").classes("w-full text-xs")

                elif isinstance(value, bool):
                    # Booleans as checkboxes
                    checkbox = ui.checkbox(label, value=value).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("bool", checkbox)

                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Numbers as number inputs
                    number_widget = ui.number(
                        label=label,
                        value=value,
                    ).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("number", number_widget)

                elif value is None or value == "":
                    # Empty values as text inputs
                    input_widget = ui.input(
                        label=label,
                        value="",
                    ).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("str", input_widget)

                else:
                    # Strings - use textarea if long, input if short
                    str_value = str(value)
                    if len(str_value) > 100 or "\n" in str_value:
                        input_widget = (
                            ui.textarea(label=label, value=str_value)
                            .classes("w-full")
                            .props("rows=3")
                        )
                    else:
                        input_widget = ui.input(label=label, value=str_value).classes("w-full")
                    self._dynamic_attr_inputs[key] = ("str", input_widget)

            # Add button to add new attribute
            with ui.row().classes("w-full mt-2 gap-2"):
                self._new_attr_key = ui.input(placeholder="New attribute name").classes("flex-1")
                ui.button(
                    icon="add",
                    on_click=self._add_new_attribute,
                ).props("flat dense")

            # Show quality scores if present (read-only display)
            quality_scores = attrs.get("quality_scores")
            if quality_scores and isinstance(quality_scores, dict):
                with ui.expansion("Quality Scores", icon="star", value=False).classes(
                    "w-full mt-2"
                ):
                    avg = quality_scores.get("average", 0)
                    ui.label(f"Average: {avg:.1f}/10").classes("font-semibold text-primary")
                    for key, value in quality_scores.items():
                        if key not in ("average", "feedback") and isinstance(value, (int, float)):
                            ui.label(f"{key.replace('_', ' ').title()}: {value:.1f}").classes(
                                "text-sm"
                            )
                    feedback = quality_scores.get("feedback", "")
                    if feedback:
                        ui.label(f"Feedback: {feedback}").classes("text-xs text-gray-500 mt-2")

            # Version history panel
            self._build_version_history_panel(entity.id)

    def _add_new_attribute(self) -> None:
        """Add a new attribute to the current entity."""
        if not hasattr(self, "_new_attr_key") or not self._new_attr_key.value:
            ui.notify("Enter an attribute name", type="warning")
            return

        key = self._new_attr_key.value.strip().lower().replace(" ", "_")
        if not key:
            return

        # Add to entity_attrs for saving
        if key not in self._entity_attrs:
            self._entity_attrs[key] = ""
            ui.notify(f"Added attribute: {key}. Click Save to persist.", type="info")
            self._refresh_entity_editor()
        else:
            ui.notify(f"Attribute '{key}' already exists", type="warning")

    def _build_version_history_panel(self, entity_id: str) -> None:
        """Build the version history panel for an entity.

        Args:
            entity_id: The entity ID to show version history for.
        """
        if not self.state.world_db:
            return

        # Use configured retention limit instead of hardcoded value
        retention_limit = self.services.settings.entity_version_retention
        versions = self.services.world.get_entity_versions(
            self.state.world_db, entity_id, limit=retention_limit
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
                            on_click=lambda v=version: self._show_version_diff(v),
                        ).props("flat dense size=xs").tooltip("View changes")

                        # Don't show revert for the latest version
                        if version.version_number < versions[0].version_number:
                            ui.button(
                                icon="restore",
                                on_click=lambda v=version: self._confirm_revert_version(v),
                            ).props("flat dense size=xs color=warning").tooltip(
                                "Revert to this version"
                            )

    def _confirm_revert_version(self, version: EntityVersion) -> None:
        """Show confirmation dialog for reverting to a version.

        Args:
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
                    on_click=lambda: self._revert_to_version(version, dialog),
                ).props("color=warning")

        dialog.open()

    def _revert_to_version(self, version: EntityVersion, dialog: ui.dialog) -> None:
        """Execute the revert to a specific version.

        Args:
            version: The version to revert to.
            dialog: The dialog to close after completion.
        """
        if not self.state.world_db:
            return

        try:
            self.services.world.revert_entity_to_version(
                self.state.world_db, version.entity_id, version.version_number
            )
            dialog.close()
            ui.notify(
                f"Reverted to version {version.version_number}",
                type="positive",
            )
            # Refresh list, editor, and graph to show updated data
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
        except ValueError as e:
            logger.error(f"Failed to revert: {e}")
            ui.notify(f"Revert failed: {e}", type="negative")

    def _show_version_diff(self, version: EntityVersion) -> None:
        """Show a dialog comparing a version with current entity state.

        Args:
            version: The version to compare.
        """
        if not self.state.world_db:
            return

        # Get current entity
        entity = self.services.world.get_entity(self.state.world_db, version.entity_id)
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
                ui.label(f"Version {version.version_number} vs Current").classes(
                    "text-lg font-bold"
                )
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
                        with ui.column().classes(
                            "flex-1 p-2 bg-green-50 dark:bg-green-900/20 rounded"
                        ):
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
                        ui.label(key.replace("_", " ").title()).classes(
                            "text-xs text-gray-600 mt-2"
                        )
                        with ui.row().classes("w-full gap-4"):
                            with ui.column().classes(
                                "flex-1 p-2 bg-red-50 dark:bg-red-900/20 rounded"
                            ):
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

    def _build_relationships_section(self) -> None:
        """Build the relationships management section."""
        with ui.expansion("Relationships", icon="link", value=False).classes("w-full"):
            # Add relationship form
            with ui.row().classes("w-full items-end gap-4 mb-4"):
                entities = self._get_entity_options()

                self._rel_source_select = ui.select(
                    label="From",
                    options=entities,
                ).classes("w-48")

                self._rel_type_select = ui.select(
                    label="Relationship",
                    options=[
                        "knows",
                        "loves",
                        "hates",
                        "located_in",
                        "owns",
                        "member_of",
                        "enemy_of",
                        "ally_of",
                        "parent_of",
                        "child_of",
                    ],
                    new_value_mode="add",
                ).classes("w-40")

                self._rel_target_select = ui.select(
                    label="To",
                    options=entities,
                ).classes("w-48")

                ui.button(
                    "Add",
                    on_click=self._add_relationship,
                    icon="add",
                ).props("color=primary")

            # Relationships table
            if self.state.world_db:
                relationships = self.services.world.get_relationships(self.state.world_db)

                if relationships:
                    columns = [
                        {"name": "from", "label": "From", "field": "from", "align": "left"},
                        {"name": "type", "label": "Type", "field": "type", "align": "center"},
                        {"name": "to", "label": "To", "field": "to", "align": "left"},
                        {"name": "actions", "label": "", "field": "actions", "align": "right"},
                    ]

                    rows = []
                    for rel in relationships:
                        source = self.services.world.get_entity(self.state.world_db, rel.source_id)
                        target = self.services.world.get_entity(self.state.world_db, rel.target_id)
                        rows.append(
                            {
                                "id": rel.id,
                                "from": source.name if source else "Unknown",
                                "type": rel.relation_type,
                                "to": target.name if target else "Unknown",
                            }
                        )

                    ui.table(columns=columns, rows=rows).classes("w-full")
                else:
                    ui.label("No relationships yet").classes("text-gray-500 dark:text-gray-400")

    def _select_entity(self, entity: Entity) -> None:
        """Select an entity for editing."""
        self.state.select_entity(entity.id)
        self._refresh_entity_list()
        self._refresh_entity_editor()
        if self._graph:
            self._graph.set_selected(entity.id)

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel with current selection."""
        if not self._editor_container:
            return

        self._editor_container.clear()
        with self._editor_container:
            self._build_entity_editor()
