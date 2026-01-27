"""World Builder page - editor mixin with entity editor methods."""

import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity, EntityVersion
from src.ui.pages.world._page import WorldPageBase
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger(__name__)


class EditorMixin(WorldPageBase):
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

    async def _show_add_dialog(self) -> None:
        """Show dialog to add new entity."""
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
                    on_click=lambda: self._add_entity(
                        dialog, name_input.value, type_select.value, desc_input.value
                    ),
                ).props("color=primary")

        dialog.open()

    def _add_entity(self, dialog: ui.dialog, name: str, entity_type: str, description: str) -> None:
        """Add a new entity."""
        if not name or not self.state.world_db:
            ui.notify("Name is required", type="warning")
            return

        try:
            entity_id = self.services.world.add_entity(
                self.state.world_db,
                entity_type=entity_type,
                name=name,
                description=description,
            )

            # Record action for undo
            self.state.record_action(
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
            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify(f"Added {name}", type="positive")
        except Exception as e:
            logger.exception(f"Failed to add entity {name}")
            ui.notify(f"Error: {e}", type="negative")

    def _collect_attrs_from_form(self, entity_type: str) -> dict[str, Any]:
        """Collect attributes from dynamic form fields.

        Args:
            entity_type: The type of entity being edited (unused, kept for compatibility).

        Returns:
            Dictionary of attributes collected from form fields.
        """
        # Start with existing attrs to preserve fields not shown in form
        attrs = self._entity_attrs.copy() if self._entity_attrs else {}

        # Collect from dynamic attribute inputs
        if hasattr(self, "_dynamic_attr_inputs"):
            for key, (value_type, widget) in self._dynamic_attr_inputs.items():
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

    def _save_entity(self) -> None:
        """Save current entity changes."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get current state for inverse data
            old_entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not old_entity:
                ui.notify("Entity not found", type="negative")
                return

            new_name = self._entity_name_input.value if self._entity_name_input else None
            new_desc = self._entity_desc_input.value if self._entity_desc_input else None
            new_type = (
                self._entity_type_select.value if self._entity_type_select else old_entity.type
            )

            # Collect attributes from form fields
            new_attrs = self._collect_attrs_from_form(new_type)

            self.services.world.update_entity(
                self.state.world_db,
                entity_id=self.state.selected_entity_id,
                name=new_name,
                description=new_desc,
                attributes=new_attrs,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.UPDATE_ENTITY,
                    data={
                        "entity_id": self.state.selected_entity_id,
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

            self._refresh_entity_list()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Entity saved", type="positive")
        except Exception as e:
            logger.exception(f"Failed to save entity {self.state.selected_entity_id}")
            ui.notify(f"Error: {e}", type="negative")

    async def _show_regenerate_dialog(self) -> None:
        """Show dialog for regenerating the selected entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            ui.notify("No entity selected", type="warning")
            return

        entity = self.services.world.get_entity(self.state.world_db, self.state.selected_entity_id)
        if not entity:
            ui.notify("Entity not found", type="negative")
            return

        # Get relationship count
        relationships = self.state.world_db.get_relationships(entity.id)
        rel_count = len(relationships)

        with ui.dialog() as dialog, ui.card().classes("w-96 p-4"):
            ui.label(f"Regenerate: {entity.name}").classes("text-lg font-semibold")

            # Mode selection
            mode_ref = {"value": "refine"}
            with ui.column().classes("w-full gap-2 mt-4"):
                ui.label("Regeneration Mode").classes("text-sm font-medium")
                mode_radio = ui.radio(
                    options={
                        "refine": "Refine existing (improve weak areas)",
                        "full": "Full regenerate (create new version)",
                        "guided": "Regenerate with guidance",
                    },
                    value="refine",
                    on_change=lambda e: mode_ref.update({"value": e.value}),
                ).classes("w-full")

            # Guidance input (visible only for guided mode)
            guidance_container = ui.column().classes("w-full mt-2")
            guidance_input_ref: dict = {"input": None}

            def update_guidance_visibility() -> None:
                """Update guidance input visibility based on mode."""
                guidance_container.clear()
                if mode_ref["value"] == "guided":
                    with guidance_container:
                        guidance_input_ref["input"] = (
                            ui.textarea(
                                label="Guidance",
                                placeholder="Describe how you want this entity to change...",
                            )
                            .classes("w-full")
                            .props("rows=3")
                        )

            mode_radio.on("update:model-value", lambda _: update_guidance_visibility())
            update_guidance_visibility()

            # Relationship warning
            if rel_count > 0:
                with ui.row().classes(
                    "w-full items-center gap-2 p-2 bg-amber-50 dark:bg-amber-900 rounded mt-4"
                ):
                    ui.icon("warning", color="amber")
                    ui.label(f"{rel_count} relationship(s) will be preserved.").classes("text-sm")

            # Quality info if available
            quality_scores = entity.attributes.get("quality_scores") if entity.attributes else None
            if quality_scores and isinstance(quality_scores, dict):
                avg = quality_scores.get("average", 0)
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    ui.label(f"Current quality: {avg:.1f}/10").classes(
                        "text-sm text-gray-500 dark:text-gray-400"
                    )

            # Actions
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                async def do_regenerate() -> None:
                    """Execute regeneration."""
                    guidance = None
                    if guidance_input_ref.get("input"):
                        guidance = guidance_input_ref["input"].value
                    dialog.close()
                    await self._execute_regenerate(entity, mode_ref["value"], guidance)

                ui.button("Regenerate", on_click=do_regenerate).props("color=primary")

        dialog.open()

    async def _execute_regenerate(self, entity: Entity, mode: str, guidance: str | None) -> None:
        """Execute entity regeneration.

        Args:
            entity: Entity to regenerate.
            mode: Regeneration mode (refine, full, guided).
            guidance: Optional guidance text for guided mode.
        """
        logger.info(f"Regenerating entity {entity.id} ({entity.name}) mode={mode}")

        if not self.state.world_db or not self.state.project:
            ui.notify("No project loaded", type="negative")
            return

        # Show progress indicator
        progress_dialog = None
        try:
            with ui.dialog() as progress_dialog, ui.card().classes("w-64 items-center p-4"):
                ui.label("Regenerating...").classes("text-center")
                ui.spinner(size="lg")
            progress_dialog.open()

            # Perform regeneration based on mode
            result = None
            if mode == "refine":
                result = await self._refine_entity(entity)
            elif mode == "guided":
                if guidance and guidance.strip():
                    result = await self._regenerate_with_guidance(entity, guidance)
                else:
                    logger.warning(
                        "Guided regeneration requested for entity %s without guidance; aborting.",
                        entity.id,
                    )
                    if progress_dialog:
                        progress_dialog.close()
                    ui.notify("Guidance text is required for guided regeneration.", type="warning")
                    return
            else:
                result = await self._regenerate_full(entity)

            if result:
                # Update entity in database (relationships preserved since ID unchanged)
                new_name = result.get("name", entity.name)
                new_description = result.get("description", entity.description)
                new_attributes = {**(entity.attributes or {}), **result.get("attributes", {})}

                self.services.world.update_entity(
                    self.state.world_db,
                    entity.id,
                    name=new_name,
                    description=new_description,
                    attributes=new_attributes,
                )

                # Invalidate graph cache for fresh tooltips
                self.state.world_db.invalidate_graph_cache()

                # Refresh UI
                self._refresh_entity_list()
                self._refresh_entity_editor()
                if self._graph:
                    self._graph.refresh()

                ui.notify(f"Regenerated {new_name}", type="positive")
            else:
                ui.notify("Regeneration failed - no result returned", type="negative")

        except Exception as e:
            logger.exception(f"Regeneration failed: {e}")
            ui.notify(f"Error: {e}", type="negative")
        finally:
            if progress_dialog:
                progress_dialog.close()

    async def _refine_entity(self, entity: Entity) -> dict | None:
        """Refine entity using quality service.

        Args:
            entity: Entity to refine.

        Returns:
            Dictionary with refined entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService to refine the entity
            refined = await self.services.world_quality.refine_entity(
                entity=entity,
                story_brief=self.state.project.brief,
            )
            return refined
        except Exception as e:
            logger.exception(f"Failed to refine entity: {e}")
            return None

    async def _regenerate_full(self, entity: Entity) -> dict | None:
        """Fully regenerate entity.

        Args:
            entity: Entity to regenerate.

        Returns:
            Dictionary with new entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService to regenerate based on entity type
            regenerated = await self.services.world_quality.regenerate_entity(
                entity=entity,
                story_brief=self.state.project.brief,
            )
            return regenerated
        except Exception as e:
            logger.exception(f"Failed to regenerate entity: {e}")
            return None

    async def _regenerate_with_guidance(self, entity: Entity, guidance: str) -> dict | None:
        """Regenerate with user guidance.

        Args:
            entity: Entity to regenerate.
            guidance: User-provided guidance text.

        Returns:
            Dictionary with regenerated entity data, or None on failure.
        """
        if not self.state.project:
            return None

        try:
            # Use WorldQualityService with custom instructions
            regenerated = await self.services.world_quality.regenerate_entity(
                entity=entity,
                story_brief=self.state.project.brief,
                custom_instructions=guidance,
            )
            return regenerated
        except Exception as e:
            logger.exception(f"Failed to regenerate entity with guidance: {e}")
            return None

    def _confirm_delete_entity(self) -> None:
        """Show confirmation dialog before deleting entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get entity name for better UX
            entity = self.services.world.get_entity(
                self.state.world_db,
                self.state.selected_entity_id,
            )
            entity_name = entity.name if entity else "this entity"

            # Count attached relationships
            all_rels = self.state.world_db.list_relationships()
            attached_rels = [
                r
                for r in all_rels
                if r.source_id == self.state.selected_entity_id
                or r.target_id == self.state.selected_entity_id
            ]
            rel_count = len(attached_rels)

            # Build message with relationship info
            message = f'Are you sure you want to delete "{entity_name}"?'
            if rel_count > 0:
                rel_word = "relationship" if rel_count == 1 else "relationships"
                message += f"\n\nThis will also delete {rel_count} attached {rel_word}."
            message += "\n\nThis action cannot be undone."

            # Custom dialog with more info
            with ui.dialog() as dialog, ui.card().classes("p-4 min-w-[400px]"):
                ui.label("Delete Entity?").classes("text-lg font-bold text-red-600")
                ui.label(message).classes(
                    "text-gray-600 dark:text-gray-400 whitespace-pre-line mt-2"
                )

                if rel_count > 0:
                    with ui.expansion("Show affected relationships", icon="link").classes(
                        "w-full mt-2"
                    ):
                        for rel in attached_rels[:10]:  # Show max 10
                            source = self.services.world.get_entity(
                                self.state.world_db, rel.source_id
                            )
                            target = self.services.world.get_entity(
                                self.state.world_db, rel.target_id
                            )
                            src_name = source.name if source else "?"
                            tgt_name = target.name if target else "?"
                            ui.label(f"{src_name} → {tgt_name} ({rel.relation_type})").classes(
                                "text-sm text-gray-500"
                            )
                        if rel_count > 10:
                            ui.label(f"... and {rel_count - 10} more").classes(
                                "text-sm text-gray-400 italic"
                            )

                def _do_delete() -> None:
                    """Close the dialog and delete the selected entity."""
                    dialog.close()
                    self._delete_entity()

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button(
                        "Delete",
                        on_click=_do_delete,
                        icon="delete",
                    ).props("color=negative")

            dialog.open()

        except Exception as e:
            logger.exception("Error showing delete confirmation")
            ui.notify(f"Error: {e}", type="negative")

    def _delete_entity(self) -> None:
        """Delete the selected entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        try:
            # Get entity data for inverse (restore) operation
            entity = self.services.world.get_entity(
                self.state.world_db, self.state.selected_entity_id
            )
            if not entity:
                ui.notify("Entity not found", type="negative")
                return

            entity_id = self.state.selected_entity_id

            self.services.world.delete_entity(
                self.state.world_db,
                entity_id,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.DELETE_ENTITY,
                    data={"entity_id": entity_id},
                    inverse_data={
                        "type": entity.type,
                        "name": entity.name,
                        "description": entity.description,
                        "attributes": entity.attributes,
                    },
                )
            )

            self.state.select_entity(None)
            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Entity deleted", type="positive")
        except Exception as e:
            logger.exception(f"Failed to delete entity {self.state.selected_entity_id}")
            ui.notify(f"Error: {e}", type="negative")

    def _add_relationship(self) -> None:
        """Add a new relationship."""
        if not self.state.world_db:
            return

        source_id = self._rel_source_select.value if self._rel_source_select else None
        rel_type = self._rel_type_select.value if self._rel_type_select else None
        target_id = self._rel_target_select.value if self._rel_target_select else None

        if not source_id or not rel_type or not target_id:
            ui.notify("All fields required", type="warning")
            return

        try:
            relationship_id = self.services.world.add_relationship(
                self.state.world_db,
                source_id=source_id,
                target_id=target_id,
                relation_type=rel_type,
            )

            # Record action for undo
            self.state.record_action(
                UndoAction(
                    action_type=ActionType.ADD_RELATIONSHIP,
                    data={
                        "relationship_id": relationship_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                    },
                    inverse_data={
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                    },
                )
            )

            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Relationship added", type="positive")
        except Exception as e:
            logger.exception("Failed to add relationship")
            ui.notify(f"Error: {e}", type="negative")

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states - implemented by UndoMixin."""
        raise NotImplementedError

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns - implemented by base class."""
        raise NotImplementedError
