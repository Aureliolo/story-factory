"""World Builder page - editor operations mixin with entity CRUD and regeneration."""

import logging
from typing import Any

from nicegui import ui

from src.memory.entities import Entity
from src.ui.pages.world._page import WorldPageBase
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger(__name__)


class EditorOpsMixin(WorldPageBase):
    """Mixin providing entity CRUD operations and regeneration for WorldPage."""

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

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states - implemented by UndoMixin."""
        raise NotImplementedError

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns - implemented by base class."""
        raise NotImplementedError
