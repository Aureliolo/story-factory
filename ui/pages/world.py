"""World Builder page - entity and relationship management."""

import logging

from nicegui import ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.html import Html
from nicegui.elements.input import Input
from nicegui.elements.json_editor import JsonEditor
from nicegui.elements.select import Select
from nicegui.elements.textarea import Textarea

from memory.entities import Entity
from services import ServiceContainer
from ui.components.entity_card import entity_list_item
from ui.components.graph import GraphComponent
from ui.graph_renderer import (
    render_centrality_result,
    render_communities_result,
    render_path_result,
)
from ui.state import ActionType, AppState, UndoAction
from ui.theme import ENTITY_COLORS

logger = logging.getLogger(__name__)


class WorldPage:
    """World Builder page for managing entities and relationships.

    Features:
    - Interactive graph visualization
    - Entity browser with filtering
    - Entity editor
    - Relationship management
    - Graph analysis tools
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize world page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        # UI references
        self._graph: GraphComponent | None = None
        self._entity_list: Column | None = None
        self._editor_container: Column | None = None
        self._entity_name_input: Input | None = None
        self._entity_type_select: Select | None = None
        self._entity_desc_input: Textarea | None = None
        self._entity_attrs_editor: JsonEditor | None = None
        self._entity_attrs: dict = {}
        self._rel_source_select: Select | None = None
        self._rel_type_select: Select | None = None
        self._rel_target_select: Select | None = None
        self._analysis_result: Html | None = None
        self._undo_btn: Button | None = None
        self._redo_btn: Button | None = None

    def build(self) -> None:
        """Build the world page UI."""
        if not self.state.has_project:
            self._build_no_project_message()
            return

        # Check if interview is complete
        if not self.state.interview_complete:
            self._build_interview_required_message()
            return

        # Responsive layout: stack on mobile, 3-column on desktop
        with ui.row().classes("w-full h-full gap-4 p-4 flex-wrap lg:flex-nowrap"):
            # Left panel - Entity browser (full width on mobile, 20% on desktop)
            with ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px]"):
                self._build_entity_browser()

            # Center panel - Graph visualization (full width on mobile, 60% on desktop)
            with ui.column().classes("w-full lg:w-3/5 gap-4 min-w-[300px]"):
                self._build_graph_section()

            # Right panel - Entity editor (full width on mobile, 20% on desktop)
            self._editor_container = ui.column().classes("w-full lg:w-1/5 gap-4 min-w-[250px]")
            with self._editor_container:
                self._build_entity_editor()

        # Bottom sections
        with ui.column().classes("w-full gap-4 p-4"):
            self._build_relationships_section()
            self._build_analysis_section()

    def _build_no_project_message(self) -> None:
        """Build message when no project is selected."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-4 py-16"):
            ui.icon("public_off", size="xl").classes("text-gray-400 dark:text-gray-500")
            ui.label("No Project Selected").classes("text-xl text-gray-500 dark:text-gray-400")
            ui.label("Select a project from the header to explore its world.").classes(
                "text-gray-400 dark:text-gray-500"
            )

    def _build_interview_required_message(self) -> None:
        """Build message when interview is not complete."""
        with ui.column().classes("w-full min-h-96 items-center justify-center gap-6 py-16"):
            ui.icon("chat", size="xl").classes("text-blue-400")
            ui.label("Complete the Interview First").classes(
                "text-xl font-semibold text-gray-700 dark:text-gray-200"
            )
            ui.label(
                "The World Builder requires story context from the interview. "
                "Complete the interview to populate your story's world."
            ).classes("text-gray-500 dark:text-gray-400 text-center max-w-md")

            ui.button(
                "Go to Interview",
                on_click=lambda: ui.navigate.to("/"),
                icon="arrow_forward",
            ).props("color=primary size=lg")

    def _build_entity_browser(self) -> None:
        """Build the entity browser panel."""
        with ui.card().classes("w-full h-full"):
            ui.label("Entity Browser").classes("text-lg font-semibold")

            # Type filter
            ui.label("Filter by Type").classes(
                "text-sm font-medium text-gray-600 dark:text-gray-400 mt-2"
            )

            for entity_type in ["character", "location", "item", "faction", "concept"]:
                color = ENTITY_COLORS[entity_type]
                ui.checkbox(
                    entity_type.title(),
                    value=entity_type in self.state.entity_filter_types,
                    on_change=lambda e, t=entity_type: self._toggle_type_filter(t, e.value),
                ).props(f'color="{color}"')

            ui.separator()

            # Search
            ui.input(
                placeholder="Search entities...",
                on_change=self._on_search,
            ).classes("w-full").props("outlined dense")

            # Entity list with proper styling
            self._entity_list = ui.column().classes(
                "w-full gap-1 overflow-auto max-h-64 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
            )
            self._refresh_entity_list()

            # Add button
            ui.button(
                "+ Add Entity",
                on_click=self._show_add_dialog,
                icon="add",
            ).props("color=primary").classes("w-full mt-2")

            # Undo/Redo buttons
            with ui.row().classes("w-full gap-2 mt-2"):
                self._undo_btn = (
                    ui.button(
                        icon="undo",
                        on_click=self._do_undo,
                    )
                    .props("flat dense")
                    .tooltip("Undo (Ctrl+Z)")
                )
                self._redo_btn = (
                    ui.button(
                        icon="redo",
                        on_click=self._do_redo,
                    )
                    .props("flat dense")
                    .tooltip("Redo (Ctrl+Y)")
                )
                self._update_undo_redo_buttons()

    def _build_graph_section(self) -> None:
        """Build the graph visualization section."""
        self._graph = GraphComponent(
            world_db=self.state.world_db,
            on_node_select=self._on_node_select,
            on_edge_select=self._on_edge_select,
            height=450,
        )
        self._graph.build()

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

            # Entity form
            self._entity_name_input = ui.input(
                label="Name",
                value=entity.name,
            ).classes("w-full")

            self._entity_type_select = ui.select(
                label="Type",
                options=["character", "location", "item", "faction", "concept"],
                value=entity.type,
            ).classes("w-full")

            self._entity_desc_input = (
                ui.textarea(
                    label="Description",
                    value=entity.description,
                )
                .classes("w-full")
                .props("rows=4")
            )

            # Attributes JSON editor
            with ui.expansion("Attributes", icon="list").classes("w-full"):
                ui.label("Edit entity attributes as JSON").classes(
                    "text-xs text-gray-500 dark:text-gray-400 mb-2"
                )

                # Initialize attrs from entity
                self._entity_attrs = entity.attributes.copy() if entity.attributes else {}

                # JSON editor for attributes
                self._entity_attrs_editor = (
                    ui.json_editor(
                        {"content": {"json": self._entity_attrs}},
                        on_change=self._on_attrs_change,
                    )
                    .classes("w-full")
                    .style("height: 200px;")
                )

            # Action buttons
            with ui.row().classes("w-full gap-2 mt-4"):
                ui.button(
                    "Save",
                    on_click=self._save_entity,
                    icon="save",
                ).props("color=primary")

                ui.button(
                    "Delete",
                    on_click=self._confirm_delete_entity,
                    icon="delete",
                ).props("color=negative outline")

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

    def _build_analysis_section(self) -> None:
        """Build the graph analysis section."""
        with ui.expansion("Analysis Tools", icon="analytics", value=False).classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("path", label="Find Path")
                ui.tab("centrality", label="Most Connected")
                ui.tab("communities", label="Communities")

            with ui.tab_panels(tabs, value="path").classes("w-full"):
                # Path finder
                with ui.tab_panel("path"):
                    with ui.row().classes("items-end gap-4"):
                        entities = self._get_entity_options()
                        path_source = ui.select(label="From", options=entities).classes("w-48")
                        path_target = ui.select(label="To", options=entities).classes("w-48")
                        ui.button(
                            "Find Path",
                            on_click=lambda: self._find_path(path_source.value, path_target.value),
                        )

                # Centrality analysis
                with ui.tab_panel("centrality"):
                    ui.button(
                        "Show Most Connected",
                        on_click=self._show_centrality,
                    )

                # Community detection
                with ui.tab_panel("communities"):
                    ui.button(
                        "Detect Communities",
                        on_click=self._show_communities,
                    )

            # Analysis result display
            self._analysis_result = ui.html(sanitize=False).classes("w-full mt-4")

    # ========== Helper Methods ==========

    def _get_entity_options(self) -> dict[str, str]:
        """Get entity options for select dropdowns."""
        if not self.state.world_db:
            return {}

        entities = self.services.world.list_entities(self.state.world_db)
        return {e.id: e.name for e in entities}

    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display."""
        if not self._entity_list or not self.state.world_db:
            return

        self._entity_list.clear()

        entities = self.services.world.list_entities(self.state.world_db)

        # Filter by type
        if self.state.entity_filter_types:
            entities = [e for e in entities if e.type in self.state.entity_filter_types]

        # Filter by search
        if self.state.entity_search_query:
            query = self.state.entity_search_query.lower()
            entities = [
                e for e in entities if query in e.name.lower() or query in e.description.lower()
            ]

        with self._entity_list:
            if not entities:
                # Check if there are entities at all or just filtered out
                all_entities = self.services.world.list_entities(self.state.world_db)
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
                    ui.label("No matching entities").classes(
                        "text-gray-500 dark:text-gray-400 text-sm"
                    )
                    ui.label("Try adjusting filters or search").classes(
                        "text-xs text-gray-400 dark:text-gray-500"
                    )
            else:
                for entity in entities:
                    entity_list_item(
                        entity=entity,
                        on_select=self._select_entity,
                        selected=entity.id == self.state.selected_entity_id,
                    )

    # ========== Event Handlers ==========

    def _toggle_type_filter(self, entity_type: str, enabled: bool) -> None:
        """Toggle entity type filter."""
        if enabled and entity_type not in self.state.entity_filter_types:
            self.state.entity_filter_types.append(entity_type)
        elif not enabled and entity_type in self.state.entity_filter_types:
            self.state.entity_filter_types.remove(entity_type)

        self._refresh_entity_list()
        if self._graph:
            self._graph.set_filter(self.state.entity_filter_types)

    def _on_search(self, e) -> None:
        """Handle search input change."""
        self.state.entity_search_query = e.value
        self._refresh_entity_list()

        # Highlight matching nodes in the graph
        if self._graph:
            self._graph.highlight_search(e.value)

    def _on_node_select(self, entity_id: str) -> None:
        """Handle graph node selection."""
        self.state.select_entity(entity_id)
        self._refresh_entity_list()
        self._refresh_entity_editor()

    def _on_edge_select(self, relationship_id: str) -> None:
        """Handle graph edge selection to show relationship editor."""
        if not self.state.world_db:
            return

        # Get all relationships and find the one that matches
        relationships = self.state.world_db.list_relationships()
        rel = next((r for r in relationships if r.id == relationship_id), None)

        if not rel:
            logger.warning(f"Relationship not found: {relationship_id}")
            return

        # Get source and target entities
        source = self.services.world.get_entity(self.state.world_db, rel.source_id)
        target = self.services.world.get_entity(self.state.world_db, rel.target_id)

        source_name = source.name if source else "Unknown"
        target_name = target.name if target else "Unknown"

        # Show relationship editor dialog
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Edit Relationship").classes("text-lg font-semibold")
            ui.label(f"{source_name} → {target_name}").classes(
                "text-sm text-gray-500 dark:text-gray-400"
            )

            ui.separator()

            # Relationship type
            rel_type_input = ui.input(
                "Relationship Type",
                value=rel.relation_type,
            ).classes("w-full")

            # Description
            rel_desc_input = ui.textarea(
                "Description",
                value=rel.description,
            ).classes("w-full")

            # Strength slider
            ui.label("Strength").classes("text-sm mt-2")
            rel_strength_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.1,
                value=rel.strength,
            ).classes("w-full")

            # Bidirectional checkbox
            rel_bidir_checkbox = ui.checkbox(
                "Bidirectional",
                value=rel.bidirectional,
            )

            ui.separator()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def save_relationship():
                    if not self.state.world_db:
                        return

                    # Record action for undo (store old values)
                    self.state.record_action(
                        UndoAction(
                            action_type=ActionType.UPDATE_RELATIONSHIP,
                            data={
                                "relationship_id": rel.id,
                                "relation_type": rel_type_input.value,
                                "description": rel_desc_input.value,
                                "strength": rel_strength_slider.value,
                                "bidirectional": rel_bidir_checkbox.value,
                            },
                            inverse_data={
                                "relation_type": rel.relation_type,
                                "description": rel.description,
                                "strength": rel.strength,
                                "bidirectional": rel.bidirectional,
                            },
                        )
                    )

                    # Update the relationship
                    self.state.world_db.update_relationship(
                        relationship_id=rel.id,
                        relation_type=rel_type_input.value,
                        description=rel_desc_input.value,
                        strength=rel_strength_slider.value,
                        bidirectional=rel_bidir_checkbox.value,
                    )
                    dialog.close()
                    if self._graph:
                        self._graph.refresh()
                    self._update_undo_redo_buttons()
                    ui.notify("Relationship updated", type="positive")

                ui.button("Save", on_click=save_relationship).props("color=primary")

                def delete_relationship():
                    if not self.state.world_db:
                        return

                    # Record action for undo
                    self.state.record_action(
                        UndoAction(
                            action_type=ActionType.DELETE_RELATIONSHIP,
                            data={"relationship_id": rel.id},
                            inverse_data={
                                "source_id": rel.source_id,
                                "target_id": rel.target_id,
                                "relation_type": rel.relation_type,
                                "description": rel.description,
                            },
                        )
                    )

                    self.services.world.delete_relationship(self.state.world_db, rel.id)
                    dialog.close()
                    if self._graph:
                        self._graph.refresh()
                    self._update_undo_redo_buttons()
                    ui.notify("Relationship deleted", type="warning")

                ui.button("Delete", on_click=delete_relationship).props("color=negative flat")

        dialog.open()

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
            desc_input = ui.textarea(label="Description").classes("w-full")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Add",
                    on_click=lambda: self._add_entity(
                        dialog, name_input.value, type_select.value, desc_input.value
                    ),
                ).props("color=primary")

        dialog.open()

    def _add_entity(self, dialog, name: str, entity_type: str, description: str) -> None:
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

    def _on_attrs_change(self, e) -> None:
        """Handle changes in the attributes JSON editor."""
        try:
            content = e.args.get("content", {})
            if "json" in content:
                self._entity_attrs = content["json"]
            elif "text" in content:
                import json

                self._entity_attrs = json.loads(content["text"])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Invalid JSON, don't update

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
            new_attrs = self._entity_attrs if self._entity_attrs else None

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

    def _confirm_delete_entity(self) -> None:
        """Show confirmation dialog before deleting entity."""
        if not self.state.selected_entity_id or not self.state.world_db:
            return

        from ui.components.common import confirmation_dialog

        try:
            # Get entity name for better UX
            entity = self.services.world.get_entity(
                self.state.world_db,
                self.state.selected_entity_id,
            )
            entity_name = entity.name if entity else "this entity"

            confirmation_dialog(
                title="Delete Entity?",
                message=f'Are you sure you want to delete "{entity_name}"? This will also remove all relationships. This cannot be undone.',
                on_confirm=self._delete_entity,
                confirm_text="Delete",
                cancel_text="Cancel",
            )
        except Exception as e:
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

    def _find_path(self, source_id: str, target_id: str) -> None:
        """Find path between two entities."""
        if not self.state.world_db or not source_id or not target_id:
            return

        path = self.services.world.find_path(self.state.world_db, source_id, target_id)

        if self._analysis_result:
            self._analysis_result.content = render_path_result(self.state.world_db, path or [])

    def _show_centrality(self) -> None:
        """Show most connected entities."""
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_centrality_result(self.state.world_db)

    def _show_communities(self) -> None:
        """Show community detection results."""
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_communities_result(self.state.world_db)

    # ========== Undo/Redo Methods ==========

    def _update_undo_redo_buttons(self) -> None:
        """Update undo/redo button states based on history."""
        if self._undo_btn:
            self._undo_btn.set_enabled(self.state.can_undo())
        if self._redo_btn:
            self._redo_btn.set_enabled(self.state.can_redo())

    def _do_undo(self) -> None:
        """Execute undo operation."""
        action = self.state.undo()
        if not action or not self.state.world_db:
            return

        try:
            if action.action_type == ActionType.ADD_ENTITY:
                # Undo add = delete
                self.services.world.delete_entity(
                    self.state.world_db,
                    action.data["entity_id"],
                )
            elif action.action_type == ActionType.DELETE_ENTITY:
                # Undo delete = add back
                self.services.world.add_entity(
                    self.state.world_db,
                    entity_type=action.inverse_data["type"],
                    name=action.inverse_data["name"],
                    description=action.inverse_data.get("description", ""),
                    attributes=action.inverse_data.get("attributes"),
                )
            elif action.action_type == ActionType.UPDATE_ENTITY:
                # Undo update = restore old values
                self.services.world.update_entity(
                    self.state.world_db,
                    entity_id=action.data["entity_id"],
                    name=action.inverse_data.get("name"),
                    description=action.inverse_data.get("description"),
                    attributes=action.inverse_data.get("attributes"),
                )
            elif action.action_type == ActionType.ADD_RELATIONSHIP:
                # Undo add = delete
                self.services.world.delete_relationship(
                    self.state.world_db,
                    action.data["relationship_id"],
                )
            elif action.action_type == ActionType.DELETE_RELATIONSHIP:
                # Undo delete = add back
                self.services.world.add_relationship(
                    self.state.world_db,
                    source_id=action.inverse_data["source_id"],
                    target_id=action.inverse_data["target_id"],
                    relation_type=action.inverse_data["relation_type"],
                    description=action.inverse_data.get("description", ""),
                )
            elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
                # Undo update = restore old values
                self.state.world_db.update_relationship(
                    relationship_id=action.data["relationship_id"],
                    relation_type=action.inverse_data.get("relation_type"),
                    description=action.inverse_data.get("description"),
                    strength=action.inverse_data.get("strength"),
                    bidirectional=action.inverse_data.get("bidirectional"),
                )

            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Undone", type="info")
        except Exception as e:
            logger.exception("Undo failed")
            ui.notify(f"Undo failed: {e}", type="negative")

    def _do_redo(self) -> None:
        """Execute redo operation."""
        action = self.state.redo()
        if not action or not self.state.world_db:
            return

        try:
            if action.action_type == ActionType.ADD_ENTITY:
                # Redo add = add again
                self.services.world.add_entity(
                    self.state.world_db,
                    entity_type=action.data["type"],
                    name=action.data["name"],
                    description=action.data.get("description", ""),
                    attributes=action.data.get("attributes"),
                )
            elif action.action_type == ActionType.DELETE_ENTITY:
                # Redo delete = delete again
                self.services.world.delete_entity(
                    self.state.world_db,
                    action.data["entity_id"],
                )
            elif action.action_type == ActionType.UPDATE_ENTITY:
                # Redo update = apply new values again
                self.services.world.update_entity(
                    self.state.world_db,
                    entity_id=action.data["entity_id"],
                    name=action.data.get("name"),
                    description=action.data.get("description"),
                    attributes=action.data.get("attributes"),
                )
            elif action.action_type == ActionType.ADD_RELATIONSHIP:
                # Redo add = add again
                self.services.world.add_relationship(
                    self.state.world_db,
                    source_id=action.data["source_id"],
                    target_id=action.data["target_id"],
                    relation_type=action.data["relation_type"],
                    description=action.data.get("description", ""),
                )
            elif action.action_type == ActionType.DELETE_RELATIONSHIP:
                # Redo delete = delete again
                self.services.world.delete_relationship(
                    self.state.world_db,
                    action.data["relationship_id"],
                )
            elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
                # Redo update = apply new values again
                self.state.world_db.update_relationship(
                    relationship_id=action.data["relationship_id"],
                    relation_type=action.data.get("relation_type"),
                    description=action.data.get("description"),
                    strength=action.data.get("strength"),
                    bidirectional=action.data.get("bidirectional"),
                )

            self._refresh_entity_list()
            self._refresh_entity_editor()
            if self._graph:
                self._graph.refresh()
            self._update_undo_redo_buttons()
            ui.notify("Redone", type="info")
        except Exception as e:
            logger.exception("Redo failed")
            ui.notify(f"Redo failed: {e}", type="negative")
