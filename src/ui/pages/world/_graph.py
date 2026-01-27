"""World Builder page - graph mixin with graph interaction methods."""

import logging

from nicegui import ui

from src.ui.components.graph import GraphComponent
from src.ui.graph_renderer import (
    render_centrality_result,
    render_communities_result,
    render_path_result,
)
from src.ui.pages.world._page import WorldPageBase
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger(__name__)

# Default value for relationship strength when creating via drag-and-drop
DEFAULT_RELATIONSHIP_STRENGTH = 0.5


class GraphMixin(WorldPageBase):
    """Mixin providing graph interaction methods for WorldPage."""

    def _build_graph_section(self) -> None:
        """Build the graph visualization section."""
        # Use larger height to match entity browser and editor
        self._graph = GraphComponent(
            world_db=self.state.world_db,
            settings=self.services.settings,
            on_node_select=self._on_node_select,
            on_edge_select=self._on_edge_select,
            on_create_relationship=self._on_create_relationship,
            on_edge_context_menu=self._on_edge_context_menu,
            height=600,  # Taller to match browser and editor panels
        )
        self._graph.build()

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
            # This can happen when clicking on a stale edge after relationships were
            # regenerated - the graph may have old edge IDs. Silently ignore.
            logger.debug(f"Relationship not found (stale edge click): {relationship_id}")
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
            rel_desc_input = (
                ui.textarea(
                    "Description",
                    value=rel.description,
                )
                .props("filled")
                .classes("w-full")
            )

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

                def save_relationship() -> None:
                    """Save relationship updates to world database."""
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

                def delete_relationship() -> None:
                    """Delete relationship from world database."""
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

    def _on_create_relationship(self, source_id: str, target_id: str) -> None:
        """Handle drag-to-connect relationship creation.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
        """
        logger.debug("Creating relationship via drag: %s -> %s", source_id, target_id)
        if not self.state.world_db:
            return

        # Get entity names for display
        source = self.services.world.get_entity(self.state.world_db, source_id)
        target = self.services.world.get_entity(self.state.world_db, target_id)

        if not source or not target:
            ui.notify("Entity not found", type="negative")
            return

        source_name = source.name
        target_name = target.name

        # Show dialog to configure relationship
        with ui.dialog() as dialog, ui.card().classes("w-96 p-4"):
            ui.label("Create Relationship").classes("text-lg font-semibold mb-2")
            ui.label(f"{source_name} → {target_name}").classes(
                "text-sm text-gray-500 dark:text-gray-400 mb-4"
            )

            ui.separator()

            # Relationship type
            rel_type_input = ui.select(
                label="Relationship Type",
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
                    "works_for",
                    "leads",
                ],
                value="knows",
                new_value_mode="add",
            ).classes("w-full")

            # Description
            rel_desc_input = (
                ui.textarea(
                    label="Description (optional)",
                    placeholder="Describe this relationship...",
                )
                .classes("w-full")
                .props("rows=2")
            )

            # Strength slider
            ui.label("Strength").classes("text-sm mt-2")
            rel_strength_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.1,
                value=DEFAULT_RELATIONSHIP_STRENGTH,
            ).classes("w-full")

            # Bidirectional checkbox
            rel_bidir_checkbox = ui.checkbox(
                "Bidirectional",
                value=False,
            )

            ui.separator()

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def create_relationship():
                    """Create the relationship between the dragged entities."""
                    if not self.state.world_db:
                        return

                    try:
                        relationship_id = self.services.world.add_relationship(
                            self.state.world_db,
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=rel_type_input.value,
                            description=rel_desc_input.value,
                        )

                        # Store actual values selected by the user
                        final_strength = rel_strength_slider.value
                        final_bidirectional = rel_bidir_checkbox.value

                        # Always update relationship to match UI-selected values
                        # (DB defaults differ from UI defaults, so we must always sync)
                        self.state.world_db.update_relationship(
                            relationship_id=relationship_id,
                            strength=final_strength,
                            bidirectional=final_bidirectional,
                        )

                        # Record action for undo with complete data
                        self.state.record_action(
                            UndoAction(
                                action_type=ActionType.ADD_RELATIONSHIP,
                                data={
                                    "relationship_id": relationship_id,
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "relation_type": rel_type_input.value,
                                    "description": rel_desc_input.value,
                                    "strength": final_strength,
                                    "bidirectional": final_bidirectional,
                                },
                                inverse_data={
                                    "relationship_id": relationship_id,
                                    "source_id": source_id,
                                    "target_id": target_id,
                                    "relation_type": rel_type_input.value,
                                    "description": rel_desc_input.value,
                                    "strength": final_strength,
                                    "bidirectional": final_bidirectional,
                                },
                            )
                        )

                        dialog.close()
                        if self._graph:
                            self._graph.refresh()
                        self._update_undo_redo_buttons()
                        ui.notify(
                            f"Created relationship: {source_name} → {target_name}", type="positive"
                        )
                    except Exception as e:
                        logger.exception("Failed to create relationship via drag")
                        ui.notify(f"Error: {e}", type="negative")

                ui.button("Create", on_click=create_relationship).props("color=primary")

        dialog.open()

    def _on_edge_context_menu(self, edge_id: str) -> None:
        """Handle edge right-click context menu.

        Args:
            edge_id: Edge/relationship ID.
        """
        logger.debug("Edge context menu triggered for: %s", edge_id)
        # Just trigger the existing edge select handler which shows the edit dialog
        self._on_edge_select(edge_id)

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
        """
        Render and display community-detection analysis in the analysis result pane.

        If no world database is available this does nothing. If an analysis result container exists, its content is replaced with the rendered community detection output for the current world.
        """
        if not self.state.world_db:
            return

        if self._analysis_result:
            self._analysis_result.content = render_communities_result(self.state.world_db)

    def _build_conflict_map_tab(self) -> None:
        """
        Constructs the Conflict Map tab UI showing relationships colored by conflict category and a link to the World Timeline.

        Renders a descriptive label, instantiates and builds a ConflictGraphComponent bound to the current world database and services (with node selection callback), and adds a button that navigates to the full World Timeline.
        """
        logger.debug("Building conflict map tab")
        from src.ui.components.conflict_graph import ConflictGraphComponent

        with ui.column().classes("w-full gap-4"):
            ui.label(
                "Visualize relationships colored by conflict category: "
                "alliances (green), rivalries (red), tensions (yellow), and neutral (blue)."
            ).classes("text-sm text-gray-600 dark:text-gray-400")

            # Conflict graph component
            conflict_graph = ConflictGraphComponent(
                world_db=self.state.world_db,
                services=self.services,
                on_node_select=self._on_node_select,
                height=400,
            )
            conflict_graph.build()

            # Link to full world timeline
            with ui.row().classes("items-center gap-4 mt-2"):
                ui.button(
                    "View World Timeline",
                    on_click=lambda: ui.navigate.to("/world-timeline"),
                    icon="timeline",
                ).props("flat")

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
