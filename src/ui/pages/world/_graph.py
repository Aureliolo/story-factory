"""Graph visualization and relationship management functions for the World page."""

import logging

from nicegui import ui

from src.ui.components.graph import GraphComponent
from src.ui.state import ActionType, UndoAction

logger = logging.getLogger("src.ui.pages.world._graph")

# Default value for relationship strength when creating via drag-and-drop
DEFAULT_RELATIONSHIP_STRENGTH = 0.5


def build_graph_section(page) -> None:
    """Build the graph visualization section.

    Args:
        page: WorldPage instance.
    """
    # Use larger height to match entity browser and editor
    page._graph = GraphComponent(
        world_db=page.state.world_db,
        settings=page.services.settings,
        on_node_select=page._on_node_select,
        on_edge_select=page._on_edge_select,
        on_create_relationship=page._on_create_relationship,
        on_edge_context_menu=page._on_edge_context_menu,
        height=600,  # Taller to match browser and editor panels
    )
    page._graph.build()


def get_entity_options(page) -> dict[str, str]:
    """Get entity options for select dropdowns.

    Args:
        page: WorldPage instance.

    Returns:
        Dictionary mapping entity ID to entity name.
    """
    if not page.state.world_db:
        return {}

    entities = page.services.world.list_entities(page.state.world_db)
    return {e.id: e.name for e in entities}


def build_relationships_section(page) -> None:
    """Build the relationships management section.

    Args:
        page: WorldPage instance.
    """
    with ui.expansion("Relationships", icon="link", value=False).classes("w-full"):
        # Add relationship form
        with ui.row().classes("w-full items-end gap-4 mb-4"):
            entities = get_entity_options(page)

            page._rel_source_select = ui.select(
                label="From",
                options=entities,
            ).classes("w-48")

            page._rel_type_select = ui.select(
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

            page._rel_target_select = ui.select(
                label="To",
                options=entities,
            ).classes("w-48")

            ui.button(
                "Add",
                on_click=lambda: add_relationship(page),
                icon="add",
            ).props("color=primary")

        # Relationships table
        if page.state.world_db:
            relationships = page.services.world.get_relationships(page.state.world_db)

            if relationships:
                columns = [
                    {"name": "from", "label": "From", "field": "from", "align": "left"},
                    {"name": "type", "label": "Type", "field": "type", "align": "center"},
                    {"name": "to", "label": "To", "field": "to", "align": "left"},
                    {"name": "actions", "label": "", "field": "actions", "align": "right"},
                ]

                rows = []
                for rel in relationships:
                    source = page.services.world.get_entity(page.state.world_db, rel.source_id)
                    target = page.services.world.get_entity(page.state.world_db, rel.target_id)
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


def on_node_select(page, entity_id: str) -> None:
    """Handle graph node selection.

    Args:
        page: WorldPage instance.
        entity_id: Selected entity ID.
    """
    page.state.select_entity(entity_id)
    page._refresh_entity_list()
    page._refresh_entity_editor()


def on_edge_select(page, relationship_id: str) -> None:
    """Handle graph edge selection to show relationship editor.

    Args:
        page: WorldPage instance.
        relationship_id: Selected relationship ID.
    """
    if not page.state.world_db:
        return

    # Get all relationships and find the one that matches
    relationships = page.state.world_db.list_relationships()
    rel = next((r for r in relationships if r.id == relationship_id), None)

    if not rel:
        # This can happen when clicking on a stale edge after relationships were
        # regenerated - the graph may have old edge IDs. Silently ignore.
        logger.debug(f"Relationship not found (stale edge click): {relationship_id}")
        return

    # Get source and target entities
    source = page.services.world.get_entity(page.state.world_db, rel.source_id)
    target = page.services.world.get_entity(page.state.world_db, rel.target_id)

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
                if not page.state.world_db:
                    return

                # Record action for undo (store old values)
                page.state.record_action(
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
                page.state.world_db.update_relationship(
                    relationship_id=rel.id,
                    relation_type=rel_type_input.value,
                    description=rel_desc_input.value,
                    strength=rel_strength_slider.value,
                    bidirectional=rel_bidir_checkbox.value,
                )
                dialog.close()
                if page._graph:
                    page._graph.refresh()
                page._update_undo_redo_buttons()
                ui.notify("Relationship updated", type="positive")

            ui.button("Save", on_click=save_relationship).props("color=primary")

            def delete_relationship() -> None:
                """Delete relationship from world database."""
                if not page.state.world_db:
                    return

                # Record action for undo
                page.state.record_action(
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

                page.services.world.delete_relationship(page.state.world_db, rel.id)
                dialog.close()
                if page._graph:
                    page._graph.refresh()
                page._update_undo_redo_buttons()
                ui.notify("Relationship deleted", type="warning")

            ui.button("Delete", on_click=delete_relationship).props("color=negative flat")

    dialog.open()


def on_create_relationship(page, source_id: str, target_id: str) -> None:
    """Handle drag-to-connect relationship creation.

    Args:
        page: WorldPage instance.
        source_id: Source entity ID.
        target_id: Target entity ID.
    """
    logger.debug("Creating relationship via drag: %s -> %s", source_id, target_id)
    if not page.state.world_db:
        return

    # Get entity names for display
    source = page.services.world.get_entity(page.state.world_db, source_id)
    target = page.services.world.get_entity(page.state.world_db, target_id)

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
                if not page.state.world_db:
                    return

                try:
                    relationship_id = page.services.world.add_relationship(
                        page.state.world_db,
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
                    page.state.world_db.update_relationship(
                        relationship_id=relationship_id,
                        strength=final_strength,
                        bidirectional=final_bidirectional,
                    )

                    # Record action for undo with complete data
                    page.state.record_action(
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
                    if page._graph:
                        page._graph.refresh()
                    page._update_undo_redo_buttons()
                    ui.notify(
                        f"Created relationship: {source_name} → {target_name}", type="positive"
                    )
                except Exception as e:
                    logger.exception("Failed to create relationship via drag")
                    ui.notify(f"Error: {e}", type="negative")

            ui.button("Create", on_click=create_relationship).props("color=primary")

    dialog.open()


def on_edge_context_menu(page, edge_id: str) -> None:
    """Handle edge right-click context menu.

    Args:
        page: WorldPage instance.
        edge_id: Edge/relationship ID.
    """
    logger.debug("Edge context menu triggered for: %s", edge_id)
    # Just trigger the existing edge select handler which shows the edit dialog
    on_edge_select(page, edge_id)


def add_relationship(page) -> None:
    """Add a new relationship from the form.

    Args:
        page: WorldPage instance.
    """
    if not page.state.world_db:
        return

    source_id = page._rel_source_select.value if page._rel_source_select else None
    rel_type = page._rel_type_select.value if page._rel_type_select else None
    target_id = page._rel_target_select.value if page._rel_target_select else None

    if not source_id or not rel_type or not target_id:
        ui.notify("All fields required", type="warning")
        return

    try:
        relationship_id = page.services.world.add_relationship(
            page.state.world_db,
            source_id=source_id,
            target_id=target_id,
            relation_type=rel_type,
        )

        # Record action for undo
        page.state.record_action(
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

        if page._graph:
            page._graph.refresh()
        page._update_undo_redo_buttons()
        ui.notify("Relationship added", type="positive")
    except Exception as e:
        logger.exception("Failed to add relationship")
        ui.notify(f"Error: {e}", type="negative")
