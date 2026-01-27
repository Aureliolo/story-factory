"""Undo/redo functions for the World page."""

import logging

from nicegui import ui

from src.ui.state import ActionType

logger = logging.getLogger("src.ui.pages.world._undo")


def update_undo_redo_buttons(page) -> None:
    """Update undo/redo button states based on history.

    Args:
        page: WorldPage instance.
    """
    if page._undo_btn:
        page._undo_btn.set_enabled(page.state.can_undo())
    if page._redo_btn:
        page._redo_btn.set_enabled(page.state.can_redo())


def do_undo(page) -> None:
    """Execute undo operation.

    Args:
        page: WorldPage instance.
    """
    action = page.state.undo()
    if not action or not page.state.world_db:
        return

    try:
        if action.action_type == ActionType.ADD_ENTITY:
            # Undo add = delete
            page.services.world.delete_entity(
                page.state.world_db,
                action.data["entity_id"],
            )
        elif action.action_type == ActionType.DELETE_ENTITY:
            # Undo delete = add back
            page.services.world.add_entity(
                page.state.world_db,
                entity_type=action.inverse_data["type"],
                name=action.inverse_data["name"],
                description=action.inverse_data.get("description", ""),
                attributes=action.inverse_data.get("attributes"),
            )
        elif action.action_type == ActionType.UPDATE_ENTITY:
            # Undo update = restore old values
            page.services.world.update_entity(
                page.state.world_db,
                entity_id=action.data["entity_id"],
                name=action.inverse_data.get("name"),
                description=action.inverse_data.get("description"),
                attributes=action.inverse_data.get("attributes"),
            )
        elif action.action_type == ActionType.ADD_RELATIONSHIP:
            # Undo add = delete
            page.services.world.delete_relationship(
                page.state.world_db,
                action.data["relationship_id"],
            )
        elif action.action_type == ActionType.DELETE_RELATIONSHIP:
            # Undo delete = add back
            page.services.world.add_relationship(
                page.state.world_db,
                source_id=action.inverse_data["source_id"],
                target_id=action.inverse_data["target_id"],
                relation_type=action.inverse_data["relation_type"],
                description=action.inverse_data.get("description", ""),
            )
        elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
            # Undo update = restore old values
            page.state.world_db.update_relationship(
                relationship_id=action.data["relationship_id"],
                relation_type=action.inverse_data.get("relation_type"),
                description=action.inverse_data.get("description"),
                strength=action.inverse_data.get("strength"),
                bidirectional=action.inverse_data.get("bidirectional"),
            )

        page._refresh_entity_list()
        page._refresh_entity_editor()
        if page._graph:
            page._graph.refresh()
        update_undo_redo_buttons(page)
        ui.notify("Undone", type="info")
    except Exception as e:
        logger.exception("Undo failed")
        ui.notify(f"Undo failed: {e}", type="negative")


def do_redo(page) -> None:
    """Execute redo operation.

    Args:
        page: WorldPage instance.
    """
    action = page.state.redo()
    if not action or not page.state.world_db:
        return

    try:
        if action.action_type == ActionType.ADD_ENTITY:
            # Redo add = add again
            page.services.world.add_entity(
                page.state.world_db,
                entity_type=action.data["type"],
                name=action.data["name"],
                description=action.data.get("description", ""),
                attributes=action.data.get("attributes"),
            )
        elif action.action_type == ActionType.DELETE_ENTITY:
            # Redo delete = delete again
            page.services.world.delete_entity(
                page.state.world_db,
                action.data["entity_id"],
            )
        elif action.action_type == ActionType.UPDATE_ENTITY:
            # Redo update = apply new values again
            page.services.world.update_entity(
                page.state.world_db,
                entity_id=action.data["entity_id"],
                name=action.data.get("name"),
                description=action.data.get("description"),
                attributes=action.data.get("attributes"),
            )
        elif action.action_type == ActionType.ADD_RELATIONSHIP:
            # Redo add = add again
            page.services.world.add_relationship(
                page.state.world_db,
                source_id=action.data["source_id"],
                target_id=action.data["target_id"],
                relation_type=action.data["relation_type"],
                description=action.data.get("description", ""),
            )
        elif action.action_type == ActionType.DELETE_RELATIONSHIP:
            # Redo delete = delete again
            page.services.world.delete_relationship(
                page.state.world_db,
                action.data["relationship_id"],
            )
        elif action.action_type == ActionType.UPDATE_RELATIONSHIP:
            # Redo update = apply new values again
            page.state.world_db.update_relationship(
                relationship_id=action.data["relationship_id"],
                relation_type=action.data.get("relation_type"),
                description=action.data.get("description"),
                strength=action.data.get("strength"),
                bidirectional=action.data.get("bidirectional"),
            )

        page._refresh_entity_list()
        page._refresh_entity_editor()
        if page._graph:
            page._graph.refresh()
        update_undo_redo_buttons(page)
        ui.notify("Redone", type="info")
    except Exception as e:
        logger.exception("Redo failed")
        ui.notify(f"Redo failed: {e}", type="negative")
