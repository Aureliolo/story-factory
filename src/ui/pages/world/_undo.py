"""World Builder page - undo mixin with undo/redo methods."""

import logging

from nicegui import ui

from src.ui.pages.world._page import WorldPageBase
from src.ui.state import ActionType

logger = logging.getLogger(__name__)


class UndoMixin(WorldPageBase):
    """Mixin providing undo/redo methods for WorldPage."""

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

    # Methods to be implemented by other mixins
    def _refresh_entity_list(self) -> None:
        """Refresh the entity list display - implemented by BrowserMixin."""
        raise NotImplementedError

    def _refresh_entity_editor(self) -> None:
        """Refresh the entity editor panel - implemented by EditorMixin."""
        raise NotImplementedError
