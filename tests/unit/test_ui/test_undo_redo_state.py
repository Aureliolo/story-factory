"""Tests for undo/redo functionality in ui/state.py."""

from src.ui.state import ActionType, AppState, UndoAction


class TestUndoRedoState:
    """Tests for undo/redo in AppState."""

    def test_record_action_adds_to_undo_stack(self):
        """Test that recording an action adds it to the undo stack."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123", "name": "Test"},
            inverse_data={"name": "Test"},
        )

        state.record_action(action)

        assert state.can_undo()
        assert not state.can_redo()

    def test_record_action_clears_redo_stack(self):
        """Test that recording a new action clears the redo stack."""
        state = AppState()

        # Add and undo an action to populate redo stack
        action1 = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "1"},
            inverse_data={},
        )
        state.record_action(action1)
        state.undo()

        assert state.can_redo()

        # Record a new action
        action2 = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "2"},
            inverse_data={},
        )
        state.record_action(action2)

        # Redo stack should be cleared
        assert not state.can_redo()

    def test_undo_returns_action(self):
        """Test that undo returns the last action."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.UPDATE_ENTITY,
            data={"entity_id": "123", "name": "New"},
            inverse_data={"name": "Old"},
        )

        state.record_action(action)
        undone = state.undo()

        assert undone is not None
        assert undone.action_type == ActionType.UPDATE_ENTITY
        assert undone.data["name"] == "New"

    def test_undo_moves_to_redo_stack(self):
        """Test that undo moves action to redo stack."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )

        state.record_action(action)
        state.undo()

        assert not state.can_undo()
        assert state.can_redo()

    def test_redo_returns_action(self):
        """Test that redo returns the action."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.DELETE_ENTITY,
            data={"entity_id": "123"},
            inverse_data={"name": "Test"},
        )

        state.record_action(action)
        state.undo()
        redone = state.redo()

        assert redone is not None
        assert redone.action_type == ActionType.DELETE_ENTITY

    def test_redo_moves_back_to_undo_stack(self):
        """Test that redo moves action back to undo stack."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )

        state.record_action(action)
        state.undo()
        state.redo()

        assert state.can_undo()
        assert not state.can_redo()

    def test_undo_empty_stack_returns_none(self):
        """Test that undo on empty stack returns None."""
        state = AppState()
        assert state.undo() is None

    def test_redo_empty_stack_returns_none(self):
        """Test that redo on empty stack returns None."""
        state = AppState()
        assert state.redo() is None

    def test_clear_history_clears_both_stacks(self):
        """Test that clear_history clears both undo and redo stacks."""
        state = AppState()

        # Add actions and undo one
        action1 = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "1"},
            inverse_data={},
        )
        action2 = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "2"},
            inverse_data={},
        )
        state.record_action(action1)
        state.record_action(action2)
        state.undo()

        assert state.can_undo()
        assert state.can_redo()

        state.clear_history()

        assert not state.can_undo()
        assert not state.can_redo()

    def test_max_undo_history_limit(self):
        """Test that undo history is limited to max size."""
        state = AppState()
        state._max_undo_history = 3

        # Add 5 actions
        for i in range(5):
            action = UndoAction(
                action_type=ActionType.ADD_ENTITY,
                data={"entity_id": str(i)},
                inverse_data={},
            )
            state.record_action(action)

        # Should only keep last 3
        assert len(state._undo_stack) == 3
        assert state._undo_stack[0].data["entity_id"] == "2"  # First is now id=2

    def test_trigger_undo_calls_callback(self):
        """Test that trigger_undo calls registered callback."""
        state = AppState()
        called = []

        def undo_callback():
            """Record that the undo callback was invoked."""
            called.append(True)

        state.on_undo(undo_callback)

        # Add an action first
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )
        state.record_action(action)

        state.trigger_undo()

        assert len(called) == 1

    def test_trigger_redo_calls_callback(self):
        """Test that trigger_redo calls registered callback."""
        state = AppState()
        called = []

        def redo_callback():
            """Record that the redo callback was invoked."""
            called.append(True)

        state.on_redo(redo_callback)

        # Add and undo an action to enable redo
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )
        state.record_action(action)
        state.undo()

        state.trigger_redo()

        assert len(called) == 1

    def test_trigger_undo_without_callback_doesnt_error(self):
        """Test that trigger_undo without callback doesn't error."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )
        state.record_action(action)

        # Should not raise error
        state.trigger_undo()

    def test_trigger_redo_without_callback_doesnt_error(self):
        """Test that trigger_redo without callback doesn't error."""
        state = AppState()
        action = UndoAction(
            action_type=ActionType.ADD_ENTITY,
            data={"entity_id": "123"},
            inverse_data={},
        )
        state.record_action(action)
        state.undo()

        # Should not raise error
        state.trigger_redo()

    def test_new_action_types(self):
        """Test that new action types are defined."""
        # Chapter actions
        assert hasattr(ActionType, "UPDATE_CHAPTER_CONTENT")
        assert hasattr(ActionType, "DELETE_CHAPTER")
        assert hasattr(ActionType, "ADD_CHAPTER")
        assert hasattr(ActionType, "UPDATE_CHAPTER_FEEDBACK")

        # Settings actions
        assert hasattr(ActionType, "UPDATE_SETTINGS")

    def test_action_type_values(self):
        """Test that action type values are correctly set."""
        assert ActionType.UPDATE_CHAPTER_CONTENT.value == "update_chapter_content"
        assert ActionType.DELETE_CHAPTER.value == "delete_chapter"
        assert ActionType.ADD_CHAPTER.value == "add_chapter"
        assert ActionType.UPDATE_CHAPTER_FEEDBACK.value == "update_chapter_feedback"
        assert ActionType.UPDATE_SETTINGS.value == "update_settings"
