# Undo/Redo Functionality

## Overview

Story Factory now supports undo/redo functionality across all pages using a centralized command pattern. Users can undo and redo actions using keyboard shortcuts (Ctrl+Z, Ctrl+Y) or page-specific buttons.

## Features

### Supported Actions

#### World Page (Entity Management)
- ✅ Add entity
- ✅ Delete entity
- ✅ Update entity
- ✅ Add relationship
- ✅ Delete relationship
- ✅ Update relationship

#### Settings Page
- ✅ Update settings (all configuration changes)

#### Write Page (Planned)
- ⏳ Update chapter content
- ⏳ Delete chapter
- ⏳ Add chapter
- ⏳ Update chapter feedback

### Keyboard Shortcuts

- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo last undone action
- **Ctrl+Shift+Z**: Alternative redo shortcut

**Note**: Keyboard shortcuts are disabled in input fields to preserve browser's native undo/redo for text editing.

## Architecture

### Command Pattern

The undo/redo system uses the Command pattern with the following components:

1. **ActionType** (`ui/state.py`): Enum defining all undoable action types
2. **UndoAction** (`ui/state.py`): Dataclass storing action data and inverse data
3. **AppState** (`ui/state.py`): Manages undo/redo stacks and callbacks
4. **Page Handlers**: Each page implements its own undo/redo logic

### State Management

```python
# AppState manages two stacks
_undo_stack: list[UndoAction]  # Actions that can be undone
_redo_stack: list[UndoAction]  # Actions that can be redone

# History is limited to 50 actions to prevent memory issues
_max_undo_history: int = 50
```

### Page Integration

Each page registers its undo/redo handlers with AppState:

```python
class WorldPage:
    def __init__(self, state: AppState, services: ServiceContainer):
        self.state = state
        # Register handlers for this page
        self.state.on_undo(self._do_undo)
        self.state.on_redo(self._do_redo)
```

### Recording Actions

When a user performs an action, the page records it:

```python
self.state.record_action(
    UndoAction(
        action_type=ActionType.ADD_ENTITY,
        data={"entity_id": entity_id, "name": name, ...},
        inverse_data={"type": type, "name": name, ...}
    )
)
```

## Implementation Details

### World Page

The World page uses entity/relationship-specific undo/redo:

- **Add Entity**: Undo deletes the entity
- **Delete Entity**: Undo recreates the entity with all attributes
- **Update Entity**: Undo restores previous values
- **Relationships**: Same pattern as entities

### Settings Page

The Settings page uses snapshot-based undo/redo:

- **Save Settings**: Captures before/after snapshots
- **Undo**: Restores previous snapshot
- **Redo**: Restores next snapshot

```python
def _save_settings(self):
    # Capture old state
    old_snapshot = self._capture_settings_snapshot()
    
    # Update settings...
    
    # Capture new state
    new_snapshot = self._capture_settings_snapshot()
    
    # Record for undo
    self.state.record_action(
        UndoAction(
            action_type=ActionType.UPDATE_SETTINGS,
            data=new_snapshot,
            inverse_data=old_snapshot
        )
    )
```

## Testing

The undo/redo functionality includes comprehensive unit tests:

- **16 tests** covering all aspects of the system
- Tests for stack management (push, pop, clear)
- Tests for callback system
- Tests for action type definitions
- Tests for history limits

Run tests with:
```bash
pytest tests/unit/test_ui/test_undo_redo_state.py -v
```

## Future Enhancements

### Write Page Support (Planned)

- Scene content editing undo/redo
- Chapter structure changes
- Feedback and notes

### Advanced Features (Future)

- **Named Checkpoints**: Allow users to create named save points
- **History Browser**: Visual history viewer with action descriptions
- **Persistent History**: Save undo history to disk (currently session-only)
- **Batch Operations**: Group related actions for single undo/redo
- **Action Descriptions**: Human-readable descriptions in UI

## Troubleshooting

### Undo/Redo Not Working

1. **Check if actions are being recorded**: Enable debug logging to see if actions are recorded
2. **Verify keyboard shortcuts**: Make sure you're not in an input field
3. **Check page handler**: Verify the page has registered its handlers

### History Not Persisting

- Undo/redo history is session-only and cleared when:
  - Page is refreshed
  - Project is switched
  - Application is restarted

### Memory Issues

- History is limited to 50 actions
- Large data (entity descriptions, settings snapshots) may use memory
- Clear history manually if needed: `state.clear_history()`

## API Reference

### AppState Methods

```python
# Record an action
state.record_action(action: UndoAction) -> None

# Check if undo/redo is available
state.can_undo() -> bool
state.can_redo() -> bool

# Perform undo/redo (returns action or None)
state.undo() -> UndoAction | None
state.redo() -> UndoAction | None

# Trigger via keyboard shortcuts
state.trigger_undo() -> None  # Calls registered callback
state.trigger_redo() -> None  # Calls registered callback

# Clear history
state.clear_history() -> None

# Register callbacks
state.on_undo(callback: Callable[[], None]) -> None
state.on_redo(callback: Callable[[], None]) -> None
```

### ActionType Enum

```python
# World/Entity actions
ADD_ENTITY = "add_entity"
DELETE_ENTITY = "delete_entity"
UPDATE_ENTITY = "update_entity"
ADD_RELATIONSHIP = "add_relationship"
DELETE_RELATIONSHIP = "delete_relationship"
UPDATE_RELATIONSHIP = "update_relationship"

# Write/Chapter actions
UPDATE_CHAPTER_CONTENT = "update_chapter_content"
DELETE_CHAPTER = "delete_chapter"
ADD_CHAPTER = "add_chapter"
UPDATE_CHAPTER_FEEDBACK = "update_chapter_feedback"

# Settings actions
UPDATE_SETTINGS = "update_settings"
```
