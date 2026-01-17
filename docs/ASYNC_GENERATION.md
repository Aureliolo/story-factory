# Async Generation with Background Tasks

## Overview
This feature allows story generation to run in the background without blocking the UI. Users can continue browsing the world, reviewing chapters, and using other parts of the application while generation is in progress.

## Features Implemented

### 1. Background Task Execution
- Generation runs in background using NiceGUI's `run.io_bound()`
- UI remains responsive during generation
- Users can navigate between pages while generation runs

### 2. Generation Status Display
- Real-time progress updates shown in UI
- Progress messages display current generation stage
- Visual indicator (progress bar) shows activity

### 3. Cancellation Support
- Cancel button available during generation
- Safe cancellation that preserves current state
- No data loss when cancelling

### 4. State Management
New fields added to `AppState`:
- `generation_cancel_requested`: Flag to request cancellation
- `generation_pause_requested`: Flag to request pause (UI only, backend pending)
- `generation_is_paused`: Current pause state (future feature)
- `generation_can_resume`: Resume capability (future feature)

Methods:
- `request_cancel_generation()`: Request cancellation
- `request_pause_generation()`: Request pause (UI ready)
- `resume_generation()`: Resume paused generation (future)
- `reset_generation_flags()`: Reset all control flags

## Architecture

### Components

#### 1. GenerationStatus Component (`ui/components/generation_status.py`)
- Displays current progress message
- Shows progress bar (indeterminate mode)
- Cancel button (functional)
- Pause/Resume button (UI only, backend pending)

#### 2. Story Service Updates (`services/story_service.py`)
- Added `cancel_check` callback parameter to `write_chapter()`
- Checks cancellation flag periodically during generation
- Raises `GenerationCancelled` exception when cancelled
- Preserves state up to cancellation point

#### 3. Write Page Updates (`ui/pages/write.py`)
- Converted `_write_current_chapter()` to true background task
- Converted `_write_all_chapters()` to true background task
- Progressive UI updates during generation
- Proper client context handling for notifications
- Prevents multiple concurrent generations

## Usage

### For Users
1. Click "Write Chapter" or "Write All Chapters"
2. Generation status appears showing progress
3. Click "Cancel" button to stop generation
4. UI remains usable - browse world, view other chapters, etc.

### For Developers

#### Checking for Cancellation
```python
def should_cancel() -> bool:
    return self.state.generation_cancel_requested

# Pass to story service
events = self.services.story.write_chapter(
    project, 
    chapter_num, 
    cancel_check=should_cancel
)
```

#### Handling GenerationCancelled
```python
from services.story_service import GenerationCancelled

try:
    events = await run.io_bound(write_chapter_blocking)
except GenerationCancelled:
    ui.notify("Generation cancelled by user", type="warning")
```

## Testing

### Unit Tests (`tests/unit/test_generation_management.py`)
- Tests for all AppState generation control methods
- Tests for GenerationCancelled exception
- All tests passing

### Smoke Tests (`tests/smoke_test_async.py`)
- Verifies imports work
- Tests AppState methods
- Validates exception handling

Run tests:
```bash
pytest tests/unit/test_generation_management.py -v
python tests/smoke_test_async.py
```

## Future Enhancements

### Pause/Resume (Deferred)
The UI has pause/resume buttons, but backend support requires:
- Checkpoint saving in orchestrator
- Resume from checkpoint functionality
- State serialization improvements

### Progress Tracking
- Determinate progress bar (show percentage)
- Estimated time remaining
- Chapter-by-chapter progress for "Write All"

### Multiple Generations
- Queue system for multiple generation requests
- Parallel generation of non-conflicting chapters
- Better conflict detection

## Known Limitations

1. **Pause/Resume**: UI buttons present but backend not implemented
2. **Progress Percentage**: Currently indeterminate, needs orchestrator updates
3. **Cancellation Granularity**: Checks between major steps, not mid-agent
4. **State Persistence**: In-progress generations lost on app restart

## Migration Notes

No breaking changes. Existing code continues to work:
- `write_chapter()` without `cancel_check` works as before
- All existing tests pass
- No database schema changes

## Dependencies

No new dependencies added. Uses existing:
- NiceGUI 3.x (`run.io_bound()` for background tasks)
- Python asyncio (for async/await)
- Existing logging infrastructure

## Performance

- Generation runs in thread pool via `run.io_bound()`
- No blocking of event loop
- Minimal memory overhead (control flags only)
- Safe concurrent UI operations

## Security

- No new security concerns
- State changes validated
- No external API calls
- Cancellation is safe (no partial states)
