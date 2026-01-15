# Story Factory - TODO

## Remaining Work

### Nice to Have

- [ ] **Graph node dragging** - Improve physics simulation for smoother draggable nodes

## Completed

### World Page Improvements (Recent)
- [x] Undo/redo system - Full undo/redo for entity and relationship CRUD operations

### World Page Improvements
- [x] Entity editor refreshes on selection - Added `_refresh_entity_editor()` method and refreshable container
- [x] Graph node selection callback - Uses NiceGUI's `emitEvent`/`ui.on()` event system
- [x] Empty state on World page - Shows helpful guidance when no entities exist
- [x] Graph search highlighting - Matching nodes highlighted with red border, non-matching dimmed
- [x] Relationship editing via edge clicks - Click edges to edit/delete relationships in dialog
- [x] Entity attributes JSON editor - Full JSON editor for entity attributes

### Graph & Visualization
- [x] Fix vis-network version mismatch (white graph)
- [x] Implement circular layout with calculated positions
- [x] Implement grid layout with calculated positions
- [x] Dark mode graph detection - Robust detection using body class, data-theme, computed colors
- [x] Graph initialization timeout - 5 second timeout with error message if CDN fails

### Security & Code Quality
- [x] HTML sanitization verified - Added `html.escape()` to all user content in graph tooltips/analysis results
- [x] Graph renderer refactor - Now returns `GraphRenderResult` dataclass instead of tuple
- [x] Logging audit complete - All services have proper logging with `logger.info/debug/warning`
- [x] Exception handling review - No bare except clauses, proper exception chaining in place
- [x] Type hints consistency - Added return types to main functions

### Test Coverage
- [x] Full test coverage review - Added 36 new tests for ModelModeService (263 tests total)
- [x] Fixed bugs found during testing - `time_seconds` and `tokens_per_second` None formatting

### Models & Services
- [x] Add "Check for Updates" button to Models page
- [x] Fix model pull progress None values crash
- [x] Auto-sync CDN versions via GitHub Action
- [x] Fix `finish_tracking` to persist performance metrics
- [x] Fix `save_custom_mode` to preserve `created_at`
- [x] Add `update_relationship` method to WorldDatabase
