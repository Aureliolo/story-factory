# Story Factory - TODO

## High Priority

### World Page Improvements
- [ ] **Entity editor doesn't refresh on selection** - Clicking different entities shows stale data. The form fields need to be updated when selection changes in `ui/pages/world.py`
- [ ] **Graph node selection callback** - Node clicks may not trigger Python callbacks properly in NiceGUI context (`ui/components/graph.py:188-205`)

### Code Quality Pass
- [ ] **Full test coverage review** - Audit all modules for missing unit tests, especially:
  - `ui/pages/*.py` - UI component tests
  - `ui/components/*.py` - Component tests
  - `services/model_mode_service.py` - More edge cases
  - `workflows/orchestrator.py` - Integration tests
- [ ] **Logging audit** - Ensure consistent logging across all modules:
  - Add `logger.info()` for important operations
  - Add `logger.debug()` for detailed flow
  - Add `logger.error()` with `exc_info=True` for all exceptions
- [ ] **Exception handling review** - Audit for:
  - Bare `except:` clauses that should be specific
  - Missing try/except around I/O operations
  - Proper exception chaining with `raise ... from e`
  - User-friendly error messages in UI

## Medium Priority

### UI/UX Improvements
- [ ] **Dark mode graph colors** - Graph detection relies on multiple DOM selectors that may not work reliably with NiceGUI (`ui/graph_renderer.py:143-146`)
- [ ] **Empty state on World page** - Show guidance when no entities exist instead of just "No entities found"
- [ ] **Graph search highlighting** - When searching entities, highlight matching nodes in the graph
- [ ] **Relationship editing** - Allow viewing/editing relationship properties by clicking edges

### Performance
- [ ] **Graph initialization timeout** - Add timeout to prevent infinite retry loop if vis.js CDN fails to load (`ui/graph_renderer.py:135-140`)

## Low Priority

### Responsive Design
- [ ] **World page mobile layout** - Fixed column widths (20%/60%/20%) break on small screens (`ui/pages/world.py:70-86`)
- [ ] **Analytics page mobile** - Test and fix mobile responsiveness

### Features
- [ ] **Entity attributes editing** - Currently read-only, need JSON editor for attributes
- [ ] **Undo/redo system** - No way to undo entity creation/deletion
- [ ] **Graph node dragging** - Enable physics simulation for draggable nodes

### Code Quality
- [ ] **Type hints consistency** - Some functions have full type hints, others use `Any`
- [ ] **Graph renderer refactor** - Returns `tuple[str, str]` for HTML/JS, should use named tuple or dataclass
- [ ] **HTML sanitization** - `ui/pages/world.py:328` uses `sanitize=False`, verify safety

## Completed

- [x] Fix vis-network version mismatch (white graph)
- [x] Implement circular layout with calculated positions
- [x] Implement grid layout with calculated positions
- [x] Add "Check for Updates" button to Models page
- [x] Fix model pull progress None values crash
- [x] Auto-sync CDN versions via GitHub Action
- [x] Fix `finish_tracking` to persist performance metrics
- [x] Fix `save_custom_mode` to preserve `created_at`
