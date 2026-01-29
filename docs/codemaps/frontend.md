# Frontend Structure

> Generated: 2026-01-24 | Updated: 2026-01-29 | Freshness: Current

## Application Entry (`ui/app.py`)

### StoryFactoryApp (`ui/app.py`)

Main NiceGUI application with path-based routing and startup timing.

```python
app = create_app(services)  # Factory function
app.run(host="127.0.0.1", port=7860)
```

**Routes:**

| Path | Page Class | Package | Purpose |
|------|------------|---------|---------|
| `/` | `WritePage` | `write/` | Main writing interface |
| `/world` | `WorldPage` | `world/` | Entity/relationship editor |
| `/timeline` | `TimelinePage` | `timeline.py` | Story timeline |
| `/world-timeline` | `WorldTimelinePage` | `world_timeline.py` | World event timeline |
| `/projects` | `ProjectsPage` | `projects.py` | Project management |
| `/settings` | `SettingsPage` | `settings/` | Configuration with undo/redo |
| `/models` | `ModelsPage` | `models/` | Ollama model management |
| `/analytics` | `AnalyticsPage` | `analytics/` | Performance metrics |
| `/templates` | `TemplatesPage` | `templates.py` | Story templates |
| `/compare` | `ComparisonPage` | `comparison.py` | Model comparison |

**Layout Pattern:**

```python
def _page_layout(self, current_path, build_content):
    self._add_styles()
    self._apply_theme()
    shortcuts = KeyboardShortcuts(self.state, self.services)
    shortcuts.register()
    header = Header(self.state, self.services, current_path)
    header.build()
    with ui.column().classes("w-full flex-grow p-0"):
        build_content()
```

## State Management (`ui/state.py`)

### AppState

Centralized UI state dataclass:

**Project State:**
- `project_id`, `project`, `world_db`

**Interview State:**
- `interview_history`, `interview_complete`, `interview_processing`

**Writing State:**
- `current_chapter`, `is_writing`, `writing_progress`
- `generation_cancel_requested`, `generation_pause_requested`

**UI Navigation:**
- `active_tab`, `active_sub_tab`

**World Builder:**
- `selected_entity_id`, `entity_filter_types`, `graph_layout`

**Undo/Redo:**
- `_undo_stack`, `_redo_stack` (max 50 actions)
- `ActionType` enum for action types

**Callbacks:**
```python
state.on_project_change(callback)
state.on_entity_select(callback)
state.on_chapter_change(callback)
state.on_undo(callback)
state.on_redo(callback)
```

## Page Packages (`ui/pages/`)

### Page Protocol

```python
class Page(Protocol):
    def build(self) -> None: ...
```

All pages receive `AppState` and `ServiceContainer` in constructor.

### settings/ (~284 lines in __init__, 7 modules)

Configuration UI with modular sections and undo/redo support.

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 284 | Main SettingsPage with flexbox layout |
| `_connection.py` | ~100 | Ollama connection settings |
| `_models.py` | ~190 | Model selection and temperature |
| `_interaction.py` | ~200 | Interaction mode and context |
| `_modes.py` | ~250 | Generation mode and adaptive learning |
| `_advanced.py` | ~830 | Advanced LLM, world gen, story structure, data integrity |
| `_persistence.py` | ~320 | Save, snapshot, restore, undo/redo |

**Features:**
- 7-section flexbox layout (2 rows of responsive cards)
- Snapshot-based undo/redo for settings changes
- Modular section building functions
- Helper methods: `_section_header()`, `_build_number_input()`

### write/ (~1,896 lines, 6 modules)

Main writing interface with generation controls.

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 164 | WritePage controller |
| `_interview.py` | 248 | Interview phase UI |
| `_structure.py` | 291 | Structure phase UI |
| `_writing.py` | 683 | Writing/revision loop UI |
| `_generation.py` | 457 | Generation progress and UX (pause/resume/cancel) |
| `_export.py` | 53 | Export dialogs |

**Features:**
- Interview chat (when status="interview")
- Chapter outline/navigation
- Content editor with version history
- Generation controls (pause/resume/cancel)
- Feedback submission for regeneration

### world/ (~5,516 lines, 13 modules)

Comprehensive entity/relationship editor.

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 342 | WorldPage controller |
| `_browser.py` | 483 | Entity browser/list view |
| `_editor.py` | 538 | Entity editor UI |
| `_editor_ops.py` | 394 | Editor operations (CRUD) |
| `_analysis.py` | 278 | World analysis dashboard |
| `_graph.py` | 492 | Relationship graph visualization |
| `_generation.py` | 366 | Entity generation UI |
| `_gen_dialogs.py` | 611 | Generation dialog components |
| `_gen_entity_types.py` | 659 | Entity type-specific generation |
| `_gen_operations.py` | 449 | Generation backend operations |
| `_import.py` | 524 | Entity import from text |
| `_calendar.py` | 217 | World calendar UI |
| `_undo.py` | 163 | World entity undo/redo |

**Features:**
- Entity list with filtering by type
- Entity detail panel with editing
- Relationship graph visualization
- Quality refinement toggle
- World calendar management
- Entity undo/redo

### analytics/ (~1,264 lines, 7 modules)

Performance metrics dashboard.

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 241 | AnalyticsPage controller |
| `_summary.py` | 108 | Summary statistics |
| `_costs.py` | 234 | Token/cost tracking |
| `_model.py` | 159 | Model performance metrics |
| `_content.py` | 79 | Content statistics |
| `_trends.py` | 347 | Trend analysis |
| `_export.py` | 96 | Export analytics |

### models/ (~1,271 lines, 4 modules)

Ollama model management.

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 163 | ModelsPage controller |
| `_listing.py` | 331 | Model list view |
| `_download.py` | 431 | Model download UI |
| `_operations.py` | 346 | Model operations (load, pull, etc.) |

**Features:**
- Installed models list
- Download new models
- Delete models
- Model tagging for roles
- Health check status

### Standalone Pages

| Page | File | Lines | Purpose |
|------|------|-------|---------|
| `ProjectsPage` | `projects.py` | ~400 | Project management, backup/restore |
| `TemplatesPage` | `templates.py` | ~300 | Template browser and preview |
| `TimelinePage` | `timeline.py` | ~350 | Story timeline visualization |
| `WorldTimelinePage` | `world_timeline.py` | ~300 | World event timeline |
| `ComparisonPage` | `comparison.py` | ~400 | Model A/B testing |

## Components (`ui/components/`)

### Header (`ui/components/header.py` ~196 lines)

Navigation and project selector:
- Tab navigation
- Dark mode toggle
- Current project display

### Chat (`ui/components/chat.py` ~256 lines)

Interview chat interface:
- Message history display
- Input with send button
- Processing indicator

### EntityCard (`ui/components/entity_card.py` ~237 lines)

Entity display component:
- Type icon
- Name and description
- Edit/delete actions

### Graph Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Graph` | `graph.py` | 478 | Generic graph visualization |
| `ConflictGraph` | `conflict_graph.py` | 488 | Conflict/relationship visualization |

**Features:**
- NetworkX-based layout
- Entity nodes with colors by type
- Relationship edges
- Force-directed and hierarchical layouts

### Timeline Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Timeline` | `timeline.py` | 626 | Story timeline visualization |
| `WorldTimeline` | `world_timeline.py` | 524 | World event timeline |

**Features:**
- Horizontal event layout
- Chapter markers
- Zoom/pan controls
- Dark mode support

### Generation Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `GenerationStatus` | `generation_status.py` | 261 | Generation progress display |
| `BuildDialog` | `build_dialog.py` | 456 | World build confirmation |

**Features:**
- Phase indicator
- Progress bar
- ETA display
- Pause/cancel controls

### Other Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Common` | `common.py` | 290 | Shared UI utilities |
| `SceneEditor` | `scene_editor.py` | 452 | Scene-level editing |
| `RecommendationDialog` | `recommendation_dialog.py` | 248 | AI recommendation UI |
| `WorldHealthDashboard` | `world_health_dashboard.py` | 361 | World health metrics |

## Keyboard Shortcuts (`ui/keyboard_shortcuts.py`)

Global shortcuts:
- `Ctrl+Z`: Undo
- `Ctrl+Y`/`Ctrl+Shift+Z`: Redo
- `Ctrl+S`: Save project
- `Ctrl+N`: New project
- Navigation shortcuts

## Theming (`ui/theme.py`)

Theme utilities:
- `get_background_class()` → CSS class
- Dark/light mode detection
- Color constants

## Graph Rendering (`ui/graph_renderer.py`)

NetworkX to visualization:
- Force-directed layout
- Hierarchical layout
- Entity type coloring
- Edge styling

## Code Statistics

| Category | Files | Modules | Lines |
|----------|-------|---------|-------|
| Page Packages | 5 | 37 | ~10,231 |
| Standalone Pages | 5 | 5 | ~1,750 |
| Components | 13 | 13 | ~4,896 |
| Core UI | 5 | 5 | ~1,200 |
| **Total** | **28** | **60** | **~18,077** |
