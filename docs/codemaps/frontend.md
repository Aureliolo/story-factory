# Frontend Structure

<!-- Generated: 2026-01-24 | Updated: 2026-02-20 | Files scanned: 222 | Token estimate: ~850 -->

## Application Entry (`ui/app.py`, 339 lines)

### StoryFactoryApp

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

## State Management (`ui/state.py`, 500 lines)

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

### settings/ (~2,892 lines, 8 modules)

Configuration UI with modular sections and undo/redo support.

| Module | Lines | Purpose |
|--------|-------|---------|
| `_advanced.py` | 931 | Advanced LLM, world gen, story structure, data integrity |
| `__init__.py` | 512 | Main SettingsPage with flexbox layout |
| `_persistence.py` | 370 | Save, snapshot, restore, undo/redo |
| `_world_validation.py` | 301 | Temporal validation settings UI |
| `_modes.py` | 248 | Generation mode and adaptive learning |
| `_interaction.py` | 221 | Interaction mode and context |
| `_models.py` | 198 | Model selection and temperature |
| `_connection.py` | 111 | Ollama connection settings |

### write/ (~1,925 lines, 6 modules)

Main writing interface with generation controls.

| Module | Lines | Purpose |
|--------|-------|---------|
| `_writing.py` | 704 | Writing/revision loop UI |
| `_generation.py` | 474 | Generation progress (pause/resume/cancel) |
| `_structure.py` | 277 | Structure phase UI |
| `_interview.py` | 252 | Interview phase UI |
| `__init__.py` | 165 | WritePage controller |
| `_export.py` | 53 | Export dialogs |

### world/ (~5,909 lines, 13 modules)

Comprehensive entity/relationship editor.

| Module | Lines | Purpose |
|--------|-------|---------|
| `_gen_entity_types.py` | 786 | Entity type-specific generation |
| `_gen_dialogs.py` | 631 | Generation dialog components |
| `_browser.py` | 583 | Entity browser/list view |
| `_editor.py` | 530 | Entity editor UI |
| `_import.py` | 527 | Entity import from text |
| `_graph.py` | 496 | Relationship graph visualization |
| `_gen_operations.py` | 468 | Generation backend operations |
| `_editor_ops.py` | 388 | Editor operations (CRUD) |
| `__init__.py` | 376 | WorldPage controller |
| `_generation.py` | 370 | Entity generation UI |
| `_analysis.py` | 362 | World analysis dashboard |
| `_calendar.py` | 229 | World calendar UI |
| `_undo.py` | 163 | World entity undo/redo |

### analytics/ (~1,320 lines, 7 modules)

Performance metrics dashboard.

| Module | Lines | Purpose |
|--------|-------|---------|
| `_trends.py` | 347 | Trend analysis |
| `__init__.py` | 301 | AnalyticsPage controller |
| `_costs.py` | 230 | Token/cost tracking |
| `_model.py` | 159 | Model performance metrics |
| `_summary.py` | 108 | Summary statistics |
| `_export.py` | 96 | Export analytics |
| `_content.py` | 79 | Content statistics |

### models/ (~1,327 lines, 4 modules)

Ollama model management.

| Module | Lines | Purpose |
|--------|-------|---------|
| `_download.py` | 431 | Model download UI |
| `_listing.py` | 391 | Model list view |
| `_operations.py` | 342 | Model operations (load, pull, etc.) |
| `__init__.py` | 163 | ModelsPage controller |

### Standalone Pages

| Page | File | Lines | Purpose |
|------|------|-------|---------|
| `ProjectsPage` | `projects.py` | 720 | Project management, backup/restore |
| `ComparisonPage` | `comparison.py` | 522 | Model A/B testing |
| `TemplatesPage` | `templates.py` | 461 | Template browser and preview |
| `TimelinePage` | `timeline.py` | 249 | Story timeline visualization |
| `WorldTimelinePage` | `world_timeline.py` | 226 | World event timeline |

## Components (`ui/components/`) ~5,241 lines

### Header (`header.py`, 193 lines)

Navigation and project selector:
- Tab navigation
- Dark mode toggle
- Current project display

### Chat (`chat.py`, 254 lines)

Interview chat interface:
- Message history display
- Input with send button
- Processing indicator

### EntityCard (`entity_card.py`, 234 lines)

Entity display component:
- Type icon
- Name and description
- Edit/delete actions

### Graph Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Graph` | `graph.py` | 549 | Generic graph visualization |
| `ConflictGraph` | `conflict_graph.py` | 521 | Conflict/relationship visualization |

### Timeline Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Timeline` | `timeline.py` | 608 | Story timeline visualization |
| `WorldTimeline` | `world_timeline.py` | 571 | World event timeline |

### Generation Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `GenerationStatus` | `generation_status.py` | 261 | Generation progress display |
| `BuildDialog` | `build_dialog.py` | 466 | World build confirmation |
| `WorldHealthDashboard` | `world_health_dashboard.py` | 573 | World health + temporal metrics |

### Other Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `Common` | `common.py` | 290 | Shared UI utilities |
| `SceneEditor` | `scene_editor.py` | 450 | Scene-level editing |
| `RecommendationDialog` | `recommendation_dialog.py` | 248 | AI recommendation UI |

## Graph Renderer (`ui/graph_renderer/`) ~914 lines

| Module | Lines | Purpose |
|--------|-------|---------|
| `_renderer.py` | 530 | NetworkX to visualization |
| `_results.py` | 237 | Render result models |
| `_layout.py` | 116 | Force-directed, hierarchical layouts |
| `__init__.py` | 31 | Package exports |

## Core UI Files

| File | Lines | Purpose |
|------|-------|---------|
| `state.py` | 500 | Centralized state with undo/redo |
| `app.py` | 339 | App routing, page layout, startup timing |
| `theme.py` | 274 | Theme utilities, dark/light mode |
| `shortcuts.py` | 244 | Shortcut definitions |
| `local_prefs.py` | 176 | Browser-local preferences |
| `keyboard_shortcuts.py` | 175 | Global keyboard shortcut registration |

## Code Statistics

| Category | Files | Modules | Lines |
|----------|-------|---------|-------|
| Page Packages | 5 | 38 | ~13,373 |
| Standalone Pages | 5 | 5 | ~2,178 |
| Components | 14 | 14 | ~5,241 |
| Graph Renderer | 4 | 4 | ~914 |
| Core UI | 6 | 6 | ~1,708 |
| **Total** | **34** | **67** | **~23,414** |
