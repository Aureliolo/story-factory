# Frontend Structure

> Generated: 2026-01-24 | Updated: 2026-01-24 | Freshness: Current

## Application Entry (`ui/app.py`)

### StoryFactoryApp (`ui/app.py:35-247`)

Main NiceGUI application with path-based routing.

```python
app = create_app(services)  # Factory function
app.run(host="127.0.0.1", port=7860)
```

**Routes:**

| Path | Page Class | Purpose |
|------|------------|---------|
| `/` | `WritePage` | Main writing interface |
| `/world` | `WorldPage` | Entity/relationship editor |
| `/timeline` | `TimelinePage` | Story timeline |
| `/projects` | `ProjectsPage` | Project management |
| `/settings` | `SettingsPage` | Configuration |
| `/models` | `ModelsPage` | Ollama model management |
| `/analytics` | `AnalyticsPage` | Performance metrics |
| `/templates` | `TemplatesPage` | Story templates |
| `/compare` | `ComparisonPage` | Model comparison |

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

### AppState (`ui/state.py:46-380`)

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

## Pages (`ui/pages/`)

### Page Protocol

```python
class Page(Protocol):
    def build(self) -> None: ...
```

All pages receive `AppState` and `ServiceContainer` in constructor.

### WritePage (`ui/pages/write.py`)

Main writing interface:
- Interview chat (when status="interview")
- Chapter outline/navigation
- Content editor with version history
- Generation controls (pause/resume/cancel)
- Feedback submission for regeneration

### WorldPage (`ui/pages/world.py`)

Entity/relationship editor:
- Entity list with filtering
- Entity detail panel
- Relationship graph visualization
- Add/edit/delete operations
- Quality refinement toggle

### ProjectsPage (`ui/pages/projects.py`)

Project management:
- Project list with metadata
- Create new project
- Load/delete projects
- Backup/restore functionality

### ModelsPage (`ui/pages/models.py`)

Ollama model management:
- Installed models list
- Download new models
- Delete models
- Model tagging for roles
- Health check status

### SettingsPage (`ui/pages/settings.py`)

Configuration UI:
- Ollama connection settings
- Per-agent model assignment
- Temperature controls
- World generation limits
- Timeout configuration

### TemplatesPage (`ui/pages/templates.py`)

Story template browser:
- Built-in templates
- Template preview
- Apply to new project

### AnalyticsPage (`ui/pages/analytics.py`)

Performance metrics:
- Model performance history
- Generation time tracking
- Quality score trends

### TimelinePage (`ui/pages/timeline.py`)

Story timeline visualization:
- Event chronology
- Chapter markers
- Entity participation

### ComparisonPage (`ui/pages/comparison.py`)

Model A/B testing:
- Side-by-side generation
- Quality comparison
- Performance metrics

## Components (`ui/components/`)

### Header (`ui/components/header.py`)

Navigation and project selector:
- Tab navigation
- Dark mode toggle
- Current project display

### Chat (`ui/components/chat.py`)

Interview chat interface:
- Message history display
- Input with send button
- Processing indicator

### EntityCard (`ui/components/entity_card.py`)

Entity display component:
- Type icon
- Name and description
- Edit/delete actions

### Graph (`ui/components/graph.py`)

Relationship graph visualization:
- NetworkX-based layout
- Entity nodes with colors by type
- Relationship edges

### Timeline (`ui/components/timeline.py`)

Timeline visualization component:
- Horizontal event layout
- Chapter markers
- Zoom/pan controls

### GenerationStatus (`ui/components/generation_status.py`)

Generation progress display:
- Phase indicator
- Progress bar
- ETA display
- Pause/cancel controls

### SceneEditor (`ui/components/scene_editor.py`)

Scene-level editing:
- Scene list with drag-drop
- Content editing
- Beat tracking

### Common (`ui/components/common.py`)

Shared UI utilities:
- Button styles
- Card layouts
- Modal dialogs

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
