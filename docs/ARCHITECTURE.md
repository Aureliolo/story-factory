# Story Factory Architecture

## Overview

Clean architecture with separation of concerns:
- **Services Layer**: Business logic, no UI knowledge
- **UI Layer**: NiceGUI components, only calls services
- **Data Layer**: Models and persistence

## Directory Structure

```
story-factory/
├── agents/                  # AI agents (unchanged)
│   ├── base.py             # Base agent class
│   ├── interviewer.py      # Story interview
│   ├── architect.py        # Story structure
│   ├── writer.py           # Content generation
│   ├── editor.py           # Content editing
│   ├── continuity.py       # Consistency checking
│   └── validator.py        # Language validation
│
├── memory/                  # Data models (unchanged)
│   ├── story_state.py      # Story state model
│   ├── entities.py         # Entity models
│   └── world_database.py   # SQLite + NetworkX
│
├── services/                # NEW: Business logic layer
│   ├── __init__.py
│   ├── project_service.py  # Project CRUD, listing
│   ├── story_service.py    # Story generation workflow
│   ├── world_service.py    # World/entity management
│   ├── model_service.py    # Ollama model operations
│   └── export_service.py   # Export to various formats
│
├── ui/                      # NEW: NiceGUI UI layer
│   ├── __init__.py
│   ├── app.py              # Main app setup
│   ├── state.py            # Centralized UI state
│   ├── theme.py            # Colors, styles
│   ├── components/         # Reusable components
│   │   ├── __init__.py
│   │   ├── header.py       # Project selector + status
│   │   ├── chat.py         # Chat interface
│   │   ├── entity_card.py  # Entity display card
│   │   └── graph.py        # Graph visualization
│   └── pages/              # Full pages
│       ├── __init__.py
│       ├── write.py        # Fundamentals + Live Writing
│       ├── world.py        # World Builder
│       ├── projects.py     # Project management
│       ├── settings.py     # Settings
│       └── models.py       # Model management
│
├── workflows/               # Orchestration (simplified)
│   └── orchestrator.py     # Coordinates agents
│
├── utils/                   # Utilities (unchanged)
│   ├── logging_config.py
│   ├── json_parser.py
│   └── error_handling.py
│
├── tests/                   # Tests
│   ├── unit/               # Unit tests
│   │   ├── test_services/
│   │   ├── test_memory/
│   │   └── test_*.py       # Unit test files
│   └── conftest.py         # Pytest fixtures
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md
│   ├── MODELS.md
│   └── UX_UI_IMPROVEMENTS.md
│
├── scripts/                 # Utility scripts
│   ├── healthcheck.py
│   └── start.ps1
│
├── settings.py              # Configuration
└── main.py                  # Entry point
```

## Layer Responsibilities

### Services Layer
- **No UI imports** - pure Python business logic
- **Receives settings via dependency injection**
- **Returns domain objects or simple types**
- **Handles all Ollama/AI interactions**

### UI Layer
- **Only calls services** - no direct agent/orchestrator access
- **Manages UI state** - what's selected, what's visible
- **Handles user events** - button clicks, form submissions
- **Renders components** - displays data from services

### Data Layer
- **Pydantic models** for validation
- **SQLite** for world database
- **JSON files** for story persistence

## Key Design Decisions

### 1. State Management
```python
# ui/state.py
class AppState:
    """Centralized UI state."""
    current_project_id: str | None = None
    current_project: StoryState | None = None
    world_db: WorldDatabase | None = None

    # UI-only state
    selected_entity_id: str | None = None
    active_tab: str = "write"
    feedback_mode: str = "per-chapter"
```

### 2. Service Container Pattern
```python
# services/__init__.py
class ServiceContainer:
    """Dependency injection container for all services.

    Creates and holds instances of all service classes,
    injecting settings as needed. This provides:
    - Single place to initialize all services
    - Easy dependency management
    - Consistent settings across services
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.project = ProjectService(settings)
        self.story = StoryService(settings)
        self.world = WorldService()
        self.model = ModelService(settings)
        self.export = ExportService()

# main.py - Services created once and passed to UI
settings = Settings.load()
services = ServiceContainer(settings)
app = create_app(services)
```

### 3. Event Flow
```
User Click → UI Handler → Service Method → Return Result → UI Update
```

### 4. No Global State
- Services are stateless (receive what they need)
- UI state is explicit and centralized in AppState
- No hidden singletons
- Settings passed via dependency injection

## Error Handling Strategy

### 1. Service Layer
Services use explicit error handling with custom exceptions:
```python
# Ollama connection errors
from utils.error_handling import handle_ollama_errors

@handle_ollama_errors(default_value=None)
def get_models() -> list[str]:
    return ollama.list()
```

### 2. UI Layer
UI components catch exceptions and show user-friendly notifications:
```python
try:
    result = self.services.story.write_chapter(state, chapter_num)
    ui.notify("Chapter written!", type="positive")
except Exception as e:
    logger.error(f"Write failed: {e}")
    ui.notify(f"Error: {e}", type="negative")
```

### 3. Agent Layer
Agents use retry logic and validation:
- Retry with exponential backoff on Ollama errors
- Response validation for language/content correctness
- Graceful degradation (skip validation if validator fails)

### 4. Logging
- Module-level loggers: `logger = logging.getLogger(__name__)`
- Centralized configuration in utils/logging_config.py
- Default: INFO level, file: logs/story_factory.log
- Configurable via CLI args: --log-level, --log-file

## API Patterns

### Services
```python
class ProjectService:
    def __init__(self, settings: Settings): ...
    def create_project(self, name: str) -> StoryState: ...
    def load_project(self, project_id: str) -> StoryState: ...
    def list_projects(self) -> list[ProjectSummary]: ...
    def delete_project(self, project_id: str) -> bool: ...
```

### UI Components
```python
def entity_card(entity: Entity, on_edit: Callable, on_delete: Callable):
    """Reusable entity card component."""
    with ui.card():
        ui.label(entity.name)
        # ...
```

### Pages
```python
class WritePage:
    def __init__(self, state: AppState, services: ServiceContainer):
        self.state = state
        self.services = services

    def build(self):
        """Build the page UI."""
        with ui.tabs() as tabs:
            with ui.tab_panel("Fundamentals"):
                self._build_fundamentals()
            with ui.tab_panel("Live Writing"):
                self._build_live_writing()
```

## Testing Strategy

### Unit Tests
- Services: Mock agents, test logic
- Models: Validation, serialization

### Integration Tests
- Full workflow with mock Ollama
- UI interaction tests

### E2E Tests (optional)
- Playwright for browser testing
