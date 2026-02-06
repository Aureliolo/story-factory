# Story Factory Architecture

## Overview

Story Factory uses clean architecture principles with clear separation of concerns across three primary layers:

- **Services Layer**: Business logic, no UI knowledge, testable in isolation
- **UI Layer**: NiceGUI components, only calls services, manages presentation
- **Data Layer**: Models and persistence, database operations

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Write  │  │  World  │  │Projects │  │Settings │  ...   │
│  │  Page   │  │  Page   │  │  Page   │  │  Page   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │             │             │             │
│       └────────────┴─────────────┴─────────────┘             │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │ AppState  │ (UI State Management)      │
│                    └─────┬─────┘                            │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Service Container                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Project    │  │    Story     │  │    World     │      │
│  │   Service    │  │   Service    │  │   Service    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
│  ┌──────▼───────┐  ┌─────▼────────┐  ┌─────▼────────┐     │
│  │    Model     │  │    Export    │  │  Analytics   │     │
│  │   Service    │  │   Service    │  │   Service    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Story Orchestrator                          │
│             (services/orchestrator/__init__.py)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Multi-Agent Production Pipeline              │  │
│  │                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐                 │  │
│  │  │ Interviewer │───▶│ Architect   │                 │  │
│  │  └─────────────┘    └──────┬──────┘                 │  │
│  │                            │                         │  │
│  │                     ┌──────▼──────┐                 │  │
│  │                     │             │                 │  │
│  │                 ┌───▼────┐   ┌───▼────┐            │  │
│  │                 │ Writer │   │ Editor │            │  │
│  │                 └───┬────┘   └───┬────┘            │  │
│  │                     │            │                  │  │
│  │                     └────┬───────┘                  │  │
│  │                          │                          │  │
│  │                    ┌─────▼──────┐                  │  │
│  │                    │ Continuity │                  │  │
│  │                    │  Checker   │                  │  │
│  │                    └────────────┘                  │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Data & Persistence                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ StoryState   │  │    World     │  │    Model     │      │
│  │  (Pydantic)  │  │   Database   │  │  Performance │      │
│  └──────────────┘  │ (SQLite +    │  │   Database   │      │
│                    │  NetworkX)   │  └──────────────┘      │
│  ┌──────────────┐  └──────────────┘                        │
│  │   Settings   │                                           │
│  │ (JSON/Pydantic)│                                         │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    External Services                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Ollama (LLM)                       │  │
│  │              http://localhost:11434                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Communication Flow

```
User Action → UI Component → AppState → Service → Agent → Ollama
                                ↓          ↓       ↓       ↓
                           UI Update ←── Result ← Response ← LLM
```

## Directory Structure

```
story-factory/
├── agents/                  # AI agents
│   ├── base.py             # Base agent class
│   ├── interviewer.py      # Story interview
│   ├── architect.py        # Story structure
│   ├── writer.py           # Content generation
│   ├── editor.py           # Content editing
│   ├── continuity.py       # Consistency checking
│   └── validator.py        # Response validation
│
├── memory/                  # Data models and persistence
│   ├── story_state.py      # Story state (Pydantic models)
│   ├── entities.py         # Entity/Relationship models
│   ├── world_database.py   # SQLite + NetworkX database
│   ├── mode_database.py    # Model performance database
│   ├── mode_models.py      # Performance tracking models
│   ├── templates.py        # Template data models
│   ├── builtin_templates.py # Built-in story templates
│   └── world_quality.py    # World quality tracking
│
├── services/                # Business logic layer
│   ├── __init__.py         # ServiceContainer
│   ├── orchestrator/       # Multi-agent orchestration
│   │   ├── __init__.py     # Main StoryOrchestrator
│   │   ├── _interview.py   # Interview phase logic
│   │   ├── _structure.py   # Architecture phase logic
│   │   ├── _writing.py     # Writing phase logic
│   │   ├── _editing.py     # Editing and continuity logic
│   │   └── _persistence.py # State persistence helpers
│   ├── story_service/      # Story generation workflow
│   ├── world_service/      # Entity/world management
│   ├── world_quality_service/ # World quality enhancement
│   ├── project_service.py  # Project CRUD operations
│   ├── model_service.py    # Ollama model operations
│   ├── export_service.py   # Export to multiple formats
│   ├── model_mode_service/ # Model performance tracking
│   ├── scoring_service.py  # Quality scoring logic
│   ├── template_service.py # Story template management
│   ├── backup_service.py   # Project backup/restore
│   ├── import_service.py   # Import entities from text
│   ├── comparison_service.py # Model comparison testing
│   ├── suggestion_service.py # AI-powered suggestions
│   ├── timeline_service.py # Timeline management
│   ├── calendar_service.py # Calendar view
│   ├── conflict_analysis_service.py # Story conflict analysis
│   ├── content_guidelines_service.py # Content guidelines
│   ├── temporal_validation_service.py # Timeline validation
│   ├── world_template_service.py # World templates
│   └── llm_client.py       # Unified LLM client
│
├── ui/                      # NiceGUI UI layer
│   ├── __init__.py
│   ├── app.py              # Main app setup
│   ├── state.py            # Centralized UI state (AppState)
│   ├── theme.py            # Colors, styles, helpers
│   ├── styles.css          # Custom CSS styles
│   ├── graph_renderer.py   # vis.js graph rendering
│   ├── keyboard_shortcuts.py # Keyboard shortcut handling
│   ├── shortcuts.py        # Shortcut registry
│   ├── components/         # Reusable components
│   │   ├── __init__.py
│   │   ├── header.py       # Project selector + navigation
│   │   ├── chat.py         # Chat interface
│   │   ├── entity_card.py  # Entity display card
│   │   ├── graph.py        # Graph visualization wrapper
│   │   └── common.py       # Shared components (loading, dialogs, etc.)
│   └── pages/              # Full pages
│       ├── __init__.py
│       ├── write.py        # Fundamentals + Live Writing
│       ├── world.py        # World Builder
│       ├── projects.py     # Project management
│       ├── templates.py    # Story templates
│       ├── timeline.py     # Event timeline
│       ├── comparison.py   # Model comparison
│       ├── settings.py     # Settings configuration
│       ├── models.py       # Model management
│       └── analytics.py    # Model performance analytics
│
├── prompts/                 # Prompt templates
│   ├── __init__.py
│   └── templates/          # Prompt template files
│
├── utils/                   # Utilities
│   ├── logging_config.py   # Logging setup and context
│   ├── json_parser.py      # LLM JSON extraction
│   ├── error_handling.py   # Decorators and helpers
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── constants.py        # Shared constants
│   ├── environment.py      # Environment validation
│   ├── message_analyzer.py # Conversation analysis
│   ├── model_utils.py      # Model name utilities
│   ├── prompt_builder.py   # Prompt construction
│   ├── prompt_registry.py  # Prompt management
│   ├── prompt_template.py  # Template system for prompts
│   ├── text_analytics.py   # Text analysis utilities
│   └── validation.py       # Data validation helpers
│
├── tests/                   # Tests
│   ├── unit/               # Unit tests (fast, isolated)
│   ├── component/          # NiceGUI component tests (User fixture)
│   ├── integration/        # Integration tests
│   ├── e2e/                # End-to-end tests
│   └── conftest.py         # Shared pytest fixtures
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md     # This file
│   ├── MODELS.md           # Model recommendations
│   ├── CODE_QUALITY.md     # Code quality issues tracker
│   ├── TEST_QUALITY.md     # Testing standards
│   └── UX_UI_IMPROVEMENTS.md # UI/UX improvements
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
    project_id: str | None = None
    project: StoryState | None = None
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
        self.world = WorldService(settings)
        self.model = ModelService(settings)
        self.export = ExportService(settings)
        self.mode = ModelModeService(settings)
        self.scoring = ScoringService(self.mode)
        self.world_quality = WorldQualityService(settings, self.mode)
        self.suggestion = SuggestionService(settings)
        self.template = TemplateService(settings)
        self.backup = BackupService(settings)
        self.import_svc = ImportService(settings, self.mode)
        self.comparison = ComparisonService(settings)

# main.py - Services created once and passed to UI
settings = Settings.load()
services = ServiceContainer(settings)
app = create_app(services)
```

### 3. Event Flow

**User Interaction Flow**:
```
User Click → UI Handler → Service Method → Return Result → UI Update
```

**Story Generation Flow**:
```
1. User inputs → Interviewer Agent → StoryBrief
2. StoryBrief → Architect Agent → Story Structure + Characters
3. Structure → Writer Agent → Chapter Draft
4. Draft → Editor Agent → Polished Chapter
5. Chapter → Continuity Agent → Validation / Issues
6. If issues → Back to Writer (max 3 iterations)
7. Final Chapter → StoryState → Save to JSON
```

**World Building Flow**:
```
1. User creates entity → WorldService.create_entity()
2. Service validates → WorldDatabase.add_entity()
3. Database stores in SQLite + adds to NetworkX graph
4. Graph updates → UI displays visualization
5. User queries relationships → PathFinding algorithm
6. Results displayed in graph component
```

**Model Selection Flow**:
```
1. User selects agent role → Settings.agent_models[role]
2. If "auto" → ModelRegistry.get_recommended_for_role()
3. Registry checks available models → Matches to role requirements
4. Returns best available model → Agent uses for generation
```

### 4. No Global State
- Services are stateless (receive what they need)
- UI state is explicit and centralized in AppState
- No hidden singletons
- Settings passed via dependency injection
- Orchestrators cached with LRU to prevent memory leaks

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

### Unit Tests (`tests/unit/`)
- Services: Mock agents, test business logic
- Models: Validation, serialization
- Utilities: JSON parsing, error handling

### Component Tests (`tests/component/`)
- NiceGUI component tests using User fixture
- Page rendering tests
- Dark mode styling verification
- Chat interface tests

### Integration Tests (`tests/integration/`)
- App startup and initialization
- Settings validation
- Service container wiring

### E2E Tests (`tests/e2e/`)
- Full browser-based testing (optional)
- Complete user workflows

## Key Design Patterns

### 1. Service Container (Dependency Injection)

**Pattern**: All services created once in a container, injected where needed.

**Benefits**:
- Single source of truth for service instances
- Easy to mock services in tests
- Clear dependency management
- No circular dependencies

**Example**:
```python
# services/__init__.py
class ServiceContainer:
    def __init__(self, settings: Settings):
        self.project = ProjectService(settings)
        self.story = StoryService(settings)
        # ... all services

# main.py
services = ServiceContainer(settings)
app = create_app(services)

# ui/pages/write.py
class WritePage:
    def __init__(self, app_state: AppState, services: ServiceContainer):
        # Access any service through the container
        self.story_service = services.story
```

### 2. Multi-Agent Orchestration

**Pattern**: Orchestrator coordinates multiple specialized AI agents in a pipeline.

**Agents**:
- **Interviewer**: Conversational, gathers requirements (temp: 0.7)
- **Architect**: Logical, designs structure (temp: 0.85)
- **Writer**: Creative, generates prose (temp: 0.9)
- **Editor**: Balanced, polishes content (temp: 0.6)
- **Continuity**: Analytical, checks consistency (temp: 0.3)

**Benefits**:
- Each agent optimized for its specific task
- Different models can be assigned per role
- Temperature tuned to task requirements
- Clear separation of responsibilities

**Flow**:
```python
orchestrator = StoryOrchestrator(settings)
orchestrator.create_new_story()
orchestrator.start_interview()  # Interviewer
orchestrator.build_story_structure()  # Architect
for event in orchestrator.write_all_chapters():  # Writer → Editor → Continuity
    # Process events
```

### 3. State Management with Pydantic

**Pattern**: Use Pydantic models for all state objects with built-in validation.

**Benefits**:
- Type safety with runtime validation
- Automatic serialization/deserialization
- Clear data contracts
- IDE support with type hints

**Example**:
```python
class StoryState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    brief: StoryBrief | None = None
    chapters: list[Chapter] = Field(default_factory=list)

    def add_chapter(self, chapter: Chapter) -> None:
        """Add chapter with validation."""
        self.chapters.append(chapter)
```

### 4. Graph-Based World Database

**Pattern**: SQLite for persistence + NetworkX for graph operations.

**Benefits**:
- Efficient storage and querying
- Graph algorithms (pathfinding, clustering)
- Relationship visualization
- Thread-safe with RLock

**Operations**:
```python
# Add entity
world_db.add_entity(character)

# Add relationship
world_db.add_relationship("char1", "char2", "ally")

# Find paths
path = world_db.find_path("char1", "char5")

# Detect communities
communities = world_db.get_communities()
```

### 5. Error Handling with Decorators

**Pattern**: Reusable decorators for common error scenarios.

**Benefits**:
- Consistent error handling
- Retry logic with exponential backoff
- Graceful degradation
- Clean code without try/except clutter

**Example**:
```python
from utils.error_handling import handle_ollama_errors, retry_with_fallback

@handle_ollama_errors(default_value=None)
@retry_with_fallback(max_retries=3)
async def generate_content(prompt: str) -> str:
    # Automatically retries on connection errors
    # Falls back to default value on failure
    return await self.client.generate(prompt)
```

### 6. Centralized UI State

**Pattern**: Single AppState class manages all UI-specific state.

**Benefits**:
- Predictable state updates
- Easy to debug
- No prop drilling
- Clear data flow

**Example**:
```python
class AppState:
    current_project: StoryState | None = None
    selected_entity_id: str | None = None
    active_tab: str = "write"

    def load_project(self, project: StoryState):
        self.current_project = project
        self.selected_entity_id = None  # Reset selection
```

### 7. LRU Caching for Orchestrators

**Pattern**: Cache orchestrator instances to prevent memory leaks.

**Why**: Creating new orchestrators for every operation leads to memory accumulation.

**Implementation**:
```python
from functools import lru_cache

@lru_cache(maxsize=10)
def get_orchestrator(project_id: str) -> StoryOrchestrator:
    # Reuses orchestrator for same project
    # Automatically evicts oldest when cache is full
    return StoryOrchestrator(settings)
```

### 8. Async/Await for Non-Blocking Operations

**Pattern**: Use async for I/O-bound operations (LLM calls, file I/O).

**Benefits**:
- UI remains responsive
- Background generation
- Parallel operations when possible

**Example**:
```python
async def generate_chapter(self, chapter_num: int) -> str:
    # Non-blocking, UI can update during generation
    draft = await self.writer.generate(prompt)
    edited = await self.editor.refine(draft)
    return edited
```

### 9. Template Method Pattern (BaseAgent)

**Pattern**: Base class defines algorithm structure, subclasses implement specifics.

**Benefits**:
- Consistent agent behavior
- Shared retry logic
- Centralized rate limiting
- Easy to add new agents

**Example**:
```python
class BaseAgent:
    def generate(self, prompt: str) -> str:
        # Template method with hooks
        self._before_generate()
        result = self._call_llm(prompt)
        self._after_generate(result)
        return result

    def _call_llm(self, prompt: str) -> str:
        # Implemented by base class
        pass

class WriterAgent(BaseAgent):
    def _before_generate(self):
        # Writer-specific setup
        self.set_temperature(0.9)
```

### 10. Observer Pattern for Event Streaming

**Pattern**: Yield events during long-running operations for UI updates.

**Benefits**:
- Real-time progress feedback
- User can see what's happening
- Cancellable operations

**Example**:
```python
def write_all_chapters(self) -> Generator[WorkflowEvent, None, None]:
    for chapter_num in range(1, total_chapters + 1):
        yield WorkflowEvent(
            agent_name="Writer",
            message=f"Writing chapter {chapter_num}..."
        )
        chapter = self._write_chapter(chapter_num)
        yield WorkflowEvent(
            agent_name="Writer",
            message=f"Chapter {chapter_num} complete"
        )
```

## Performance Considerations

### Memory Management

1. **LRU Caching**: Orchestrators cached to prevent memory leaks
2. **Model Unloading**: Ollama automatically manages VRAM
3. **Context Windows**: Configurable to balance quality vs. memory
4. **Lazy Loading**: UI components load data on-demand

### Optimization Strategies

1. **Rate Limiting**: Max 2 concurrent LLM requests
2. **Batch Operations**: Group similar operations when possible
3. **Caching**: Cache frequently accessed data (project lists, settings)
4. **Streaming**: Use generators for large operations
5. **Background Tasks**: Run long operations asynchronously

### Scalability

Current limitations:
- Single user (local application)
- One active story generation at a time (VRAM)
- File-based storage (suitable for personal use)

Future considerations:
- Database backend (PostgreSQL) for multi-user
- Job queue (Redis) for multiple concurrent stories
- Cloud deployment (GPU instances)

## Security Considerations

### Local-First Security

- **No external API calls**: All processing local
- **No data collection**: Complete privacy
- **No authentication**: Single-user desktop app
- **File permissions**: Standard OS-level security

### Input Validation

- **Pydantic models**: Runtime type validation
- **File paths**: Validated to prevent path traversal
- **User input**: Sanitized before LLM prompts
- **JSON parsing**: Safe parsing with error handling

### Future Considerations

For multi-user deployment:
- Authentication (JWT, OAuth)
- Authorization (role-based access)
- API rate limiting
- Input sanitization for XSS
- SQL injection prevention (already using parameterized queries)

## Extensibility

### Adding New Features

The architecture makes it easy to add:

1. **New Agent**: Extend `BaseAgent`, add to orchestrator
2. **New Service**: Create service, add to `ServiceContainer`
3. **New UI Page**: Create page class, add route in `app.py`
4. **New Export Format**: Add method to `ExportService`
5. **New Analytics**: Extend `ScoringService` or `ModelModeService`

### Plugin System (Future)

Potential plugin architecture:
- Plugin discovery via entry points
- Agent plugins (custom agents)
- Export plugins (custom formats)
- Template plugins (custom genres)
- UI theme plugins

## Related Documentation

- **[README.md](../README.md)**: Getting started, features, usage
- **[MODELS.md](MODELS.md)**: Model selection and recommendations
- **[TROUBLESHOOTING.md](../TROUBLESHOOTING.md)**: Common issues and solutions
- **[CONTRIBUTING.md](../CONTRIBUTING.md)**: Development guidelines
