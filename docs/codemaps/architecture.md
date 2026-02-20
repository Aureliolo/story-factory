# Story Factory Architecture

<!-- Generated: 2026-01-24 | Updated: 2026-02-20 | Files scanned: 222 | Token estimate: ~900 -->

## System Overview

Story Factory is a local AI-powered multi-agent system for generating stories using Ollama. Five specialized agents collaborate through an iterative write-edit-check loop.

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py (Entry Point)                  │
│  - CLI mode: StoryOrchestrator direct control               │
│  - Web mode: NiceGUI on http://localhost:7860               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              ServiceContainer (DI Container)                │
│  settings/ → Settings.load() → ServiceContainer(settings)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Services │    │   UI    │     │ Agents  │
    │  Layer  │    │  Layer  │     │  Layer  │
    └─────────┘     └─────────┘     └─────────┘
```

## Layer Architecture

### Entry Layer
| File | Purpose |
|------|---------|
| `main.py` | CLI/Web entrypoint, argument parsing, startup timing |
| `settings/` | Configuration package (8 modules, ~2,600 lines) |

### Services Layer (`services/`)

Business logic, no UI imports. Receives settings via DI. **21 services** in container.

| Service | Package/File | Responsibility |
|---------|--------------|---------------|
| `ProjectService` | `project_service.py` | Project CRUD, persistence |
| `StoryService` | `story_service/` | Story generation workflow (5 modules) |
| `WorldService` | `world_service/` | Entity/relationship management (10 modules) |
| `ModelService` | `model_service.py` | Ollama model operations |
| `ExportService` | `export_service/` | Format exports (6 modules) |
| `ModelModeService` | `model_mode_service/` | Model performance tracking (6 modules) |
| `ScoringService` | `scoring_service.py` | Quality scoring |
| `WorldQualityService` | `world_quality_service/` | Entity quality enhancement (20 modules) |
| `SuggestionService` | `suggestion_service.py` | AI-powered suggestions |
| `TemplateService` | `template_service.py` | Story template management |
| `BackupService` | `backup_service.py` | Project backup/restore |
| `ImportService` | `import_service.py` | Import entities from text |
| `ComparisonService` | `comparison_service.py` | Model comparison testing |
| `TimelineService` | `timeline_service.py` | Timeline aggregation and temporal context |
| `ConflictAnalysisService` | `conflict_analysis_service.py` | Relationship classification |
| `WorldTemplateService` | `world_template_service.py` | World template management |
| `ContentGuidelinesService` | `content_guidelines_service.py` | Content standards enforcement |
| `CalendarService` | `calendar_service.py` | Fictional calendar generation |
| `TemporalValidationService` | `temporal_validation_service.py` | Temporal consistency validation |
| `EmbeddingService` | `embedding_service.py` | Vector embeddings via Ollama |
| `ContextRetrievalService` | `context_retrieval_service.py` | RAG context retrieval for agent prompts |

### UI Layer (`ui/`)
NiceGUI components, calls services only, manages state via AppState.

| Page | Route | Package | Purpose |
|------|-------|---------|---------|
| `WritePage` | `/` | `write/` (6 modules) | Main writing interface |
| `WorldPage` | `/world` | `world/` (13 modules) | Entity/relationship editor |
| `TimelinePage` | `/timeline` | `timeline.py` | Story timeline visualization |
| `WorldTimelinePage` | `/world-timeline` | `world_timeline.py` | World event timeline |
| `ProjectsPage` | `/projects` | `projects.py` | Project management |
| `SettingsPage` | `/settings` | `settings/` (8 modules) | Configuration UI with undo/redo |
| `ModelsPage` | `/models` | `models/` (4 modules) | Ollama model management |
| `AnalyticsPage` | `/analytics` | `analytics/` (7 modules) | Performance metrics |
| `TemplatesPage` | `/templates` | `templates.py` | Template browser |
| `ComparisonPage` | `/compare` | `comparison.py` | Model comparison |

### Agents Layer (`agents/`)
AI agent implementations extending BaseAgent with retry logic.

| Agent | Role | Temperature |
|-------|------|-------------|
| `InterviewerAgent` | Gathers story requirements | 0.7 |
| `ArchitectAgent` | Designs structure/characters (modularized) | 0.85 |
| `WriterAgent` | Writes prose | 0.9 |
| `EditorAgent` | Polishes and refines | 0.6 |
| `ContinuityAgent` | Detects plot holes, temporal validation | 0.3 |
| `ValidatorAgent` | Validates AI responses | 0.1 |

### Orchestration Layer (`services/orchestrator/`)
Orchestration of agent pipelines.

| Component | Package | Purpose |
|-----------|---------|---------|
| `StoryOrchestrator` | `services/orchestrator/` (6 modules) | Coordinates agents through story creation |
| `WorkflowEvent` | `services/orchestrator/` | Event model for UI updates |

### Memory Layer (`memory/`)
Pydantic models and persistent storage.

| Component | Package/File | Purpose |
|-----------|--------------|---------|
| `StoryState` | `story_state.py` | Complete story context model |
| `StoryBrief` | `story_state.py` | Initial story configuration |
| `Chapter`, `Scene` | `story_state.py` | Content structure models |
| `Character`, `Faction` | `story_state.py` | Entity models |
| `WorldDatabase` | `world_database/` (11 modules) | SQLite + NetworkX + sqlite-vec graph DB |
| `Entity`, `Relationship` | `entities.py` | Core entity models |
| `ModeDatabase` | `mode_database/` (10 modules) | Model performance tracking |
| `WorldCalendar` | `world_calendar.py` | Fictional calendar system |
| `WorldSettings` | `world_settings.py` | World configuration |
| `WorldHealth` | `world_health.py` | World health metrics |

## Agent Workflow

```
User Input → Interviewer → Architect → [Writer → Editor → Continuity] × N → Final Story
                                            └────── revision loop ───────┘
```

**RAG Context Pipeline:**

```plaintext
WorldDatabase (sqlite-vec) → EmbeddingService → ContextRetrievalService → StoryOrchestrator → Agent prompts
```

**Phase Weights (Progress):**
- Interview: 10%
- Architect: 15%
- Writer: 50%
- Editor: 15%
- Continuity: 10%

## Key Patterns

1. **Service Container DI**: 21 services instantiated once, injected into UI
2. **AppState**: Centralized UI state with callbacks for reactivity
3. **BaseAgent**: Retry logic, rate limiting (max 2 concurrent), configurable timeout
4. **WorldDatabase**: SQLite + NetworkX + sqlite-vec, thread-safe with RLock, versioning
5. **ModeDatabase**: Model performance tracking with cost analytics
6. **LRU Cache**: Orchestrator caching to prevent memory leaks
7. **Prompt Registry**: Centralized YAML template management with Jinja2
8. **Exception Hierarchy**: `StoryFactoryError` → `LLMError`, `ExportError`, etc.
9. **Instructor Integration**: Pydantic-validated structured LLM outputs
10. **Quality Refinement Loop**: Iterative entity improvement with scoring
11. **Auto Model Selection**: Tag-based model selection per agent role
12. **Temporal Validation**: Calendar-aware consistency checking wired into continuity
13. **Undo/Redo System**: Snapshot-based state management for settings and world entities
14. **Settings Backup**: Auto-backup before writes, merge-with-defaults on load
15. **RAG Pipeline**: Vector similarity search for contextual agent prompts
16. **Lifecycle Data**: Temporal attributes on all entity types.

## Data Flow

```plaintext
settings/ → Settings.load() → ServiceContainer
                                      ↓
         ┌────────────────────────────┼────────────────────────────┐
         ↓                            ↓                            ↓
  ProjectService              StoryService                 WorldService
  (stories/*.json)           (orchestrator/)              (worlds/*.sqlite)
         ↓                            ↓                            ↓
   StoryState                    Agents[]                   WorldDatabase
         ↓                     ↓          ↑                        ↓
     AppState ←─── UI Pages ←─ Events  RAG context ←── ContextRetrieval ←── EmbeddingService
```

## File Organization

```plaintext
story-factory/
├── main.py                  # Entry point with startup timing
├── settings/                # Configuration package (8 modules, ~2,600 lines)
│   ├── _settings.py         # Core Settings dataclass
│   ├── _validation.py       # Validation logic
│   ├── _model_registry.py   # RECOMMENDED_MODELS catalog
│   ├── _backup.py           # Settings backup/restore
│   ├── _paths.py            # Path resolution
│   ├── _types.py            # TypedDicts (ModelInfo, etc.)
│   ├── __init__.py          # Package exports
│   └── _utils.py            # Helper functions
├── agents/                  # AI agents (10 files, 1 package)
│   └── architect/           # Modularized architect agent
├── memory/                  # Data models (42 files, 4 packages)
│   ├── mode_database/       # Model performance tracking (10 modules)
│   ├── world_database/      # Entity/relationship storage (11 modules)
│   ├── world_quality/       # Quality models (5 modules)
│   └── builtin_templates/   # Template registry (2 modules)
├── services/                # Business logic (18 files, 6 packages)
│   ├── export_service/      # Export formats (6 modules)
│   ├── model_mode_service/  # Mode management (6 modules)
│   ├── orchestrator/        # Story orchestration (6 modules)
│   ├── story_service/       # Story workflow (5 modules)
│   ├── world_quality_service/ # Entity quality (20 modules)
│   └── world_service/       # World management (10 modules)
├── ui/                      # NiceGUI interface
│   ├── app.py               # App routing with timing
│   ├── state.py             # Centralized state with undo/redo
│   ├── components/          # Reusable UI components (14 files)
│   ├── graph_renderer/      # Graph visualization (4 modules)
│   └── pages/               # Route handlers (5 packages)
│       ├── analytics/       # Analytics dashboard (7 modules)
│       ├── models/          # Model management (4 modules)
│       ├── settings/        # Configuration UI (8 modules)
│       ├── world/           # World editor (13 modules)
│       └── write/           # Writing interface (6 modules)
├── utils/                   # Shared utilities (19 files)
├── prompts/                 # 42 YAML prompt templates
├── tests/                   # Test suites
│   ├── unit/                # Unit tests
│   ├── component/           # UI component tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
└── output/                  # Generated content
    ├── stories/             # JSON story files
    ├── worlds/              # SQLite world DBs
    └── backups/             # Project backups
```

## Code Statistics

| Layer | Files | Packages | Lines |
|-------|-------|----------|-------|
| Services | 71 | 6 | ~25,415 |
| UI Pages | 48 | 5 | ~13,373 |
| UI Components | 14 | 1 | ~5,241 |
| UI Core + Graph | 10 | 1 | ~2,622 |
| Memory Models | 42 | 4 | ~14,269 |
| Agents | 10 | 1 | ~3,253 |
| Settings | 8 | 1 | ~2,600 |
| Utils | 19 | 0 | ~4,463 |
| Prompts | 2 | 0 | ~232 |
| **Total** | **222** | **19** | **~73,487** |
