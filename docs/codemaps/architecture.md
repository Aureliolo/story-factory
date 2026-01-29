# Story Factory Architecture

> Generated: 2026-01-24 | Updated: 2026-01-29 | Freshness: Current

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
│  settings.py → Settings.load() → ServiceContainer(settings) │
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
| `settings.py` | Configuration, model registry, validation (~1282 lines) |

### Services Layer (`services/`)
Business logic, no UI imports. Receives settings via DI. **19 services** in container.

| Service | Package/File | Responsibility |
|---------|--------------|---------------|
| `ProjectService` | `project_service.py` | Project CRUD, persistence |
| `StoryService` | `story_service/` | Story generation workflow (5 modules) |
| `WorldService` | `world_service/` | Entity/relationship management (6 modules) |
| `ModelService` | `model_service.py` | Ollama model operations |
| `ExportService` | `export_service/` | Format exports (5 modules) |
| `ModelModeService` | `model_mode_service/` | Model performance tracking (6 modules) |
| `ScoringService` | `scoring_service.py` | Quality scoring |
| `WorldQualityService` | `world_quality_service/` | Entity quality enhancement (9 modules) |
| `SuggestionService` | `suggestion_service.py` | AI-powered suggestions |
| `TemplateService` | `template_service.py` | Story template management |
| `BackupService` | `backup_service.py` | Project backup/restore |
| `ImportService` | `import_service.py` | Import entities from text |
| `ComparisonService` | `comparison_service.py` | Model comparison testing |
| `TimelineService` | `timeline_service.py` | Timeline event management |
| `ConflictAnalysisService` | `conflict_analysis_service.py` | Relationship classification |
| `WorldTemplateService` | `world_template_service.py` | World template management |
| `ContentGuidelinesService` | `content_guidelines_service.py` | Content standards enforcement |
| `CalendarService` | `calendar_service.py` | Fictional calendar generation |
| `TemporalValidationService` | `temporal_validation_service.py` | Temporal consistency validation |

### UI Layer (`ui/`)
NiceGUI components, calls services only, manages state via AppState.

| Page | Route | Package | Purpose |
|------|-------|---------|---------|
| `WritePage` | `/` | `write/` (6 modules) | Main writing interface |
| `WorldPage` | `/world` | `world/` (13 modules) | Entity/relationship editor |
| `TimelinePage` | `/timeline` | `timeline.py` | Story timeline visualization |
| `WorldTimelinePage` | `/world-timeline` | `world_timeline.py` | World event timeline |
| `ProjectsPage` | `/projects` | `projects.py` | Project management |
| `SettingsPage` | `/settings` | `settings/` (7 modules) | Configuration UI with undo/redo |
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
| `ContinuityAgent` | Detects plot holes | 0.3 |
| `ValidatorAgent` | Validates AI responses | 0.1 |

### Workflows Layer (`workflows/`)
Orchestration of agent pipelines.

| Component | Package | Purpose |
|-----------|---------|---------|
| `StoryOrchestrator` | `orchestrator/` (6 modules) | Coordinates agents through story creation |
| `WorkflowEvent` | `orchestrator/` | Event model for UI updates |

### Memory Layer (`memory/`)
Pydantic models and persistent storage.

| Component | Package/File | Purpose |
|-----------|--------------|---------|
| `StoryState` | `story_state.py` | Complete story context model |
| `StoryBrief` | `story_state.py` | Initial story configuration |
| `Chapter`, `Scene` | `story_state.py` | Content structure models |
| `Character`, `Faction` | `story_state.py` | Entity models |
| `WorldDatabase` | `world_database/` (7 modules) | SQLite + NetworkX graph DB |
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

**Phase Weights (Progress):**
- Interview: 10%
- Architect: 15%
- Writer: 50%
- Editor: 15%
- Continuity: 10%

## Key Patterns

1. **Service Container DI**: 19 services instantiated once, injected into UI
2. **AppState**: Centralized UI state with callbacks for reactivity
3. **BaseAgent**: Retry logic, rate limiting (max 2 concurrent), configurable timeout
4. **WorldDatabase**: SQLite + NetworkX, thread-safe with RLock, versioning support
5. **ModeDatabase**: Model performance tracking with cost analytics
6. **LRU Cache**: Orchestrator caching to prevent memory leaks
7. **Prompt Registry**: Centralized YAML template management with Jinja2
8. **Exception Hierarchy**: `StoryFactoryError` → `LLMError`, `ExportError`, etc.
9. **Instructor Integration**: Pydantic-validated structured LLM outputs
10. **Quality Refinement Loop**: Iterative entity improvement with scoring
11. **Auto Model Selection**: Tag-based model selection per agent role
12. **Temporal Validation**: Calendar-aware consistency checking
13. **Undo/Redo System**: Snapshot-based state management for settings and world entities

## Data Flow

```
settings.json → Settings.load() → ServiceContainer
                                        ↓
           ┌────────────────────────────┼────────────────────────────┐
           ↓                            ↓                            ↓
    ProjectService              StoryService                 WorldService
    (stories/*.json)           (orchestrator/)              (worlds/*.sqlite)
           ↓                            ↓                            ↓
     StoryState                    Agents[]                   WorldDatabase
           ↓                            ↓                            ↓
       AppState ←───── UI Pages ←────── WorkflowEvents
```

## File Organization

```
story-factory/
├── main.py                  # Entry point with startup timing
├── settings.py              # Configuration (~1282 lines)
├── agents/                  # AI agents (8 files, 1 package)
│   └── architect/          # Modularized architect agent
├── memory/                  # Data models (14 files, 3 packages)
│   ├── mode_database/      # Model performance tracking (10 modules)
│   ├── world_database/     # Entity/relationship storage (7 modules)
│   └── world_quality/      # Quality models (3 modules)
├── services/                # Business logic (15 files, 6 packages)
│   ├── export_service/     # Export formats (5 modules)
│   ├── model_mode_service/ # Mode management (6 modules)
│   ├── orchestrator/       # Story orchestration (6 modules)
│   ├── story_service/      # Story workflow (5 modules)
│   ├── world_quality_service/ # Entity quality (9 modules)
│   └── world_service/      # World management (6 modules)
├── ui/                      # NiceGUI interface
│   ├── app.py              # App routing with timing
│   ├── state.py            # Centralized state with undo/redo
│   ├── components/         # Reusable UI components (13 files)
│   └── pages/              # Route handlers (6 packages)
│       ├── analytics/      # Analytics dashboard (7 modules)
│       ├── models/         # Model management (4 modules)
│       ├── settings/       # Configuration UI (7 modules)
│       ├── world/          # World editor (13 modules)
│       └── write/          # Writing interface (6 modules)
├── workflows/               # Orchestration (1 package)
├── utils/                   # Shared utilities (16 files)
├── prompts/                 # Prompt templates
├── tests/                   # Test suites
│   ├── unit/               # Unit tests
│   ├── component/          # UI component tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
└── output/                  # Generated content
    ├── stories/            # JSON story files
    ├── worlds/             # SQLite world DBs
    └── backups/            # Project backups
```

## Code Statistics

| Layer | Files | Packages | Lines |
|-------|-------|----------|-------|
| Services | 31 | 6 | ~15,270 |
| UI Pages | 47 | 6 | ~10,671 |
| UI Components | 13 | 1 | ~4,896 |
| Memory Models | 28 | 3 | ~11,932 |
| **Total** | **119** | **16** | **~42,769** |
