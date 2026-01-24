# Story Factory Architecture

> Generated: 2026-01-24 | Updated: 2026-01-24 | Freshness: Current

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
| `main.py:1-251` | CLI/Web entrypoint, argument parsing |
| `settings.py:1-1282` | Configuration, model registry, validation |

### Services Layer (`services/`)
Business logic, no UI imports. Receives settings via DI.

| Service | Responsibility |
|---------|---------------|
| `ProjectService` | Project CRUD, persistence |
| `StoryService` | Story generation workflow |
| `WorldService` | Entity/relationship management |
| `ModelService` | Ollama model operations |
| `ExportService` | Format exports (MD, PDF, EPUB) |
| `ModelModeService` | Model performance tracking |
| `ScoringService` | Quality scoring |
| `WorldQualityService` | World quality enhancement |
| `SuggestionService` | AI-powered suggestions |
| `TemplateService` | Story template management |
| `BackupService` | Project backup/restore |
| `ImportService` | Import entities from text |
| `ComparisonService` | Model comparison testing |

### UI Layer (`ui/`)
NiceGUI components, calls services only, manages state via AppState.

| Component | Route | Purpose |
|-----------|-------|---------|
| `WritePage` | `/` | Main writing interface |
| `WorldPage` | `/world` | Entity/relationship editor |
| `TimelinePage` | `/timeline` | Story timeline visualization |
| `ProjectsPage` | `/projects` | Project management |
| `SettingsPage` | `/settings` | Configuration UI |
| `ModelsPage` | `/models` | Ollama model management |
| `AnalyticsPage` | `/analytics` | Performance metrics |
| `TemplatesPage` | `/templates` | Template browser |
| `ComparisonPage` | `/compare` | Model comparison |

### Agents Layer (`agents/`)
AI agent implementations extending BaseAgent with retry logic.

| Agent | Role | Temperature |
|-------|------|-------------|
| `InterviewerAgent` | Gathers story requirements | 0.7 |
| `ArchitectAgent` | Designs structure/characters | 0.85 |
| `WriterAgent` | Writes prose | 0.9 |
| `EditorAgent` | Polishes and refines | 0.6 |
| `ContinuityAgent` | Detects plot holes | 0.3 |
| `ValidatorAgent` | Validates AI responses | 0.1 |

### Workflows Layer (`workflows/`)
Orchestration of agent pipelines.

| Component | Purpose |
|-----------|---------|
| `StoryOrchestrator` | Coordinates agents through story creation |
| `WorkflowEvent` | Event model for UI updates |

### Memory Layer (`memory/`)
Pydantic models and persistent storage.

| Component | Purpose |
|-----------|---------|
| `StoryState` | Complete story context model |
| `StoryBrief` | Initial story configuration |
| `Chapter`, `Scene` | Content structure models |
| `Character`, `Faction` | Entity models |
| `WorldDatabase` | SQLite + NetworkX graph DB |
| `Entity`, `Relationship` | Core entity models |
| `ModeDatabase` | Model performance tracking |

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

1. **Service Container DI**: All services instantiated once, injected into UI
2. **AppState**: Centralized UI state with callbacks for reactivity
3. **BaseAgent**: Retry logic, rate limiting (max 2 concurrent), configurable timeout
4. **WorldDatabase**: SQLite + NetworkX, thread-safe with RLock
5. **LRU Cache**: Orchestrator caching to prevent memory leaks
6. **Prompt Registry**: Centralized YAML template management with Jinja2
7. **Exception Hierarchy**: `StoryFactoryError` → `LLMError`, `ExportError`, etc.
8. **Instructor Integration**: Pydantic-validated structured LLM outputs
9. **Quality Refinement Loop**: Iterative entity improvement with scoring
10. **Auto Model Selection**: Tag-based model selection per agent role

## Data Flow

```
settings.json → Settings.load() → ServiceContainer
                                        ↓
           ┌────────────────────────────┼────────────────────────────┐
           ↓                            ↓                            ↓
    ProjectService              StoryService                 WorldService
    (stories/*.json)           (StoryOrchestrator)         (worlds/*.sqlite)
           ↓                            ↓                            ↓
     StoryState                    Agents[]                   WorldDatabase
           ↓                            ↓                            ↓
       AppState ←───── UI Pages ←────── WorkflowEvents
```

## File Organization

```
story-factory/
├── main.py              # Entry point
├── settings.py          # Configuration (53KB)
├── agents/              # AI agents (7 files)
├── memory/              # Data models (9 files)
├── services/            # Business logic (14 files)
├── ui/                  # NiceGUI interface
│   ├── app.py          # App routing
│   ├── state.py        # Centralized state
│   ├── components/     # Reusable UI components
│   └── pages/          # Route handlers
├── workflows/           # Orchestration (2 files)
├── utils/               # Shared utilities (16 files)
├── prompts/             # Prompt templates
├── tests/               # Test suites
│   ├── unit/           # Unit tests
│   ├── component/      # UI component tests
│   ├── integration/    # Integration tests
│   └── e2e/            # End-to-end tests
└── output/              # Generated content
    ├── stories/        # JSON story files
    ├── worlds/         # SQLite world DBs
    └── backups/        # Project backups
```
