# Backend Structure

> Generated: 2026-01-24 | Freshness: Current

## Services (`services/`)

### ServiceContainer (`services/__init__.py:27-74`)

DI container for all services. Initialized with Settings, creates all service instances.

```python
services = ServiceContainer(settings)
services.project.list_projects()
services.story.start_interview(state)
```

### Core Services

| Service | File | Key Methods |
|---------|------|-------------|
| `ProjectService` | `project_service.py` | `create_project()`, `load_project()`, `save_project()`, `delete_project()` |
| `StoryService` | `story_service.py` | `start_interview()`, `process_response()`, `build_structure()`, `write_chapter()` |
| `WorldService` | `world_service.py` | `add_entity()`, `update_entity()`, `generate_world()`, `WorldBuildOptions` |
| `ModelService` | `model_service.py` | `list_models()`, `pull_model()`, `delete_model()`, `check_health()` |
| `ExportService` | `export_service.py` | `export_markdown()`, `export_pdf()`, `export_epub()` |

### Analysis Services

| Service | File | Purpose |
|---------|------|---------|
| `ModelModeService` | `model_mode_service.py` | Model performance tracking, mode-based selection |
| `ScoringService` | `scoring_service.py` | Quality scoring for generated content |
| `WorldQualityService` | `world_quality_service.py` | Iterative quality refinement |
| `ComparisonService` | `comparison_service.py` | A/B model comparison testing |

### Utility Services

| Service | File | Purpose |
|---------|------|---------|
| `SuggestionService` | `suggestion_service.py` | AI-powered writing suggestions |
| `TemplateService` | `template_service.py` | Story template management |
| `BackupService` | `backup_service.py` | Project backup/restore |
| `ImportService` | `import_service.py` | Entity extraction from text |

### LLM Client (`services/llm_client.py`)

Direct Ollama integration for services that need raw LLM access.

## Agents (`agents/`)

### BaseAgent (`agents/base.py:81-485`)

Abstract base for all agents with:
- Rate limiting via semaphore (max 2 concurrent)
- Retry logic with exponential backoff
- Prompt template integration
- Instructor-based structured output
- Performance logging

Key methods:
- `generate(prompt, context, temperature)` → str
- `generate_structured(prompt, response_model)` → Pydantic model
- `render_prompt(task, **kwargs)` → str

### Agent Implementations

| Agent | File | Role | Default Temp |
|-------|------|------|--------------|
| `InterviewerAgent` | `interviewer.py` | Story requirements gathering | 0.7 |
| `ArchitectAgent` | `architect.py` | World/plot/character design | 0.85 |
| `WriterAgent` | `writer.py` | Prose generation | 0.9 |
| `EditorAgent` | `editor.py` | Content refinement | 0.6 |
| `ContinuityAgent` | `continuity.py` | Plot hole detection | 0.3 |
| `ValidatorAgent` | `validator.py` | Response validation | 0.1 |

## Workflows (`workflows/`)

### StoryOrchestrator (`workflows/orchestrator.py:51-1567`)

Main coordinator for the story generation pipeline.

**Phases:**
1. Interview → `start_interview()`, `process_interview_response()`, `finalize_interview()`
2. Architecture → `build_story_structure()`, `generate_more_characters()`, `generate_locations()`
3. Writing → `write_chapter()`, `write_all_chapters()`, `write_short_story()`
4. Editing → `edit_passage()`, `get_edit_suggestions()`, `continue_chapter()`
5. Review → `review_full_story()`

**Persistence:**
- `save_story()`, `load_story()`, `autosave()`
- `export_to_markdown()`, `export_to_epub()`, `export_to_pdf()`

**Events:**
- `WorkflowEvent` dataclass with progress tracking
- Deque-based event history (max 100 events)

## Memory Models (`memory/`)

### StoryState (`memory/story_state.py:404-622`)

Complete story context:
- `brief`: StoryBrief | None
- `characters`: list[Character]
- `chapters`: list[Chapter]
- `plot_points`: list[PlotPoint]
- `established_facts`: list[str]
- `outline_variations`: list[OutlineVariation]

### WorldDatabase (`memory/world_database.py:62-700+`)

SQLite + NetworkX hybrid storage:
- Thread-safe with RLock
- WAL mode for concurrency
- Schema versioning/migrations
- Graph queries via NetworkX

Tables: `entities`, `relationships`, `events`, `event_participants`, `schema_version`

### Entity Models (`memory/entities.py`)

- `Entity`: character, location, item, faction, concept
- `Relationship`: source → target with type, strength
- `WorldEvent`: Timeline events with participants

## Utilities (`utils/`)

| Utility | File | Purpose |
|---------|------|---------|
| Exceptions | `exceptions.py` | Centralized exception hierarchy |
| JSON Parser | `json_parser.py` | LLM response JSON extraction |
| Validation | `validation.py` | Input validation helpers |
| Error Handling | `error_handling.py` | Decorators: `@handle_ollama_errors`, `@retry_with_fallback` |
| Logging Config | `logging_config.py` | Logging setup, performance context manager |
| Prompt Registry | `prompt_registry.py` | Template loading and rendering |
| Prompt Builder | `prompt_builder.py` | Dynamic prompt construction |
| Constants | `constants.py` | Language codes, shared constants |
| Environment | `environment.py` | Environment checks (Python version, deps) |
| Text Analytics | `text_analytics.py` | Reading level, complexity metrics |
| Message Analyzer | `message_analyzer.py` | Language/content inference |

## Configuration (`settings.py`)

### Settings Class (`settings.py:231-1103`)

Dataclass with ~100+ configurable values:
- Ollama connection (URL, timeouts)
- Per-agent model selection (`agent_models` dict)
- Per-agent temperatures (`agent_temperatures` dict)
- World generation limits (min/max counts)
- LLM token limits per task
- Quality thresholds

Key methods:
- `load(use_cache=True)` → Settings
- `save()` → JSON file
- `validate()` → raises ValueError
- `get_model_for_agent(role)` → str
- `get_temperature_for_agent(role)` → float

### Model Registry (`settings.py:85-219`)

`RECOMMENDED_MODELS`: Curated model catalog with:
- Quality/speed ratings
- VRAM requirements
- Role-specific tags

### Auto-Selection (`settings.py:1005-1089`)

Tag-based model selection:
1. Check `use_per_agent_models` setting
2. Lookup `agent_models[role]`
3. If "auto", find installed models tagged for role
4. Select highest quality that fits VRAM
