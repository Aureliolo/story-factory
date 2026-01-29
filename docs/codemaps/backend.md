# Backend Structure

> Generated: 2026-01-24 | Updated: 2026-01-29 | Freshness: Current

## Services (`services/`)

### ServiceContainer (`services/__init__.py:32-96`)

DI container for all services. Initialized with Settings, creates 19 service instances.

```python
services = ServiceContainer(settings)
services.project.list_projects()
services.story.start_interview(state)
services.calendar.generate_calendar(brief)
services.temporal_validation.validate_entity(entity, calendar)
```

### Core Services

| Service | Location | Key Methods |
|---------|----------|-------------|
| `ProjectService` | `project_service.py` | `create_project()`, `load_project()`, `save_project()`, `delete_project()` |
| `StoryService` | `story_service/` | `start_interview()`, `process_response()`, `build_structure()`, `write_chapter()` |
| `WorldService` | `world_service/` | `add_entity()`, `update_entity()`, `generate_world()`, `calculate_health()` |
| `ModelService` | `model_service.py` | `list_models()`, `pull_model()`, `delete_model()`, `check_health()` |
| `ExportService` | `export_service/` | `export_markdown()`, `export_pdf()`, `export_epub()`, `export_docx()` |

### Analysis Services

| Service | Location | Purpose |
|---------|----------|---------|
| `ModelModeService` | `model_mode_service/` | Model performance tracking, mode-based selection |
| `ScoringService` | `scoring_service.py` | Quality scoring for generated content |
| `WorldQualityService` | `world_quality_service/` | Iterative quality refinement |
| `ComparisonService` | `comparison_service.py` | A/B model comparison testing |
| `ConflictAnalysisService` | `conflict_analysis_service.py` | Relationship classification and conflict visualization |

### Utility Services

| Service | Location | Purpose |
|---------|----------|---------|
| `SuggestionService` | `suggestion_service.py` | AI-powered writing suggestions |
| `TemplateService` | `template_service.py` | Story template management |
| `BackupService` | `backup_service.py` | Project backup/restore |
| `ImportService` | `import_service.py` | Entity extraction from text |
| `TimelineService` | `timeline_service.py` | Timeline event management |
| `WorldTemplateService` | `world_template_service.py` | World template management |
| `ContentGuidelinesService` | `content_guidelines_service.py` | Content standards enforcement |
| `CalendarService` | `calendar_service.py` | Fictional calendar generation |
| `TemporalValidationService` | `temporal_validation_service.py` | Temporal consistency validation |

### LLM Client (`services/llm_client.py`)

Shared Instructor client for structured LLM outputs in services.

```python
# Get cached Instructor client
client = get_instructor_client(settings)

# Generate structured output with Pydantic validation
result = generate_structured(
    settings, model, prompt, ResponseModel,
    system_prompt="...", temperature=0.1
)
```

## Service Packages

### story_service/ (~1,261 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 429 | Main StoryService with workflow orchestration |
| `_interview.py` | 168 | Interview phase logic |
| `_structure.py` | 222 | Structure phase logic |
| `_writing.py` | 219 | Writing phase logic |
| `_editing.py` | 223 | Editing phase logic |

### world_service/ (~2,113 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 624 | Main WorldService with entity/relationship CRUD |
| `_build.py` | 447 | World building orchestration |
| `_entities.py` | 316 | Entity management |
| `_extraction.py` | 311 | Entity extraction from LLM output |
| `_graph.py` | 200 | Graph operations |
| `_health.py` | 215 | World health metrics calculation |

### export_service/ (~1,063 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 296 | Main ExportService class |
| `_docx.py` | 121 | DOCX export format |
| `_epub.py` | 146 | EPUB export format |
| `_pdf.py` | 131 | PDF export format |
| `_text.py` | 184 | Text export format |
| `_types.py` | 185 | Type definitions and shared utilities |

### model_mode_service/ (~1,726 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 418 | Main ModelModeService |
| `_modes.py` | 421 | Generation mode management |
| `_scoring.py` | 310 | Mode scoring and evaluation |
| `_learning.py` | 284 | Adaptive learning logic |
| `_analytics.py` | 196 | Mode analytics |
| `_vram.py` | 97 | VRAM tracking |

### world_quality_service/ (~4,732 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 763 | Main orchestrator |
| `_validation.py` | 752 | Quality validation rules |
| `_batch.py` | 613 | Batch processing |
| `_character.py` | 398 | Character-specific quality |
| `_faction.py` | 537 | Faction-specific quality |
| `_concept.py` | 405 | Concept-specific quality |
| `_location.py` | 380 | Location-specific quality |
| `_item.py` | 406 | Item-specific quality |
| `_relationship.py` | 478 | Relationship-specific quality |

### orchestrator/ (~2,086 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 716 | Main StoryOrchestrator, coordinates agents |
| `_persistence.py` | 433 | Story state persistence |
| `_writing.py` | 452 | Writing loop coordination |
| `_editing.py` | 219 | Editing loop coordination |
| `_structure.py` | 182 | Structure determination |
| `_interview.py` | 84 | Interview phase |

## Agents (`agents/`)

### BaseAgent (`agents/base.py`)

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
| `ArchitectAgent` | `architect/` | World/plot/character design | 0.85 |
| `WriterAgent` | `writer.py` | Prose generation | 0.9 |
| `EditorAgent` | `editor.py` | Content refinement | 0.6 |
| `ContinuityAgent` | `continuity.py` | Plot hole detection | 0.3 |
| `ValidatorAgent` | `validator.py` | Response validation | 0.1 |

### ArchitectAgent Package (`agents/architect/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Main ArchitectAgent class |
| `_structure.py` | Story structure generation |
| `_world.py` | World building methods |

## Memory Models (`memory/`)

### StoryState (`memory/story_state.py`)

Complete story context:
- `brief`: StoryBrief | None
- `characters`: list[Character]
- `chapters`: list[Chapter]
- `plot_points`: list[PlotPoint]
- `established_facts`: list[str]
- `outline_variations`: list[OutlineVariation]

### Database Packages

#### WorldDatabase (`memory/world_database/`)

SQLite + NetworkX hybrid storage (~2,600 lines):

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 688 | Main WorldDatabase class, graph operations |
| `_entities.py` | 366 | Entity CRUD operations |
| `_relationships.py` | 405 | Relationship management |
| `_versions.py` | 318 | Entity versioning for backups |
| `_graph.py` | 534 | NetworkX graph operations |
| `_events.py` | 164 | Timeline event management |
| `_io.py` | 125 | I/O utilities |

Features:
- Thread-safe with RLock
- WAL mode for concurrency
- Schema versioning/migrations
- Graph queries via NetworkX
- Entity versioning for backups

#### ModeDatabase (`memory/mode_database/`)

Model performance tracking (~3,396 lines):

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 590 | Main ModeDatabase class |
| `_schema.py` | 243 | SQLite schema and migrations |
| `_cost_tracking.py` | 502 | Token and cost tracking |
| `_performance.py` | 250 | Performance metrics |
| `_scoring.py` | 641 | Mode scoring |
| `_prompt_metrics.py` | 308 | Per-prompt metrics |
| `_recommendations.py` | 188 | Mode recommendations |
| `_refinement.py` | 213 | Refinement effectiveness tracking |
| `_custom_modes.py` | 142 | Custom mode management |
| `_world_entity.py` | 319 | World entity quality tracking with temporal fields |

### Entity Models (`memory/entities.py`)

- `Entity`: character, location, item, faction, concept
- `Relationship`: source → target with type, strength
- `WorldEvent`: Timeline events with participants

### New Memory Models

| Model | File | Purpose |
|-------|------|---------|
| `WorldCalendar` | `world_calendar.py` | Fictional calendar with months, eras |
| `WorldSettings` | `world_settings.py` | World configuration settings |
| `WorldHealth` | `world_health.py` | World health metrics and scoring |

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
| Prompt Template | `prompt_template.py` | YAML-based Jinja2 templates |

## Configuration (`settings.py`)

### Settings Class

Dataclass with ~100+ configurable values:
- Ollama connection (URL, timeouts)
- Per-agent model selection (`agent_models` dict)
- Per-agent temperatures (`agent_temperatures` dict)
- World generation limits (min/max counts)
- LLM token limits per task
- Quality thresholds
- Temporal validation settings

Key methods:
- `load(use_cache=True)` → Settings
- `save()` → JSON file
- `validate()` → raises ValueError
- `get_model_for_agent(role)` → str
- `get_temperature_for_agent(role)` → float

### Model Registry

`RECOMMENDED_MODELS`: Curated model catalog with:
- Quality/speed ratings
- VRAM requirements
- Role-specific tags

### Auto-Selection

Tag-based model selection:
1. Check `use_per_agent_models` setting
2. Lookup `agent_models[role]`
3. If "auto", find installed models tagged for role
4. Select highest quality that fits VRAM

## Prompt Template System (`prompts/templates/`)

### PromptTemplate (`utils/prompt_template.py`)

YAML-based prompt templates with Jinja2 rendering.

```python
@dataclass
class PromptTemplate:
    name: str           # Template name (e.g., "write_chapter")
    version: str        # For metrics tracking
    agent: str          # Agent role (e.g., "writer")
    task: str           # Task identifier
    template: str       # Jinja2 template string
    required_variables: list[str]
    optional_variables: list[str]
    is_system_prompt: bool
```

### PromptRegistry (`utils/prompt_registry.py`)

Central registry for template loading and lookup.

```python
registry = PromptRegistry()  # Auto-loads from prompts/templates/
prompt = registry.render("writer", "write_chapter", chapter_number=1, ...)
system = registry.render_system("editor")
```

### Template Organization (~60+ templates)

```
prompts/templates/
├── architect/          # 10 templates
│   ├── system.yaml
│   ├── create_characters.yaml
│   ├── create_plot_outline.yaml
│   └── ...
├── continuity/         # 9 templates
│   ├── system.yaml
│   ├── check_chapter.yaml
│   └── ...
├── editor/             # 5 templates
├── interviewer/        # 4 templates
├── suggestion/         # 4 templates
├── validator/          # 2 templates
├── writer/             # 5 templates
└── world_quality/      # 21+ templates
    ├── character/
    ├── location/
    ├── faction/
    ├── item/
    ├── concept/
    ├── calendar/       # NEW: Calendar generation
    └── shared/
```
