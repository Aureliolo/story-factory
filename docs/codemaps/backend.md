# Backend Structure

<!-- Generated: 2026-01-24 | Updated: 2026-02-20 | Files scanned: 222 | Token estimate: ~950 -->

## Services (`services/`)

### ServiceContainer (`services/__init__.py`)

DI container for all 21 services. Initialized with Settings, wires dependencies.

```python
services = ServiceContainer(settings)
services.project.list_projects()
services.story.start_interview(state)
services.calendar.generate_calendar(brief)
services.temporal_validation.validate_entity(entity, calendar)
services.context_retrieval.retrieve_context(query, world_db, state)
```

### Core Services

| Service | Location | Key Methods |
|---------|----------|-------------|
| `ProjectService` | `project_service.py` (468) | `create_project()`, `load_project()`, `save_project()`, `delete_project()` |
| `StoryService` | `story_service/` (1,202) | `start_interview()`, `process_response()`, `build_structure()`, `write_chapter()` |
| `WorldService` | `world_service/` (3,420) | `add_entity()`, `update_entity()`, `generate_world()`, `calculate_health()` |
| `ModelService` | `model_service.py` (710) | `list_models()`, `pull_model()`, `delete_model()`, `check_health()` |
| `ExportService` | `export_service/` (1,063) | `export_markdown()`, `export_pdf()`, `export_epub()`, `export_docx()` |

### Analysis Services

| Service | Location | Purpose |
|---------|----------|---------|
| `ModelModeService` | `model_mode_service/` (1,800) | Model performance tracking, mode-based selection |
| `ScoringService` | `scoring_service.py` (409) | Quality scoring for generated content |
| `WorldQualityService` | `world_quality_service/` (8,061) | Iterative quality refinement |
| `ComparisonService` | `comparison_service.py` (348) | A/B model comparison testing |
| `ConflictAnalysisService` | `conflict_analysis_service.py` (654) | Relationship classification and conflict visualization |

### RAG Services

| Service | Location | Purpose |
|---------|----------|---------|
| `EmbeddingService` | `embedding_service.py` (447) | Vector embeddings via Ollama, auto-embedding callbacks |
| `ContextRetrievalService` | `context_retrieval_service.py` (395) | KNN vector search for relevant entities/relationships |

### Utility Services

| Service | Location | Purpose |
|---------|----------|---------|
| `SuggestionService` | `suggestion_service.py` (283) | AI-powered writing suggestions |
| `TemplateService` | `template_service.py` (402) | Story template management |
| `BackupService` | `backup_service.py` (723) | Project backup/restore |
| `ImportService` | `import_service.py` (565) | Entity extraction from text |
| `TimelineService` | `timeline_service.py` (445) | Timeline aggregation, temporal context for agents |
| `WorldTemplateService` | `world_template_service.py` (250) | World template management |
| `ContentGuidelinesService` | `content_guidelines_service.py` (358) | Content standards enforcement |
| `CalendarService` | `calendar_service.py` (252) | Fictional calendar generation |
| `TemporalValidationService` | `temporal_validation_service.py` (472) | Temporal consistency validation |

### LLM Client (`services/llm_client.py`, 254 lines)

Shared Instructor client for structured LLM outputs in services.

```python
client = get_instructor_client(settings)
result = generate_structured(settings, model, prompt, ResponseModel, system_prompt="...", temperature=0.1)
```

## Service Packages

### story_service/ (~1,202 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 444 | Main StoryService with workflow orchestration |
| `_interview.py` | 168 | Interview phase logic |
| `_structure.py` | 136 | Structure phase logic |
| `_writing.py` | 231 | Writing phase logic |
| `_editing.py` | 223 | Editing phase logic |

### world_service/ (~3,420 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `_build.py` | 905 | World building orchestration |
| `__init__.py` | 744 | Main WorldService with entity/relationship CRUD |
| `_health.py` | 331 | World health metrics calculation |
| `_entities.py` | 316 | Entity management |
| `_event_helpers.py` | 305 | Event generation helpers |
| `_extraction.py` | 233 | Entity extraction from LLM output |
| `_graph.py` | 182 | Graph operations |
| `_orphan_recovery.py` | 181 | Orphan entity recovery |
| `_lifecycle_helpers.py` | 154 | Lifecycle/temporal data helpers |
| `_name_matching.py` | 69 | Fuzzy entity name matching |

### world_quality_service/ (~8,061 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `_validation.py` | 864 | Quality validation rules |
| `__init__.py` | 844 | Main orchestrator |
| `_relationship.py` | 804 | Relationship-specific quality |
| `_batch.py` | 767 | Batch processing |
| `_quality_loop.py` | 514 | Generic quality refinement loop |
| `_calendar.py` | 462 | Calendar quality scoring |
| `_faction.py` | 423 | Faction-specific quality |
| `_character.py` | 404 | Character-specific quality |
| `_event.py` | 360 | Event-specific quality |
| `_chapter_quality.py` | 326 | Chapter quality assessment |
| `_location.py` | 318 | Location-specific quality |
| `_concept.py` | 317 | Concept-specific quality |
| `_item.py` | 316 | Item-specific quality |
| `_plot.py` | 246 | Plot quality assessment |
| `_model_cache.py` | 233 | Model caching for quality ops |
| `_common.py` | 209 | Shared quality utilities |
| `_model_resolver.py` | 181 | Model resolution for quality agents |
| `_analytics.py` | 171 | Quality analytics |
| `_entity_delegates.py` | 170 | Entity type delegation |
| `_formatting.py` | 132 | Output formatting helpers |

### orchestrator/ (~2,285 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 750 | Main StoryOrchestrator, coordinates agents |
| `_writing.py` | 615 | Writing loop with timeline context |
| `_persistence.py` | 431 | Story state persistence |
| `_editing.py` | 227 | Editing loop with temporal validation |
| `_structure.py` | 180 | Structure determination |
| `_interview.py` | 82 | Interview phase |

### export_service/ (~1,063 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 296 | Main ExportService class |
| `_types.py` | 185 | Type definitions and shared utilities |
| `_text.py` | 184 | Text export format |
| `_epub.py` | 146 | EPUB export format |
| `_pdf.py` | 131 | PDF export format |
| `_docx.py` | 121 | DOCX export format |

### model_mode_service/ (~1,800 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `_modes.py` | 466 | Generation mode management |
| `__init__.py` | 424 | Main ModelModeService |
| `_scoring.py` | 310 | Mode scoring and evaluation |
| `_learning.py` | 285 | Adaptive learning logic |
| `_analytics.py` | 196 | Mode analytics |
| `_vram.py` | 119 | VRAM tracking |

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
| `ContinuityAgent` | `continuity.py` | Plot hole + temporal validation | 0.3 |
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

#### WorldDatabase (`memory/world_database/`) ~3,787 lines

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | 919 | Main WorldDatabase class, graph operations |
| `_graph.py` | 535 | NetworkX graph operations |
| `_embeddings.py` | 425 | sqlite-vec vector embeddings |
| `_entities.py` | 400 | Entity CRUD operations |
| `_relationships.py` | 354 | Relationship management |
| `_versions.py` | 321 | Entity versioning for backups |
| `_events.py` | 235 | Timeline event management |
| `_schema.py` | 231 | Schema definitions and migrations |
| `_io.py` | 145 | I/O utilities |
| `_cycles.py` | 116 | Circular relationship detection |
| `_attributes.py` | 106 | Entity attribute helpers |

#### ModeDatabase (`memory/mode_database/`) ~3,381 lines

| Module | Lines | Purpose |
|--------|-------|---------|
| `_scoring.py` | 641 | Mode scoring |
| `__init__.py` | 590 | Main ModeDatabase class |
| `_cost_tracking.py` | 502 | Token and cost tracking |
| `_world_entity.py` | 319 | World entity quality tracking with temporal fields |
| `_prompt_metrics.py` | 308 | Per-prompt metrics |
| `_performance.py` | 250 | Performance metrics |
| `_schema.py` | 228 | SQLite schema and migrations |
| `_refinement.py` | 213 | Refinement effectiveness tracking |
| `_recommendations.py` | 188 | Mode recommendations |
| `_custom_modes.py` | 142 | Custom mode management |

### Entity Models (`memory/entities.py`)

- `Entity`: character, location, item, faction, concept (with lifecycle/temporal attributes)
- `Relationship`: source → target with type, strength
- `WorldEvent`: Timeline events with participants

### Other Memory Models

| Model | File | Purpose |
|-------|------|---------|
| `WorldCalendar` | `world_calendar.py` | Fictional calendar with months, eras |
| `WorldSettings` | `world_settings.py` | World configuration settings |
| `WorldHealth` | `world_health.py` | World health metrics and scoring |
| `TimelineTypes` | `timeline_types.py` | Timeline event types |
| `ContentGuidelines` | `content_guidelines.py` | Content rules models |
| `ConflictTypes` | `conflict_types.py` | Conflict classification models |
| `CostModels` | `cost_models.py` | Token cost tracking models |
| `ArcTemplates` | `arc_templates.py` | Character arc templates |

## Settings Package (`settings/`) ~2,600 lines

| Module | Lines | Purpose |
|--------|-------|---------|
| `_validation.py` | 897 | Load-time validation, merge-with-defaults |
| `_settings.py` | 865 | Core Settings dataclass (~100+ fields) |
| `_model_registry.py` | 275 | RECOMMENDED_MODELS catalog |
| `_utils.py` | 204 | Helper functions |
| `_backup.py` | 172 | Auto-backup before writes, restore on corruption |
| `_types.py` | 124 | TypedDicts (ModelInfo, etc.) |
| `__init__.py` | 45 | Package exports |
| `_paths.py` | 18 | Path resolution |

Key methods:
- `load(use_cache=True)` → Settings (merges user JSON with defaults)
- `save()` → JSON file (auto-backup before write)
- `validate()` → raises ValueError
- `get_model_for_agent(role)` → str
- `get_temperature_for_agent(role)` → float

## Utilities (`utils/`) ~4,463 lines

| Utility | File | Purpose |
|---------|------|---------|
| Exceptions | `exceptions.py` | Centralized exception hierarchy |
| JSON Parser | `json_parser.py` | LLM response JSON extraction |
| Validation | `validation.py` | Input validation helpers |
| Error Handling | `error_handling.py` | `@handle_ollama_errors`, `@retry_with_fallback` |
| Logging Config | `logging_config.py` | Logging setup, performance context manager |
| Prompt Registry | `prompt_registry.py` | Template loading and rendering |
| Prompt Builder | `prompt_builder.py` | Dynamic prompt construction |
| Prompt Template | `prompt_template.py` | YAML-based Jinja2 templates |
| Constants | `constants.py` | Language codes, shared constants |
| Environment | `environment.py` | Environment checks (Python version, deps) |
| Text Analytics | `text_analytics.py` | Reading level, complexity metrics |
| Message Analyzer | `message_analyzer.py` | Language/content inference |
| Model Utils | `model_utils.py` | Model selection helpers |
| Retry Strategies | `retry_strategies.py` | Configurable retry logic |
| Similarity | `similarity.py` | Text similarity functions |
| Streaming | `streaming.py` | LLM streaming utilities |
| Circuit Breaker | `circuit_breaker.py` | Fault tolerance pattern |
| SQLite Vec Loader | `sqlite_vec_loader.py` | sqlite-vec extension loading |

## Prompt Template System (`prompts/templates/`)

42 YAML templates organized by agent role:

```
prompts/templates/
├── architect/      # 11 templates (world, characters, plot, relationships, locations)
├── continuity/     # 9 templates (chapter check, full story, facts, arcs, dialogue)
├── editor/         # 5 templates (edit, suggestions, consistency)
├── interviewer/    # 4 templates (questions, response, brief)
├── suggestion/     # 4 templates (suggestions, categories, project names)
├── validator/      # 4 templates (validation, guidelines, contradictions)
└── writer/         # 5 templates (chapter, scene, short story, continue)
```
