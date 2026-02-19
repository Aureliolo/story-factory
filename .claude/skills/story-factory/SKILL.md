# Story Factory Development Workflows

Common multi-step workflows derived from commit history patterns. All file paths verified against the actual codebase.

## Add New Entity Type

When adding a new entity type that participates in the quality loop (like locations, factions, items, concepts, events):

1. **Create/update Pydantic models** in `src/memory/story_state.py` or `src/memory/entities.py`
2. **Add quality scores model** in `src/memory/world_quality/_entity_scores.py`
3. **Create quality service module** in `src/services/world_quality_service/_entityname.py` implementing the create/judge/refine pattern (see existing `_location.py`, `_faction.py`, `_item.py`, `_concept.py`, `_event.py` for reference)
4. **Add entity to build pipeline** in `src/services/world_service/_build.py` — follows the same `_create_fn` / `_judge_fn` / `_refine_fn` pattern via `quality_refinement_loop()`
5. **Add UI page** in `src/ui/pages/world/` for browsing/editing the new entity type
6. **Add comprehensive tests** in `tests/unit/` mirroring the source structure

## Database Schema Migration

When modifying the WorldDatabase schema:

1. **Update schema version and add migration** in `src/memory/world_database/_schema.py`
2. **Add database operations** in `src/memory/world_database/` — use a dedicated submodule if the operations are substantial (e.g., `_cycles.py`, `_events.py`, `_embeddings.py`)
3. **Update `__init__.py`** in `src/memory/world_database/` to expose new operations
4. **Update related services** that consume the new data
5. **Add migration tests and integration tests**

## Quality Loop Improvements

When modifying quality scoring or the refinement loop:

1. **Update loop logic** in `src/services/world_quality_service/_quality_loop.py`
2. **Modify scoring models** in `src/memory/world_quality/_models.py` or `src/memory/world_quality/_entity_scores.py`
3. **Update analytics** in `src/services/world_quality_service/_analytics.py`
4. **Update settings** in `src/settings.py` and expose in `src/ui/pages/settings/`
5. **Add tests** for new scoring/loop behaviors

## World Building Pipeline Improvements

When improving entity generation, relationships, or pipeline reliability:

1. **Update build pipeline** in `src/services/world_service/_build.py`
2. **Modify entity generation** in the relevant `src/services/world_quality_service/_entityname.py` module
3. **Update relationship generation** in `src/services/world_quality_service/_relationship.py`
4. **Update health metrics** in `src/services/world_service/_health.py`
5. **Add build tests** covering the pipeline changes

## Investigation Script Creation

When diagnosing model behavior, performance, or data issues:

1. **Create investigation script** as `scripts/investigate_*.py` (see existing: `investigate_vram_usage.py`, `investigate_build_performance.py`, `investigate_schema_compliance.py`, etc.)
2. **Use shared helpers** from `scripts/_ollama_helpers.py` for Ollama interactions
3. **Respect the 80% GPU residency rule** (`MIN_GPU_RESIDENCY` in `src/services/model_mode_service/_vram.py`) when testing models
4. **Document findings** and implement fixes in the main codebase

## UI Component Enhancement

When adding or improving UI pages/components:

1. **Update component** in `src/ui/components/` or `src/ui/pages/`
2. **Modify related services** for data fetching — services live in `src/services/`, never import UI modules
3. **Update graph renderer** in `src/ui/graph_renderer/` if visualization changes are needed
4. **Add component tests** in `tests/component/` using NiceGUI User fixture

## Naming Conventions

| Element | Convention | Examples |
|---------|-----------|----------|
| Files/modules | snake_case | `story_state.py`, `world_database.py` |
| Private modules | underscore prefix | `_schema.py`, `_quality_loop.py`, `_analytics.py` |
| Classes | PascalCase | `StoryState`, `WorldDatabase`, `BaseAgent` |
| Functions/methods | snake_case | `build_world()`, `generate_structured()` |
| Constants | SCREAMING_SNAKE_CASE | `MIN_GPU_RESIDENCY`, `RECOMMENDED_MODELS` |
| Test files | test_ prefix | `test_settings.py`, `test_world_service.py` |

## Git Conventions

- **Conventional commits**: `type: description` — types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`
- **PR references**: Include `(#123)` in commit messages when related to a PR/issue
- **Never amend or force push** — always create new commits
