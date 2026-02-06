# Code Quality Issues & Fixes

This document tracks code quality issues identified during comprehensive review and their resolutions.

## HIGH Priority

### 1.1 SQL Injection Risk
- **Status:** Fixed
- **File:** `memory/world_database.py`
- **Issue:** Dynamic SQL construction using f-strings in `update_entity`
- **Fix:** Validate column names against explicit whitelist before query construction
- **Test:** `tests/unit/test_memory/test_world_database.py::test_update_entity_rejects_invalid_columns`

### 1.2 Database Connection Leaks
- **Status:** Fixed
- **Files:** `memory/world_database.py`, `services/__init__.py`
- **Issue:** WorldDatabase connections not always closed properly
- **Fix:** Added `__del__` safety net, `shutdown()` to ServiceContainer
- **Test:** `tests/unit/test_resource_cleanup.py`

### 1.3 Memory Leak - Orchestrator Cache
- **Status:** Fixed
- **File:** `services/story_service.py`
- **Issue:** `_orchestrators` dict grows unbounded
- **Fix:** Implemented LRU cache with max_size=10
- **Test:** `tests/unit/test_services/test_story_service.py::test_orchestrator_lru_eviction`

### 1.4 SQLite Thread Safety
- **Status:** Fixed
- **File:** `memory/world_database.py`
- **Issue:** No locking with `check_same_thread=False`
- **Fix:** Added `threading.RLock` around all database operations
- **Test:** `tests/unit/test_thread_safety.py`

### 1.5 Settings Validation
- **Status:** Fixed
- **File:** `settings.py`
- **Issue:** User settings silently overwritten on validation error
- **Fix:** Log warnings, attempt repair by merging valid fields, only overwrite as last resort
- **Test:** `tests/unit/test_settings.py::test_partial_settings_recovery`

## MEDIUM Priority

### 2.1 Graph Rebuild Optimization
- **Status:** Fixed
- **File:** `memory/world_database.py`
- **Issue:** Full graph rebuild on every mutation
- **Fix:** Incremental graph updates (add/remove nodes/edges directly)
- **Test:** `tests/unit/test_memory/test_world_database.py::test_incremental_graph_update`

### 2.2 Entity Validation
- **Status:** Fixed
- **File:** `memory/world_database.py`
- **Issue:** Missing validation on entity_type, attributes depth/size
- **Fix:** Whitelist entity types, limit attributes depth (3) and size (10KB)
- **Test:** `tests/unit/test_memory/test_world_database.py::test_entity_validation`

### 2.3 Configurable Timeouts
- **Status:** Fixed
- **Files:** `settings.py`, `agents/base.py`
- **Issue:** Hardcoded timeouts (120s, 10s)
- **Fix:** Added `ollama_timeout`, `subprocess_timeout` to Settings
- **Test:** `tests/unit/test_settings.py::test_timeout_settings`

### 2.4 Rate Limiting
- **Status:** Fixed
- **File:** `agents/base.py`
- **Issue:** No limit on concurrent LLM requests
- **Fix:** Added semaphore with max_concurrent=2
- **Test:** `tests/unit/test_agents/test_base_agent.py::test_rate_limiting`

### 2.5 Language Mapping Deduplication
- **Status:** Fixed
- **Files:** `utils/constants.py`, `services/orchestrator`, `services/export_service/`
- **Issue:** Identical `lang_map` in two places
- **Fix:** Moved to shared `utils/constants.py`
- **Test:** Both usages import from shared location

## LOW Priority

### 3.1 CLI Logging
- **Status:** Fixed
- **File:** `main.py`
- **Issue:** Print statements instead of logger
- **Fix:** Replaced print with logger, added CLI formatter
- **Test:** `tests/integration/test_main_startup.py`

### 3.2 Exception Hierarchy
- **Status:** Fixed
- **File:** `utils/exceptions.py`
- **Issue:** Inconsistent exception patterns
- **Fix:** Documented and enforced hierarchy
- **Test:** Verified inheritance in exception classes

### 3.3 XSS in Export
- **Status:** Fixed
- **File:** `services/export_service.py`
- **Issue:** Paragraph content not escaped in EPUB
- **Fix:** Added `html.escape()` to paragraph content
- **Test:** `tests/unit/test_services/test_export_service.py::test_html_special_chars_escaped`

### 3.4 Database Migrations
- **Status:** Fixed
- **Files:** `memory/world_database.py`, `memory/mode_database.py`
- **Issue:** No schema versioning
- **Fix:** Added `schema_version` table and migration support
- **Test:** `tests/unit/test_memory/test_migrations.py`

### 3.5 Chapter Validation
- **Status:** Fixed
- **File:** `services/orchestrator`
- **Issue:** Unhelpful error for invalid chapter numbers
- **Fix:** Added bounds check with descriptive error message
- **Test:** `tests/unit/test_orchestrator.py::test_chapter_bounds_error`

## Summary

All 15 code quality issues have been addressed:

| Priority | Total | Fixed |
|----------|-------|-------|
| HIGH     | 5     | 5     |
| MEDIUM   | 5     | 5     |
| LOW      | 5     | 5     |

### Verification

```bash
# All tests pass
pytest tests/ -v

# No lint errors
ruff check .

# Code is formatted
ruff format --check .
```

### New Files Created

| File | Purpose |
|------|---------|
| `utils/exceptions.py` | Centralized exception hierarchy |
| `utils/constants.py` | Shared constants (language codes) |

### Key Improvements

1. **Security**: SQL injection prevention, XSS escaping in exports
2. **Reliability**: Thread-safe database, connection leak prevention
3. **Performance**: Incremental graph updates, LRU cache for orchestrators
4. **Scalability**: Rate limiting for LLM requests, configurable timeouts
5. **Maintainability**: Exception hierarchy, deduplicated constants, improved logging

## Agent Method Integrations

Previously unused agent methods have been fully integrated into the workflow:

### Full Story Review (`check_full_story`)
- **File:** `agents/continuity.py:128`
- **Integration:** `services/orchestrator:write_all_chapters()`
- **Purpose:** Final story-wide continuity check after all chapters complete
- **Checks:** Unresolved plot threads, incomplete character arcs, foreshadowing payoff, timeline consistency
- **Service:** `StoryService.review_full_story()`

### Scene Continuation (`continue_scene`)
- **File:** `agents/writer.py:134`
- **Integration:** `services/orchestrator:continue_chapter()`
- **Purpose:** Continue writing from where text left off
- **Features:** Optional direction parameter, maintains voice/style, seamless continuation
- **Service:** `StoryService.continue_chapter()`

### Passage Editing (`edit_passage`)
- **File:** `agents/editor.py:83`
- **Integration:** `services/orchestrator:edit_passage()`
- **Purpose:** Targeted editing of specific text passages
- **Features:** Optional focus area (dialogue, pacing, description)
- **Service:** `StoryService.edit_passage()`

### Edit Suggestions (`get_edit_suggestions`)
- **File:** `agents/editor.py:102`
- **Integration:** `services/orchestrator:get_edit_suggestions()`
- **Purpose:** Review mode - get suggestions without applying changes
- **Output:** Specific suggestions with quotes and improvements
- **Service:** `StoryService.get_edit_suggestions()`

### Workflow Changes

The `write_all_chapters()` method now includes a final review step:
1. Write all chapters (existing)
2. **NEW: Call `check_full_story()` for final continuity review**
3. Report any issues found (doesn't block completion)
4. Mark story complete

New orchestrator methods exposed via `StoryService`:
- `continue_chapter(chapter_num, direction)` - Continue writing
- `edit_passage(text, focus)` - Targeted editing
- `get_edit_suggestions(text)` - Review mode
- `review_full_story()` - On-demand full review
