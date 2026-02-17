# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

**Never defer work.** Do not suggest "this can be done later" or "consider for a future PR". Complete all requested changes fully.

**No placeholder code.** Every piece of code must be fully functional. No `# TODO`, no `pass` as placeholders, no "implement later" comments.

**No partial implementations.** If you start refactoring something, finish it completely. Don't leave work as "pending" or "remaining".

**No boilerplate.** Don't generate generic/template code that needs to be filled in. Write the actual implementation.

**No default fallbacks.** Never use `.get(key, default_value)` patterns for configuration values. All configurable values must be explicitly defined in settings with proper validation. If a value is missing, raise an error rather than silently using a default.

**Everything needs logging.** All functions and methods should have appropriate logging for debugging and traceability. Use `logger.debug()` for routine operations, `logger.info()` for significant events, `logger.warning()` for unexpected but recoverable situations, and `logger.error()` for failures.

**Everything needs tests.** All new code must have corresponding unit tests. When modifying existing code, update related tests. Tests should cover both happy paths and edge cases.

**No git history rewriting.** Never use `git commit --amend` or `git push --force`. Always create new commits to fix issues. This preserves history and avoids force-push problems.

**No bypassing CI.** Never use `git push --no-verify` or modify test coverage thresholds to make tests pass. If tests fail, fix the actual issue. Pre-push hooks exist to catch problems before they reach CI.

**No skipping coverage.** Never exclude files from coverage requirements to make tests pass. Write proper tests to achieve 100% coverage. Acceptable exclusions are: (1) `# pragma: no cover` on pure UI widget creation code (code that only creates GUI widgets with no business logic), and (2) standalone utility scripts in `scripts/` that are not part of the main application. If coverage is failing for core application code, write tests for the uncovered code.

**Settings must have UI.** Any new settings added to `src/settings.py` MUST be exposed in the Settings page UI (`src/ui/pages/settings.py`). Users should be able to configure all settings through the web interface, not just by editing JSON files.

**Always update existing documentation.** When making changes that affect architecture, services, key files, settings, or workflows, update the relevant sections of existing documentation (this CLAUDE.md, README.md, etc.) to reflect those changes. Never create new standalone documentation files unless explicitly requested, but always keep existing docs accurate and up to date.

## Project Overview

Story Factory is a local AI-powered multi-agent system for generating stories using Ollama. Five specialized agents (Interviewer, Architect, Writer, Editor, Continuity Checker) collaborate through an iterative write-edit-check loop.

## Common Commands

```bash
# Run application
python main.py              # Web UI on http://localhost:7860
python main.py --cli        # CLI mode

# Testing
pytest                      # Run all tests
pytest tests/unit/test_settings.py  # Run single test file
pytest --cov=. --cov-report=term    # With coverage

# Code quality
ruff format .               # Format code
ruff check .                # Lint
ruff check --fix .          # Auto-fix lint issues
make check                  # Format + lint + test
```

## Architecture

**Clean architecture with Service Container pattern for dependency injection:**

```
main.py
  └── ServiceContainer(settings)  # Creates all services once
       ├── ProjectService         # Project CRUD
       ├── StoryService           # Story generation workflow
       ├── WorldService           # Entity/relationship management
       ├── ModelService           # Ollama model operations
       ├── ExportService          # Export formats
       ├── ModelModeService       # Model performance tracking
       ├── ScoringService         # Quality scoring
       ├── WorldQualityService    # World quality enhancement
       ├── SuggestionService      # AI-powered suggestions
       ├── TemplateService        # Story template management
       ├── BackupService          # Project backup/restore
       ├── ImportService          # Import entities from text
       ├── ComparisonService      # Model comparison testing
       ├── EmbeddingService       # Vector embeddings via Ollama
       └── ContextRetrievalService # RAG context retrieval for agent prompts
```

**Layer responsibilities:**
- **src/services/**: Business logic layer - no UI imports, receives settings via DI; includes orchestrator.py
- **src/ui/**: NiceGUI components - only calls services, manages UI state via AppState
- **src/agents/**: AI agent implementations - extend BaseAgent, use retry logic
- **src/memory/**: Pydantic models (StoryState, Character, Chapter) and WorldDatabase (SQLite + NetworkX + sqlite-vec)
- **src/prompts/**: YAML prompt templates loaded at runtime

**Key patterns:**
- Pages implement `build()` method, receive `AppState` and `ServiceContainer`
- Centralized UI state in `src/ui/state.py` (AppState class)
- JSON extraction from LLM responses via `src/utils/json_parser.py`
- Error handling decorators: `@handle_ollama_errors`, `@retry_with_fallback`
- Thread-safe database operations with `threading.RLock`
- LRU cache for orchestrators to prevent memory leaks
- Rate limiting for concurrent LLM requests (max 2 concurrent)
- Centralized exception hierarchy in `src/utils/exceptions.py`

## Agent Workflow

```
User Input → Interviewer → Architect → [Writer → Editor → Continuity] × N → Final Story
                                            └────── revision loop ───────┘
```

**RAG Context Pipeline:**

The writing agents receive semantically relevant world context via a RAG pipeline:
```text
WorldDatabase (sqlite-vec) → EmbeddingService → ContextRetrievalService → StoryOrchestrator → Agent prompts
```
- `EmbeddingService` generates vector embeddings via Ollama and attaches auto-embedding callbacks to `WorldDatabase`
- `ContextRetrievalService` performs KNN vector search to retrieve relevant entities/relationships for the current writing task
- `StoryOrchestrator` retrieves RAG context before each agent call and passes it as `world_context` to Writer, Editor, and Continuity agents
- RAG retrieval failures are non-fatal — agents proceed with empty context if retrieval fails

## Threading Model

The codebase uses proper thread-safety patterns for concurrent operations:
- Double-checked locking pattern for singleton initialization
- `threading.RLock` for database operations in `WorldDatabase`
- `threading.Semaphore` for LLM rate limiting in `BaseAgent`
- YAML templates loaded once at startup under lock protection

Note: These patterns also work with experimental free-threaded Python builds (no-GIL, `python3.14t`), though we don't recommend using experimental features in production. PyYAML 6.0.3+ includes thread-safe C extensions for those builds.

## Code Style

- **Line length**: 100 characters (Ruff)
- **Python**: 3.14+
- **Type hints**: Encouraged but not enforced (gradual adoption)
- **Imports**: Auto-sorted by Ruff
- **Dependency parsing**: Use `packaging.requirements.Requirement` instead of regex - handles all version specifiers (`==`, `>=`, `~=`, etc.) correctly.

## Cross-Platform Gotchas

- `subprocess.CREATE_NO_WINDOW` is Windows-only - use `getattr(subprocess, "CREATE_NO_WINDOW", 0)` for cross-platform code
- Pre-commit hooks run only on staged files locally, but pre-commit.ci runs on full repo - use `pass_filenames: false` for tools that should check entire directories
- **Never create a file called `nul`** (or `NUL`, `con`, `prn`, `aux`, etc.) on Windows — these are reserved device names. A `nul` file in the repo root causes testmon (and other path-resolution tools) to crash with `ValueError: path is on mount '\\\\.\\NUL'`. If this happens, delete the file (`rm nul`) and the error resolves immediately.

## After Making Changes

1. `ruff format .` - Format code
2. `ruff check .` - Lint
3. `pytest` - Verify tests pass

## Key Files

- `src/settings.py`: Settings management, model registry, configurable timeouts
- `src/settings.json`: User configuration (gitignored, copy from `src/settings.example.json`)
- `src/ui/state.py`: Centralized UI state (AppState)
- `src/services/__init__.py`: ServiceContainer for dependency injection
- `src/services/orchestrator/`: StoryOrchestrator coordinates agents through the story creation pipeline
- `src/agents/base.py`: BaseAgent with retry logic, rate limiting, configurable timeout
- `src/utils/exceptions.py`: Centralized exception hierarchy (StoryFactoryError, LLMError, etc.)
- `src/utils/constants.py`: Shared constants (language codes, etc.)
- `src/memory/world_database.py`: SQLite + NetworkX + sqlite-vec with thread safety
- `src/services/context_retrieval_service.py`: RAG context retrieval via vector similarity search
- `src/services/embedding_service.py`: Vector embedding generation via Ollama

## Testing

- Unit tests in `tests/unit/` with `test_*.py` naming
- Component tests in `tests/component/` using NiceGUI User fixture
- Integration tests in `tests/integration/`
- Mock Ollama in tests to avoid requiring running instance
- Shared fixtures in `tests/conftest.py`
- **Always run tests in background** to avoid blocking on long test runs when working on todo lists
- **Never run full test suite scans** - only run tests for specific files when needed (e.g., `pytest tests/unit/test_settings.py`). Full test runs take too long and should only be done by CI.
- **Single tasks: do them directly. Multiple tasks: use agents.** If there is only one thing to fix/change, do it inline in the main conversation. Only spawn background agents when there are multiple independent tasks to parallelize.

**Test patterns:**
- Use `threading.Event` instead of `time.sleep()` for thread synchronization in tests
- When testing Windows-specific code paths, patch `sys.platform` to `"win32"`.

**Test mocking gotchas:**
- Tests must use **fake model names** (e.g., `test-model:8b`, `fake-writer:latest`) — never use real model IDs from `RECOMMENDED_MODELS` in test code. The `mock_ollama_globally` autouse fixture in conftest.py mocks `Settings.get_model_tags()` so fake names resolve to all required role tags.
- `mock_ollama_globally` fixture in conftest.py is autouse — all tests automatically mock Ollama
- Ollama API responses use both dict (`models.get("models")`) and object (`response.models`) patterns — mocks must support both

## Ollama Integration

- Default endpoint: `http://localhost:11434`
- Recommended model: `huihui_ai/dolphin3-abliterated:8b`
- Temperature varies by agent role (writer: 0.9, editor: 0.6, continuity: 0.3)
- Context size: 32768 tokens default
- **80% GPU residency rule**: Never run a model unless at least 80% of it fits in GPU VRAM. Models split heavily between GPU and system RAM run drastically slower (5-10x). For a 24 GB GPU, max model size is ~30 GB. This is enforced in `select_model_with_size_preference()` and should be respected by all investigation scripts. The constant `MIN_GPU_RESIDENCY` lives in `src/services/model_mode_service/_vram.py`.

## Data Storage

- Stories: `output/stories/` (JSON files, UUIDs as IDs)
- World databases: `output/worlds/` (SQLite)
- Logs: `output/logs/story_factory.log`
- Backups: `output/backups/`
