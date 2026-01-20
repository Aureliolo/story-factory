# Story Factory - Copilot Instructions

This is a Python-based multi-agent system for generating stories using local AI models via Ollama. Five specialized AI agents (Interviewer, Architect, Writer, Editor, Continuity Checker) collaborate through an iterative write-edit-check loop to create short stories, novellas, and novels with continuous refinement.

## Critical Rules

**Never defer work.** Do not suggest "this can be done later" or "consider for a future PR". Complete all requested changes fully.

**No placeholder code.** Every piece of code must be fully functional. No `# TODO`, no `pass` as placeholders, no "implement later" comments.

**No partial implementations.** If you start refactoring something, finish it completely. Don't leave work as "pending" or "remaining".

**No boilerplate.** Don't generate generic/template code that needs to be filled in. Write the actual implementation.

**No default fallbacks.** Never use `.get(key, default_value)` patterns for configuration values. All configurable values must be explicitly defined in settings with proper validation. If a value is missing, raise an error rather than silently using a default.

**Everything needs logging.** All functions and methods should have appropriate logging for debugging and traceability. Use `logger.debug()` for routine operations, `logger.info()` for significant events, `logger.warning()` for unexpected but recoverable situations, and `logger.error()` for failures.

**Everything needs tests.** All new code must have corresponding unit tests. When modifying existing code, update related tests. Tests should cover both happy paths and edge cases.

**Run tests in background.** When working on multi-step tasks, always run tests in background to avoid blocking on long test runs.

## Code Standards

### Python Best Practices
- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate (Pydantic models for data validation in `memory/story_state.py`, dataclasses in `settings.py`)
- Write clear, descriptive variable and function names
- Keep functions focused and single-purpose
- Use docstrings for classes and complex functions
- Line length: 100 characters (enforced by Ruff)
- Python version: 3.14+
- **NEVER** add `from __future__ import annotations` - Python 3.14+ has native support for deferred annotation evaluation and does not require this import

### Required Before Each Commit
**CRITICAL**: After making code changes, always run in this order:
1. `ruff format .` - Format code with Ruff
2. `ruff check .` - Lint code (use `ruff check --fix .` to auto-fix)
3. `pytest` - Run tests to verify nothing broke
4. Commit and push changes
5. **MUST** verify CI passes on GitHub (check Actions tab) - never leave a PR with failing CI

**MANDATORY CI VERIFICATION**: After every push, you MUST check that CI passes. If CI fails, fix the issue immediately and push again until all checks are green. Never walk away from a failing CI pipeline.

### Coverage Requirements
**100% test coverage is MANDATORY for every commit**. The CI enforces 100% coverage on core modules (`agents/`, `services/`, `workflows/`, `memory/`, `utils/`, `settings.py`). If your code reduces coverage, the CI will fail and you must add tests before the PR can be merged.

### Development Flow
- **Install dependencies**: `pip install -r requirements.txt`
- **Build**: Not applicable (Python project, no build step)
- **Test**: `pytest` (runs all tests in `tests/` directory)
- **Test with coverage**: `pytest --cov=. --cov-report=term --cov-fail-under=100 tests/unit tests/smoke tests/integration`
- **Full CI check**: `make test-ci` (runs tests with 100% coverage enforcement on core modules)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Run the application**: `python main.py` (starts web UI on http://localhost:7860)
- **Run in CLI mode**: `python main.py --cli`

### Testing Guidelines
- Write unit tests for new functionality using pytest
- Place test files in appropriate directories:
  - `tests/unit/` - Unit tests with `test_*.py` naming convention
  - `tests/smoke/` - Quick startup validation tests
  - `tests/integration/` - Integration tests for component interactions
  - `tests/e2e/` - End-to-end browser tests (require playwright)
- Mock Ollama API calls in tests to avoid requiring a running Ollama instance
- Use pytest fixtures for test setup (shared fixtures in `tests/conftest.py`)
- Core modules (`agents/`, `services/`, `workflows/`, `memory/`, `utils/`, `settings.py`) must maintain 100% test coverage
- UI components (`ui/`) are excluded from coverage requirements until NiceGUI component tests are added

## Repository Structure

- `main.py`: Entry point for the application (supports both web UI and CLI modes)
- `settings.py`: Settings management and model registry using dataclasses
- `settings.json`: User configuration file (not in git, copy from `settings.example.json`)
- `requirements.txt`: Python dependencies (runtime, testing, and code quality)
- `pyproject.toml`: Tool configuration (ruff, mypy, pytest, coverage)
- `Makefile`: Development task shortcuts
- `docs/`: Documentation files
  - `MODELS.md`: Model recommendations and usage guidelines
  - `ARCHITECTURE.md`: System architecture documentation
  - `UX_UI_IMPROVEMENTS.md`: UI/UX feature documentation
- `scripts/`: Utility scripts
  - `healthcheck.py`: System health check utility
  - `start.ps1`: PowerShell launcher script
- `agents/`: AI agent implementations (extend BaseAgent)
  - `base.py`: Base agent class with retry logic, rate limiting, configurable timeout
  - `interviewer.py`: Gathers story requirements from users
  - `architect.py`: Designs world, characters, and plot structure
  - `writer.py`: Generates prose content
  - `editor.py`: Polishes and refines writing
  - `continuity.py`: Detects plot holes and inconsistencies
  - `validator.py`: Validates AI responses
- `workflows/`: Agent coordination and orchestration
  - `orchestrator.py`: Manages the multi-agent workflow with LRU cache to prevent memory leaks
- `memory/`: Story state and world management
  - `story_state.py`: Pydantic models for story context (StoryState, Character, Chapter)
  - `entities.py`: Entity and relationship models
  - `world_database.py`: SQLite + NetworkX world database with thread safety and schema migrations
- `services/`: Business logic layer (no UI imports, receives settings via DI)
  - `__init__.py`: ServiceContainer for dependency injection
  - `project_service.py`: Project CRUD operations
  - `story_service.py`: Story generation workflow
  - `world_service.py`: Entity management
  - `model_service.py`: Ollama model operations
  - `export_service.py`: Export to various formats (markdown, text, HTML, EPUB, PDF)
- `utils/`: Utility modules
  - `json_parser.py`: JSON extraction and parsing from LLM responses
  - `logging_config.py`: Logging configuration and setup
  - `error_handling.py`: Error handling decorators (@handle_ollama_errors, @retry_with_fallback)
  - `exceptions.py`: Centralized exception hierarchy (StoryFactoryError, LLMError, etc.)
  - `constants.py`: Shared constants (language codes, etc.)
- `ui/`: User interface components (NiceGUI 3.x)
  - `app.py`: Main NiceGUI application
  - `state.py`: Centralized UI state management (AppState class)
  - `theme.py`: Colors, styles, and theme utilities
  - `pages/`: Page components implementing `build()` method (write, world, projects, settings, models, analytics)
  - `components/`: Reusable UI components (header, chat, graph, common)
- `output/`: Generated outputs (gitignored)
  - `stories/`: Story output files (JSON with UUIDs as IDs)
  - `worlds/`: World database files (SQLite)
- `tests/`: Test suite using pytest (849+ tests, 100% coverage on core modules)
  - `unit/`: Unit tests
  - `smoke/`: Quick startup validation tests
  - `integration/`: Integration tests
  - `e2e/`: End-to-end browser tests
  - `conftest.py`: Shared pytest fixtures
- `logs/`: Application logs (written to `logs/story_factory.log`)
- `.github/workflows/`: CI/CD workflows
  - `ci.yml`: Test and code quality checks (100% coverage on core modules)

## Architecture & Key Patterns

### Clean Architecture with Service Container Pattern

The application uses dependency injection via ServiceContainer:

```python
main.py
  └── ServiceContainer(settings)  # Creates all services once
       ├── ProjectService         # Project CRUD
       ├── StoryService           # Story generation workflow
       ├── WorldService           # Entity/relationship management
       ├── ModelService           # Ollama model operations
       └── ExportService          # Export formats
```

**Layer responsibilities:**
- **services/**: Business logic layer - no UI imports, receives settings via DI
- **ui/**: NiceGUI components - only calls services, manages UI state via AppState
- **agents/**: AI agent implementations - extend BaseAgent, use retry logic
- **memory/**: Pydantic models (StoryState, Character, Chapter) and WorldDatabase (SQLite + NetworkX)
- **workflows/**: StoryOrchestrator coordinates agents through the story creation pipeline

### Key Design Patterns

1. **Service Container Pattern**: All services created once in `services/__init__.py`, injected into pages
2. **Centralized UI State**: `ui/state.py` manages AppState singleton
3. **Base Agent Pattern**: All agents extend `agents/base.py` with retry logic and rate limiting
4. **Error Handling Decorators**: `@handle_ollama_errors`, `@retry_with_fallback` in `utils/error_handling.py`
5. **JSON Extraction**: Use `utils/json_parser.py` for parsing LLM responses (handles markdown code blocks)
6. **Thread-Safe Database**: WorldDatabase uses `threading.RLock` for concurrent access
7. **LRU Cache**: Orchestrators cached to prevent memory leaks
8. **Rate Limiting**: Max 2 concurrent LLM requests to prevent overload

### NiceGUI UI Pattern

- **Framework**: Uses NiceGUI 3.x for the web interface
- **Page Structure**: Pages implement `build()` method, receive `AppState` and `ServiceContainer`
- **UI Elements**: Import from `nicegui import ui`
- **HTML Elements**: Use `ui.html(content, sanitize=False)` for trusted HTML with JavaScript
- **Component Types**: `ui.card()`, `ui.button()`, `ui.input()`, `ui.label()`, etc.
- **Async Operations**: Use `async def` for methods that call async services
- **Notifications**: Use `ui.notify(message, type="positive|negative|warning|info")`
- **Testing**: UI changes should be tested by running `python main.py` and checking http://localhost:7860

## Agent Workflow

The multi-agent story generation workflow:

```
User Input → Interviewer → Architect → [Writer → Editor → Continuity] × N → Final Story
                                            └────── revision loop ───────┘
```

1. **Interview Phase**: Interviewer gathers story requirements (genre, tone, length, characters)
2. **Architecture Phase**: Architect creates world-building, character profiles, plot outline
3. **Writing Phase**: For each chapter (iterative refinement):
   - Writer drafts the content
   - Editor polishes the prose
   - Continuity Checker validates consistency
   - If issues found, loop back to Writer (max 3 iterations)
4. **Output**: Complete story exported in preferred format (markdown, text, EPUB, PDF)

### Agent Temperatures
- **Writer**: 0.9 (high creativity)
- **Editor**: 0.6 (balanced)
- **Continuity**: 0.3 (strict, analytical)
- **Architect**: 0.85 (creative but structured)
- **Interviewer**: 0.7 (conversational)

## Key Guidelines

1. **Ollama Integration**:
   - All AI agents use Ollama for local LLM serving
   - Default endpoint: `http://localhost:11434`
   - Recommended model: `huihui_ai/dolphin3-abliterated:8b`
   - Context size: 32768 tokens default
   - Respect existing model configuration patterns

2. **Agent Architecture**:
   - Each agent has a specific role - maintain separation of concerns
   - Don't mix agent responsibilities
   - Use the base agent class for common functionality
   - Follow the established agent interface patterns
   - All agents extend `BaseAgent` from `agents/base.py`

3. **State Management**:
   - Story state is maintained through `memory/story_state.py` module
   - Use Pydantic models for validation (StoryState, Character, Chapter, etc.)
   - World data stored in SQLite + NetworkX (thread-safe operations)

4. **Error Handling**:
   - Handle Ollama connection errors gracefully
   - Use decorators: `@handle_ollama_errors`, `@retry_with_fallback`
   - Centralized exceptions in `utils/exceptions.py`
   - Provide informative error messages

5. **Configuration**:
   - Settings managed through dataclasses in `settings.py`
   - User configuration in `settings.json` (copy from `settings.example.json`)
   - All configurable values must be explicitly defined - no `.get()` defaults

6. **Dependencies**:
   - Minimize external dependencies
   - When adding new dependencies, add them to `requirements.txt`
   - Pin dependency versions with ranges (e.g., `>=4.0.0,<7.0.0`)

7. **Documentation**:
   - Update README.md for significant feature changes
   - Update docs/MODELS.md when adding model recommendations
   - Keep docstrings up to date with code changes
   - Document complex algorithms and business logic

8. **Logging**:
   - Logs written to `logs/story_factory.log`
   - Use `logger = logging.getLogger(__name__)`
   - Log levels: debug (routine), info (significant), warning (unexpected), error (failures)

9. **JSON Parsing**:
   - Use `utils/json_parser.py` for extracting JSON from LLM responses
   - LLMs may include JSON in markdown code blocks or with surrounding text
   - Handle malformed JSON gracefully
