# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

**Never defer work.** Do not suggest "this can be done later" or "consider for a future PR". Complete all requested changes fully.

**No placeholder code.** Every piece of code must be fully functional. No `# TODO`, no `pass` as placeholders, no "implement later" comments.

**No partial implementations.** If you start refactoring something, finish it completely. Don't leave work as "pending" or "remaining".

**No boilerplate.** Don't generate generic/template code that needs to be filled in. Write the actual implementation.

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
       └── ExportService          # Export formats
```

**Layer responsibilities:**
- **services/**: Business logic layer - no UI imports, receives settings via DI
- **ui/**: NiceGUI components - only calls services, manages UI state via AppState
- **agents/**: AI agent implementations - extend BaseAgent, use retry logic
- **memory/**: Pydantic models (StoryState, Character, Chapter) and WorldDatabase (SQLite + NetworkX)
- **workflows/**: StoryOrchestrator coordinates agents through the story creation pipeline

**Key patterns:**
- Pages implement `build()` method, receive `AppState` and `ServiceContainer`
- Centralized UI state in `ui/state.py` (AppState class)
- JSON extraction from LLM responses via `utils/json_parser.py`
- Error handling decorators: `@handle_ollama_errors`, `@retry_with_fallback`

## Agent Workflow

```
User Input → Interviewer → Architect → [Writer → Editor → Continuity] × N → Final Story
                                            └────── revision loop ───────┘
```

## Code Style

- **Line length**: 100 characters (Ruff)
- **Python**: 3.13+
- **Type hints**: Encouraged but not enforced (gradual adoption)
- **Imports**: Auto-sorted by Ruff

## After Making Changes

1. `ruff format .` - Format code
2. `ruff check .` - Lint
3. `pytest` - Verify tests pass

## Key Files

- `settings.py`: Settings management and model registry (dataclass-based)
- `settings.json`: User configuration (gitignored, copy from `settings.example.json`)
- `ui/state.py`: Centralized UI state (AppState)
- `services/__init__.py`: ServiceContainer for dependency injection
- `agents/base.py`: BaseAgent with retry logic and common functionality

## Testing

- Tests in `tests/unit/` with `test_*.py` naming
- Mock Ollama in tests to avoid requiring running instance
- Shared fixtures in `tests/conftest.py`

## Ollama Integration

- Default endpoint: `http://localhost:11434`
- Recommended model: `huihui_ai/qwen3-abliterated:8b`
- Temperature varies by agent role (writer: 0.9, editor: 0.6, continuity: 0.3)
- Context size: 32768 tokens default

## Data Storage

- Stories: `output/stories/` (JSON files, UUIDs as IDs)
- World databases: `output/worlds/` (SQLite)
- Logs: `logs/story_factory.log`
