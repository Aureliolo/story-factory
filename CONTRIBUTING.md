# Contributing to Story Factory

This document provides guidelines and best practices for working on the Story Factory codebase, primarily designed for AI coding assistants and the repository owner.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Aureliolo/story-factory.git
cd story-factory

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .

# Run the application
python main.py
```

## Development Workflow

### 1. Making Changes

1. **Create a branch** for your changes (handled by AI assistants automatically)
2. **Make minimal, focused changes** - change as few lines as possible
3. **Write tests** for new functionality
4. **Run tests** after each change: `pytest`
5. **Format code**: `ruff format .`
6. **Lint code**: `ruff check .` (or `ruff check --fix .` to auto-fix)
7. **Commit frequently** with descriptive messages

### 2. Code Quality Standards

All code must pass:
- ✅ **Tests**: `pytest` (100 tests must pass)
- ✅ **Formatting**: `ruff format --check .`
- ✅ **Linting**: `ruff check .`

### 3. After Every Code Change

**CRITICAL**: Always run this sequence after making changes:

```bash
ruff format .
ruff check .
pytest
git add .
git commit -m "descriptive message"
git push
```

Then verify CI passes on GitHub Actions.

## Project Structure

```
story-factory/
├── agents/              # AI agent implementations
│   ├── base.py          # BaseAgent with LLM communication
│   ├── interviewer.py   # Story requirements gathering
│   ├── architect.py     # Structure and character design
│   ├── writer.py        # Prose generation
│   ├── editor.py        # Refinement and polish
│   └── continuity.py    # Plot hole detection
├── workflows/
│   └── orchestrator.py  # Agent coordination
├── memory/
│   ├── story_state.py   # Story state management (Pydantic)
│   ├── entities.py      # Entity/Relationship models
│   └── world_database.py # SQLite + NetworkX database
├── services/            # Business logic layer
│   ├── project_service.py
│   ├── story_service.py
│   ├── world_service.py
│   ├── model_service.py
│   └── export_service.py
├── ui/                  # NiceGUI web interface
│   ├── app.py           # Main application
│   ├── state.py         # UI state management
│   ├── theme.py         # Colors and styles
│   ├── pages/           # Page components
│   └── components/      # Reusable UI components
├── utils/               # Utility modules
│   ├── json_parser.py   # JSON extraction from LLM responses
│   ├── logging_config.py
│   ├── error_handling.py
│   └── prompt_builder.py # Prompt construction helper
└── tests/               # Test suite (pytest)
    └── unit/            # Unit tests
```

## Architecture Patterns

### Agent Pattern

All agents inherit from `BaseAgent` and follow this structure:

```python
class MyAgent(BaseAgent):
    def __init__(self, model: str | None = None, settings=None):
        super().__init__(
            name="AgentName",
            role="Agent Role",
            system_prompt=SYSTEM_PROMPT,
            agent_role="agent_name",  # for settings lookup
            model=model,
            settings=settings,
        )
    
    def my_method(self, story_state: StoryState) -> ReturnType:
        # Validate brief
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        
        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_language_requirement(brief.language)
        builder.add_brief_requirements(brief)
        # ... add more sections
        
        prompt = builder.build()
        response = self.generate(prompt)
        return response
```

### Service Pattern

Services use dependency injection via `ServiceContainer`:

```python
class MyService:
    def __init__(self, settings: Settings | None = None):
        """Initialize service with settings."""
        self.settings = settings or Settings.load()
    
    def my_method(self, ...):
        # Business logic here
        pass
```

### UI Pattern (NiceGUI)

Pages are created as classes with dependency injection:

```python
def create_my_page(services: ServiceContainer, state: AppState):
    """Create and return the page UI."""
    
    with ui.card():
        ui.label("My Page")
        # ... UI elements
    
    # Event handlers
    def on_button_click():
        # Handle event
        pass
    
    return container  # Return reference if needed
```

## Testing Guidelines

### Test Structure

```python
class TestMyFeature:
    """Tests for MyFeature."""
    
    def test_specific_behavior(self):
        """Should demonstrate specific behavior."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_my_feature.py

# Run with coverage
pytest --cov=. --cov-report=term

# Run specific test
pytest tests/unit/test_my_feature.py::TestMyFeature::test_specific_behavior -v
```

### Mocking Ollama

When testing agents, mock Ollama API calls:

```python
def test_my_agent(monkeypatch):
    """Test agent without calling Ollama."""
    
    # Mock the generate method
    def mock_generate(self, prompt, context=None, temperature=None):
        return "Mocked response"
    
    monkeypatch.setattr(BaseAgent, "generate", mock_generate)
    
    # Now test the agent
    agent = MyAgent()
    result = agent.my_method(story_state)
    assert result == "Mocked response"
```

## Common Development Tasks

### Adding a New Agent Method

1. Use `PromptBuilder.ensure_brief()` for validation
2. Build prompts with `PromptBuilder` class
3. Call `self.generate(prompt)` for LLM response
4. Parse JSON responses with `utils.json_parser` utilities
5. Add comprehensive docstrings
6. Write unit tests

### Adding a New Service

1. Accept `settings: Settings | None` in `__init__`
2. Store as `self.settings = settings or Settings.load()`
3. Add to `ServiceContainer` in `services/__init__.py`
4. Write unit tests in `tests/unit/test_services/`

### Adding a New UI Page

1. Create page in `ui/pages/my_page.py`
2. Accept `services: ServiceContainer` and `state: AppState` parameters
3. Use `state` for shared UI state
4. Use `services` to call business logic
5. Use `ui.notify()` for user feedback
6. Register in `ui/app.py`

### Adding a New Setting

1. Add field to `Settings` dataclass in `settings.py`
2. Update `validate()` method with validation rules
3. Add UI controls in `ui/pages/settings.py`
4. Update `settings.example.json`
5. Write validation tests

## Code Style Guidelines

### Python Style

- **Line length**: 100 characters (enforced by ruff)
- **Imports**: Organized with isort (via ruff)
- **Type hints**: Use where helpful, not required everywhere
- **Docstrings**: Use for classes and complex functions
- **F-strings**: Preferred for string formatting
- **Comprehensions**: Use when they improve readability

### Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Type variables**: `T`, `ModelT`, etc.

### Comments

- Write code that is self-explanatory
- Add comments only when:
  - Explaining *why* something is done (not *what*)
  - Complex algorithms or non-obvious logic
  - Workarounds or temporary solutions
  - Important context for future maintainers

### Error Handling

```python
# Use specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except ConnectionError as e:
    logger.warning(f"Connection failed: {e}")
    return default_value

# Use error handling decorators from utils.error_handling
@handle_ollama_errors(default_return=("", "Error"), raise_on_error=False)
def my_ollama_call():
    # Ollama API call
    pass
```

## Debugging Tips

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed diagnostic info")
logger.info("General informational messages")
logger.warning("Warning messages for unexpected situations")
logger.error("Error messages for failures")
```

Logs are written to `logs/story_factory.log`.

### UI Debugging

1. Check browser console for JavaScript errors
2. Use `ui.notify()` to display debug info in UI
3. Check network tab for failed requests
4. Inspect NiceGUI state with browser dev tools

### Agent Debugging

1. Log prompts being sent to LLM: `logger.debug(f"Prompt: {prompt}")`
2. Log raw LLM responses: `logger.debug(f"Response: {response}")`
3. Test prompt building separately from LLM calls
4. Use smaller/faster models for debugging

## CI/CD

GitHub Actions runs on every push:

1. **Tests**: All tests must pass
2. **Code Quality**: Ruff formatting and linting must pass
3. **Coverage**: Coverage reports uploaded to Codecov

Check the Actions tab on GitHub to see CI results.

## Performance Considerations

### LLM Calls

- Use appropriate temperature settings (higher for creative, lower for structured)
- Implement retry logic with exponential backoff (handled by `BaseAgent`)
- Cache responses when appropriate
- Use smaller models for validation tasks

### Database

- WorldDatabase uses SQLite - keep queries simple
- Batch operations when possible
- Close database connections properly

### UI

- Avoid expensive operations in UI thread
- Use async operations for long-running tasks
- Implement progress indicators for user feedback

## Security Best Practices

- **Never commit secrets** to repository
- **Validate all user input** in services layer
- **Use parameterized SQL queries** (handled by WorldDatabase)
- **Sanitize file paths** when reading/writing files
- **Check settings validation** before using values

## Getting Help

- Check existing tests for examples
- Review similar features in the codebase
- Check GitHub Issues for known problems
- Consult architecture docs in `docs/ARCHITECTURE.md`

## Anti-Patterns to Avoid

❌ **Don't**:
- Bypass settings validation
- Create temporary files in project directory (use `/tmp`)
- Add TODO comments without a plan to fix
- Copy-paste code instead of refactoring common patterns
- Skip tests "temporarily"
- Commit commented-out code
- Mix UI logic with business logic
- Hard-code configuration values

✅ **Do**:
- Use utilities like `PromptBuilder` for common patterns
- Keep functions focused and single-purpose
- Write tests before fixing bugs
- Refactor when you see duplication
- Follow existing patterns in the codebase
- Document non-obvious decisions
- Validate inputs at service boundaries
- Use type hints for complex functions
