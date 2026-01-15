# Contributing to Story Factory

Thank you for your interest in contributing to Story Factory! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aureliolo/story-factory.git
   cd story-factory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama and pull a model**
   ```bash
   # See README.md for Ollama installation
   ollama pull huihui_ai/qwen3-abliterated:8b
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term --cov-report=html

# Run specific test file
pytest tests/test_settings.py

# Or use make
make test
make test-cov
```

### Code Quality

We use automated code quality tools to maintain consistent code style:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Or use make
make format
make lint
make check  # Run both linting and tests
```

### Code Style

- **Line length**: 100 characters
- **Formatting**: Ruff (automatic)
- **Import sorting**: Ruff (automatic)
- **Linting**: Ruff
- **Type hints**: Encouraged but not required (we're gradually adding them)

### Project Structure

- `agents/`: AI agent implementations (Writer, Editor, etc.)
- `workflows/`: Agent coordination and orchestration
- `memory/`: Story state management
- `utils/`: Utility modules (JSON parsing, logging, etc.)
- `ui/`: NiceGUI web interface
- `tests/`: Test suite
- `settings.py`: Settings management and model registry

## Writing Tests

- Place tests in the `tests/unit/` directory
- Follow the naming convention: `test_*.py`
- Use pytest fixtures for common setup (see `tests/conftest.py`)
- Write descriptive test names and docstrings
- Test both success and error cases
- Use the Arrange-Act-Assert pattern

### Testing Patterns

**Mocking Ollama API calls**:
```python
def test_agent_without_ollama(monkeypatch):
    """Test agent logic without calling Ollama."""
    def mock_generate(self, prompt, context=None, temperature=None):
        return "Mocked LLM response"
    
    monkeypatch.setattr(BaseAgent, "generate", mock_generate)
    
    agent = WriterAgent()
    result = agent.write_chapter(story_state, chapter)
    assert "Mocked LLM response" in result
```

**Using fixtures**:
```python
def test_export_to_markdown_file(sample_story_state):
    """Should export story to markdown file."""
    orchestrator = StoryOrchestrator()
    orchestrator.story_state = sample_story_state
    
    result = orchestrator.export_story_to_file(format="markdown")
    assert Path(result).exists()
```

**Parametrized tests**:
```python
@pytest.mark.parametrize("input_value,expected", [
    ("short_story", 1),
    ("novella", 7),
    ("novel", 20),
])
def test_chapter_count(input_value, expected):
    """Should return correct chapter count for story length."""
    result = get_chapter_count(input_value)
    assert result == expected
```

## Architecture Patterns

### Agent Pattern

All agents inherit from `BaseAgent` and use `PromptBuilder` for constructing prompts:

```python
from utils.prompt_builder import PromptBuilder

class MyAgent(BaseAgent):
    def my_method(self, story_state: StoryState) -> str:
        # Validate brief exists
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        
        # Build prompt using fluent API
        builder = PromptBuilder()
        builder.add_language_requirement(brief.language)
        builder.add_brief_requirements(brief)
        builder.add_text("Additional instructions...")
        
        prompt = builder.build()
        return self.generate(prompt)
```

### Service Pattern

Services use dependency injection:

```python
class MyService:
    def __init__(self, settings: Settings | None = None):
        """Initialize service with settings."""
        self.settings = settings or Settings.load()
```

All services are initialized through `ServiceContainer`:
```python
services = ServiceContainer(settings)
services.project.list_projects()
services.story.start_interview(state)
```

## Logging

Use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing started")
logger.warning("Potential issue detected")
logger.error("Operation failed", exc_info=True)
```

For performance tracking:
```python
from utils.logging_config import log_performance

with log_performance(logger, "story_generation"):
    generate_story()  # Will log duration
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow code style guidelines

3. **Run tests and linting**
   ```bash
   make check
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## CI/CD

Our GitHub Actions workflow automatically:
- Runs tests with coverage
- Checks code formatting (ruff format)
- Runs linting (ruff check)

Make sure all checks pass before requesting review.

## Areas for Contribution

We welcome contributions in these areas:

- **New features**: Export formats, analytics, UI improvements
- **Agent improvements**: Better prompts, new agent types
- **Testing**: Increase test coverage, integration tests
- **Documentation**: Tutorials, examples, API docs
- **Bug fixes**: Check GitHub issues
- **Performance**: Optimization, caching, resource management

## Questions?

- Open an issue on GitHub
- Check existing issues and discussions
- Review the README and documentation

Thank you for contributing! 🎉
