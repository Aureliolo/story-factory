# Contributing to Story Factory

Thank you for your interest in contributing! This is a hobby project, but contributions are very welcome.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Code of Conduct

Be respectful, constructive, and kind. This is a learning project - mistakes are okay, and all skill levels are welcome.

## Getting Started

1. **Check existing issues**: Look for issues labeled `good first issue` or `help wanted`
2. **Read the documentation**: Familiarize yourself with [README.md](README.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Set up development environment**: Follow the [Development Setup](#development-setup) section

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- **Search existing issues** to avoid duplicates
- **Verify the bug** with the latest version
- **Gather information**: OS, Python version, GPU, error logs

Create an issue with:
- **Clear title**: Describe the problem concisely
- **Steps to reproduce**: Exact steps to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**:
  - OS and version
  - Python version (`python --version`)
  - Ollama version (`ollama --version`)
  - GPU info (`nvidia-smi`)
- **Error logs**: From `logs/story_factory.log`
- **Screenshots**: If UI-related

### Suggesting Features

Feature requests are welcome! Include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought of
- **Implementation ideas**: If you have technical suggestions

### Contributing Code

Areas that need help:
- **Bug fixes**: Check issues labeled `bug`
- **Documentation**: Improve guides, add examples
- **Tests**: Increase coverage, add edge cases
- **UI improvements**: Better UX, accessibility
- **Performance**: Optimization, caching
- **New features**: After discussing in issues/discussions

## Development Setup

### Prerequisites

- Python 3.12+ (3.14+ recommended)
- Git
- Ollama installed and running
- At least one model pulled (e.g., `huihui_ai/dolphin3-abliterated:8b`)

### Setup Steps

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/story-factory.git
   cd story-factory
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

4. **Install dependencies**:
   ```bash
   pip install -e ".[all]"
   ```

5. **Copy settings**:
   ```bash
   cp settings.example.json settings.json
   ```

6. **Run tests** to verify setup:
   ```bash
   pytest
   ```

7. **Create a branch**:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/bug-description
   ```

## Coding Standards

### Python Style

- **Formatter**: Ruff (replaces Black)
- **Linter**: Ruff (replaces Flake8, isort, etc.)
- **Type hints**: Use where appropriate
- **Line length**: 100 characters
- **Python version**: 3.12+ (3.14+ recommended)

### Code Organization

Follow the existing architecture:
- **Services layer**: Business logic, no UI imports
- **UI layer**: NiceGUI components, calls services only
- **Agents**: AI interactions, extend BaseAgent
- **Utils**: Helper functions, pure Python

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Code Quality

Before committing:

1. **Format code**:
   ```bash
   ruff format .
   ```

2. **Lint code**:
   ```bash
   ruff check .
   ```
   Fix any issues:
   ```bash
   ruff check --fix .
   ```

3. **Type check** (optional but recommended):
   ```bash
   mypy .
   ```

4. **Run tests**:
   ```bash
   pytest
   ```

### Imports

Group imports in this order:
1. Standard library
2. Third-party packages
3. Local modules

```python
import logging
from pathlib import Path

from pydantic import BaseModel
from nicegui import ui

from services.story_service import StoryService
from utils.logging_config import setup_logging
```

### Docstrings

Use Google-style docstrings for classes and complex functions:

```python
def process_story(story_state: StoryState, options: dict) -> str:
    """Process a story state and generate output.

    Args:
        story_state: The current story state.
        options: Processing options.

    Returns:
        The processed story content.

    Raises:
        ValueError: If story_state is invalid.
    """
    pass
```

### Error Handling

- Use specific exceptions, not bare `except:`
- Log errors with appropriate level
- Provide helpful error messages
- Use custom exceptions from `utils/exceptions.py`

```python
import logging
from utils.exceptions import LLMError

logger = logging.getLogger(__name__)

try:
    result = generate_content()
except ConnectionError as e:
    logger.error(f"Failed to connect to Ollama: {e}")
    raise LLMError(f"Connection failed: {e}") from e
```

### Logging

Use the logging module, not print statements:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed information for debugging")
logger.info("General informational messages")
logger.warning("Warning messages for unexpected situations")
logger.error("Error messages for failures")
```

## Testing Guidelines

### Test Coverage

- **Core modules** require **100% coverage**: `agents/`, `services/`, `services/orchestrator/`, `memory/`, `utils/`, `settings.py`
- **UI modules** excluded from coverage requirements (until NiceGUI component tests are added)

### Writing Tests

1. **Test file location**: Mirror source structure in `tests/`
   - `agents/writer.py` → `tests/unit/test_writer.py`

2. **Test naming**: `test_<what_is_being_tested>`
   ```python
   def test_create_story_success():
       """Test successful story creation."""
       pass

   def test_create_story_with_invalid_input():
       """Test story creation with invalid input."""
       pass
   ```

3. **Mock external dependencies**:
   ```python
   from unittest.mock import patch, MagicMock

   @patch("agents.base.ollama.Client")
   def test_agent_generation(mock_ollama):
       mock_ollama.return_value.generate.return_value = {"response": "test"}
       # ... test code
   ```

4. **Use fixtures** for common setup:
   ```python
   import pytest

   @pytest.fixture
   def sample_story_state():
       return StoryState(title="Test Story")

   def test_something(sample_story_state):
       # Use the fixture
       pass
   ```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/unit/test_story_service.py

# Specific test
pytest tests/unit/test_story_service.py::test_create_story

# With coverage
pytest --cov=. --cov-report=term

# Coverage report (HTML)
pytest --cov=. --cov-report=html
# Open htmlcov/index.html

# Fast tests only (smoke tests)
pytest tests/smoke/

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Structure

```python
class TestStoryService:
    """Tests for StoryService."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create a StoryService instance."""
        settings = Settings()
        return StoryService(settings, tmp_path)

    def test_create_story_success(self, service):
        """Test successful story creation."""
        # Arrange
        title = "Test Story"

        # Act
        result = service.create_story(title)

        # Assert
        assert result is not None
        assert result.title == title
```

## Submitting Changes

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run full test suite**:
   ```bash
   pytest --cov=. --cov-report=term
   ```
4. **Format and lint**:
   ```bash
   ruff format .
   ruff check .
   ```
5. **Check coverage**: Ensure core modules have 100%
6. **Manual testing**: Run the app and test your changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for custom story templates

- Add TemplateService for managing templates
- Add UI for creating templates from projects
- Add tests with 100% coverage

Closes #123
```

Format:
- **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Subject**: Brief description (50 chars or less)
- **Body**: Detailed explanation (optional)
- **Footer**: Issue references (optional)

### Pull Request Process

1. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

2. **Create Pull Request** on GitHub:
   - Clear title describing the change
   - Description of what and why
   - Link to related issues
   - Screenshots for UI changes

3. **PR template** (if applicable):
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring

   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)

   ## Screenshots (if applicable)
   [Add screenshots here]

   Closes #issue_number
   ```

4. **Code Review**:
   - Address reviewer feedback
   - Make requested changes
   - Push updates to the same branch

5. **After Approval**:
   - Squash commits if requested
   - Maintainer will merge

## Documentation

### What to Document

- **New features**: Update README.md
- **API changes**: Update docstrings and architecture docs
- **Configuration**: Update settings.example.json and docs
- **Breaking changes**: Clearly document migration path

### Documentation Files

- **README.md**: Overview, quick start, basic usage
- **docs/ARCHITECTURE.md**: System design, patterns
- **docs/MODELS.md**: Model recommendations
- **TROUBLESHOOTING.md**: Common issues and solutions
- **CONTRIBUTING.md**: This file
- **Docstrings**: In-code documentation

### Documentation Standards

- **Clear and concise**: Easy to understand
- **Examples**: Show, don't just tell
- **Up-to-date**: Keep in sync with code
- **Well-structured**: Use headers, lists, code blocks
- **Links**: Reference related docs

### Writing Examples

Show concrete examples:

```python
# Good: Shows actual usage
story_service.create_story(
    title="My Story",
    genre="Fantasy",
    length="novel"
)

# Bad: Abstract description
story_service.create_story(parameters)
```

## Project-Specific Guidelines

### Adding a New Agent

1. Create agent class in `agents/`
2. Extend `BaseAgent`
3. Implement required methods
4. Add tests with mocked Ollama
5. Update `StoryOrchestrator` if needed
6. Document in README.md

### Adding a New UI Page

1. Create page class in `ui/pages/`
2. Follow NiceGUI patterns
3. Use `AppState` for state management
4. Call services, not agents directly
5. Add navigation link in `ui/app.py`
6. Update UI documentation

### Adding a New Service

1. Create service in `services/`
2. Add to `ServiceContainer`
3. Inject dependencies via constructor
4. No UI imports in services
5. Add comprehensive tests
6. Update architecture docs

## Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For general questions
- **Documentation**: Check existing docs first

Thank you for contributing to Story Factory! 🎉
