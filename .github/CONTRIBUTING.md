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
   pip install -r requirements-dev.txt
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
black .

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
- **Formatting**: Black (automatic)
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

- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Use pytest fixtures for common setup
- Write descriptive test names
- Test both success and error cases

Example:
```python
def test_export_to_markdown_file(sample_story_state):
    """Should export story to markdown file."""
    orchestrator = StoryOrchestrator()
    orchestrator.story_state = sample_story_state

    result = orchestrator.export_story_to_file(format="markdown")
    assert Path(result).exists()
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
- Checks code formatting (black)
- Runs linting (ruff)
- Reports coverage to Codecov

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
