# Contributing to Story Factory

Thank you for your interest in contributing to Story Factory! This guide will help you get started.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Be respectful, inclusive, and constructive. We're building this together!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/story-factory.git
   cd story-factory
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Aureliolo/story-factory.git
   ```

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Ollama installed and running
- At least one model pulled (recommend `huihui_ai/qwen3-abliterated:8b` for testing)
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Copy example settings**:
   ```bash
   cp settings.example.json settings.json
   ```

4. **Run tests** to verify setup:
   ```bash
   pytest
   ```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

Key directories:
```
story-factory/
├── agents/          # AI agent implementations
├── workflows/       # Orchestration logic
├── memory/          # State management
├── utils/           # Utilities (validation, logging, metrics)
├── ui/              # Gradio web interface
├── tests/           # Test suite
├── output/          # Generated stories (gitignored)
└── logs/            # Log files (gitignored)
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-epub-export` - New feature
- `bugfix/fix-continuity-check` - Bug fix
- `docs/update-readme` - Documentation
- `refactor/improve-validation` - Code refactoring

### 2. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_validators.py

# Run with coverage
pytest --cov=. --cov-report=html

# Test in CLI mode
python main.py --cli

# Test in UI mode
python main.py
```

### 4. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable names

### Code Organization

- One class per file (except small related classes)
- Group imports: standard library, third-party, local
- Keep functions focused (single responsibility)
- Add docstrings to all public functions/classes

### Example Function

```python
def process_chapter(
    chapter: Chapter,
    story_state: StoryState,
    max_retries: int = 3
) -> tuple[str, bool]:
    """Process a chapter through the editing pipeline.
    
    Args:
        chapter: The chapter to process
        story_state: Current story state for context
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (processed_content, success)
        
    Raises:
        ValidationError: If chapter is invalid
        LLMError: If generation fails after retries
    """
    # Implementation...
```

### Docstring Format

Use Google-style docstrings:
```python
"""Brief description.

Longer description if needed.

Args:
    param1: Description
    param2: Description

Returns:
    Description of return value

Raises:
    ExceptionType: When this happens
"""
```

## Testing

### Test Structure

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

### Test Categories

```python
class TestValidators:
    """Tests for input validation."""
    
    def test_valid_input(self):
        """Should accept valid input."""
        assert validate_something("valid") is True
    
    def test_rejects_invalid_input(self):
        """Should reject invalid input."""
        with pytest.raises(ValidationError):
            validate_something("invalid")
```

### Writing Good Tests

1. **Arrange-Act-Assert** pattern:
   ```python
   def test_something():
       # Arrange
       data = create_test_data()
       
       # Act
       result = process(data)
       
       # Assert
       assert result == expected
   ```

2. **Test edge cases**: empty strings, None, large values, etc.
3. **Use fixtures** for common setup:
   ```python
   @pytest.fixture
   def sample_story():
       return StoryState(id="test-123", ...)
   ```

4. **Mock external dependencies**:
   ```python
   @patch('agents.base.ollama.Client')
   def test_agent(mock_client):
       mock_client.return_value.chat.return_value = {"message": {"content": "test"}}
       # Test...
   ```

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, methods
- Include type hints
- Document exceptions that can be raised
- Add inline comments for complex logic

### User Documentation

Update relevant documentation when adding features:
- `README.md` - User-facing features
- `ARCHITECTURE.md` - System design
- `TROUBLESHOOTING.md` - Common issues
- Docstrings - API documentation

### Documentation Style

- Use clear, concise language
- Include examples
- Keep formatting consistent
- Test code examples

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass
   ```bash
   pytest
   ```

2. **Check code style**: Follow PEP 8
   ```bash
   # Optional: use black for formatting
   pip install black
   black .
   ```

3. **Update documentation**: Reflect your changes

4. **Update CHANGELOG**: Add entry describing changes

5. **Rebase on main**: Ensure no conflicts
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub

3. **Fill out PR template**:
   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues
   - Screenshots for UI changes
   - Testing notes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Tested manually (describe how)

## Screenshots
(If applicable)

## Checklist
- [ ] Code follows project style
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** must pass (tests, etc.)
2. **Code review** by maintainer(s)
3. **Address feedback** if requested
4. **Approval** from maintainer
5. **Merge** by maintainer

## Types of Contributions

### Bug Fixes

- Include reproduction steps
- Add test case that fails before fix, passes after
- Reference issue number in commit

### New Features

- Discuss in an issue first for major features
- Keep features focused and modular
- Add comprehensive tests
- Update documentation

### Documentation

- Fix typos, improve clarity
- Add examples
- Keep formatting consistent
- Test code examples

### Performance Improvements

- Include benchmarks showing improvement
- Don't sacrifice readability without significant gain
- Add performance tests if applicable

## Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for design context
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Story Factory! 🎉
