---
applyTo: "**/test_*.py"
---

## Test File Requirements

When writing or modifying test files in this project, follow these guidelines:

### Test Structure
1. **File Organization**: Test files should mirror the structure of the code they test
   - `tests/unit/test_<module>.py` for unit tests
   - `tests/integration/test_<feature>.py` for integration tests
   - `tests/smoke/test_<component>.py` for smoke tests
   - `tests/e2e/test_<workflow>.py` for end-to-end tests

2. **Naming Conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*` (e.g., `TestStoryService`)
   - Test functions: `test_*` (e.g., `test_create_project`)

### Testing Standards

1. **Coverage Requirements**:
   - Core modules must maintain **100% test coverage**
   - Core modules include: `agents/`, `services/`, `workflows/`, `memory/`, `utils/`, `settings.py`
   - UI components (`ui/`) are excluded from coverage requirements

2. **Test Independence**:
   - Each test should be independent and not rely on other tests
   - Use pytest fixtures for test setup (shared fixtures in `tests/conftest.py`)
   - Clean up resources in teardown/fixtures

3. **Mocking Guidelines**:
   - **Always mock Ollama API calls** - tests should not require a running Ollama instance
   - Mock external dependencies (file system operations, network calls, etc.)
   - Use `unittest.mock` for mocking (`patch`, `MagicMock`)
   - Mock the Ollama client: `patch("agents.base.ollama.Client")`

4. **Fixtures**:
   - Place shared fixtures in `tests/conftest.py`
   - Use fixture scopes appropriately (`function`, `class`, `module`, `session`)
   - Document complex fixtures with docstrings

5. **Test Quality**:
   - Test both happy paths and edge cases
   - Test error conditions and exceptions
   - Use descriptive test names that explain what is being tested
   - Include assertions that verify the expected behavior
   - Avoid testing implementation details - focus on behavior

6. **Async Tests**:
   - Use `async def` for async test functions
   - pytest-asyncio is configured with `asyncio_mode = "auto"`
   - Mark async tests with `@pytest.mark.asyncio` if needed

### Example Test Pattern

```python
import pytest
from unittest.mock import patch, MagicMock
from services.story_service import StoryService

class TestStoryService:
    """Tests for StoryService."""

    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama API calls."""
        with patch("agents.base.ollama.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.generate.return_value = {"response": "test"}
            yield mock_instance

    def test_create_story_success(self, mock_ollama, tmp_path):
        """Test successful story creation."""
        # Arrange
        settings = Settings()
        service = StoryService(settings, tmp_path)

        # Act
        result = service.create_story("Test Story")

        # Assert
        assert result is not None
        assert result.title == "Test Story"
        mock_ollama.generate.assert_called()

    def test_create_story_error(self, mock_ollama):
        """Test story creation with API error."""
        # Arrange
        mock_ollama.generate.side_effect = Exception("API Error")

        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            service.create_story("Test Story")
```

### Common Test Patterns

1. **Testing Services**: Mock the Ollama client and external dependencies
2. **Testing Agents**: Mock the base agent's LLM calls and verify prompts
3. **Testing Memory/State**: Use in-memory databases or temp directories
4. **Testing Workflows**: Mock agent responses and verify orchestration
5. **Testing UI Components**: Use NiceGUI User fixture for component testing

### Running Tests

- All tests: `pytest`
- Specific test file: `pytest tests/unit/test_story_service.py`
- With coverage: `pytest --cov=. --cov-report=term`
- CI command: `make test-ci` (enforces 100% coverage on core modules)
