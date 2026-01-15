# Testing Guide

This guide covers testing patterns, strategies, and best practices for the Story Factory project.

## Overview

Story Factory uses **pytest** as its testing framework with the following structure:

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests
│   ├── test_agents/     # (Future: agent tests)
│   ├── test_memory/     # Memory/database tests
│   ├── test_services/   # Service layer tests
│   ├── test_error_handling.py
│   ├── test_json_parser.py
│   ├── test_orchestrator.py
│   ├── test_prompt_builder.py
│   ├── test_settings.py
│   └── test_ui_features.py
└── integration/         # (Future: integration tests)
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_settings.py

# Run specific test class
pytest tests/unit/test_settings.py::TestSettings

# Run specific test
pytest tests/unit/test_settings.py::TestSettings::test_default_values

# Run tests matching pattern
pytest -k "test_settings"

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

### Coverage

```bash
# Run with coverage
pytest --cov=. --cov-report=term

# Generate HTML coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Show missing lines
pytest --cov=. --cov-report=term-missing
```

## Test Organization

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
def test_my_feature():
    """Should demonstrate specific behavior."""
    # Arrange - Set up test data and conditions
    input_data = create_test_data()
    expected_output = "expected result"
    
    # Act - Execute the code being tested
    result = my_function(input_data)
    
    # Assert - Verify the results
    assert result == expected_output
```

### Test Naming

- **Test files**: `test_<module_name>.py`
- **Test classes**: `Test<ClassName>`
- **Test functions**: `test_<what_it_tests>()`

Use descriptive docstrings:

```python
def test_create_project_with_custom_name():
    """Should create project with user-specified name."""
    # Test implementation
```

### Test Classes

Group related tests in classes:

```python
class TestSettings:
    """Tests for Settings class."""
    
    def test_default_values(self):
        """Should have sensible default values."""
        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
    
    def test_validate_raises_on_invalid_url(self):
        """Should raise ValueError for invalid Ollama URL."""
        settings = Settings(ollama_url="not-a-url")
        with pytest.raises(ValueError, match="Invalid URL"):
            settings.validate()
```

## Common Testing Patterns

### 1. Testing with Fixtures

Use fixtures for shared test data:

```python
# In conftest.py
@pytest.fixture
def sample_story_state():
    """Create a sample StoryState for testing."""
    return StoryState(
        id="test-123",
        brief=StoryBrief(
            premise="Test story",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novel",
            content_rating="none",
        ),
    )

# In test file
def test_my_feature(sample_story_state):
    """Should process story state correctly."""
    result = process_story(sample_story_state)
    assert result is not None
```

### 2. Testing Exceptions

```python
def test_raises_on_invalid_input():
    """Should raise ValueError for invalid input."""
    with pytest.raises(ValueError, match="Invalid input"):
        my_function(invalid_input)

def test_raises_specific_error():
    """Should raise custom exception."""
    with pytest.raises(CustomError) as exc_info:
        dangerous_function()
    
    # Check exception details
    assert "expected error message" in str(exc_info.value)
```

### 3. Testing with Temporary Files

```python
def test_file_operations(tmp_path):
    """Should create and read files correctly."""
    # tmp_path is a pytest fixture providing temporary directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    result = read_file(test_file)
    assert result == "test content"
```

### 4. Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("input_value,expected", [
    ("short_story", 1),
    ("novella", 7),
    ("novel", 20),
])
def test_chapter_count_for_length(input_value, expected):
    """Should return correct chapter count for story length."""
    result = get_chapter_count(input_value)
    assert result == expected
```

### 5. Mocking External Dependencies

#### Mocking Ollama API Calls

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

#### Mocking File System

```python
def test_save_without_filesystem(monkeypatch, tmp_path):
    """Test save logic without affecting real filesystem."""
    
    # Redirect saves to temporary directory
    monkeypatch.setattr("settings.STORIES_DIR", tmp_path)
    
    orchestrator = StoryOrchestrator()
    filepath = orchestrator.save_story()
    
    assert filepath.exists()
```

#### Mocking External Processes

```python
def test_ollama_list_models(monkeypatch):
    """Test model listing without Ollama installed."""
    
    def mock_run(*args, **kwargs):
        class MockResult:
            stdout = "model1:latest\nmodel2:7b\n"
            returncode = 0
        return MockResult()
    
    monkeypatch.setattr("subprocess.run", mock_run)
    
    models = get_installed_models()
    assert "model1:latest" in models
```

## Testing Different Layers

### Testing Utilities

Pure functions are easiest to test:

```python
# tests/unit/test_json_parser.py
def test_extract_json_from_code_block():
    """Should extract JSON from markdown code block."""
    text = '''Here is some JSON:
    ```json
    {"key": "value"}
    ```
    '''
    
    result = extract_json(text)
    assert result == {"key": "value"}
```

### Testing Settings

```python
# tests/unit/test_settings.py
def test_settings_validation():
    """Should validate settings constraints."""
    settings = Settings(context_size=999)  # Too small
    
    with pytest.raises(ValueError, match="context_size must be between"):
        settings.validate()
```

### Testing Services

```python
# tests/unit/test_services/test_project_service.py
def test_create_project(tmp_path, monkeypatch):
    """Should create new project successfully."""
    monkeypatch.setattr("settings.STORIES_DIR", tmp_path)
    
    service = ProjectService()
    state = service.create_project("Test Project")
    
    assert state.project_name == "Test Project"
    assert state.id is not None
```

### Testing Memory/Database

```python
# tests/unit/test_memory/test_world_database.py
def test_add_entity(tmp_path):
    """Should add entity to database."""
    db_path = tmp_path / "test.db"
    db = WorldDatabase(db_path)
    
    entity_id = db.add_entity(
        entity_type="character",
        name="Test Character",
        description="A test character"
    )
    
    entity = db.get_entity(entity_id)
    assert entity.name == "Test Character"
```

### Testing Agents

When testing agents, **always mock** the Ollama calls:

```python
def test_architect_creates_world(monkeypatch, sample_story_state):
    """Should create world description."""
    
    def mock_generate(self, prompt, context=None, temperature=None):
        return "A detailed fantasy world with magic and dragons."
    
    monkeypatch.setattr(BaseAgent, "generate", mock_generate)
    
    agent = ArchitectAgent()
    world = agent.create_world(sample_story_state)
    
    assert "fantasy world" in world.lower()
```

### Testing UI (Future)

UI testing will use NiceGUI's testing utilities:

```python
# Future pattern for UI tests
async def test_ui_component():
    """Should render component correctly."""
    from nicegui import ui
    from nicegui.testing import UserFixture
    
    # Create UI
    with ui.card():
        ui.label("Test Label")
    
    # Test interactions
    # (Implementation depends on NiceGUI testing capabilities)
```

## Testing Best Practices

### ✅ Do

- **Test one thing per test** - Keep tests focused
- **Use descriptive test names** - Name should describe expected behavior
- **Write tests before fixing bugs** - Reproduce bug, then fix
- **Test edge cases** - Empty inputs, None, maximum values, etc.
- **Mock external dependencies** - Don't rely on network, filesystem, etc.
- **Keep tests fast** - Unit tests should run in milliseconds
- **Test error paths** - Not just happy path
- **Use fixtures for common data** - Avoid duplication

### ❌ Don't

- **Don't test implementation details** - Test behavior, not internal structure
- **Don't write flaky tests** - Tests should be deterministic
- **Don't make tests depend on each other** - Each test should be independent
- **Don't skip assertion** - Every test should have at least one assert
- **Don't test framework code** - Trust pytest, Pydantic, etc.
- **Don't make real API calls** - Always mock external services
- **Don't commit commented-out tests** - Delete or fix them

## Test Coverage Goals

Current coverage: ~70% (aiming for 80%+)

### High Priority Coverage

- ✅ **Utils** (90%+ coverage) - Pure functions, easy to test
- ✅ **Settings** (90%+ coverage) - Critical configuration
- ⚠️ **Services** (70% coverage) - Business logic layer
- ⚠️ **Agents** (40% coverage) - Need more mocked tests
- ❌ **UI** (10% coverage) - Complex to test, lower priority
- ❌ **Workflows** (30% coverage) - Integration-heavy

### Coverage Gaps

Areas needing more tests:

1. **Agent error handling** - Test retry logic, connection failures
2. **Workflow orchestration** - Test agent coordination
3. **Database edge cases** - Test concurrent access, corrupted data
4. **Export formats** - Test EPUB, PDF generation
5. **UI state management** - Test state transitions

## Debugging Failing Tests

### 1. Read the Error Message

```bash
# Run failing test with verbose output
pytest tests/unit/test_my_feature.py::test_failing_test -vv
```

### 2. Use Print Debugging

```python
def test_my_feature():
    """Test that's failing."""
    result = my_function()
    print(f"DEBUG: result = {result}")  # Will show with -s flag
    assert result == expected
```

Run with: `pytest tests/unit/test_my_feature.py -s`

### 3. Use pytest's Debug Mode

```bash
# Drop into debugger on failure
pytest --pdb
```

### 4. Isolate the Problem

```python
def test_isolated():
    """Minimal reproduction of the bug."""
    # Simplest possible test case
    assert simple_function() == expected_value
```

### 5. Check Fixtures

```bash
# Show fixture setup/teardown
pytest --setup-show
```

## Continuous Integration

GitHub Actions runs tests on every push:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: pytest --cov=. --cov-report=xml
```

Tests must pass before PRs can be merged.

## Writing Test Documentation

Good test documentation helps future maintainers:

```python
class TestPromptBuilder:
    """Tests for the PromptBuilder class.
    
    PromptBuilder consolidates common prompt patterns used across agents,
    reducing code duplication and ensuring consistency.
    """
    
    def test_add_language_requirement(self):
        """Should add language enforcement section.
        
        This test verifies that the language requirement is properly
        formatted and includes the correct language name.
        """
        builder = PromptBuilder()
        result = builder.add_language_requirement("Spanish").build()
        
        assert "LANGUAGE: Spanish" in result
        assert "Write ALL content in Spanish" in result
```

## Test Data Management

### Creating Test Data

Use builder pattern for complex objects:

```python
def create_test_story_state(**overrides):
    """Create a StoryState with sensible defaults.
    
    Args:
        **overrides: Override any default values.
    
    Returns:
        Configured StoryState for testing.
    """
    defaults = {
        "id": "test-123",
        "project_name": "Test Project",
        "brief": StoryBrief(
            premise="Test premise",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novel",
            content_rating="none",
        ),
    }
    defaults.update(overrides)
    return StoryState(**defaults)
```

Usage:

```python
def test_with_custom_genre():
    """Test with different genre."""
    state = create_test_story_state(
        brief__genre="Sci-Fi"  # Would need to update builder
    )
```

### Test Data Files

For large test data, use JSON files:

```python
def test_with_data_file():
    """Test using external test data."""
    test_data_path = Path(__file__).parent / "test_data" / "sample.json"
    with open(test_data_path) as f:
        data = json.load(f)
    
    result = process_data(data)
    assert result["status"] == "success"
```

## Common Testing Challenges

### Challenge: Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test asynchronous function."""
    result = await my_async_function()
    assert result is not None
```

### Challenge: Testing Database Transactions

```python
def test_database_rollback(tmp_path):
    """Test that failed operations don't corrupt database."""
    db = WorldDatabase(tmp_path / "test.db")
    
    try:
        db.add_entity("character", "Test", "Desc")
        raise Exception("Simulated error")
    except Exception:
        pass
    
    # Database should still be usable
    entities = db.get_all_entities()
    assert isinstance(entities, list)
```

### Challenge: Testing Random Behavior

```python
def test_random_with_seed(monkeypatch):
    """Test function with random elements."""
    import random
    random.seed(42)  # Make test deterministic
    
    result = function_with_randomness()
    assert result in expected_range
```

## Quick Reference

### Pytest Markers

```python
@pytest.mark.skip("Reason for skipping")
def test_skip_this():
    pass

@pytest.mark.skipif(condition, reason="Only run on Linux")
def test_linux_only():
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_expected_to_fail():
    pass

@pytest.mark.parametrize("input,expected", [...])
def test_multiple_cases(input, expected):
    pass
```

### Assertions

```python
# Equality
assert value == expected
assert value != unexpected

# Membership
assert item in collection
assert item not in collection

# Truthiness
assert value
assert not value
assert value is None
assert value is not None

# Exceptions
with pytest.raises(ValueError):
    risky_function()

# Approximate equality (for floats)
assert value == pytest.approx(3.14, rel=0.01)
```

### Fixtures

```python
# Scopes: function (default), class, module, session
@pytest.fixture(scope="module")
def expensive_fixture():
    """Created once per module."""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()

# Autouse fixtures run automatically
@pytest.fixture(autouse=True)
def setup_logging():
    """Run before every test."""
    setup_test_logging()
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
