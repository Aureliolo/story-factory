# NiceGUI Testing Strategy Plan

## Problem Statement

Current tests don't catch runtime NiceGUI errors like:
- `ui.html()` missing required `sanitize` parameter
- `<script>` tags in ui.html() being rejected
- Background task UI context loss
- Component rendering failures

Unit tests mock NiceGUI, smoke tests only check imports, and E2E tests require full browser setup.

## NiceGUI Testing Framework Overview

NiceGUI provides two testing approaches via pytest plugins:

### 1. User Fixture (Recommended - Fast)
- Lightweight Python-based simulation
- No browser required
- Unit test-like performance
- Tests actual component construction and interactions

```python
from nicegui.testing import User

async def test_page_loads(user: User):
    await user.open('/settings')
    await user.should_see('Ollama URL')
    user.find(ui.button).click()
```

### 2. Screen Fixture (Comprehensive - Slow)
- Actual headless browser (Selenium/ChromeDriver)
- Tests browser-specific behavior
- Full JavaScript execution
- Required for vis-network graph testing

```python
from nicegui.testing import Screen

def test_graph_renders(screen: Screen):
    screen.open('/world')
    screen.should_contain('graph-container')
```

## Proposed Test Structure

```
tests/
├── unit/               # Pure logic tests (mock NiceGUI)
├── smoke/              # Import/construction tests
├── integration/        # Service integration
├── component/          # NEW: NiceGUI User fixture tests
│   ├── conftest.py     # pytest plugin activation
│   ├── test_chat.py    # Chat component tests
│   ├── test_graph.py   # Graph component tests
│   ├── test_header.py  # Header tests
│   └── test_pages/
│       ├── test_write_page.py
│       ├── test_world_page.py
│       └── test_settings_page.py
└── e2e/                # Full browser tests (existing)
```

## Implementation Plan

### Phase 1: Setup pytest Configuration

Update `pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
main_file = main.py
addopts = -p nicegui.testing.user_plugin
```

Update `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "component: NiceGUI component tests using User fixture",
]
```

### Phase 2: Create Component Test Base

`tests/component/conftest.py`:
```python
import pytest

pytest_plugins = ['nicegui.testing.user_plugin']

@pytest.fixture
def test_services():
    """Create test service container with mocked Ollama."""
    from services import ServiceContainer
    from settings import Settings

    settings = Settings()
    settings.ollama_url = "http://mock:11434"
    return ServiceContainer(settings)
```

### Phase 3: Component Tests to Add

#### Critical (Would have caught our bugs):

1. **Graph Component Tests**
   - Test `ui.html()` is called with correct parameters
   - Test vis-network script loading
   - Test `sanitize=False` is used

2. **Chat Component Tests**
   - Test `add_message()` works in background task context
   - Test `show_typing()` toggles visibility
   - Test client context is captured on build

3. **Page Load Tests**
   - Test each page loads without errors
   - Test navigation between pages
   - Test project selector updates

#### Example Test Cases:

```python
# tests/component/test_graph.py
async def test_graph_component_builds(user: User, test_services):
    """Graph component builds without errors."""
    from ui.components.graph import GraphComponent

    @ui.page('/test-graph')
    def test_page():
        graph = GraphComponent(height=300)
        graph.build()

    await user.open('/test-graph')
    # If we get here without exception, build succeeded

async def test_graph_with_world_db(user: User, test_world_db):
    """Graph renders with world database."""
    from ui.components.graph import mini_graph

    @ui.page('/test-mini-graph')
    def test_page():
        mini_graph(test_world_db, height=200)

    await user.open('/test-mini-graph')
    await user.should_see('graph-container')
```

```python
# tests/component/test_pages/test_write_page.py
async def test_write_page_loads(user: User, test_services):
    """Write page loads with project selected."""
    await user.open('/')
    await user.should_see('No Project Selected')

async def test_fundamentals_tab(user: User, test_services, test_project):
    """Fundamentals tab shows interview section."""
    # Setup project in state
    await user.open('/')
    await user.should_see('Interview')
    await user.should_see('World Overview')
```

### Phase 4: CI Integration

Add component tests to CI pipeline:
```yaml
- name: Run component tests
  run: pytest tests/component/ -v --timeout=30
```

### Phase 5: Screen Tests for JavaScript

For features requiring actual browser (vis-network):
```python
# tests/e2e/test_graph_browser.py
def test_vis_network_initializes(screen: Screen):
    """vis-network graph initializes in browser."""
    screen.open('/world')
    screen.wait(1)  # Wait for JS

    # Check vis-network is loaded
    result = screen.selenium.execute_script(
        "return typeof vis !== 'undefined'"
    )
    assert result is True
```

## Test Priority Matrix

| Test Type | Speed | Catches Runtime Errors | JS Testing | Priority |
|-----------|-------|----------------------|------------|----------|
| User Fixture | Fast | Yes | No | HIGH |
| Screen Fixture | Slow | Yes | Yes | MEDIUM |
| Unit (mocked) | Fast | No | No | Keep existing |

## Effort Estimate

1. **Setup pytest config**: 30 min
2. **Create conftest.py**: 30 min
3. **Write 10 core component tests**: 2-3 hours
4. **Add to CI**: 15 min

## Expected Benefits

1. **Catch construction errors** - Like `ui.html()` parameter issues
2. **Catch context errors** - Background task UI updates
3. **Fast feedback** - User fixture runs in seconds
4. **No browser deps for most tests** - Simpler CI

## References

- [NiceGUI Testing Documentation](https://nicegui.io/documentation/section_testing)
- [NiceGUI Tests README](https://github.com/zauberzeug/nicegui/blob/main/tests/README.md)
- [User Fixture Examples](https://nicegui.io/documentation/user)
