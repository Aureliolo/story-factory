"""Tests for background task tracking in AppState."""

import logging
import threading
from collections.abc import Generator

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.ui.state import AppState
from src.utils.exceptions import BackgroundTaskActiveError


def _is_busy(state: AppState) -> bool:
    """Check if state is busy (wrapper to prevent mypy narrowing)."""
    return state.is_busy


def _make_project() -> StoryState:
    """Create a minimal StoryState for testing."""
    return StoryState(
        id="test-project",
        brief=StoryBrief(
            genre="fantasy",
            tone="epic",
            premise="A test story",
            target_length="novella",
            setting_time="Medieval era",
            setting_place="A kingdom",
            content_rating="none",
        ),
    )


@pytest.fixture
def world_db(tmp_path) -> Generator[WorldDatabase]:
    """Create a WorldDatabase that auto-closes after the test."""
    db = WorldDatabase(tmp_path / "test.db")
    yield db
    if not db._closed:
        db.close()


class TestBackgroundTaskCounter:
    """Tests for begin/end background task counter."""

    def test_begin_end_counter(self):
        """Test that begin increments and end decrements the counter."""
        state = AppState()
        assert not _is_busy(state)

        state.begin_background_task("task_a")
        assert _is_busy(state)

        state.end_background_task("task_a")
        assert not _is_busy(state)

    def test_is_busy_reflects_counter(self):
        """Test is_busy property reflects whether tasks are active."""
        state = AppState()
        assert not _is_busy(state)

        state.begin_background_task("build")
        assert _is_busy(state)

        state.end_background_task("build")
        assert not _is_busy(state)

    def test_multiple_concurrent_tasks(self):
        """Test counter handles overlapping tasks correctly."""
        state = AppState()

        state.begin_background_task("build")
        state.begin_background_task("generate")
        assert _is_busy(state)

        state.end_background_task("build")
        assert _is_busy(state)  # generate still running

        state.end_background_task("generate")
        assert not _is_busy(state)

    def test_end_without_begin_logs_warning_and_stays_at_zero(self, caplog):
        """Test that ending a task without begin logs a warning and does not go negative."""
        state = AppState()
        with caplog.at_level(logging.WARNING, logger="src.ui.state"):
            state.end_background_task("phantom_task")

        assert not _is_busy(state)
        assert "end_background_task called with no active tasks" in caplog.text

        # Counter should still work normally after a spurious end
        state.begin_background_task("real_task")
        assert _is_busy(state)
        state.end_background_task("real_task")
        assert not _is_busy(state)


class TestSetProjectGuard:
    """Tests for set_project raising when busy."""

    def test_set_project_raises_when_busy(self, world_db):
        """Test that set_project raises BackgroundTaskActiveError when tasks are active."""
        state = AppState()
        state.begin_background_task("build")

        project = _make_project()

        with pytest.raises(BackgroundTaskActiveError):
            state.set_project("proj-1", project, world_db)

        state.end_background_task("build")

    def test_set_project_succeeds_when_idle(self, world_db):
        """Test that set_project works normally when no tasks are active."""
        state = AppState()
        project = _make_project()

        state.set_project("proj-1", project, world_db)

        assert state.project_id == "proj-1"
        assert state.project is project
        assert state.world_db is world_db


class TestClearProjectGuard:
    """Tests for clear_project raising when busy."""

    def test_clear_project_raises_when_busy(self, world_db):
        """Test that clear_project raises BackgroundTaskActiveError when tasks are active."""
        state = AppState()
        project = _make_project()
        state.set_project("proj-1", project, world_db)

        state.begin_background_task("generation")

        with pytest.raises(BackgroundTaskActiveError):
            state.clear_project()

        # Project should still be set
        assert state.project_id == "proj-1"

        state.end_background_task("generation")
        state.clear_project()
        assert state.project_id is None

    def test_clear_project_succeeds_when_idle(self, world_db):
        """Test that clear_project works normally when no tasks are active."""
        state = AppState()
        project = _make_project()
        state.set_project("proj-1", project, world_db)
        assert state.project_id == "proj-1"

        state.clear_project()
        assert state.project_id is None


class TestThreadSafety:
    """Tests for thread safety of background task counter."""

    def test_thread_safety(self):
        """Test concurrent begin/end from multiple threads."""
        state = AppState()
        num_threads = 20
        iterations = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            """Begin and end tasks in a tight loop."""
            barrier.wait()
            for _ in range(iterations):
                state.begin_background_task("stress")
                state.end_background_task("stress")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all threads complete, counter should be back to zero
        assert not _is_busy(state)


class TestProjectListCache:
    """Tests for project list caching in AppState."""

    def test_cache_returns_cached_value_within_ttl(self):
        """Test that cached value is returned within the 2-second TTL."""
        state = AppState()
        call_count = 0

        def fetch_projects():
            """Fetch mock projects and track call count."""
            nonlocal call_count
            call_count += 1
            return [{"id": "1", "name": "Project 1"}]

        # First call should fetch
        result1 = state.get_cached_projects(fetch_projects)
        assert call_count == 1
        assert len(result1) == 1

        # Second call within TTL should return cached value
        result2 = state.get_cached_projects(fetch_projects)
        assert call_count == 1  # Still 1, not 2
        assert result2 is result1  # Same object

    def test_cache_invalidation(self):
        """Test that invalidate_project_cache clears the cache."""
        state = AppState()
        call_count = 0

        def fetch_projects():
            """Fetch mock projects with incrementing ID."""
            nonlocal call_count
            call_count += 1
            return [{"id": str(call_count)}]

        # First call
        state.get_cached_projects(fetch_projects)
        assert call_count == 1

        # Invalidate
        state.invalidate_project_cache()

        # Next call should fetch again
        result2 = state.get_cached_projects(fetch_projects)
        assert call_count == 2
        assert result2[0]["id"] == "2"

    def test_cache_returns_empty_list_correctly(self):
        """Test that empty project list is cached correctly."""
        state = AppState()
        call_count = 0

        def fetch_empty():
            """Fetch empty list and track call count."""
            nonlocal call_count
            call_count += 1
            return []

        result1 = state.get_cached_projects(fetch_empty)
        assert call_count == 1
        assert result1 == []

        # Empty list should still be cached
        state.get_cached_projects(fetch_empty)
        assert call_count == 1  # Still 1

    def test_cache_logs_on_hit(self, caplog):
        """Test that cache hit is logged at debug level."""
        state = AppState()

        def fetch():
            """Return mock project list."""
            return [{"id": "1"}]

        with caplog.at_level(logging.DEBUG, logger="src.ui.state"):
            state.get_cached_projects(fetch)
            state.get_cached_projects(fetch)

        assert "Returning cached project list" in caplog.text

    def test_cache_logs_on_refresh(self, caplog):
        """Test that cache refresh is logged at debug level."""
        state = AppState()

        def fetch():
            """Return mock project list with 2 items."""
            return [{"id": "1"}, {"id": "2"}]

        with caplog.at_level(logging.DEBUG, logger="src.ui.state"):
            state.get_cached_projects(fetch)

        assert "Refreshed project list cache with 2 projects" in caplog.text

    def test_cache_logs_on_invalidate(self, caplog):
        """Test that cache invalidation is logged at debug level."""
        state = AppState()
        state._project_list_cache = [{"id": "1"}]

        with caplog.at_level(logging.DEBUG, logger="src.ui.state"):
            state.invalidate_project_cache()

        assert "Project list cache invalidated" in caplog.text
