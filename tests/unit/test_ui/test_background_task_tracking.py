"""Tests for background task tracking in AppState."""

import threading

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.ui.state import AppState
from src.utils.exceptions import BackgroundTaskActiveError


def _is_busy(state: AppState) -> bool:
    """Check if state is busy (wrapper to prevent mypy narrowing)."""
    return state.is_busy


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

    def test_end_without_begin_clamps_to_zero(self):
        """Test that ending a task without begin does not go negative."""
        state = AppState()
        state.end_background_task("phantom_task")
        assert not _is_busy(state)

        # Counter should still work normally after a spurious end
        state.begin_background_task("real_task")
        assert _is_busy(state)
        state.end_background_task("real_task")
        assert not _is_busy(state)


class TestSetProjectGuard:
    """Tests for set_project raising when busy."""

    def test_set_project_raises_when_busy(self, tmp_path):
        """Test that set_project raises BackgroundTaskActiveError when tasks are active."""
        state = AppState()
        state.begin_background_task("build")

        db = WorldDatabase(tmp_path / "test.db")
        project = StoryState(
            id="test-project",
            brief=StoryBrief(
                genre="fantasy",
                tone="epic",
                premise="A hero's journey",
                target_length="novella",
                setting_time="Medieval era",
                setting_place="A magical kingdom",
                content_rating="none",
            ),
        )

        with pytest.raises(BackgroundTaskActiveError):
            state.set_project("proj-1", project, db)

        state.end_background_task("build")
        db.close()

    def test_set_project_succeeds_when_idle(self, tmp_path):
        """Test that set_project works normally when no tasks are active."""
        state = AppState()
        db = WorldDatabase(tmp_path / "test.db")
        project = StoryState(
            id="test-project",
            brief=StoryBrief(
                genre="sci-fi",
                tone="dark",
                premise="Space exploration",
                target_length="novel",
                setting_time="Far future",
                setting_place="Deep space",
                content_rating="mild",
            ),
        )

        state.set_project("proj-1", project, db)

        assert state.project_id == "proj-1"
        assert state.project is project
        assert state.world_db is db

        db.close()


class TestClearProjectGuard:
    """Tests for clear_project raising when busy."""

    def test_clear_project_raises_when_busy(self, tmp_path):
        """Test that clear_project raises BackgroundTaskActiveError when tasks are active."""
        state = AppState()
        db = WorldDatabase(tmp_path / "test.db")
        project = StoryState(
            id="test-project",
            brief=StoryBrief(
                genre="mystery",
                tone="noir",
                premise="A detective story",
                target_length="short_story",
                setting_time="1940s",
                setting_place="Los Angeles",
                content_rating="moderate",
            ),
        )
        state.set_project("proj-1", project, db)

        state.begin_background_task("generation")

        with pytest.raises(BackgroundTaskActiveError):
            state.clear_project()

        # Project should still be set
        assert state.project_id == "proj-1"

        state.end_background_task("generation")
        state.clear_project()
        assert state.project_id is None

    def test_clear_project_succeeds_when_idle(self, tmp_path):
        """Test that clear_project works normally when no tasks are active."""
        state = AppState()
        db = WorldDatabase(tmp_path / "test.db")
        project = StoryState(
            id="test-project",
            brief=StoryBrief(
                genre="romance",
                tone="light",
                premise="A love story",
                target_length="novella",
                setting_time="Present day",
                setting_place="Paris",
                content_rating="mild",
            ),
        )
        state.set_project("proj-1", project, db)
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
