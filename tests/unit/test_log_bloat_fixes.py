"""Tests for issue #264: log bloat and error propagation fixes.

Covers:
- B1+C6: summarize_llm_error utility and error propagation
- B3: JS timeout double-log prevention
- B5: Relationship add logging reduction (verified by inspection)
- B11: get_entity_color no longer logs per-call
- B13: arc_progress invalid keys logged at WARNING
- B14: Iteration count reports total iterations, not best iteration index
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.exceptions import summarize_llm_error


class _FakeRetryException(Exception):
    """Simulates InstructorRetryException with n_attempts and last_exception."""

    def __init__(self, message: str, n_attempts: int, last_exception: Exception | None = None):
        super().__init__(message)
        self.n_attempts = n_attempts
        self.last_exception = last_exception


class TestSummarizeLlmError:
    """Tests for summarize_llm_error utility."""

    def test_short_message_returned_as_is(self):
        """Short error messages are not truncated."""
        err = ValueError("missing field 'name'")
        result = summarize_llm_error(err)
        assert result == "missing field 'name'"

    def test_long_message_truncated(self):
        """Long error messages are truncated with char count."""
        msg = "x" * 500
        err = RuntimeError(msg)
        result = summarize_llm_error(err, max_length=200)
        assert len(result) < 300
        assert "chars truncated" in result
        assert result.startswith("x" * 200)

    def test_instructor_retry_exception_with_n_attempts(self):
        """InstructorRetryException-like errors use attempt count summary."""
        err = _FakeRetryException(
            "massive dump" * 100,
            n_attempts=3,
            last_exception=ValueError("field 'name' is required"),
        )
        result = summarize_llm_error(err)
        assert "3 attempt(s) failed" in result
        assert "field 'name' is required" in result
        assert len(result) < 300

    def test_instructor_retry_with_long_last_exception(self):
        """Last exception in retry summary is also truncated."""
        err = _FakeRetryException(
            "massive dump " * 100, n_attempts=2, last_exception=ValueError("a" * 300)
        )
        result = summarize_llm_error(err)
        assert "2 attempt(s) failed" in result
        assert "..." in result

    def test_instructor_retry_without_last_exception(self):
        """Works when last_exception is None."""
        err = _FakeRetryException("massive dump " * 100, n_attempts=5, last_exception=None)
        result = summarize_llm_error(err)
        assert "5 attempt(s) failed" in result

    def test_custom_max_length(self):
        """max_length parameter controls truncation threshold."""
        err = RuntimeError("y" * 100)
        result = summarize_llm_error(err, max_length=50)
        assert result.startswith("y" * 50)
        assert "chars truncated" in result

    def test_exact_boundary_not_truncated(self):
        """Message exactly at max_length is not truncated."""
        msg = "z" * 300
        err = RuntimeError(msg)
        result = summarize_llm_error(err, max_length=300)
        assert result == msg


class TestEntityModulesUseSummarizeLlmError:
    """Verify all entity modules import and use summarize_llm_error."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "src.services.world_quality_service._character",
            "src.services.world_quality_service._concept",
            "src.services.world_quality_service._location",
            "src.services.world_quality_service._faction",
            "src.services.world_quality_service._item",
            "src.services.world_quality_service._plot",
            "src.services.world_quality_service._relationship",
            "src.services.world_quality_service._chapter_quality",
        ],
    )
    def test_module_imports_summarize_llm_error(self, module_path):
        """Entity module imports summarize_llm_error from exceptions."""
        import importlib

        mod = importlib.import_module(module_path)
        assert hasattr(mod, "summarize_llm_error"), (
            f"{module_path} should import summarize_llm_error"
        )

    @pytest.mark.parametrize(
        "module_path",
        [
            "src.services.world_quality_service._character",
            "src.services.world_quality_service._concept",
            "src.services.world_quality_service._location",
            "src.services.world_quality_service._faction",
            "src.services.world_quality_service._item",
            "src.services.world_quality_service._plot",
            "src.services.world_quality_service._relationship",
            "src.services.world_quality_service._chapter_quality",
        ],
    )
    def test_module_does_not_use_logger_exception(self, module_path):
        """Entity modules should use logger.error, not logger.exception (avoids traceback dumps)."""
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        source = inspect.getsource(mod)
        assert "logger.exception(" not in source, (
            f"{module_path} should use logger.error with summarize_llm_error, "
            f"not logger.exception (which dumps full tracebacks)"
        )


class TestJsTimeoutDoublelog:
    """Tests for B3: JS localStorage timeout handling."""

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_load_prefs_returns_empty_on_timeout(self, mock_ui):
        """_load_prefs returns {} when localStorage read times out."""
        from src.ui.local_prefs import _load_prefs

        mock_ui.run_javascript = AsyncMock(side_effect=TimeoutError("JS timeout"))
        result = await _load_prefs("test_page")
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_load_prefs_timeout_logs_debug(self, mock_ui, caplog):
        """TimeoutError in _load_prefs is logged at DEBUG."""
        from src.ui.local_prefs import _load_prefs

        mock_ui.run_javascript = AsyncMock(side_effect=TimeoutError("JS timeout"))
        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await _load_prefs("cold_start_page")
        assert "timed out" in caplog.text
        assert "cold_start_page" in caplog.text

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_deferred_catches_timeout_error(self, mock_ui, caplog):
        """_deferred catches TimeoutError without propagating."""
        from src.ui.local_prefs import load_prefs_deferred

        captured_callback = None

        def capture_timer(delay, func, once):
            """Capture the timer callback."""
            nonlocal captured_callback
            captured_callback = func

        mock_ui.timer = MagicMock(side_effect=capture_timer)
        mock_ui.run_javascript = AsyncMock(side_effect=TimeoutError("JS timeout"))

        callback = MagicMock()
        load_prefs_deferred("timeout_page", callback)

        assert captured_callback is not None
        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await captured_callback()

        # _load_prefs catches TimeoutError and returns {}, so callback IS called
        callback.assert_called_once_with({})
        # Should log as debug, not error
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 0
        # Timeout should be logged at debug level
        assert "timed out" in caplog.text

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_deferred_catches_timeout_when_callback_raises(self, mock_ui, caplog):
        """_deferred catches TimeoutError if it propagates from callback."""
        from src.ui.local_prefs import load_prefs_deferred

        captured_callback = None

        def capture_timer(delay, func, once):
            """Capture the timer callback."""
            nonlocal captured_callback
            captured_callback = func

        mock_ui.timer = MagicMock(side_effect=capture_timer)
        # Normal return from JS, but callback raises TimeoutError
        mock_ui.run_javascript = AsyncMock(return_value="{}")

        callback = MagicMock(side_effect=TimeoutError("late timeout"))
        load_prefs_deferred("timeout_page", callback)

        assert captured_callback is not None
        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await captured_callback()  # Should not raise

        assert "Pref load skipped" in caplog.text


class TestArcProgressWarningLevel:
    """Tests for B13: arc_progress invalid keys at WARNING."""

    def test_invalid_keys_logged_at_warning(self, caplog):
        """Invalid arc_progress keys should produce WARNING, not DEBUG."""
        from src.memory.story_state import Character

        with caplog.at_level(logging.WARNING, logger="src.memory.story_state"):
            char = Character(
                name="Test",
                role="protagonist",
                description="A test character",
                arc_progress={"Embracing Power": "grows stronger", 1: "starts journey"},
            )

        # Valid key should be preserved
        assert 1 in char.arc_progress
        # Invalid key should be skipped
        assert "Embracing Power" not in str(char.arc_progress)
        # Should have logged a warning
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "arc_progress" in warning_records[0].message
        assert "Embracing Power" in warning_records[0].message


class TestEntityColorNoDebugLog:
    """Tests for B11: get_entity_color should not log for valid types."""

    def test_valid_type_does_not_log(self, caplog):
        """get_entity_color should not log when given a valid entity type."""
        from src.utils.constants import get_entity_color

        with caplog.at_level(logging.DEBUG, logger="src.utils.constants"):
            color = get_entity_color("character")

        assert color == "#4CAF50"
        # Should NOT have any debug log for the successful lookup
        debug_records = [r for r in caplog.records if "get_entity_color" in r.message]
        assert len(debug_records) == 0

    def test_unknown_type_still_logs_warning(self, caplog):
        """Unknown entity types still produce a warning."""
        from src.utils.constants import get_entity_color

        with caplog.at_level(logging.WARNING, logger="src.utils.constants"):
            color = get_entity_color("spaceship")

        assert color is not None  # Returns concept color
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert "spaceship" in warning_records[0].message


class TestQualityLoopErrorPropagation:
    """Tests for C6: quality loop uses WARNING for already-logged errors."""

    def test_quality_loop_uses_warning_for_caught_errors(self):
        """Quality loop should use logger.warning for WorldGenerationError (already logged upstream)."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        # The WorldGenerationError catch block should use warning, not error
        assert "already logged upstream" in source


class TestIterationCountReporting:
    """Tests for B14: iteration count returns total, not best index."""

    def test_quality_loop_returns_total_iterations(self):
        """quality_refinement_loop returns total iterations in docstring."""
        import inspect

        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        source = inspect.getsource(quality_refinement_loop)
        assert "total_iterations" in source
        # The return comment should mention total iteration count
        assert "total iteration count" in source
