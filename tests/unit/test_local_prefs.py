"""Tests for src.ui.local_prefs — generic localStorage persistence."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ui.local_prefs import (
    _INITIAL_DELAY,
    _JS_TIMEOUT,
    _MAX_RETRIES,
    _PREFIX,
    _RETRY_DELAY,
    _load_prefs,
    _storage_key,
    load_prefs_deferred,
    save_pref,
    save_prefs,
)


class TestStorageKey:
    """Tests for _storage_key helper."""

    def test_prefix_applied(self):
        """Key includes the sf_prefs_ prefix."""
        assert _storage_key("entity_browser") == f"{_PREFIX}entity_browser"

    def test_different_pages_produce_different_keys(self):
        """Distinct page keys produce distinct storage keys."""
        assert _storage_key("graph") != _storage_key("timeline")


class TestSavePref:
    """Tests for save_pref (single field)."""

    @patch("src.ui.local_prefs.ui")
    def test_calls_run_javascript(self, mock_ui):
        """save_pref emits a JavaScript snippet via ui.run_javascript."""
        save_pref("entity_browser", "sort_by", "name")
        mock_ui.run_javascript.assert_called_once()
        js = mock_ui.run_javascript.call_args[0][0]
        assert "sf_prefs_entity_browser" in js
        assert '"sort_by"' in js
        assert '"name"' in js

    @patch("src.ui.local_prefs.ui")
    def test_guards_against_non_object_stored_value(self, mock_ui):
        """JS resets to {} if stored value parses to non-object (array, number, null)."""
        save_pref("page", "key", "value")
        js = mock_ui.run_javascript.call_args[0][0]
        # Must check typeof and Array.isArray
        assert "typeof raw==='object'" in js
        assert "Array.isArray(raw)" in js

    @patch("src.ui.local_prefs.ui")
    def test_serialises_bool(self, mock_ui):
        """Boolean values are JSON-serialised correctly."""
        save_pref("page", "flag", True)
        js = mock_ui.run_javascript.call_args[0][0]
        assert "true" in js

    @patch("src.ui.local_prefs.ui")
    def test_serialises_list(self, mock_ui):
        """List values are JSON-serialised correctly."""
        save_pref("page", "types", ["a", "b"])
        js = mock_ui.run_javascript.call_args[0][0]
        assert '["a", "b"]' in js

    @patch("src.ui.local_prefs.ui")
    def test_serialises_int(self, mock_ui):
        """Integer values are JSON-serialised correctly."""
        save_pref("page", "min_quality", 7)
        js = mock_ui.run_javascript.call_args[0][0]
        assert "7" in js


class TestSavePrefs:
    """Tests for save_prefs (multiple fields at once)."""

    @patch("src.ui.local_prefs.ui")
    def test_calls_run_javascript_once(self, mock_ui):
        """save_prefs emits a single JS call for all fields."""
        save_prefs("page", {"a": 1, "b": "two"})
        mock_ui.run_javascript.assert_called_once()

    @patch("src.ui.local_prefs.ui")
    def test_contains_all_fields(self, mock_ui):
        """The emitted JS includes all provided field values."""
        save_prefs("page", {"x": True, "y": [1, 2]})
        js = mock_ui.run_javascript.call_args[0][0]
        assert "sf_prefs_page" in js
        # The JSON dump of the fields dict should appear in the JS
        assert '"x"' in js
        assert '"y"' in js

    @patch("src.ui.local_prefs.ui")
    def test_guards_against_non_object_stored_value(self, mock_ui):
        """JS resets to {} if stored value parses to non-object (array, number, null)."""
        save_prefs("page", {"a": 1})
        js = mock_ui.run_javascript.call_args[0][0]
        assert "typeof raw==='object'" in js
        assert "Array.isArray(raw)" in js


class TestLoadPrefs:
    """Tests for _load_prefs (async helper with retry logic)."""

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_returns_empty_dict_when_nothing_stored(self, mock_ui):
        """Returns {} when localStorage has no entry."""
        mock_ui.run_javascript = AsyncMock(return_value=None)
        result = await _load_prefs("missing_page")
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_returns_stored_dict(self, mock_ui):
        """Returns the parsed dict when valid JSON is stored."""
        stored = {"sort_by": "name", "descending": True}
        mock_ui.run_javascript = AsyncMock(return_value=json.dumps(stored))
        result = await _load_prefs("page")
        assert result == stored

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_returns_empty_dict_on_corrupt_json(self, mock_ui):
        """Returns {} when stored JSON is corrupt."""
        mock_ui.run_javascript = AsyncMock(return_value="not valid json{{{")
        result = await _load_prefs("page")
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_returns_empty_dict_when_stored_value_is_not_dict(self, mock_ui):
        """Returns {} when stored JSON is valid but not an object."""
        mock_ui.run_javascript = AsyncMock(return_value=json.dumps([1, 2, 3]))
        result = await _load_prefs("page")
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_returns_empty_dict_on_empty_string(self, mock_ui):
        """Returns {} when localStorage returns empty string."""
        mock_ui.run_javascript = AsyncMock(return_value="")
        result = await _load_prefs("page")
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_uses_json_serialised_key(self, mock_ui):
        """_load_prefs passes the key through JSON.dumps to prevent injection."""
        mock_ui.run_javascript = AsyncMock(return_value=None)
        await _load_prefs("test_page")
        js_call = mock_ui.run_javascript.call_args[0][0]
        # Key should be JSON-encoded (double-quoted), not single-quoted
        assert '"sf_prefs_test_page"' in js_call

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_passes_js_timeout(self, mock_ui):
        """Verify that the timeout parameter is forwarded to run_javascript."""
        mock_ui.run_javascript = AsyncMock(return_value=None)
        await _load_prefs("page")
        _, kwargs = mock_ui.run_javascript.call_args
        assert kwargs["timeout"] == _JS_TIMEOUT


class TestLoadPrefsRetry:
    """Tests for _load_prefs timeout retry logic."""

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_retries_on_timeout_then_succeeds(self, mock_ui, mock_sleep):
        """First attempt times out, second attempt succeeds."""
        prefs = {"theme": "dark"}
        mock_ui.run_javascript = AsyncMock(
            side_effect=[TimeoutError("JS timed out"), json.dumps(prefs)]
        )

        result = await _load_prefs("page1")

        assert result == prefs
        assert mock_ui.run_javascript.await_count == 2
        mock_sleep.assert_awaited_once_with(_RETRY_DELAY)

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_all_retries_exhausted(self, mock_ui, mock_sleep):
        """All attempts time out — returns empty dict."""
        mock_ui.run_javascript = AsyncMock(side_effect=TimeoutError("JS timed out"))

        result = await _load_prefs("page1")

        assert result == {}
        assert mock_ui.run_javascript.await_count == _MAX_RETRIES
        # Sleep is called between attempts, so (_MAX_RETRIES - 1) times
        assert mock_sleep.await_count == _MAX_RETRIES - 1

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_logs_debug_on_first_timeout(self, mock_ui, mock_sleep, caplog):
        """First timeout is logged at DEBUG, not WARNING."""
        prefs = {"ok": True}
        mock_ui.run_javascript = AsyncMock(
            side_effect=[TimeoutError("JS timed out"), json.dumps(prefs)]
        )

        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await _load_prefs("test_key")

        debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
        warning_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("timed out" in m.message and "attempt 1" in m.message for m in debug_messages)
        assert not any("timed out" in m.message for m in warning_messages)

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_logs_warning_on_final_failure(self, mock_ui, mock_sleep, caplog):
        """Final exhaustion is logged at WARNING."""
        mock_ui.run_javascript = AsyncMock(side_effect=TimeoutError("JS timed out"))

        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await _load_prefs("test_key")

        warning_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("preferences lost" in m.message for m in warning_messages)

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_returns_empty_on_timeout_then_empty_storage(self, mock_ui, mock_sleep):
        """First attempt times out, second returns None (no data stored)."""
        mock_ui.run_javascript = AsyncMock(side_effect=[TimeoutError("JS timed out"), None])

        result = await _load_prefs("page1")

        assert result == {}
        assert mock_ui.run_javascript.await_count == 2


class TestLoadPrefsDeferred:
    """Tests for load_prefs_deferred."""

    @patch("src.ui.local_prefs.ui")
    def test_schedules_timer_with_correct_delay(self, mock_ui):
        """load_prefs_deferred creates a one-shot ui.timer with _INITIAL_DELAY."""
        callback = MagicMock()
        load_prefs_deferred("page", callback)
        mock_ui.timer.assert_called_once()
        args, kwargs = mock_ui.timer.call_args
        assert args[0] == _INITIAL_DELAY
        assert kwargs.get("once") is True

    @patch("src.ui.local_prefs.ui")
    def test_callback_not_called_immediately(self, mock_ui):
        """The callback is not invoked synchronously."""
        callback = MagicMock()
        load_prefs_deferred("page", callback)
        callback.assert_not_called()


class TestDeferredRuntimeErrorHandling:
    """Tests for RuntimeError handling in deferred callbacks."""

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_deferred_catches_runtime_error_in_callback(self, mock_ui, caplog):
        """Verify _deferred actually catches RuntimeError from callback."""
        # Capture the timer callback
        captured_callback = None

        def capture_timer(delay, func, once):
            """Capture the timer callback for later invocation."""
            nonlocal captured_callback
            captured_callback = func

        mock_ui.timer = MagicMock(side_effect=capture_timer)
        mock_ui.run_javascript = AsyncMock(return_value="{}")

        callback = MagicMock(side_effect=RuntimeError("UI destroyed"))
        load_prefs_deferred("test_page", callback)

        # Invoke the captured _deferred function - should not raise
        assert captured_callback is not None, "Timer callback was not captured"
        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await captured_callback()

        assert "Pref load skipped" in caplog.text

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs.ui")
    async def test_deferred_catches_runtime_error_in_load_prefs(self, mock_ui, caplog):
        """Verify _deferred catches RuntimeError from _load_prefs itself."""
        # Capture the timer callback
        captured_callback = None

        def capture_timer(delay, func, once):
            """Capture the timer callback for later invocation."""
            nonlocal captured_callback
            captured_callback = func

        mock_ui.timer = MagicMock(side_effect=capture_timer)
        mock_ui.run_javascript = AsyncMock(side_effect=RuntimeError("Parent slot deleted"))

        callback = MagicMock()
        load_prefs_deferred("destroyed_page", callback)

        # Invoke the captured _deferred function - should not raise
        assert captured_callback is not None, "Timer callback was not captured"
        with caplog.at_level(logging.DEBUG, logger="src.ui.local_prefs"):
            await captured_callback()

        assert "Pref load skipped" in caplog.text
        callback.assert_not_called()  # Callback never reached due to RuntimeError

    @pytest.mark.asyncio
    @patch("src.ui.local_prefs._load_prefs", new_callable=AsyncMock)
    @patch("src.ui.local_prefs.ui")
    async def test_deferred_callback_receives_prefs(self, mock_ui, mock_load):
        """The _deferred coroutine calls callback with loaded prefs."""
        prefs = {"sort": "desc"}
        mock_load.return_value = prefs

        captured_callback = None

        def capture_timer(delay, func, once):
            """Capture the timer callback for later invocation."""
            nonlocal captured_callback
            captured_callback = func

        mock_ui.timer = MagicMock(side_effect=capture_timer)
        callback = MagicMock()

        load_prefs_deferred("page1", callback)

        assert captured_callback is not None
        await captured_callback()

        callback.assert_called_once_with(prefs)


class TestConstants:
    """Verify module constants have sane values."""

    def test_initial_delay_in_sane_range(self):
        """_INITIAL_DELAY is between 0.1s and 2s (enough for browser init)."""
        assert 0.1 < _INITIAL_DELAY <= 2.0

    def test_js_timeout_greater_than_default(self):
        """_JS_TIMEOUT exceeds the default NiceGUI 1s timeout."""
        assert _JS_TIMEOUT > 1.0

    def test_max_retries_at_least_two(self):
        """At least two attempts are made before giving up."""
        assert _MAX_RETRIES >= 2

    def test_retry_delay_is_positive(self):
        """_RETRY_DELAY is a positive number of seconds."""
        assert _RETRY_DELAY > 0
