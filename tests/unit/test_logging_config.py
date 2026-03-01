"""Tests for utils/logging_config.py."""

import logging

import pytest

from src.utils.logging_config import (
    ContextFilter,
    _context_filter,
    _suppress_noisy_loggers,
    log_context,
    log_performance,
    reset_logger_suppression,
    set_log_level,
    setup_logging,
)


class TestContextFilter:
    """Tests for ContextFilter class."""

    def test_filter_with_correlation_id(self):
        """Test filter adds correlation ID when set."""
        filter_instance = ContextFilter()
        filter_instance.correlation_id = "test-123"

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.correlation_id == "test-123"  # type: ignore[attr-defined]

    def test_filter_without_correlation_id(self):
        """Test filter uses dash when no correlation ID set."""
        filter_instance = ContextFilter()
        filter_instance.correlation_id = None

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.correlation_id == "-"  # type: ignore[attr-defined]


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default settings."""
        # Save original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            setup_logging(level="DEBUG", log_file=None)  # No file to avoid creating files

            assert root_logger.level == logging.DEBUG
            # Should have at least console handler
            assert len(root_logger.handlers) >= 1
        finally:
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

    def test_setup_logging_with_file(self, tmp_path):
        """Test setup_logging with file handler."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            log_file = tmp_path / "test.log"
            setup_logging(level="INFO", log_file=str(log_file))

            # Should have console and file handlers
            assert len(root_logger.handlers) >= 2
            assert log_file.exists()
        finally:
            # Close file handlers to release the file
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

    def test_setup_logging_no_file(self):
        """Test setup_logging with no file logging."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            setup_logging(level="WARNING", log_file=None)

            assert root_logger.level == logging.WARNING
            # Should have only console handler
            assert len(root_logger.handlers) == 1
        finally:
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

    def test_setup_logging_default_file(self, tmp_path, monkeypatch):
        """Test setup_logging uses default log file when log_file='default'."""
        from src.utils import logging_config

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        # Monkeypatch the default log file location
        default_log = tmp_path / "default.log"
        monkeypatch.setattr(logging_config, "DEFAULT_LOG_FILE", default_log)

        try:
            setup_logging(level="INFO", log_file="default")

            # Should have created log file at default location
            assert default_log.exists()
        finally:
            # Close file handlers
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

    def test_setup_logging_removes_existing_handlers(self):
        """Test setup_logging removes existing handlers."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            # Add some dummy handlers
            dummy_handler = logging.StreamHandler()
            root_logger.addHandler(dummy_handler)
            handler_count_before = len(root_logger.handlers)

            setup_logging(level="INFO", log_file=None)

            # Should have removed all old handlers and added new ones
            assert handler_count_before > 0
            # All handlers should have our context filter
            for handler in root_logger.handlers:
                filters = handler.filters
                assert any(isinstance(f, ContextFilter) for f in filters)
        finally:
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

    def test_setup_logging_invalid_level_raises_value_error(self):
        """setup_logging should raise ValueError for invalid level names."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID_LEVEL", log_file=None)


class TestLogContext:
    """Tests for log_context context manager."""

    def test_log_context_with_provided_id(self):
        """Test log_context with provided correlation ID."""
        original_id = _context_filter.correlation_id

        try:
            with log_context("my-custom-id") as ctx_id:
                assert ctx_id == "my-custom-id"
                assert _context_filter.correlation_id == "my-custom-id"

            # Should restore original after context
            assert _context_filter.correlation_id == original_id
        finally:
            _context_filter.correlation_id = original_id

    def test_log_context_generates_id(self):
        """Test log_context generates UUID when not provided."""
        original_id = _context_filter.correlation_id

        try:
            with log_context() as ctx_id:
                assert ctx_id is not None
                assert len(ctx_id) == 8  # UUID[:8]
                assert _context_filter.correlation_id == ctx_id

            # Should restore original after context
            assert _context_filter.correlation_id == original_id
        finally:
            _context_filter.correlation_id = original_id

    def test_log_context_restores_on_exception(self):
        """Test log_context restores original ID on exception."""
        original_id = _context_filter.correlation_id
        _context_filter.correlation_id = "original"

        try:
            with pytest.raises(ValueError):
                with log_context("exception-id"):
                    assert _context_filter.correlation_id == "exception-id"
                    raise ValueError("Test error")

            # Should restore original after exception
            assert _context_filter.correlation_id == "original"
        finally:
            _context_filter.correlation_id = original_id


class TestLogPerformance:
    """Tests for log_performance context manager."""

    def test_log_performance_success(self, caplog):
        """Test log_performance logs success."""
        logger = logging.getLogger("test_perf")

        with caplog.at_level(logging.INFO):
            with log_performance(logger, "test_operation"):
                pass  # Simulate work

        assert "test_operation: Starting" in caplog.text
        assert "test_operation: Completed" in caplog.text

    def test_log_performance_failure(self, caplog):
        """Test log_performance logs failure on exception."""
        logger = logging.getLogger("test_perf")

        with caplog.at_level(logging.INFO):
            with pytest.raises(RuntimeError):
                with log_performance(logger, "failing_operation"):
                    raise RuntimeError("Something went wrong")

        assert "failing_operation: Starting" in caplog.text
        assert "failing_operation: Failed" in caplog.text
        assert "Something went wrong" in caplog.text

    def test_log_performance_includes_duration(self, caplog):
        """Test log_performance includes duration."""
        import time

        logger = logging.getLogger("test_perf")

        with caplog.at_level(logging.INFO):
            with log_performance(logger, "timed_op"):
                time.sleep(0.01)  # Small delay to ensure measurable time

        # Check duration is logged
        assert "Completed in" in caplog.text


class TestSetLogLevel:
    """Tests for set_log_level() runtime level change."""

    @pytest.fixture(autouse=True)
    def _managed_logging(self):
        """Save and restore root logger state around each test."""
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers[:]
        original_handler_levels = {h: h.level for h in original_handlers}
        try:
            yield
        finally:
            root_logger.handlers = original_handlers
            for handler, level in original_handler_levels.items():
                handler.setLevel(level)
            root_logger.setLevel(original_level)

    def test_set_log_level_changes_root_logger(self):
        """set_log_level should change the root logger level."""
        root_logger = logging.getLogger()
        setup_logging(level="INFO", log_file=None)
        assert root_logger.level == logging.INFO

        set_log_level("DEBUG")
        assert root_logger.level == logging.DEBUG

        set_log_level("WARNING")
        assert root_logger.level == logging.WARNING

    def test_set_log_level_changes_handler_levels(self):
        """set_log_level should update all handler levels."""
        root_logger = logging.getLogger()
        setup_logging(level="INFO", log_file=None)
        set_log_level("DEBUG")

        for handler in root_logger.handlers:
            assert handler.level == logging.DEBUG

    def test_set_log_level_suppresses_third_party(self):
        """Third-party loggers should stay at WARNING after set_log_level('DEBUG')."""
        setup_logging(level="INFO", log_file=None)
        set_log_level("DEBUG")

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("nicegui").level == logging.WARNING

    def test_set_log_level_logs_change(self, caplog):
        """set_log_level should log the level change."""
        # Don't call setup_logging() here — it removes all root handlers
        # including pytest's caplog handler, preventing log capture.
        with caplog.at_level(logging.INFO):
            set_log_level("WARNING")

        assert "Log level changed to WARNING" in caplog.text

    def test_set_log_level_invalid_raises_value_error(self):
        """set_log_level should raise ValueError for invalid level names."""
        setup_logging(level="INFO", log_file=None)
        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level("INVALID_LEVEL")

    def test_set_log_level_same_level_noop(self, caplog):
        """set_log_level should be a no-op when the level is already set."""
        # Don't call setup_logging() — it removes caplog's handler.
        # Instead, manually set root logger and handler levels.
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        for h in root_logger.handlers:
            h.setLevel(logging.WARNING)

        # Use logger-specific capture to avoid caplog.at_level altering the
        # root logger level, which would defeat the "already at WARNING" check.
        caplog.set_level(logging.DEBUG, logger="src.utils.logging_config")
        set_log_level("WARNING")

        # Level should still be WARNING (no change)
        assert root_logger.level == logging.WARNING
        # The "Log level changed" info message must NOT appear
        assert "Log level changed" not in caplog.text
        # The debug trace for the no-op should appear
        assert "already set to" in caplog.text
        # Third-party loggers must remain suppressed even in the no-op path
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("nicegui").level == logging.WARNING


class TestSuppressNoisyLoggersIdempotent:
    """Tests for _suppress_noisy_loggers idempotency (L4)."""

    @pytest.fixture(autouse=True)
    def _reset_suppression(self):
        """Reset the suppression flag before and after each test."""
        reset_logger_suppression()
        yield
        reset_logger_suppression()

    def test_suppress_noisy_loggers_always_reapplies_but_logs_once(self, caplog):
        """_suppress_noisy_loggers always re-applies WARNING levels (safety net).

        The suppression is always applied so set_log_level() can fix levels
        reset by third-party code, but the debug log message only fires once.
        """
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level

        try:
            httpx_logger.setLevel(logging.DEBUG)

            # First call: should set to WARNING and log the debug message
            with caplog.at_level(logging.DEBUG, logger="src.utils.logging_config"):
                _suppress_noisy_loggers()
            assert httpx_logger.level == logging.WARNING
            first_messages = [r.message for r in caplog.records]
            assert any("Suppressing noisy" in m for m in first_messages)

            # Temporarily reset httpx to DEBUG
            httpx_logger.setLevel(logging.DEBUG)
            caplog.clear()

            # Second call: should re-apply WARNING but NOT log the message again
            with caplog.at_level(logging.DEBUG, logger="src.utils.logging_config"):
                _suppress_noisy_loggers()
            assert httpx_logger.level == logging.WARNING  # re-suppressed
            second_messages = [r.message for r in caplog.records]
            assert not any("Suppressing noisy" in m for m in second_messages)
        finally:
            httpx_logger.setLevel(original_level)
