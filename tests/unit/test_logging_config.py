"""Tests for utils/logging_config.py."""

import logging

import pytest

from utils.logging_config import (
    ContextFilter,
    _context_filter,
    log_context,
    log_performance,
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
        from utils import logging_config

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
