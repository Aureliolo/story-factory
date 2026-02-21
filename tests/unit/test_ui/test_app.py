"""Tests for the StoryFactoryApp exception handler."""

import logging
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from src.ui.app import StoryFactoryApp


class TestExceptionHandler:
    """Tests for the global exception handler in StoryFactoryApp."""

    @pytest.fixture
    def handler(self) -> Callable[[Exception], None]:
        """Create a StoryFactoryApp and extract its exception handler.

        Returns:
            The handle_exception function registered via app.on_exception.
        """
        mock_services = MagicMock()
        mock_services.settings.last_project_id = None

        with (
            patch("src.ui.app.app") as mock_app,
            patch("src.ui.app.ui"),
        ):
            sfa = StoryFactoryApp(mock_services)
            sfa.build()

            mock_app.on_exception.assert_called_once()
            handler: Callable[[Exception], None] = mock_app.on_exception.call_args.args[0]
            return handler

    def test_parent_slot_error_logs_debug(self, handler, caplog):
        """RuntimeError with 'parent slot' is logged at DEBUG, not ERROR."""
        error = RuntimeError("The parent slot of this element has been deleted")

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            handler(error)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        debug_records = [
            r for r in caplog.records if r.levelno == logging.DEBUG and r.name == "src.ui.app"
        ]
        assert len(error_records) == 0
        assert any("teardown" in r.message.lower() for r in debug_records)

    def test_has_been_deleted_error_logs_debug(self, handler, caplog):
        """RuntimeError with 'has been deleted' is logged at DEBUG, not ERROR."""
        error = RuntimeError("Element has been deleted from the page")

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            handler(error)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        debug_records = [
            r for r in caplog.records if r.levelno == logging.DEBUG and r.name == "src.ui.app"
        ]
        assert len(error_records) == 0
        assert any("teardown" in r.message.lower() for r in debug_records)

    def test_other_runtime_error_logs_error(self, handler, caplog):
        """Non-teardown RuntimeError is logged at ERROR level."""
        error = RuntimeError("Something else went wrong")

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            handler(error)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1
        assert "Unhandled UI exception" in error_records[0].message

    def test_mixed_case_teardown_error_logs_debug(self, handler, caplog):
        """Case-insensitive matching: mixed-case teardown message is logged at DEBUG."""
        error = RuntimeError("The Parent Slot was removed during teardown")

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            handler(error)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        debug_records = [
            r for r in caplog.records if r.levelno == logging.DEBUG and r.name == "src.ui.app"
        ]
        assert len(error_records) == 0
        assert any("teardown" in r.message.lower() for r in debug_records)

    def test_non_runtime_error_logs_error(self, handler, caplog):
        """Non-RuntimeError exceptions are logged at ERROR level."""
        error = ValueError("Bad value in UI")

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            handler(error)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1
        assert "Unhandled UI exception" in error_records[0].message
