"""Tests for RuntimeError guards on build dialog progress callbacks."""

import logging
from unittest.mock import MagicMock, PropertyMock

from src.ui.components.build_dialog import safe_progress_update


class TestSafeProgressUpdate:
    """Tests for the safe_progress_update helper that guards against destroyed UI elements."""

    def test_swallows_runtime_error_from_destroyed_elements(self, caplog):
        """Test that safe_progress_update catches RuntimeError from destroyed UI elements.

        Simulates the scenario where a dialog is closed while a background build
        is still running. The function should catch RuntimeError and log a debug
        message without re-raising.
        """
        progress_label = MagicMock()
        type(progress_label).text = PropertyMock(
            side_effect=RuntimeError("The parent slot of the element has been deleted")
        )

        progress_bar = MagicMock()
        type(progress_bar).value = PropertyMock(
            side_effect=RuntimeError("The parent slot of the element has been deleted")
        )

        with caplog.at_level(logging.DEBUG, logger="src.ui.components.build_dialog"):
            # Should NOT raise
            safe_progress_update(progress_label, progress_bar, "Building...", 0.5)

        assert "Progress update skipped: UI element destroyed" in caplog.text

    def test_updates_live_elements_normally(self):
        """Test that safe_progress_update sets values on live UI elements."""
        progress_label = MagicMock()
        progress_bar = MagicMock()

        safe_progress_update(progress_label, progress_bar, "Building locations...", 0.5)

        assert progress_label.text == "Building locations..."
        assert progress_bar.value == 0.5

    def test_partial_failure_still_catches(self, caplog):
        """Test that if only the bar raises, the error is still caught."""
        progress_label = MagicMock()
        progress_bar = MagicMock()
        type(progress_bar).value = PropertyMock(side_effect=RuntimeError("Element destroyed"))

        with caplog.at_level(logging.DEBUG, logger="src.ui.components.build_dialog"):
            safe_progress_update(progress_label, progress_bar, "Step 3/10", 0.3)

        # Label was set before bar raised
        assert progress_label.text == "Step 3/10"
        assert "Progress update skipped: UI element destroyed" in caplog.text
