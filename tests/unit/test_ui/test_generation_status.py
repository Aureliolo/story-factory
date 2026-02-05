"""Tests for GenerationStatus component."""

import logging
from unittest.mock import MagicMock

import pytest

from src.ui.components.generation_status import GenerationStatus
from src.ui.state import AppState


@pytest.fixture
def status_component():
    """Create GenerationStatus with mocked UI elements."""
    state = AppState()
    status = GenerationStatus(state)
    # Mock the progress bar
    status._progress_bar = MagicMock()
    return status


class TestProgressValidation:
    """Tests for progress value validation in GenerationStatus."""

    def test_valid_progress_zero(self, status_component):
        """Test that 0.0 is accepted as valid progress."""
        event = MagicMock()
        event.message = "Starting"
        event.phase = None
        event.progress = 0.0
        event.chapter_number = None
        event.eta_seconds = None

        status_component.update_from_event(event)

        status_component._progress_bar.update.assert_called()

    def test_valid_progress_one(self, status_component):
        """Test that 1.0 is accepted as valid progress."""
        event = MagicMock()
        event.message = "Complete"
        event.phase = None
        event.progress = 1.0
        event.chapter_number = None
        event.eta_seconds = None

        status_component.update_from_event(event)

        status_component._progress_bar.update.assert_called()

    def test_valid_progress_mid_value(self, status_component):
        """Test that 0.5 is accepted as valid progress."""
        event = MagicMock()
        event.message = "In progress"
        event.phase = None
        event.progress = 0.5
        event.chapter_number = None
        event.eta_seconds = None

        status_component.update_from_event(event)

        status_component._progress_bar.update.assert_called()

    def test_invalid_progress_negative(self, status_component, caplog):
        """Test that negative progress values are rejected with warning."""
        event = MagicMock()
        event.message = "Error"
        event.phase = None
        event.progress = -0.5
        event.chapter_number = None
        event.eta_seconds = None

        with caplog.at_level(logging.WARNING):
            status_component.update_from_event(event)

        assert "Invalid progress value" in caplog.text
        # Progress bar should not be updated with invalid value
        status_component._progress_bar.update.assert_not_called()

    def test_invalid_progress_greater_than_one(self, status_component, caplog):
        """Test that progress > 1.0 is rejected with warning."""
        event = MagicMock()
        event.message = "Error"
        event.phase = None
        event.progress = 1.5
        event.chapter_number = None
        event.eta_seconds = None

        with caplog.at_level(logging.WARNING):
            status_component.update_from_event(event)

        assert "Invalid progress value" in caplog.text
        status_component._progress_bar.update.assert_not_called()

    def test_none_progress_no_update(self, status_component):
        """Test that None progress doesn't trigger update."""
        event = MagicMock()
        event.message = "Status"
        event.phase = None
        event.progress = None
        event.chapter_number = None
        event.eta_seconds = None

        status_component.update_from_event(event)

        # set_progress should not be called when progress is None
        status_component._progress_bar.update.assert_not_called()


class TestSetProgressClamping:
    """Tests for the set_progress method's clamping behavior."""

    def test_set_progress_clamps_negative_to_zero(self, status_component, caplog):
        """Test that set_progress clamps negative values to 0 with warning."""
        with caplog.at_level(logging.WARNING):
            status_component.set_progress(-0.5)

        assert status_component._progress_bar.value == 0.0
        assert "clamped" in caplog.text

    def test_set_progress_clamps_high_to_one(self, status_component, caplog):
        """Test that set_progress clamps values > 1 to 1 with warning."""
        with caplog.at_level(logging.WARNING):
            status_component.set_progress(2.0)

        assert status_component._progress_bar.value == 1.0
        assert "clamped" in caplog.text

    def test_set_progress_accepts_valid_value(self, status_component):
        """Test that set_progress accepts valid values without warning."""
        status_component.set_progress(0.75)

        assert status_component._progress_bar.value == 0.75
