"""Tests for generation management functionality."""

import pytest

from src.services.story_service import GenerationCancelled
from src.ui.state import AppState


class TestAppStateGenerationControl:
    """Tests for AppState generation control methods."""

    @pytest.fixture
    def app_state(self):
        """Create AppState instance."""
        return AppState()

    def test_initial_generation_flags(self, app_state):
        """Test generation flags are initialized correctly."""
        assert app_state.generation_cancel_requested is False
        assert app_state.generation_pause_requested is False
        assert app_state.generation_is_paused is False
        assert app_state.generation_can_resume is False

    def test_request_cancel_generation(self, app_state):
        """Test requesting cancellation sets the flag."""
        app_state.request_cancel_generation()
        assert app_state.generation_cancel_requested is True

    def test_request_pause_generation(self, app_state):
        """Test requesting pause sets the flag."""
        app_state.request_pause_generation()
        assert app_state.generation_pause_requested is True

    def test_resume_generation(self, app_state):
        """Test resuming generation clears pause flags."""
        app_state.generation_is_paused = True
        app_state.generation_pause_requested = True

        app_state.resume_generation()

        assert app_state.generation_is_paused is False
        assert app_state.generation_pause_requested is False

    def test_reset_generation_flags(self, app_state):
        """Test resetting all generation flags."""
        # Set all flags
        app_state.generation_cancel_requested = True
        app_state.generation_pause_requested = True
        app_state.generation_is_paused = True
        app_state.generation_can_resume = True

        # Reset
        app_state.reset_generation_flags()

        # Verify all are reset
        assert app_state.generation_cancel_requested is False
        assert app_state.generation_pause_requested is False
        assert app_state.generation_is_paused is False
        assert app_state.generation_can_resume is False


class TestGenerationCancellation:
    """Tests for generation cancellation."""

    def test_generation_cancelled_exception(self):
        """Test GenerationCancelled exception can be raised."""
        with pytest.raises(GenerationCancelled) as exc_info:
            raise GenerationCancelled("Test cancellation")

        assert "Test cancellation" in str(exc_info.value)

    def test_generation_cancelled_is_exception(self):
        """Test GenerationCancelled is an Exception subclass."""
        exc = GenerationCancelled("Test")
        assert isinstance(exc, Exception)
