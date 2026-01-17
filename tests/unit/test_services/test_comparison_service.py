"""Tests for comparison service."""

import pytest
from unittest.mock import MagicMock, patch

from services.comparison_service import ComparisonService, ComparisonResult, ComparisonRecord
from memory.story_state import StoryState, Chapter
from settings import Settings
from workflows.orchestrator import WorkflowEvent


@pytest.fixture
def settings():
    """Create test settings."""
    settings = Settings()
    settings.agent_models = {
        "writer": "test-model-1",
        "editor": "test-model-2",
    }
    return settings


@pytest.fixture
def comparison_service(settings):
    """Create comparison service."""
    return ComparisonService(settings)


@pytest.fixture
def story_state():
    """Create a test story state."""
    state = StoryState(id="test-story", project_name="Test Project")
    state.chapters = [
        Chapter(number=1, title="Chapter 1", outline="Test chapter outline"),
        Chapter(number=2, title="Chapter 2", outline="Another chapter"),
    ]
    return state


class TestComparisonService:
    """Test comparison service functionality."""

    def test_initialization(self, settings):
        """Test service initialization."""
        service = ComparisonService(settings)
        assert service.settings == settings
        assert len(service._comparison_history) == 0

    def test_validate_models_minimum(self, comparison_service, story_state):
        """Test that at least 2 models are required."""
        with pytest.raises(ValueError, match="At least 2 models required"):
            list(
                comparison_service.generate_chapter_comparison(
                    state=story_state,
                    chapter_num=1,
                    models=["model-1"],
                )
            )

    def test_validate_models_maximum(self, comparison_service, story_state):
        """Test that maximum 4 models are allowed."""
        with pytest.raises(ValueError, match="Maximum 4 models allowed"):
            list(
                comparison_service.generate_chapter_comparison(
                    state=story_state,
                    chapter_num=1,
                    models=["m1", "m2", "m3", "m4", "m5"],
                )
            )

    def test_validate_models_duplicates(self, comparison_service, story_state):
        """Test that duplicate models are rejected."""
        with pytest.raises(ValueError, match="Duplicate models"):
            list(
                comparison_service.generate_chapter_comparison(
                    state=story_state,
                    chapter_num=1,
                    models=["model-1", "model-1"],
                )
            )

    @patch("services.comparison_service.StoryOrchestrator")
    def test_generate_chapter_comparison(
        self, mock_orchestrator_class, comparison_service, story_state
    ):
        """Test generating chapter comparison."""
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock chapter writing events
        def write_chapter_side_effect(chapter_num):
            # Yield some events
            yield WorkflowEvent(
                event_type="agent_start",
                agent_name="Writer",
                message="Starting",
                chapter_number=chapter_num,
            )
            yield WorkflowEvent(
                event_type="agent_complete",
                agent_name="System",
                message="Complete",
                chapter_number=chapter_num,
            )

        mock_orchestrator.write_chapter.side_effect = write_chapter_side_effect

        # Update story state to have chapter content
        story_state.chapters[0].content = "Generated chapter content"

        # Generate comparison
        models = ["model-1", "model-2"]
        generator = comparison_service.generate_chapter_comparison(
            state=story_state,
            chapter_num=1,
            models=models,
        )

        # Consume all events
        events = list(generator)

        # Should have events for both models
        assert len(events) > 0

        # Check that we got progress events
        model_ids = {e["model_id"] for e in events if "model_id" in e}
        assert "model-1" in model_ids
        assert "model-2" in model_ids

        # Check that orchestrator was called
        assert mock_orchestrator.write_chapter.call_count == 2

    @patch("services.comparison_service.StoryOrchestrator")
    def test_generate_with_error_handling(
        self, mock_orchestrator_class, comparison_service, story_state
    ):
        """Test that errors during generation are captured."""
        # Mock orchestrator that raises an error
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        def write_chapter_error(chapter_num):
            raise RuntimeError("Test error")

        mock_orchestrator.write_chapter.side_effect = write_chapter_error

        # Generate comparison
        generator = comparison_service.generate_chapter_comparison(
            state=story_state,
            chapter_num=1,
            models=["model-1", "model-2"],
        )

        # Get the final record
        events = list(generator)
        record = events[-1] if events else None

        # Record should exist even with errors
        assert isinstance(record, ComparisonRecord)
        assert len(record.results) == 2

        # Results should have errors
        for result in record.results.values():
            assert result.error is not None

    def test_select_winner(self, comparison_service, story_state):
        """Test selecting a winner from comparison."""
        # Create a mock comparison record
        record = ComparisonRecord(
            id="test-comparison",
            timestamp=None,
            chapter_number=1,
            models=["model-1", "model-2"],
            results={
                "model-1": ComparisonResult(
                    model_id="model-1",
                    content="Content 1",
                    word_count=100,
                    generation_time=10.0,
                ),
                "model-2": ComparisonResult(
                    model_id="model-2",
                    content="Content 2",
                    word_count=120,
                    generation_time=12.0,
                ),
            },
        )
        comparison_service._comparison_history.append(record)

        # Select a winner
        comparison_service.select_winner(
            comparison_id="test-comparison",
            selected_model="model-1",
            user_notes="Better prose",
        )

        # Verify selection
        assert record.selected_model == "model-1"
        assert record.user_notes == "Better prose"

    def test_select_winner_invalid_comparison(self, comparison_service):
        """Test selecting winner with invalid comparison ID."""
        with pytest.raises(ValueError, match="Comparison record not found"):
            comparison_service.select_winner(
                comparison_id="invalid-id",
                selected_model="model-1",
            )

    def test_select_winner_invalid_model(self, comparison_service):
        """Test selecting winner with invalid model ID."""
        record = ComparisonRecord(
            id="test-comparison",
            timestamp=None,
            chapter_number=1,
            models=["model-1"],
            results={
                "model-1": ComparisonResult(
                    model_id="model-1",
                    content="Content",
                    word_count=100,
                    generation_time=10.0,
                )
            },
        )
        comparison_service._comparison_history.append(record)

        with pytest.raises(ValueError, match="not in comparison results"):
            comparison_service.select_winner(
                comparison_id="test-comparison",
                selected_model="invalid-model",
            )

    def test_get_comparison_history(self, comparison_service):
        """Test retrieving comparison history."""
        # Add some records
        for i in range(3):
            record = ComparisonRecord(
                id=f"test-{i}",
                timestamp=None,
                chapter_number=i + 1,
                models=["model-1", "model-2"],
                results={},
            )
            comparison_service._comparison_history.append(record)

        # Get history
        history = comparison_service.get_comparison_history()

        # Should be in reverse order (newest first)
        assert len(history) == 3
        assert history[0].id == "test-2"
        assert history[1].id == "test-1"
        assert history[2].id == "test-0"

    def test_get_comparison(self, comparison_service):
        """Test retrieving a specific comparison."""
        record = ComparisonRecord(
            id="test-comparison",
            timestamp=None,
            chapter_number=1,
            models=["model-1"],
            results={},
        )
        comparison_service._comparison_history.append(record)

        # Get the comparison
        found = comparison_service.get_comparison("test-comparison")
        assert found is not None
        assert found.id == "test-comparison"

        # Try to get non-existent
        not_found = comparison_service.get_comparison("invalid-id")
        assert not_found is None

    def test_get_model_win_rate(self, comparison_service):
        """Test calculating model win rate."""
        # Create comparisons with winners
        for i in range(10):
            record = ComparisonRecord(
                id=f"test-{i}",
                timestamp=None,
                chapter_number=i + 1,
                models=["model-1", "model-2"],
                results={
                    "model-1": ComparisonResult(
                        model_id="model-1",
                        content="",
                        word_count=100,
                        generation_time=10.0,
                    ),
                    "model-2": ComparisonResult(
                        model_id="model-2",
                        content="",
                        word_count=100,
                        generation_time=10.0,
                    ),
                },
            )
            # Model-1 wins 7 out of 10
            if i < 7:
                record.selected_model = "model-1"
            else:
                record.selected_model = "model-2"

            comparison_service._comparison_history.append(record)

        # Check win rates
        model1_rate = comparison_service.get_model_win_rate("model-1")
        assert model1_rate == 70.0

        model2_rate = comparison_service.get_model_win_rate("model-2")
        assert model2_rate == 30.0

        # Model not in any comparison
        model3_rate = comparison_service.get_model_win_rate("model-3")
        assert model3_rate == 0.0

    def test_clear_history(self, comparison_service):
        """Test clearing comparison history."""
        # Add some records
        for i in range(3):
            record = ComparisonRecord(
                id=f"test-{i}",
                timestamp=None,
                chapter_number=i + 1,
                models=["model-1"],
                results={},
            )
            comparison_service._comparison_history.append(record)

        assert len(comparison_service._comparison_history) == 3

        # Clear history
        comparison_service.clear_history()

        # Verify cleared
        assert len(comparison_service._comparison_history) == 0
        assert len(comparison_service.get_comparison_history()) == 0

    def test_cancellation_support(self, comparison_service, story_state):
        """Test that cancellation is supported."""
        cancel_flag = False

        def cancel_check():
            return cancel_flag

        with patch("services.comparison_service.StoryOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_orch.return_value = mock_instance

            # Simulate cancellation after first model
            call_count = [0]

            def write_chapter_side_effect(chapter_num):
                call_count[0] += 1
                if call_count[0] > 1:
                    nonlocal cancel_flag
                    cancel_flag = True
                yield WorkflowEvent(
                    event_type="agent_complete",
                    agent_name="System",
                    message="Complete",
                    chapter_number=chapter_num,
                )

            mock_instance.write_chapter.side_effect = write_chapter_side_effect

            # Generate with cancellation
            generator = comparison_service.generate_chapter_comparison(
                state=story_state,
                chapter_num=1,
                models=["model-1", "model-2", "model-3"],
                cancel_check=cancel_check,
            )

            events = list(generator)

            # Should have stopped early
            # Only model-1 should have completed
            assert len([e for e in events if e.get("completed")]) <= 2
