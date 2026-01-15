"""Unit tests for ScoringService."""

from unittest.mock import MagicMock

import pytest

from memory.mode_models import QualityScores
from services.scoring_service import ScoringService


class TestScoringService:
    """Tests for ScoringService."""

    @pytest.fixture
    def mock_mode_service(self) -> MagicMock:
        """Create a mock ModelModeService."""
        mock = MagicMock()
        mock.record_generation.return_value = 1
        mock.record_implicit_signal.return_value = None
        mock.judge_quality.return_value = QualityScores(prose_quality=8.0)
        mock.calculate_consistency_score.return_value = 9.0
        mock.update_quality_scores.return_value = None
        mock.update_performance_metrics.return_value = None
        return mock

    @pytest.fixture
    def service(self, mock_mode_service: MagicMock) -> ScoringService:
        """Create a ScoringService with mocked dependencies."""
        return ScoringService(mock_mode_service)

    def test_start_tracking(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test starting score tracking."""
        score_id = service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
            genre="fantasy",
        )

        assert score_id == 1
        mock_mode_service.record_generation.assert_called_once_with(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
            genre="fantasy",
        )

    def test_start_tracking_stores_active_score(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test that start_tracking stores the score ID."""
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )

        assert service.get_active_score_id("ch-1", "writer") == 1

    def test_finish_tracking_stores_original_content(self, service: ScoringService) -> None:
        """Test that finish_tracking stores content for edit distance."""
        content = "Original chapter content"
        service.finish_tracking(
            score_id=1,
            content=content,
            tokens_generated=500,
            time_seconds=10.0,
            chapter_id="ch-1",
        )

        # The original content should be stored
        assert service._original_content.get("ch-1") == content

    def test_finish_tracking_persists_performance_metrics(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test that finish_tracking persists performance metrics to database."""
        service.finish_tracking(
            score_id=42,
            content="Chapter content",
            tokens_generated=500,
            time_seconds=10.0,
            chapter_id="ch-1",
        )

        # Verify performance metrics were persisted
        mock_mode_service.update_performance_metrics.assert_called_once_with(
            42,
            tokens_generated=500,
            time_seconds=10.0,
        )

    def test_on_regenerate(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test recording regenerate signal."""
        # First start tracking
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )

        # Then record regenerate
        service.on_regenerate("ch-1", "writer")

        mock_mode_service.record_implicit_signal.assert_called_once_with(1, was_regenerated=True)

    def test_on_regenerate_no_active_score(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test on_regenerate with no active score."""
        service.on_regenerate("nonexistent", "writer")
        mock_mode_service.record_implicit_signal.assert_not_called()

    def test_on_edit(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test recording edit signal."""
        # Start tracking and store original content
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )
        service._original_content["ch-1"] = "Original content here"

        # Record edit
        service.on_edit("ch-1", "Modified content here")

        mock_mode_service.record_implicit_signal.assert_called_once()
        call_args = mock_mode_service.record_implicit_signal.call_args
        assert call_args[0][0] == 1  # score_id
        assert "edit_distance" in call_args[1]

    def test_on_edit_no_original(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test on_edit with no original content."""
        service.on_edit("ch-1", "Some content")
        mock_mode_service.record_implicit_signal.assert_not_called()

    def test_on_accept(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test recording accept signal."""
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )

        service.on_accept("ch-1")

        mock_mode_service.record_implicit_signal.assert_called_once_with(1, edit_distance=0)

    def test_on_rate_valid(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test recording valid rating."""
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )

        service.on_rate("ch-1", 4)

        mock_mode_service.record_implicit_signal.assert_called_once_with(1, user_rating=4)

    def test_on_rate_invalid(self, service: ScoringService, mock_mode_service: MagicMock) -> None:
        """Test recording invalid rating."""
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )

        # Invalid ratings should not be recorded
        service.on_rate("ch-1", 0)
        service.on_rate("ch-1", 6)

        mock_mode_service.record_implicit_signal.assert_not_called()

    def test_judge_chapter_quality(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test judging chapter quality."""
        scores = service.judge_chapter_quality(
            content="Chapter content",
            genre="fantasy",
            tone="dark",
            themes=["redemption"],
            score_id=1,
        )

        assert scores.prose_quality == 8.0
        mock_mode_service.judge_quality.assert_called_once()
        mock_mode_service.update_quality_scores.assert_called_once_with(1, scores)

    def test_judge_chapter_quality_no_score_id(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test judging quality without score ID."""
        scores = service.judge_chapter_quality(
            content="Chapter content",
            genre="fantasy",
            tone="dark",
            themes=["redemption"],
        )

        assert scores.prose_quality == 8.0
        mock_mode_service.update_quality_scores.assert_not_called()

    def test_calculate_consistency_score(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test calculating consistency score."""
        issues = [
            {"description": "Minor issue", "severity": "low"},
            {"description": "Major issue", "severity": "high"},
        ]

        score = service.calculate_consistency_score(issues, score_id=1)

        assert score == 9.0
        mock_mode_service.calculate_consistency_score.assert_called_once_with(issues)
        mock_mode_service.update_quality_scores.assert_called_once()

    def test_clear_chapter_tracking(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test clearing tracking data for a chapter."""
        # Set up tracking data
        service.start_tracking(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
        )
        service.start_tracking(
            project_id="test-project",
            agent_role="editor",
            model_id="test-model",
            chapter_id="ch-1",
        )
        service._original_content["ch-1"] = "Content"

        # Clear tracking
        service.clear_chapter_tracking("ch-1")

        # Verify cleared
        assert service.get_active_score_id("ch-1", "writer") is None
        assert service.get_active_score_id("ch-1", "editor") is None
        assert "ch-1" not in service._original_content

    def test_calculate_edit_distance(self, service: ScoringService) -> None:
        """Test edit distance calculation."""
        # Identical strings should have 0 distance
        distance = service._calculate_edit_distance("hello", "hello")
        assert distance == 0

        # Similar strings should have some distance
        distance = service._calculate_edit_distance("hello", "hallo")
        assert distance >= 0

        # Very different strings should have higher distance
        similar_distance = service._calculate_edit_distance("hello world", "hello worlds")
        different_distance = service._calculate_edit_distance("hello", "goodbye completely")
        assert different_distance > similar_distance

    def test_enforce_tracking_limits(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test that tracking limits are enforced to prevent memory leaks."""
        # Set a low MAX_TRACKED_CHAPTERS for testing
        service.MAX_TRACKED_CHAPTERS = 3

        # Track more chapters than the limit
        for i in range(5):
            chapter_id = f"ch-{i}"
            service.start_tracking(
                project_id="test-project",
                agent_role="writer",
                model_id="test-model",
                chapter_id=chapter_id,
            )
            service._original_content[chapter_id] = f"Content {i}"

        # Trigger the limit enforcement
        service._enforce_tracking_limits()

        # Should only have 3 chapters left (the most recent ones)
        remaining_chapters = {key.split(":")[0] for key in service._active_scores}
        assert len(remaining_chapters) <= service.MAX_TRACKED_CHAPTERS

        # Oldest chapters should be removed (ch-0, ch-1)
        assert "ch-0" not in service._original_content
        assert "ch-1" not in service._original_content

        # Most recent chapters should still be present
        assert (
            "ch-4" in service._original_content
            or len(service._active_scores) <= service.MAX_TRACKED_CHAPTERS
        )

    def test_finish_tracking_enforces_limits(
        self, service: ScoringService, mock_mode_service: MagicMock
    ) -> None:
        """Test that finish_tracking calls _enforce_tracking_limits."""
        # Set a low MAX_TRACKED_CHAPTERS for testing
        service.MAX_TRACKED_CHAPTERS = 2

        # Track multiple chapters
        for i in range(3):
            chapter_id = f"ch-{i}"
            service.start_tracking(
                project_id="test-project",
                agent_role="writer",
                model_id="test-model",
                chapter_id=chapter_id,
            )
            # finish_tracking should enforce limits
            service.finish_tracking(
                score_id=i + 1,
                content=f"Content {i}",
                tokens_generated=100,
                time_seconds=1.0,
                chapter_id=chapter_id,
            )

        # Should only have MAX_TRACKED_CHAPTERS chapters
        remaining_chapters = {key.split(":")[0] for key in service._active_scores}
        assert len(remaining_chapters) <= service.MAX_TRACKED_CHAPTERS
