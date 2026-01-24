"""Tests for StoryService regeneration with feedback."""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import Chapter, Character, StoryBrief, StoryState
from src.services.orchestrator import WorkflowEvent
from src.services.story_service import GenerationCancelled, StoryService
from src.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def story_service(settings):
    """Create StoryService instance."""
    return StoryService(settings)


@pytest.fixture
def sample_story_with_content():
    """Create story state with a chapter that has content."""
    brief = StoryBrief(
        premise="A detective solves a mystery",
        genre="Mystery",
        tone="Dark",
        themes=["Justice"],
        setting_time="1940s",
        setting_place="Los Angeles",
        target_length="novella",
        language="English",
        content_rating="mature",
    )

    state = StoryState(
        id="test-story-regen",
        project_name="Test Story",
        brief=brief,
        status="writing",
        characters=[
            Character(
                name="Jack Stone",
                role="protagonist",
                description="A grizzled detective",
                personality_traits=["cynical", "determined"],
                goals=["Find the truth"],
            ),
        ],
        chapters=[
            Chapter(
                number=1,
                title="The Client",
                outline="Introduction to the case",
                content="Jack Stone sat in his dimly lit office, waiting.",
                word_count=9,
            ),
        ],
    )
    return state


class TestRegenerateChapterWithFeedback:
    """Tests for regenerate_chapter_with_feedback method."""

    @patch("src.services.story_service.StoryOrchestrator")
    def test_regenerate_saves_version_before_regenerating(
        self, mock_orchestrator_class, story_service, sample_story_with_content
    ):
        """Test that current content is saved as a version before regenerating."""
        # Setup mock orchestrator
        mock_orch = MagicMock()
        mock_orchestrator_class.return_value = mock_orch
        mock_orch.story_state = sample_story_with_content

        # Mock write_chapter to return events
        def mock_write_chapter(chapter_num, feedback=None):
            """Simulate chapter writing by updating content and yielding workflow events."""
            # Simulate writing new content
            chapter = sample_story_with_content.chapters[0]
            chapter.content = "Regenerated content with feedback applied."
            chapter.word_count = 6
            yield WorkflowEvent(event_type="agent_start", agent_name="Writer", message="Writing...")
            yield WorkflowEvent(
                event_type="agent_complete", agent_name="System", message="Complete"
            )

        mock_orch.write_chapter = mock_write_chapter

        chapter = sample_story_with_content.chapters[0]
        original_content = chapter.content
        assert len(chapter.versions) == 0

        # Regenerate with feedback
        feedback = "Make it more exciting with action"
        list(story_service.regenerate_chapter_with_feedback(sample_story_with_content, 1, feedback))

        # Verify versions were saved (original + regenerated result)
        # First version is original content (saved before regeneration, no feedback)
        # Second version is regenerated content (with feedback that produced it)
        assert len(chapter.versions) == 2
        first_version = chapter.versions[0]
        assert first_version.content == original_content
        assert first_version.feedback == ""  # No feedback for the original version

        second_version = chapter.versions[1]
        assert second_version.feedback == feedback  # Feedback associated with the result

    @patch("src.services.story_service.StoryOrchestrator")
    def test_regenerate_calls_orchestrator_with_feedback(
        self, mock_orchestrator_class, story_service, sample_story_with_content
    ):
        """Test that orchestrator is called with the feedback."""
        mock_orch = MagicMock()
        mock_orchestrator_class.return_value = mock_orch
        mock_orch.story_state = sample_story_with_content

        write_calls = []

        def mock_write_chapter(chapter_num, feedback=None):
            """Track write_chapter calls and yield a completion event."""
            write_calls.append((chapter_num, feedback))
            chapter = sample_story_with_content.chapters[0]
            chapter.content = "New content"
            yield WorkflowEvent(
                event_type="agent_complete", agent_name="System", message="Complete"
            )

        mock_orch.write_chapter = mock_write_chapter

        feedback = "Add more suspense"
        list(story_service.regenerate_chapter_with_feedback(sample_story_with_content, 1, feedback))

        # Verify write_chapter was called with feedback
        assert len(write_calls) == 1
        assert write_calls[0] == (1, feedback)

    def test_regenerate_requires_existing_content(self, story_service):
        """Test that regeneration fails if chapter has no content."""
        brief = StoryBrief(
            premise="Test",
            genre="Test",
            tone="Test",
            setting_time="Now",
            setting_place="Here",
            target_length="novella",
            language="English",
            content_rating="none",
        )

        state = StoryState(
            id="test-empty",
            brief=brief,
            chapters=[
                Chapter(number=1, title="Empty", outline="No content", content=""),
            ],
        )

        with pytest.raises(ValueError, match="has no content to regenerate"):
            list(story_service.regenerate_chapter_with_feedback(state, 1, "Feedback"))

    def test_regenerate_requires_valid_chapter(self, story_service, sample_story_with_content):
        """Test that regeneration fails for non-existent chapter."""
        with pytest.raises(ValueError, match="Chapter 99 not found"):
            list(
                story_service.regenerate_chapter_with_feedback(
                    sample_story_with_content, 99, "Feedback"
                )
            )

    def test_regenerate_requires_feedback(self, story_service, sample_story_with_content):
        """Test that feedback is required."""
        with pytest.raises(ValueError):
            list(story_service.regenerate_chapter_with_feedback(sample_story_with_content, 1, ""))

    @patch("src.services.story_service.StoryOrchestrator")
    def test_regenerate_rollback_on_cancellation(
        self, mock_orchestrator_class, story_service, sample_story_with_content
    ):
        """Test that content is rolled back if generation is cancelled."""
        mock_orch = MagicMock()
        mock_orchestrator_class.return_value = mock_orch
        mock_orch.story_state = sample_story_with_content

        chapter = sample_story_with_content.chapters[0]

        # Track whether cancel was checked
        cancel_checked = [False]

        def mock_write_chapter(chapter_num, feedback=None):
            """Yield an event to trigger the cancel check during regeneration."""
            # Yield an event so the cancel check runs
            yield WorkflowEvent(event_type="agent_start", agent_name="Writer", message="Writing...")
            # After the event, the service checks for cancellation
            # Since our cancel_check returns True, GenerationCancelled should be raised

        mock_orch.write_chapter = mock_write_chapter

        # Create a cancel check that returns True to simulate cancellation
        def should_cancel():
            """Return True to simulate user-requested cancellation."""
            cancel_checked[0] = True
            return True

        # Expect GenerationCancelled exception
        with pytest.raises(GenerationCancelled):
            list(
                story_service.regenerate_chapter_with_feedback(
                    sample_story_with_content, 1, "Feedback", cancel_check=should_cancel
                )
            )

        # Verify cancel was checked
        assert cancel_checked[0] is True
        # Content should be preserved (rolled back happens in the exception handler)
        # A version should have been saved before the attempt
        assert len(chapter.versions) >= 1
