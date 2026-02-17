"""Tests for StoryService."""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import Chapter, Character, OutlineVariation, StoryBrief, StoryState
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
def sample_brief():
    """Create a sample story brief."""
    return StoryBrief(
        premise="A detective solves a mystery",
        genre="Mystery",
        subgenres=["Noir"],
        tone="Dark",
        themes=["Justice"],
        setting_time="1940s",
        setting_place="Los Angeles",
        target_length="novella",
        language="English",
        content_rating="mature",
        content_preferences=[],
        content_avoid=[],
    )


@pytest.fixture
def sample_story_state(sample_brief):
    """Create a sample story state."""
    return StoryState(
        id="test-story-001",
        project_name="Test Story",
        brief=sample_brief,
        status="interview",
    )


@pytest.fixture
def sample_story_with_chapters(sample_brief):
    """Create story state with chapters."""
    state = StoryState(
        id="test-story-002",
        project_name="Test Story With Chapters",
        brief=sample_brief,
        status="writing",
        characters=[
            Character(
                name="Jack Stone",
                role="protagonist",
                description="A grizzled detective",
                personality_traits=["cynical", "determined"],
                goals=["Find the truth"],
                relationships={"Vera": "client"},
            ),
        ],
        chapters=[
            Chapter(number=1, title="The Client", outline="Introduction", content=""),
            Chapter(number=2, title="The Chase", outline="Action", content=""),
        ],
    )
    return state


class TestStoryServiceInit:
    """Tests for StoryService initialization."""

    def test_init_with_settings(self, settings):
        """Test service initializes with settings."""
        service = StoryService(settings)
        assert service.settings == settings
        assert service._orchestrators == {}


class TestStoryServiceOrchestratorCache:
    """Tests for orchestrator caching behavior."""

    def test_creates_new_orchestrator(self, story_service, sample_story_state):
        """Test creates new orchestrator for new story."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            story_service._get_orchestrator(sample_story_state)

            MockOrchestrator.assert_called_once()
            assert sample_story_state.id in story_service._orchestrators

    def test_reuses_existing_orchestrator(self, story_service, sample_story_state):
        """Test reuses existing orchestrator for same story."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            orch1 = story_service._get_orchestrator(sample_story_state)
            orch2 = story_service._get_orchestrator(sample_story_state)

            assert MockOrchestrator.call_count == 1  # Only created once
            assert orch1 == orch2

    def test_evicts_oldest_orchestrator_when_full(self, story_service, sample_brief, settings):
        """Test LRU eviction when cache is full."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = MagicMock()

            cache_size = settings.orchestrator_cache_size

            # Fill the cache
            for i in range(cache_size + 2):
                state = StoryState(id=f"story-{i}", brief=sample_brief, status="interview")
                story_service._get_orchestrator(state)

            # Cache should be at max size
            assert len(story_service._orchestrators) == cache_size

            # First story should have been evicted
            assert "story-0" not in story_service._orchestrators
            assert "story-1" not in story_service._orchestrators


class TestStoryServiceSyncState:
    """Tests for _sync_state method."""

    def test_syncs_all_relevant_fields(self, story_service, sample_story_state):
        """Test syncs all story state fields from orchestrator."""
        mock_orch = MagicMock()
        mock_orch.story_state = StoryState(
            id=sample_story_state.id,
            brief=sample_story_state.brief,
            status="writing",
            characters=[Character(name="Test", role="protagonist", description="Test")],
            chapters=[Chapter(number=1, title="Test", outline="Test")],
            plot_summary="A summary",
            world_description="A world",
        )

        story_service._sync_state(mock_orch, sample_story_state)

        assert sample_story_state.status == "writing"
        assert len(sample_story_state.characters) == 1
        assert sample_story_state.plot_summary == "A summary"

    def test_handles_none_orchestrator_state(self, story_service, sample_story_state):
        """Test handles orchestrator with no story state."""
        mock_orch = MagicMock()
        mock_orch.story_state = None

        # Should not raise
        story_service._sync_state(mock_orch, sample_story_state)


class TestStoryServiceInterview:
    """Tests for interview phase methods."""

    def test_start_interview(self, story_service, sample_story_state):
        """Test starts interview and returns questions."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.start_interview.return_value = "What story would you like to write?"
            MockOrchestrator.return_value = mock_orch

            result = story_service.start_interview(sample_story_state)

            assert result == "What story would you like to write?"
            assert len(sample_story_state.interview_history) == 1
            assert sample_story_state.interview_history[0]["role"] == "assistant"

    def test_process_interview(self, story_service, sample_story_state):
        """Test processes interview response."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.process_interview_response.return_value = ("Tell me more!", False)
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            response, is_complete = story_service.process_interview(
                sample_story_state, "I want a mystery story"
            )

            assert response == "Tell me more!"
            assert is_complete is False
            assert len(sample_story_state.interview_history) == 2
            assert sample_story_state.interview_history[0]["role"] == "user"
            assert sample_story_state.interview_history[1]["role"] == "assistant"

    def test_finalize_interview(self, story_service, sample_story_state, sample_brief):
        """Test finalizes interview and returns brief."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.finalize_interview.return_value = sample_brief
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            brief = story_service.finalize_interview(sample_story_state)

            assert brief == sample_brief

    def test_continue_interview(self, story_service, sample_story_state):
        """Test continues interview after completion."""
        sample_story_state.brief = StoryBrief(
            premise="A mystery",
            genre="Mystery",
            tone="Dark",
            setting_time="1940s",
            setting_place="LA",
            target_length="novella",
            language="English",
            content_rating="mature",
        )

        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.process_interview_response.return_value = ("Noted!", True)
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            response = story_service.continue_interview(
                sample_story_state, "Actually, make it a comedy"
            )

            assert response == "Noted!"

    def test_continue_interview_raises_without_brief(self, story_service, sample_story_state):
        """Test continue_interview raises when no brief exists."""
        sample_story_state.brief = None

        with pytest.raises(ValueError, match="brief"):
            story_service.continue_interview(sample_story_state, "More info")


class TestStoryServiceOutline:
    """Tests for outline retrieval."""

    def test_get_outline(self, story_service, sample_story_state):
        """Test gets story outline summary."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.get_outline_summary.return_value = "Story outline here..."
            MockOrchestrator.return_value = mock_orch

            outline = story_service.get_outline(sample_story_state)

            assert outline == "Story outline here..."


class TestStoryServiceWriting:
    """Tests for writing phase methods."""

    def test_get_full_story(self, story_service, sample_story_with_chapters):
        """Test gets full story text."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.get_full_story.return_value = "The full story content..."
            MockOrchestrator.return_value = mock_orch

            story = story_service.get_full_story(sample_story_with_chapters)

            assert story == "The full story content..."

    def test_get_chapter_content(self, story_service, sample_story_with_chapters):
        """Test gets specific chapter content."""
        sample_story_with_chapters.chapters[0].content = "Chapter 1 content"

        content = story_service.get_chapter_content(sample_story_with_chapters, 1)

        assert content == "Chapter 1 content"

    def test_get_chapter_content_not_found(self, story_service, sample_story_with_chapters):
        """Test returns None for non-existent chapter."""
        content = story_service.get_chapter_content(sample_story_with_chapters, 99)

        assert content is None

    def test_get_statistics(self, story_service, sample_story_state):
        """Test gets story statistics."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.get_statistics.return_value = {
                "total_words": 5000,
                "chapters": 3,
            }
            MockOrchestrator.return_value = mock_orch

            stats = story_service.get_statistics(sample_story_state)

            assert stats["total_words"] == 5000
            assert stats["chapters"] == 3


class TestStoryServiceReviews:
    """Tests for review management."""

    def test_add_review(self, story_service, sample_story_with_chapters):
        """Test adding a review to a story."""
        story_service.add_review(
            sample_story_with_chapters,
            review_type="user_feedback",
            content="This chapter needs more action",
            chapter_num=1,
        )

        assert len(sample_story_with_chapters.reviews) == 1
        assert sample_story_with_chapters.reviews[0]["type"] == "user_feedback"
        assert sample_story_with_chapters.reviews[0]["content"] == "This chapter needs more action"
        assert sample_story_with_chapters.reviews[0]["chapter"] == 1

    def test_add_review_without_chapter(self, story_service, sample_story_with_chapters):
        """Test adding a general review without chapter number."""
        story_service.add_review(
            sample_story_with_chapters,
            review_type="user_note",
            content="Overall pacing is good",
            chapter_num=None,
        )

        assert len(sample_story_with_chapters.reviews) == 1
        assert sample_story_with_chapters.reviews[0]["chapter"] is None

    def test_add_multiple_reviews(self, story_service, sample_story_with_chapters):
        """Test adding multiple reviews."""
        story_service.add_review(
            sample_story_with_chapters, "user_feedback", "Chapter 1 feedback", 1
        )
        story_service.add_review(
            sample_story_with_chapters, "user_feedback", "Chapter 2 feedback", 2
        )
        story_service.add_review(
            sample_story_with_chapters, "ai_suggestion", "Consider adding more dialogue", 1
        )

        assert len(sample_story_with_chapters.reviews) == 3

    def test_get_reviews_all(self, story_service, sample_story_with_chapters):
        """Test getting all reviews."""
        story_service.add_review(sample_story_with_chapters, "user_feedback", "Feedback 1", 1)
        story_service.add_review(sample_story_with_chapters, "user_feedback", "Feedback 2", 2)

        reviews = story_service.get_reviews(sample_story_with_chapters)

        assert len(reviews) == 2

    def test_get_reviews_by_chapter(self, story_service, sample_story_with_chapters):
        """Test getting reviews for a specific chapter."""
        story_service.add_review(
            sample_story_with_chapters, "user_feedback", "Chapter 1 feedback", 1
        )
        story_service.add_review(
            sample_story_with_chapters, "user_feedback", "Chapter 2 feedback", 2
        )
        story_service.add_review(
            sample_story_with_chapters, "user_note", "Another chapter 1 note", 1
        )

        chapter_1_reviews = story_service.get_reviews(sample_story_with_chapters, chapter_num=1)

        assert len(chapter_1_reviews) == 2
        assert all(r["chapter"] == 1 for r in chapter_1_reviews)


class TestStoryServiceTitleGeneration:
    """Tests for title generation."""

    def test_generate_title_suggestions(self, story_service, sample_story_state):
        """Test generates title suggestions."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.generate_title_suggestions.return_value = [
                "The Dark Mystery",
                "Shadows of LA",
                "The Final Clue",
            ]
            MockOrchestrator.return_value = mock_orch

            titles = story_service.generate_title_suggestions(sample_story_state)

            assert len(titles) == 3
            assert "The Dark Mystery" in titles


class TestStoryServiceCleanup:
    """Tests for cleanup methods."""

    def test_cleanup_orchestrator(self, story_service, sample_story_state):
        """Test cleans up orchestrator for story."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            MockOrchestrator.return_value = MagicMock()

            # Create orchestrator
            story_service._get_orchestrator(sample_story_state)
            assert sample_story_state.id in story_service._orchestrators

            # Clean up
            story_service.cleanup_orchestrator(sample_story_state)

            assert sample_story_state.id not in story_service._orchestrators

    def test_cleanup_nonexistent_orchestrator(self, story_service, sample_story_state):
        """Test cleanup handles non-existent orchestrator gracefully."""
        # Should not raise
        story_service.cleanup_orchestrator(sample_story_state)


class TestStoryServiceGenerators:
    """Tests for generator-based methods."""

    def test_write_chapter_generator(self, story_service, sample_story_with_chapters):
        """Test write_chapter yields events."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            # Mock the generator
            def mock_write_chapter(chapter_num):
                """Yield mock workflow events for chapter writing."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Starting"
                )
                yield WorkflowEvent(
                    event_type="agent_complete", agent_name="Writer", message="Done"
                )
                yield WorkflowEvent(
                    event_type="agent_complete", agent_name="System", message="Complete"
                )

            mock_orch.write_chapter = mock_write_chapter
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(story_service.write_chapter(sample_story_with_chapters, 1))

            assert len(events) == 3
            assert events[0].event_type == "agent_start"

    def test_write_all_chapters_generator(self, story_service, sample_story_with_chapters):
        """Test write_all_chapters yields events."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_all():
                """Yield mock workflow events for writing all chapters."""
                yield WorkflowEvent(
                    event_type="chapter_start", agent_name="System", message="Chapter 1"
                )
                yield WorkflowEvent(
                    event_type="chapter_complete", agent_name="System", message="Done"
                )

            mock_orch.write_all_chapters = mock_write_all
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(story_service.write_all_chapters(sample_story_with_chapters))

            assert len(events) >= 1

    def test_write_short_story_generator(self, story_service, sample_story_state):
        """Test write_short_story yields events."""
        sample_story_state.chapters = [Chapter(number=1, title="Story", outline="The whole story")]

        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_short():
                """Yield mock workflow events for short story writing."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Starting"
                )
                yield WorkflowEvent(
                    event_type="agent_complete", agent_name="System", message="Complete"
                )

            mock_orch.write_short_story = mock_write_short
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            events = list(story_service.write_short_story(sample_story_state))

            assert len(events) >= 1

    def test_continue_chapter_generator(self, story_service, sample_story_with_chapters):
        """Test continue_chapter yields events."""
        sample_story_with_chapters.chapters[0].content = "Chapter content so far..."

        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_continue(chapter_num, direction=None):
                """Yield mock workflow events for chapter continuation."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Continuing"
                )
                yield WorkflowEvent(
                    event_type="agent_complete",
                    agent_name="Writer",
                    message="Done",
                    data={"continuation": "More text..."},
                )

            mock_orch.continue_chapter = mock_continue
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(
                story_service.continue_chapter(sample_story_with_chapters, 1, "Add action")
            )

            assert len(events) == 2
            assert events[0].event_type == "agent_start"
            assert events[1].event_type == "agent_complete"

    def test_edit_passage_generator(self, story_service, sample_story_with_chapters):
        """Test edit_passage yields events."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_edit(text, focus=None):
                """Yield mock workflow events for passage editing."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Editor", message="Editing"
                )
                yield WorkflowEvent(
                    event_type="agent_complete",
                    agent_name="Editor",
                    message="Done",
                    data={"edited_text": "Improved text"},
                )

            mock_orch.edit_passage = mock_edit
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(
                story_service.edit_passage(
                    sample_story_with_chapters, "Original text", focus="dialogue"
                )
            )

            assert len(events) == 2
            assert events[0].agent_name == "Editor"

    def test_get_edit_suggestions_generator(self, story_service, sample_story_with_chapters):
        """Test get_edit_suggestions yields events."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_suggestions(text):
                """Yield mock workflow events for edit suggestions."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Editor", message="Reviewing"
                )
                yield WorkflowEvent(
                    event_type="agent_complete",
                    agent_name="Editor",
                    message="Done",
                    data={"suggestions": "Consider adding more dialogue"},
                )

            mock_orch.get_edit_suggestions = mock_suggestions
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(
                story_service.get_edit_suggestions(sample_story_with_chapters, "Text to review")
            )

            assert len(events) == 2

    def test_review_full_story_generator(self, story_service, sample_story_with_chapters):
        """Test review_full_story yields events."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_review():
                """Yield mock workflow events for full story review."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Continuity", message="Reviewing"
                )
                yield WorkflowEvent(
                    event_type="agent_complete",
                    agent_name="Continuity",
                    message="Done",
                    data={
                        "issues": [{"description": "Timeline inconsistency", "severity": "minor"}]
                    },
                )

            mock_orch.review_full_story = mock_review
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(story_service.review_full_story(sample_story_with_chapters))

            assert len(events) == 2
            assert events[1].data["issues"][0]["description"] == "Timeline inconsistency"


class TestStoryServiceExceptionHandling:
    """Tests for exception handling paths."""

    def test_start_interview_raises_on_orchestrator_error(self, story_service, sample_story_state):
        """Test start_interview propagates exceptions from orchestrator."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.start_interview.side_effect = RuntimeError("LLM connection failed")
            MockOrchestrator.return_value = mock_orch

            with pytest.raises(RuntimeError, match="LLM connection failed"):
                story_service.start_interview(sample_story_state)

    def test_process_interview_completes_with_is_complete_true(
        self, story_service, sample_story_state
    ):
        """Test process_interview logs completion when is_complete is True."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.process_interview_response.return_value = (
                "Great! Here's your brief.",
                True,
            )
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            response, is_complete = story_service.process_interview(
                sample_story_state, "Yes, that sounds perfect"
            )

            assert is_complete is True
            assert response == "Great! Here's your brief."

    def test_process_interview_raises_on_orchestrator_error(
        self, story_service, sample_story_state
    ):
        """Test process_interview propagates exceptions from orchestrator."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.process_interview_response.side_effect = ValueError("Invalid input")
            MockOrchestrator.return_value = mock_orch

            with pytest.raises(ValueError, match="Invalid input"):
                story_service.process_interview(sample_story_state, "Some response")

    def test_finalize_interview_raises_on_orchestrator_error(
        self, story_service, sample_story_state
    ):
        """Test finalize_interview propagates exceptions from orchestrator."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.finalize_interview.side_effect = RuntimeError("Failed to generate brief")
            MockOrchestrator.return_value = mock_orch

            with pytest.raises(RuntimeError, match="Failed to generate brief"):
                story_service.finalize_interview(sample_story_state)


class TestStoryServiceOutlineVariations:
    """Tests for outline variation methods."""

    def test_generate_outline_variations(self, story_service, sample_story_state):
        """Test generates outline variations from architect."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_architect = MagicMock()

            # Create mock variations
            mock_variations = [
                OutlineVariation(
                    id="var-1",
                    name="Variation 1",
                    plot_summary="A dark mystery",
                    ai_rationale="Classic noir approach",
                ),
                OutlineVariation(
                    id="var-2",
                    name="Variation 2",
                    plot_summary="A redemption arc",
                    ai_rationale="Character-driven approach",
                ),
            ]
            mock_architect.generate_outline_variations.return_value = mock_variations
            mock_orch.architect = mock_architect
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            variations = story_service.generate_outline_variations(sample_story_state, count=2)

            assert len(variations) == 2
            assert variations[0].name == "Variation 1"
            assert variations[1].name == "Variation 2"
            mock_architect.generate_outline_variations.assert_called_once_with(
                sample_story_state, count=2
            )
            # Verify variations were added to state
            assert len(sample_story_state.outline_variations) == 2

    def test_generate_outline_variations_raises_without_brief(self, story_service):
        """Test generate_outline_variations raises when no brief exists."""
        state = StoryState(id="test-no-brief", status="interview")

        with pytest.raises(ValueError, match="brief"):
            story_service.generate_outline_variations(state)

    def test_select_variation_success(self, story_service, sample_story_state):
        """Test selecting a variation as canonical."""
        # Add a variation to the state
        variation = OutlineVariation(
            id="var-to-select",
            name="Selected Variation",
            plot_summary="The selected plot",
            world_description="A dark world",
            characters=[
                Character(name="Hero", role="protagonist", description="The main character")
            ],
            chapters=[Chapter(number=1, title="Beginning", outline="Start")],
        )
        sample_story_state.outline_variations.append(variation)

        result = story_service.select_variation(sample_story_state, "var-to-select")

        assert result is True
        assert sample_story_state.selected_variation_id == "var-to-select"
        assert sample_story_state.plot_summary == "The selected plot"
        assert sample_story_state.world_description == "A dark world"

    def test_select_variation_not_found(self, story_service, sample_story_state):
        """Test selecting a non-existent variation returns False."""
        result = story_service.select_variation(sample_story_state, "non-existent-id")

        assert result is False

    def test_rate_variation_success(self, story_service, sample_story_state):
        """Test rating a variation successfully."""
        # Add a variation to rate
        variation = OutlineVariation(id="var-to-rate", name="Rate Me")
        sample_story_state.outline_variations.append(variation)

        result = story_service.rate_variation(
            sample_story_state,
            "var-to-rate",
            rating=4,
            notes="Great plot twist!",
        )

        assert result is True
        assert variation.user_rating == 4
        assert variation.user_notes == "Great plot twist!"

    def test_rate_variation_clamps_rating(self, story_service, sample_story_state):
        """Test rating is clamped between 0 and 5."""
        variation = OutlineVariation(id="var-clamp", name="Clamp Test")
        sample_story_state.outline_variations.append(variation)

        # Test rating above 5
        story_service.rate_variation(sample_story_state, "var-clamp", rating=10)
        assert variation.user_rating == 5

        # Test rating below 0
        story_service.rate_variation(sample_story_state, "var-clamp", rating=-3)
        assert variation.user_rating == 0

    def test_rate_variation_without_notes(self, story_service, sample_story_state):
        """Test rating variation without notes doesn't overwrite existing notes."""
        variation = OutlineVariation(
            id="var-no-notes", name="No Notes", user_notes="Existing notes"
        )
        sample_story_state.outline_variations.append(variation)

        result = story_service.rate_variation(sample_story_state, "var-no-notes", rating=3)

        assert result is True
        assert variation.user_rating == 3
        assert variation.user_notes == "Existing notes"

    def test_rate_variation_not_found(self, story_service, sample_story_state):
        """Test rating non-existent variation returns False."""
        result = story_service.rate_variation(sample_story_state, "non-existent", rating=5)

        assert result is False

    def test_toggle_variation_favorite_on(self, story_service, sample_story_state):
        """Test toggling favorite on."""
        variation = OutlineVariation(id="var-fav", name="Fav Test", is_favorite=False)
        sample_story_state.outline_variations.append(variation)

        result = story_service.toggle_variation_favorite(sample_story_state, "var-fav")

        assert result is True
        assert variation.is_favorite is True

    def test_toggle_variation_favorite_off(self, story_service, sample_story_state):
        """Test toggling favorite off."""
        variation = OutlineVariation(id="var-unfav", name="Unfav Test", is_favorite=True)
        sample_story_state.outline_variations.append(variation)

        result = story_service.toggle_variation_favorite(sample_story_state, "var-unfav")

        assert result is True
        assert variation.is_favorite is False

    def test_toggle_variation_favorite_not_found(self, story_service, sample_story_state):
        """Test toggling favorite on non-existent variation returns False."""
        result = story_service.toggle_variation_favorite(sample_story_state, "non-existent")

        assert result is False

    def test_create_merged_variation(self, story_service, sample_story_state):
        """Test creating a merged variation."""
        # Add source variations
        var1 = OutlineVariation(
            id="var-source-1",
            name="Source 1",
            world_description="A dark world",
            characters=[Character(name="Hero", role="protagonist", description="The hero")],
        )
        var2 = OutlineVariation(
            id="var-source-2",
            name="Source 2",
            plot_summary="An epic adventure",
            chapters=[Chapter(number=1, title="Start", outline="Beginning")],
        )
        sample_story_state.outline_variations.extend([var1, var2])

        merged = story_service.create_merged_variation(
            sample_story_state,
            name="Merged Story",
            source_elements={
                "var-source-1": ["world", "characters"],
                "var-source-2": ["plot", "chapters"],
            },
        )

        assert merged.name == "Merged Story"
        assert merged.world_description == "A dark world"
        assert len(merged.characters) == 1
        assert merged.plot_summary == "An epic adventure"
        assert len(merged.chapters) == 1


class TestStoryServiceCancellation:
    """Tests for cancellation handling in generator methods."""

    def test_write_chapter_cancellation(self, story_service, sample_story_with_chapters):
        """Test write_chapter raises GenerationCancelled when cancelled."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            # Mock generator that yields one event before cancellation check
            def mock_write_chapter(chapter_num):
                """Yield mock workflow events for chapter writing cancellation test."""
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Starting"
                )
                yield WorkflowEvent(
                    event_type="agent_progress", agent_name="Writer", message="Writing"
                )

            mock_orch.write_chapter = mock_write_chapter
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            # Create a cancel check that returns True after the first event
            call_count = [0]

            def cancel_check():
                """Return True after first call to simulate cancellation."""
                call_count[0] += 1
                return call_count[0] > 1

            gen = story_service.write_chapter(
                sample_story_with_chapters, 1, cancel_check=cancel_check
            )

            # First event should be yielded
            first_event = next(gen)
            assert first_event.event_type == "agent_start"

            # Second call should raise GenerationCancelled
            with pytest.raises(GenerationCancelled) as exc_info:
                next(gen)

            assert exc_info.value.chapter_num == 1

    def test_write_all_chapters_cancellation(self, story_service, sample_story_with_chapters):
        """Test write_all_chapters raises GenerationCancelled when cancelled."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_all():
                """Yield mock workflow events for write all chapters cancellation test."""
                yield WorkflowEvent(
                    event_type="chapter_start", agent_name="System", message="Chapter 1"
                )
                yield WorkflowEvent(
                    event_type="agent_progress", agent_name="Writer", message="Writing"
                )

            mock_orch.write_all_chapters = mock_write_all
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            call_count = [0]

            def cancel_check():
                """Return True after first call to simulate cancellation."""
                call_count[0] += 1
                return call_count[0] > 1

            gen = story_service.write_all_chapters(
                sample_story_with_chapters, cancel_check=cancel_check
            )

            # First event should be yielded
            first_event = next(gen)
            assert first_event.event_type == "chapter_start"

            # Second call should raise GenerationCancelled
            with pytest.raises(GenerationCancelled, match="Write all chapters cancelled"):
                next(gen)


class TestStoryServiceRegenerateChapter:
    """Tests for regenerate_chapter_with_feedback method."""

    def test_regenerate_chapter_success(self, story_service, sample_story_with_chapters):
        """Test regenerating a chapter with feedback."""
        # Set up chapter with existing content
        sample_story_with_chapters.chapters[0].content = "Original chapter content here."
        sample_story_with_chapters.chapters[0].word_count = 4

        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_chapter(chapter_num, feedback=None):
                """Yield mock workflow events and update chapter content for regeneration."""
                # Simulate writing new content
                sample_story_with_chapters.chapters[0].content = "New improved content."
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Regenerating"
                )
                yield WorkflowEvent(
                    event_type="agent_complete", agent_name="System", message="Complete"
                )

            mock_orch.write_chapter = mock_write_chapter
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            events = list(
                story_service.regenerate_chapter_with_feedback(
                    sample_story_with_chapters, 1, "Add more dialogue"
                )
            )

            assert len(events) == 2
            # Verify version was saved
            chapter = sample_story_with_chapters.chapters[0]
            assert len(chapter.versions) >= 1

    def test_regenerate_chapter_not_found(self, story_service, sample_story_with_chapters):
        """Test regenerating non-existent chapter raises ValueError."""
        with pytest.raises(ValueError, match="Chapter 99 not found"):
            list(
                story_service.regenerate_chapter_with_feedback(
                    sample_story_with_chapters, 99, "Some feedback"
                )
            )

    def test_regenerate_chapter_no_content(self, story_service, sample_story_with_chapters):
        """Test regenerating chapter without content raises ValueError."""
        # Ensure chapter has no content
        sample_story_with_chapters.chapters[0].content = ""

        with pytest.raises(ValueError, match="has no content to regenerate"):
            list(
                story_service.regenerate_chapter_with_feedback(
                    sample_story_with_chapters, 1, "Some feedback"
                )
            )

    def test_regenerate_chapter_cancellation_rollback(
        self, story_service, sample_story_with_chapters
    ):
        """Test regeneration rollback on cancellation."""
        original_content = "Original chapter content."
        sample_story_with_chapters.chapters[0].content = original_content
        sample_story_with_chapters.chapters[0].word_count = 3

        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_chapter(chapter_num, feedback=None):
                """Yield mock workflow events with partial content for rollback test."""
                # Simulate partial writing
                sample_story_with_chapters.chapters[0].content = "Partial new content..."
                yield WorkflowEvent(
                    event_type="agent_start", agent_name="Writer", message="Regenerating"
                )
                yield WorkflowEvent(
                    event_type="agent_progress", agent_name="Writer", message="In progress"
                )

            mock_orch.write_chapter = mock_write_chapter
            mock_orch.story_state = sample_story_with_chapters
            MockOrchestrator.return_value = mock_orch

            call_count = [0]

            def cancel_check():
                """Return True after first call to trigger cancellation and rollback."""
                call_count[0] += 1
                return call_count[0] > 1

            gen = story_service.regenerate_chapter_with_feedback(
                sample_story_with_chapters, 1, "Add action", cancel_check=cancel_check
            )

            # First event should be yielded
            first_event = next(gen)
            assert first_event.event_type == "agent_start"

            # Second call should raise and rollback
            with pytest.raises(GenerationCancelled):
                next(gen)

            # Content should be rolled back to original
            chapter = sample_story_with_chapters.chapters[0]
            assert chapter.content == original_content


class TestStoryServiceWorldGeneration:
    """Tests for world generation methods."""

    def test_generate_more_characters(self, story_service, sample_story_state):
        """Test generating additional characters."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            new_chars = [
                Character(name="Sidekick", role="supporting", description="A helpful friend"),
                Character(name="Villain", role="antagonist", description="The main threat"),
            ]
            mock_orch.generate_more_characters.return_value = new_chars
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            result = story_service.generate_more_characters(sample_story_state, count=2)

            assert len(result) == 2
            assert result[0].name == "Sidekick"
            assert result[1].name == "Villain"
            mock_orch.generate_more_characters.assert_called_once_with(2)

    def test_generate_locations(self, story_service, sample_story_state):
        """Test generating locations."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            locations = [
                {"name": "Dark Alley", "description": "A shadowy backstreet"},
                {"name": "Jazz Club", "description": "A smoky nightclub"},
                {"name": "Detective Office", "description": "A cramped office"},
            ]
            mock_orch.generate_locations.return_value = locations
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            result = story_service.generate_locations(sample_story_state, count=3)

            assert len(result) == 3
            assert result[0]["name"] == "Dark Alley"
            mock_orch.generate_locations.assert_called_once_with(3)

    def test_generate_relationships(self, story_service, sample_story_state):
        """Test generating relationships between entities."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            relationships = [
                {"source": "Jack", "target": "Vera", "type": "romantic_interest"},
                {"source": "Jack", "target": "Chief", "type": "reports_to"},
            ]
            mock_orch.generate_relationships.return_value = relationships
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            entity_names = ["Jack", "Vera", "Chief"]
            existing_rels = [("Jack", "Vera")]

            result = story_service.generate_relationships(
                sample_story_state,
                entity_names=entity_names,
                existing_rels=existing_rels,
                count=5,
            )

            assert len(result) == 2
            mock_orch.generate_relationships.assert_called_once_with(entity_names, existing_rels, 5)

    def test_rebuild_world(self, story_service, sample_story_state):
        """Test rebuilding the entire world."""
        with patch("src.services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            result = story_service.rebuild_world(sample_story_state)

            assert result == sample_story_state
            mock_orch.rebuild_world.assert_called_once()


class TestGenerationCancelledException:
    """Tests for GenerationCancelled exception."""

    def test_basic_initialization(self):
        """Test basic exception initialization."""
        exc = GenerationCancelled()
        assert str(exc) == "Generation cancelled"
        assert exc.chapter_num is None
        assert exc.progress_state == {}

    def test_with_message(self):
        """Test exception with custom message."""
        exc = GenerationCancelled("Custom cancellation message")
        assert str(exc) == "Custom cancellation message"

    def test_with_chapter_num(self):
        """Test exception with chapter number."""
        exc = GenerationCancelled("Chapter cancelled", chapter_num=3)
        assert exc.chapter_num == 3
        assert str(exc) == "Chapter cancelled"

    def test_with_progress_state(self):
        """Test exception with progress state."""
        progress = {"words_written": 500, "percent_complete": 45}
        exc = GenerationCancelled("Cancelled mid-write", progress_state=progress)
        assert exc.progress_state == progress
        assert exc.progress_state["words_written"] == 500

    def test_full_initialization(self):
        """Test exception with all parameters."""
        progress = {"current_scene": 2, "total_scenes": 5}
        exc = GenerationCancelled(
            "Full cancellation",
            chapter_num=7,
            progress_state=progress,
        )
        assert str(exc) == "Full cancellation"
        assert exc.chapter_num == 7
        assert exc.progress_state == progress


class TestStoryServiceLearning:
    """Tests for learning system integration."""

    def test_complete_project_no_mode_service(self, story_service, sample_story_state):
        """Test complete_project when no mode_service is configured."""
        result = story_service.complete_project(sample_story_state)

        assert result["project_id"] == sample_story_state.id
        assert result["status"] == "complete"
        assert result["pending_recommendations"] == []
        assert sample_story_state.status == "complete"

    def test_complete_project_with_mode_service(self, settings, sample_story_state):
        """Test complete_project with mode_service that returns recommendations."""
        from unittest.mock import MagicMock

        from src.memory.mode_models import RecommendationType, TuningRecommendation

        mock_mode_service = MagicMock()
        mock_rec = TuningRecommendation(
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="huihui_ai/dolphin3-abliterated:8b",
            suggested_value="vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            reason="Better quality",
            confidence=0.9,
            affected_role="writer",
        )
        mock_mode_service.on_project_complete.return_value = [mock_rec]
        mock_mode_service.handle_recommendations.return_value = [mock_rec]

        service = StoryService(settings, mode_service=mock_mode_service)
        result = service.complete_project(sample_story_state)

        assert result["project_id"] == sample_story_state.id
        assert result["status"] == "complete"
        assert len(result["pending_recommendations"]) == 1
        mock_mode_service.on_project_complete.assert_called_once()
        mock_mode_service.handle_recommendations.assert_called_once()

    def test_complete_project_with_exception(self, settings, sample_story_state):
        """Test complete_project handles exceptions gracefully."""
        from unittest.mock import MagicMock

        mock_mode_service = MagicMock()
        mock_mode_service.on_project_complete.side_effect = Exception("Learning failed")

        service = StoryService(settings, mode_service=mock_mode_service)
        result = service.complete_project(sample_story_state)

        # Should still complete but with empty recommendations
        assert result["project_id"] == sample_story_state.id
        assert result["status"] == "complete"
        assert result["pending_recommendations"] == []

    def test_on_story_complete_no_recommendations(self, settings, sample_story_state):
        """Test _on_story_complete when no recommendations are generated."""
        from unittest.mock import MagicMock

        mock_mode_service = MagicMock()
        mock_mode_service.on_project_complete.return_value = []

        service = StoryService(settings, mode_service=mock_mode_service)
        result = service._on_story_complete(sample_story_state)

        assert result is None
        mock_mode_service.handle_recommendations.assert_not_called()

    def test_complete_project_validates_state(self, story_service):
        """Test complete_project validates state parameter."""
        with pytest.raises(ValueError, match="'state' cannot be None"):
            story_service.complete_project(None)

    def test_complete_project_validates_state_type(self, story_service):
        """Test complete_project validates state is StoryState."""
        with pytest.raises(TypeError, match="'state' must be StoryState"):
            story_service.complete_project({"id": "fake"})


class TestStoryServiceRAGIntegration:
    """Tests for StoryService RAG context_retrieval wiring."""

    def test_init_with_context_retrieval(self, settings):
        """Test StoryService accepts context_retrieval parameter."""
        mock_cr = MagicMock()
        svc = StoryService(settings, context_retrieval=mock_cr)
        assert svc.context_retrieval is mock_cr

    def test_init_without_context_retrieval(self, settings):
        """Test StoryService works without context_retrieval (default None)."""
        svc = StoryService(settings)
        assert svc.context_retrieval is None

    def test_orchestrator_gets_context_retrieval(self, settings, sample_story_state):
        """Test _get_orchestrator passes context_retrieval to orchestrator."""
        mock_cr = MagicMock()
        svc = StoryService(settings, context_retrieval=mock_cr)
        orc = svc._get_orchestrator(sample_story_state)
        assert orc.context_retrieval is mock_cr

    def test_write_chapter_passes_world_db(self, settings, sample_story_state):
        """Test write_chapter uses set_project_context to thread world_db."""
        svc = StoryService(settings)
        mock_world_db = MagicMock()

        # Mock the orchestrator's write_chapter to return events
        event = WorkflowEvent(
            event_type="agent_complete",
            agent_name="System",
            message="Done",
        )
        with patch.object(svc, "_get_orchestrator") as mock_get_orc:
            mock_orc = MagicMock()
            mock_orc.write_chapter.return_value = iter([event])
            mock_get_orc.return_value = mock_orc

            events = list(svc.write_chapter(sample_story_state, 1, world_db=mock_world_db))

            assert len(events) > 0
            # Verify set_project_context was called with world_db
            mock_orc.set_project_context.assert_called_once_with(
                world_db=mock_world_db, story_state=sample_story_state
            )

    def test_write_all_chapters_passes_world_db(self, settings, sample_story_state):
        """Test write_all_chapters uses set_project_context to thread world_db."""
        svc = StoryService(settings)
        mock_world_db = MagicMock()

        event = WorkflowEvent(
            event_type="agent_complete",
            agent_name="System",
            message="Done",
        )
        with patch.object(svc, "_get_orchestrator") as mock_get_orc:
            mock_orc = MagicMock()
            mock_orc.write_all_chapters.return_value = iter([event])
            mock_get_orc.return_value = mock_orc

            events = list(svc.write_all_chapters(sample_story_state, world_db=mock_world_db))

            assert len(events) > 0
            mock_orc.set_project_context.assert_called_once_with(
                world_db=mock_world_db, story_state=sample_story_state
            )

    def test_write_short_story_passes_world_db(self, settings, sample_story_state):
        """Test write_short_story uses set_project_context to thread world_db."""
        svc = StoryService(settings)
        mock_world_db = MagicMock()

        event = WorkflowEvent(
            event_type="agent_complete",
            agent_name="System",
            message="Done",
        )
        with patch.object(svc, "_get_orchestrator") as mock_get_orc:
            mock_orc = MagicMock()
            mock_orc.write_short_story.return_value = iter([event])
            mock_get_orc.return_value = mock_orc

            events = list(svc.write_short_story(sample_story_state, world_db=mock_world_db))

            assert len(events) > 0
            mock_orc.set_project_context.assert_called_once_with(
                world_db=mock_world_db, story_state=sample_story_state
            )

    def test_regenerate_chapter_passes_world_db(self, settings, sample_story_state):
        """Test regenerate_chapter_with_feedback uses set_project_context."""
        svc = StoryService(settings)
        mock_world_db = MagicMock()

        # Give the story state a chapter with content to regenerate
        sample_story_state.chapters.append(
            Chapter(
                number=1, title="Test Chapter", outline="Test outline", content="Original content"
            )
        )

        event = WorkflowEvent(
            event_type="agent_complete",
            agent_name="System",
            message="Done",
        )
        with patch.object(svc, "_get_orchestrator") as mock_get_orc:
            mock_orc = MagicMock()
            mock_orc.write_chapter.return_value = iter([event])
            mock_get_orc.return_value = mock_orc

            list(
                svc.regenerate_chapter_with_feedback(
                    sample_story_state, 1, "make it better", world_db=mock_world_db
                )
            )

            mock_orc.set_project_context.assert_called_once_with(
                world_db=mock_world_db, story_state=sample_story_state
            )
