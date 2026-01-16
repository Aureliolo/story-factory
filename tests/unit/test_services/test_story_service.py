"""Tests for StoryService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memory.story_state import Chapter, Character, StoryBrief, StoryState
from memory.world_database import WorldDatabase
from services.story_service import StoryService
from settings import Settings
from workflows.orchestrator import WorkflowEvent


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


@pytest.fixture
def world_db(tmp_path: Path):
    """Create a temporary WorldDatabase."""
    return WorldDatabase(tmp_path / "test_world.db")


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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            story_service._get_orchestrator(sample_story_state)

            MockOrchestrator.assert_called_once()
            assert sample_story_state.id in story_service._orchestrators

    def test_reuses_existing_orchestrator(self, story_service, sample_story_state):
        """Test reuses existing orchestrator for same story."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            orch1 = story_service._get_orchestrator(sample_story_state)
            orch2 = story_service._get_orchestrator(sample_story_state)

            assert MockOrchestrator.call_count == 1  # Only created once
            assert orch1 == orch2

    def test_evicts_oldest_orchestrator_when_full(self, story_service, sample_brief, settings):
        """Test LRU eviction when cache is full."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.start_interview.return_value = "What story would you like to write?"
            MockOrchestrator.return_value = mock_orch

            result = story_service.start_interview(sample_story_state)

            assert result == "What story would you like to write?"
            assert len(sample_story_state.interview_history) == 1
            assert sample_story_state.interview_history[0]["role"] == "assistant"

    def test_process_interview(self, story_service, sample_story_state):
        """Test processes interview response."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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

        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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


class TestStoryServiceStructure:
    """Tests for structure building phase."""

    def test_build_structure(self, story_service, sample_story_state, world_db):
        """Test builds story structure."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.story_state = sample_story_state
            MockOrchestrator.return_value = mock_orch

            result = story_service.build_structure(sample_story_state, world_db)

            mock_orch.build_story_structure.assert_called_once()
            assert result == sample_story_state

    def test_build_structure_raises_without_brief(self, story_service, world_db):
        """Test build_structure raises when no brief exists."""
        state = StoryState(id="test", status="interview")

        with pytest.raises(ValueError, match="brief"):
            story_service.build_structure(state, world_db)

    def test_extract_entities_to_world(self, story_service, sample_story_with_chapters, world_db):
        """Test extracts characters to world database."""
        story_service._extract_entities_to_world(sample_story_with_chapters, world_db)

        entities = world_db.list_entities(entity_type="character")
        assert len(entities) == 1
        assert entities[0].name == "Jack Stone"

    def test_extract_entities_skips_existing(
        self, story_service, sample_story_with_chapters, world_db
    ):
        """Test skips extraction if entity already exists in world database."""
        # Add the character first
        world_db.add_entity("character", "Jack Stone", "A detective")

        # Extract should skip the existing entity
        story_service._extract_entities_to_world(sample_story_with_chapters, world_db)

        # Should still be just one entity
        entities = world_db.list_entities(entity_type="character")
        assert len(entities) == 1

    def test_extract_entities_with_relationships(self, story_service, world_db):
        """Test extracts character relationships to world database when related entity exists."""
        # Pre-add Bob so Alice's relationship can be created
        bob_id = world_db.add_entity("character", "Bob", "Friend character")

        state = StoryState(
            id="test-rel",
            project_name="Relationship Test",
            brief=StoryBrief(
                premise="A story",
                genre="Drama",
                tone="Serious",
                setting_time="Present",
                setting_place="City",
                target_length="novel",
                language="English",
                content_rating="general",
            ),
            status="writing",
            characters=[
                Character(
                    name="Alice",
                    role="protagonist",
                    description="Main character",
                    personality_traits=["kind"],
                    goals=["happiness"],
                    relationships={"Bob": "friend"},
                ),
            ],
        )

        story_service._extract_entities_to_world(state, world_db)

        entities = world_db.list_entities(entity_type="character")
        assert len(entities) == 2

        # Check relationships were added
        alice = next((e for e in entities if e.name == "Alice"), None)
        assert alice is not None

        # Check relationship exists from Alice to Bob
        relationships = world_db.get_relationships(alice.id)
        assert len(relationships) > 0
        assert any(r.target_id == bob_id or r.source_id == bob_id for r in relationships)

    def test_get_outline(self, story_service, sample_story_state):
        """Test gets story outline summary."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.get_outline_summary.return_value = "Story outline here..."
            MockOrchestrator.return_value = mock_orch

            outline = story_service.get_outline(sample_story_state)

            assert outline == "Story outline here..."


class TestStoryServiceWriting:
    """Tests for writing phase methods."""

    def test_get_full_story(self, story_service, sample_story_with_chapters):
        """Test gets full story text."""
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            # Mock the generator
            def mock_write_chapter(chapter_num):
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_all():
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

        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_write_short():
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

        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_continue(chapter_num, direction=None):
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_edit(text, focus=None):
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_suggestions(text):
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
        with patch("services.story_service.StoryOrchestrator") as MockOrchestrator:
            mock_orch = MagicMock()

            def mock_review():
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
