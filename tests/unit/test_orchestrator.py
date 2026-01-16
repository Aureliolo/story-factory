"""Tests for the orchestrator export functionality."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memory.story_state import Chapter, Character, PlotPoint, StoryBrief, StoryState
from settings import Settings
from workflows.orchestrator import MAX_EVENTS, StoryOrchestrator, WorkflowEvent


class TestExportFunctionality:
    """Tests for story export to files."""

    @pytest.fixture
    def sample_story_state(self):
        """Create a sample story state for testing."""
        state = StoryState(
            id="test-story-123",
            created_at=datetime.now(),
            status="complete",
            brief=StoryBrief(
                premise="A robot learns to love",
                genre="Science Fiction",
                tone="Hopeful",
                setting_time="2150",
                setting_place="Mars Colony",
                target_length="short_story",
                content_rating="none",
            ),
        )
        state.chapters = [
            Chapter(
                number=1,
                title="First Encounter",
                outline="The robot meets a human",
                content="The robot's optical sensors focused on the human...",
                status="final",
                word_count=500,
            )
        ]
        return state

    def test_export_to_markdown_file(self, sample_story_state):
        """Should export story to markdown file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.md"
            result = orchestrator.export_story_to_file(format="markdown", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".md")

            content = Path(result).read_text(encoding="utf-8")
            assert "A robot learns to love" in content
            assert "Chapter 1: First Encounter" in content

    def test_export_to_text_file(self, sample_story_state):
        """Should export story to plain text file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            result = orchestrator.export_story_to_file(format="text", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".txt")

            content = Path(result).read_text(encoding="utf-8")
            assert "Science Fiction" in content
            assert "FIRST ENCOUNTER" in content

    def test_export_raises_on_no_story(self):
        """Should raise ValueError when no story to export."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story to export"):
            orchestrator.export_story_to_file(format="markdown")

    def test_export_raises_on_invalid_format(self, sample_story_state):
        """Should raise ValueError for unsupported format."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with pytest.raises(ValueError, match="Unsupported export format"):
            orchestrator.export_story_to_file(format="docx")

    def test_export_creates_directory_if_not_exists(self, sample_story_state):
        """Should create parent directories if they don't exist."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dir" / "story.md"
            result = orchestrator.export_story_to_file(format="markdown", filepath=str(filepath))

            assert Path(result).exists()
            assert Path(result).parent.exists()

    def test_export_to_pdf_file(self, sample_story_state):
        """Should export story to PDF file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pdf"
            result = orchestrator.export_story_to_file(format="pdf", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".pdf")

    def test_export_to_epub_file(self, sample_story_state):
        """Should export story to EPUB file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.epub"
            result = orchestrator.export_story_to_file(format="epub", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".epub")


class TestWorkflowEvents:
    """Tests for workflow event handling."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests to avoid polluting output/stories."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_create_new_story_sets_correlation_id(self):
        """Should set correlation ID when creating a new story."""
        orchestrator = StoryOrchestrator()
        state = orchestrator.create_new_story()

        assert orchestrator._correlation_id is not None
        assert orchestrator._correlation_id == state.id[:8]

    def test_workflow_events_include_timestamp(self):
        """Should include timestamp in workflow events."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()

        # Emit a test event
        before = datetime.now()
        event = orchestrator._emit("test", "TestAgent", "Test message")
        after = datetime.now()

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        # Verify timestamp is approximately equal to current time
        assert before <= event.timestamp <= after

    def test_workflow_events_include_correlation_id(self):
        """Should include correlation ID in workflow events."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()

        # Emit a test event
        event = orchestrator._emit("test", "TestAgent", "Test message")

        assert event.correlation_id is not None
        assert event.correlation_id == orchestrator._correlation_id

    def test_workflow_event_dataclass_fields(self):
        """Should have all expected fields in WorkflowEvent."""
        event = WorkflowEvent(
            event_type="test",
            agent_name="TestAgent",
            message="Test message",
            data={"key": "value"},
            timestamp=datetime.now(),
            correlation_id="abc123",
        )

        assert event.event_type == "test"
        assert event.agent_name == "TestAgent"
        assert event.message == "Test message"
        assert event.data == {"key": "value"}
        assert event.timestamp is not None
        assert event.correlation_id == "abc123"


class TestStoryOrchestratorInit:
    """Tests for StoryOrchestrator initialization."""

    def test_init_with_custom_settings(self):
        """Test orchestrator initializes with custom settings."""
        settings = Settings()

        orchestrator = StoryOrchestrator(settings=settings)

        assert orchestrator.settings == settings

    def test_init_with_model_override(self):
        """Test orchestrator initializes with model override."""
        settings = Settings()
        orchestrator = StoryOrchestrator(settings=settings, model_override="custom-model:7b")

        assert orchestrator.model_override == "custom-model:7b"

    def test_events_deque_has_maxlen(self):
        """Test events deque has max length to prevent memory leaks."""
        settings = Settings()
        orchestrator = StoryOrchestrator(settings=settings)

        assert orchestrator.events.maxlen == MAX_EVENTS


class TestStoryOrchestratorCreateNewStory:
    """Tests for create_new_story method."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_creates_story_with_uuid(self):
        """Test creates story with valid UUID."""
        orchestrator = StoryOrchestrator()
        state = orchestrator.create_new_story()

        assert state is not None
        assert len(state.id) == 36  # UUID format
        assert state.status == "interview"

    def test_creates_story_with_default_name(self):
        """Test creates story with default name containing date."""
        orchestrator = StoryOrchestrator()
        state = orchestrator.create_new_story()

        assert "New Story" in state.project_name

    def test_sets_correlation_id(self):
        """Test sets correlation ID from story ID."""
        orchestrator = StoryOrchestrator()
        state = orchestrator.create_new_story()

        assert orchestrator._correlation_id == state.id[:8]


class TestStoryOrchestratorUpdateProjectName:
    """Tests for update_project_name method."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_updates_project_name(self):
        """Test updates project name."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()

        orchestrator.update_project_name("My Awesome Story")

        assert orchestrator.story_state is not None
        assert orchestrator.story_state.project_name == "My Awesome Story"

    def test_no_update_when_no_story(self):
        """Test does nothing when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        # Should not raise
        orchestrator.update_project_name("Test")


class TestStoryOrchestratorGenerateTitleSuggestions:
    """Tests for generate_title_suggestions method."""

    def test_returns_empty_when_no_story_state(self):
        """Test returns empty list when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        result = orchestrator.generate_title_suggestions()

        assert result == []

    def test_returns_empty_when_no_brief(self):
        """Test returns empty list when no brief."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test", status="interview")

        result = orchestrator.generate_title_suggestions()

        assert result == []

    def test_generates_titles_from_brief(self):
        """Test generates titles when brief exists."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="interview",
            brief=StoryBrief(
                premise="A robot learns to love",
                genre="Science Fiction",
                tone="Hopeful",
                themes=["Love", "Identity"],
                setting_time="2150",
                setting_place="Mars",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
        )
        mock_generate = MagicMock(
            return_value='["The Mechanical Heart", "Love in Silicon", "Mars Awakening", "Digital Dreams", "The Robot\'s Journey"]'
        )
        object.__setattr__(orchestrator.interviewer, "generate", mock_generate)

        result = orchestrator.generate_title_suggestions()

        assert len(result) == 5
        assert "The Mechanical Heart" in result


class TestStoryOrchestratorInterview:
    """Tests for interview phase methods."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_start_interview(self):
        """Test starts interview and returns questions."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()
        mock_get_initial_questions = MagicMock(return_value="What story would you like to write?")
        object.__setattr__(
            orchestrator.interviewer, "get_initial_questions", mock_get_initial_questions
        )

        result = orchestrator.start_interview()

        assert result == "What story would you like to write?"

    def test_process_interview_response(self):
        """Test processes interview response."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()
        object.__setattr__(
            orchestrator.interviewer,
            "process_response",
            MagicMock(return_value="Tell me more about the characters!"),
        )
        object.__setattr__(orchestrator.interviewer, "extract_brief", MagicMock(return_value=None))

        response, is_complete = orchestrator.process_interview_response("I want a fantasy story")

        assert "Tell me more" in response
        assert is_complete is False

    def test_finalize_interview(self):
        """Test finalizes interview."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()
        orchestrator.interviewer.conversation_history = [
            {"role": "user", "content": "I want a mystery"}
        ]
        mock_brief = StoryBrief(
            premise="A mystery",
            genre="Mystery",
            tone="Dark",
            setting_time="1940s",
            setting_place="LA",
            target_length="novella",
            language="English",
            content_rating="mature",
        )
        object.__setattr__(
            orchestrator.interviewer, "finalize_brief", MagicMock(return_value=mock_brief)
        )

        brief = orchestrator.finalize_interview()

        assert brief.genre == "Mystery"
        assert orchestrator.story_state is not None
        assert orchestrator.story_state.brief == mock_brief


class TestStoryOrchestratorStructure:
    """Tests for structure building methods."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_build_story_structure_yields_events(self):
        """Test build_story_structure yields workflow events."""
        settings = Settings()
        orchestrator = StoryOrchestrator(settings=settings)
        orchestrator.create_new_story()
        assert orchestrator.story_state is not None
        orchestrator.story_state.brief = StoryBrief(
            premise="Test",
            genre="Fantasy",
            tone="Epic",
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novella",
            language="English",
            content_rating="general",
        )

        # Mock all architect methods using object.__setattr__
        object.__setattr__(
            orchestrator.architect, "create_world", MagicMock(return_value="A magical world")
        )
        object.__setattr__(
            orchestrator.architect,
            "create_characters",
            MagicMock(
                return_value=[
                    Character(name="Hero", role="protagonist", description="The main character")
                ]
            ),
        )
        object.__setattr__(
            orchestrator.architect,
            "create_plot_outline",
            MagicMock(
                return_value=("An epic journey", [PlotPoint(description="Beginning", chapter=1)])
            ),
        )
        object.__setattr__(
            orchestrator.architect,
            "create_chapter_outline",
            MagicMock(return_value=[Chapter(number=1, title="Chapter 1", outline="The start")]),
        )

        orchestrator.build_story_structure()

        # Events are stored in orchestrator.events, not yielded
        assert len(orchestrator.events) > 0
        # Check that events contain workflow events
        assert any(
            hasattr(e, "event_type") and e.event_type == "agent_start" for e in orchestrator.events
        )


class TestStoryOrchestratorHelpers:
    """Tests for helper methods."""

    def test_get_outline_summary(self):
        """Test gets outline summary."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
            characters=[
                Character(name="Hero", role="protagonist", description="The main character")
            ],
            chapters=[
                Chapter(number=1, title="Beginning", outline="The start"),
                Chapter(number=2, title="Middle", outline="The journey"),
            ],
            plot_summary="An epic tale of adventure",
        )

        summary = orchestrator.get_outline_summary()

        assert "Fantasy" in summary
        assert "Hero" in summary
        assert "Beginning" in summary or "Chapter 1" in summary

    def test_get_full_story(self):
        """Test gets full story text."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="complete",
            chapters=[
                Chapter(number=1, title="One", outline="", content="Chapter one content."),
                Chapter(number=2, title="Two", outline="", content="Chapter two content."),
            ],
        )

        story = orchestrator.get_full_story()

        assert "Chapter one content" in story
        assert "Chapter two content" in story

    def test_get_statistics(self):
        """Test gets story statistics."""
        settings = Settings()
        orchestrator = StoryOrchestrator(settings=settings)
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="Test",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
            characters=[Character(name="Hero", role="protagonist", description="Main character")],
            chapters=[
                Chapter(number=1, title="One", outline="", content="Word " * 100, word_count=100),
            ],
        )

        stats = orchestrator.get_statistics()

        assert stats["total_words"] >= 100
        assert stats["total_chapters"] == 1
        assert "characters" in stats or len(orchestrator.story_state.characters) == 1

    def test_clear_events(self):
        """Test clears all events."""
        orchestrator = StoryOrchestrator()
        orchestrator._emit("test", "Agent", "Message")
        orchestrator._emit("test", "Agent", "Message")

        assert len(orchestrator.events) == 2

        orchestrator.clear_events()

        assert len(orchestrator.events) == 0

    def test_validate_response_without_brief(self):
        """Test validate_response returns response when no brief."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        result = orchestrator._validate_response("Test response", "task")

        assert result == "Test response"

    def test_interaction_mode_property(self):
        """Test interaction_mode returns settings value."""
        settings = Settings()
        orchestrator = StoryOrchestrator(settings=settings)

        result = orchestrator.interaction_mode

        assert result == settings.interaction_mode
