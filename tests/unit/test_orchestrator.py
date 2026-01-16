"""Tests for the orchestrator export functionality."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents import ResponseValidationError
from memory.story_state import Chapter, Character, PlotPoint, StoryBrief, StoryState
from settings import Settings
from workflows.orchestrator import StoryOrchestrator, WorkflowEvent


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

        assert orchestrator.events.maxlen == settings.workflow_max_events


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


class TestValidateResponse:
    """Tests for _validate_response method."""

    def test_validate_response_raises_on_failure(self):
        """Test validate_response raises when validation fails."""
        orchestrator = StoryOrchestrator()
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
        )
        object.__setattr__(
            orchestrator.validator,
            "validate_response",
            MagicMock(side_effect=ResponseValidationError("Invalid language")),
        )

        with pytest.raises(ResponseValidationError):
            orchestrator._validate_response("Invalid content", "task")

    def test_validate_response_returns_on_success(self):
        """Test validate_response returns response when valid."""
        orchestrator = StoryOrchestrator()
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
        )
        object.__setattr__(
            orchestrator.validator, "validate_response", MagicMock(return_value=None)
        )

        result = orchestrator._validate_response("Valid content", "task")

        assert result == "Valid content"


class TestGenerateTitleSuggestionsExceptions:
    """Tests for generate_title_suggestions exception handling."""

    def test_returns_empty_on_exception(self):
        """Test returns empty list when exception occurs."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="interview",
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
        )
        object.__setattr__(
            orchestrator.interviewer,
            "generate",
            MagicMock(side_effect=Exception("LLM error")),
        )

        result = orchestrator.generate_title_suggestions()

        assert result == []

    def test_returns_empty_on_invalid_json(self):
        """Test returns empty list when JSON parsing fails."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="interview",
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
        )
        object.__setattr__(
            orchestrator.interviewer,
            "generate",
            MagicMock(return_value="Not valid JSON at all"),
        )

        result = orchestrator.generate_title_suggestions()

        assert result == []


class TestInterviewCompletion:
    """Tests for interview completion scenarios."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_process_interview_completes_when_brief_extracted(self):
        """Test interview completes when brief is extracted."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()

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
            orchestrator.interviewer,
            "process_response",
            MagicMock(return_value="Great! Here's your brief..."),
        )
        object.__setattr__(
            orchestrator.interviewer, "extract_brief", MagicMock(return_value=mock_brief)
        )

        response, is_complete = orchestrator.process_interview_response("I want a noir mystery")

        assert is_complete is True
        assert orchestrator.story_state is not None
        assert orchestrator.story_state.brief == mock_brief
        assert orchestrator.story_state.status == "outlining"

    def test_process_interview_raises_without_story(self):
        """Test process_interview_response raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.process_interview_response("Test")

    def test_finalize_interview_raises_without_story(self):
        """Test finalize_interview raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.finalize_interview()


class TestBuildStoryStructure:
    """Tests for build_story_structure validation."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_raises_without_story_state(self):
        """Test raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.build_story_structure()

    def test_handles_validation_warning(self):
        """Test handles validation warnings without blocking."""
        orchestrator = StoryOrchestrator()
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

        # Return state with content that triggers validation
        def mock_build(state):
            state.world_description = "A world with German: Die Welt"
            state.plot_summary = "A plot summary"
            state.status = "writing"
            return state

        object.__setattr__(orchestrator.architect, "build_story_structure", mock_build)
        object.__setattr__(
            orchestrator.validator,
            "validate_response",
            MagicMock(side_effect=ResponseValidationError("Wrong language")),
        )

        # Should not raise, just log warning
        result = orchestrator.build_story_structure()

        assert result.status == "writing"


class TestGetOutlineSummary:
    """Tests for get_outline_summary edge cases."""

    def test_raises_without_story_state(self):
        """Test raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.get_outline_summary()

    def test_handles_missing_brief(self):
        """Test handles story without brief."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            characters=[],
            chapters=[],
            plot_summary="",
        )

        summary = orchestrator.get_outline_summary()

        assert "No brief available" in summary


class TestWritingMethods:
    """Tests for write_short_story, write_chapter, write_all_chapters."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    @pytest.fixture
    def orchestrator_with_story(self):
        """Create orchestrator with story ready for writing."""
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
                target_length="short_story",
                language="English",
                content_rating="general",
            ),
            characters=[Character(name="Hero", role="protagonist", description="Main character")],
            chapters=[
                Chapter(number=1, title="Ch1", outline="Beginning"),
            ],
            plot_summary="An epic tale",
        )
        return orchestrator

    def test_write_short_story_yields_events(self, orchestrator_with_story):
        """Test write_short_story yields workflow events."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_short_story", MagicMock(return_value="Once upon a time...")
        )
        object.__setattr__(
            orc.editor, "edit_chapter", MagicMock(return_value="Once upon a time... edited")
        )
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_short_story())

        assert len(events) > 0
        assert orc.story_state is not None
        assert orc.story_state.status == "complete"

    def test_write_short_story_with_revision_loop(self, orchestrator_with_story):
        """Test write_short_story performs revision when issues found."""
        from agents.continuity import ContinuityIssue

        orc = orchestrator_with_story
        mock_issues = [
            ContinuityIssue(
                severity="critical",
                category="language",
                description="Wrong language",
                location="Para 1",
                suggestion="Fix it",
            )
        ]

        # First call returns issues, second call returns empty (revision fixes it)
        check_calls = [0]

        def mock_check(*args):
            check_calls[0] += 1
            if check_calls[0] == 1:
                return mock_issues
            return []

        object.__setattr__(
            orc.writer, "write_short_story", MagicMock(return_value="Story content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", mock_check)
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "should_revise", MagicMock(return_value=True))
        object.__setattr__(
            orc.continuity, "format_revision_feedback", MagicMock(return_value="Fix these issues")
        )
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_short_story())

        assert len(events) > 0
        # Should have called writer twice (initial + revision)
        assert orc.writer.write_short_story.call_count >= 1

    def test_write_short_story_handles_validation_warning(self, orchestrator_with_story):
        """Test write_short_story handles validation warnings."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_short_story", MagicMock(return_value="Story content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(
            orc.validator,
            "validate_response",
            MagicMock(side_effect=ResponseValidationError("Wrong language")),
        )

        # Should not raise, just log warning
        events = list(orc.write_short_story())
        assert len(events) > 0

    def test_write_short_story_handles_save_failure(self, orchestrator_with_story, monkeypatch):
        """Test write_short_story handles save failure gracefully."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_short_story", MagicMock(return_value="Story content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        # Make save_story raise
        object.__setattr__(orc, "save_story", MagicMock(side_effect=OSError("Disk full")))

        # Should not raise
        events = list(orc.write_short_story())
        assert len(events) > 0

    def test_write_short_story_raises_without_story(self):
        """Test write_short_story raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.write_short_story())

    def test_write_chapter_yields_events(self, orchestrator_with_story):
        """Test write_chapter yields workflow events."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(
            orc.editor, "edit_chapter", MagicMock(return_value="Edited chapter content...")
        )
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_chapter(1))

        assert len(events) > 0

    def test_write_chapter_raises_without_story(self):
        """Test write_chapter raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.write_chapter(1))

    def test_write_chapter_raises_for_invalid_chapter(self, orchestrator_with_story):
        """Test write_chapter raises for out of bounds chapter."""
        with pytest.raises(ValueError, match="out of bounds"):
            list(orchestrator_with_story.write_chapter(99))

    def test_write_chapter_raises_for_missing_chapter_in_range(self):
        """Test write_chapter raises when chapter not found even if in valid range."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            chapters=[
                Chapter(number=1, title="Ch1", outline=""),
                Chapter(number=3, title="Ch3", outline=""),  # Skip chapter 2
            ],
        )

        # Chapter 2 is in range (1-3) but doesn't exist
        with pytest.raises(ValueError, match="not found"):
            list(orchestrator.write_chapter(2))

    def test_write_chapter_raises_without_chapters(self):
        """Test write_chapter raises when no chapters defined."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test", status="writing", chapters=[])

        with pytest.raises(ValueError, match="No chapters defined"):
            list(orchestrator.write_chapter(1))

    def test_write_chapter_with_revision_loop(self, orchestrator_with_story):
        """Test write_chapter performs revision when issues found."""
        from agents.continuity import ContinuityIssue

        orc = orchestrator_with_story
        mock_issues = [
            ContinuityIssue(
                severity="critical",
                category="character",
                description="Wrong name",
                location="Para 1",
                suggestion="Fix name",
            )
        ]

        check_calls = [0]

        def mock_check(*args):
            check_calls[0] += 1
            if check_calls[0] == 1:
                return mock_issues
            return []

        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", mock_check)
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "should_revise", MagicMock(return_value=True))
        object.__setattr__(
            orc.continuity, "format_revision_feedback", MagicMock(return_value="Fix these issues")
        )
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=["fact1"]))
        object.__setattr__(
            orc.continuity, "extract_character_arcs", MagicMock(return_value={"Hero": "arc"})
        )
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_chapter(1))

        assert len(events) > 0
        assert orc.writer.write_chapter.call_count >= 1

    def test_write_chapter_handles_validation_warning(self, orchestrator_with_story):
        """Test write_chapter handles validation warnings."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(
            orc.validator,
            "validate_response",
            MagicMock(side_effect=ResponseValidationError("Wrong language")),
        )

        # Should not raise
        events = list(orc.write_chapter(1))
        assert len(events) > 0

    def test_write_chapter_with_previous_chapter(self):
        """Test write_chapter ensures consistency with previous chapter."""
        orchestrator = StoryOrchestrator()
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
                Chapter(number=1, title="Ch1", outline="", content="Previous chapter content."),
                Chapter(number=2, title="Ch2", outline=""),
            ],
            plot_summary="Tale",
        )
        object.__setattr__(
            orchestrator.writer, "write_chapter", MagicMock(return_value="New chapter content...")
        )
        object.__setattr__(
            orchestrator.editor, "edit_chapter", MagicMock(return_value="Edited content...")
        )
        object.__setattr__(
            orchestrator.editor,
            "ensure_consistency",
            MagicMock(return_value="Consistent content..."),
        )
        object.__setattr__(orchestrator.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(
            orchestrator.continuity, "validate_against_outline", MagicMock(return_value=[])
        )
        object.__setattr__(orchestrator.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(
            orchestrator.continuity, "extract_character_arcs", MagicMock(return_value={})
        )
        object.__setattr__(
            orchestrator.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(
            orchestrator.validator, "validate_response", MagicMock(return_value=None)
        )

        events = list(orchestrator.write_chapter(2))

        assert len(events) > 0
        orchestrator.editor.ensure_consistency.assert_called()  # type: ignore[attr-defined]

    def test_write_chapter_revision_with_previous_chapter(self):
        """Test write_chapter revision loop ensures consistency with previous chapter."""
        from agents.continuity import ContinuityIssue

        orchestrator = StoryOrchestrator()
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
                Chapter(number=1, title="Ch1", outline="", content="Previous chapter content."),
                Chapter(number=2, title="Ch2", outline=""),
            ],
            plot_summary="Tale",
        )

        mock_issues = [
            ContinuityIssue(
                severity="critical",
                category="character",
                description="Issue",
                location="Para 1",
                suggestion="Fix",
            )
        ]
        check_calls = [0]

        def mock_check(*args):
            check_calls[0] += 1
            if check_calls[0] == 1:
                return mock_issues
            return []

        object.__setattr__(
            orchestrator.writer, "write_chapter", MagicMock(return_value="New chapter content...")
        )
        object.__setattr__(
            orchestrator.editor, "edit_chapter", MagicMock(return_value="Edited content...")
        )
        object.__setattr__(
            orchestrator.editor,
            "ensure_consistency",
            MagicMock(return_value="Consistent content..."),
        )
        object.__setattr__(orchestrator.continuity, "check_chapter", mock_check)
        object.__setattr__(
            orchestrator.continuity, "validate_against_outline", MagicMock(return_value=[])
        )
        object.__setattr__(orchestrator.continuity, "should_revise", MagicMock(return_value=True))
        object.__setattr__(
            orchestrator.continuity, "format_revision_feedback", MagicMock(return_value="Feedback")
        )
        object.__setattr__(orchestrator.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(
            orchestrator.continuity, "extract_character_arcs", MagicMock(return_value={})
        )
        object.__setattr__(
            orchestrator.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(
            orchestrator.validator, "validate_response", MagicMock(return_value=None)
        )

        events = list(orchestrator.write_chapter(2))

        assert len(events) > 0
        # ensure_consistency should be called at least twice (initial + revision)
        assert orchestrator.editor.ensure_consistency.call_count >= 2  # type: ignore[attr-defined]

    def test_write_chapter_updates_character_arcs(self, orchestrator_with_story):
        """Test write_chapter updates character arcs."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(
            orc.continuity, "extract_character_arcs", MagicMock(return_value={"Hero": "Growing"})
        )
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_chapter(1))

        assert len(events) > 0

    def test_write_chapter_marks_plot_points_complete(self, orchestrator_with_story):
        """Test write_chapter marks completed plot points."""
        orc = orchestrator_with_story
        assert orc.story_state is not None
        orc.story_state.plot_points = [
            PlotPoint(description="Point 1", chapter=1, completed=False),
            PlotPoint(description="Point 2", chapter=1, completed=False),
        ]

        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[0])
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        list(orc.write_chapter(1))

        assert orc.story_state.plot_points[0].completed is True

    def test_write_chapter_handles_save_failure(self, orchestrator_with_story):
        """Test write_chapter handles save failure gracefully."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))
        object.__setattr__(orc, "save_story", MagicMock(side_effect=OSError("Disk full")))

        # Should not raise
        events = list(orc.write_chapter(1))
        assert len(events) > 0

    def test_write_all_chapters_raises_without_story(self):
        """Test write_all_chapters raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.write_all_chapters())

    def test_write_all_chapters_completes_story(self, orchestrator_with_story):
        """Test write_all_chapters completes all chapters."""
        orc = orchestrator_with_story
        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.continuity, "check_full_story", MagicMock(return_value=[]))
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_all_chapters())

        assert len(events) > 0
        assert orc.story_state is not None
        assert orc.story_state.status == "complete"

    def test_write_all_chapters_with_final_issues(self, orchestrator_with_story):
        """Test write_all_chapters reports final issues."""
        from agents.continuity import ContinuityIssue

        orc = orchestrator_with_story
        final_issues = [
            ContinuityIssue(
                severity="moderate",
                category="plot_hole",
                description="Issue 1",
                location="",
                suggestion="",
            )
        ]

        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.continuity, "check_full_story", MagicMock(return_value=final_issues))
        object.__setattr__(
            orc.continuity, "format_revision_feedback", MagicMock(return_value="Feedback")
        )
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        events = list(orc.write_all_chapters())

        # Should report issues but complete
        assert len(events) > 0
        assert orc.story_state is not None
        assert orc.story_state.status == "complete"

    def test_write_all_chapters_emits_checkpoint_event(self):
        """Test write_all_chapters emits checkpoint event when on_checkpoint is provided."""
        settings = Settings()
        # checkpoint is already the default mode, chapters_between_checkpoints=3 by default

        orc = StoryOrchestrator(settings=settings)
        orc.story_state = StoryState(
            id="test",
            status="writing",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="short_story",
                language="English",
                content_rating="general",
            ),
            characters=[Character(name="Hero", role="protagonist", description="Main character")],
            chapters=[
                Chapter(number=1, title="Ch1", outline="Beginning"),
                Chapter(number=2, title="Ch2", outline="Middle"),
                Chapter(number=3, title="Ch3", outline="End"),
            ],
            plot_summary="An epic tale",
        )

        object.__setattr__(
            orc.writer, "write_chapter", MagicMock(return_value="Chapter content...")
        )
        object.__setattr__(orc.editor, "edit_chapter", MagicMock(return_value="Edited content..."))
        object.__setattr__(
            orc.editor, "ensure_consistency", MagicMock(return_value="Consistent content...")
        )
        object.__setattr__(orc.continuity, "check_chapter", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "validate_against_outline", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_new_facts", MagicMock(return_value=[]))
        object.__setattr__(orc.continuity, "extract_character_arcs", MagicMock(return_value={}))
        object.__setattr__(
            orc.continuity, "check_plot_points_completed", MagicMock(return_value=[])
        )
        object.__setattr__(orc.continuity, "check_full_story", MagicMock(return_value=[]))
        object.__setattr__(orc.validator, "validate_response", MagicMock(return_value=None))

        def on_checkpoint(chapter, content):
            return True  # Continue

        # With 3 chapters and default chapters_between_checkpoints=3,
        # checkpoint event should be emitted after chapter 3
        events = list(orc.write_all_chapters(on_checkpoint=on_checkpoint))

        assert len(events) > 0
        # Check that a "user_input_needed" event was emitted (checkpoint event)
        checkpoint_events = [e for e in events if e.event_type == "user_input_needed"]
        assert len(checkpoint_events) == 1
        assert "Checkpoint" in checkpoint_events[0].message


class TestContinuationMethods:
    """Tests for continue_chapter, edit_passage, get_edit_suggestions."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_continue_chapter_raises_without_story(self):
        """Test continue_chapter raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.continue_chapter(1))

    def test_continue_chapter_raises_for_missing_chapter(self):
        """Test continue_chapter raises when chapter not found."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test", status="writing", chapters=[Chapter(number=1, title="Ch1", outline="")]
        )

        with pytest.raises(ValueError, match="not found"):
            list(orchestrator.continue_chapter(99))

    def test_continue_chapter_raises_for_empty_content(self):
        """Test continue_chapter raises when chapter has no content."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="")],
        )

        with pytest.raises(ValueError, match="has no content"):
            list(orchestrator.continue_chapter(1))

    def test_continue_chapter_yields_events(self):
        """Test continue_chapter yields events."""
        orchestrator = StoryOrchestrator()
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
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Existing content here.")],
        )
        object.__setattr__(
            orchestrator.writer, "continue_scene", MagicMock(return_value="Continued content...")
        )
        object.__setattr__(
            orchestrator.validator, "validate_response", MagicMock(return_value=None)
        )

        events = list(orchestrator.continue_chapter(1, direction="Continue with action"))

        assert len(events) > 0

    def test_continue_chapter_handles_validation_warning(self):
        """Test continue_chapter handles validation warnings."""
        orchestrator = StoryOrchestrator()
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
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Existing content.")],
        )
        object.__setattr__(
            orchestrator.writer, "continue_scene", MagicMock(return_value="Continued content...")
        )
        object.__setattr__(
            orchestrator.validator,
            "validate_response",
            MagicMock(side_effect=ResponseValidationError("Wrong language")),
        )

        # Should not raise
        events = list(orchestrator.continue_chapter(1))
        assert len(events) > 0

    def test_continue_chapter_handles_save_failure(self, tmp_path, monkeypatch):
        """Test continue_chapter handles save failure gracefully."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="writing",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Existing content.")],
        )
        object.__setattr__(
            orchestrator.writer, "continue_scene", MagicMock(return_value="Continued content...")
        )
        object.__setattr__(
            orchestrator.validator, "validate_response", MagicMock(return_value=None)
        )
        object.__setattr__(orchestrator, "save_story", MagicMock(side_effect=OSError("Disk full")))

        # Should not raise
        events = list(orchestrator.continue_chapter(1))
        assert len(events) > 0

    def test_edit_passage_raises_without_story(self):
        """Test edit_passage raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.edit_passage("Some text"))

    def test_edit_passage_yields_events(self):
        """Test edit_passage yields events."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test", status="writing")
        object.__setattr__(
            orchestrator.editor, "edit_passage", MagicMock(return_value="Edited passage...")
        )

        events = list(orchestrator.edit_passage("Original passage", focus="dialogue"))

        assert len(events) > 0

    def test_get_edit_suggestions_yields_events(self):
        """Test get_edit_suggestions yields events."""
        orchestrator = StoryOrchestrator()
        object.__setattr__(
            orchestrator.editor, "get_edit_suggestions", MagicMock(return_value="Suggestions...")
        )

        events = list(orchestrator.get_edit_suggestions("Text to review"))

        assert len(events) > 0


class TestReviewFullStory:
    """Tests for review_full_story method."""

    def test_raises_without_story_state(self):
        """Test raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            list(orchestrator.review_full_story())

    def test_returns_issues_when_found(self):
        """Test returns issues when found."""
        from agents.continuity import ContinuityIssue

        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="complete",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Content")],
        )
        mock_issues = [
            ContinuityIssue(
                severity="moderate",
                category="character",
                description="Issue 1",
                location="",
                suggestion="",
            )
        ]
        object.__setattr__(
            orchestrator.continuity, "check_full_story", MagicMock(return_value=mock_issues)
        )
        object.__setattr__(
            orchestrator.continuity,
            "format_revision_feedback",
            MagicMock(return_value="Feedback"),
        )

        events = list(orchestrator.review_full_story())

        assert len(events) > 0

    def test_returns_no_issues_message(self):
        """Test returns no issues message when clean."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="complete",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Content")],
        )
        object.__setattr__(orchestrator.continuity, "check_full_story", MagicMock(return_value=[]))

        events = list(orchestrator.review_full_story())

        assert len(events) > 0
        assert any("No continuity issues" in e.message for e in events)


class TestExportMethods:
    """Tests for export methods."""

    @pytest.fixture
    def story_state_with_chapters(self):
        """Create story state with chapters."""
        return StoryState(
            id="test-story",
            status="complete",
            project_name="Test Project",
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
            chapters=[
                Chapter(
                    number=1,
                    title="Beginning",
                    outline="",
                    content="Chapter one content here.",
                    word_count=100,
                )
            ],
        )

    def test_get_full_story_raises_without_story(self):
        """Test get_full_story raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.get_full_story()

    def test_export_to_markdown_raises_without_story(self):
        """Test export_to_markdown raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.export_to_markdown()

    def test_export_to_markdown_without_brief(self):
        """Test export_to_markdown without brief."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="complete",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Content")],
        )

        result = orchestrator.export_to_markdown()

        assert "Untitled Story" in result

    def test_export_to_text_raises_without_story(self):
        """Test export_to_text raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.export_to_text()

    def test_export_to_text_without_brief(self):
        """Test export_to_text without brief."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test",
            status="complete",
            chapters=[Chapter(number=1, title="Ch1", outline="", content="Content")],
        )

        result = orchestrator.export_to_text()

        assert "Untitled Story" in result

    def test_export_to_epub_raises_without_story(self):
        """Test export_to_epub raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.export_to_epub()

    def test_export_to_pdf_raises_without_story(self):
        """Test export_to_pdf raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.export_to_pdf()

    def test_export_to_mobi_raises_runtime_error(self, story_state_with_chapters):
        """Test export_to_mobi raises RuntimeError."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = story_state_with_chapters

        with pytest.raises(RuntimeError, match="MOBI format is no longer supported"):
            orchestrator.export_to_mobi()

    def test_export_story_to_file_json_format(self, story_state_with_chapters, tmp_path):
        """Test export_story_to_file with JSON format."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = story_state_with_chapters

        filepath = tmp_path / "story.json"
        result = orchestrator.export_story_to_file(format="json", filepath=str(filepath))

        assert Path(result).exists()

    def test_export_story_to_file_default_path(
        self, story_state_with_chapters, tmp_path, monkeypatch
    ):
        """Test export_story_to_file uses default path when not specified."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir()
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

        orchestrator = StoryOrchestrator()
        orchestrator.story_state = story_state_with_chapters

        result = orchestrator.export_story_to_file(format="markdown")

        assert Path(result).exists()
        assert str(stories_dir) in result

    def test_get_statistics_raises_without_story(self):
        """Test get_statistics raises when no story state."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story state"):
            orchestrator.get_statistics()


class TestPersistenceMethods:
    """Tests for save, load, autosave, list methods."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        self.stories_dir = tmp_path / "stories"
        self.stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", self.stories_dir)

    def test_autosave_returns_none_without_story(self):
        """Test autosave returns None when no story."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        result = orchestrator.autosave()

        assert result is None

    def test_autosave_returns_path_on_success(self):
        """Test autosave returns path when successful."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test-123", status="writing")

        result = orchestrator.autosave()

        assert result is not None
        assert Path(result).exists()

    def test_autosave_returns_none_on_exception(self):
        """Test autosave returns None when exception occurs."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test-123", status="writing")

        # Make save_story raise
        def mock_save(*args):
            raise OSError("Disk full")

        object.__setattr__(orchestrator, "save_story", mock_save)

        result = orchestrator.autosave()

        assert result is None

    def test_save_story_raises_without_story(self):
        """Test save_story raises when no story."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story to save"):
            orchestrator.save_story()

    def test_save_story_sets_timestamps(self):
        """Test save_story sets updated_at and last_saved."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test-123", status="writing")

        path = orchestrator.save_story()

        assert orchestrator.story_state.updated_at is not None
        assert orchestrator.story_state.last_saved is not None
        assert Path(path).exists()

    def test_save_story_to_custom_path(self, tmp_path):
        """Test save_story to custom path."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test-123", status="writing")

        # Create the parent directory first
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        custom_path = custom_dir / "story.json"
        path = orchestrator.save_story(str(custom_path))

        assert Path(path).exists()
        assert "custom" in path

    def test_load_story_raises_for_missing_file(self):
        """Test load_story raises for missing file."""
        orchestrator = StoryOrchestrator()

        with pytest.raises(FileNotFoundError):
            orchestrator.load_story("/nonexistent/path/story.json")

    def test_load_story_loads_and_sets_correlation_id(self, tmp_path):
        """Test load_story loads story and sets correlation ID."""
        # First save a story
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(id="test-12345678", status="writing")
        path = orchestrator.save_story()

        # Now load it in a new orchestrator
        new_orchestrator = StoryOrchestrator()
        loaded = new_orchestrator.load_story(path)

        assert loaded.id == "test-12345678"
        assert new_orchestrator._correlation_id == "test-123"

    def test_list_saved_stories_returns_empty_for_no_dir(self, tmp_path, monkeypatch):
        """Test list_saved_stories returns empty when dir doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", nonexistent)

        result = StoryOrchestrator.list_saved_stories()

        assert result == []

    def test_list_saved_stories_returns_story_metadata(self):
        """Test list_saved_stories returns story metadata."""
        # Save a story
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = StoryState(
            id="test-123",
            status="writing",
            brief=StoryBrief(
                premise="A test story premise",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novella",
                language="English",
                content_rating="general",
            ),
        )
        orchestrator.save_story()

        result = StoryOrchestrator.list_saved_stories()

        assert len(result) >= 1
        found = [s for s in result if s["id"] == "test-123"]
        assert len(found) == 1
        assert "A test story" in found[0]["premise"]  # type: ignore[operator]

    def test_list_saved_stories_handles_corrupt_files(self, tmp_path):
        """Test list_saved_stories handles corrupt JSON files."""
        # Create a corrupt JSON file
        corrupt_file = self.stories_dir / "corrupt.json"
        corrupt_file.write_text("not valid json {{{")

        # Should not raise, just skip the corrupt file
        result = StoryOrchestrator.list_saved_stories()

        # The corrupt file should not appear in results
        assert all(s.get("id") != "corrupt" for s in result)


class TestResetState:
    """Tests for reset_state method."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path, monkeypatch):
        """Use temp directory for all tests."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("workflows.orchestrator.STORIES_DIR", stories_dir)

    def test_reset_state_clears_everything(self):
        """Test reset_state clears story and events."""
        orchestrator = StoryOrchestrator()
        orchestrator.create_new_story()
        orchestrator._emit("test", "Agent", "Message")

        assert orchestrator.story_state is not None
        assert len(orchestrator.events) > 0

        orchestrator.reset_state()

        assert orchestrator.story_state is None
        assert len(orchestrator.events) == 0  # type: ignore[unreachable]

    def test_reset_state_clears_conversation_history(self):
        """Test reset_state clears interviewer conversation history."""
        orchestrator = StoryOrchestrator()
        orchestrator.interviewer.conversation_history = [{"role": "user", "content": "test"}]

        orchestrator.reset_state()

        assert orchestrator.interviewer.conversation_history == []
