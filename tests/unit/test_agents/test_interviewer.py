"""Tests for InterviewerAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.interviewer import InterviewerAgent
from memory.story_state import StoryBrief
from settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def interviewer(settings):
    """Create InterviewerAgent with mocked Ollama client."""
    with patch("agents.base.ollama.Client"):
        agent = InterviewerAgent(model="test-model", settings=settings)
        return agent


class TestInterviewerAgentInit:
    """Tests for InterviewerAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("agents.base.ollama.Client"):
            agent = InterviewerAgent(settings=settings)
            assert agent.name == "Interviewer"
            assert agent.role == "Story Requirements Gatherer"
            assert agent.conversation_history == []

    def test_init_with_custom_model(self, settings):
        """Test agent initializes with custom model."""
        with patch("agents.base.ollama.Client"):
            agent = InterviewerAgent(model="custom:model", settings=settings)
            assert agent.model == "custom:model"


class TestInterviewerGetInitialQuestions:
    """Tests for get_initial_questions method."""

    def test_returns_generated_questions(self, interviewer):
        """Test returns LLM-generated questions."""
        interviewer.generate = MagicMock(return_value="Welcome! What story would you like?")

        result = interviewer.get_initial_questions()

        assert result == "Welcome! What story would you like?"
        interviewer.generate.assert_called_once()
        # Verify the prompt contains key elements
        call_args = interviewer.generate.call_args[0][0]
        assert "greeting" in call_args.lower() or "story idea" in call_args.lower()


class TestInterviewerProcessResponse:
    """Tests for process_response method."""

    def test_adds_user_response_to_history(self, interviewer):
        """Test user response is added to conversation history."""
        interviewer.generate = MagicMock(
            return_value="Interesting! Tell me more about the setting."
        )

        interviewer.process_response("I want a fantasy story about dragons")

        assert len(interviewer.conversation_history) == 2
        assert interviewer.conversation_history[0]["role"] == "user"
        assert "dragons" in interviewer.conversation_history[0]["content"]
        assert interviewer.conversation_history[1]["role"] == "assistant"

    def test_includes_context_in_prompt(self, interviewer):
        """Test context is included in the prompt when provided."""
        interviewer.generate = MagicMock(return_value="Got it!")

        interviewer.process_response(
            "I want an adult romance story", context="Language: English\nContent Rating: adult"
        )

        call_args = interviewer.generate.call_args[0][0]
        assert "ALREADY DETERMINED" in call_args
        assert "English" in call_args

    def test_returns_assistant_response(self, interviewer):
        """Test returns the generated response."""
        expected = "Great premise! What genre are you thinking?"
        interviewer.generate = MagicMock(return_value=expected)

        result = interviewer.process_response("A mystery in Victorian London")

        assert result == expected


class TestInterviewerExtractBrief:
    """Tests for extract_brief method."""

    def test_extracts_valid_json_brief(self, interviewer):
        """Test extracts brief from JSON in response."""
        response = """Here's your story brief:
```json
{
    "premise": "A detective solves crimes",
    "genre": "Mystery",
    "subgenres": ["Noir"],
    "tone": "Dark",
    "themes": ["Justice"],
    "setting_time": "1920s",
    "setting_place": "Chicago",
    "target_length": "novella",
    "language": "English",
    "content_rating": "mature",
    "content_preferences": [],
    "content_avoid": [],
    "additional_notes": ""
}
```"""

        brief = interviewer.extract_brief(response)

        assert brief is not None
        assert brief.premise == "A detective solves crimes"
        assert brief.genre == "Mystery"
        assert brief.target_length == "novella"

    def test_returns_none_for_no_json(self, interviewer):
        """Test returns None when no JSON found."""
        response = "Let me ask you a few more questions about your story..."

        brief = interviewer.extract_brief(response)

        assert brief is None

    def test_returns_none_for_invalid_json(self, interviewer):
        """Test returns None for malformed JSON."""
        response = '```json\n{"premise": "incomplete json'

        brief = interviewer.extract_brief(response)

        assert brief is None


class TestInterviewerFinalizeBrief:
    """Tests for finalize_brief method."""

    def test_generates_and_extracts_brief(self, interviewer):
        """Test generates brief from conversation summary."""
        mock_brief = StoryBrief(
            premise="An epic fantasy adventure",
            genre="Fantasy",
            subgenres=["Epic"],
            tone="Grand",
            themes=["Heroism"],
            setting_time="Medieval",
            setting_place="Magical Kingdom",
            target_length="novel",
            language="English",
            content_rating="general",
            content_preferences=[],
            content_avoid=[],
            additional_notes="",
        )
        interviewer.generate_structured = MagicMock(return_value=mock_brief)

        brief = interviewer.finalize_brief(
            "User wants fantasy, epic adventure, magic kingdom setting"
        )

        assert brief is not None
        assert brief.genre == "Fantasy"
        assert brief.target_length == "novel"
        interviewer.generate_structured.assert_called_once()

    def test_raises_on_generation_failure(self, interviewer):
        """Test raises LLMGenerationError when structured generation fails."""
        from utils.exceptions import LLMGenerationError

        interviewer.generate_structured = MagicMock(
            side_effect=LLMGenerationError("Validation failed")
        )

        with pytest.raises(LLMGenerationError, match="Validation failed"):
            interviewer.finalize_brief("Some conversation summary")


class TestInterviewerConversationFlow:
    """Integration tests for full conversation flow."""

    def test_full_interview_flow(self, interviewer):
        """Test complete interview conversation flow."""
        # First: Get initial questions
        interviewer.generate = MagicMock(return_value="Welcome! What's your story idea?")
        questions = interviewer.get_initial_questions()
        assert "story" in questions.lower()

        # Second: Process user response
        interviewer.generate = MagicMock(return_value="Interesting! Tell me about the characters.")
        interviewer.process_response("I want a sci-fi story about space explorers")
        assert len(interviewer.conversation_history) == 2

        # Third: Process follow-up
        interviewer.generate = MagicMock(return_value="Great details! Ready for the brief?")
        interviewer.process_response("The main character is a brave captain")
        assert len(interviewer.conversation_history) == 4

        # Fourth: Generate final brief using structured generation
        mock_brief = StoryBrief(
            premise="Space explorers discover alien life",
            genre="Science Fiction",
            subgenres=["Space Opera"],
            tone="Adventurous",
            themes=["Exploration", "First Contact"],
            setting_time="Future",
            setting_place="Deep Space",
            target_length="novella",
            language="English",
            content_rating="general",
            content_preferences=[],
            content_avoid=[],
            additional_notes="",
        )
        interviewer.generate_structured = MagicMock(return_value=mock_brief)
        brief = interviewer.finalize_brief(str(interviewer.conversation_history))

        assert brief.genre == "Science Fiction"
        assert "Space" in brief.setting_place
