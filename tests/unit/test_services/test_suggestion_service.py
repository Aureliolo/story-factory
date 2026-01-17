"""Tests for suggestion service."""

from unittest.mock import MagicMock, patch

import pytest

from memory.story_state import Chapter, Character, StoryBrief, StoryState
from services.suggestion_service import SuggestionService
from settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.get_model_for_agent.return_value = "test-model"
    settings.get_temperature_for_agent.return_value = 0.7
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_timeout = 30
    return settings


@pytest.fixture
def suggestion_service(mock_settings):
    """Create suggestion service with mock settings."""
    return SuggestionService(mock_settings)


@pytest.fixture
def sample_story_state():
    """Create a sample story state for testing."""
    state = StoryState(
        id="test-story",
        project_name="Test Story",
        brief=StoryBrief(
            premise="A detective investigates a mysterious murder in a futuristic city.",
            genre="Science Fiction Mystery",
            tone="Dark and suspenseful",
            setting_time="2150",
            setting_place="Neo-Tokyo",
            target_length="novella",
            content_rating="mature",
        ),
        characters=[
            Character(
                name="Detective Sarah Chen",
                role="protagonist",
                description="A brilliant but troubled detective",
                goals=["Solve the murder", "Clear her own name"],
            ),
            Character(
                name="Dr. Marcus Vale",
                role="antagonist",
                description="A rogue AI researcher",
                goals=["Complete the experiment"],
            ),
        ],
        chapters=[
            Chapter(
                number=1,
                title="The Body in the Lab",
                outline="Sarah discovers a murdered scientist in a high-tech laboratory.",
                content="The rain hammered against the windows as Detective Sarah Chen stepped into the lab...",
                status="drafting",
            )
        ],
        current_chapter=1,
    )
    return state


def test_build_context(suggestion_service, sample_story_state):
    """Test context building from story state."""
    context = suggestion_service._build_context(sample_story_state)

    # Verify key elements are included
    assert "A detective investigates a mysterious murder" in context
    assert "Science Fiction Mystery" in context
    assert "Dark and suspenseful" in context
    assert "Neo-Tokyo" in context
    assert "Detective Sarah Chen" in context
    assert "The Body in the Lab" in context


def test_build_context_minimal_state(suggestion_service):
    """Test context building with minimal story state."""
    minimal_state = StoryState(id="minimal", project_name="Minimal Story")
    context = suggestion_service._build_context(minimal_state)

    # Should not crash with empty state
    assert isinstance(context, str)


def test_fallback_suggestions_all_categories(suggestion_service, sample_story_state):
    """Test fallback suggestions for all categories."""
    categories = ["plot", "character", "scene", "transition"]
    suggestions = suggestion_service._fallback_suggestions(sample_story_state, categories)

    # Verify all categories are present
    assert set(suggestions.keys()) == set(categories)

    # Verify each category has suggestions
    for category in categories:
        assert len(suggestions[category]) > 0
        assert all(isinstance(s, str) for s in suggestions[category])


def test_fallback_suggestions_single_category(suggestion_service, sample_story_state):
    """Test fallback suggestions for single category."""
    suggestions = suggestion_service._fallback_suggestions(sample_story_state, ["plot"])

    assert "plot" in suggestions
    assert len(suggestions["plot"]) > 0


def test_fallback_suggestions_uses_character_names(suggestion_service, sample_story_state):
    """Test that fallback suggestions use actual character names."""
    suggestions = suggestion_service._fallback_suggestions(
        sample_story_state, ["plot", "character"]
    )

    # At least one suggestion should mention one of the character names
    all_suggestions = suggestions["plot"] + suggestions["character"]
    suggestion_text = " ".join(all_suggestions)

    assert "Detective Sarah Chen" in suggestion_text or "Dr. Marcus Vale" in suggestion_text


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_success(mock_agent_class, suggestion_service, sample_story_state):
    """Test successful suggestion generation."""
    # Mock agent response
    mock_agent = MagicMock()
    mock_agent.chat.return_value = """{
        "plot": ["What if Sarah discovers the killer is an AI?", "A second murder complicates the investigation."],
        "character": ["Show Sarah's personal struggles with the case.", "Reveal Dr. Vale's true motivations."],
        "scene": ["Build tension through environmental details.", "Use dialogue to reveal hidden agendas."],
        "transition": ["Jump to the next crime scene.", "End on a cliffhanger."]
    }"""
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Verify all categories are present
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions
    assert "transition" in suggestions

    # Verify suggestions are lists of strings
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) > 0


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_single_category(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test generating suggestions for a single category."""
    mock_agent = MagicMock()
    mock_agent.chat.return_value = (
        '["What if the detective becomes the suspect?", "An unexpected ally appears."]'
    )
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state, category="plot")

    # Should only have the requested category
    assert "plot" in suggestions
    assert isinstance(suggestions["plot"], list)


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_invalid_json(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test handling of invalid JSON response."""
    mock_agent = MagicMock()
    mock_agent.chat.return_value = "This is not valid JSON at all"
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should fall back to template suggestions
    assert "plot" in suggestions
    assert "character" in suggestions
    assert len(suggestions["plot"]) > 0


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_llm_error(mock_agent_class, suggestion_service, sample_story_state):
    """Test handling of LLM errors."""
    mock_agent = MagicMock()
    mock_agent.chat.side_effect = Exception("LLM connection failed")
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should fall back to template suggestions
    assert "plot" in suggestions
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) > 0


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_partial_response(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test handling of partial category response."""
    mock_agent = MagicMock()
    mock_agent.chat.return_value = (
        '{"plot": ["Good plot idea"], "character": ["Good character idea"]}'
    )
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should have all categories, using fallback for missing ones
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions  # Should be filled by fallback
    assert "transition" in suggestions  # Should be filled by fallback
