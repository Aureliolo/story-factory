"""Tests for suggestion service."""

from unittest.mock import MagicMock, patch

import pytest

from memory.story_state import Chapter, Character, PlotPoint, StoryBrief, StoryState
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
    mock_agent.generate.return_value = """{
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
def test_generate_suggestions_single_category_list_response(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test generating suggestions for a single category when LLM returns a list."""
    mock_agent = MagicMock()
    mock_agent.generate.return_value = (
        '["What if the detective becomes the suspect?", "An unexpected ally appears."]'
    )
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state, category="plot")

    # Should only have the requested category
    assert "plot" in suggestions
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) == 2


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_single_category_dict_response(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test generating suggestions for a single category when LLM returns a dict with category key."""
    mock_agent = MagicMock()
    mock_agent.generate.return_value = (
        '{"plot": ["A hidden conspiracy emerges.", "The suspect has an alibi."]}'
    )
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state, category="plot")

    # Should only have the requested category
    assert "plot" in suggestions
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) == 2
    assert "A hidden conspiracy emerges." in suggestions["plot"]


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_single_category_unexpected_structure(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test fallback when single category response has unexpected structure."""
    mock_agent = MagicMock()
    # Return a dict that doesn't contain the requested category
    mock_agent.generate.return_value = '{"character": ["Some character suggestion"]}'
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state, category="plot")

    # Should fall back to template suggestions for plot
    assert "plot" in suggestions
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) > 0


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_invalid_json(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test handling of invalid JSON response."""
    mock_agent = MagicMock()
    mock_agent.generate.return_value = "This is not valid JSON at all"
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
    mock_agent.generate.side_effect = Exception("LLM connection failed")
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
    mock_agent.generate.return_value = (
        '{"plot": ["Good plot idea"], "character": ["Good character idea"]}'
    )
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should have all categories, using fallback for missing ones
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions  # Should be filled by fallback
    assert "transition" in suggestions  # Should be filled by fallback


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_multi_category_non_dict_response(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test fallback when multi-category response is not a dict."""
    mock_agent = MagicMock()
    # Return a list instead of a dict for multi-category request
    mock_agent.generate.return_value = '["suggestion1", "suggestion2"]'
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should fall back to template suggestions for all categories
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions
    assert "transition" in suggestions
    assert isinstance(suggestions["plot"], list)
    assert len(suggestions["plot"]) > 0


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_invalid_category_value(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test fallback when a category value is not a list."""
    mock_agent = MagicMock()
    # Return a dict where one category has a non-list value
    mock_agent.generate.return_value = """{
        "plot": ["Valid plot suggestion"],
        "character": "This should be a list not a string",
        "scene": ["Valid scene suggestion"],
        "transition": null
    }"""
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Should have all categories
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions
    assert "transition" in suggestions

    # plot and scene should have the LLM values
    assert "Valid plot suggestion" in suggestions["plot"]
    assert "Valid scene suggestion" in suggestions["scene"]

    # character and transition should fall back to templates
    assert isinstance(suggestions["character"], list)
    assert isinstance(suggestions["transition"], list)


def test_build_context_with_plot_points(suggestion_service):
    """Test context building includes plot points."""
    state = StoryState(
        id="test",
        project_name="Test Story",
        plot_points=[
            PlotPoint(description="The hero discovers the truth", completed=False),
            PlotPoint(description="The villain is revealed", completed=True),
            PlotPoint(description="Final confrontation", completed=False),
        ],
    )

    context = suggestion_service._build_context(state)

    # Should include the first uncompleted plot point
    assert "The hero discovers the truth" in context
    # Should not include completed plot points
    assert "The villain is revealed" not in context


def test_build_context_with_established_facts(suggestion_service):
    """Test context building includes established facts."""
    state = StoryState(
        id="test",
        project_name="Test Story",
        established_facts=[
            "The city was founded in 2050",
            "AI gained sentience in 2080",
            "The detective's partner was killed",
            "The murder weapon was a laser pistol",
            "The suspect has an alibi",
            "There is a mole in the department",
        ],
    )

    context = suggestion_service._build_context(state)

    # Should include the most recent facts (last 5)
    assert "AI gained sentience in 2080" in context
    assert "The detective's partner was killed" in context
    assert "The murder weapon was a laser pistol" in context
    assert "The suspect has an alibi" in context
    assert "There is a mole in the department" in context
    # First fact should be excluded (only last 5)
    assert "The city was founded in 2050" not in context


def test_build_context_with_all_plot_points_completed(suggestion_service):
    """Test context building when all plot points are completed."""
    state = StoryState(
        id="test",
        project_name="Test Story",
        plot_points=[
            PlotPoint(description="Plot point 1", completed=True),
            PlotPoint(description="Plot point 2", completed=True),
        ],
    )

    context = suggestion_service._build_context(state)

    # Should not include plot points section when all are completed
    assert "Upcoming Plot Point" not in context


@patch("services.suggestion_service.BaseAgent")
def test_generate_suggestions_empty_json_response(
    mock_agent_class, suggestion_service, sample_story_state
):
    """Test fallback when extract_json returns empty/falsy value."""
    mock_agent = MagicMock()
    # Return something that extract_json will parse but return empty/falsy
    mock_agent.generate.return_value = "{}"
    mock_agent_class.return_value = mock_agent

    suggestions = suggestion_service.generate_suggestions(sample_story_state)

    # Empty dict is falsy, should fall back
    # Actually, empty dict {} is truthy in Python but will fail validation
    # Let's verify it still works
    assert "plot" in suggestions
    assert "character" in suggestions
    assert "scene" in suggestions
    assert "transition" in suggestions


class TestGenerateProjectNames:
    """Tests for generate_project_names method."""

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_success(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test successful project name generation."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = """[
            "Neon Shadows",
            "The Chen Files",
            "Circuit Breaker",
            "Synthwave Murder",
            "Digital Detective",
            "Chrome and Blood",
            "The Vale Conspiracy",
            "Neo-Tokyo Noir",
            "Silicon Requiem",
            "Ghost in the Lab"
        ]"""
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state, count=10)

        assert len(names) == 10
        assert "Neon Shadows" in names
        assert "The Chen Files" in names
        assert all(isinstance(n, str) for n in names)

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_limited_count(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test project name generation with limited count."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = """[
            "Title One",
            "Title Two",
            "Title Three",
            "Title Four",
            "Title Five"
        ]"""
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state, count=3)

        # Should return at most 'count' names
        assert len(names) <= 3

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_invalid_json(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test fallback when LLM returns invalid JSON."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "Not valid JSON at all"
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state)

        # Should return fallback suggestions
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_llm_error(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test fallback when LLM throws error."""
        mock_agent = MagicMock()
        mock_agent.generate.side_effect = Exception("LLM connection failed")
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state)

        # Should return fallback suggestions
        assert len(names) > 0

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_empty_response(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test fallback when LLM returns empty list."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "[]"
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state)

        # Should return fallback suggestions since empty list is not useful
        assert len(names) > 0

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_non_list_response(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test fallback when LLM returns non-list JSON."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = '{"title": "Single Title"}'
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state)

        # Should return fallback suggestions
        assert len(names) > 0

    def test_fallback_project_names(self, suggestion_service, sample_story_state):
        """Test fallback project names use story context."""
        names = suggestion_service._fallback_project_names(sample_story_state, count=5)

        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)
        # Should include genre-based names
        assert any("Science Fiction Mystery" in n for n in names)

    def test_fallback_project_names_minimal_state(self, suggestion_service):
        """Test fallback project names with minimal state."""
        minimal_state = StoryState(id="minimal", project_name="Minimal")
        names = suggestion_service._fallback_project_names(minimal_state, count=5)

        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_strips_whitespace(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test that generated names are stripped of whitespace."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = """[
            "  Title with leading spaces",
            "Title with trailing spaces   ",
            "  Both ends  "
        ]"""
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state, count=3)

        # Names should be stripped
        assert "Title with leading spaces" in names
        assert "Title with trailing spaces" in names
        assert "Both ends" in names

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_filters_empty(
        self, mock_agent_class, suggestion_service, sample_story_state
    ):
        """Test that empty names are filtered out."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = """[
            "Valid Title",
            "",
            "   ",
            "Another Valid Title",
            null
        ]"""
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(sample_story_state, count=10)

        # Should only contain non-empty names
        assert "Valid Title" in names
        assert "Another Valid Title" in names
        assert "" not in names
        assert "   " not in names

    @patch("services.suggestion_service.BaseAgent")
    def test_generate_project_names_with_plot_points(self, mock_agent_class, suggestion_service):
        """Test project name generation includes plot points in context."""
        state = StoryState(
            id="test-with-plot",
            project_name="Test Story",
            brief=StoryBrief(
                premise="A hero's journey",
                genre="Fantasy",
                tone="Epic",
                setting_time="Medieval",
                setting_place="Kingdom of Eldoria",
                target_length="novel",
                content_rating="general",
            ),
            plot_points=[
                PlotPoint(description="The hero finds the sword", completed=False),
                PlotPoint(description="The villain strikes", completed=False),
                PlotPoint(description="Final battle", completed=False),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.generate.return_value = '["The Sword Edge", "Villain Shadow"]'
        mock_agent_class.return_value = mock_agent

        names = suggestion_service.generate_project_names(state, count=2)

        assert len(names) == 2
        # Verify the prompt included plot points by checking the call
        call_args = mock_agent.generate.call_args[0][0]
        assert "Key Plot Points" in call_args
        assert "The hero finds the sword" in call_args
