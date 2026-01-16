"""Tests for ArchitectAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.architect import ArchitectAgent
from memory.story_state import Chapter, Character, PlotPoint, StoryBrief, StoryState
from settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def architect(settings):
    """Create ArchitectAgent with mocked Ollama client."""
    with patch("agents.base.ollama.Client"):
        agent = ArchitectAgent(model="test-model", settings=settings)
        return agent


@pytest.fixture
def sample_story_state():
    """Create a sample story state for testing."""
    brief = StoryBrief(
        premise="A young wizard discovers a hidden magical academy",
        genre="Fantasy",
        subgenres=["Coming of Age", "Adventure"],
        tone="Whimsical",
        themes=["Self-discovery", "Friendship"],
        setting_time="Modern Day",
        setting_place="England",
        target_length="novella",
        language="English",
        content_rating="general",
        content_preferences=["Magic", "School Setting"],
        content_avoid=["Gore"],
    )
    return StoryState(
        id="test-story-001",
        project_name="Test Story",
        brief=brief,
        status="architect",
    )


class TestArchitectAgentInit:
    """Tests for ArchitectAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("agents.base.ollama.Client"):
            agent = ArchitectAgent(settings=settings)
            assert agent.name == "Architect"
            assert agent.role == "Story Structure Designer"

    def test_has_schema_constants(self, architect):
        """Test agent has JSON schema constants."""
        assert isinstance(architect.CHARACTER_SCHEMA, str) and len(architect.CHARACTER_SCHEMA) > 0
        assert isinstance(architect.PLOT_POINT_SCHEMA, str) and len(architect.PLOT_POINT_SCHEMA) > 0
        assert isinstance(architect.CHAPTER_SCHEMA, str) and len(architect.CHAPTER_SCHEMA) > 0


class TestArchitectCreateWorld:
    """Tests for create_world method."""

    def test_generates_world_description(self, architect, sample_story_state):
        """Test generates world-building content."""
        world_desc = """The magical academy exists in a hidden dimension accessible only through specific portals.

Key rules of this world:
- Magic manifests as visible auras around practitioners
- Each student has a unique magical affinity
- The academy grounds shift and change with the seasons

The atmosphere is one of wonder and mystery, with ancient stone corridors and floating candles illuminating the halls."""

        architect.generate = MagicMock(return_value=world_desc)

        result = architect.create_world(sample_story_state)

        assert "magical" in result.lower() or "academy" in result.lower()
        architect.generate.assert_called_once()
        # Verify prompt includes key elements
        call_args = architect.generate.call_args[0][0]
        assert "Fantasy" in call_args
        assert "England" in call_args

    def test_raises_without_brief(self, architect):
        """Test raises error when brief is missing."""
        state = StoryState(id="test", status="architect")

        with pytest.raises(ValueError, match="brief"):
            architect.create_world(state)


class TestArchitectCreateCharacters:
    """Tests for create_characters method."""

    def test_parses_character_json(self, architect, sample_story_state):
        """Test parses character JSON from response."""
        json_response = """Here are the characters:
```json
[
    {
        "name": "Oliver Grey",
        "role": "protagonist",
        "description": "A curious 16-year-old with untapped magical potential",
        "personality_traits": ["curious", "brave", "loyal"],
        "goals": ["Master magic", "Find belonging"],
        "relationships": {},
        "arc_notes": "Grows from uncertain outsider to confident wizard"
    },
    {
        "name": "Professor Nightshade",
        "role": "supporting",
        "description": "Stern but caring headmaster of the academy",
        "personality_traits": ["wise", "stern", "protective"],
        "goals": ["Guide students", "Protect the academy"],
        "relationships": {"Oliver Grey": "Mentor"},
        "arc_notes": "Reveals hidden connection to Oliver's past"
    }
]
```"""
        architect.generate = MagicMock(return_value=json_response)

        characters = architect.create_characters(sample_story_state)

        assert len(characters) == 2
        assert characters[0].name == "Oliver Grey"
        assert characters[0].role == "protagonist"
        assert "curious" in characters[0].personality_traits

    def test_returns_empty_list_on_parse_failure(self, architect, sample_story_state):
        """Test returns empty list when JSON parsing fails."""
        architect.generate = MagicMock(return_value="No valid JSON here")

        characters = architect.create_characters(sample_story_state)

        assert characters == []


class TestArchitectCreatePlotOutline:
    """Tests for create_plot_outline method."""

    def test_includes_world_preview_when_long(self, architect, sample_story_state):
        """Test includes truncated world preview when world description is long."""
        # Create a long world description (over 500 chars)
        long_world_desc = "A " * 300  # 600 chars
        sample_story_state.world_description = long_world_desc

        response = """Plot summary here.

```json
[{"description": "Event 1", "chapter": 1}]
```"""
        sample_story_state.characters = [
            Character(name="Hero", role="protagonist", description="Main char")
        ]
        architect.generate = MagicMock(return_value=response)

        architect.create_plot_outline(sample_story_state)

        # Verify the prompt includes truncated world with ellipsis
        call_args = architect.generate.call_args[0][0]
        assert "WORLD" in call_args
        assert "..." in call_args

    def test_returns_summary_and_plot_points(self, architect, sample_story_state):
        """Test returns both plot summary and plot points."""
        response = """Oliver discovers a mysterious letter inviting him to an academy he never knew existed. This sets him on a journey of self-discovery where he must confront his fears and embrace his true nature.

```json
[
    {"description": "Oliver receives the mysterious invitation", "chapter": 1},
    {"description": "First day at the magical academy", "chapter": 2},
    {"description": "Oliver discovers a hidden power within himself", "chapter": 4},
    {"description": "The final confrontation with the dark force", "chapter": 7},
    {"description": "Oliver accepts his role as protector", "chapter": 7}
]
```"""
        sample_story_state.characters = [
            Character(name="Oliver Grey", role="protagonist", description="Young wizard")
        ]
        architect.generate = MagicMock(return_value=response)

        summary, plot_points = architect.create_plot_outline(sample_story_state)

        assert "Oliver" in summary or "academy" in summary.lower()
        assert len(plot_points) == 5
        assert plot_points[0].description == "Oliver receives the mysterious invitation"
        assert plot_points[0].chapter == 1


class TestArchitectCreateChapterOutline:
    """Tests for create_chapter_outline method."""

    def test_creates_correct_number_of_chapters(self, architect, sample_story_state):
        """Test creates chapters based on target length."""
        # Novella should create 7 chapters
        json_response = """```json
[
    {"number": 1, "title": "The Letter", "outline": "Oliver receives the mysterious invitation"},
    {"number": 2, "title": "The Journey", "outline": "Travel to the hidden academy"},
    {"number": 3, "title": "First Day", "outline": "Introduction to the magical world"},
    {"number": 4, "title": "The Discovery", "outline": "Oliver finds his unique power"},
    {"number": 5, "title": "The Training", "outline": "Intensive magical training begins"},
    {"number": 6, "title": "The Revelation", "outline": "Dark secrets are revealed"},
    {"number": 7, "title": "The Final Test", "outline": "Oliver faces his ultimate challenge"}
]
```"""
        sample_story_state.plot_summary = "An epic journey of discovery"
        sample_story_state.plot_points = [PlotPoint(description="Beginning", chapter=1)]
        sample_story_state.characters = [
            Character(name="Oliver", role="protagonist", description="Young wizard")
        ]
        architect.generate = MagicMock(return_value=json_response)

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 7
        assert chapters[0].title == "The Letter"
        assert chapters[6].number == 7

    def test_short_story_creates_one_chapter(self, architect, sample_story_state):
        """Test short story creates single chapter."""
        sample_story_state.brief.target_length = "short_story"
        json_response = """```json
[
    {"number": 1, "title": "The Complete Story", "outline": "The entire narrative in one chapter"}
]
```"""
        sample_story_state.plot_summary = "A complete story"
        sample_story_state.plot_points = []
        sample_story_state.characters = []
        architect.generate = MagicMock(return_value=json_response)

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 1


class TestArchitectBuildStoryStructure:
    """Tests for build_story_structure method."""

    def test_builds_complete_structure(self, architect, sample_story_state):
        """Test builds complete story structure with all components."""
        # Mock all sub-methods
        architect.create_world = MagicMock(return_value="A magical world...\n- Rule 1\n- Rule 2")
        architect.create_characters = MagicMock(
            return_value=[
                Character(name="Hero", role="protagonist", description="The main character")
            ]
        )
        architect.create_plot_outline = MagicMock(
            return_value=(
                "The hero's journey begins",
                [PlotPoint(description="Beginning", chapter=1)],
            )
        )
        architect.create_chapter_outline = MagicMock(
            return_value=[Chapter(number=1, title="Chapter 1", outline="The beginning")]
        )

        result = architect.build_story_structure(sample_story_state)

        assert result.world_description == "A magical world...\n- Rule 1\n- Rule 2"
        assert len(result.characters) == 1
        assert result.plot_summary == "The hero's journey begins"
        assert len(result.chapters) == 1
        assert result.status == "writing"

    def test_extracts_world_rules(self, architect, sample_story_state):
        """Test extracts world rules from description."""
        world_response = """The world of magic...

Key rules:
- Magic requires spoken words
- Only those with the gift can see magical creatures
- The veil between worlds is thin at midnight

The atmosphere is mysterious."""

        architect.create_world = MagicMock(return_value=world_response)
        architect.create_characters = MagicMock(return_value=[])
        architect.create_plot_outline = MagicMock(return_value=("Summary", []))
        architect.create_chapter_outline = MagicMock(return_value=[])

        result = architect.build_story_structure(sample_story_state)

        assert len(result.world_rules) > 0
        assert any("Magic" in rule or "magic" in rule for rule in result.world_rules)


class TestArchitectGenerateVariations:
    """Tests for generate_outline_variations method."""

    def test_generates_multiple_variations(self, architect, sample_story_state):
        """Test generates requested number of variations."""

        # Mock the generate method to return variation responses
        def mock_generate(prompt):
            return """
            RATIONALE: This variation focuses on character-driven narrative.

            WORLD DESCRIPTION: A modern magical academy hidden in London.

            KEY RULES:
            - Magic is hereditary
            - Students sorted by affinity

            ```json
            [
                {"name": "Alex", "role": "protagonist", "description": "Student",
                 "personality_traits": ["curious"], "goals": ["master magic"],
                 "relationships": {}, "arc_notes": "Growth"}
            ]
            ```

            PLOT SUMMARY: Alex discovers their magical heritage and attends academy.

            ```json
            [
                {"description": "Discovery", "chapter": 1}
            ]
            ```

            ```json
            [
                {"number": 1, "title": "Discovery", "outline": "Alex finds letter"}
            ]
            ```
            """

        architect.generate = MagicMock(side_effect=mock_generate)

        variations = architect.generate_outline_variations(sample_story_state, count=3)

        assert len(variations) == 3
        assert all(hasattr(v, "id") for v in variations)
        assert all(hasattr(v, "name") for v in variations)
        assert all(hasattr(v, "ai_rationale") for v in variations)

    def test_variation_contains_structure(self, architect, sample_story_state):
        """Test each variation contains complete structure."""

        def mock_generate(prompt):
            return """
            RATIONALE: Fast-paced action variant.

            WORLD: Modern city with hidden magic.

            CHARACTERS:
            ```json
            [
                {"name": "Hero", "role": "protagonist", "description": "Fighter",
                 "personality_traits": ["brave"], "goals": ["save world"],
                 "relationships": {}, "arc_notes": "Becomes leader"}
            ]
            ```

            PLOT: Hero fights dark forces.

            PLOT POINTS:
            ```json
            [{"description": "First battle", "chapter": 1}]
            ```

            CHAPTERS:
            ```json
            [{"number": 1, "title": "Battle", "outline": "Fight begins"}]
            ```
            """

        architect.generate = MagicMock(side_effect=mock_generate)

        variations = architect.generate_outline_variations(sample_story_state, count=3)

        assert len(variations) == 3
        var = variations[0]
        assert var.world_description != ""
        assert len(var.characters) > 0
        assert var.plot_summary != ""
        assert len(var.plot_points) > 0
        assert len(var.chapters) > 0

    def test_variations_have_different_focus(self, architect, sample_story_state):
        """Test variations get different focus prompts."""
        prompts_used = []

        def capture_generate(prompt):
            prompts_used.append(prompt)
            return """
            RATIONALE: Test variation.
            WORLD: Test world.
            ```json
            [{"name": "T", "role": "protagonist", "description": "Test",
               "personality_traits": [], "goals": [], "relationships": {}, "arc_notes": ""}]
            ```
            PLOT: Test.
            ```json
            [{"description": "Test", "chapter": 1}]
            ```
            ```json
            [{"number": 1, "title": "Test", "outline": "Test"}]
            ```
            """

        architect.generate = MagicMock(side_effect=capture_generate)

        architect.generate_outline_variations(sample_story_state, count=3)

        assert len(prompts_used) == 3
        # Check that different focus keywords appear
        assert any("Traditional" in p for p in prompts_used)
        assert any("Non-linear" in p for p in prompts_used)
        assert any("Fast-paced" in p for p in prompts_used)
