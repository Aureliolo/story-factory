"""Tests for ArchitectAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.architect import ArchitectAgent
from memory.story_state import (
    Chapter,
    ChapterList,
    Character,
    CharacterList,
    PlotOutline,
    PlotPoint,
    StoryBrief,
    StoryState,
)
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

    def test_creates_characters_via_structured_generation(self, architect, sample_story_state):
        """Test creates characters using generate_structured."""
        mock_characters = CharacterList(
            characters=[
                Character(
                    name="Oliver Grey",
                    role="protagonist",
                    description="A curious 16-year-old with untapped magical potential",
                    personality_traits=["curious", "brave", "loyal"],
                    goals=["Master magic", "Find belonging"],
                    relationships={},
                    arc_notes="Grows from uncertain outsider to confident wizard",
                ),
                Character(
                    name="Professor Nightshade",
                    role="supporting",
                    description="Stern but caring headmaster of the academy",
                    personality_traits=["wise", "stern", "protective"],
                    goals=["Guide students", "Protect the academy"],
                    relationships={"Oliver Grey": "Mentor"},
                    arc_notes="Reveals hidden connection to Oliver's past",
                ),
                Character(
                    name="Luna Swift",
                    role="love_interest",
                    description="Quick-witted student with a mysterious past",
                    personality_traits=["clever", "secretive", "loyal"],
                    goals=["Uncover family secrets", "Master rare magic"],
                    relationships={"Oliver Grey": "Friend"},
                    arc_notes="Becomes Oliver's closest ally",
                ),
                Character(
                    name="Marcus Thorn",
                    role="antagonist",
                    description="Ambitious rival with hidden motives",
                    personality_traits=["ambitious", "cunning", "jealous"],
                    goals=["Become top student", "Gain power"],
                    relationships={"Oliver Grey": "Rival"},
                    arc_notes="Reveals deeper vulnerability",
                ),
            ]
        )
        architect.generate_structured = MagicMock(return_value=mock_characters)

        characters = architect.create_characters(sample_story_state)

        assert len(characters) == 4
        assert characters[0].name == "Oliver Grey"
        assert characters[0].role == "protagonist"
        assert "curious" in characters[0].personality_traits
        architect.generate_structured.assert_called_once()

    def test_raises_error_on_generation_failure(self, architect, sample_story_state):
        """Test raises error when structured generation fails."""
        from utils.exceptions import LLMGenerationError

        architect.generate_structured = MagicMock(
            side_effect=LLMGenerationError("Validation failed")
        )

        with pytest.raises(LLMGenerationError):
            architect.create_characters(sample_story_state)

    def test_raises_error_when_not_enough_characters(self, architect, sample_story_state):
        """Test raises LLMGenerationError when LLM can't generate minimum characters."""
        from utils.exceptions import LLMGenerationError

        # Mock returning only 2 characters (less than minimum of 4)
        mock_result = CharacterList(
            characters=[
                Character(name="A", role="protagonist", description="Test"),
                Character(name="B", role="antagonist", description="Test"),
            ]
        )
        architect.generate_structured = MagicMock(return_value=mock_result)

        with pytest.raises(LLMGenerationError, match="Failed to generate enough characters"):
            architect.create_characters(sample_story_state)

        # Should have retried 3 times
        assert architect.generate_structured.call_count == 3


class TestArchitectCreatePlotOutline:
    """Tests for create_plot_outline method."""

    def test_includes_world_preview_when_long(self, architect, sample_story_state):
        """Test includes truncated world preview when world description is long."""
        # Create a long world description (over 500 chars)
        long_world_desc = "A " * 300  # 600 chars
        sample_story_state.world_description = long_world_desc

        mock_result = PlotOutline(
            plot_summary="Plot summary here.",
            plot_points=[PlotPoint(description="Event 1", chapter=1)],
        )
        sample_story_state.characters = [
            Character(name="Hero", role="protagonist", description="Main char")
        ]
        architect.generate_structured = MagicMock(return_value=mock_result)

        architect.create_plot_outline(sample_story_state)

        # Verify the prompt includes truncated world with ellipsis
        call_args = architect.generate_structured.call_args[0][0]
        assert "WORLD" in call_args
        assert "..." in call_args

    def test_returns_summary_and_plot_points(self, architect, sample_story_state):
        """Test returns both plot summary and plot points."""
        mock_result = PlotOutline(
            plot_summary="Oliver discovers a mysterious letter inviting him to an academy he never knew existed. This sets him on a journey of self-discovery.",
            plot_points=[
                PlotPoint(description="Oliver receives the mysterious invitation", chapter=1),
                PlotPoint(description="First day at the magical academy", chapter=2),
                PlotPoint(description="Oliver discovers a hidden power within himself", chapter=4),
                PlotPoint(description="The final confrontation with the dark force", chapter=7),
                PlotPoint(description="Oliver accepts his role as protector", chapter=7),
            ],
        )
        sample_story_state.characters = [
            Character(name="Oliver Grey", role="protagonist", description="Young wizard")
        ]
        architect.generate_structured = MagicMock(return_value=mock_result)

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
        mock_result = ChapterList(
            chapters=[
                Chapter(number=1, title="The Letter", outline="Oliver receives the invitation"),
                Chapter(number=2, title="The Journey", outline="Travel to the hidden academy"),
                Chapter(number=3, title="First Day", outline="Introduction to the magical world"),
                Chapter(number=4, title="The Discovery", outline="Oliver finds his unique power"),
                Chapter(number=5, title="The Training", outline="Intensive magical training"),
                Chapter(number=6, title="The Revelation", outline="Dark secrets are revealed"),
                Chapter(number=7, title="The Final Test", outline="Oliver faces his challenge"),
            ]
        )
        sample_story_state.plot_summary = "An epic journey of discovery"
        sample_story_state.plot_points = [PlotPoint(description="Beginning", chapter=1)]
        sample_story_state.characters = [
            Character(name="Oliver", role="protagonist", description="Young wizard")
        ]
        architect.generate_structured = MagicMock(return_value=mock_result)

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 7
        assert chapters[0].title == "The Letter"
        assert chapters[6].number == 7

    def test_short_story_creates_one_chapter(self, architect, sample_story_state):
        """Test short story creates single chapter."""
        sample_story_state.brief.target_length = "short_story"
        mock_result = ChapterList(
            chapters=[
                Chapter(number=1, title="The Complete Story", outline="The entire narrative in one")
            ]
        )
        sample_story_state.plot_summary = "A complete story"
        sample_story_state.plot_points = []
        sample_story_state.characters = []
        architect.generate_structured = MagicMock(return_value=mock_result)

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 1

    def test_iteratively_generates_chapters_when_needed(self, architect, sample_story_state):
        """Test generates chapters iteratively if LLM returns fewer than needed."""
        sample_story_state.plot_summary = "An epic journey"
        sample_story_state.plot_points = [PlotPoint(description="Beginning")]
        sample_story_state.characters = [
            Character(name="Hero", role="protagonist", description="Main")
        ]

        # First call returns 3 chapters, second call returns 4 more
        first_result = ChapterList(
            chapters=[
                Chapter(number=1, title="Ch1", outline="First"),
                Chapter(number=2, title="Ch2", outline="Second"),
                Chapter(number=3, title="Ch3", outline="Third"),
            ]
        )
        second_result = ChapterList(
            chapters=[
                Chapter(number=1, title="Ch4", outline="Fourth"),
                Chapter(number=2, title="Ch5", outline="Fifth"),
                Chapter(number=3, title="Ch6", outline="Sixth"),
                Chapter(number=4, title="Ch7", outline="Seventh"),
            ]
        )
        architect.generate_structured = MagicMock(side_effect=[first_result, second_result])

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 7
        # Check that chapters are properly renumbered
        assert chapters[0].number == 1
        assert chapters[3].number == 4
        assert chapters[6].number == 7
        # Should have been called twice
        assert architect.generate_structured.call_count == 2

    def test_raises_error_when_not_enough_chapters(self, architect, sample_story_state):
        """Test raises LLMGenerationError when unable to generate enough chapters."""
        from utils.exceptions import LLMGenerationError

        sample_story_state.plot_summary = "An epic journey"
        sample_story_state.plot_points = [PlotPoint(description="Beginning")]
        sample_story_state.characters = [
            Character(name="Hero", role="protagonist", description="Main")
        ]

        # Always return empty chapters
        mock_result = ChapterList(chapters=[])
        architect.generate_structured = MagicMock(return_value=mock_result)

        with pytest.raises(LLMGenerationError, match="Failed to generate enough chapters"):
            architect.create_chapter_outline(sample_story_state)

    def test_handles_zero_chapters_iteration(self, architect, sample_story_state):
        """Test handles iteration where LLM returns 0 chapters and retries."""
        sample_story_state.brief.target_length = "short_story"  # Only needs 1 chapter
        sample_story_state.plot_summary = "A short story"
        sample_story_state.plot_points = []
        sample_story_state.characters = []

        # First call returns 0, second call returns 1
        empty_result = ChapterList(chapters=[])
        success_result = ChapterList(
            chapters=[Chapter(number=1, title="The Story", outline="Complete")]
        )
        architect.generate_structured = MagicMock(side_effect=[empty_result, success_result])

        chapters = architect.create_chapter_outline(sample_story_state)

        assert len(chapters) == 1
        assert architect.generate_structured.call_count == 2


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

    def test_generate_outline_variations_all_five(self, architect, sample_story_state):
        """Test generates all five variation types."""
        prompts_used = []

        def capture_generate(prompt):
            prompts_used.append(prompt)
            return """
            RATIONALE: Variation test.
            WORLD: Test world.
            CHARACTERS:
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

        architect.generate_outline_variations(sample_story_state, count=5)

        assert len(prompts_used) == 5
        # Check that all focus keywords appear (variations 0-4)
        assert any("Traditional" in p for p in prompts_used)  # i=0
        assert any("Non-linear" in p for p in prompts_used)  # i=1
        assert any("Fast-paced" in p for p in prompts_used)  # i=2
        assert any("Character-driven" in p for p in prompts_used)  # i=3
        assert any("Ensemble" in p for p in prompts_used)  # i=4

    def test_generate_outline_variations_count_out_of_range(self, architect, sample_story_state):
        """Test raises ValueError when count is out of valid range."""
        with pytest.raises(ValueError, match="count must be between"):
            architect.generate_outline_variations(sample_story_state, count=10)


class TestParseVariationResponseEdgeCases:
    """Tests for _parse_variation_response method edge cases and exception handling."""

    def test_character_parsing_failure_individual_item(self, architect, sample_story_state):
        """Test handles individual character parsing failure (lines 434-437)."""
        # Response has character JSON where one item is malformed
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [
            {"name": "Valid", "role": "protagonist", "description": "Good"},
            {"invalid_field_only": "This will fail Character validation"}
        ]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        [{"description": "Event", "chapter": 1}]
        ```

        CHAPTERS:
        ```json
        [{"number": 1, "title": "Test", "outline": "Test"}]
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have parsed the valid character, skipped the invalid one
        assert len(variation.characters) == 1
        assert variation.characters[0].name == "Valid"

    def test_character_extraction_failure_outer_exception(self, architect, sample_story_state):
        """Test handles outer character extraction failure (lines 436-437)."""
        # Response has completely broken character JSON that extract_json_list can't parse
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [{"this is not valid json at all: ]]]]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        [{"description": "Event", "chapter": 1}]
        ```

        CHAPTERS:
        ```json
        [{"number": 1, "title": "Test", "outline": "Test"}]
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have empty characters due to extraction failure
        assert len(variation.characters) == 0

    def test_plot_point_parsing_failure_individual_item(self, architect, sample_story_state):
        """Test handles individual plot point parsing failure (lines 462-465)."""
        # Response has plot points JSON where one item is malformed
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [{"name": "Hero", "role": "protagonist", "description": "Main"}]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        [
            {"description": "Valid event", "chapter": 1},
            {"missing_description_field": "This should fail PlotPoint validation"}
        ]
        ```

        CHAPTERS:
        ```json
        [{"number": 1, "title": "Test", "outline": "Test"}]
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have parsed the valid plot point, skipped the invalid one
        assert len(variation.plot_points) == 1
        assert variation.plot_points[0].description == "Valid event"

    def test_plot_point_extraction_failure_outer_exception(self, architect, sample_story_state):
        """Test handles outer plot point extraction failure (lines 464-465)."""
        # Response has broken plot points JSON
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [{"name": "Hero", "role": "protagonist", "description": "Main"}]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        not valid json at all {{{{
        ```

        CHAPTERS:
        ```json
        [{"number": 1, "title": "Test", "outline": "Test"}]
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have empty plot points due to extraction failure
        assert len(variation.plot_points) == 0

    def test_chapter_parsing_failure_individual_item(self, architect, sample_story_state):
        """Test handles individual chapter parsing failure (lines 482-485)."""
        # Response has chapters JSON where one item is malformed
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [{"name": "Hero", "role": "protagonist", "description": "Main"}]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        [{"description": "Event", "chapter": 1}]
        ```

        CHAPTERS:
        ```json
        [
            {"number": 1, "title": "Valid Chapter", "outline": "This is valid"},
            {"missing_required_fields": "This should fail Chapter validation"}
        ]
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have parsed the valid chapter, skipped the invalid one
        assert len(variation.chapters) == 1
        assert variation.chapters[0].title == "Valid Chapter"

    def test_chapter_extraction_failure_outer_exception(self, architect, sample_story_state):
        """Test handles outer chapter extraction failure (lines 484-485)."""
        # Response has broken chapters JSON
        response = """
        RATIONALE: Test variation.

        WORLD DESCRIPTION: A magical world.

        CHARACTERS:
        ```json
        [{"name": "Hero", "role": "protagonist", "description": "Main"}]
        ```

        PLOT SUMMARY: Test plot.

        PLOT POINTS:
        ```json
        [{"description": "Event", "chapter": 1}]
        ```

        CHAPTERS:
        ```json
        [[[[[not valid at all
        ```
        """

        brief = sample_story_state.brief
        variation = architect._parse_variation_response(response, 1, brief)

        # Should have empty chapters due to extraction failure
        assert len(variation.chapters) == 0


class TestGenerateMoreCharacters:
    """Tests for generate_more_characters method (lines 535-565)."""

    def test_generates_new_characters(self, architect, sample_story_state):
        """Test generates new characters complementing existing ones."""
        mock_result = CharacterList(
            characters=[
                Character(
                    name="Marcus Vale",
                    role="supporting",
                    description="A mysterious librarian with knowledge of ancient texts",
                    personality_traits=["secretive", "helpful", "wise"],
                    goals=["Protect forbidden knowledge", "Guide worthy students"],
                    relationships={"Oliver Grey": "Mentor figure"},
                    arc_notes="Reveals his true identity as a guardian",
                ),
                Character(
                    name="Luna Frost",
                    role="supporting",
                    description="A talented student with ice magic abilities",
                    personality_traits=["aloof", "competitive", "loyal"],
                    goals=["Master her powers", "Prove herself"],
                    relationships={"Oliver Grey": "Rival turned ally"},
                    arc_notes="Learns to trust others",
                ),
            ]
        )
        architect.generate_structured = MagicMock(return_value=mock_result)
        existing_names = ["Oliver Grey", "Professor Nightshade"]

        characters = architect.generate_more_characters(sample_story_state, existing_names, count=2)

        assert len(characters) == 2
        assert characters[0].name == "Marcus Vale"
        assert characters[1].name == "Luna Frost"
        # Verify prompt includes existing names to avoid duplicates
        call_args = architect.generate_structured.call_args[0][0]
        assert "Oliver Grey" in call_args
        assert "Professor Nightshade" in call_args
        assert "EXISTING CHARACTERS" in call_args

    def test_validates_story_state_not_none(self, architect):
        """Test raises error when story_state is None."""
        with pytest.raises(ValueError, match="story_state"):
            architect.generate_more_characters(None, ["Name"], count=2)

    def test_validates_story_state_type(self, architect):
        """Test raises error when story_state is wrong type."""
        with pytest.raises(TypeError, match="story_state"):
            architect.generate_more_characters("not a story state", ["Name"], count=2)

    def test_validates_existing_names_not_none(self, architect, sample_story_state):
        """Test raises error when existing_names is None."""
        with pytest.raises(ValueError, match="existing_names"):
            architect.generate_more_characters(sample_story_state, None, count=2)

    def test_validates_count_positive(self, architect, sample_story_state):
        """Test raises error when count is not positive."""
        with pytest.raises(ValueError, match="count"):
            architect.generate_more_characters(sample_story_state, ["Name"], count=0)

    def test_includes_language_requirement(self, architect, sample_story_state):
        """Test prompt includes language requirement from brief."""
        mock_result = CharacterList(
            characters=[
                Character(
                    name="Test",
                    role="supporting",
                    description="A test character",
                    personality_traits=["brave"],
                    goals=["survive"],
                    arc_notes="none",
                    relationships={},
                )
            ]
        )
        architect.generate_structured = MagicMock(return_value=mock_result)

        architect.generate_more_characters(sample_story_state, ["Hero"], count=1)

        call_args = architect.generate_structured.call_args[0][0]
        assert "English" in call_args


class TestGenerateLocations:
    """Tests for generate_locations method (lines 580-615)."""

    def test_generates_locations(self, architect, sample_story_state):
        """Test generates location dictionaries."""
        json_response = """Here are the key locations:
```json
[
    {
        "name": "The Hidden Library",
        "type": "location",
        "description": "An ancient library concealed within the academy's walls, accessible only through a secret passage.",
        "significance": "Contains forbidden texts and serves as a meeting place for secret societies."
    },
    {
        "name": "The Crystal Cavern",
        "type": "location",
        "description": "A vast underground cave filled with glowing crystals that amplify magical energy.",
        "significance": "Students must pass a trial here to unlock their full potential."
    },
    {
        "name": "The Twilight Tower",
        "type": "location",
        "description": "The tallest tower in the academy, where time seems to move differently.",
        "significance": "The headmaster's private quarters and site of the final confrontation."
    }
]
```"""
        architect.generate = MagicMock(return_value=json_response)

        locations = architect.generate_locations(sample_story_state, existing_locations=[], count=3)

        assert len(locations) == 3
        assert locations[0]["name"] == "The Hidden Library"
        assert locations[1]["name"] == "The Crystal Cavern"
        assert locations[2]["name"] == "The Twilight Tower"

    def test_includes_world_description_when_present(self, architect, sample_story_state):
        """Test includes world description preview in prompt."""
        sample_story_state.world_description = "A magical academy hidden in the mountains..."
        architect.generate = MagicMock(return_value="```json\n[]\n```")

        architect.generate_locations(sample_story_state, [], count=2)

        call_args = architect.generate.call_args[0][0]
        assert "WORLD" in call_args
        assert "magical academy" in call_args

    def test_includes_existing_locations_when_provided(self, architect, sample_story_state):
        """Test includes existing locations to avoid duplicates."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")
        existing = ["The Great Hall", "The Dungeon"]

        architect.generate_locations(sample_story_state, existing, count=2)

        call_args = architect.generate.call_args[0][0]
        assert "EXISTING LOCATIONS" in call_args
        assert "The Great Hall" in call_args
        assert "The Dungeon" in call_args

    def test_returns_empty_list_on_parse_failure(self, architect, sample_story_state):
        """Test returns empty list when JSON parsing fails."""
        architect.generate = MagicMock(return_value="No valid JSON here at all")

        locations = architect.generate_locations(sample_story_state, [], count=2)

        assert locations == []

    def test_handles_no_existing_locations(self, architect, sample_story_state):
        """Test works correctly when no existing locations provided."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")

        architect.generate_locations(sample_story_state, [], count=2)

        call_args = architect.generate.call_args[0][0]
        # Should NOT include "EXISTING LOCATIONS" section when empty
        assert "EXISTING LOCATIONS" not in call_args

    def test_handles_no_world_description(self, architect, sample_story_state):
        """Test works correctly when no world description present."""
        sample_story_state.world_description = ""
        architect.generate = MagicMock(return_value="```json\n[]\n```")

        architect.generate_locations(sample_story_state, [], count=2)

        call_args = architect.generate.call_args[0][0]
        # Should NOT include "WORLD" section when empty
        # The word WORLD should still appear in the premise about "world"
        # but the section header pattern "WORLD:" should not appear
        assert "WORLD:" not in call_args


class TestGenerateRelationships:
    """Tests for generate_relationships method (lines 635-667)."""

    def test_generates_relationships(self, architect, sample_story_state):
        """Test generates relationship dictionaries."""
        json_response = """Here are the relationships:
```json
[
    {
        "source": "Oliver Grey",
        "target": "Professor Nightshade",
        "relation_type": "knows",
        "description": "Professor is secretly Oliver's guardian"
    },
    {
        "source": "Oliver Grey",
        "target": "Luna Frost",
        "relation_type": "allies_with",
        "description": "Former rivals who became close friends"
    },
    {
        "source": "Luna Frost",
        "target": "Marcus Vale",
        "relation_type": "hates",
        "description": "Suspects him of dark intentions"
    }
]
```"""
        architect.generate = MagicMock(return_value=json_response)
        entity_names = ["Oliver Grey", "Professor Nightshade", "Luna Frost", "Marcus Vale"]

        relationships = architect.generate_relationships(
            sample_story_state,
            entity_names,
            existing_relationships=[],
            count=3,
        )

        assert len(relationships) == 3
        assert relationships[0]["source"] == "Oliver Grey"
        assert relationships[0]["target"] == "Professor Nightshade"
        assert relationships[1]["relation_type"] == "allies_with"

    def test_includes_existing_relationships(self, architect, sample_story_state):
        """Test includes existing relationships to avoid duplicates."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")
        entity_names = ["A", "B", "C"]
        existing = [("A", "B"), ("B", "C")]

        architect.generate_relationships(sample_story_state, entity_names, existing, count=2)

        call_args = architect.generate.call_args[0][0]
        assert "EXISTING RELATIONSHIPS" in call_args
        assert "A → B" in call_args
        assert "B → C" in call_args

    def test_handles_empty_existing_relationships(self, architect, sample_story_state):
        """Test works correctly with no existing relationships."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")
        entity_names = ["Character1", "Character2"]

        architect.generate_relationships(sample_story_state, entity_names, [], count=2)

        call_args = architect.generate.call_args[0][0]
        # Should NOT include "EXISTING RELATIONSHIPS" section when empty
        assert "EXISTING RELATIONSHIPS" not in call_args

    def test_returns_empty_list_on_parse_failure(self, architect, sample_story_state):
        """Test returns empty list when JSON parsing fails."""
        architect.generate = MagicMock(return_value="No valid JSON at all")
        entity_names = ["A", "B"]

        relationships = architect.generate_relationships(
            sample_story_state, entity_names, [], count=2
        )

        assert relationships == []

    def test_includes_entity_names_in_prompt(self, architect, sample_story_state):
        """Test prompt includes all available entity names."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")
        entity_names = ["Hero", "Villain", "Mentor", "Sidekick"]

        architect.generate_relationships(sample_story_state, entity_names, [], count=3)

        call_args = architect.generate.call_args[0][0]
        assert "AVAILABLE ENTITIES" in call_args
        assert "Hero" in call_args
        assert "Villain" in call_args
        assert "Mentor" in call_args
        assert "Sidekick" in call_args

    def test_limits_existing_relationships_display(self, architect, sample_story_state):
        """Test limits existing relationships display to 20."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")
        entity_names = ["A", "B"]
        # Create 25 existing relationships
        existing = [(f"Entity{i}", f"Entity{i + 1}") for i in range(25)]

        architect.generate_relationships(sample_story_state, entity_names, existing, count=2)

        call_args = architect.generate.call_args[0][0]
        # Should only show first 20 relationships
        assert "Entity0 → Entity1" in call_args
        assert "Entity19 → Entity20" in call_args
        # Entity24 → Entity25 should NOT be in prompt (it's the 25th, 0-indexed 24)
        assert "Entity24 → Entity25" not in call_args

    def test_includes_language_requirement(self, architect, sample_story_state):
        """Test prompt includes language requirement from brief."""
        architect.generate = MagicMock(return_value="```json\n[]\n```")

        architect.generate_relationships(sample_story_state, ["A", "B"], [], count=2)

        call_args = architect.generate.call_args[0][0]
        assert "English" in call_args
