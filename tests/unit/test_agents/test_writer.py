"""Tests for WriterAgent."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.writer import WriterAgent
from src.memory.story_state import Chapter, Character, Scene, StoryBrief, StoryState
from src.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def writer(settings):
    """Create WriterAgent with mocked Ollama client."""
    with patch("src.agents.base.ollama.Client"):
        agent = WriterAgent(model="test-model", settings=settings)
        return agent


@pytest.fixture
def sample_story_state():
    """Create a sample story state with chapters."""
    brief = StoryBrief(
        premise="A detective investigates a series of mysterious disappearances",
        genre="Mystery",
        subgenres=["Noir", "Thriller"],
        tone="Dark and suspenseful",
        themes=["Justice", "Truth"],
        setting_time="1940s",
        setting_place="Los Angeles",
        target_length="novella",
        language="English",
        content_rating="mature",
        content_preferences=["Suspense", "Plot twists"],
        content_avoid=["Gore"],
    )
    state = StoryState(
        id="test-story-001",
        project_name="The Missing",
        brief=brief,
        status="writing",
        characters=[
            Character(
                name="Jack Stone",
                role="protagonist",
                description="Grizzled private detective with a troubled past",
                personality_traits=["cynical", "determined", "honorable"],
                goals=["Find the truth", "Redemption"],
            ),
            Character(
                name="Vera Lane",
                role="love_interest",
                description="Mysterious woman who hires Jack",
                personality_traits=["enigmatic", "intelligent", "secretive"],
                goals=["Find her sister"],
            ),
        ],
        chapters=[
            Chapter(number=1, title="The Client", outline="Vera walks into Jack's office"),
            Chapter(number=2, title="First Lead", outline="Jack follows the first clue"),
            Chapter(number=3, title="The Truth", outline="The mystery is solved"),
        ],
    )
    state.plot_summary = (
        "A detective uncovers a dark conspiracy while searching for missing persons."
    )
    return state


class TestWriterAgentInit:
    """Tests for WriterAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("src.agents.base.ollama.Client"):
            agent = WriterAgent(settings=settings)
            assert agent.name == "Writer"
            assert agent.role == "Prose Craftsman"

    def test_init_with_custom_model(self, settings):
        """Test agent initializes with custom model."""
        with patch("src.agents.base.ollama.Client"):
            agent = WriterAgent(model="creative-model:7b", settings=settings)
            assert agent.model == "creative-model:7b"


class TestWriterWriteChapter:
    """Tests for write_chapter method."""

    def test_generates_chapter_content(self, writer, sample_story_state):
        """Test generates chapter prose."""
        chapter_content = """The rain hammered against the window of Jack's cramped office on Third Street. The neon sign outside flickered, casting an intermittent red glow across the worn floorboards.

The door creaked open, and she walked in. Vera Lane, with eyes that held secrets and a dress that held promises. She placed a photograph on his desk—a young woman, barely twenty, with the same haunting eyes.

"Find her," she said, her voice barely above a whisper. "Find my sister."

Jack picked up the photograph, studied it for a long moment. Something about this case felt different. Wrong.

"I'll need a retainer," he said, reaching for his cigarette case.

She slid an envelope across the desk. It was thick with bills.

"Money isn't the problem, Mr. Stone. Time is."
"""
        writer.generate = MagicMock(return_value=chapter_content)

        chapter = sample_story_state.chapters[0]
        result = writer.write_chapter(sample_story_state, chapter)

        assert len(result) > 100
        writer.generate.assert_called_once()
        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "Chapter 1" in prompt or "The Client" in prompt
        assert "English" in prompt

    def test_uses_revision_temperature_for_revisions(self, writer, sample_story_state):
        """Test uses lower temperature when revising."""
        writer.generate = MagicMock(return_value="Revised chapter content...")

        chapter = sample_story_state.chapters[0]
        writer.write_chapter(
            sample_story_state, chapter, revision_feedback="Make the dialogue more noir-style"
        )

        call_args = writer.generate.call_args
        assert call_args[1].get("temperature") == writer.settings.revision_temperature

    def test_includes_previous_chapter_context(self, writer, sample_story_state):
        """Test includes previous chapter content for continuity."""
        # Add content to first chapter
        sample_story_state.chapters[0].content = "Jack lit a cigarette and stared at the rain..."
        writer.generate = MagicMock(return_value="Chapter 2 content...")

        chapter = sample_story_state.chapters[1]
        writer.write_chapter(sample_story_state, chapter)

        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "PREVIOUS CHAPTER" in prompt or "Chapter 1" in prompt.upper() or "rain" in prompt

    def test_raises_without_brief(self, writer):
        """Test raises error when brief is missing."""
        state = StoryState(
            id="test",
            status="writing",
            chapters=[Chapter(number=1, title="Test", outline="Test outline")],
        )

        with pytest.raises(ValueError, match="brief"):
            writer.write_chapter(state, state.chapters[0])


class TestWriterWriteShortStory:
    """Tests for write_short_story method."""

    def test_generates_complete_short_story(self, writer, sample_story_state):
        """Test generates a complete short story."""
        short_story = """The case started like they all do—with a dame and a problem.

Vera Lane walked into my office on a Tuesday, the kind of day when the fog rolled in off the Pacific and settled over the city like a burial shroud. She had legs that went all the way up and eyes that had seen things no woman should see.

"My sister disappeared three weeks ago," she said, her voice steady despite the trembling in her hands. "The police won't help. They say she ran off with some sailor."

I took the case. I shouldn't have. But something in those eyes told me she was telling the truth.

Three days later, I found myself in a warehouse on the docks, surrounded by men who didn't appreciate my questions. But the truth has a way of surfacing, like a body in the harbor.

Her sister was alive. Running from the same people who had killed their father—a cop who got too close to the truth about who really ran this city.

I got them both out. It cost me a black eye and two cracked ribs, but some things are worth the price.

When Vera thanked me, there were tears in her eyes. "How can I ever repay you?"

"The retainer covered it," I said, lighting a cigarette. "That's how this works."

She smiled then, a real smile, and walked out of my office and out of my life.

Another case closed. Another truth dragged into the light.

The rain started again as I poured myself a drink. In this city, the rain never really stops.
"""
        writer.generate = MagicMock(return_value=short_story)

        result = writer.write_short_story(sample_story_state)

        assert len(result) > 500
        writer.generate.assert_called_once()
        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "short story" in prompt.lower()

    def test_uses_revision_feedback(self, writer, sample_story_state):
        """Test incorporates revision feedback."""
        writer.generate = MagicMock(return_value="Revised story...")

        writer.write_short_story(
            sample_story_state, revision_feedback="Add more atmosphere and sensory details"
        )

        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "atmosphere" in prompt.lower() or "REVISION" in prompt


class TestWriterContinueScene:
    """Tests for continue_scene method."""

    def test_continues_from_existing_text(self, writer, sample_story_state):
        """Test continues scene from existing text."""
        existing_text = """Jack followed the trail to an abandoned warehouse on Fifth Street. The door hung off its hinges, creaking in the wind.

He drew his gun and stepped inside. The darkness was absolute."""

        continuation = """His eyes adjusted slowly. Crates lined the walls, covered in dust and cobwebs. Something moved in the shadows.

"Stone?" A voice echoed through the empty space. "I've been expecting you."

Jack's finger tightened on the trigger. "Show yourself."

A figure emerged from behind the crates. Not the killer he expected, but someone far more dangerous—Chief Bradley, badge gleaming even in the dim light."""

        writer.generate = MagicMock(return_value=continuation)

        result = writer.continue_scene(sample_story_state, existing_text)

        assert len(result) > 100
        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "warehouse" in prompt.lower() or "darkness" in prompt.lower()

    def test_uses_direction_when_provided(self, writer, sample_story_state):
        """Test uses direction hint for scene continuation."""
        writer.generate = MagicMock(return_value="Continued with action...")

        writer.continue_scene(
            sample_story_state,
            "Jack stood at the edge of the roof...",
            direction="Add a chase sequence",
        )

        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "DIRECTION" in prompt or "chase" in prompt.lower()

    def test_continues_naturally_without_direction(self, writer, sample_story_state):
        """Test continues naturally when no direction provided."""
        writer.generate = MagicMock(return_value="Natural continuation...")

        writer.continue_scene(sample_story_state, "The phone rang, cutting through the silence...")

        call_args = writer.generate.call_args
        prompt = call_args[0][0]
        assert "naturally" in prompt.lower() or "next beat" in prompt.lower()


class TestWriterSceneAware:
    """Tests for scene-aware writing functionality."""

    @pytest.fixture
    def chapter_with_scenes(self):
        """Create a chapter with scene structure."""
        return Chapter(
            number=1,
            title="The Investigation Begins",
            outline="Jack investigates the disappearance",
            scenes=[
                Scene(
                    id="scene-1",
                    title="Meeting the Client",
                    goal="Introduce Vera and establish the mystery",
                    pov_character="Jack Stone",
                    location="Jack's office",
                    beats=[
                        "Vera enters the office",
                        "She shows a photo of her sister",
                        "Jack accepts the case",
                    ],
                    order=0,
                ),
                Scene(
                    id="scene-2",
                    title="First Clue",
                    goal="Jack finds the first lead",
                    pov_character="Jack Stone",
                    location="Sister's apartment",
                    beats=["Jack searches the apartment", "Discovers a hidden letter"],
                    order=1,
                ),
            ],
        )

    def test_detects_scene_structure(self, writer, sample_story_state, chapter_with_scenes):
        """Test that writer detects when scenes are defined."""
        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # Should be called once per scene
        assert writer.generate.call_count == 2

    def test_writes_scenes_sequentially(self, writer, sample_story_state, chapter_with_scenes):
        """Test that scenes are written in order."""
        scene_contents = ["First scene content...", "Second scene content..."]
        writer.generate = MagicMock(side_effect=scene_contents)

        result = writer.write_chapter(sample_story_state, chapter_with_scenes)

        assert "First scene content" in result
        assert "Second scene content" in result
        # Scenes should be separated by double newlines
        assert "\n\n" in result

    def test_includes_scene_goals_in_prompt(self, writer, sample_story_state, chapter_with_scenes):
        """Test that scene goals are included in generation prompts."""
        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # Check first scene call
        first_call_prompt = writer.generate.call_args_list[0][0][0]
        assert "Introduce Vera" in first_call_prompt or "SCENE GOAL" in first_call_prompt

    def test_includes_scene_beats_in_prompt(self, writer, sample_story_state, chapter_with_scenes):
        """Test that scene beats are included in generation prompts."""
        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # Check first scene call
        first_call_prompt = writer.generate.call_args_list[0][0][0]
        assert "Vera enters" in first_call_prompt or "KEY BEATS" in first_call_prompt

    def test_includes_scene_pov_in_prompt(self, writer, sample_story_state, chapter_with_scenes):
        """Test that POV character is included in prompts."""
        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        first_call_prompt = writer.generate.call_args_list[0][0][0]
        assert "Jack Stone" in first_call_prompt or "POV CHARACTER" in first_call_prompt

    def test_includes_scene_location_in_prompt(
        self, writer, sample_story_state, chapter_with_scenes
    ):
        """Test that location is included in prompts."""
        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        first_call_prompt = writer.generate.call_args_list[0][0][0]
        assert "office" in first_call_prompt.lower() or "LOCATION" in first_call_prompt

    def test_maintains_continuity_between_scenes(
        self, writer, sample_story_state, chapter_with_scenes
    ):
        """Test that previous scene context is passed to next scene."""
        scene_contents = [
            "First scene with specific ending phrase...",
            "Second scene content...",
        ]
        writer.generate = MagicMock(side_effect=scene_contents)

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # Check second scene call includes previous scene context
        second_call_prompt = writer.generate.call_args_list[1][0][0]
        assert "PREVIOUS SCENE" in second_call_prompt or "ending phrase" in second_call_prompt

    def test_updates_scene_content_and_metadata(
        self, writer, sample_story_state, chapter_with_scenes
    ):
        """Test that scene content and metadata are updated."""
        scene_contents = ["First scene content...", "Second scene content..."]
        writer.generate = MagicMock(side_effect=scene_contents)

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # Check that scene content was updated
        assert chapter_with_scenes.scenes[0].content == "First scene content..."
        assert chapter_with_scenes.scenes[1].content == "Second scene content..."
        # Check word count
        assert chapter_with_scenes.scenes[0].word_count > 0
        # Check status
        assert chapter_with_scenes.scenes[0].status == "drafted"

    def test_fallback_to_chapter_level_without_scenes(self, writer, sample_story_state):
        """Test that writer falls back to chapter-level when no scenes defined."""
        writer.generate = MagicMock(return_value="Full chapter content...")

        chapter = sample_story_state.chapters[0]
        # Ensure no scenes
        assert len(chapter.scenes) == 0

        result = writer.write_chapter(sample_story_state, chapter)

        # Should only call generate once for the whole chapter
        assert writer.generate.call_count == 1
        assert result == "Full chapter content..."

    def test_write_scene_directly(self, writer, sample_story_state, chapter_with_scenes):
        """Test writing a single scene directly."""
        writer.generate = MagicMock(return_value="Direct scene content...")

        scene = chapter_with_scenes.scenes[0]
        result = writer.write_scene(
            story_state=sample_story_state,
            chapter=chapter_with_scenes,
            scene=scene,
        )

        assert result == "Direct scene content..."
        assert writer.generate.call_count == 1

        # Check that scene-specific info is in the prompt
        prompt = writer.generate.call_args[0][0]
        assert "Meeting the Client" in prompt or scene.title in prompt

    def test_scene_with_previous_chapter_context(
        self, writer, sample_story_state, chapter_with_scenes
    ):
        """Test first scene includes previous chapter context."""
        # Add content to previous chapter
        sample_story_state.chapters[0].content = "Previous chapter ending text..."
        # Update chapter number to make it chapter 2
        chapter_with_scenes.number = 2

        writer.generate = MagicMock(return_value="Scene content...")

        writer.write_chapter(sample_story_state, chapter_with_scenes)

        # First scene should include previous chapter context
        first_call_prompt = writer.generate.call_args_list[0][0][0]
        assert "PREVIOUS CHAPTER" in first_call_prompt or "ending text" in first_call_prompt
