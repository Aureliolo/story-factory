"""Tests for WriterAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.writer import WriterAgent
from memory.story_state import Chapter, Character, StoryBrief, StoryState
from settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def writer(settings):
    """Create WriterAgent with mocked Ollama client."""
    with patch("agents.base.ollama.Client"):
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
        with patch("agents.base.ollama.Client"):
            agent = WriterAgent(settings=settings)
            assert agent.name == "Writer"
            assert agent.role == "Prose Craftsman"

    def test_init_with_custom_model(self, settings):
        """Test agent initializes with custom model."""
        with patch("agents.base.ollama.Client"):
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
