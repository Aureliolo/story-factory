"""Tests for EditorAgent."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.editor import EditorAgent
from src.memory.story_state import StoryBrief, StoryState
from src.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def editor(settings):
    """Create EditorAgent with mocked Ollama client."""
    with patch("src.agents.base.ollama.Client"):
        agent = EditorAgent(model="test-model", settings=settings)
        return agent


@pytest.fixture
def sample_story_state():
    """Create a sample story state for testing."""
    brief = StoryBrief(
        premise="A romance between rival bakers",
        genre="Romance",
        subgenres=["Comedy", "Contemporary"],
        tone="Light and humorous",
        themes=["Love", "Competition"],
        setting_time="Present Day",
        setting_place="Small town",
        target_length="novella",
        language="English",
        content_rating="teen",
        content_preferences=["Banter", "Slow burn"],
        content_avoid=["Heavy drama"],
    )
    state = StoryState(
        id="test-story-001",
        project_name="The Bake-Off",
        brief=brief,
        status="editing",
        established_facts=[
            "Emma owns a bakery on Main Street",
            "Jake opened a rival bakery across the road",
            "The town is hosting a bake-off competition",
        ],
    )
    return state


class TestEditorAgentInit:
    """Tests for EditorAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("src.agents.base.ollama.Client"):
            agent = EditorAgent(settings=settings)
            assert agent.name == "Editor"
            assert agent.role == "Prose Polisher"

    def test_init_with_custom_model(self, settings):
        """Test agent initializes with custom model."""
        with patch("src.agents.base.ollama.Client"):
            agent = EditorAgent(model="editor-model:7b", settings=settings)
            assert agent.model == "editor-model:7b"


class TestEditorEditChapter:
    """Tests for edit_chapter method."""

    def test_returns_edited_content(self, editor, sample_story_state):
        """Test returns edited chapter content."""
        original = """Emma walked into the kitchen. She was tired. She started baking. The flour was everywhere."""

        edited = """Emma trudged into her kitchen, exhaustion weighing on her shoulders after another sleepless night. She reached for her favorite mixing bowl, the ceramic one with the tiny chip on the rim, and began measuring flour. Within minutes, a fine white dusting covered every surface—the counter, her apron, even the tip of her nose."""

        editor.generate = MagicMock(return_value=edited)

        result = editor.edit_chapter(sample_story_state, original)

        assert len(result) > len(original)
        editor.generate.assert_called_once()

    def test_prompt_includes_genre_and_tone(self, editor, sample_story_state):
        """Test editing prompt includes story context."""
        editor.generate = MagicMock(return_value="Edited content...")

        editor.edit_chapter(sample_story_state, "Some chapter content")

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "Romance" in prompt
        assert "Light" in prompt or "humorous" in prompt

    def test_raises_without_brief(self, editor):
        """Test raises error when brief is missing."""
        state = StoryState(id="test", status="editing")

        with pytest.raises(ValueError, match="brief"):
            editor.edit_chapter(state, "Content to edit")

    def test_preserves_language_instruction(self, editor, sample_story_state):
        """Test includes language preservation instruction."""
        editor.generate = MagicMock(return_value="Edited...")

        editor.edit_chapter(sample_story_state, "Content")

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "English" in prompt
        assert "translate" in prompt.lower() or "language" in prompt.lower()


class TestEditorEditPassage:
    """Tests for edit_passage method."""

    def test_edits_passage_with_default_language(self, editor):
        """Test edits passage using default English."""
        original = "She went to the store. She bought things. She went home."
        edited = "She hurried to the corner market, filling her basket with fresh ingredients before making her way back through the winding streets."

        editor.generate = MagicMock(return_value=edited)

        result = editor.edit_passage(original)

        assert result == edited
        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "English" in prompt

    def test_edits_passage_with_focus(self, editor):
        """Test includes focus area in prompt."""
        editor.generate = MagicMock(return_value="Improved dialogue...")

        editor.edit_passage('"Hi," he said. "Hi," she said.', focus="Improve dialogue variety")

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "FOCUS" in prompt
        assert "dialogue" in prompt.lower()

    def test_edits_passage_with_custom_language(self, editor):
        """Test edits passage in specified language."""
        editor.generate = MagicMock(return_value="Texto editado...")

        editor.edit_passage("Ella fue a la tienda.", language="Spanish")

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "Spanish" in prompt


class TestEditorGetEditSuggestions:
    """Tests for get_edit_suggestions method."""

    def test_returns_suggestions(self, editor):
        """Test returns editing suggestions."""
        text = """The man ran fast. He was very fast. The dog also ran fast."""
        suggestions = """1. Repetitive sentence structure: All three sentences follow "Subject + ran + fast" pattern.
   Suggestion: Vary sentence structure: "The man sprinted down the street, his legs pumping furiously. Behind him, the dog matched his pace with ease."

2. Overuse of "fast": The word appears three times.
   Suggestion: Use varied descriptions: "swift," "rapid," "with incredible speed"

3. Telling instead of showing: "He was very fast" is pure telling.
   Suggestion: Show his speed through action or comparison: "His feet barely seemed to touch the ground."""

        editor.generate = MagicMock(return_value=suggestions)

        result = editor.get_edit_suggestions(text)

        assert "Repetitive" in result or "sentence" in result.lower()
        call_args = editor.generate.call_args
        assert call_args[1].get("temperature") == 0.5

    def test_truncates_long_text(self, editor):
        """Test truncates very long text for analysis."""
        long_text = "A" * 10000
        editor.generate = MagicMock(return_value="Suggestions...")

        editor.get_edit_suggestions(long_text)

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        # Should be truncated based on settings
        assert len(prompt) < 10000


class TestEditorEnsureConsistency:
    """Tests for ensure_consistency method."""

    def test_checks_consistency_with_previous(self, editor, sample_story_state):
        """Test checks consistency between content sections."""
        previous = """Emma carefully arranged the croissants in the display case. Her blonde hair was tied back in a neat ponytail."""

        new_content = """Emma pushed her dark hair out of her eyes as she mixed the batter."""

        corrected = (
            """Emma pushed a stray strand of blonde hair out of her eyes as she mixed the batter."""
        )

        editor.generate = MagicMock(return_value=corrected)

        result = editor.ensure_consistency(new_content, previous, sample_story_state)

        assert "blonde" in result
        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "PREVIOUS CONTENT" in prompt
        assert "NEW CONTENT" in prompt

    def test_includes_established_facts(self, editor, sample_story_state):
        """Test includes established facts for reference."""
        editor.generate = MagicMock(return_value="Consistent content...")

        editor.ensure_consistency(
            "New chapter content", "Previous chapter ending", sample_story_state
        )

        call_args = editor.generate.call_args
        prompt = call_args[0][0]
        assert "ESTABLISHED FACTS" in prompt
        assert "Emma owns a bakery" in prompt

    def test_raises_without_brief(self, editor):
        """Test raises error when brief is missing."""
        state = StoryState(id="test", status="editing")

        with pytest.raises(ValueError, match="brief"):
            editor.ensure_consistency("New", "Previous", state)


class TestEditorIntegration:
    """Integration tests for editor workflow."""

    def test_edit_then_ensure_consistency(self, editor, sample_story_state):
        """Test editing followed by consistency check."""
        # First: Edit the chapter
        original = "Emma baked a cake. It was good."
        edited = "Emma carefully folded the flour into the butter, creating the perfect base for her signature chocolate layer cake."
        editor.generate = MagicMock(return_value=edited)

        edited_content = editor.edit_chapter(sample_story_state, original)

        # Second: Check consistency with new content
        previous = "She had perfected this recipe over ten years of practice."
        consistent = "The consistency check result."
        editor.generate = MagicMock(return_value=consistent)

        editor.ensure_consistency(edited_content, previous, sample_story_state)

        assert editor.generate.call_count == 1  # Reset between calls
