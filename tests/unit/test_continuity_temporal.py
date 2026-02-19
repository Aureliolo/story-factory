"""Tests for world_context parameter in ContinuityAgent.check_full_story()."""

from unittest.mock import patch

import pytest

from src.agents.continuity import ContinuityAgent, ContinuityIssueList
from src.memory.story_state import Chapter, StoryBrief, StoryState


@pytest.fixture
def mock_story_state():
    """Create a mock story state with chapters."""
    state = StoryState(id="test-story-1")
    state.brief = StoryBrief(
        premise="A hero's journey",
        genre="fantasy",
        tone="epic",
        num_chapters=2,
        language="English",
        setting_time="Medieval",
        setting_place="Fantasy realm",
        target_length="short_story",
        content_rating="teen",
    )
    state.characters = []
    state.chapters = [
        Chapter(number=1, title="Beginning", outline="", content="The hero woke up."),
        Chapter(number=2, title="End", outline="", content="The hero triumphed."),
    ]
    return state


class TestCheckFullStoryWorldContext:
    """Tests for world_context parameter in check_full_story."""

    def test_world_context_included_in_prompt(self, mock_story_state):
        """check_full_story includes world_context in the LLM prompt."""
        agent = ContinuityAgent(model="test-model:8b")

        captured_prompts = []

        def mock_generate_structured(prompt, model_class, **kwargs):
            """Capture prompts and return empty issue list."""
            captured_prompts.append(prompt)
            return ContinuityIssueList(issues=[])

        with patch.object(agent, "generate_structured", side_effect=mock_generate_structured):
            with patch.object(agent, "extract_dialogue_patterns", return_value={}):
                with patch.object(agent, "check_character_voice", return_value=[]):
                    agent.check_full_story(
                        mock_story_state,
                        world_context="TIMELINE:\n- Hero: born Year 100\n- Villain: born Year 50",
                    )

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "[START CONTEXT]" in prompt
        assert "RETRIEVED WORLD CONTEXT:" in prompt
        assert "[END CONTEXT]" in prompt
        assert "Hero: born Year 100" in prompt
        assert "Villain: born Year 50" in prompt

    def test_no_world_context_omits_block(self, mock_story_state):
        """check_full_story without world_context does not include context block."""
        agent = ContinuityAgent(model="test-model:8b")

        captured_prompts = []

        def mock_generate_structured(prompt, model_class, **kwargs):
            """Capture prompts and return empty issue list."""
            captured_prompts.append(prompt)
            return ContinuityIssueList(issues=[])

        with patch.object(agent, "generate_structured", side_effect=mock_generate_structured):
            with patch.object(agent, "extract_dialogue_patterns", return_value={}):
                with patch.object(agent, "check_character_voice", return_value=[]):
                    agent.check_full_story(mock_story_state)

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "RETRIEVED WORLD CONTEXT:" not in prompt

    def test_empty_world_context_omits_block(self, mock_story_state):
        """check_full_story with empty world_context does not include context block."""
        agent = ContinuityAgent(model="test-model:8b")

        captured_prompts = []

        def mock_generate_structured(prompt, model_class, **kwargs):
            """Capture prompts and return empty issue list."""
            captured_prompts.append(prompt)
            return ContinuityIssueList(issues=[])

        with patch.object(agent, "generate_structured", side_effect=mock_generate_structured):
            with patch.object(agent, "extract_dialogue_patterns", return_value={}):
                with patch.object(agent, "check_character_voice", return_value=[]):
                    agent.check_full_story(mock_story_state, world_context="")

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "RETRIEVED WORLD CONTEXT:" not in prompt

    def test_backward_compatibility_no_kwarg(self, mock_story_state):
        """check_full_story works without world_context kwarg (backward compat)."""
        agent = ContinuityAgent(model="test-model:8b")

        def mock_generate_structured(prompt, model_class, **kwargs):
            """Return empty issue list."""
            return ContinuityIssueList(issues=[])

        with patch.object(agent, "generate_structured", side_effect=mock_generate_structured):
            with patch.object(agent, "extract_dialogue_patterns", return_value={}):
                with patch.object(agent, "check_character_voice", return_value=[]):
                    # Call without world_context
                    issues = agent.check_full_story(mock_story_state)

        assert isinstance(issues, list)
