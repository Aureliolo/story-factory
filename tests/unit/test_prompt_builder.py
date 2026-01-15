"""Tests for PromptBuilder utility."""

import pytest

from memory.story_state import Character, StoryBrief, StoryState
from utils.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Tests for the PromptBuilder class."""

    def test_add_language_requirement(self):
        """Should add language enforcement section."""
        builder = PromptBuilder()
        result = builder.add_language_requirement("Spanish").build()

        assert "LANGUAGE: Spanish" in result
        assert "Write ALL content in Spanish" in result

    def test_add_story_context_with_all_sections(self):
        """Should add complete story context when all data available."""
        state = StoryState(
            id="test",
            brief=StoryBrief(
                premise="A test story",
                genre="Fantasy",
                tone="Epic",
                themes=["courage", "friendship"],
                setting_time="Medieval",
                setting_place="Kingdom",
                target_length="novel",
                content_rating="moderate",
            ),
            characters=[
                Character(
                    name="Hero",
                    role="protagonist",
                    description="A brave warrior",
                    personality_traits=["brave", "loyal"],
                    goals=["Save the kingdom"],
                )
            ],
            world_description="A magical realm with dragons",
        )

        builder = PromptBuilder()
        result = builder.add_story_context(state).build()

        assert "STORY CONTEXT" in result
        assert "A test story" in result
        assert "Fantasy" in result
        assert "Epic" in result
        assert "courage, friendship" in result
        assert "CHARACTERS" in result
        assert "Hero" in result
        assert "WORLD" in result
        assert "magical realm" in result

    def test_add_story_context_selective_sections(self):
        """Should only include requested sections."""
        state = StoryState(
            id="test",
            brief=StoryBrief(
                premise="Test",
                genre="Sci-Fi",
                tone="Dark",
                setting_time="Future",
                setting_place="Space",
                target_length="short_story",
                content_rating="none",
            ),
            characters=[Character(name="Jane", role="protagonist", description="Smart")],
            world_description="Space station",
        )

        builder = PromptBuilder()
        result = builder.add_story_context(
            state, include_brief=True, include_characters=False, include_world=False
        ).build()

        assert "STORY CONTEXT" in result
        assert "Test" in result
        assert "CHARACTERS" not in result
        assert "WORLD" not in result

    def test_add_brief_requirements(self):
        """Should format brief requirements correctly."""
        brief = StoryBrief(
            premise="Test",
            genre="Romance",
            tone="Light",
            setting_time="Modern",
            setting_place="City",
            target_length="novella",
            content_rating="explicit",
            content_preferences=["happy ending", "slow burn"],
            content_avoid=["love triangle"],
        )

        builder = PromptBuilder()
        result = builder.add_brief_requirements(brief).build()

        assert "GENRE: Romance" in result
        assert "TONE: Light" in result
        assert "CONTENT RATING: explicit" in result
        assert "Include: happy ending, slow burn" in result
        assert "Avoid: love triangle" in result

    def test_add_character_summary(self):
        """Should format character summaries with details."""
        characters = [
            Character(
                name="Alice",
                role="protagonist",
                description="Detective",
                personality_traits=["analytical", "determined"],
                goals=["Solve the case", "Find redemption"],
            ),
            Character(
                name="Bob",
                role="antagonist",
                description="Criminal mastermind",
            ),
        ]

        builder = PromptBuilder()
        result = builder.add_character_summary(characters).build()

        assert "CHARACTERS:" in result
        assert "Alice (protagonist): Detective" in result
        assert "Traits: analytical, determined" in result
        assert "Goals: Solve the case, Find redemption" in result
        assert "Bob (antagonist): Criminal mastermind" in result

    def test_add_character_summary_empty_list(self):
        """Should handle empty character list gracefully."""
        builder = PromptBuilder()
        result = builder.add_character_summary([]).build()

        assert result == ""

    def test_add_json_output_format(self):
        """Should add JSON format instructions."""
        schema = '{"key": "value", "items": []}'
        builder = PromptBuilder()
        result = builder.add_json_output_format(schema).build()

        assert "OUTPUT FORMAT:" in result
        assert "JSON" in result
        assert schema in result

    def test_add_revision_notes(self):
        """Should add revision instructions when feedback provided."""
        feedback = "Fix the pacing in chapter 3"
        builder = PromptBuilder()
        result = builder.add_revision_notes(feedback).build()

        assert "REVISION REQUESTED:" in result
        assert "Fix the pacing in chapter 3" in result
        assert "Address these issues" in result

    def test_add_revision_notes_empty(self):
        """Should not add section when no feedback provided."""
        builder = PromptBuilder()
        result = builder.add_revision_notes("").build()

        assert "REVISION" not in result

    def test_add_section(self):
        """Should add custom titled section."""
        builder = PromptBuilder()
        result = builder.add_section("CUSTOM TITLE", "Custom content here").build()

        assert "CUSTOM TITLE:" in result
        assert "Custom content here" in result

    def test_add_text(self):
        """Should add raw text section."""
        builder = PromptBuilder()
        result = builder.add_text("This is raw text").build()

        assert "This is raw text" in result

    def test_method_chaining(self):
        """Should support method chaining."""
        result = (
            PromptBuilder()
            .add_language_requirement("French")
            .add_text("Write a story")
            .add_section("NOTE", "Be creative")
            .build()
        )

        assert "LANGUAGE: French" in result
        assert "Write a story" in result
        assert "NOTE:" in result
        assert "Be creative" in result

    def test_build_combines_sections_with_newlines(self):
        """Should separate sections with double newlines."""
        result = (
            PromptBuilder()
            .add_text("Section 1")
            .add_text("Section 2")
            .add_text("Section 3")
            .build()
        )

        assert "Section 1\n\nSection 2\n\nSection 3" in result

    def test_ensure_brief_with_valid_state(self):
        """Should return brief when state is valid."""
        state = StoryState(
            id="test",
            brief=StoryBrief(
                premise="Test",
                genre="Fantasy",
                tone="Epic",
                setting_time="Past",
                setting_place="Castle",
                target_length="novel",
                content_rating="none",
            ),
        )

        brief = PromptBuilder.ensure_brief(state, "TestAgent")
        assert brief is not None
        assert brief.premise == "Test"

    def test_ensure_brief_raises_on_no_state(self):
        """Should raise ValueError when state is None."""
        with pytest.raises(ValueError, match="TestAgent requires a story brief"):
            PromptBuilder.ensure_brief(None, "TestAgent")

    def test_ensure_brief_raises_on_no_brief(self):
        """Should raise ValueError when brief is None."""
        state = StoryState(id="test", brief=None)

        with pytest.raises(ValueError, match="TestAgent requires a story brief"):
            PromptBuilder.ensure_brief(state, "TestAgent")

    def test_ensure_brief_default_agent_name(self):
        """Should use default agent name in error message."""
        with pytest.raises(ValueError, match="Agent requires a story brief"):
            PromptBuilder.ensure_brief(None)
