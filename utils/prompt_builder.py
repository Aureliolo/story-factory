"""Prompt building utilities to reduce duplication across agents."""

from __future__ import annotations

from memory.story_state import Character, StoryBrief, StoryState


class PromptBuilder:
    """Builds structured prompts with common patterns to reduce code duplication.

    This class helps agents construct prompts in a consistent way, avoiding
    repeated language enforcement, context building, and formatting patterns.
    """

    def __init__(self) -> None:
        """Initialize an empty prompt builder.

        Creates a new prompt builder with an empty sections list.
        """
        self.sections: list[str] = []

    def add_language_requirement(self, language: str) -> PromptBuilder:
        """Add language enforcement section.

        This reduces the 13+ instances of language requirement duplication
        across agents (architect, writer, editor, continuity).

        Args:
            language: Target language for all content (e.g., "English", "Spanish")

        Returns:
            Self for method chaining
        """
        requirement = f"LANGUAGE: {language} - Write ALL content in {language}. All prose, dialogue, names, descriptions, and narration must be in {language}."
        self.sections.append(requirement)
        return self

    def add_story_context(
        self,
        story_state: StoryState,
        include_brief: bool = True,
        include_characters: bool = True,
        include_world: bool = True,
    ) -> PromptBuilder:
        """Add standardized story context block.

        Consolidates the manual context extraction patterns found across
        architect.py, writer.py, editor.py.

        Args:
            story_state: Current story state
            include_brief: Include story brief section
            include_characters: Include character summaries
            include_world: Include world description

        Returns:
            Self for method chaining
        """
        context_parts = []

        if include_brief and story_state.brief:
            brief = story_state.brief
            context_parts.append(
                f"STORY CONTEXT:\n"
                f"Premise: {brief.premise}\n"
                f"Genre: {brief.genre}\n"
                f"Tone: {brief.tone}"
            )
            if brief.themes:
                context_parts.append(f"Themes: {', '.join(brief.themes)}")

        if include_characters and story_state.characters:
            char_list = "\n".join(f"- {c.name}: {c.description}" for c in story_state.characters)
            context_parts.append(f"CHARACTERS:\n{char_list}")

        if include_world and story_state.world_description:
            context_parts.append(f"WORLD:\n{story_state.world_description}")

        if context_parts:
            self.sections.append("\n\n".join(context_parts))

        return self

    def add_brief_requirements(self, brief: StoryBrief) -> PromptBuilder:
        """Add story brief requirements (genre, tone, content rating).

        Args:
            brief: Story brief with requirements

        Returns:
            Self for method chaining
        """
        requirements = [
            f"GENRE: {brief.genre}",
            f"TONE: {brief.tone}",
            f"CONTENT RATING: {brief.content_rating}",
        ]

        if brief.content_preferences:
            requirements.append(f"Include: {', '.join(brief.content_preferences)}")
        if brief.content_avoid:
            requirements.append(f"Avoid: {', '.join(brief.content_avoid)}")

        self.sections.append("\n".join(requirements))
        return self

    def add_character_summary(self, characters: list[Character]) -> PromptBuilder:
        """Add formatted character summary.

        Replaces manual formatting found in multiple agents.

        Args:
            characters: List of characters to summarize

        Returns:
            Self for method chaining
        """
        if not characters:
            return self

        char_lines = []
        for char in characters:
            char_lines.append(f"- {char.name} ({char.role}): {char.description}")
            if char.personality_traits:
                char_lines.append(f"  Traits: {', '.join(char.personality_traits)}")
            if char.goals:
                char_lines.append(f"  Goals: {', '.join(char.goals)}")

        summary = "CHARACTERS:\n" + "\n".join(char_lines)
        self.sections.append(summary)
        return self

    def add_json_output_format(self, schema_example: str) -> PromptBuilder:
        """Add JSON output format instructions.

        Args:
            schema_example: Example JSON structure as a string

        Returns:
            Self for method chaining
        """
        instruction = f"OUTPUT FORMAT:\nProvide your response as JSON:\n{schema_example}"
        self.sections.append(instruction)
        return self

    def add_revision_notes(self, feedback: str) -> PromptBuilder:
        """Add revision instruction section.

        Consolidates the repeated revision note patterns found in
        writer.py and editor.py.

        Args:
            feedback: Revision feedback to address

        Returns:
            Self for method chaining
        """
        if feedback:
            revision = f"REVISION REQUESTED:\n{feedback}\n\nAddress these issues while rewriting."
            self.sections.append(revision)
        return self

    def add_section(self, title: str, content: str) -> PromptBuilder:
        """Add a custom section with title.

        Args:
            title: Section title
            content: Section content

        Returns:
            Self for method chaining
        """
        self.sections.append(f"{title}:\n{content}")
        return self

    def add_text(self, text: str) -> PromptBuilder:
        """Add raw text as a section.

        Args:
            text: Text to add

        Returns:
            Self for method chaining
        """
        self.sections.append(text)
        return self

    def build(self) -> str:
        """Combine all sections into final prompt.

        Returns:
            Complete prompt string with sections separated by double newlines
        """
        return "\n\n".join(self.sections)

    @staticmethod
    def ensure_brief(story_state: StoryState | None, agent_name: str = "Agent") -> StoryBrief:
        """Validate that a story brief exists, raising descriptive error if not.

        This replaces the 12+ instances of manual brief validation across agents.

        Args:
            story_state: Story state to check
            agent_name: Name of agent for error message

        Returns:
            The story brief if it exists

        Raises:
            ValueError: If story state or brief is missing
        """
        if not story_state or not story_state.brief:
            raise ValueError(
                f"Story brief must be created before using {agent_name}. "
                "Please complete the interview phase first."
            )
        return story_state.brief
