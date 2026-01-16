"""Writer Agent - Writes the actual prose."""

import logging

from memory.story_state import Chapter, StoryState
from utils.prompt_builder import PromptBuilder

from .base import BaseAgent

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are the Writer, a skilled prose craftsman who brings stories to life.

CRITICAL: Always write ALL content in the specified language. Every word of prose, dialogue, and narration must be in that language.

Your strengths:
- Vivid, sensory descriptions that immerse readers
- Natural, character-distinct dialogue
- Varied sentence structure and pacing
- Show don't tell - convey emotion through action and detail
- Smooth scene transitions
- Maintaining consistent voice and tone throughout

You write prose that serves the story's genre and tone. You understand that different genres have different conventions:
- Literary fiction: introspection, metaphor, subtext
- Thriller: short sentences, tension, momentum
- Romance: emotional beats, chemistry, longing
- Erotica: sensual detail, building anticipation, emotional connection
- Horror: atmosphere, dread, the unseen
- Fantasy: world details woven naturally, wonder

For mature content, you write intimate scenes with:
- Emotional truth - what characters feel, not just what they do
- Building tension and anticipation
- Character voice maintained even in heated moments
- Variety in pacing and intensity
- Natural integration into the larger narrative

Write complete, polished prose ready for the editor."""


class WriterAgent(BaseAgent):
    """Agent that writes the actual prose."""

    def __init__(self, model: str | None = None, settings=None):
        super().__init__(
            name="Writer",
            role="Prose Craftsman",
            system_prompt=WRITER_SYSTEM_PROMPT,
            agent_role="writer",
            model=model,
            settings=settings,
        )

    def write_chapter(
        self,
        story_state: StoryState,
        chapter: Chapter,
        revision_feedback: str | None = None,
    ) -> str:
        """Write or revise a single chapter."""
        logger.info(
            f"Writing chapter {chapter.number}: '{chapter.title}'"
            + (" (revision)" if revision_feedback else "")
        )
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        context = story_state.get_context_summary()

        # Get previous chapter summary if exists
        prev_chapter_summary = ""
        if chapter.number > 1:
            prev = next((c for c in story_state.chapters if c.number == chapter.number - 1), None)
            if prev and prev.content:
                ctx_chars = self.settings.previous_chapter_context_chars
                prev_chapter_summary = (
                    f"PREVIOUS CHAPTER ENDED WITH:\n...{prev.content[-ctx_chars:]}"
                )

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text(f'Write Chapter {chapter.number}: "{chapter.title}"')
        builder.add_language_requirement(brief.language)
        builder.add_section("CHAPTER OUTLINE", chapter.outline)
        builder.add_text(f"STORY CONTEXT:\n{context}")

        if prev_chapter_summary:
            builder.add_text(prev_chapter_summary)

        builder.add_brief_requirements(brief)
        builder.add_revision_notes(revision_feedback or "")

        builder.add_text(
            f"Write the complete chapter in {brief.language}. Include:\n"
            "- Scene-setting description\n"
            "- Character actions and dialogue\n"
            "- Internal thoughts/feelings where appropriate\n"
            "- Smooth transitions between scenes\n"
            "- A hook or tension point at the end to pull readers forward\n\n"
            "Target length: 1500-2500 words for this chapter.\n"
            "Write in third person past tense unless the outline specifies otherwise.\n"
            "Do not include the chapter title or number in your output - just the prose."
        )

        prompt = builder.build()

        # Use lower temperature for revisions (more focused output)
        temp = self.settings.revision_temperature if revision_feedback else None
        content = self.generate(prompt, context, temperature=temp)
        logger.info(f"Chapter {chapter.number} written ({len(content)} chars)")
        return content

    def write_short_story(
        self,
        story_state: StoryState,
        revision_feedback: str | None = None,
    ) -> str:
        """Write a complete short story in one pass."""
        logger.info("Writing complete short story" + (" (revision)" if revision_feedback else ""))
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        context = story_state.get_context_summary()

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text(f"Write a complete short story based on this premise:\n{brief.premise}")
        builder.add_language_requirement(brief.language)
        builder.add_brief_requirements(brief)
        builder.add_text(f"SETTING: {brief.setting_place}, {brief.setting_time}")
        builder.add_character_summary(story_state.characters)
        builder.add_section("PLOT OUTLINE", story_state.plot_summary)
        builder.add_revision_notes(revision_feedback or "")

        builder.add_text(
            f"Write a complete, polished short story in {brief.language} (2000-4000 words).\n"
            "Include a strong opening hook, rising tension, a satisfying climax, and resolution.\n"
            "Show character growth and explore the themes naturally through the narrative.\n\n"
            "Write only the story prose - no titles, headers, or meta-commentary."
        )

        prompt = builder.build()

        # Use lower temperature for revisions (more focused output)
        temp = self.settings.revision_temperature if revision_feedback else None
        content = self.generate(prompt, context, temperature=temp)
        logger.info(f"Short story written ({len(content)} chars)")
        return content

    def continue_scene(
        self,
        story_state: StoryState,
        current_text: str,
        direction: str | None = None,
    ) -> str:
        """Continue writing from where the text left off."""
        logger.info(f"Continuing scene from {len(current_text)} chars of existing text")
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        context = story_state.get_context_summary()

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text("Continue this scene:")
        builder.add_language_requirement(brief.language)
        builder.add_text(current_text[-1500:])

        if direction:
            builder.add_section("DIRECTION", direction)
        else:
            builder.add_text("Continue naturally to the next beat.")

        builder.add_brief_requirements(brief)
        builder.add_text(
            f"Continue seamlessly from where the text ends. Write 500-1000 more words in {brief.language}.\n"
            "Maintain the same voice, tense, and style."
        )

        prompt = builder.build()
        continuation = self.generate(prompt, context)
        logger.debug(f"Scene continued ({len(continuation)} chars added)")
        return continuation
