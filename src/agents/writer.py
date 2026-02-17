"""Writer Agent - Writes the actual prose."""

import logging

from src.memory.story_state import Chapter, Scene, StoryState
from src.settings import Settings
from src.utils.prompt_builder import PromptBuilder
from src.utils.validation import validate_not_empty, validate_not_none, validate_type

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

    def __init__(self, model: str | None = None, settings: Settings | None = None) -> None:
        """Initialize the Writer agent.

        Args:
            model: Override model to use. If None, uses settings-based model for writer.
            settings: Application settings. If None, loads default settings.
        """
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
        world_context: str = "",
    ) -> str:
        """Write or revise a single chapter.

        If the chapter has scenes defined, generates content scene-by-scene.
        Otherwise, generates the entire chapter at once.

        Args:
            story_state: Current story state.
            chapter: Chapter to write.
            revision_feedback: Optional revision feedback.
            world_context: Optional RAG-retrieved world context for prompt enrichment.
        """
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)
        validate_not_none(chapter, "chapter")
        validate_type(chapter, "chapter", Chapter)
        logger.info(
            f"Writing chapter {chapter.number}: '{chapter.title}'"
            + (" (revision)" if revision_feedback else "")
        )

        # Check if chapter has scenes defined
        if chapter.scenes:
            logger.info(
                f"Chapter {chapter.number} has {len(chapter.scenes)} scenes, writing scene-by-scene"
            )
            return self._write_chapter_with_scenes(
                story_state, chapter, revision_feedback, world_context
            )
        else:
            logger.info(f"Chapter {chapter.number} has no scenes, writing as single unit")
            return self._write_chapter_whole(story_state, chapter, revision_feedback, world_context)

    def _write_chapter_whole(
        self,
        story_state: StoryState,
        chapter: Chapter,
        revision_feedback: str | None = None,
        world_context: str = "",
    ) -> str:
        """Write chapter as a single unit (original behavior)."""
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

        if world_context:
            logger.debug(
                "Injecting world context into chapter %d prompt (%d chars)",
                chapter.number,
                len(world_context),
            )
            builder.add_text(f"WORLD CONTEXT:\n{world_context}")

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

    def _write_chapter_with_scenes(
        self,
        story_state: StoryState,
        chapter: Chapter,
        revision_feedback: str | None = None,
        world_context: str = "",
    ) -> str:
        """Write chapter scene-by-scene when scenes are defined."""
        scene_contents: list[str] = []

        for scene in chapter.scenes:
            scene_num = scene.order + 1  # 1-indexed for display
            logger.info(f"Writing scene {scene_num} of chapter {chapter.number}: '{scene.title}'")

            # Get context from previous scene if exists
            prev_scene_context = ""
            if scene.order > 0 and scene_contents:
                # Use last 500 chars from previous scene for continuity
                prev_scene_context = f"PREVIOUS SCENE ENDED WITH:\n...{scene_contents[-1][-500:]}\n"
            # Get context from previous chapter if this is the first scene
            elif scene.order == 0 and chapter.number > 1:
                prev = next(
                    (c for c in story_state.chapters if c.number == chapter.number - 1), None
                )
                if prev and prev.content:
                    ctx_chars = self.settings.previous_chapter_context_chars
                    prev_scene_context = (
                        f"PREVIOUS CHAPTER ENDED WITH:\n...{prev.content[-ctx_chars:]}\n"
                    )

            scene_content = self.write_scene(
                story_state=story_state,
                chapter=chapter,
                scene=scene,
                prev_context=prev_scene_context,
                revision_feedback=revision_feedback if scene.order == 0 else None,
                world_context=world_context,
            )
            scene_contents.append(scene_content)

            # Update scene content and word count
            scene.content = scene_content
            scene.word_count = len(scene_content.split())
            scene.status = "drafted"

        # Combine all scenes with smooth transitions
        full_content = "\n\n".join(scene_contents)
        logger.info(
            f"Chapter {chapter.number} complete with {len(chapter.scenes)} scenes ({len(full_content)} chars)"
        )
        return full_content

    def write_scene(
        self,
        story_state: StoryState,
        chapter: Chapter,
        scene: Scene,
        prev_context: str = "",
        revision_feedback: str | None = None,
        world_context: str = "",
    ) -> str:
        """Write a single scene within a chapter."""
        scene_num = scene.order + 1  # 1-indexed for display
        logger.debug(f"Generating scene {scene_num}: '{scene.title}'")
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        context = story_state.get_context_summary()

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text(
            f'Write Scene {scene_num} of Chapter {chapter.number}: "{scene.title or "Untitled Scene"}"'
        )
        builder.add_language_requirement(brief.language)

        # Add scene-specific information
        if scene.goal:
            builder.add_section("SCENE GOAL", scene.goal)

        if scene.beats:
            beats_text = "\n".join(f"- {beat}" for beat in scene.beats)
            builder.add_section("KEY BEATS TO HIT", beats_text)

        if scene.pov_character:
            builder.add_text(f"POV CHARACTER: {scene.pov_character}")

        if scene.location:
            builder.add_text(f"LOCATION: {scene.location}")

        builder.add_section("CHAPTER CONTEXT", chapter.outline)
        builder.add_text(f"STORY CONTEXT:\n{context}")

        if world_context:
            logger.debug(
                "Injecting world context into scene %d prompt (%d chars)",
                scene_num,
                len(world_context),
            )
            builder.add_text(f"WORLD CONTEXT:\n{world_context}")

        if prev_context:
            builder.add_text(prev_context)

        builder.add_brief_requirements(brief)
        builder.add_revision_notes(revision_feedback or "")

        builder.add_text(
            f"Write this scene in {brief.language}. Focus on:\n"
            "- Achieving the scene goal\n"
            "- Hitting the specified story beats naturally\n"
            "- Vivid, sensory descriptions\n"
            "- Character voice and emotion\n"
            "- Smooth flow from previous content\n\n"
            "Target length: 500-1000 words for this scene.\n"
            "Write in third person past tense unless specified otherwise.\n"
            "Do not include scene numbers or titles in your output - just the prose."
        )

        prompt = builder.build()

        # Use lower temperature for revisions (more focused output)
        temp = self.settings.revision_temperature if revision_feedback else None
        content = self.generate(prompt, context, temperature=temp)
        logger.debug(f"Scene {scene_num} written ({len(content)} chars)")
        return content

    def write_short_story(
        self,
        story_state: StoryState,
        revision_feedback: str | None = None,
        world_context: str = "",
    ) -> str:
        """Write a complete short story in one pass."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

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

        if world_context:
            logger.debug(
                "Injecting world context into short story prompt (%d chars)",
                len(world_context),
            )
            builder.add_text(f"WORLD CONTEXT:\n{world_context}")

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
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)
        validate_not_empty(current_text, "current_text")

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
