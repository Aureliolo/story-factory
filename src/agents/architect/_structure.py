"""StructureMixin - Methods for plot outlines, chapter structure, and story building."""

import logging

from src.agents.architect._agent import ArchitectAgentBase
from src.memory.story_state import (
    Chapter,
    ChapterList,
    PlotOutline,
    PlotPoint,
    StoryState,
)
from src.utils.exceptions import LLMGenerationError
from src.utils.prompt_builder import PromptBuilder
from src.utils.validation import validate_not_none, validate_type

logger = logging.getLogger(__name__)


class StructureMixin(ArchitectAgentBase):
    """Mixin providing plot and chapter structure methods."""

    def create_plot_outline(self, story_state: StoryState) -> tuple[str, list[PlotPoint]]:
        """Create the main plot outline and key plot points."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        logger.info("Creating plot outline")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text("Create a plot outline for this story.")
        builder.add_language_requirement(brief.language)
        builder.add_text(f"PREMISE: {brief.premise}")
        builder.add_text(f"LENGTH: {brief.target_length}")
        builder.add_brief_requirements(brief)
        builder.add_character_summary(story_state.characters)

        if story_state.world_description:
            world_preview = story_state.world_description[
                : self.settings.world_description_summary_length
            ]
            if len(story_state.world_description) > self.settings.world_description_summary_length:
                world_preview += "..."
            builder.add_section("WORLD", world_preview)

        builder.add_text(
            f"Create a complete plot outline with summary and key plot points.\n"
            f"Write all text in {brief.language}.\n"
            f"Output as JSON:"
        )
        builder.add_json_output_format(self.PLOT_POINT_SCHEMA)
        builder.add_text(
            "Make sure the plot serves the themes and gives characters room to grow.\n"
            f"For mature content at level '{brief.content_rating}', integrate intimate moments naturally into the arc."
        )

        prompt = builder.build()
        result = self.generate_structured(prompt, PlotOutline)
        logger.info(f"Created plot outline with {len(result.plot_points)} plot points")

        return result.plot_summary, result.plot_points

    def create_chapter_outline(self, story_state: StoryState) -> list[Chapter]:
        """Create detailed chapter outlines."""
        logger.info("Creating chapter outlines")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

        # Use project-specific chapter count if available, otherwise use length-based default
        num_chapters: int
        if story_state.target_chapters is not None:
            num_chapters = story_state.target_chapters
            logger.debug(f"Using project-specific chapter count: {num_chapters}")
        else:
            length_map = {
                "short_story": self.settings.chapters_short_story,
                "novella": self.settings.chapters_novella,
                "novel": self.settings.chapters_novel,
            }
            chapter_count = length_map.get(brief.target_length)
            if chapter_count is None:
                logger.warning(
                    f"Unknown target_length '{brief.target_length}', using novella chapter count"
                )
                num_chapters = self.settings.chapters_novella
            else:
                num_chapters = chapter_count
            logger.debug(
                f"Using length-based chapter count: {num_chapters} for {brief.target_length}"
            )

        plot_points_text = "\n".join(f"- {p.description}" for p in story_state.plot_points)

        # Build prompt with stronger emphasis on generating exactly the right number of chapters
        builder = PromptBuilder()
        builder.add_text(
            f"Create EXACTLY {num_chapters} chapter outlines for this {brief.target_length}. "
            f"You MUST output {num_chapters} chapters - no more, no less."
        )
        builder.add_language_requirement(brief.language)
        builder.add_section("PLOT SUMMARY", story_state.plot_summary)
        builder.add_section("KEY PLOT POINTS", plot_points_text)
        builder.add_text(f"CHARACTERS: {', '.join(c.name for c in story_state.characters)}")

        builder.add_text(
            f"Output a JSON object with a 'chapters' array containing EXACTLY {num_chapters} chapters. "
            f"Each chapter needs (all text in {brief.language}):"
        )
        builder.add_json_output_format(self.CHAPTER_SCHEMA)
        builder.add_text(
            "Ensure good pacing - build tension, vary intensity, place climactic moments appropriately. "
            f"Remember: output EXACTLY {num_chapters} chapters."
        )

        # Generate chapters iteratively if needed (LLM may not produce all at once)
        all_chapters: list[Chapter] = []
        max_iterations = num_chapters * 2  # Safety limit

        for iteration in range(max_iterations):
            if len(all_chapters) >= num_chapters:  # pragma: no cover
                break  # Defensive check - end-of-loop check usually triggers first

            remaining = num_chapters - len(all_chapters)
            # Update prompt for remaining chapters
            if len(all_chapters) > 0:
                builder_iter = PromptBuilder()
                builder_iter.add_text(
                    f"Continue the chapter outline. You have {len(all_chapters)} chapters so far. "
                    f"Create the next {remaining} chapter(s) to complete the {num_chapters}-chapter outline."
                )
                builder_iter.add_language_requirement(brief.language)
                builder_iter.add_section(
                    "EXISTING CHAPTERS",
                    "\n".join(f"- Ch{c.number}: {c.title}" for c in all_chapters),
                )
                builder_iter.add_section("PLOT SUMMARY", story_state.plot_summary)
                builder_iter.add_text(
                    f"Create chapters {len(all_chapters) + 1} through {num_chapters}. "
                    f"Output JSON with a 'chapters' array:"
                )
                builder_iter.add_json_output_format(self.CHAPTER_SCHEMA)
                prompt = builder_iter.build()
            else:
                prompt = builder.build()

            result = self.generate_structured(prompt, ChapterList)

            if len(result.chapters) == 0:
                logger.warning(f"Iteration {iteration + 1}: Got 0 chapters, retrying...")
                continue

            # Renumber chapters to continue from where we left off
            for i, chapter in enumerate(result.chapters):
                chapter.number = len(all_chapters) + i + 1
            all_chapters.extend(result.chapters)
            logger.info(
                f"Iteration {iteration + 1}: Got {len(result.chapters)} chapters, total: {len(all_chapters)}"
            )

            if len(all_chapters) >= num_chapters:
                break

        if len(all_chapters) < num_chapters:
            raise LLMGenerationError(
                f"Failed to generate enough chapters after {max_iterations} iterations. "
                f"Got {len(all_chapters)}, needed {num_chapters}. "
                f"Try using a different model or adjusting the settings."
            )

        # Trim to exact count if we got more
        chapters = all_chapters[:num_chapters]
        logger.info(f"Created {len(chapters)} chapter outlines")
        return chapters

    def build_story_structure(self, story_state: StoryState) -> StoryState:
        """Complete story structure building process."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        logger.info(f"Building complete story structure for: {story_state.id}")
        # Create world
        world_response = self.create_world(story_state)
        story_state.world_description = world_response

        # Extract world rules (simple heuristic)
        rules = []
        for line in world_response.split("\n"):
            if line.strip().startswith(("-", "*", "\u2022")):
                rules.append(line.strip().lstrip("-*\u2022 "))
        story_state.world_rules = rules[:10]

        # Create characters
        story_state.characters = self.create_characters(story_state)

        # Create plot
        plot_summary, plot_points = self.create_plot_outline(story_state)
        story_state.plot_summary = plot_summary
        story_state.plot_points = plot_points

        # Create chapters
        story_state.chapters = self.create_chapter_outline(story_state)

        story_state.status = "writing"
        logger.info(
            f"Story structure complete: {len(story_state.characters)} characters, "
            f"{len(story_state.chapters)} chapters"
        )
        return story_state
