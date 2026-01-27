"""Story structure functions for the ArchitectAgent.

Handles plot outline creation, chapter outline creation,
outline variation generation, and variation response parsing.
"""

import logging
import re
import uuid

from src.memory.story_state import (
    Chapter,
    ChapterList,
    Character,
    OutlineVariation,
    PlotOutline,
    PlotPoint,
    StoryState,
)
from src.utils.exceptions import LLMGenerationError
from src.utils.json_parser import extract_json_list
from src.utils.prompt_builder import PromptBuilder
from src.utils.validation import validate_not_none, validate_type

logger = logging.getLogger("src.agents.architect._structure")


def create_plot_outline(agent, story_state: StoryState) -> tuple[str, list[PlotPoint]]:
    """Create the main plot outline and key plot points.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief, characters, and world.

    Returns:
        Tuple of (plot_summary, list of PlotPoint objects).
    """
    validate_not_none(story_state, "story_state")
    validate_type(story_state, "story_state", StoryState)

    logger.info("Creating plot outline")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

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
            : agent.settings.world_description_summary_length
        ]
        if len(story_state.world_description) > agent.settings.world_description_summary_length:
            world_preview += "..."
        builder.add_section("WORLD", world_preview)

    builder.add_text(
        f"Create a complete plot outline with summary and key plot points.\n"
        f"Write all text in {brief.language}.\n"
        f"Output as JSON:"
    )
    builder.add_json_output_format(agent.PLOT_POINT_SCHEMA)
    builder.add_text(
        "Make sure the plot serves the themes and gives characters room to grow.\n"
        f"For mature content at level '{brief.content_rating}', integrate intimate moments "
        f"naturally into the arc."
    )

    prompt = builder.build()
    result = agent.generate_structured(prompt, PlotOutline)
    logger.info(f"Created plot outline with {len(result.plot_points)} plot points")

    return result.plot_summary, result.plot_points


def create_chapter_outline(agent, story_state: StoryState) -> list[Chapter]:
    """Create detailed chapter outlines.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief, plot, and characters.

    Returns:
        List of Chapter objects.
    """
    logger.info("Creating chapter outlines")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    # Use project-specific chapter count if available, otherwise use length-based default
    num_chapters: int
    if story_state.target_chapters is not None:
        num_chapters = story_state.target_chapters
        logger.debug(f"Using project-specific chapter count: {num_chapters}")
    else:
        length_map = {
            "short_story": agent.settings.chapters_short_story,
            "novella": agent.settings.chapters_novella,
            "novel": agent.settings.chapters_novel,
        }
        chapter_count = length_map.get(brief.target_length)
        if chapter_count is None:
            logger.warning(
                f"Unknown target_length '{brief.target_length}', using novella chapter count"
            )
            num_chapters = agent.settings.chapters_novella
        else:
            num_chapters = chapter_count
        logger.debug(f"Using length-based chapter count: {num_chapters} for {brief.target_length}")

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
    builder.add_json_output_format(agent.CHAPTER_SCHEMA)
    builder.add_text(
        "Ensure good pacing - build tension, vary intensity, place climactic moments "
        "appropriately. "
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
                f"Create the next {remaining} chapter(s) to complete the "
                f"{num_chapters}-chapter outline."
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
            builder_iter.add_json_output_format(agent.CHAPTER_SCHEMA)
            prompt = builder_iter.build()
        else:
            prompt = builder.build()

        result = agent.generate_structured(prompt, ChapterList)

        if len(result.chapters) == 0:
            logger.warning(f"Iteration {iteration + 1}: Got 0 chapters, retrying...")
            continue

        # Renumber chapters to continue from where we left off
        for i, chapter in enumerate(result.chapters):
            chapter.number = len(all_chapters) + i + 1
        all_chapters.extend(result.chapters)
        logger.info(
            f"Iteration {iteration + 1}: Got {len(result.chapters)} chapters, "
            f"total: {len(all_chapters)}"
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


def generate_outline_variations(
    agent,
    story_state: StoryState,
    count: int = 3,
) -> list[OutlineVariation]:
    """Generate multiple variations of the story outline.

    Each variation will have different approaches to:
    - Character dynamics and relationships
    - Plot structure and pacing
    - Chapter breakdown
    - Tone and atmosphere variations

    Args:
        agent: The ArchitectAgent instance.
        story_state: Story state with completed brief.
        count: Number of variations to generate (must be 3-5).

    Returns:
        List of OutlineVariation objects.

    Raises:
        ValueError: If count is not within configured range.
    """
    # Validate count parameter - must be within settings range
    if not agent.settings.outline_variations_min <= count <= agent.settings.outline_variations_max:
        raise ValueError(
            f"count must be between {agent.settings.outline_variations_min} and "
            f"{agent.settings.outline_variations_max}, got {count}"
        )

    logger.info(f"Generating {count} outline variations")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    variations = []

    for i in range(count):
        logger.debug(f"Generating variation {i + 1}/{count}")

        # Build variation-specific prompt
        builder = PromptBuilder()
        builder.add_text(
            f"Create variation #{i + 1} of {count} different story outline approaches."
        )
        builder.add_language_requirement(brief.language)

        # Add differentiation guidance
        if i == 0:
            builder.add_text(
                "FOCUS: Traditional narrative structure with clear hero's journey. "
                "Emphasize character growth and redemption arcs."
            )
        elif i == 1:
            builder.add_text(
                "FOCUS: Non-linear storytelling with flashbacks and multiple perspectives. "
                "Emphasize mystery and gradual revelation."
            )
        elif i == 2:
            builder.add_text(
                "FOCUS: Fast-paced action-driven plot with high stakes. "
                "Emphasize tension and external conflicts."
            )
        elif i == 3:
            builder.add_text(
                "FOCUS: Character-driven intimate story with internal conflicts. "
                "Emphasize relationships and emotional depth."
            )
        elif i == 4:
            builder.add_text(
                "FOCUS: Ensemble cast with interwoven storylines. "
                "Emphasize complex character dynamics and parallel plots."
            )

        builder.add_text(f"PREMISE: {brief.premise}")
        builder.add_text(f"LENGTH: {brief.target_length}")
        builder.add_brief_requirements(brief)

        # Request complete structure
        builder.add_text(
            "\nProvide a COMPLETE outline including:\n"
            "1. A brief rationale (2-3 sentences) explaining what makes this variation unique\n"
            "2. World description (2-3 paragraphs)\n"
            "3. Characters as JSON\n"
            "4. Plot summary (1-2 paragraphs)\n"
            "5. Key plot points as JSON\n"
            "6. Chapter outlines as JSON"
        )

        prompt = builder.build()
        response = agent.generate(prompt)

        # Parse the response to extract components
        variation = _parse_variation_response(response, i + 1, brief)
        variations.append(variation)

        logger.info(f"Variation {i + 1} complete: {variation.name}")

    return variations


def _parse_variation_response(
    response: str,
    variation_number: int,
    brief,
) -> OutlineVariation:
    """Parse LLM response into an OutlineVariation object.

    Args:
        response: Raw LLM response.
        variation_number: Which variation number this is.
        brief: Story brief for context.

    Returns:
        OutlineVariation object.
    """
    # Extract rationale (first paragraph or section before JSON)
    rationale_match = re.search(
        r"(?:rationale|unique|approach|focus)[:\s]+(.*?)(?=\n\n|characters|world|```)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    rationale = (
        rationale_match.group(1).strip()
        if rationale_match
        else f"Variation {variation_number} with unique narrative approach"
    )

    # Extract world description
    world_match = re.search(
        r"(?:world description|world)[:\s]+(.*?)(?=\n\n|characters|```json|key rules)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    world_description = world_match.group(1).strip() if world_match else ""

    # Extract world rules (bullet points from rules section)
    rules = []
    rules_section_match = re.search(
        r"(?:world rules|rules|key rules)[:\s]*(.*?)(?=\n\n[A-Z]|characters|```json|plot)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if rules_section_match:
        rules_text = rules_section_match.group(1)
        for line in rules_text.split("\n"):
            if line.strip().startswith(("-", "*", "\u2022")):
                rule = line.strip().lstrip("-*\u2022 ")
                if len(rule) > 10:
                    rules.append(rule)

    # Parse characters - look for CHARACTERS section specifically
    characters = []
    char_section_match = re.search(
        r"characters:?\s*```json\s*(.*?)\s*```",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if char_section_match:
        char_json = char_section_match.group(1)
        # Use strict=False for variation parsing - it's optional (returns None on failure)
        char_data = extract_json_list(f"```json\n{char_json}\n```", strict=False)
        if char_data:
            for item in char_data:
                try:
                    characters.append(Character(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse character in variation: {e}")

    # Extract plot summary
    plot_match = re.search(
        r"(?:plot summary|plot)[:\s]+(.*?)(?=\n\n|plot points|```json|key plot)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    plot_summary = plot_match.group(1).strip() if plot_match else ""

    # Parse plot points - look for PLOT POINTS section
    plot_points = []
    pp_section_match = re.search(
        r"plot points:?\s*```json\s*(.*?)\s*```",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if pp_section_match:
        pp_json = pp_section_match.group(1)
        # Use strict=False for variation parsing - it's optional (returns None on failure)
        pp_data = extract_json_list(f"```json\n{pp_json}\n```", strict=False)
        if pp_data:
            for item in pp_data:
                try:
                    plot_points.append(PlotPoint(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse plot point in variation: {e}")

    # Parse chapters - look for CHAPTERS section
    chapters = []
    ch_section_match = re.search(
        r"chapters:?\s*```json\s*(.*?)\s*```",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if ch_section_match:
        ch_json = ch_section_match.group(1)
        # Use strict=False for variation parsing - it's optional (returns None on failure)
        ch_data = extract_json_list(f"```json\n{ch_json}\n```", strict=False)
        if ch_data:
            for item in ch_data:
                try:
                    chapters.append(Chapter(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse chapter in variation: {e}")

    # Create variation object
    variation = OutlineVariation(
        id=str(uuid.uuid4()),
        name=f"Variation {variation_number}",
        world_description=world_description,
        world_rules=rules[:10],
        characters=characters,
        plot_summary=plot_summary,
        plot_points=plot_points,
        chapters=chapters,
        ai_rationale=rationale,
    )

    return variation
