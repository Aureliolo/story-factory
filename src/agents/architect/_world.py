"""World-building functions for the ArchitectAgent.

Handles world creation, character generation, location generation,
and relationship generation.
"""

import logging
import random
from typing import TYPE_CHECKING, Any

from src.memory.story_state import (
    Character,
    CharacterCreationList,
    StoryState,
)
from src.utils.json_parser import extract_json_list
from src.utils.prompt_builder import PromptBuilder
from src.utils.validation import validate_not_none, validate_positive, validate_type

if TYPE_CHECKING:
    from . import ArchitectAgent

logger = logging.getLogger(__name__)


def create_world(agent: ArchitectAgent, story_state: StoryState) -> str:
    """Create the world-building document.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief.

    Returns:
        World description text.
    """
    validate_not_none(story_state, "story_state")
    validate_type(story_state, "story_state", StoryState)

    logger.info(f"Creating world for story: {story_state.project_name or story_state.id}")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    # Build prompt using PromptBuilder
    builder = PromptBuilder()
    builder.add_text("Create detailed world-building for this story.")
    builder.add_language_requirement(brief.language)
    builder.add_text(f"PREMISE: {brief.premise}")
    builder.add_text(
        f"GENRE: {brief.genre} (subgenres: {', '.join(brief.subgenres)})\n"
        f"SETTING: {brief.setting_place}, {brief.setting_time}"
    )
    builder.add_text(f"TONE: {brief.tone}")
    builder.add_text(f"THEMES: {', '.join(brief.themes)}")

    builder.add_text(
        "Create:\n"
        "1. A vivid description of the world/setting (2-3 paragraphs)\n"
        "2. Key rules or facts about this world (5-10 bullet points)\n"
        "3. The atmosphere and mood that should permeate the story\n\n"
        "Make it immersive and specific to the genre."
    )

    prompt = builder.build()
    response = agent.generate(prompt)
    logger.debug(f"World creation complete ({len(response)} chars)")
    return response


def create_characters(
    agent: ArchitectAgent,
    story_state: StoryState,
    protagonist_arc_id: str | None = None,
    antagonist_arc_id: str | None = None,
) -> list[Character]:
    """Design the main characters.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief.
        protagonist_arc_id: Optional arc template ID for protagonist (e.g., "hero_journey").
        antagonist_arc_id: Optional arc template ID for antagonist (e.g., "mirror").

    Returns:
        List of Character objects.
    """
    from src.utils.exceptions import LLMGenerationError

    validate_not_none(story_state, "story_state")
    validate_type(story_state, "story_state", StoryState)

    logger.info("Creating characters for story")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    # Get arc templates if specified (using helper to reduce duplication)
    protagonist_arc_guidance, protagonist_arc_template = agent._get_arc_guidance(
        protagonist_arc_id, "protagonist"
    )
    antagonist_arc_guidance, antagonist_arc_template = agent._get_arc_guidance(
        antagonist_arc_id, "antagonist"
    )

    # Build prompt using PromptBuilder
    builder = PromptBuilder()
    builder.add_text("Design the main characters for this story.")
    builder.add_language_requirement(brief.language)
    builder.add_text(f"PREMISE: {brief.premise}")
    builder.add_brief_requirements(brief)

    # Add arc template guidance if available
    if protagonist_arc_guidance:
        builder.add_text("\n=== PROTAGONIST ARC GUIDANCE ===")
        builder.add_text(protagonist_arc_guidance)
        builder.add_text(
            "\nDesign the protagonist to follow this arc pattern. Their traits, goals, "
            "and arc_notes should align with the arc stages described above."
        )

    if antagonist_arc_guidance:
        builder.add_text("\n=== ANTAGONIST ARC GUIDANCE ===")
        builder.add_text(antagonist_arc_guidance)
        builder.add_text(
            "\nDesign the antagonist to follow this arc pattern. Their traits, goals, "
            "and arc_notes should align with the arc stages described above."
        )

    # Use project-specific settings if available, otherwise fall back to global settings
    min_chars = (
        story_state.target_characters_min
        if story_state.target_characters_min is not None
        else agent.settings.world_gen_characters_min
    )
    max_chars = (
        story_state.target_characters_max
        if story_state.target_characters_max is not None
        else agent.settings.world_gen_characters_max
    )
    logger.debug(f"Character count range: {min_chars}-{max_chars}")

    # Add random naming variety hint to avoid repetitive names across generations
    naming_styles = [
        "Use unexpected, fresh names - avoid common fantasy names like Elara, Kael, Thorne, or Lyra.",
        "Draw inspiration from diverse cultures for unique names.",
        "Create memorable names that reflect each character's personality.",
        "Use short, punchy names for some characters and longer, elaborate names for others.",
        "Mix naming conventions - some formal, some nicknames, some titles.",
    ]
    naming_hint = random.choice(naming_styles)

    builder.add_text(
        f"Create EXACTLY {min_chars} to {max_chars} main characters. You MUST output at least "
        f"{min_chars} characters. {naming_hint} "
        f"Output a JSON object with a 'characters' array. Each character needs "
        f"(all text values in {brief.language}):"
    )
    builder.add_json_output_format(agent.CHARACTER_SCHEMA)

    arc_reminder = ""
    if protagonist_arc_guidance or antagonist_arc_guidance:
        arc_reminder = " Ensure character arcs align with the arc guidance provided above."

    builder.add_text(
        f"Make them complex, with flaws and desires that create conflict.{arc_reminder} "
        f"Remember: output at least {min_chars} characters."
    )

    prompt = builder.build()

    # Try up to 3 times if we don't get enough characters
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        result = agent.generate_structured(prompt, CharacterCreationList)
        if len(result.characters) >= min_chars:
            # Convert to full Character objects
            characters = result.to_characters()

            # Set arc_type on characters only if a valid template was found
            for char in characters:
                if char.role == "protagonist" and protagonist_arc_template:
                    char.arc_type = protagonist_arc_id
                elif char.role == "antagonist" and antagonist_arc_template:
                    char.arc_type = antagonist_arc_id

            # Trim to max_chars if the LLM returned too many
            if len(characters) > max_chars:
                logger.info(f"Trimming characters from {len(characters)} to {max_chars}")
                characters = characters[:max_chars]

            logger.info(f"Created {len(characters)} characters: {[c.name for c in characters]}")
            return characters
        logger.warning(
            f"Attempt {attempt}: Got {len(result.characters)} characters, expected {min_chars}. "
            f"{'Retrying...' if attempt < max_attempts else 'Giving up.'}"
        )

    # If we still don't have enough after retries, raise an error
    raise LLMGenerationError(
        f"Failed to generate enough characters after {max_attempts} attempts. "
        f"Got {len(result.characters)}, needed at least {min_chars}. "
        f"Try using a different model or adjusting the settings."
    )


def generate_more_characters(
    agent: ArchitectAgent, story_state: StoryState, existing_names: list[str], count: int = 2
) -> list[Character]:
    """Generate additional characters that complement existing ones.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief.
        existing_names: Names of existing characters to avoid duplicates.
        count: Number of new characters to generate.

    Returns:
        List of new Character objects.
    """
    validate_not_none(story_state, "story_state")
    validate_type(story_state, "story_state", StoryState)
    validate_not_none(existing_names, "existing_names")
    validate_positive(count, "count")

    logger.info(f"Generating {count} more characters for story")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    builder = PromptBuilder()
    builder.add_text(
        f"Create {count} NEW supporting characters for this story. "
        "These should complement and interact with existing characters."
    )
    builder.add_language_requirement(brief.language)
    builder.add_text(f"PREMISE: {brief.premise}")
    builder.add_section("EXISTING CHARACTERS (do NOT recreate)", ", ".join(existing_names))

    builder.add_text(f"Create {count} NEW characters. Output JSON (all text in {brief.language}):")
    builder.add_json_output_format(agent.CHARACTER_SCHEMA)
    builder.add_text(
        "Make these characters interesting and give them connections to existing characters. "
        "Consider: mentors, rivals, allies, family members, or mysterious figures."
    )

    prompt = builder.build()
    result = agent.generate_structured(prompt, CharacterCreationList)
    characters = result.to_characters()
    logger.info(f"Generated {len(characters)} new characters: {[c.name for c in characters]}")
    return characters


def generate_locations(
    agent: ArchitectAgent, story_state: StoryState, existing_locations: list[str], count: int = 3
) -> list[dict[str, Any]]:
    """Generate locations for the story world.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief and world description.
        existing_locations: Names of existing locations to avoid duplicates.
        count: Number of locations to generate.

    Returns:
        List of location dictionaries with name, type, description, significance.
    """
    logger.info(f"Generating {count} locations for story")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    builder = PromptBuilder()
    builder.add_text(
        f"Create {count} important locations for this story's world. "
        "These should be places where key scenes will happen."
    )
    builder.add_language_requirement(brief.language)
    builder.add_text(f"PREMISE: {brief.premise}")

    if story_state.world_description:
        world_preview = story_state.world_description[:500]
        builder.add_section("WORLD", world_preview)

    if existing_locations:
        builder.add_section("EXISTING LOCATIONS (do NOT recreate)", ", ".join(existing_locations))

    builder.add_text(f"Create {count} NEW locations. Output JSON (all text in {brief.language}):")
    builder.add_json_output_format(agent.LOCATION_SCHEMA)
    builder.add_text(
        "Include a mix of: main settings, secret places, meeting spots, dangerous areas. "
        "Make each location atmospheric and memorable."
    )

    prompt = builder.build()
    response = agent.generate(prompt)

    # Parse JSON response - use strict=False since location generation is supplementary
    locations = extract_json_list(response, strict=False) or []
    logger.info(f"Generated {len(locations)} locations")
    return locations


def generate_relationships(
    agent: ArchitectAgent,
    story_state: StoryState,
    entity_names: list[str],
    existing_relationships: list[tuple[str, str]],
    count: int = 5,
) -> list[dict[str, Any]]:
    """Generate relationships between existing entities.

    Args:
        agent: The ArchitectAgent instance.
        story_state: Current story state with brief.
        entity_names: Names of all entities that can have relationships.
        existing_relationships: List of (source, target) tuples to avoid duplicates.
        count: Number of relationships to generate.

    Returns:
        List of relationship dictionaries.
    """
    logger.info(f"Generating {count} relationships between entities")
    brief = PromptBuilder.ensure_brief(story_state, agent.name)

    # Format existing relationships
    existing_rel_strs = [f"{s} → {t}" for s, t in existing_relationships]

    builder = PromptBuilder()
    builder.add_text(
        f"Create {count} meaningful relationships between characters/entities in this story."
    )
    builder.add_language_requirement(brief.language)
    builder.add_text(f"PREMISE: {brief.premise}")
    builder.add_section("AVAILABLE ENTITIES", ", ".join(entity_names))

    if existing_rel_strs:
        builder.add_section(
            "EXISTING RELATIONSHIPS (avoid these)", "\n".join(existing_rel_strs[:20])
        )

    builder.add_text(f"Create {count} NEW relationships. Output JSON (in {brief.language}):")
    builder.add_json_output_format(agent.RELATIONSHIP_SCHEMA)
    builder.add_text(
        "Create interesting dynamics: allies, rivals, secret connections, family ties, "
        "romantic interests, professional relationships. Each should add depth to the story."
    )

    prompt = builder.build()
    response = agent.generate(prompt)

    # Parse JSON response - use strict=False since relationship generation is supplementary
    relationships = extract_json_list(response, strict=False) or []
    logger.info(f"Generated {len(relationships)} relationships")
    return relationships
