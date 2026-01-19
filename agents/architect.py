"""Story Architect Agent - Creates story structure, characters, and outlines."""

from __future__ import annotations


import logging
import random
import re
import uuid
from typing import Any

from memory.story_state import Chapter, Character, OutlineVariation, PlotPoint, StoryState
from settings import Settings
from utils.json_parser import extract_json_list, parse_json_list_to_models
from utils.prompt_builder import PromptBuilder
from utils.validation import validate_not_none, validate_positive, validate_type

from .base import BaseAgent

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM_PROMPT = """You are the Story Architect, a master storyteller who designs compelling narrative structures.

Your responsibilities:
1. Create vivid, detailed world-building
2. Design memorable characters with clear arcs and motivations
3. Craft plot outlines with proper pacing (setup, rising action, climax, resolution)
4. Plan chapter/scene structures appropriate for the story length
5. Plant seeds for foreshadowing and payoffs

CRITICAL: Always write ALL content in the specified language. Every word, name, description, and dialogue must be in that language.

You understand genre conventions and know when to follow or subvert them.
You create characters with depth - flaws, desires, fears, and growth potential.
You think about themes and how they manifest through plot and character.

For mature content, you integrate intimate scenes naturally into the narrative arc - they serve character development and plot.

Output your plans in structured formats (JSON when requested) so other team members can execute them."""


class ArchitectAgent(BaseAgent):
    """Agent that designs story structure, characters, and outlines."""

    # JSON schema constants for structured outputs
    CHARACTER_SCHEMA = """[
    {
        "name": "Full Name",
        "role": "protagonist|antagonist|love_interest|supporting",
        "description": "Physical and personality description (2-3 sentences)",
        "personality_traits": ["trait1", "trait2", "trait3"],
        "goals": ["what they want", "what they need"],
        "relationships": {"other_character": "relationship description"},
        "arc_notes": "How this character should change through the story"
    }
]"""

    PLOT_POINT_SCHEMA = """[
    {"description": "Inciting incident - ...", "chapter": 1},
    {"description": "First plot point - ...", "chapter": 2},
    {"description": "Midpoint twist - ...", "chapter": null},
    {"description": "Crisis - ...", "chapter": null},
    {"description": "Climax - ...", "chapter": null},
    {"description": "Resolution - ...", "chapter": null}
]"""

    CHAPTER_SCHEMA = """[
    {
        "number": 1,
        "title": "Chapter Title",
        "outline": "Detailed outline of what happens in this chapter (3-5 sentences). Include key scenes, character moments, and how it advances the plot."
    }
]"""

    def __init__(self, model: str | None = None, settings: Settings | None = None) -> None:
        """Initialize the Architect agent.

        Args:
            model: Override model to use. If None, uses settings-based model for architect.
            settings: Application settings. If None, loads default settings.
        """
        super().__init__(
            name="Architect",
            role="Story Structure Designer",
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
            agent_role="architect",
            model=model,
            settings=settings,
        )

    def create_world(self, story_state: StoryState) -> str:
        """Create the world-building document."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        logger.info(f"Creating world for story: {story_state.project_name or story_state.id}")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

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
        response = self.generate(prompt)
        logger.debug(f"World creation complete ({len(response)} chars)")
        return response

    def create_characters(self, story_state: StoryState) -> list[Character]:
        """Design the main characters."""
        validate_not_none(story_state, "story_state")
        validate_type(story_state, "story_state", StoryState)

        logger.info("Creating characters for story")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text("Design the main characters for this story.")
        builder.add_language_requirement(brief.language)
        builder.add_text(f"PREMISE: {brief.premise}")
        builder.add_brief_requirements(brief)

        min_chars = self.settings.world_gen_characters_min
        max_chars = self.settings.world_gen_characters_max

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
            f"Create {min_chars}-{max_chars} main characters. {naming_hint} "
            f"For each, output JSON (all text values in {brief.language}):"
        )
        builder.add_json_output_format(self.CHARACTER_SCHEMA)
        builder.add_text("Make them complex, with flaws and desires that create conflict.")

        prompt = builder.build()
        response = self.generate(prompt)
        characters = parse_json_list_to_models(response, Character)
        logger.info(f"Created {len(characters)} characters: {[c.name for c in characters]}")
        return characters

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
            f"Create:\n"
            f"1. A compelling plot summary (1-2 paragraphs) in {brief.language}\n"
            f"2. Key plot points as JSON (descriptions in {brief.language}):"
        )
        builder.add_json_output_format(self.PLOT_POINT_SCHEMA)
        builder.add_text(
            "Make sure the plot serves the themes and gives characters room to grow.\n"
            f"For mature content at level '{brief.content_rating}', integrate intimate moments naturally into the arc."
        )

        prompt = builder.build()
        response = self.generate(prompt)

        # Extract plot summary (everything before JSON)
        plot_summary = re.split(r"```json", response)[0].strip()

        # Extract plot points
        plot_points = parse_json_list_to_models(response, PlotPoint)
        logger.info(f"Created plot outline with {len(plot_points)} plot points")

        return plot_summary, plot_points

    def create_chapter_outline(self, story_state: StoryState) -> list[Chapter]:
        """Create detailed chapter outlines."""
        logger.info("Creating chapter outlines")
        brief = PromptBuilder.ensure_brief(story_state, self.name)
        length_map = {
            "short_story": self.settings.chapters_short_story,
            "novella": self.settings.chapters_novella,
            "novel": self.settings.chapters_novel,
        }
        num_chapters = length_map.get(brief.target_length, self.settings.chapters_default)
        logger.debug(f"Target: {num_chapters} chapters for {brief.target_length}")

        plot_points_text = "\n".join(f"- {p.description}" for p in story_state.plot_points)

        # Build prompt using PromptBuilder
        builder = PromptBuilder()
        builder.add_text(f"Create a {num_chapters}-chapter outline for this story.")
        builder.add_language_requirement(brief.language)
        builder.add_section("PLOT SUMMARY", story_state.plot_summary)
        builder.add_section("KEY PLOT POINTS", plot_points_text)
        builder.add_text(f"CHARACTERS: {', '.join(c.name for c in story_state.characters)}")

        builder.add_text(f"For each chapter, output JSON (all text in {brief.language}):")
        builder.add_json_output_format(self.CHAPTER_SCHEMA)
        builder.add_text(
            "Ensure good pacing - build tension, vary intensity, place climactic moments appropriately."
        )

        prompt = builder.build()
        response = self.generate(prompt)
        chapters = parse_json_list_to_models(response, Chapter)
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
            if line.strip().startswith(("-", "*", "•")):
                rules.append(line.strip().lstrip("-*• "))
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

    def generate_outline_variations(
        self,
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
            story_state: Story state with completed brief.
            count: Number of variations to generate (must be 3-5).

        Returns:
            List of OutlineVariation objects.

        Raises:
            ValueError: If count is not within configured range.
        """
        # Validate count parameter - must be within settings range
        if (
            not self.settings.outline_variations_min
            <= count
            <= self.settings.outline_variations_max
        ):
            raise ValueError(
                f"count must be between {self.settings.outline_variations_min} and "
                f"{self.settings.outline_variations_max}, got {count}"
            )

        logger.info(f"Generating {count} outline variations")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

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
            response = self.generate(prompt)

            # Parse the response to extract components
            variation = self._parse_variation_response(response, i + 1, brief)
            variations.append(variation)

            logger.info(f"Variation {i + 1} complete: {variation.name}")

        return variations

    def _parse_variation_response(
        self,
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
                if line.strip().startswith(("-", "*", "•")):
                    rule = line.strip().lstrip("-*• ")
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

    # JSON schema for locations
    LOCATION_SCHEMA = """[
    {
        "name": "Location Name",
        "type": "location",
        "description": "Detailed description of the location (2-3 sentences)",
        "significance": "Why this place matters to the story"
    }
]"""

    # JSON schema for relationships
    RELATIONSHIP_SCHEMA = """[
    {
        "source": "Character/Entity Name 1",
        "target": "Character/Entity Name 2",
        "relation_type": "knows|loves|hates|allies_with|enemies_with|located_in|owns|member_of",
        "description": "Description of the relationship"
    }
]"""

    def generate_more_characters(
        self, story_state: StoryState, existing_names: list[str], count: int = 2
    ) -> list[Character]:
        """Generate additional characters that complement existing ones.

        Args:
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
        brief = PromptBuilder.ensure_brief(story_state, self.name)

        builder = PromptBuilder()
        builder.add_text(
            f"Create {count} NEW supporting characters for this story. "
            "These should complement and interact with existing characters."
        )
        builder.add_language_requirement(brief.language)
        builder.add_text(f"PREMISE: {brief.premise}")
        builder.add_section("EXISTING CHARACTERS (do NOT recreate)", ", ".join(existing_names))

        builder.add_text(
            f"Create {count} NEW characters. Output JSON (all text in {brief.language}):"
        )
        builder.add_json_output_format(self.CHARACTER_SCHEMA)
        builder.add_text(
            "Make these characters interesting and give them connections to existing characters. "
            "Consider: mentors, rivals, allies, family members, or mysterious figures."
        )

        prompt = builder.build()
        response = self.generate(prompt)
        characters = parse_json_list_to_models(response, Character)
        logger.info(f"Generated {len(characters)} new characters: {[c.name for c in characters]}")
        return characters

    def generate_locations(
        self, story_state: StoryState, existing_locations: list[str], count: int = 3
    ) -> list[dict[str, Any]]:
        """Generate locations for the story world.

        Args:
            story_state: Current story state with brief and world description.
            existing_locations: Names of existing locations to avoid duplicates.
            count: Number of locations to generate.

        Returns:
            List of location dictionaries with name, type, description, significance.
        """
        logger.info(f"Generating {count} locations for story")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

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
            builder.add_section(
                "EXISTING LOCATIONS (do NOT recreate)", ", ".join(existing_locations)
            )

        builder.add_text(
            f"Create {count} NEW locations. Output JSON (all text in {brief.language}):"
        )
        builder.add_json_output_format(self.LOCATION_SCHEMA)
        builder.add_text(
            "Include a mix of: main settings, secret places, meeting spots, dangerous areas. "
            "Make each location atmospheric and memorable."
        )

        prompt = builder.build()
        response = self.generate(prompt)

        # Parse JSON response - use strict=False since location generation is supplementary
        locations = extract_json_list(response, strict=False) or []
        logger.info(f"Generated {len(locations)} locations")
        return locations

    def generate_relationships(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_relationships: list[tuple[str, str]],
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate relationships between existing entities.

        Args:
            story_state: Current story state with brief.
            entity_names: Names of all entities that can have relationships.
            existing_relationships: List of (source, target) tuples to avoid duplicates.
            count: Number of relationships to generate.

        Returns:
            List of relationship dictionaries.
        """
        logger.info(f"Generating {count} relationships between entities")
        brief = PromptBuilder.ensure_brief(story_state, self.name)

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
        builder.add_json_output_format(self.RELATIONSHIP_SCHEMA)
        builder.add_text(
            "Create interesting dynamics: allies, rivals, secret connections, family ties, "
            "romantic interests, professional relationships. Each should add depth to the story."
        )

        prompt = builder.build()
        response = self.generate(prompt)

        # Parse JSON response - use strict=False since relationship generation is supplementary
        relationships = extract_json_list(response, strict=False) or []
        logger.info(f"Generated {len(relationships)} relationships")
        return relationships
