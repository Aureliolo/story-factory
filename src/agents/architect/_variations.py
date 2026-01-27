"""VariationsMixin - Methods for generating and parsing outline variations."""

import logging
import re
import uuid

from src.agents.architect._agent import ArchitectAgentBase
from src.memory.story_state import (
    Chapter,
    Character,
    OutlineVariation,
    PlotPoint,
    StoryState,
)
from src.utils.json_parser import extract_json_list
from src.utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class VariationsMixin(ArchitectAgentBase):
    """Mixin providing outline variation generation methods."""

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
