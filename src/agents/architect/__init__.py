"""Story Architect Agent - Creates story structure, characters, and outlines."""

import logging
from typing import Any

from src.memory.arc_templates import (
    CharacterArcTemplate,
    format_arc_guidance,
    get_arc_template,
)
from src.memory.story_state import (
    Chapter,
    Character,
    OutlineVariation,
    PlotPoint,
    StoryState,
)
from src.settings import Settings
from src.utils.validation import validate_not_none, validate_type

from ..base import BaseAgent
from ._structure import (
    _parse_variation_response,
    create_chapter_outline,
    create_plot_outline,
    generate_outline_variations,
)
from ._world import (
    create_characters,
    create_world,
    generate_locations,
    generate_more_characters,
    generate_relationships,
)

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
        "personality_traits": [
            {"trait": "trait description", "category": "core|flaw|quirk"}
        ],
        "goals": ["what they want", "what they need"],
        "relationships": {"other_character": "relationship description"},
        "arc_notes": "How this character should change through the story"
    }
]"""

    PLOT_POINT_SCHEMA = """{
    "plot_summary": "A compelling 1-2 paragraph summary of the story...",
    "plot_points": [
        {"description": "Inciting incident - ...", "chapter": 1},
        {"description": "First plot point - ...", "chapter": 2},
        {"description": "Midpoint twist - ...", "chapter": null},
        {"description": "Crisis - ...", "chapter": null},
        {"description": "Climax - ...", "chapter": null},
        {"description": "Resolution - ...", "chapter": null}
    ]
}"""

    CHAPTER_SCHEMA = """[
    {
        "number": 1,
        "title": "Chapter Title",
        "outline": "Detailed outline of what happens in this chapter (3-5 sentences). Include key scenes, character moments, and how it advances the plot."
    }
]"""

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

    def _get_arc_guidance(
        self, arc_id: str | None, role_name: str
    ) -> tuple[str | None, CharacterArcTemplate | None]:
        """Get arc guidance text and template for a character role.

        Args:
            arc_id: The arc template ID to look up (e.g., "hero_journey").
            role_name: The role name for logging (e.g., "protagonist", "antagonist").

        Returns:
            Tuple of (guidance_text, arc_template). Both are None if arc_id is None
            or template not found.
        """
        if not arc_id:
            return None, None

        arc_template = get_arc_template(arc_id)
        if arc_template:
            guidance = format_arc_guidance(arc_template)
            logger.info(f"Using {role_name} arc template: {arc_id}")
            return guidance, arc_template
        else:
            logger.warning(f"{role_name.capitalize()} arc template not found: {arc_id}")
            return None, None

    def create_world(self, story_state: StoryState) -> str:
        """Create the world-building document."""
        return create_world(self, story_state)

    def create_characters(
        self,
        story_state: StoryState,
        protagonist_arc_id: str | None = None,
        antagonist_arc_id: str | None = None,
    ) -> list[Character]:
        """Design the main characters.

        Args:
            story_state: Current story state with brief.
            protagonist_arc_id: Optional arc template ID for protagonist (e.g., "hero_journey").
            antagonist_arc_id: Optional arc template ID for antagonist (e.g., "mirror").

        Returns:
            List of Character objects.
        """
        return create_characters(self, story_state, protagonist_arc_id, antagonist_arc_id)

    def create_plot_outline(self, story_state: StoryState) -> tuple[str, list[PlotPoint]]:
        """Create the main plot outline and key plot points."""
        return create_plot_outline(self, story_state)

    def create_chapter_outline(self, story_state: StoryState) -> list[Chapter]:
        """Create detailed chapter outlines."""
        return create_chapter_outline(self, story_state)

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

    def _parse_variation_response(
        self,
        response: str,
        variation_number: int,
        brief,
    ) -> OutlineVariation:
        """Parse LLM response into an OutlineVariation object."""
        return _parse_variation_response(response, variation_number, brief)

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
        return generate_outline_variations(self, story_state, count)

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
        return generate_more_characters(self, story_state, existing_names, count)

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
        return generate_locations(self, story_state, existing_locations, count)

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
        return generate_relationships(
            self, story_state, entity_names, existing_relationships, count
        )


__all__ = ["ARCHITECT_SYSTEM_PROMPT", "ArchitectAgent"]
