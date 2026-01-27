"""Base ArchitectAgent class with core initialization and utility methods."""

import logging

from src.memory.arc_templates import (
    CharacterArcTemplate,
    format_arc_guidance,
    get_arc_template,
)
from src.settings import Settings

from ..base import BaseAgent

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


class ArchitectAgentBase(BaseAgent):
    """Base class for ArchitectAgent with initialization and core utilities."""

    # Expose schema constants as class attributes for backward compatibility
    CHARACTER_SCHEMA = CHARACTER_SCHEMA
    PLOT_POINT_SCHEMA = PLOT_POINT_SCHEMA
    CHAPTER_SCHEMA = CHAPTER_SCHEMA
    LOCATION_SCHEMA = LOCATION_SCHEMA
    RELATIONSHIP_SCHEMA = RELATIONSHIP_SCHEMA

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
