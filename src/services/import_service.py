"""Import service - extract entities from existing text."""

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.memory.story_state import StoryState
from src.services.llm_client import generate_structured
from src.services.model_mode_service import ModelModeService
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError
from src.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


# Pydantic models for extracted entities


class ExtractedCharacter(BaseModel):
    """A character extracted from prose text."""

    name: str
    role: str  # protagonist, antagonist, supporting
    description: str
    relationships: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.7
    needs_review: bool = False


class ExtractedCharacterList(BaseModel):
    """Wrapper for list of extracted characters.

    Handles LLMs returning a single object instead of a wrapped list.
    """

    characters: list[ExtractedCharacter]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single ExtractedCharacter object in a list if needed."""
        if isinstance(data, dict) and "characters" not in data:
            if "name" in data and "role" in data:
                logger.debug("Wrapping single ExtractedCharacter in ExtractedCharacterList")
                return {"characters": [data]}
        return data


class ExtractedLocation(BaseModel):
    """A location extracted from prose text."""

    name: str
    type: str = "location"
    description: str
    significance: str = ""
    confidence: float = 0.7
    needs_review: bool = False


class ExtractedLocationList(BaseModel):
    """Wrapper for list of extracted locations.

    Handles LLMs returning a single object instead of a wrapped list.
    """

    locations: list[ExtractedLocation]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single ExtractedLocation object in a list if needed."""
        if isinstance(data, dict) and "locations" not in data:
            if "name" in data and "description" in data:
                logger.debug("Wrapping single ExtractedLocation in ExtractedLocationList")
                return {"locations": [data]}
        return data


class ExtractedItem(BaseModel):
    """An item extracted from prose text."""

    name: str
    type: str = "item"
    description: str
    significance: str = ""
    properties: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    needs_review: bool = False


class ExtractedItemList(BaseModel):
    """Wrapper for list of extracted items.

    Handles LLMs returning a single object instead of a wrapped list.
    """

    items: list[ExtractedItem]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single ExtractedItem object in a list if needed."""
        if isinstance(data, dict) and "items" not in data:
            if "name" in data and "description" in data:
                logger.debug("Wrapping single ExtractedItem in ExtractedItemList")
                return {"items": [data]}
        return data


class ExtractedRelationship(BaseModel):
    """A relationship between characters/entities."""

    source: str
    target: str
    relation_type: str
    description: str
    confidence: float = 0.7
    needs_review: bool = False


class ExtractedRelationshipList(BaseModel):
    """Wrapper for list of extracted relationships.

    Handles LLMs returning a single object instead of a wrapped list.
    """

    relationships: list[ExtractedRelationship]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_object(cls, data: Any) -> Any:
        """Wrap a single ExtractedRelationship object in a list if needed."""
        if isinstance(data, dict) and "relationships" not in data:
            if "source" in data and "target" in data:
                logger.debug("Wrapping single ExtractedRelationship in ExtractedRelationshipList")
                return {"relationships": [data]}
        return data


class ImportService:
    """Service for importing entities from existing text.

    Uses AI to extract characters, locations, items, and relationships
    from prose text, with uncertainty flagging for user review.
    """

    def __init__(self, settings: Settings, mode_service: ModelModeService):
        """Initialize ImportService.

        Args:
            settings: Application settings.
            mode_service: Model mode service for model selection.
        """
        logger.debug("Initializing ImportService")
        self.settings = settings
        self.mode_service = mode_service
        logger.debug("ImportService initialized successfully")

    def _get_model(self) -> str:
        """Get the model to use for entity extraction."""
        return self.mode_service.get_model_for_agent("writer")

    def extract_characters(
        self,
        text: str,
        story_state: StoryState | None = None,
    ) -> list[dict[str, Any]]:
        """Extract characters from prose text.

        Args:
            text: The prose text to analyze.
            story_state: Optional story state for context (genre, tone, etc.).

        Returns:
            List of character dictionaries with:
            - name: Character name
            - role: protagonist/antagonist/supporting
            - description: Physical and personality description
            - relationships: Dict of relationships to other characters
            - confidence: float 0-1 indicating extraction confidence
            - needs_review: bool if confidence is low

        Raises:
            WorldGenerationError: If extraction fails.
        """
        validate_not_empty(text, "text")
        logger.info(f"Extracting characters from text ({len(text)} chars)")

        # Build context from story state if available
        context = ""
        if story_state and story_state.brief:
            brief = story_state.brief
            context = f"""
Story Context:
- Genre: {brief.genre}
- Tone: {brief.tone}
- Themes: {", ".join(brief.themes)}
- Setting: {brief.setting_place}, {brief.setting_time}
"""

        prompt = f"""Analyze this text and extract ALL characters mentioned.

{context}

TEXT:
{text}

For each character, determine:
1. Their name (full name if given)
2. Their role: "protagonist" (main character), "antagonist" (opposes protagonist), or "supporting" (other characters)
3. Physical and personality description based on the text
4. Relationships to other characters (if any are mentioned or implied)
5. Your confidence in this extraction (0.0-1.0):
   - 1.0: Explicitly named with clear details
   - 0.7-0.9: Named but limited details
   - 0.4-0.6: Mentioned but ambiguous or unclear
   - Below 0.4: Very uncertain

Set needs_review to true if confidence < 0.7.
Be thorough - include all named characters and even unnamed roles if significant (e.g., "the guard")."""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for character extraction")

            result = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=ExtractedCharacterList,
                temperature=self.settings.temp_import_extraction,
            )

            # Convert to dict format and apply confidence threshold
            data = []
            for char in result.characters:
                char_dict = char.model_dump()
                # Apply confidence threshold
                if char_dict["confidence"] < self.settings.import_confidence_threshold:
                    char_dict["needs_review"] = True
                data.append(char_dict)

            logger.info(f"Extracted {len(data)} characters from text")
            return data

        except Exception as e:
            logger.error(f"Character extraction error: {e}")
            raise WorldGenerationError(f"Character extraction failed: {e}") from e

    def extract_locations(
        self,
        text: str,
        story_state: StoryState | None = None,
    ) -> list[dict[str, Any]]:
        """Extract locations from prose text.

        Args:
            text: The prose text to analyze.
            story_state: Optional story state for context.

        Returns:
            List of location dictionaries with:
            - name: Location name
            - type: "location"
            - description: Description from text
            - significance: Why this location matters
            - confidence: float 0-1 indicating extraction confidence
            - needs_review: bool if confidence is low

        Raises:
            WorldGenerationError: If extraction fails.
        """
        validate_not_empty(text, "text")
        logger.info(f"Extracting locations from text ({len(text)} chars)")

        # Build context from story state if available
        context = ""
        if story_state and story_state.brief:
            brief = story_state.brief
            context = f"""
Story Context:
- Setting: {brief.setting_place}, {brief.setting_time}
- Genre: {brief.genre}
"""

        prompt = f"""Analyze this text and extract ALL significant locations mentioned.

{context}

TEXT:
{text}

For each location, determine:
1. Its name (as mentioned in the text)
2. Description from the text (atmosphere, appearance, etc.)
3. Why this location matters to the story
4. Your confidence in this extraction (0.0-1.0):
   - 1.0: Explicitly named and described
   - 0.7-0.9: Named with some details
   - 0.4-0.6: Mentioned but limited detail
   - Below 0.4: Very uncertain

Set needs_review to true if confidence < 0.7.
Include cities, buildings, rooms, natural features - any place that matters to the story."""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for location extraction")

            result = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=ExtractedLocationList,
                temperature=self.settings.temp_import_extraction,
            )

            # Convert to dict format and apply confidence threshold
            data = []
            for loc in result.locations:
                loc_dict = loc.model_dump()
                if loc_dict["confidence"] < 0.7:
                    loc_dict["needs_review"] = True
                data.append(loc_dict)

            logger.info(f"Extracted {len(data)} locations from text")
            return data

        except Exception as e:
            logger.error(f"Location extraction error: {e}")
            raise WorldGenerationError(f"Location extraction failed: {e}") from e

    def extract_items(
        self,
        text: str,
        story_state: StoryState | None = None,
    ) -> list[dict[str, Any]]:
        """Extract significant items from prose text.

        Args:
            text: The prose text to analyze.
            story_state: Optional story state for context.

        Returns:
            List of item dictionaries with:
            - name: Item name
            - type: "item"
            - description: Physical description
            - significance: Why this item matters
            - properties: List of special properties
            - confidence: float 0-1 indicating extraction confidence
            - needs_review: bool if confidence is low

        Raises:
            WorldGenerationError: If extraction fails.
        """
        validate_not_empty(text, "text")
        logger.info(f"Extracting items from text ({len(text)} chars)")

        # Build context from story state if available
        context = ""
        if story_state and story_state.brief:
            brief = story_state.brief
            context = f"Story Genre: {brief.genre}\n"

        prompt = f"""Analyze this text and extract ALL significant items/objects mentioned.

{context}

TEXT:
{text}

Focus on items that are:
- Given specific names or detailed descriptions
- Important to the plot or characters
- Have special properties or significance
- Repeatedly mentioned

For each item, determine:
1. Its name
2. Physical description from the text
3. Why it matters to the story
4. Special properties (if any)
5. Your confidence (0.0-1.0)

Set needs_review to true if confidence < 0.7.
Only include items that are actually significant - avoid mundane everyday objects unless they're plot-relevant."""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for item extraction")

            result = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=ExtractedItemList,
                temperature=self.settings.temp_import_extraction,
            )

            # Convert to dict format and apply confidence threshold
            data = []
            for item in result.items:
                item_dict = item.model_dump()
                if item_dict["confidence"] < 0.7:
                    item_dict["needs_review"] = True
                data.append(item_dict)

            logger.info(f"Extracted {len(data)} items from text")
            return data

        except Exception as e:
            logger.error(f"Item extraction error: {e}")
            raise WorldGenerationError(f"Item extraction failed: {e}") from e

    def infer_relationships(
        self,
        characters: list[dict[str, Any]],
        text: str,
    ) -> list[dict[str, Any]]:
        """Infer relationships between characters based on text.

        Args:
            characters: List of character dictionaries extracted from text.
            text: The original text for context.

        Returns:
            List of relationship dictionaries with:
            - source: Source character name
            - target: Target character name
            - relation_type: Type of relationship
            - description: Relationship description
            - confidence: float 0-1 indicating extraction confidence
            - needs_review: bool if confidence is low

        Raises:
            WorldGenerationError: If inference fails.
        """
        if not characters:
            logger.debug("No characters provided for relationship inference")
            return []

        validate_not_empty(text, "text")
        logger.info(f"Inferring relationships between {len(characters)} characters")

        char_names = [c.get("name", "Unknown") for c in characters if isinstance(c, dict)]

        prompt = f"""Based on this text and the characters identified, infer relationships between them.

CHARACTERS:
{", ".join(char_names)}

TEXT:
{text}

For each relationship you find, provide:
1. Source character name
2. Target character name
3. Relationship type: knows, loves, hates, allies_with, enemies_with, parent_of, child_of, works_for, etc.
4. Description of the relationship based on the text
5. Confidence (0.0-1.0):
   - 1.0: Explicitly stated in text
   - 0.7-0.9: Strongly implied
   - 0.4-0.6: Weakly implied
   - Below 0.4: Uncertain

Set needs_review to true if confidence < 0.7.
Only include relationships that are actually mentioned or clearly implied in the text."""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for relationship inference")

            result = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=ExtractedRelationshipList,
                temperature=self.settings.temp_import_extraction,
            )

            # Convert to dict format and apply confidence threshold
            data = []
            for rel in result.relationships:
                rel_dict = rel.model_dump()
                if rel_dict["confidence"] < 0.7:
                    rel_dict["needs_review"] = True
                data.append(rel_dict)

            logger.info(f"Inferred {len(data)} relationships from text")
            return data

        except Exception as e:
            logger.error(f"Relationship inference error: {e}")
            raise WorldGenerationError(f"Relationship inference failed: {e}") from e

    def extract_all(
        self,
        text: str,
        story_state: StoryState | None = None,
    ) -> dict[str, Any]:
        """Extract all entities (characters, locations, items, relationships) from text.

        Args:
            text: The prose text to analyze.
            story_state: Optional story state for context.

        Returns:
            Dictionary with:
            - characters: List of character dicts
            - locations: List of location dicts
            - items: List of item dicts
            - relationships: List of relationship dicts
            - summary: Summary statistics

        Raises:
            WorldGenerationError: If extraction fails.
        """
        validate_not_empty(text, "text")
        logger.info(f"Extracting all entities from text ({len(text)} chars)")

        try:
            # Extract all entity types
            characters = self.extract_characters(text, story_state)
            locations = self.extract_locations(text, story_state)
            items = self.extract_items(text, story_state)
            relationships = self.infer_relationships(characters, text)

            # Calculate summary stats
            needs_review_count = sum(
                1
                for entity in (characters + locations + items + relationships)
                if isinstance(entity, dict) and entity.get("needs_review", False)
            )

            summary = {
                "total_entities": len(characters) + len(locations) + len(items),
                "characters": len(characters),
                "locations": len(locations),
                "items": len(items),
                "relationships": len(relationships),
                "needs_review": needs_review_count,
            }

            logger.info(
                f"Extraction complete: {summary['characters']} chars, "
                f"{summary['locations']} locs, {summary['items']} items, "
                f"{summary['relationships']} rels ({needs_review_count} need review)"
            )

            return {
                "characters": characters,
                "locations": locations,
                "items": items,
                "relationships": relationships,
                "summary": summary,
            }

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in full entity extraction: {e}")
            raise WorldGenerationError(f"Unexpected extraction error: {e}") from e
