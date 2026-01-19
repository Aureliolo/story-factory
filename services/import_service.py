"""Import service - extract entities from existing text."""

import logging
from typing import Any

import ollama

from memory.story_state import StoryState
from services.model_mode_service import ModelModeService
from settings import Settings
from utils.exceptions import WorldGenerationError
from utils.json_parser import extract_json_list
from utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


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
        self._client: ollama.Client | None = None
        logger.debug("ImportService initialized successfully")

    @property
    def client(self) -> ollama.Client:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = ollama.Client(
                host=self.settings.ollama_url,
                timeout=float(self.settings.ollama_timeout),
            )
        return self._client

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

Return ONLY valid JSON array with NO markdown formatting:
[
  {{
    "name": "Character Name",
    "role": "protagonist|antagonist|supporting",
    "description": "Physical and personality description from the text",
    "relationships": {{"other_character_name": "relationship_type"}},
    "confidence": 0.9,
    "needs_review": false
  }}
]

If a character lacks details, include what you have and flag with needs_review: true if confidence < 0.7.
Be thorough - include all named characters and even unnamed roles if significant (e.g., "the guard").
"""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for character extraction")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.temp_import_extraction,
                    "num_predict": self.settings.llm_tokens_character_create
                    * self.settings.import_character_token_multiplier,
                },
            )

            data = extract_json_list(response["response"], strict=False)
            if not data:
                logger.error(f"Character extraction returned no data: {data}")
                raise WorldGenerationError(f"Invalid character extraction response: {data}")

            # Post-process: validate required fields and flag low confidence items
            for char in data:
                if isinstance(char, dict):
                    # Validate required field exists - LLM must provide confidence
                    if "confidence" not in char:
                        logger.warning(
                            "Character extraction missing confidence field for '%s', flagging for review",
                            char.get("name", "unknown"),
                        )
                        char["confidence"] = self.settings.import_default_confidence
                        char["needs_review"] = True
                    elif char["confidence"] < self.settings.import_confidence_threshold:
                        char["needs_review"] = True
                    else:
                        char.setdefault("needs_review", False)

            logger.info(f"Extracted {len(data)} characters from text")
            return data

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Character extraction LLM error: {e}")
            raise WorldGenerationError(f"LLM error during character extraction: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Character extraction JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid character extraction response: {e}") from e
        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in character extraction: {e}")
            raise WorldGenerationError(f"Unexpected character extraction error: {e}") from e

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

Return ONLY valid JSON array with NO markdown formatting:
[
  {{
    "name": "Location Name",
    "type": "location",
    "description": "Description based on the text",
    "significance": "Why this place matters",
    "confidence": 0.9,
    "needs_review": false
  }}
]

If a location lacks details, include what you have and flag with needs_review: true if confidence < 0.7.
Include cities, buildings, rooms, natural features - any place that matters to the story.
"""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for location extraction")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.temp_import_extraction,
                    "num_predict": self.settings.llm_tokens_location_create * 3,
                },
            )

            data = extract_json_list(response["response"], strict=False)
            if not data:
                logger.error(f"Location extraction returned no data: {data}")
                raise WorldGenerationError(f"Invalid location extraction response: {data}")

            # Post-process: validate required fields and flag low confidence items
            for loc in data:
                if isinstance(loc, dict):
                    # Validate required field exists - LLM must provide confidence
                    if "confidence" not in loc:
                        logger.warning(
                            "Location extraction missing confidence field for '%s', flagging for review",
                            loc.get("name", "unknown"),
                        )
                        loc["confidence"] = 0.5
                        loc["needs_review"] = True
                    elif loc["confidence"] < 0.7:
                        loc["needs_review"] = True
                    else:
                        loc.setdefault("needs_review", False)

            logger.info(f"Extracted {len(data)} locations from text")
            return data

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Location extraction LLM error: {e}")
            raise WorldGenerationError(f"LLM error during location extraction: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Location extraction JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid location extraction response: {e}") from e
        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in location extraction: {e}")
            raise WorldGenerationError(f"Unexpected location extraction error: {e}") from e

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

Return ONLY valid JSON array with NO markdown formatting:
[
  {{
    "name": "Item Name",
    "type": "item",
    "description": "Physical description from text",
    "significance": "Why this item matters",
    "properties": ["property1", "property2"],
    "confidence": 0.9,
    "needs_review": false
  }}
]

Flag with needs_review: true if confidence < 0.7.
Only include items that are actually significant - avoid mundane everyday objects unless they're plot-relevant.
"""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for item extraction")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.temp_import_extraction,
                    "num_predict": self.settings.llm_tokens_item_create * 3,
                },
            )

            data = extract_json_list(response["response"], strict=False)
            if not data:
                logger.error(f"Item extraction returned no data: {data}")
                raise WorldGenerationError(f"Invalid item extraction response: {data}")

            # Post-process: validate required fields and flag low confidence items
            for item in data:
                if isinstance(item, dict):
                    # Validate required field exists - LLM must provide confidence
                    if "confidence" not in item:
                        logger.warning(
                            "Item extraction missing confidence field for '%s', flagging for review",
                            item.get("name", "unknown"),
                        )
                        item["confidence"] = 0.5
                        item["needs_review"] = True
                    elif item["confidence"] < 0.7:
                        item["needs_review"] = True
                    else:
                        item.setdefault("needs_review", False)

            logger.info(f"Extracted {len(data)} items from text")
            return data

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Item extraction LLM error: {e}")
            raise WorldGenerationError(f"LLM error during item extraction: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Item extraction JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid item extraction response: {e}") from e
        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in item extraction: {e}")
            raise WorldGenerationError(f"Unexpected item extraction error: {e}") from e

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

Return ONLY valid JSON array with NO markdown formatting:
[
  {{
    "source": "Character A",
    "target": "Character B",
    "relation_type": "relationship_type",
    "description": "Description from the text",
    "confidence": 0.9,
    "needs_review": false
  }}
]

Flag with needs_review: true if confidence < 0.7.
Only include relationships that are actually mentioned or clearly implied in the text.
"""

        try:
            model = self._get_model()
            logger.debug(f"Using model {model} for relationship inference")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.temp_import_extraction,
                    "num_predict": self.settings.llm_tokens_relationship_create * 4,
                },
            )

            data = extract_json_list(response["response"], strict=False)
            if not data:
                logger.error(f"Relationship inference returned no data: {data}")
                raise WorldGenerationError(f"Invalid relationship inference response: {data}")

            # Post-process: validate required fields and flag low confidence items
            for rel in data:
                if isinstance(rel, dict):
                    # Validate required field exists - LLM must provide confidence
                    if "confidence" not in rel:
                        logger.warning(
                            "Relationship inference missing confidence field for '%s' -> '%s', flagging for review",
                            rel.get("source", "unknown"),
                            rel.get("target", "unknown"),
                        )
                        rel["confidence"] = 0.5
                        rel["needs_review"] = True
                    elif rel["confidence"] < 0.7:
                        rel["needs_review"] = True
                    else:
                        rel.setdefault("needs_review", False)

            logger.info(f"Inferred {len(data)} relationships from text")
            return data

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Relationship inference LLM error: {e}")
            raise WorldGenerationError(f"LLM error during relationship inference: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Relationship inference JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid relationship inference response: {e}") from e
        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in relationship inference: {e}")
            raise WorldGenerationError(f"Unexpected relationship inference error: {e}") from e

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
