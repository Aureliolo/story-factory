"""World Quality Service - multi-model iteration for world building quality.

Implements a generate-judge-refine loop using:
- Creator model: High temperature (0.9) for creative generation
- Judge model: Low temperature (0.1) for consistent evaluation
- Refinement: Incorporates feedback to improve entities
"""

import logging
import time
from typing import Any

import ollama

from memory.mode_database import ModeDatabase
from memory.story_state import Character, StoryState
from memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RelationshipQualityScores,
)
from services.model_mode_service import ModelModeService
from settings import Settings
from utils.exceptions import WorldGenerationError
from utils.json_parser import extract_json

logger = logging.getLogger(__name__)


class WorldQualityService:
    """Service for quality-controlled world entity generation.

    Uses a multi-model iteration loop:
    1. Creator generates initial entity (high temperature)
    2. Judge evaluates quality (low temperature)
    3. If below threshold, refine with feedback and repeat
    4. Return entity with quality scores
    """

    def __init__(self, settings: Settings, mode_service: ModelModeService):
        """Initialize WorldQualityService.

        Args:
            settings: Application settings.
            mode_service: Model mode service for model selection.
        """
        self.settings = settings
        self.mode_service = mode_service
        self._client: ollama.Client | None = None
        self._analytics_db: ModeDatabase | None = None

    @property
    def analytics_db(self) -> ModeDatabase:
        """Get or create analytics database instance."""
        if self._analytics_db is None:
            self._analytics_db = ModeDatabase()
        return self._analytics_db

    def record_entity_quality(
        self,
        project_id: str,
        entity_type: str,
        entity_name: str,
        scores: dict[str, Any],
        iterations: int,
        generation_time: float,
    ) -> None:
        """Record entity quality score to analytics database.

        Args:
            project_id: The project ID.
            entity_type: Type of entity (character, location, faction, etc.)
            entity_name: Name of the entity.
            scores: Quality scores dictionary.
            iterations: Number of refinement iterations used.
            generation_time: Time in seconds to generate.
        """
        try:
            self.analytics_db.record_world_entity_score(
                project_id=project_id,
                entity_type=entity_type,
                entity_name=entity_name,
                model_id=self._get_creator_model(),
                scores=scores,
                iterations_used=iterations,
                generation_time_seconds=generation_time,
                feedback=scores.get("feedback", ""),
            )
            logger.debug(
                f"Recorded {entity_type} '{entity_name}' quality to analytics "
                f"(avg: {scores.get('average', 0):.1f})"
            )
        except Exception as e:
            # Don't fail generation if analytics recording fails
            logger.warning(f"Failed to record entity quality to analytics: {e}")

    @property
    def client(self) -> ollama.Client:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = ollama.Client(
                host=self.settings.ollama_url,
                timeout=float(self.settings.ollama_timeout),
            )
        return self._client

    def get_config(self) -> RefinementConfig:
        """Get refinement configuration from settings."""
        return RefinementConfig.from_settings(self.settings)

    def _get_creator_model(self) -> str:
        """Get the model to use for creative generation."""
        return self.mode_service.get_model_for_agent("writer")

    def _get_judge_model(self) -> str:
        """Get the model to use for quality judgment."""
        return self.mode_service.get_model_for_agent("validator")

    # ========== CHARACTER GENERATION WITH QUALITY ==========

    def generate_character_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[Character, CharacterQualityScores, int]:
        """Generate a character with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing characters to avoid.

        Returns:
            Tuple of (Character, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If character generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for character generation")

        logger.info(f"Generating character with quality threshold {config.quality_threshold}")

        iteration = 0
        character: Character | None = None
        scores: CharacterQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    # Initial generation
                    character = self._create_character(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    # Refinement based on feedback
                    if character and scores:
                        character = self._refine_character(
                            character,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if character is None:
                    last_error = f"Character creation returned None on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Judge quality - this can raise if parsing fails
                scores = self._judge_character_quality(
                    character, story_state, config.judge_temperature
                )

                logger.info(
                    f"Character '{character.name}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (threshold {config.quality_threshold})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Character '{character.name}' met quality threshold")
                    return character, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Character generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # If we get here, all iterations failed or didn't meet quality threshold
        if character is None or scores is None:
            raise WorldGenerationError(
                f"Failed to generate character after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # We have a character but it didn't meet threshold - return it with a warning
        logger.warning(
            f"Character '{character.name}' did not meet quality threshold "
            f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
        )
        return character, scores, iteration

    def _create_character(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> Character | None:
        """Create a new character using the creator model.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            temperature: Generation temperature.

        Returns:
            New Character or None on failure.
        """
        brief = story_state.brief
        if not brief:
            return None

        prompt = f"""Create a compelling NEW character for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
SETTING: {brief.setting_place}, {brief.setting_time}

EXISTING CHARACTERS (do NOT recreate): {", ".join(existing_names) if existing_names else "None"}

Create a character with:
1. Deep psychological complexity - internal contradictions, layers
2. Clear goals - what they want vs what they need
3. Meaningful flaws that drive conflict
4. Uniqueness - not a genre archetype
5. Arc potential - room for transformation

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Full Name",
    "role": "protagonist|antagonist|love_interest|supporting",
    "description": "Physical and personality description (2-3 sentences)",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "goals": ["external want", "internal need"],
    "relationships": {{}},
    "arc_notes": "How this character should change through the story"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 500},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return Character(
                    name=data.get("name", "Unknown"),
                    role=data.get("role", "supporting"),
                    description=data.get("description", ""),
                    personality_traits=data.get("personality_traits", []),
                    goals=data.get("goals", []),
                    relationships=data.get("relationships", {}),
                    arc_notes=data.get("arc_notes", ""),
                )
        except Exception as e:
            logger.exception(f"Character creation failed: {e}")

        return None

    def _judge_character_quality(
        self,
        character: Character,
        story_state: StoryState,
        temperature: float,
    ) -> CharacterQualityScores:
        """Judge character quality using the validator model.

        Args:
            character: Character to evaluate.
            story_state: Current story state for context.
            temperature: Judge temperature (low for consistency).

        Returns:
            CharacterQualityScores with ratings and feedback.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are a literary critic evaluating character quality for a {genre} story.

CHARACTER TO EVALUATE:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

Rate each dimension 0-10:
- DEPTH: Psychological complexity, internal contradictions, layers
- GOALS: Clarity, story relevance, want vs need tension
- FLAWS: Meaningful vulnerabilities that drive conflict
- UNIQUENESS: Distinctiveness from genre archetypes
- ARC_POTENTIAL: Room for transformation and growth

Provide specific, actionable feedback for improvement.

Output ONLY valid JSON:
{{"depth": 7.5, "goals": 8.0, "flaws": 6.5, "uniqueness": 7.0, "arc_potential": 8.5, "feedback": "Specific improvement suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 300},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                # Validate all required fields are present
                required_fields = ["depth", "goals", "flaws", "uniqueness", "arc_potential"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Character judge response missing required fields: {missing}"
                    )

                return CharacterQualityScores(
                    depth=float(data["depth"]),
                    goals=float(data["goals"]),
                    flaws=float(data["flaws"]),
                    uniqueness=float(data["uniqueness"]),
                    arc_potential=float(data["arc_potential"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Character quality judgment failed: {e}")
            raise WorldGenerationError(f"Character quality judgment failed: {e}") from e

    def _refine_character(
        self,
        character: Character,
        scores: CharacterQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> Character:
        """Refine a character based on quality feedback.

        Args:
            character: Character to refine.
            scores: Quality scores with feedback.
            story_state: Current story state.
            temperature: Refinement temperature.

        Returns:
            Refined Character.
        """
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this character based on quality feedback.

ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

QUALITY SCORES (0-10):
- Depth: {scores.depth}
- Goals: {scores.goals}
- Flaws: {scores.flaws}
- Uniqueness: {scores.uniqueness}
- Arc Potential: {scores.arc_potential}

FEEDBACK: {scores.feedback}

WEAK AREAS TO IMPROVE: {", ".join(weak) if weak else "None - minor improvements only"}

Keep the name and role, but enhance the weak areas.
Make the character more compelling while maintaining consistency.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{character.name}",
    "role": "{character.role}",
    "description": "Improved description",
    "personality_traits": ["improved", "traits"],
    "goals": ["improved goals"],
    "relationships": {{}},
    "arc_notes": "Improved arc notes"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 500},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return Character(
                    name=data.get("name", character.name),
                    role=data.get("role", character.role),
                    description=data.get("description", character.description),
                    personality_traits=data.get("personality_traits", character.personality_traits),
                    goals=data.get("goals", character.goals),
                    relationships=data.get("relationships", character.relationships),
                    arc_notes=data.get("arc_notes", character.arc_notes),
                )
        except Exception as e:
            logger.exception(f"Character refinement failed: {e}")

        return character

    # ========== LOCATION GENERATION WITH QUALITY ==========

    def generate_location_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], LocationQualityScores, int]:
        """Generate a location with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing locations to avoid.

        Returns:
            Tuple of (location_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If location generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for location generation")

        logger.info(f"Generating location with quality threshold {config.quality_threshold}")

        iteration = 0
        location: dict[str, Any] = {}
        scores: LocationQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    location = self._create_location(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if location and scores:
                        location = self._refine_location(
                            location,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if not location.get("name"):
                    last_error = f"Location creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                scores = self._judge_location_quality(
                    location, story_state, config.judge_temperature
                )

                logger.info(
                    f"Location '{location.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f}"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Location '{location.get('name')}' met quality threshold")
                    return location, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Location generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # If we get here, all iterations failed or didn't meet quality threshold
        if not location.get("name") or scores is None:
            raise WorldGenerationError(
                f"Failed to generate location after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # We have a location but it didn't meet threshold - return it with a warning
        logger.warning(
            f"Location '{location.get('name')}' did not meet quality threshold "
            f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
        )
        return location, scores, iteration

    def _create_location(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new location using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a compelling location for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}

EXISTING LOCATIONS (do NOT recreate): {", ".join(existing_names) if existing_names else "None"}

Create a location with:
1. Rich atmosphere - sensory details, mood
2. Narrative significance - symbolic or plot meaning
3. Strong story relevance - connections to themes/characters
4. Distinctiveness - memorable unique qualities

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Location Name",
    "type": "location",
    "description": "Detailed description with sensory details (2-3 sentences)",
    "significance": "Why this place matters to the story"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Location creation failed: {e}")

        return {}

    def _judge_location_quality(
        self,
        location: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> LocationQualityScores:
        """Judge location quality using the validator model.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a location for a {genre} story.

LOCATION TO EVALUATE:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

Rate each dimension 0-10:
- ATMOSPHERE: Sensory richness, mood, immersion
- SIGNIFICANCE: Plot or symbolic meaning
- STORY_RELEVANCE: Connections to themes and characters
- DISTINCTIVENESS: Memorable, unique qualities

Provide specific improvement feedback.

Output ONLY valid JSON:
{{"atmosphere": 7.5, "significance": 8.0, "story_relevance": 7.0, "distinctiveness": 6.5, "feedback": "Specific suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 200},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                # Validate all required fields are present
                required_fields = [
                    "atmosphere",
                    "significance",
                    "story_relevance",
                    "distinctiveness",
                ]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Location judge response missing required fields: {missing}"
                    )

                return LocationQualityScores(
                    atmosphere=float(data["atmosphere"]),
                    significance=float(data["significance"]),
                    story_relevance=float(data["story_relevance"]),
                    distinctiveness=float(data["distinctiveness"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from location judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Location quality judgment failed: {e}")
            raise WorldGenerationError(f"Location quality judgment failed: {e}") from e

    def _refine_location(
        self,
        location: dict[str, Any],
        scores: LocationQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a location based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this location based on quality feedback.

ORIGINAL LOCATION:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

QUALITY SCORES (0-10):
- Atmosphere: {scores.atmosphere}
- Significance: {scores.significance}
- Story Relevance: {scores.story_relevance}
- Distinctiveness: {scores.distinctiveness}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{location.get("name", "Unknown")}",
    "type": "location",
    "description": "Improved description",
    "significance": "Improved significance"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Location refinement failed: {e}")

        return location

    # ========== RELATIONSHIP GENERATION WITH QUALITY ==========

    def generate_relationship_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
    ) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
        """Generate a relationship with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            entity_names: Names of entities that can have relationships.
            existing_rels: Existing (source, target) pairs to avoid.

        Returns:
            Tuple of (relationship_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If relationship generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for relationship generation")

        if len(entity_names) < 2:
            raise ValueError("Need at least 2 entities for relationship generation")

        logger.info(f"Generating relationship with quality threshold {config.quality_threshold}")

        iteration = 0
        relationship: dict[str, Any] = {}
        scores: RelationshipQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    relationship = self._create_relationship(
                        story_state, entity_names, existing_rels, config.creator_temperature
                    )
                else:
                    if relationship and scores:
                        relationship = self._refine_relationship(
                            relationship,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if not relationship.get("source") or not relationship.get("target"):
                    last_error = (
                        f"Relationship creation returned incomplete on iteration {iteration + 1}"
                    )
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Check for duplicate relationship
                source = relationship.get("source", "")
                target = relationship.get("target", "")
                rel_type = relationship.get("relation_type", "knows")
                if self._is_duplicate_relationship(source, target, rel_type, existing_rels):
                    last_error = f"Generated duplicate relationship {source} -> {target}"
                    logger.warning(last_error)
                    iteration += 1
                    continue

                scores = self._judge_relationship_quality(
                    relationship, story_state, config.judge_temperature
                )

                logger.info(
                    f"Relationship '{source} -> {target}' "
                    f"iteration {iteration + 1}: score {scores.average:.1f}"
                )

                if scores.average >= config.quality_threshold:
                    logger.info("Relationship met quality threshold")
                    return relationship, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Relationship generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # If we get here, all iterations failed or didn't meet quality threshold
        if not relationship.get("source") or not relationship.get("target") or scores is None:
            raise WorldGenerationError(
                f"Failed to generate relationship after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # We have a relationship but it didn't meet threshold - return it with a warning
        logger.warning(
            f"Relationship '{relationship.get('source')} -> {relationship.get('target')}' "
            f"did not meet quality threshold ({scores.average:.1f} < {config.quality_threshold}), "
            f"returning anyway"
        )
        return relationship, scores, iteration

    def _is_duplicate_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        existing_rels: list[tuple[str, str]],
    ) -> bool:
        """Check if a relationship already exists (in either direction for same type).

        Args:
            source: Source entity name.
            target: Target entity name.
            rel_type: Relationship type.
            existing_rels: List of existing (source, target) pairs.

        Returns:
            True if this relationship already exists.
        """
        for existing_source, existing_target in existing_rels:
            # Check both directions
            same_pair = (source == existing_source and target == existing_target) or (
                source == existing_target and target == existing_source
            )
            if same_pair:
                return True
        return False

    def _create_relationship(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new relationship using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        existing_rel_strs = [f"{s} -> {t}" for s, t in existing_rels]

        prompt = f"""Create a compelling relationship between entities for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

AVAILABLE ENTITIES: {", ".join(entity_names)}
EXISTING RELATIONSHIPS (avoid): {", ".join(existing_rel_strs[:10]) if existing_rel_strs else "None"}

Create a relationship with:
1. Tension - conflict potential
2. Complex dynamics - power balance, history
3. Story potential - opportunities for scenes
4. Authenticity - believable connection

Output ONLY valid JSON (all text in {brief.language}):
{{
    "source": "Entity Name 1",
    "target": "Entity Name 2",
    "relation_type": "knows|loves|hates|allies_with|enemies_with|located_in|owns|member_of",
    "description": "Description of the relationship with history and dynamics"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 300},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Relationship creation failed: {e}")

        return {}

    def _judge_relationship_quality(
        self,
        relationship: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> RelationshipQualityScores:
        """Judge relationship quality using the validator model.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a relationship for a {genre} story.

RELATIONSHIP TO EVALUATE:
Source: {relationship.get("source", "Unknown")}
Target: {relationship.get("target", "Unknown")}
Type: {relationship.get("relation_type", "unknown")}
Description: {relationship.get("description", "")}

Rate each dimension 0-10:
- TENSION: Conflict potential
- DYNAMICS: Complexity, power balance, history
- STORY_POTENTIAL: Opportunities for scenes and development
- AUTHENTICITY: Believability of the connection

Provide specific improvement feedback.

Output ONLY valid JSON:
{{"tension": 7.5, "dynamics": 8.0, "story_potential": 7.0, "authenticity": 8.5, "feedback": "Specific suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 200},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                # Validate all required fields are present
                required_fields = ["tension", "dynamics", "story_potential", "authenticity"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Relationship judge response missing required fields: {missing}"
                    )

                return RelationshipQualityScores(
                    tension=float(data["tension"]),
                    dynamics=float(data["dynamics"]),
                    story_potential=float(data["story_potential"]),
                    authenticity=float(data["authenticity"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from relationship judge: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Relationship quality judgment failed: {e}")
            raise WorldGenerationError(f"Relationship quality judgment failed: {e}") from e

    def _refine_relationship(
        self,
        relationship: dict[str, Any],
        scores: RelationshipQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a relationship based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this relationship based on quality feedback.

ORIGINAL RELATIONSHIP:
Source: {relationship.get("source", "Unknown")}
Target: {relationship.get("target", "Unknown")}
Type: {relationship.get("relation_type", "unknown")}
Description: {relationship.get("description", "")}

QUALITY SCORES (0-10):
- Tension: {scores.tension}
- Dynamics: {scores.dynamics}
- Story Potential: {scores.story_potential}
- Authenticity: {scores.authenticity}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep source/target/type, enhance the description and weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "source": "{relationship.get("source", "Unknown")}",
    "target": "{relationship.get("target", "Unknown")}",
    "relation_type": "{relationship.get("relation_type", "knows")}",
    "description": "Improved description with more depth"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 300},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Relationship refinement failed: {e}")

        return relationship

    # ========== FACTION GENERATION WITH QUALITY ==========

    def generate_faction_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], FactionQualityScores, int]:
        """Generate a faction with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing factions to avoid.

        Returns:
            Tuple of (faction_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If faction generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for faction generation")

        logger.info(f"Generating faction with quality threshold {config.quality_threshold}")

        iteration = 0
        faction: dict[str, Any] = {}
        scores: FactionQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    faction = self._create_faction(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if faction and scores:
                        faction = self._refine_faction(
                            faction,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if not faction.get("name"):
                    last_error = f"Faction creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                scores = self._judge_faction_quality(faction, story_state, config.judge_temperature)

                logger.info(
                    f"Faction '{faction.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f}"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Faction '{faction.get('name')}' met quality threshold")
                    return faction, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Faction generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        if not faction.get("name") or scores is None:
            raise WorldGenerationError(
                f"Failed to generate faction after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        logger.warning(
            f"Faction '{faction.get('name')}' did not meet quality threshold "
            f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
        )
        return faction, scores, iteration

    def _create_faction(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new faction using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a compelling faction/organization for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

EXISTING FACTIONS (do NOT recreate): {", ".join(existing_names) if existing_names else "None"}

Create a faction with:
1. Internal coherence - clear structure, beliefs, and rules
2. World influence - meaningful impact on the setting
3. Conflict potential - natural tensions with other groups
4. Distinctiveness - unique identity and aesthetics

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Faction Name",
    "type": "faction",
    "description": "Description of the faction, its history, and purpose (2-3 sentences)",
    "leader": "Name or title of leader (if any)",
    "goals": ["primary goal", "secondary goal"],
    "values": ["core value 1", "core value 2"]
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Faction creation failed: {e}")

        return {}

    def _judge_faction_quality(
        self,
        faction: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> FactionQualityScores:
        """Judge faction quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a faction for a {genre} story.

FACTION TO EVALUATE:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

Rate each dimension 0-10:
- COHERENCE: Internal consistency, clear structure
- INFLUENCE: World impact, power level
- CONFLICT_POTENTIAL: Story conflict opportunities
- DISTINCTIVENESS: Memorable, unique qualities

Provide specific improvement feedback.

Output ONLY valid JSON:
{{"coherence": 7.5, "influence": 8.0, "conflict_potential": 7.0, "distinctiveness": 6.5, "feedback": "Specific suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 200},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                required_fields = [
                    "coherence",
                    "influence",
                    "conflict_potential",
                    "distinctiveness",
                ]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Faction judge response missing required fields: {missing}"
                    )

                return FactionQualityScores(
                    coherence=float(data["coherence"]),
                    influence=float(data["influence"]),
                    conflict_potential=float(data["conflict_potential"]),
                    distinctiveness=float(data["distinctiveness"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from faction judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Faction quality judgment failed: {e}")
            raise WorldGenerationError(f"Faction quality judgment failed: {e}") from e

    def _refine_faction(
        self,
        faction: dict[str, Any],
        scores: FactionQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a faction based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this faction based on quality feedback.

ORIGINAL FACTION:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

QUALITY SCORES (0-10):
- Coherence: {scores.coherence}
- Influence: {scores.influence}
- Conflict Potential: {scores.conflict_potential}
- Distinctiveness: {scores.distinctiveness}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{faction.get("name", "Unknown")}",
    "type": "faction",
    "description": "Improved description",
    "leader": "Improved leader",
    "goals": ["improved goals"],
    "values": ["improved values"]
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Faction refinement failed: {e}")

        return faction

    # ========== ITEM GENERATION WITH QUALITY ==========

    def generate_item_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ItemQualityScores, int]:
        """Generate an item with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing items to avoid.

        Returns:
            Tuple of (item_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If item generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for item generation")

        logger.info(f"Generating item with quality threshold {config.quality_threshold}")

        iteration = 0
        item: dict[str, Any] = {}
        scores: ItemQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    item = self._create_item(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if item and scores:
                        item = self._refine_item(
                            item,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if not item.get("name"):
                    last_error = f"Item creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                scores = self._judge_item_quality(item, story_state, config.judge_temperature)

                logger.info(
                    f"Item '{item.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f}"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Item '{item.get('name')}' met quality threshold")
                    return item, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Item generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        if not item.get("name") or scores is None:
            raise WorldGenerationError(
                f"Failed to generate item after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        logger.warning(
            f"Item '{item.get('name')}' did not meet quality threshold "
            f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
        )
        return item, scores, iteration

    def _create_item(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new item using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a significant item/object for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

EXISTING ITEMS (do NOT recreate): {", ".join(existing_names) if existing_names else "None"}

Create an item with:
1. Significance - meaningful role in the plot or character development
2. Uniqueness - distinctive appearance or properties
3. Narrative potential - opportunities for scenes and conflict
4. Integration - fits naturally into the world

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Item Name",
    "type": "item",
    "description": "Physical description and history (2-3 sentences)",
    "significance": "Why this item matters to the story",
    "properties": ["special property 1", "special property 2"]
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Item creation failed: {e}")

        return {}

    def _judge_item_quality(
        self,
        item: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> ItemQualityScores:
        """Judge item quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating an item for a {genre} story.

ITEM TO EVALUATE:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {", ".join(item.get("properties", []))}

Rate each dimension 0-10:
- SIGNIFICANCE: Story importance, plot relevance
- UNIQUENESS: Distinctive qualities
- NARRATIVE_POTENTIAL: Opportunities for scenes
- INTEGRATION: How well it fits the world

Provide specific improvement feedback.

Output ONLY valid JSON:
{{"significance": 7.5, "uniqueness": 8.0, "narrative_potential": 7.0, "integration": 6.5, "feedback": "Specific suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 200},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                required_fields = [
                    "significance",
                    "uniqueness",
                    "narrative_potential",
                    "integration",
                ]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Item judge response missing required fields: {missing}"
                    )

                return ItemQualityScores(
                    significance=float(data["significance"]),
                    uniqueness=float(data["uniqueness"]),
                    narrative_potential=float(data["narrative_potential"]),
                    integration=float(data["integration"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from item judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Item quality judgment failed: {e}")
            raise WorldGenerationError(f"Item quality judgment failed: {e}") from e

    def _refine_item(
        self,
        item: dict[str, Any],
        scores: ItemQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine an item based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this item based on quality feedback.

ORIGINAL ITEM:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {", ".join(item.get("properties", []))}

QUALITY SCORES (0-10):
- Significance: {scores.significance}
- Uniqueness: {scores.uniqueness}
- Narrative Potential: {scores.narrative_potential}
- Integration: {scores.integration}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{item.get("name", "Unknown")}",
    "type": "item",
    "description": "Improved description",
    "significance": "Improved significance",
    "properties": ["improved properties"]
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Item refinement failed: {e}")

        return item

    # ========== CONCEPT GENERATION WITH QUALITY ==========

    def generate_concept_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ConceptQualityScores, int]:
        """Generate a concept with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing concepts to avoid.

        Returns:
            Tuple of (concept_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If concept generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for concept generation")

        logger.info(f"Generating concept with quality threshold {config.quality_threshold}")

        iteration = 0
        concept: dict[str, Any] = {}
        scores: ConceptQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                if iteration == 0:
                    concept = self._create_concept(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if concept and scores:
                        concept = self._refine_concept(
                            concept,
                            scores,
                            story_state,
                            config.refinement_temperature,
                        )

                if not concept.get("name"):
                    last_error = f"Concept creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                scores = self._judge_concept_quality(concept, story_state, config.judge_temperature)

                logger.info(
                    f"Concept '{concept.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f}"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Concept '{concept.get('name')}' met quality threshold")
                    return concept, scores, iteration + 1

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Concept generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        if not concept.get("name") or scores is None:
            raise WorldGenerationError(
                f"Failed to generate concept after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        logger.warning(
            f"Concept '{concept.get('name')}' did not meet quality threshold "
            f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
        )
        return concept, scores, iteration

    def _create_concept(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new concept using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a thematic concept/idea for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

EXISTING CONCEPTS (do NOT recreate): {", ".join(existing_names) if existing_names else "None"}

Create a concept that:
1. Is relevant to the story's themes
2. Has philosophical depth
3. Can manifest in concrete ways in the story
4. Resonates emotionally with readers

Examples: "The Corruption of Power", "Inherited Trauma", "The Price of Immortality"

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Concept Name",
    "type": "concept",
    "description": "What this concept means in the context of the story (2-3 sentences)",
    "manifestations": "How this concept appears in the story - through events, characters, or symbols"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Concept creation failed: {e}")

        return {}

    def _judge_concept_quality(
        self,
        concept: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> ConceptQualityScores:
        """Judge concept quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a thematic concept for a {genre} story.

CONCEPT TO EVALUATE:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

Rate each dimension 0-10:
- RELEVANCE: Alignment with story themes
- DEPTH: Philosophical richness
- MANIFESTATION: How well it can appear in story
- RESONANCE: Emotional impact potential

Provide specific improvement feedback.

Output ONLY valid JSON:
{{"relevance": 7.5, "depth": 8.0, "manifestation": 7.0, "resonance": 6.5, "feedback": "Specific suggestions..."}}"""

        try:
            model = self._get_judge_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 200},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                required_fields = ["relevance", "depth", "manifestation", "resonance"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Concept judge response missing required fields: {missing}"
                    )

                return ConceptQualityScores(
                    relevance=float(data["relevance"]),
                    depth=float(data["depth"]),
                    manifestation=float(data["manifestation"]),
                    resonance=float(data["resonance"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from concept judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Concept quality judgment failed: {e}")
            raise WorldGenerationError(f"Concept quality judgment failed: {e}") from e

    def _refine_concept(
        self,
        concept: dict[str, Any],
        scores: ConceptQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a concept based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this concept based on quality feedback.

ORIGINAL CONCEPT:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

QUALITY SCORES (0-10):
- Relevance: {scores.relevance}
- Depth: {scores.depth}
- Manifestation: {scores.manifestation}
- Resonance: {scores.resonance}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{concept.get("name", "Unknown")}",
    "type": "concept",
    "description": "Improved description",
    "manifestations": "Improved manifestations"
}}"""

        try:
            model = self._get_creator_model()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 400},
            )

            data = extract_json(response["response"])
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.exception(f"Concept refinement failed: {e}")

        return concept

    # ========== BATCH OPERATIONS ==========

    def generate_factions_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
    ) -> list[tuple[dict[str, Any], FactionQualityScores]]:
        """Generate multiple factions with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of factions to generate.

        Returns:
            List of (faction_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no factions could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating faction {i + 1}/{count} with quality refinement")
                start_time = time.time()
                faction, scores, iterations = self.generate_faction_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((faction, scores))
                faction_name = faction.get("name", "Unknown")
                names.append(faction_name)
                logger.info(
                    f"Faction '{faction_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="faction",
                    entity_name=faction_name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate faction {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any factions. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} factions. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_items_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
    ) -> list[tuple[dict[str, Any], ItemQualityScores]]:
        """Generate multiple items with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of items to generate.

        Returns:
            List of (item_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no items could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating item {i + 1}/{count} with quality refinement")
                start_time = time.time()
                item, scores, iterations = self.generate_item_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((item, scores))
                item_name = item.get("name", "Unknown")
                names.append(item_name)
                logger.info(
                    f"Item '{item_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="item",
                    entity_name=item_name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate item {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(f"Failed to generate any items. Errors: {'; '.join(errors)}")

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} items. {len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_concepts_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
    ) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
        """Generate multiple concepts with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of concepts to generate.

        Returns:
            List of (concept_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no concepts could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating concept {i + 1}/{count} with quality refinement")
                start_time = time.time()
                concept, scores, iterations = self.generate_concept_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((concept, scores))
                concept_name = concept.get("name", "Unknown")
                names.append(concept_name)
                logger.info(
                    f"Concept '{concept_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="concept",
                    entity_name=concept_name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate concept {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any concepts. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} concepts. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_characters_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
    ) -> list[tuple[Character, CharacterQualityScores]]:
        """Generate multiple characters with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of characters to generate.

        Returns:
            List of (Character, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no characters could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating character {i + 1}/{count} with quality refinement")
                start_time = time.time()
                char, scores, iterations = self.generate_character_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((char, scores))
                names.append(char.name)
                logger.info(
                    f"Character '{char.name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="character",
                    entity_name=char.name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate character {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any characters. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} characters. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_locations_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
    ) -> list[tuple[dict[str, Any], LocationQualityScores]]:
        """Generate multiple locations with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of locations to generate.

        Returns:
            List of (location_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no locations could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating location {i + 1}/{count} with quality refinement")
                start_time = time.time()
                loc, scores, iterations = self.generate_location_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((loc, scores))
                loc_name = loc.get("name", "Unknown")
                names.append(loc_name)
                logger.info(
                    f"Location '{loc_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="location",
                    entity_name=loc_name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate location {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any locations. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} locations. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_relationships_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        count: int = 5,
    ) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
        """Generate multiple relationships with quality refinement.

        Args:
            story_state: Current story state.
            entity_names: Entity names available for relationships.
            existing_rels: Existing relationships to avoid.
            count: Number of relationships to generate.

        Returns:
            List of (relationship_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no relationships could be generated.
        """
        results = []
        rels = existing_rels.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating relationship {i + 1}/{count} with quality refinement")
                start_time = time.time()
                rel, scores, iterations = self.generate_relationship_with_quality(
                    story_state, entity_names, rels
                )
                elapsed = time.time() - start_time
                results.append((rel, scores))
                rels.append((rel.get("source", ""), rel.get("target", "")))
                rel_name = f"{rel.get('source', '')} -> {rel.get('target', '')}"
                logger.info(
                    f"Relationship '{rel_name}' complete "
                    f"after {iterations} iteration(s), quality: {scores.average:.1f}"
                )
                # Record to analytics
                self.record_entity_quality(
                    project_id=story_state.id,
                    entity_type="relationship",
                    entity_name=rel_name,
                    scores=scores.to_dict(),
                    iterations=iterations,
                    generation_time=elapsed,
                )
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate relationship {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any relationships. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} relationships. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    # ========== MINI DESCRIPTION GENERATION ==========

    def generate_mini_description(
        self,
        name: str,
        entity_type: str,
        full_description: str,
    ) -> str:
        """Generate a short 10-15 word mini description for hover tooltips.

        Uses a fast model with low temperature for consistent, concise output.

        Args:
            name: Entity name.
            entity_type: Type of entity (character, location, etc.).
            full_description: Full description to summarize.

        Returns:
            A short summary (10-15 words max).
        """
        if not full_description or len(full_description.split()) <= 15:
            # Already short enough, just return trimmed version
            words = full_description.split()[:15]
            return " ".join(words)

        prompt = f"""Summarize in EXACTLY 10-15 words for a tooltip preview.

ENTITY: {name} ({entity_type})
FULL DESCRIPTION: {full_description}

Write a punchy, informative summary. NO quotes, NO formatting, just the summary text.
Example: "Mysterious rogue haunted by past betrayals, seeking redemption through unlikely alliances"

SUMMARY:"""

        try:
            model = self._get_judge_model()  # Use fast validator model
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 50},
            )
            summary: str = str(response["response"]).strip()
            # Clean up any quotes or formatting
            summary = summary.strip("\"'").strip()
            # Ensure it's not too long
            words = summary.split()
            if len(words) > 18:
                summary = " ".join(words[:15]) + "..."
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate mini description: {e}")
            # Fallback: truncate description
            words = full_description.split()[:15]
            return " ".join(words) + ("..." if len(full_description.split()) > 15 else "")

    def generate_mini_descriptions_batch(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Generate mini descriptions for a batch of entities.

        Args:
            entities: List of entity dicts with 'name', 'type', 'description'.

        Returns:
            Dict mapping entity names to mini descriptions.
        """
        results = {}
        for entity in entities:
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "entity")
            description = entity.get("description", "")

            if description:
                mini_desc = self.generate_mini_description(name, entity_type, description)
                results[name] = mini_desc
                logger.debug(f"Generated mini description for {name}: {mini_desc}")

        return results
