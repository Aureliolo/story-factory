"""Utility methods mixin for WorldQualityService."""

import asyncio
import logging
import time
from typing import Any

from src.memory.entities import Entity
from src.memory.story_state import StoryBrief
from src.memory.world_quality import RefinementConfig
from src.utils.json_parser import extract_json

from ._base import WorldQualityServiceBase

logger = logging.getLogger(__name__)


class UtilsMixin(WorldQualityServiceBase):
    """Mixin providing utility methods for WorldQualityService."""

    def _format_existing_names(self, existing_names: list[str]) -> str:
        """
        Format a list of existing faction names into a prompt-ready string that highlights names to avoid duplicating.

        Parameters:
            existing_names (list[str]): Existing faction names to include in the prompt.

        Returns:
            str: Newline-separated names each prefixed with "-". If `existing_names` is empty, returns
            "None yet - you are creating the first faction."
        """
        if not existing_names:
            logger.debug("Formatting existing faction names: none provided")
            return "None yet - you are creating the first faction."

        formatted = []
        for name in existing_names:
            # Also show variations that should be avoided
            formatted.append(f"- {name}")

        logger.debug("Formatted %d existing faction names for prompt", len(formatted))
        return "\n".join(formatted)

    def _format_existing_names_warning(self, existing_names: list[str], entity_type: str) -> str:
        """
        Format existing names with explicit DO NOT examples for consistent duplicate prevention.

        Parameters:
            existing_names (list[str]): Existing entity names to avoid.
            entity_type (str): Type of entity (concept, item, location, faction) for context.

        Returns:
            str: Formatted warning string with examples of what NOT to use.
        """
        if not existing_names:
            logger.debug("Formatting existing %s names: none provided", entity_type)
            return f"EXISTING {entity_type.upper()}S: None yet - you are creating the first {entity_type}."

        formatted_names = "\n".join(f"  - {name}" for name in existing_names)

        # Generate example DO NOT variations from the first name
        example_name = existing_names[0] if existing_names else "Example"
        do_not_examples = [
            f'"{example_name}" (exact match)',
            f'"{example_name.upper()}" (case variation)',
            f'"The {example_name}" (prefix variation)',
        ]

        logger.debug("Formatted %d existing %s names for prompt", len(existing_names), entity_type)

        return f"""EXISTING {entity_type.upper()}S (DO NOT DUPLICATE OR CREATE SIMILAR NAMES):
{formatted_names}

DO NOT USE names like:
{chr(10).join(f"  - {ex}" for ex in do_not_examples)}

Create a COMPLETELY DIFFERENT {entity_type} name."""

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
            A short summary (configured word limit).
        """
        max_words = self.settings.mini_description_words_max
        if not full_description or len(full_description.split()) <= max_words:
            # Already short enough, just return trimmed version
            words = full_description.split()[:max_words]
            return " ".join(words)

        prompt = f"""Summarize in EXACTLY 10-{max_words} words for a tooltip preview.

ENTITY: {name} ({entity_type})
FULL DESCRIPTION: {full_description}

Write a punchy, informative summary. NO quotes, NO formatting, just the summary text.

SUMMARY:"""

        try:
            model = self._get_judge_model(entity_type=entity_type)  # Use fast validator model
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.world_quality_judge_temp,
                    "num_predict": self.settings.llm_tokens_mini_description,
                },
            )
            summary: str = str(response["response"]).strip()
            # Clean up any quotes or formatting
            summary = summary.strip("\"'").strip()
            # Ensure it's not too long
            words = summary.split()
            if len(words) > max_words + 3:
                summary = " ".join(words[:max_words]) + "..."
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate mini description: {e}")
            # Fallback: truncate description
            words = full_description.split()[:max_words]
            return " ".join(words) + ("..." if len(full_description.split()) > max_words else "")

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
        # Filter entities that have descriptions
        entities_with_desc = [e for e in entities if e.get("description")]
        total_count = len(entities_with_desc)

        logger.info(f"Starting mini description generation for {total_count} entities")
        start_time = time.time()

        results = {}
        for i, entity in enumerate(entities_with_desc):
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "entity")
            description = entity.get("description", "")

            logger.debug(
                f"Generating mini description {i + 1}/{total_count}: {entity_type} '{name}'"
            )

            mini_desc = self.generate_mini_description(name, entity_type, description)
            results[name] = mini_desc
            logger.debug(f"Generated mini description for {name}: {mini_desc[:50]}...")

        elapsed = time.time() - start_time
        avg_time = elapsed / max(len(results), 1)
        logger.info(
            f"Completed mini description generation: {len(results)} descriptions "
            f"in {elapsed:.2f}s ({avg_time:.2f}s avg)"
        )

        return results

    async def refine_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
    ) -> dict[str, Any] | None:
        """Refine an existing entity to improve its quality.

        Uses the quality service's refinement loop with the existing entity as a base.

        Args:
            entity: Entity to refine.
            story_brief: Story brief for context.

        Returns:
            Dictionary with refined entity data (name, description, attributes), or None on failure.
        """
        if not entity or not story_brief:
            logger.warning("refine_entity called with missing entity or story_brief")
            return None

        entity_type = entity.type.lower()
        logger.info(f"Refining {entity_type}: {entity.name}")

        try:
            # Get quality scores to identify weak areas
            current_scores = (entity.attributes or {}).get("quality_scores")
            feedback = ""
            if current_scores:
                # Build feedback from low scores
                weak_areas = []
                for key, value in current_scores.items():
                    if key not in ("average", "feedback") and isinstance(value, (int, float)):
                        if value < 7:
                            weak_areas.append(f"{key.replace('_', ' ')} ({value:.1f})")
                if weak_areas:
                    feedback = f"Focus on improving: {', '.join(weak_areas)}"

            # Build refinement prompt
            refinement_prompt = f"""Refine this {entity_type} for a story. Keep the core identity but improve quality.

Current {entity_type}:
- Name: {entity.name}
- Description: {entity.description}
- Attributes: {entity.attributes}
- Role: {(entity.attributes or {}).get("role", "unknown")}

{f"Feedback: {feedback}" if feedback else "Improve overall quality and depth."}

Story context:
- Title: {getattr(story_brief, "title", "Unknown")}
- Genre: {getattr(story_brief, "genre", "Unknown")}
- Themes: {getattr(story_brief, "themes", "Unknown")}

Return a JSON object with:
- name: string (can refine slightly but keep recognizable)
- description: string (improved, more detailed)
- attributes: object (enhanced attributes for this {entity_type} type)"""

            # Use creator model for refinement
            model_id = self._get_creator_model(entity_type)
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=refinement_prompt,
                options={
                    "temperature": config.refinement_temperature,
                    "num_predict": 2048,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response when refining {entity.name}")
                return None

            # Parse response
            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse refinement response for {entity.name}")
                return None

            logger.info(f"Successfully refined {entity_type}: {entity.name}")
            return result

        except Exception as e:
            logger.exception(f"Failed to refine entity {entity.name}: {e}")
            return None

    async def regenerate_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
        custom_instructions: str | None = None,
    ) -> dict[str, Any] | None:
        """Fully regenerate an entity with AI.

        Creates a new version of the entity while preserving its role in the story.

        Args:
            entity: Entity to regenerate.
            story_brief: Story brief for context.
            custom_instructions: Optional user guidance for regeneration.

        Returns:
            Dictionary with regenerated entity data (name, description, attributes), or None on failure.
        """
        if not entity or not story_brief:
            logger.warning("regenerate_entity called with missing entity or story_brief")
            return None

        entity_type = entity.type.lower()
        logger.info(f"Regenerating {entity_type}: {entity.name}")

        try:
            # Build regeneration prompt
            instruction_text = custom_instructions or "Create a fresh, high-quality version."

            regeneration_prompt = f"""Create a new version of this {entity_type} for a story.

Original {entity_type} (for reference):
- Name: {entity.name}
- Type: {entity.type}
- Description: {entity.description}
- Role: {(entity.attributes or {}).get("role", "unknown")}

Instructions: {instruction_text}

Story context:
- Title: {getattr(story_brief, "title", "Unknown")}
- Genre: {getattr(story_brief, "genre", "Unknown")}
- Themes: {getattr(story_brief, "themes", "Unknown")}
- Setting: {getattr(story_brief, "setting", "Unknown")}

Return a JSON object with:
- name: string (new name, can be similar or different)
- description: string (detailed, engaging description)
- attributes: object (full attributes for this {entity_type} type including role, traits, etc.)"""

            # Use creator model
            model_id = self._get_creator_model(entity_type)
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=regeneration_prompt,
                options={
                    "temperature": config.creator_temperature,
                    "num_predict": 2048,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response when regenerating {entity.name}")
                return None

            # Parse response
            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse regeneration response for {entity.name}")
                return None

            logger.info(
                f"Successfully regenerated {entity_type}: {entity.name} -> {result.get('name', 'unknown')}"
            )
            return result

        except Exception as e:
            logger.exception(f"Failed to regenerate entity {entity.name}: {e}")
            return None
