"""Validation and consistency checking mixin for WorldQualityService."""

import asyncio
import logging
from typing import Any

from src.memory.entities import Entity
from src.memory.story_state import StoryBrief
from src.memory.world_quality import RefinementConfig
from src.utils.json_parser import extract_json

from ._base import WorldQualityServiceBase

logger = logging.getLogger(__name__)


class ValidationMixin(WorldQualityServiceBase):
    """Mixin providing entity validation and consistency checking."""

    async def suggest_relationships_for_entity(
        self,
        entity: Entity,
        available_entities: list[Entity],
        existing_relationships: list[dict[str, str]],
        story_brief: StoryBrief | None,
        num_suggestions: int = 3,
    ) -> list[dict[str, Any]]:
        """Suggest meaningful relationships for an entity based on narrative context.

        Uses LLM to propose relationships based on entity descriptions,
        story context, and existing world structure.

        Args:
            entity: The entity to suggest relationships for.
            available_entities: Other entities that could be relationship targets.
            existing_relationships: List of existing relationships (as dicts with
                source_name, target_name, relation_type).
            story_brief: Story brief for context (optional).
            num_suggestions: Number of suggestions to generate.

        Returns:
            List of relationship suggestion dicts with keys:
            - target_entity_name: Name of suggested target
            - target_entity_id: ID of suggested target
            - relation_type: Type of relationship
            - description: Relationship description
            - confidence: Confidence score (0.0-1.0)
            - bidirectional: Whether relationship is bidirectional
        """
        logger.info(f"Suggesting relationships for entity: {entity.name}")

        # Filter out the entity itself from available entities
        targets = [e for e in available_entities if e.id != entity.id]
        if not targets:
            logger.debug("No available target entities for relationship suggestions")
            return []

        # Get relationship minimums from settings - log warning if not configured
        entity_type = entity.type.lower()
        entity_role = (entity.attributes or {}).get("role", "default")
        minimums = self.settings.relationship_minimums.get(entity_type, {})

        if entity_type not in self.settings.relationship_minimums:
            logger.warning(
                f"No relationship_minimums configured for entity type '{entity_type}', "
                f"using fallback. Consider adding to settings.relationship_minimums."
            )

        if entity_role in minimums:
            min_rels = minimums[entity_role]
        elif "default" in minimums:
            min_rels = minimums["default"]
            logger.debug(f"Role '{entity_role}' not in minimums for '{entity_type}', using default")
        else:
            # No configuration found - log warning and use sensible default
            min_rels = 2
            logger.warning(
                f"No relationship minimum configured for {entity_type}/{entity_role}, "
                f"using fallback value of 2"
            )

        # Enforce max_relationships_per_entity by limiting suggestions
        max_rels = self.settings.max_relationships_per_entity
        current_rel_count = len(
            [
                r
                for r in existing_relationships
                if r.get("source_name") == entity.name or r.get("target_name") == entity.name
            ]
        )
        effective_suggestions = min(num_suggestions, max(0, max_rels - current_rel_count))

        if effective_suggestions <= 0:
            logger.debug(f"Entity {entity.name} already at max relationships ({max_rels})")
            return []

        # Build prompt with entity IDs for stable reference
        entities_context = "\n".join(
            f"- [{e.id}] {e.name} ({e.type}): {e.description[:100]}..."
            for e in targets[:20]  # Limit to avoid context overflow
        )

        existing_rels_context = "\n".join(
            f"- {r.get('relation_type', 'unknown')}: {r.get('source_name')} -> {r.get('target_name')}"
            for r in existing_relationships[:15]
        )

        prompt = f"""Suggest meaningful relationships for this entity.

TARGET ENTITY:
Name: {entity.name}
Type: {entity.type}
Description: {entity.description}

AVAILABLE ENTITIES FOR RELATIONSHIPS (use the bracketed ID when specifying targets):
{entities_context}

{"EXISTING RELATIONSHIPS (avoid duplicates):" + chr(10) + existing_rels_context if existing_relationships else ""}

STORY CONTEXT:
- Premise: {story_brief.premise[:100] if story_brief and story_brief.premise else "Unknown"}...
- Genre: {story_brief.genre if story_brief else "Unknown"}
- Themes: {story_brief.themes if story_brief else "Unknown"}

Suggest {effective_suggestions} new relationships for "{entity.name}".
Consider: allies, rivals, family, romance, professional ties, hidden connections.
Minimum recommended relationships for {entity_type}/{entity_role}: {min_rels}
Maximum relationships per entity: {max_rels} (current: {current_rel_count})

IMPORTANT: Respond with ONLY the JSON object below. No markdown code blocks, no explanations, no additional text.

{{
    "suggestions": [
        {{
            "target_entity_id": "uuid-from-brackets",
            "target_entity_name": "Entity Name",
            "relation_type": "...",
            "description": "...",
            "confidence": 0.85,
            "bidirectional": true
        }}
    ]
}}"""

        try:
            model_id = self._get_creator_model(entity_type="relationship")
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": config.creator_temperature,
                    "num_predict": 1024,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response for relationship suggestions for {entity.name}")
                return []

            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse relationship suggestions for {entity.name}")
                return []

            suggestions = result.get("suggestions", [])

            # Enrich suggestions with target entity IDs using ID-first, then fuzzy matching
            enriched_suggestions = []
            fuzzy_threshold = self.settings.fuzzy_match_threshold
            for suggestion in suggestions:
                target_id = suggestion.get("target_entity_id", "")
                target_name = suggestion.get("target_entity_name", "")

                # First try: resolve by ID if provided
                target_entity = None
                if target_id:
                    target_entity = next((e for e in targets if e.id == target_id), None)

                # Second try: exact case-insensitive name match
                if not target_entity and target_name:
                    target_entity = next(
                        (e for e in targets if e.name.lower() == target_name.lower()),
                        None,
                    )

                # Third try: fuzzy name matching
                if not target_entity and target_name:
                    from difflib import SequenceMatcher

                    best_match = None
                    best_score = 0.0
                    search_name = target_name.lower().strip()
                    for e in targets:
                        entity_name = e.name.lower().strip()
                        score = SequenceMatcher(None, search_name, entity_name).ratio()
                        if score > best_score and score >= fuzzy_threshold:
                            best_score = score
                            best_match = e
                    if best_match:
                        logger.debug(
                            f"Fuzzy matched '{target_name}' -> '{best_match.name}' (score={best_score:.2f})"
                        )
                        target_entity = best_match

                if target_entity:
                    # Coerce confidence to float with fallback and clamping
                    raw_confidence = suggestion.get("confidence", 0.5)
                    try:
                        confidence = float(raw_confidence)
                    except (TypeError, ValueError):
                        confidence = 0.5
                    confidence = max(0.0, min(1.0, confidence))

                    # Coerce bidirectional to boolean
                    raw_bidirectional = suggestion.get("bidirectional", False)
                    if isinstance(raw_bidirectional, str):
                        bidirectional = raw_bidirectional.strip().lower() in ("true", "1", "yes")
                    elif isinstance(raw_bidirectional, (int, bool)):
                        bidirectional = bool(raw_bidirectional)
                    else:
                        bidirectional = False

                    enriched_suggestions.append(
                        {
                            "source_entity_id": entity.id,
                            "source_entity_name": entity.name,
                            "target_entity_id": target_entity.id,
                            "target_entity_name": target_entity.name,
                            "relation_type": suggestion.get("relation_type", "related_to"),
                            "description": suggestion.get("description", ""),
                            "confidence": confidence,
                            "bidirectional": bidirectional,
                        }
                    )
                else:
                    logger.warning(
                        f"Could not resolve target entity: id={target_id}, name={target_name}"
                    )

            logger.info(
                f"Generated {len(enriched_suggestions)} relationship suggestions for {entity.name}"
            )
            return enriched_suggestions

        except Exception as e:
            logger.exception(f"Failed to suggest relationships for {entity.name}: {e}")
            return []

    async def extract_entity_claims(
        self,
        entity: Entity,
    ) -> list[dict[str, str]]:
        """Extract factual claims from an entity's description and attributes.

        Args:
            entity: Entity to extract claims from.

        Returns:
            List of claim dicts with keys: entity_id, entity_name, entity_type, claim, source_text.
        """
        logger.debug(f"Extracting claims from entity: {entity.name}")

        prompt = f"""Extract key factual claims from this entity.

ENTITY:
Name: {entity.name}
Type: {entity.type}
Description: {entity.description}
Attributes: {entity.attributes}

Extract 3-5 factual claims about this entity (things that must be true about them).
Focus on: identity, relationships, abilities, history, location, appearance.

Return JSON:
{{
    "claims": [
        "Claim 1 (e.g., 'Character X is the king of Y')",
        "Claim 2 (e.g., 'Character X has blue eyes')",
        "Claim 3"
    ]
}}"""

        try:
            model_id = self._get_judge_model(entity_type=entity.type.lower())

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            )

            response_text = str(response.get("response", ""))
            result = extract_json(response_text)

            if not result or not isinstance(result, dict):
                return []

            claims_list = result.get("claims", [])

            # Type check: ensure claims_list is actually a list before iterating
            if not isinstance(claims_list, (list, tuple)):
                logger.warning(
                    f"Expected claims to be a list but got {type(claims_list).__name__} "
                    f"for entity {entity.name} (id={entity.id}). Response: {response_text[:200]}"
                )
                return []

            return [
                {
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "entity_type": entity.type,
                    "claim": claim,
                    "source_text": entity.description[:200],
                }
                for claim in claims_list
                if isinstance(claim, str)
            ]

        except Exception as e:
            logger.exception(f"Failed to extract claims from {entity.name}: {e}")
            return []

    async def check_contradiction(
        self,
        claim_a: dict[str, str],
        claim_b: dict[str, str],
    ) -> dict[str, Any] | None:
        """Check if two claims contradict each other.

        Args:
            claim_a: First claim dict.
            claim_b: Second claim dict.

        Returns:
            Contradiction dict if found, None otherwise.
        """
        prompt = f"""Analyze whether these two claims contradict each other.

CLAIM A:
Entity: {claim_a.get("entity_name")} ({claim_a.get("entity_type")})
Claim: {claim_a.get("claim")}

CLAIM B:
Entity: {claim_b.get("entity_name")} ({claim_b.get("entity_type")})
Claim: {claim_b.get("claim")}

Analyze for contradictions considering:
1. Direct logical contradictions
2. Temporal inconsistencies
3. Character/attribute conflicts
4. Location/geography conflicts
5. Relationship contradictions

Return JSON:
{{
    "is_contradiction": true/false,
    "severity": "low" | "medium" | "high" | "critical",
    "explanation": "Why these claims do or don't contradict",
    "resolution_suggestion": "How to resolve if contradictory",
    "confidence": 0.0-1.0
}}"""

        try:
            model_id = self._get_judge_model(entity_type="validator")

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            )

            response_text = str(response.get("response", ""))
            result = extract_json(response_text)

            if not result or not isinstance(result, dict):
                return None

            # Properly coerce is_contradiction to boolean (string "false" is truthy!)
            raw_is_contradiction = result.get("is_contradiction", False)
            if isinstance(raw_is_contradiction, str):
                is_contradiction = raw_is_contradiction.strip().lower() in ("true", "1", "yes")
            elif isinstance(raw_is_contradiction, (int, bool)):
                is_contradiction = bool(raw_is_contradiction)
            else:
                is_contradiction = False

            if is_contradiction:
                # Coerce confidence to float with fallback and clamping
                try:
                    confidence = float(result.get("confidence", 0.5))
                except (TypeError, ValueError):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))

                return {
                    "claim_a": claim_a,
                    "claim_b": claim_b,
                    "severity": result.get("severity", "medium"),
                    "explanation": result.get("explanation", ""),
                    "resolution_suggestion": result.get("resolution_suggestion", ""),
                    "confidence": confidence,
                }

            return None

        except Exception as e:
            logger.exception("Failed to check contradiction: %s", e)
            return None

    async def validate_entity_consistency(
        self,
        entities: list[Entity],
        max_comparisons: int = 50,
    ) -> list[dict[str, Any]]:
        """Validate consistency across entities by detecting contradictions.

        Extracts claims from each entity and cross-references them pairwise
        to detect contradictions.

        Args:
            entities: List of entities to validate.
            max_comparisons: Maximum number of claim pairs to compare (for performance).

        Returns:
            List of detected contradictions.
        """
        logger.info(f"Validating consistency across {len(entities)} entities")

        # Extract claims from all entities
        all_claims: list[dict[str, str]] = []
        for entity in entities:
            claims = await self.extract_entity_claims(entity)
            all_claims.extend(claims)

        logger.debug(f"Extracted {len(all_claims)} claims from {len(entities)} entities")

        if len(all_claims) < 2:
            return []

        # Compare claims pairwise (with limit for performance)
        contradictions: list[dict[str, Any]] = []
        comparisons_made = 0

        for i, claim_a in enumerate(all_claims):
            for claim_b in all_claims[i + 1 :]:
                if comparisons_made >= max_comparisons:
                    break

                # Skip comparing claims from the same entity
                if claim_a.get("entity_id") == claim_b.get("entity_id"):
                    continue

                contradiction = await self.check_contradiction(claim_a, claim_b)
                if contradiction:
                    contradictions.append(contradiction)

                comparisons_made += 1

            if comparisons_made >= max_comparisons:
                break

        logger.info(f"Found {len(contradictions)} contradictions in {comparisons_made} comparisons")
        return contradictions
