"""Validation, mini-description, entity refinement/regeneration, relationship suggestion, and contradiction detection functions."""

import asyncio
import logging
import time
from typing import Any

from src.memory.entities import Entity
from src.memory.story_state import StoryBrief
from src.memory.world_quality import RefinementConfig
from src.utils.json_parser import clean_llm_text, extract_json

logger = logging.getLogger(__name__)


def generate_mini_description(
    svc,
    name: str,
    entity_type: str,
    full_description: str,
) -> str:
    """Generate a short 10-15 word mini description for hover tooltips.

    Uses a fast model with low temperature for consistent, concise output.

    Args:
        svc: WorldQualityService instance.
        name: Entity name.
        entity_type: Type of entity (character, location, etc.).
        full_description: Full description to summarize.

    Returns:
        A short summary (configured word limit).
    """
    max_words = svc.settings.mini_description_words_max
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
        model = svc._get_judge_model(entity_type=entity_type)  # Use fast validator model
        logger.debug(
            "Generating mini description for %s '%s' (model=%s)",
            entity_type,
            name,
            model,
        )
        response = svc.client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": svc.settings.world_quality_judge_temp,
                "num_predict": svc.settings.llm_tokens_mini_description,
            },
        )
        summary: str = str(response["response"]).strip()
        # Clean think tags and LLM artifacts
        summary = clean_llm_text(summary)
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
    svc,
    entities: list[dict[str, Any]],
) -> dict[str, str]:
    """Generate mini descriptions for a batch of entities.

    Args:
        svc: WorldQualityService instance.
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

        logger.debug(f"Generating mini description {i + 1}/{total_count}: {entity_type} '{name}'")

        mini_desc = svc.generate_mini_description(name, entity_type, description)
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
    svc,
    entity: Entity | None,
    story_brief: StoryBrief | None,
) -> dict[str, Any] | None:
    """Refine an existing entity to improve its quality.

    Uses the quality service's refinement loop with the existing entity as a base.

    Args:
        svc: WorldQualityService instance.
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
        model_id = svc._get_creator_model(entity_type)
        config = RefinementConfig.from_settings(svc.settings)

        logger.debug(
            "Refining entity '%s' via LLM (model=%s, temp=%.1f)",
            entity.name,
            model_id,
            config.refinement_temperature,
        )
        response = await asyncio.to_thread(
            svc.client.generate,
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
            logger.warning(
                "Failed to parse refinement response for %s: %s",
                entity.name,
                response_text[:200],
            )
            return None

        logger.info(f"Successfully refined {entity_type}: {entity.name}")
        return result

    except Exception as e:
        logger.exception(f"Failed to refine entity {entity.name}: {e}")
        return None


async def regenerate_entity(
    svc,
    entity: Entity | None,
    story_brief: StoryBrief | None,
    custom_instructions: str | None = None,
) -> dict[str, Any] | None:
    """Fully regenerate an entity with AI.

    Creates a new version of the entity while preserving its role in the story.

    Args:
        svc: WorldQualityService instance.
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
        model_id = svc._get_creator_model(entity_type)
        config = RefinementConfig.from_settings(svc.settings)

        logger.debug(
            "Regenerating entity '%s' via LLM (model=%s, temp=%.1f)",
            entity.name,
            model_id,
            config.creator_temperature,
        )
        response = await asyncio.to_thread(
            svc.client.generate,
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
            logger.warning(
                "Failed to parse regeneration response for %s: %s",
                entity.name,
                response_text[:200],
            )
            return None

        logger.info(
            f"Successfully regenerated {entity_type}: {entity.name} -> {result.get('name', 'unknown')}"
        )
        return result

    except Exception as e:
        logger.exception(f"Failed to regenerate entity {entity.name}: {e}")
        return None


async def suggest_relationships_for_entity(
    svc,
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
        svc: WorldQualityService instance.
        entity: The entity to suggest relationships for.
        available_entities: Other entities that could be relationship targets.
        existing_relationships: List of existing relationships (as dicts with
            source_name, target_name, relation_type).
        story_brief: Story brief for context (optional).
        num_suggestions: Number of suggestions to generate.

    Returns:
        List of relationship suggestion dicts.
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
    minimums = svc.settings.relationship_minimums.get(entity_type, {})

    if entity_type not in svc.settings.relationship_minimums:
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
    max_rels = svc.settings.max_relationships_per_entity
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
        model_id = svc._get_creator_model(entity_type="relationship")
        config = RefinementConfig.from_settings(svc.settings)

        logger.debug(
            "Suggesting relationships for '%s' via LLM (model=%s)",
            entity.name,
            model_id,
        )
        response = await asyncio.to_thread(
            svc.client.generate,
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
            logger.warning(
                "Failed to parse relationship suggestions for %s: %s",
                entity.name,
                response_text[:200],
            )
            return []

        suggestions = result.get("suggestions", [])

        # Enrich suggestions with target entity IDs using ID-first, then fuzzy matching
        enriched_suggestions = []
        fuzzy_threshold = svc.settings.fuzzy_match_threshold
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
    svc,
    entity: Entity,
) -> list[dict[str, str]]:
    """Extract factual claims from an entity's description and attributes.

    Args:
        svc: WorldQualityService instance.
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
        model_id = svc._get_judge_model(entity_type=entity.type.lower())

        logger.debug(
            "Extracting claims for '%s' via LLM (model=%s)",
            entity.name,
            model_id,
        )
        response = await asyncio.to_thread(
            svc.client.generate,
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
            logger.warning(
                "Failed to parse claims JSON for '%s': %s",
                entity.name,
                response_text[:200],
            )
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
    svc,
    claim_a: dict[str, str],
    claim_b: dict[str, str],
) -> dict[str, Any] | None:
    """Check if two claims contradict each other.

    Args:
        svc: WorldQualityService instance.
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
        model_id = svc._get_judge_model(entity_type="validator")

        logger.debug("Checking contradiction via LLM (model=%s)", model_id)
        response = await asyncio.to_thread(
            svc.client.generate,
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
            logger.warning("Failed to parse contradiction JSON: %s", response_text[:200])
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
    svc,
    entities: list[Entity],
    max_comparisons: int = 50,
) -> list[dict[str, Any]]:
    """Validate consistency across entities by detecting contradictions.

    Extracts claims from each entity and cross-references them pairwise
    to detect contradictions.

    Args:
        svc: WorldQualityService instance.
        entities: List of entities to validate.
        max_comparisons: Maximum number of claim pairs to compare (for performance).

    Returns:
        List of detected contradictions.
    """
    logger.info(f"Validating consistency across {len(entities)} entities")

    # Extract claims from all entities
    all_claims: list[dict[str, str]] = []
    for entity in entities:
        claims = await extract_entity_claims(svc, entity)
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

            contradiction = await check_contradiction(svc, claim_a, claim_b)
            if contradiction:
                contradictions.append(contradiction)

            comparisons_made += 1

        if comparisons_made >= max_comparisons:
            break

    logger.info(f"Found {len(contradictions)} contradictions in {comparisons_made} comparisons")
    return contradictions
