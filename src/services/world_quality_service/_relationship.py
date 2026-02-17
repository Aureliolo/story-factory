"""Relationship generation, judgment, and refinement functions."""

import logging
from collections import Counter
from typing import Any

import ollama

from src.memory.conflict_types import (
    VALID_RELATIONSHIP_TYPES,
    ConflictCategory,
    classify_relationship,
    normalize_relation_type,
)
from src.memory.story_state import StoryState
from src.memory.world_quality import RelationshipQualityScores
from src.services.llm_client import generate_structured, warn_if_prompt_too_large
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error
from src.utils.json_parser import extract_json, unwrap_single_json

logger = logging.getLogger(__name__)


def generate_relationship_with_quality(
    svc,
    story_state: StoryState,
    entity_names: list[str],
    existing_rels: list[tuple[str, str, str]],
) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
    """Generate a relationship with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    relationships until the quality threshold is met or stopping criteria is reached.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        entity_names: Names of entities that can have relationships.
        existing_rels: Existing (source, target, relation_type) 3-tuples to avoid.

    Returns:
        Tuple of (relationship_dict, quality_scores, iterations_used).

    Raises:
        ValueError: If story_state.brief is missing or fewer than two entity names are provided.
        WorldGenerationError: If relationship generation fails after all retries.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for relationship generation")

    if len(entity_names) < 2:
        raise ValueError("Need at least 2 entities for relationship generation")

    # Track rejected duplicate pairs across loop iterations via mutable closure
    rejected_pairs: list[tuple[str, str, str]] = []

    def _create(retries: int) -> dict[str, Any]:
        """Create a new relationship, passing rejected pairs to avoid duplicates.

        Uses stable temperature (no escalation) because escalating temperature
        degrades JSON compliance for relationship generation.
        """
        combined_rels = existing_rels + rejected_pairs
        result: dict[str, Any] = svc._create_relationship(
            story_state,
            entity_names,
            combined_rels,
            config.creator_temperature,  # Stable: no escalation for relationships
        )
        return result

    def _is_empty(rel: dict[str, Any]) -> bool:
        """Check if relationship is empty or a duplicate that should be rejected.

        Returns True if source/target missing or if the pair already exists,
        tracking rejected pairs to avoid regenerating them.
        """
        if not rel.get("source") or not rel.get("target"):
            return True
        # Check for duplicate relationship (includes previously rejected pairs)
        source = rel.get("source", "")
        target = rel.get("target", "")
        raw_type = rel.get("relation_type") or "related_to"
        rel_type = normalize_relation_type(raw_type)
        combined_rels = existing_rels + rejected_pairs
        if _is_duplicate_relationship(source, target, combined_rels):
            logger.warning("Generated duplicate relationship %s -> %s, rejecting", source, target)
            rejected_pairs.append((source, target, rel_type))
            return True
        return False

    return quality_refinement_loop(
        entity_type="relationship",
        create_fn=_create,
        judge_fn=lambda rel: svc._judge_relationship_quality(
            rel,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda rel, scores, iteration: svc._refine_relationship(
            rel,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda rel: f"{rel.get('source', '?')} -> {rel.get('target', '?')}",
        serialize=lambda rel: rel.copy(),
        is_empty=_is_empty,
        score_cls=RelationshipQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def _is_duplicate_relationship(
    source: str,
    target: str,
    existing_rels: list[tuple[str, str, str]],
) -> bool:
    """Check if a relationship already exists between two entities (in either direction).

    Args:
        source: Source entity name.
        target: Target entity name.
        existing_rels: List of existing (source, target, relation_type) 3-tuples.

    Returns:
        True if this source/target pair already exists in either direction.
    """
    for existing_source, existing_target, _rel_type in existing_rels:
        # Check both directions
        same_pair = (source == existing_source and target == existing_target) or (
            source == existing_target and target == existing_source
        )
        if same_pair:
            return True
    return False


def _compute_diversity_hint(existing_types: list[str]) -> str:
    """Compute a diversity hint based on the distribution of relationship categories.

    When one conflict category is under-represented (< 15% of total),
    returns a hint suggesting the LLM prioritize that category.

    Args:
        existing_types: List of relationship type strings from existing relationships.

    Returns:
        A prompt hint string, or empty string if distribution is balanced or too few rels.
    """
    if len(existing_types) < 3:
        return ""

    cats = Counter(classify_relationship(t) for t in existing_types)
    total = sum(cats.values())

    # Check for under-represented categories (excluding NEUTRAL which is informational)
    category_hints = {
        ConflictCategory.RIVALRY: "conflicts, betrayals, or opposition",
        ConflictCategory.TENSION: "distrust, competition, or manipulation",
        ConflictCategory.ALLIANCE: "loyalty, cooperation, or protection",
    }
    for cat, examples in category_hints.items():
        if cats.get(cat, 0) / total < 0.15:
            logger.debug(
                "Diversity hint: %s under-represented (%d/%d = %.0f%%)",
                cat.value,
                cats.get(cat, 0),
                total,
                (cats.get(cat, 0) / total) * 100,
            )
            return (
                f"\nHINT: The world lacks {cat.value} relationships. "
                f"Prioritize creating a {cat.value}-type relationship ({examples})."
            )

    return ""


def _compute_entity_frequency_hint(
    entity_names: list[str],
    existing_rels: list[tuple[str, str, str]],
) -> str:
    """Compute a prompt hint highlighting under-connected and over-connected entities.

    Counts how many relationships each entity already appears in (as source or target)
    and generates guidance for the LLM to prioritize under-connected entities and avoid
    over-connected ones, reducing focal-character bias.

    Args:
        entity_names: All entity names available for relationships.
        existing_rels: Existing (source, target, relation_type) 3-tuples.

    Returns:
        A prompt hint string, or empty string if fewer than 3 entities or no relationships.
    """
    if len(entity_names) < 3 or not existing_rels:
        logger.debug(
            "Skipping entity frequency hint: %d entities, %d existing rels",
            len(entity_names),
            len(existing_rels),
        )
        return ""

    # Count how many relationships each entity participates in
    freq: Counter[str] = Counter()
    for source, target, _rel_type in existing_rels:
        if source in entity_names:
            freq[source] += 1
        if target in entity_names:
            freq[target] += 1

    # Identify under-connected (0-1 rels) and over-connected (4+) entities
    under_connected = sorted(name for name in entity_names if freq[name] <= 1)
    over_connected = sorted(name for name in entity_names if freq[name] >= 4)

    if not under_connected and not over_connected:
        logger.debug("Entity frequency distribution is balanced, no hint needed")
        return ""

    parts: list[str] = ["\nENTITY CONNECTION BALANCE:"]
    if under_connected:
        logger.debug(
            "Under-connected entities (PRIORITY): %s",
            ", ".join(under_connected),
        )
        parts.append(
            "PRIORITY — these entities have few connections, prefer using them: "
            + ", ".join(under_connected)
        )
    if over_connected:
        logger.debug(
            "Over-connected entities (AVOID): %s",
            ", ".join(over_connected),
        )
        parts.append(
            "AVOID — these entities already have many connections: " + ", ".join(over_connected)
        )

    return "\n".join(parts)


def _create_relationship(
    svc,
    story_state: StoryState,
    entity_names: list[str],
    existing_rels: list[tuple[str, str, str]],
    temperature: float,
) -> dict[str, Any]:
    """Create a new relationship using the creator model.

    Args:
        svc: WorldQualityService instance for model selection and LLM calls.
        story_state: Story state with brief; returns ``{}`` if brief is missing.
        entity_names: Available entity names to choose as source and target.
        existing_rels: Existing (source, target, relation_type) 3-tuples used to
            avoid duplicates and inform diversity hints.
        temperature: Sampling temperature for the creator model.

    Returns:
        Relationship dict with keys ``source``, ``target``, ``relation_type``
        (normalized to controlled vocabulary), and ``description``.

    Raises:
        WorldGenerationError: If the creator model fails or returns unparsable JSON.
    """
    logger.debug(
        "Creating relationship for story %s (%d entities available)",
        story_state.id,
        len(entity_names),
    )
    brief = story_state.brief
    if not brief:
        return {}

    # Build forbidden pairs block with direction and type for clarity
    existing_rel_strs = [f"- {s} -> {t} ({rt})" for s, t, rt in existing_rels]
    existing_pairs_block = "\n".join(existing_rel_strs) if existing_rel_strs else "None"

    type_list = ", ".join(VALID_RELATIONSHIP_TYPES)

    # Compute diversity hint from existing relationship types
    existing_types = [rel_type for _s, _t, rel_type in existing_rels if rel_type]
    diversity_hint = _compute_diversity_hint(existing_types)

    # Compute entity frequency hint to reduce focal-character bias
    frequency_hint = _compute_entity_frequency_hint(entity_names, existing_rels)

    # For small entity sets, compute and list valid unused pairs
    unused_pairs_block = ""
    if len(entity_names) <= 20:
        from itertools import permutations

        existing_pair_set = set()
        for s, t, _rt in existing_rels:
            existing_pair_set.add((s, t))
            existing_pair_set.add((t, s))

        # Count connections per entity for sorting
        entity_freq: Counter[str] = Counter()
        for s, t, _rt in existing_rels:
            if s in entity_names:
                entity_freq[s] += 1
            if t in entity_names:
                entity_freq[t] += 1

        # Sort unused pairs so under-connected entities appear first
        raw_unused = [
            (a, b) for a, b in permutations(entity_names, 2) if (a, b) not in existing_pair_set
        ]
        raw_unused.sort(key=lambda pair: entity_freq[pair[0]] + entity_freq[pair[1]])
        unused_pairs = [f"- {a} -> {b}" for a, b in raw_unused[:30]]

        if unused_pairs:
            unused_pairs_block = (
                "\n\nAVAILABLE UNUSED PAIRS (choose ONLY from these):\n" + "\n".join(unused_pairs)
            )

    prompt = f"""Create a compelling relationship between entities for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

AVAILABLE ENTITIES: {", ".join(entity_names)}

FORBIDDEN ENTITY PAIRS (you MUST NOT use any of these pairs in either direction):
{existing_pairs_block}{unused_pairs_block}

Create a relationship with:
1. Tension - conflict potential
2. Complex dynamics - power balance, history
3. Story potential - opportunities for scenes
4. Authenticity - believable connection

IMPORTANT: Return exactly ONE relationship as a single JSON object. Do NOT return an array.

Output ONLY valid JSON (all text in {brief.language}):
{{
    "source": "Entity Name 1",
    "target": "Entity Name 2",
    "relation_type": "One of: {type_list}",
    "description": "Description of the relationship with history and dynamics"
}}{diversity_hint}{frequency_hint}"""

    try:
        model = svc._get_creator_model(entity_type="relationship")

        # Warn proactively if prompt may exceed context window
        warn_if_prompt_too_large(
            prompt, model, svc.settings.context_size, svc.settings.llm_tokens_relationship_create
        )

        response = svc.client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": svc.settings.llm_tokens_relationship_create,
            },
        )

        raw_response = response["response"]
        data = extract_json(raw_response, strict=False)
        data = unwrap_single_json(data)
        if data and isinstance(data, dict):
            # Normalize relation_type to controlled vocabulary
            raw_type = data.get("relation_type", "related_to")
            data["relation_type"] = normalize_relation_type(raw_type)
            result: dict[str, Any] = data
            return result
        else:
            # Detect likely truncation: unbalanced braces suggest output was cut off
            open_braces = raw_response.count("{") - raw_response.count("}")
            if open_braces > 0:
                logger.error(
                    "Relationship creation JSON appears truncated "
                    "(unbalanced braces: %d unclosed). "
                    "Token limit llm_tokens_relationship_create=%d may be too low. "
                    "Raw response tail: ...%s",
                    open_braces,
                    svc.settings.llm_tokens_relationship_create,
                    raw_response[-100:],
                )
            else:
                logger.error(f"Relationship creation returned invalid JSON structure: {data}")
            raise WorldGenerationError(f"Invalid relationship JSON structure: {data}")
    except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
        logger.error("Relationship creation LLM error for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"LLM error during relationship creation: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error("Relationship creation JSON parsing error for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"Invalid relationship response format: {e}") from e
    except WorldGenerationError:
        # Re-raise domain exceptions as-is
        raise
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Unexpected error in relationship creation for story %s: %s", story_state.id, summary
        )
        raise WorldGenerationError(f"Unexpected relationship creation error: {summary}") from e


def _judge_relationship_quality(
    svc,
    relationship: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> RelationshipQualityScores:
    """Judge relationship quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.

    Args:
        svc: WorldQualityService instance.
        relationship: Relationship dict to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        RelationshipQualityScores with ratings and feedback.

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

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- tension: Conflict potential
- dynamics: Complexity, power balance, history
- story_potential: Opportunities for scenes and development
- authenticity: Believability of the connection — do their shared experiences, motivations, and emotional responses feel earned and internally consistent?

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"tension": <float 0-10>, "dynamics": <float 0-10>, "story_potential": <float 0-10>, "authenticity": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="relationship")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> RelationshipQualityScores:
        """Execute a single judge call for relationship quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=RelationshipQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Relationship quality judgment failed for %s->%s: %s",
                    relationship.get("source") or "Unknown",
                    relationship.get("target") or "Unknown",
                    summary,
                )
            else:
                logger.error(
                    "Relationship quality judgment failed for %s->%s: %s",
                    relationship.get("source") or "Unknown",
                    relationship.get("target") or "Unknown",
                    summary,
                )
            raise WorldGenerationError(f"Relationship quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, RelationshipQualityScores, judge_config)


def _refine_relationship(
    svc,
    relationship: dict[str, Any],
    scores: RelationshipQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine a relationship based on quality feedback."""
    logger.debug(
        "Refining relationship '%s' <-> '%s' for story %s",
        relationship.get("source", "?"),
        relationship.get("target", "?"),
        story_state.id,
    )
    brief = story_state.brief
    threshold = svc.get_config().get_threshold("relationship")

    # Build specific improvement instructions from feedback
    improvement_focus = []
    if scores.tension < threshold:
        improvement_focus.append("Add competing interests, unresolved grievances, power imbalances")
    if scores.dynamics < threshold:
        improvement_focus.append("Add complexity — history, power shifts, secrets")
    if scores.story_potential < threshold:
        improvement_focus.append("Create more scene opportunities — betrayal, alliance, revelation")
    if scores.authenticity < threshold:
        improvement_focus.append(
            "Add shared history, believable motivation for the bond — "
            "ensure emotional responses feel earned and internally consistent"
        )

    prompt = f"""TASK: Improve this relationship to score HIGHER on the weak dimensions.

ORIGINAL RELATIONSHIP:
Source: {relationship.get("source", "Unknown")}
Target: {relationship.get("target", "Unknown")}
Type: {relationship.get("relation_type", "unknown")}
Description: {relationship.get("description", "")}

CURRENT SCORES (need {threshold}+ in all areas):
- Tension: {scores.tension}/10
- Dynamics: {scores.dynamics}/10
- Story Potential: {scores.story_potential}/10
- Authenticity: {scores.authenticity}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep source: "{relationship.get("source", "Unknown")}"
2. Keep target: "{relationship.get("target", "Unknown")}"
3. Keep type: "{relationship.get("relation_type", "knows")}"
4. Make SUBSTANTIAL improvements to weak areas
5. Add concrete details, not vague generalities
6. Output in {brief.language if brief else "English"}

Output ONLY valid JSON:
{{
    "source": "{relationship.get("source", "Unknown")}",
    "target": "{relationship.get("target", "Unknown")}",
    "relation_type": "{relationship.get("relation_type", "knows")}",
    "description": "Improved description with more depth"
}}"""

    try:
        model = svc._get_creator_model(entity_type="relationship")

        # Warn proactively if prompt may exceed context window
        warn_if_prompt_too_large(
            prompt, model, svc.settings.context_size, svc.settings.llm_tokens_relationship_refine
        )

        response = svc.client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={
                "temperature": temperature,
                "num_predict": svc.settings.llm_tokens_relationship_refine,
            },
        )

        data = extract_json(response["response"], strict=False)
        if data and isinstance(data, dict):
            raw_type = data.get("relation_type", "related_to")
            data["relation_type"] = normalize_relation_type(raw_type)
            result: dict[str, Any] = data
            return result
        else:
            logger.error(f"Relationship refinement returned invalid JSON structure: {data}")
            raise WorldGenerationError(f"Invalid relationship refinement JSON structure: {data}")
    except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
        logger.error(
            "Relationship refinement LLM error for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"LLM error during relationship refinement: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error(
            "Relationship refinement JSON parsing error for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"Invalid relationship refinement response format: {e}") from e
    except WorldGenerationError:
        # Re-raise domain exceptions as-is
        raise
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Unexpected error in relationship refinement for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            summary,
        )
        raise WorldGenerationError(f"Unexpected relationship refinement error: {summary}") from e
