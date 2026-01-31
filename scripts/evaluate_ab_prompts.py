#!/usr/bin/env python3
"""A/B test current vs improved refine prompts on the same seed entities.

Tests current refine prompts vs improved variants to determine whether
prompt changes can improve refinement effectiveness.

Protocol:
1. Generate N seed entities per type
2. Judge each seed -> baseline score
3. Refine with CURRENT production prompt -> judge -> score A
4. Refine with IMPROVED prompt variant -> judge -> score B
5. Compare A vs B

Usage:
    python scripts/evaluate_ab_prompts.py [options]
      --entity-types faction,concept  (default: faction, concept, item, location)
      --count-per-type 3              (default: 3)
      --output results.json           (default: output/diagnostics/<timestamp>_ab.json)
      --verbose
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_refinement import (
    ALL_ENTITY_TYPES,
    make_canonical_brief,
    make_story_state,
)
from src.memory.story_state import (
    Character,
    Concept,
    Faction,
    Item,
    Location,
    StoryState,
)
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
)
from src.services import ServiceContainer
from src.services.llm_client import generate_structured
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings

logger = logging.getLogger(__name__)

# Default: test the 4 dict-based entity types (Type B in the plan) plus character (Type A)
DEFAULT_AB_TYPES = ["character", "faction", "concept", "item", "location"]


def _format_improvements(
    improvement_focus: list[str], fallback: str = "- Minor enhancements only"
) -> str:
    """Format improvement focus items as a newline-separated list.

    Args:
        improvement_focus: List of improvement instruction strings.
        fallback: Text to use when no improvements are needed.

    Returns:
        Formatted string with one improvement per line, prefixed with "- ".
    """
    if not improvement_focus:
        return fallback
    return "\n".join(f"- {imp}" for imp in improvement_focus)


def improved_refine_character(
    svc: WorldQualityService,
    character: Character,
    scores: CharacterQualityScores,
    story_state: StoryState,
    temperature: float,
) -> Character:
    """Improved character refinement prompt (Type A fix).

    Changes vs production:
    - Adds numeric scores per dimension
    - Adds explicit threshold target matching actual config (not 9+)
    - Includes judge feedback more prominently
    - Derives specific improvement instructions from dimension scores
    """
    brief = story_state.brief
    config = svc.get_config()
    threshold = config.quality_threshold

    # Build actionable improvement instructions from scores (like Type B does)
    improvement_focus = []
    if scores.depth < threshold:
        improvement_focus.append(
            f"DEPTH ({scores.depth}/10): Add more internal contradictions, hidden motivations, "
            f"or psychological layers"
        )
    if scores.goals < threshold:
        improvement_focus.append(
            f"GOALS ({scores.goals}/10): Clarify what the character wants vs what they need; "
            f"make goals more story-relevant"
        )
    if scores.flaws < threshold:
        improvement_focus.append(
            f"FLAWS ({scores.flaws}/10): Add meaningful vulnerabilities that drive specific "
            f"conflicts, not just surface-level weaknesses"
        )
    if scores.uniqueness < threshold:
        improvement_focus.append(
            f"UNIQUENESS ({scores.uniqueness}/10): Differentiate from genre archetypes with "
            f"unexpected traits, background, or worldview"
        )
    if scores.arc_potential < threshold:
        improvement_focus.append(
            f"ARC POTENTIAL ({scores.arc_potential}/10): Create clearer transformation paths "
            f"with specific catalysts for change"
        )

    # Format relationships as "Name: description" lines
    rel_lines = (
        "\n".join(f"  {name}: {desc}" for name, desc in character.relationships.items())
        if character.relationships
        else "  (none yet)"
    )

    prompt = f"""TASK: Improve this character to meet quality threshold {threshold}/10 average.

ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Relationships:
{rel_lines}
Arc Notes: {character.arc_notes}

CURRENT SCORES (target: {threshold}+ average, currently {scores.average:.1f}):
- Depth: {scores.depth}/10
- Goals: {scores.goals}/10
- Flaws: {scores.flaws}/10
- Uniqueness: {scores.uniqueness}/10
- Arc Potential: {scores.arc_potential}/10

JUDGE'S SPECIFIC FEEDBACK (address each point):
{scores.feedback}

PRIORITY IMPROVEMENTS (focus on these dimensions):
{_format_improvements(improvement_focus, "- All dimensions above threshold, make minor enhancements")}

REQUIREMENTS:
1. Keep the name "{character.name}" and role "{character.role}"
2. Make SUBSTANTIAL changes to description, traits, goals, and arc_notes
3. Do NOT just rephrase existing content - add NEW details and complexity
4. The "relationships" field must be a flat dict mapping character names to relationship descriptions (e.g. {{"Sera": "Trusted mentor"}}) - NOT lists, NOT "Allies"/"Enemies" categories
5. Write all text in {brief.language if brief else "English"}"""

    model = svc._get_creator_model(entity_type="character")
    return generate_structured(
        settings=svc.settings,
        model=model,
        prompt=prompt,
        response_model=Character,
        temperature=temperature,
    )


def improved_refine_faction(
    svc: WorldQualityService,
    faction: dict[str, Any],
    scores: FactionQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Improved faction refinement prompt (Type B fix).

    Changes vs production:
    - Uses actual quality threshold instead of "need 9+"
    - Includes judge feedback more prominently with action items
    - Derives improvement instructions from feedback text
    """
    brief = story_state.brief
    config = svc.get_config()
    threshold = config.quality_threshold

    improvement_focus = []
    if scores.coherence < threshold:
        improvement_focus.append(
            f"COHERENCE ({scores.coherence}/10): Strengthen internal logic, add clearer "
            f"organizational structure and rules"
        )
    if scores.influence < threshold:
        improvement_focus.append(
            f"INFLUENCE ({scores.influence}/10): Expand world impact with specific power "
            f"mechanisms and territorial/political reach"
        )
    if scores.conflict_potential < threshold:
        improvement_focus.append(
            f"CONFLICT ({scores.conflict_potential}/10): Add internal tensions, rival "
            f"relationships, and specific friction points"
        )
    if scores.distinctiveness < threshold:
        improvement_focus.append(
            f"DISTINCTIVENESS ({scores.distinctiveness}/10): Add unique rituals, symbols, "
            f"jargon, or cultural practices that set this faction apart"
        )

    prompt = f"""TASK: Improve this faction to meet quality threshold {threshold}/10 average.
Current average: {scores.average:.1f}/10.

ORIGINAL FACTION:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

CURRENT SCORES (target: {threshold}+ average):
- Coherence: {scores.coherence}/10
- Influence: {scores.influence}/10
- Conflict Potential: {scores.conflict_potential}/10
- Distinctiveness: {scores.distinctiveness}/10

JUDGE'S SPECIFIC FEEDBACK (address each point):
{scores.feedback}

PRIORITY IMPROVEMENTS:
{_format_improvements(improvement_focus, "- All dimensions above threshold, make minor enhancements")}

REQUIREMENTS:
1. Keep the exact name: "{faction.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements - add new concrete details, not vague generalities
3. The description should be at least 3 sentences with specific world details
4. Output in {brief.language if brief else "English"}

Return ONLY the improved faction."""

    model = svc._get_creator_model(entity_type="faction")
    refined = generate_structured(
        settings=svc.settings,
        model=model,
        prompt=prompt,
        response_model=Faction,
        temperature=temperature,
    )
    result = refined.model_dump()
    result["name"] = faction.get("name", "Unknown")
    result["type"] = "faction"
    return result


def improved_refine_concept(
    svc: WorldQualityService,
    concept: dict[str, Any],
    scores: ConceptQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Improved concept refinement prompt."""
    brief = story_state.brief
    config = svc.get_config()
    threshold = config.quality_threshold

    improvement_focus = []
    if scores.relevance < threshold:
        improvement_focus.append(
            f"RELEVANCE ({scores.relevance}/10): Connect more explicitly to story themes"
        )
    if scores.depth < threshold:
        improvement_focus.append(
            f"DEPTH ({scores.depth}/10): Add philosophical nuance and paradoxes"
        )
    if scores.manifestation < threshold:
        improvement_focus.append(
            f"MANIFESTATION ({scores.manifestation}/10): Describe specific scenes or events "
            f"where this concept becomes tangible"
        )
    if scores.resonance < threshold:
        improvement_focus.append(
            f"RESONANCE ({scores.resonance}/10): Add universal human truths that readers "
            f"connect with emotionally"
        )

    prompt = f"""TASK: Improve this concept to meet quality threshold {threshold}/10 average.
Current average: {scores.average:.1f}/10.

ORIGINAL CONCEPT:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

CURRENT SCORES (target: {threshold}+ average):
- Relevance: {scores.relevance}/10
- Depth: {scores.depth}/10
- Manifestation: {scores.manifestation}/10
- Resonance: {scores.resonance}/10

JUDGE'S SPECIFIC FEEDBACK (address each point):
{scores.feedback}

PRIORITY IMPROVEMENTS:
{_format_improvements(improvement_focus)}

REQUIREMENTS:
1. Keep the exact name: "{concept.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements with concrete details
3. Output in {brief.language if brief else "English"}

Return ONLY the improved concept."""

    model = svc._get_creator_model(entity_type="concept")
    refined = generate_structured(
        settings=svc.settings,
        model=model,
        prompt=prompt,
        response_model=Concept,
        temperature=temperature,
    )
    result = refined.model_dump()
    result["name"] = concept.get("name", "Unknown")
    result["type"] = "concept"
    return result


def improved_refine_item(
    svc: WorldQualityService,
    item: dict[str, Any],
    scores: ItemQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Improved item refinement prompt."""
    brief = story_state.brief
    config = svc.get_config()
    threshold = config.quality_threshold

    improvement_focus = []
    if scores.significance < threshold:
        improvement_focus.append(
            f"SIGNIFICANCE ({scores.significance}/10): Tie the item to specific plot events "
            f"or character arcs"
        )
    if scores.uniqueness < threshold:
        improvement_focus.append(
            f"UNIQUENESS ({scores.uniqueness}/10): Add distinctive sensory details or unusual "
            f"properties"
        )
    if scores.narrative_potential < threshold:
        improvement_focus.append(
            f"NARRATIVE POTENTIAL ({scores.narrative_potential}/10): Create conflict around "
            f"possession, use, or discovery"
        )
    if scores.integration < threshold:
        improvement_focus.append(
            f"INTEGRATION ({scores.integration}/10): Ground in world lore, history, or culture"
        )

    prompt = f"""TASK: Improve this item to meet quality threshold {threshold}/10 average.
Current average: {scores.average:.1f}/10.

ORIGINAL ITEM:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {", ".join(item.get("properties", [])) if isinstance(item.get("properties"), list) else str(item.get("properties", ""))}

CURRENT SCORES (target: {threshold}+ average):
- Significance: {scores.significance}/10
- Uniqueness: {scores.uniqueness}/10
- Narrative Potential: {scores.narrative_potential}/10
- Integration: {scores.integration}/10

JUDGE'S SPECIFIC FEEDBACK (address each point):
{scores.feedback}

PRIORITY IMPROVEMENTS:
{_format_improvements(improvement_focus)}

REQUIREMENTS:
1. Keep the exact name: "{item.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements
3. Output in {brief.language if brief else "English"}

Return ONLY the improved item."""

    model = svc._get_creator_model(entity_type="item")
    refined = generate_structured(
        settings=svc.settings,
        model=model,
        prompt=prompt,
        response_model=Item,
        temperature=temperature,
    )
    result = refined.model_dump()
    result["name"] = item.get("name", "Unknown")
    result["type"] = "item"
    return result


def improved_refine_location(
    svc: WorldQualityService,
    location: dict[str, Any],
    scores: LocationQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Improved location refinement prompt."""
    brief = story_state.brief
    config = svc.get_config()
    threshold = config.quality_threshold

    improvement_focus = []
    if scores.atmosphere < threshold:
        improvement_focus.append(
            f"ATMOSPHERE ({scores.atmosphere}/10): Add specific sensory details "
            f"(sounds, smells, light, texture)"
        )
    if scores.significance < threshold:
        improvement_focus.append(
            f"SIGNIFICANCE ({scores.significance}/10): Connect to plot events or symbolic meaning"
        )
    if scores.story_relevance < threshold:
        improvement_focus.append(
            f"STORY RELEVANCE ({scores.story_relevance}/10): Link to character arcs "
            f"or thematic elements"
        )
    if scores.distinctiveness < threshold:
        improvement_focus.append(
            f"DISTINCTIVENESS ({scores.distinctiveness}/10): Add unique architectural, "
            f"natural, or cultural features"
        )

    prompt = f"""TASK: Improve this location to meet quality threshold {threshold}/10 average.
Current average: {scores.average:.1f}/10.

ORIGINAL LOCATION:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

CURRENT SCORES (target: {threshold}+ average):
- Atmosphere: {scores.atmosphere}/10
- Significance: {scores.significance}/10
- Story Relevance: {scores.story_relevance}/10
- Distinctiveness: {scores.distinctiveness}/10

JUDGE'S SPECIFIC FEEDBACK (address each point):
{scores.feedback}

PRIORITY IMPROVEMENTS:
{_format_improvements(improvement_focus)}

REQUIREMENTS:
1. Keep the exact name: "{location.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements with concrete sensory details
3. Output in {brief.language if brief else "English"}

Return ONLY the improved location."""

    model = svc._get_creator_model(entity_type="location")
    refined = generate_structured(
        settings=svc.settings,
        model=model,
        prompt=prompt,
        response_model=Location,
        temperature=temperature,
    )
    result = refined.model_dump()
    result["name"] = location.get("name", "Unknown")
    result["type"] = "location"
    return result


# Map entity types to improved refine functions
IMPROVED_REFINERS = {
    "character": improved_refine_character,
    "faction": improved_refine_faction,
    "concept": improved_refine_concept,
    "item": improved_refine_item,
    "location": improved_refine_location,
}


def run_ab_test(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
    entity_index: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run A/B test for a single entity.

    1. Create seed entity
    2. Judge baseline
    3. Refine with current (A) -> judge
    4. Refine with improved (B) -> judge
    5. Compare

    Args:
        svc_container: Service container.
        story_state: Story state.
        entity_type: Entity type.
        entity_index: Index for logging.
        verbose: Verbose output.

    Returns:
        Dict with A/B test results.
    """
    wqs = svc_container.world_quality
    config = wqs.get_config()
    existing_names: list[str] = []

    result: dict[str, Any] = {
        "entity_type": entity_type,
        "entity_index": entity_index,
        "entity_name": "",
        "baseline_score": None,
        "score_a_current": None,
        "score_b_improved": None,
        "delta_a": None,
        "delta_b": None,
        "delta_ab": None,
        "baseline_feedback": "",
        "feedback_a": "",
        "feedback_b": "",
        "error": None,
    }

    try:
        # Step 1: Create seed entity
        if entity_type == "character":
            entity_obj = wqs._create_character(
                story_state, existing_names, config.creator_temperature
            )
            entity_data = entity_obj.model_dump() if entity_obj else None
        elif entity_type == "faction":
            entity_data = wqs._create_faction(
                story_state, existing_names, config.creator_temperature
            )
        elif entity_type == "location":
            entity_data = wqs._create_location(
                story_state, existing_names, config.creator_temperature
            )
        elif entity_type == "item":
            entity_data = wqs._create_item(story_state, existing_names, config.creator_temperature)
        elif entity_type == "concept":
            entity_data = wqs._create_concept(
                story_state, existing_names, config.creator_temperature
            )
        else:
            result["error"] = f"A/B testing not supported for {entity_type}"
            return result

        if not entity_data:
            result["error"] = "Seed creation returned empty"
            return result

        name = entity_data.get("name", "Unknown")
        result["entity_name"] = name

        # Step 2: Judge baseline
        if entity_type == "character":
            baseline_scores = wqs._judge_character_quality(
                Character(**entity_data), story_state, config.judge_temperature
            )
        elif entity_type == "faction":
            baseline_scores = wqs._judge_faction_quality(
                entity_data, story_state, config.judge_temperature
            )
        elif entity_type == "location":
            baseline_scores = wqs._judge_location_quality(
                entity_data, story_state, config.judge_temperature
            )
        elif entity_type == "item":
            baseline_scores = wqs._judge_item_quality(
                entity_data, story_state, config.judge_temperature
            )
        elif entity_type == "concept":
            baseline_scores = wqs._judge_concept_quality(
                entity_data, story_state, config.judge_temperature
            )
        else:
            result["error"] = f"Judging not supported for {entity_type}"
            return result

        if baseline_scores is None:
            result["error"] = "Baseline judge returned None"
            logger.warning("Baseline judge failed for %s #%d", entity_type, entity_index)
            return result

        result["baseline_score"] = round(baseline_scores.average, 2)
        result["baseline_feedback"] = baseline_scores.feedback

        if verbose:
            print(f"    Baseline: {baseline_scores.average:.1f}")

        # Step 3: Refine with CURRENT production prompt (A)
        refine_temp = config.get_refinement_temperature(2)
        if entity_type == "character":
            refined_a_obj = wqs._refine_character(
                Character(**entity_data), baseline_scores, story_state, refine_temp
            )
            refined_a = refined_a_obj.model_dump() if refined_a_obj else None
        elif entity_type == "faction":
            refined_a = wqs._refine_faction(entity_data, baseline_scores, story_state, refine_temp)
        elif entity_type == "location":
            refined_a = wqs._refine_location(entity_data, baseline_scores, story_state, refine_temp)
        elif entity_type == "item":
            refined_a = wqs._refine_item(entity_data, baseline_scores, story_state, refine_temp)
        elif entity_type == "concept":
            refined_a = wqs._refine_concept(entity_data, baseline_scores, story_state, refine_temp)
        else:
            refined_a = None

        scores_a = None
        if refined_a:
            if entity_type == "character":
                scores_a = wqs._judge_character_quality(
                    Character(**refined_a), story_state, config.judge_temperature
                )
            elif entity_type == "faction":
                scores_a = wqs._judge_faction_quality(
                    refined_a, story_state, config.judge_temperature
                )
            elif entity_type == "location":
                scores_a = wqs._judge_location_quality(
                    refined_a, story_state, config.judge_temperature
                )
            elif entity_type == "item":
                scores_a = wqs._judge_item_quality(refined_a, story_state, config.judge_temperature)
            elif entity_type == "concept":
                scores_a = wqs._judge_concept_quality(
                    refined_a, story_state, config.judge_temperature
                )
            else:
                scores_a = baseline_scores

            if scores_a is None:
                logger.warning(
                    "Judge returned None for refined-A %s #%d, skipping A comparison",
                    entity_type,
                    entity_index,
                )
            else:
                result["score_a_current"] = round(scores_a.average, 2)
                result["feedback_a"] = scores_a.feedback
                result["delta_a"] = round(scores_a.average - baseline_scores.average, 2)

                if verbose:
                    print(
                        f"    Current (A): {scores_a.average:.1f} (delta={result['delta_a']:+.1f})"
                    )
        else:
            logger.warning(
                "Current refiner (A) returned None for %s #%d, skipping A comparison",
                entity_type,
                entity_index,
            )

        # Step 4: Refine with IMPROVED prompt (B) from same seed
        improved_refiner = IMPROVED_REFINERS.get(entity_type)
        if improved_refiner:
            if entity_type == "character":
                refined_b = improved_refiner(  # type: ignore[operator]
                    wqs, Character(**entity_data), baseline_scores, story_state, refine_temp
                )
                refined_b_data = (
                    refined_b.model_dump() if hasattr(refined_b, "model_dump") else refined_b
                )
            else:
                refined_b_data = improved_refiner(  # type: ignore[operator]
                    wqs, entity_data, baseline_scores, story_state, refine_temp
                )

            if refined_b_data:
                if entity_type == "character":
                    scores_b = wqs._judge_character_quality(
                        Character(**refined_b_data), story_state, config.judge_temperature
                    )
                elif entity_type == "faction":
                    scores_b = wqs._judge_faction_quality(
                        refined_b_data, story_state, config.judge_temperature
                    )
                elif entity_type == "location":
                    scores_b = wqs._judge_location_quality(
                        refined_b_data, story_state, config.judge_temperature
                    )
                elif entity_type == "item":
                    scores_b = wqs._judge_item_quality(
                        refined_b_data, story_state, config.judge_temperature
                    )
                elif entity_type == "concept":
                    scores_b = wqs._judge_concept_quality(
                        refined_b_data, story_state, config.judge_temperature
                    )
                else:
                    scores_b = baseline_scores

                if scores_b is None:
                    logger.warning(
                        "Judge returned None for refined-B %s #%d, skipping B comparison",
                        entity_type,
                        entity_index,
                    )
                else:
                    result["score_b_improved"] = round(scores_b.average, 2)
                    result["feedback_b"] = scores_b.feedback
                    result["delta_b"] = round(scores_b.average - baseline_scores.average, 2)
                    if scores_a is not None and result["score_a_current"] is not None:
                        result["delta_ab"] = round(scores_b.average - scores_a.average, 2)

                    if verbose:
                        ab_str = (
                            f"A-B={result['delta_ab']:+.1f}"
                            if result["delta_ab"] is not None
                            else "A=N/A"
                        )
                        print(
                            f"    Improved (B): {scores_b.average:.1f} "
                            f"(delta={result['delta_b']:+.1f}, {ab_str})"
                        )
        else:
            result["error"] = f"No improved refiner for {entity_type}"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("A/B test error for %s #%d: %s", entity_type, entity_index, e)

    return result


def main() -> None:
    """Main entry point for the A/B prompt evaluation script."""
    parser = argparse.ArgumentParser(description="A/B test current vs improved refine prompts.")
    parser.add_argument(
        "--entity-types",
        type=str,
        help=f"Comma-separated entity types (default: {', '.join(DEFAULT_AB_TYPES)})",
    )
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=3,
        help="Number of seed entities per type (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: output/diagnostics/<timestamp>_ab.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress to console",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse entity types
    if args.entity_types:
        entity_types = [t.strip() for t in args.entity_types.split(",")]
        invalid = [t for t in entity_types if t not in ALL_ENTITY_TYPES]
        if invalid:
            print(f"ERROR: Invalid entity types: {invalid}")
            sys.exit(1)
        # Filter to types that have improved refiners
        unsupported = [t for t in entity_types if t not in IMPROVED_REFINERS]
        if unsupported:
            print(f"WARNING: No improved refiner for: {unsupported}. Skipping.")
            entity_types = [t for t in entity_types if t in IMPROVED_REFINERS]
    else:
        entity_types = DEFAULT_AB_TYPES

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"ab_prompts_{timestamp}.json"

    # Initialize services
    print("Loading settings and initializing services...")
    settings = Settings.load()
    svc = ServiceContainer(settings)
    config = svc.world_quality.get_config()

    # Create canonical story state
    brief = make_canonical_brief()
    story_state = make_story_state(brief)

    print(f"Quality threshold: {config.quality_threshold}")
    print(f"Entity types: {entity_types}")
    print(f"Count per type: {args.count_per_type}")
    print(f"Output: {output_path}")
    print()

    # Run A/B tests
    ab_results: list[dict[str, Any]] = []
    total = len(entity_types) * args.count_per_type
    completed = 0

    for et in entity_types:
        print(f"--- A/B testing {et} ({args.count_per_type} seeds) ---")
        for i in range(args.count_per_type):
            completed += 1
            print(f"  [{completed}/{total}] {et} #{i + 1}...")
            result = run_ab_test(svc, story_state, et, i + 1, verbose=args.verbose)
            ab_results.append(result)

            if result.get("error"):
                print(f"    ERROR: {result['error']}")

    # Compute per-type summary
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in ab_results:
        by_type.setdefault(r["entity_type"], []).append(r)

    type_summary: dict[str, dict[str, Any]] = {}
    for et, results in by_type.items():
        valid = [r for r in results if not r.get("error")]
        if not valid:
            type_summary[et] = {"count": 0, "error": "All runs failed"}
            continue

        baselines = [r["baseline_score"] for r in valid if r["baseline_score"] is not None]
        deltas_a = [r["delta_a"] for r in valid if r["delta_a"] is not None]
        deltas_b = [r["delta_b"] for r in valid if r["delta_b"] is not None]
        deltas_ab = [r["delta_ab"] for r in valid if r["delta_ab"] is not None]

        # Only count wins/ties for runs where both A and B scored
        comparable = [
            r
            for r in valid
            if r["score_a_current"] is not None and r["score_b_improved"] is not None
        ]

        type_summary[et] = {
            "count": len(valid),
            "avg_baseline": round(sum(baselines) / len(baselines), 2) if baselines else 0.0,
            "avg_delta_a_current": round(sum(deltas_a) / len(deltas_a), 2) if deltas_a else 0.0,
            "avg_delta_b_improved": round(sum(deltas_b) / len(deltas_b), 2) if deltas_b else 0.0,
            "avg_delta_ab": round(sum(deltas_ab) / len(deltas_ab), 2) if deltas_ab else 0.0,
            "b_wins": sum(1 for r in comparable if r["score_b_improved"] > r["score_a_current"]),
            "a_wins": sum(1 for r in comparable if r["score_a_current"] > r["score_b_improved"]),
            "ties": sum(1 for r in comparable if r["score_a_current"] == r["score_b_improved"]),
        }

    # Build output
    output = {
        "run_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "model_creator": svc.world_quality._get_creator_model(),
            "model_judge": svc.world_quality._get_judge_model(),
            "quality_threshold": config.quality_threshold,
            "entity_types": entity_types,
            "count_per_type": args.count_per_type,
        },
        "ab_results": ab_results,
        "type_summary": type_summary,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")

    # Print summary
    print("\n=== A/B PROMPT TEST SUMMARY ===")
    print(f"{'Type':<15} {'Baseline':<10} {'Delta A':<10} {'Delta B':<10} {'A-B':<8} {'B Wins':<8}")
    print("-" * 61)
    for et in entity_types:
        s = type_summary.get(et, {})
        if s.get("error"):
            print(f"{et:<15} ERROR: {s['error']}")
            continue
        print(
            f"{et:<15} {s.get('avg_baseline', 0):<10.1f} "
            f"{s.get('avg_delta_a_current', 0):<+10.2f} "
            f"{s.get('avg_delta_b_improved', 0):<+10.2f} "
            f"{s.get('avg_delta_ab', 0):<+8.2f} "
            f"{s.get('b_wins', 0)}/{s.get('count', 0)}"
        )

    # Decision guidance
    print("\n=== DECISION GUIDANCE ===")
    for et in entity_types:
        s = type_summary.get(et, {})
        delta_ab = float(s.get("avg_delta_ab", 0))
        if delta_ab > 0.5:
            print(f"{et}: IMPROVED prompts help significantly (delta={delta_ab:+.2f})")
            print("  -> IMPLEMENT prompt changes for this entity type")
        elif delta_ab > 0.2:
            print(f"{et}: IMPROVED prompts show modest improvement (delta={delta_ab:+.2f})")
            print("  -> Consider implementing, but check if judge noise explains the gap")
        elif delta_ab > -0.2:
            print(f"{et}: No significant difference (delta={delta_ab:+.2f})")
            print("  -> Prompt changes alone won't fix this type")
        else:
            print(f"{et}: IMPROVED prompts are WORSE (delta={delta_ab:+.2f})")
            print("  -> Current prompts are better; investigate other causes")


if __name__ == "__main__":
    main()
