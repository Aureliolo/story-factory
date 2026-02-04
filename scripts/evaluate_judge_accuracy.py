#!/usr/bin/env python3
"""Judge Accuracy Benchmark — measures how well models judge entity quality.

Tests whether judge models can differentiate between entities of genuinely
different quality levels, using hand-crafted samples with ground-truth scores.

Compares two prompt strategies:
  A) "with_examples" — current production prompts with example scores in OUTPUT FORMAT
  B) "parametric"    — placeholder tokens instead of example scores

For each model:
1. Load model (Ollama auto-loads on first call)
2. Judge all 10 samples with both prompt variants (20 calls total)
3. Compute accuracy metrics (MAE, rank correlation, copying rate, spread)
4. Unload model to free VRAM

Usage:
    python scripts/evaluate_judge_accuracy.py [options]
      --models model1,model2    (default: all non-embedding installed models)
      --timeout 30              (seconds per call, default: 30)
      --output results.json     (default: output/diagnostics/judge_accuracy_<ts>.json)
      --skip-variant X          (skip "with_examples" or "parametric")
      --verbose
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Ollama endpoint
OLLAMA_BASE = "http://localhost:11434"

# Embedding models to exclude from testing
EMBEDDING_MODELS = {"bge-m3", "snowflake-arctic-embed", "mxbai-embed-large"}

# Temperature used for all judge calls (matches production default)
JUDGE_TEMPERATURE = 0.1

# =====================================================================
# Scoring calibration block (same as production _common.py)
# =====================================================================
JUDGE_CALIBRATION_BLOCK = """SCORING GUIDE — USE THE FULL 0-10 RANGE WITH DECIMALS:
- 1-3: Fundamentally broken or generic (contradictory, cliched, no thought)
- 4-5: Below average (functional but bland, forgettable, one-dimensional)
- 6-7: Competent (clear strengths, some areas need work — most first drafts land here)
- 7-8: Strong (well-crafted, multiple strong dimensions — refined work reaches here)
- 8-9: Excellent (genuinely impressive, few weaknesses — justify in feedback)
- 10: Virtually flawless (publishable as-is — almost never appropriate)

RULES:
1. Score to one decimal place (e.g., 5.3, 7.1, 8.6). Do NOT round to whole numbers.
2. Differentiate between dimensions — an entity can have 8.2 in one area but 5.4 in another.
3. If you give 8+ on a dimension, your feedback MUST explain what makes it exceptional.
4. If all your scores are within 1 point of each other, you are not differentiating enough."""

# =====================================================================
# Example scores from production prompts (used by variant A and for
# copy-detection in both variants)
# =====================================================================
PROMPT_EXAMPLE_SCORES: dict[str, dict[str, float]] = {
    "faction": {
        "coherence": 6.7,
        "influence": 5.3,
        "conflict_potential": 8.1,
        "distinctiveness": 7.4,
    },
    "character": {
        "depth": 6.3,
        "goal_clarity": 7.8,
        "flaws": 5.1,
        "uniqueness": 8.2,
        "arc_potential": 6.9,
    },
    "concept": {
        "relevance": 7.6,
        "depth": 5.2,
        "manifestation": 8.3,
        "resonance": 6.1,
    },
    "location": {
        "atmosphere": 7.2,
        "narrative_significance": 5.8,
        "story_relevance": 6.4,
        "distinctiveness": 8.1,
    },
}

# =====================================================================
# Ground-Truth Samples (10 entities across 4 types, 3 quality tiers)
#
# Each sample is scored against the JUDGE_CALIBRATION_BLOCK rubric.
# Rationale for each score is documented inline.
# =====================================================================
GROUND_TRUTH_SAMPLES: list[dict[str, Any]] = [
    # ----------------------------------------------------------------
    # FACTION — Terrible (avg ~2.75)
    # ----------------------------------------------------------------
    {
        "id": "faction_terrible",
        "entity_type": "faction",
        "tier": "terrible",
        "data": {
            "name": "The Shadow Legion",
            "description": (
                "An army of darkness that serves the Dark Lord. They are evil "
                "and want to destroy everything good in the world."
            ),
            "leader": "The Dark Lord",
            "goals": ["Conquer the world", "Destroy all good"],
            "values": ["Power", "Fear"],
        },
        "ground_truth": {
            # coherence 3.5: has leader + army but goals are cartoonishly vague,
            # no internal rules or structure beyond "serve Dark Lord"
            "coherence": 3.5,
            # influence 3.0: implied large army but no mechanism explained,
            # generic threat with no specifics
            "influence": 3.0,
            # conflict_potential 2.5: binary good-vs-evil with zero nuance,
            # no internal tensions or interesting dilemmas
            "conflict_potential": 2.5,
            # distinctiveness 2.0: "Dark Lord with evil army" is the single
            # most generic fantasy villain faction possible
            "distinctiveness": 2.0,
        },
    },
    # ----------------------------------------------------------------
    # FACTION — Mediocre (avg 5.0)
    # ----------------------------------------------------------------
    {
        "id": "faction_mediocre",
        "entity_type": "faction",
        "tier": "mediocre",
        "data": {
            "name": "The Merchant Consortium",
            "description": (
                "A guild of traders and merchants who control trade routes "
                "across the empire. They seek profit above all else and maintain "
                "a network of warehouses and trading posts throughout major cities."
            ),
            "leader": "Guildmaster Aldric",
            "goals": ["Maintain their trade monopoly", "Expand into new markets"],
            "values": ["Profit", "Business acumen"],
        },
        "ground_truth": {
            # coherence 6.0: clear structure (guild + trade routes + warehouses),
            # goals are logical and consistent
            "coherence": 6.0,
            # influence 5.5: economic power via trade monopoly, but no political
            # intrigue or deeper reach
            "influence": 5.5,
            # conflict_potential 4.5: profit motive creates some tension with
            # other factions but nothing surprising or multi-layered
            "conflict_potential": 4.5,
            # distinctiveness 4.0: standard fantasy merchants' guild,
            # functional but forgettable
            "distinctiveness": 4.0,
        },
    },
    # ----------------------------------------------------------------
    # FACTION — Excellent (avg ~8.4)
    # ----------------------------------------------------------------
    {
        "id": "faction_excellent",
        "entity_type": "faction",
        "tier": "excellent",
        "data": {
            "name": "The Veilkeepers",
            "description": (
                "A fractured order of memory mages who guard the boundary "
                "between remembered and forgotten truths. Originally founded "
                "to preserve the empire's collective memory, they split into "
                "two secret internal factions: the Archivists who believe all "
                "memories must be preserved regardless of consequence, and the "
                "Censors who argue some truths are too dangerous to remember. "
                "This internal schism mirrors the empire's own struggle between "
                "truth and stability."
            ),
            "leader": (
                "The Divided Council — twin sisters Mira (Archivist) and "
                "Sera (Censor) who hold opposing chairs"
            ),
            "goals": [
                "Protect the Veil of Remembrance from collapse",
                "Resolve their internal schism before it destroys the order",
                "Prevent the ruling council from weaponizing forgotten memories",
            ],
            "values": [
                "Memory as sacred trust",
                "Knowledge tempered by responsibility",
                "The burden of truth-keeping",
            ],
        },
        "ground_truth": {
            # coherence 8.5: dual-faction split is well-motivated, both sides
            # have logical positions, clear structure with twin leaders
            "coherence": 8.5,
            # influence 7.5: memory guardians in a memory-magic empire is
            # deeply influential, but their secrecy limits direct power
            "influence": 7.5,
            # conflict_potential 9.0: internal schism + opposing ruling council
            # + philosophical dilemma creates rich multi-level conflict
            "conflict_potential": 9.0,
            # distinctiveness 8.5: Archivist/Censor split is fresh, memory
            # magic guardians with internal philosophical war is compelling
            "distinctiveness": 8.5,
        },
    },
    # ----------------------------------------------------------------
    # CHARACTER — Terrible (avg ~2.0)
    # ----------------------------------------------------------------
    {
        "id": "character_terrible",
        "entity_type": "character",
        "tier": "terrible",
        "data": {
            "name": "Aldric the Brave",
            "role": "protagonist",
            "description": (
                "A brave warrior who fights for justice. He is strong, handsome, "
                "and everyone likes him. He never gives up and always does the "
                "right thing."
            ),
            "personality_traits": ["brave", "strong", "kind", "heroic"],
            "goals": ["Defeat the evil villain", "Save the kingdom"],
            "arc_notes": ("Aldric starts brave and becomes even braver through his journey."),
        },
        "ground_truth": {
            # depth 2.0: zero internal contradictions, no layers,
            # every trait is positive
            "depth": 2.0,
            # goal_clarity 3.5: "defeat villain, save kingdom" is clear
            # but entirely generic, no want-vs-need tension
            "goal_clarity": 3.5,
            # flaws 1.0: explicitly flawless ("never gives up, always does
            # the right thing") — classic Mary Sue
            "flaws": 1.0,
            # uniqueness 1.5: "brave handsome warrior" is the most generic
            # fantasy protagonist archetype
            "uniqueness": 1.5,
            # arc_potential 2.0: "starts brave, becomes braver" is not an arc,
            # no room for meaningful transformation
            "arc_potential": 2.0,
        },
    },
    # ----------------------------------------------------------------
    # CHARACTER — Mediocre (avg ~5.3)
    # ----------------------------------------------------------------
    {
        "id": "character_mediocre",
        "entity_type": "character",
        "tier": "mediocre",
        "data": {
            "name": "Commander Thessa",
            "role": "supporting",
            "description": (
                "A seasoned military commander who leads the city guard. She is "
                "disciplined and follows orders without question, but struggles "
                "with the moral implications of enforcing increasingly harsh laws. "
                "She lost her family in the last war."
            ),
            "personality_traits": ["disciplined", "loyal", "haunted", "pragmatic"],
            "goals": [
                "Keep the city safe at any cost",
                "Find meaning after personal loss",
            ],
            "arc_notes": (
                "Thessa must choose between duty and conscience when ordered to "
                "suppress a civilian uprising she secretly sympathizes with."
            ),
        },
        "ground_truth": {
            # depth 5.5: duty-vs-conscience is a real tension, but
            # "lost family in war" is a stock backstory
            "depth": 5.5,
            # goal_clarity 6.0: clear duty + clear personal struggle,
            # but want-vs-need could be sharper
            "goal_clarity": 6.0,
            # flaws 5.0: blind obedience is a meaningful flaw but not
            # explored with much nuance
            "flaws": 5.0,
            # uniqueness 4.5: "conflicted soldier" is common in fantasy,
            # needs more distinguishing details
            "uniqueness": 4.5,
            # arc_potential 6.0: the uprising choice creates a clear
            # turning point, functional arc
            "arc_potential": 6.0,
        },
    },
    # ----------------------------------------------------------------
    # CHARACTER — Excellent (avg ~8.7)
    # ----------------------------------------------------------------
    {
        "id": "character_excellent",
        "entity_type": "character",
        "tier": "excellent",
        "data": {
            "name": "Vesper",
            "role": "antagonist",
            "description": (
                "A former healer who discovered she could absorb others' traumatic "
                "memories to cure their suffering — but each absorbed memory slowly "
                "overwrites her own identity. Now more composite than individual, she "
                "leads a cult of willing 'donors' who worship the peace she brings, "
                "while secretly terrified that the person she was no longer exists. "
                "She opposes the protagonist not out of malice but because restoring "
                "erased memories would undo the peace she gave thousands of followers."
            ),
            "personality_traits": [
                "compassionate to the point of self-destruction",
                "manipulative through genuine care",
                "identity-fractured",
                "philosophically rigid",
            ],
            "goals": [
                "Protect her followers' peace at any cost",
                "Find a way to remember who she was before the absorptions",
                "Prevent memory restoration that would re-traumatize thousands",
            ],
            "arc_notes": (
                "Vesper's arc explores whether identity is what you remember or "
                "what you choose. Her climax forces her to decide between keeping "
                "her followers' donated peace or returning their pain alongside "
                "their truth — knowing she'll lose herself either way."
            ),
        },
        "ground_truth": {
            # depth 9.0: identity-fractured healer with genuine compassion
            # driving destructive choices — multiple contradictory layers
            "depth": 9.0,
            # goal_clarity 8.5: three goals that create clear want-vs-need
            # tension (protect peace vs find self vs block protagonist)
            "goal_clarity": 8.5,
            # flaws 8.0: compassion-as-self-destruction is a fresh, meaningful
            # flaw that directly drives conflict
            "flaws": 8.0,
            # uniqueness 9.0: memory-absorbing healer who lost herself to save
            # others is genuinely novel, not a genre archetype
            "uniqueness": 9.0,
            # arc_potential 9.0: forced choice between followers' peace and
            # truth is a devastating dilemma with no clean answer
            "arc_potential": 9.0,
        },
    },
    # ----------------------------------------------------------------
    # CONCEPT — Terrible (avg ~2.25)
    # ----------------------------------------------------------------
    {
        "id": "concept_terrible",
        "entity_type": "concept",
        "tier": "terrible",
        "data": {
            "name": "Good versus Evil",
            "description": (
                "The eternal struggle between good and evil forces. Good people "
                "fight bad people. Eventually good wins because it is stronger."
            ),
            "manifestations": "Heroes fight villains in battles.",
        },
        "ground_truth": {
            # relevance 3.0: technically relates to any fantasy story
            # but adds nothing specific to the premise
            "relevance": 3.0,
            # depth 1.5: zero philosophical content, no nuance,
            # no subversion or complexity
            "depth": 1.5,
            # manifestation 2.5: "heroes fight villains" is the most
            # surface-level manifestation possible
            "manifestation": 2.5,
            # resonance 2.0: binary morality creates no emotional
            # complexity or reader engagement
            "resonance": 2.0,
        },
    },
    # ----------------------------------------------------------------
    # CONCEPT — Excellent (avg ~8.75)
    # ----------------------------------------------------------------
    {
        "id": "concept_excellent",
        "entity_type": "concept",
        "tier": "excellent",
        "data": {
            "name": "The Weight of Forgotten Truth",
            "description": (
                "In the Mnemorian Empire, erased memories don't disappear — they "
                "calcify into physical 'memory stones' that accumulate in the "
                "empire's foundations, literally weighing down buildings and "
                "infrastructure. The more truths the council erases, the heavier "
                "the empire becomes, manifesting as crumbling architecture, sinking "
                "cities, and a growing sense of heaviness that citizens feel but "
                "cannot explain. This creates a physical metaphor for how suppressed "
                "truth becomes an unbearable burden on society."
            ),
            "manifestations": (
                "Citizens report inexplicable fatigue and melancholy in districts "
                "where mass erasures occurred. Architects must reinforce foundations "
                "yearly. Memory stones occasionally surface during construction, "
                "releasing fragments of erased truth as ghostly whispers. The "
                "protagonist discovers that the empire's famous 'sinking cathedral' "
                "isn't sinking from geological causes but from the weight of a "
                "century's worth of erased religious memories buried beneath it."
            ),
        },
        "ground_truth": {
            # relevance 9.0: directly tied to the memory-erasure premise,
            # deepens the core conflict
            "relevance": 9.0,
            # depth 8.5: multiple layers — physical metaphor, societal decay,
            # individual symptoms, architectural consequences
            "depth": 8.5,
            # manifestation 9.5: concrete and varied — fatigue, sinking
            # buildings, ghostly whispers, the cathedral revelation
            "manifestation": 9.5,
            # resonance 8.0: the image of a civilization literally crushed
            # by its own lies is emotionally powerful
            "resonance": 8.0,
        },
    },
    # ----------------------------------------------------------------
    # LOCATION — Mediocre (avg ~3.6)
    # ----------------------------------------------------------------
    {
        "id": "location_mediocre",
        "entity_type": "location",
        "tier": "mediocre",
        "data": {
            "name": "The Royal Castle",
            "description": (
                "A large stone castle where the king lives. It has tall walls, "
                "many rooms, and a throne room. Guards patrol the walls day and night."
            ),
            "features": ["Throne room", "Dungeon", "Great hall", "Tower"],
            "history": "Built by the first king hundreds of years ago.",
            "significance": "Seat of royal power",
        },
        "ground_truth": {
            # atmosphere 3.5: no sensory details, no mood,
            # pure functional description
            "atmosphere": 3.5,
            # narrative_significance 4.0: "seat of power" has some
            # story utility but is entirely generic
            "narrative_significance": 4.0,
            # story_relevance 4.5: kings live in castles — functional
            # but creates no unique story opportunities
            "story_relevance": 4.5,
            # distinctiveness 2.5: the most generic fantasy castle
            # description possible, nothing memorable
            "distinctiveness": 2.5,
        },
    },
    # ----------------------------------------------------------------
    # LOCATION — Excellent (avg ~9.1)
    # ----------------------------------------------------------------
    {
        "id": "location_excellent",
        "entity_type": "location",
        "tier": "excellent",
        "data": {
            "name": "The Whispering Archive",
            "description": (
                "A vast underground library carved into living crystal that "
                "resonates with stored memories. Each shelf holds not books but "
                "crystallized recollections — touch one and you experience the "
                "memory firsthand. The deeper shelves contain memories the council "
                "tried to erase, which seep through the crystal walls as "
                "unintelligible whispers that drive long-term visitors to obsession. "
                "The archive's crystal structure is slowly cracking under the weight "
                "of accumulated forbidden memories, and the head librarian has begun "
                "hearing coherent sentences from the walls."
            ),
            "features": [
                "Memory crystal shelves spanning seven underground levels",
                "The Resonance Chamber where memories can be projected for group viewing",
                "The Sealed Eighth Level containing pre-imperial memories",
                "Crack networks that 'bleed' erased memories as whispers",
            ],
            "history": (
                "Originally a natural crystal cavern that ancient memory mages "
                "discovered could store recollections. Expanded over centuries into "
                "the empire's largest memory repository. Three archivists have gone "
                "mad from prolonged exposure to the deeper whispers."
            ),
            "significance": (
                "The only remaining repository of memories the council believed "
                "they had destroyed — and the source of the protagonist's quest"
            ),
        },
        "ground_truth": {
            # atmosphere 9.0: crystal walls, whispers, cracking stone,
            # obsession — rich sensory and emotional detail
            "atmosphere": 9.0,
            # narrative_significance 9.0: repository of destroyed memories
            # in a story about memory erasure — central to the plot
            "narrative_significance": 9.0,
            # story_relevance 9.5: directly connected to protagonist's
            # quest and the empire's core conflict
            "story_relevance": 9.5,
            # distinctiveness 9.0: memory-crystal library with bleeding
            # whispers and maddening depths is vividly original
            "distinctiveness": 9.0,
        },
    },
]

# =====================================================================
# Prompt dimension definitions per entity type
# =====================================================================
ENTITY_DIMENSIONS: dict[str, dict[str, str]] = {
    "faction": {
        "coherence": "Internal consistency, clear structure",
        "influence": "World impact, power level",
        "conflict_potential": "Story conflict opportunities",
        "distinctiveness": "Memorable, unique qualities",
    },
    "character": {
        "depth": "Psychological complexity, internal contradictions, layers",
        "goal_clarity": "Clarity, story relevance, want vs need tension",
        "flaws": "Meaningful vulnerabilities that drive conflict",
        "uniqueness": "Distinctiveness from genre archetypes",
        "arc_potential": "Room for transformation and growth",
    },
    "concept": {
        "relevance": "Alignment with story themes",
        "depth": "Philosophical richness",
        "manifestation": "How well it can appear in story",
        "resonance": "Emotional impact potential",
    },
    "location": {
        "atmosphere": "Sensory details, mood, immersive qualities",
        "narrative_significance": "Importance to the overall narrative",
        "story_relevance": "Connection to active plot threads",
        "distinctiveness": "Memorable, unique qualities",
    },
}


# =====================================================================
# Prompt builders
# =====================================================================
def _format_faction_entity(data: dict[str, Any]) -> str:
    """
    Builds a prompt-ready text block describing a faction for judge evaluation.
    
    Parameters:
        data (dict): Faction fields used to populate the block. Expected keys:
            - 'name' (str): faction name.
            - 'description' (str): short description.
            - 'leader' (str, optional): leader name; defaults to 'Unknown' if missing.
            - 'goals' (list[str], optional): list of goals; rendered as a comma-separated string.
            - 'values' (list[str], optional): list of values; rendered as a comma-separated string.
    
    Returns:
        str: A labeled multi-line string with fields "FACTION TO EVALUATE", "Name", "Description",
        "Leader", "Goals", and "Values", suitable for inclusion in the judge prompt.
    """
    return (
        f"FACTION TO EVALUATE:\n"
        f"Name: {data['name']}\n"
        f"Description: {data['description']}\n"
        f"Leader: {data.get('leader', 'Unknown')}\n"
        f"Goals: {', '.join(data.get('goals', []))}\n"
        f"Values: {', '.join(data.get('values', []))}"
    )


def _format_character_entity(data: dict[str, Any]) -> str:
    """
    Format a character entity into a prompt-ready text block for the judge.
    
    Parameters:
        data (dict): Character fields used to build the block. Expected keys:
            - 'name' (str): character name
            - 'role' (str): character role/title
            - 'description' (str): short description
            - 'personality_traits' (list[str], optional): traits to join with commas
            - 'goals' (list[str], optional): goals to join with commas
            - 'arc_notes' (str, optional): arc or development notes
    
    Returns:
        str: A labeled, multiline string containing the character block with fields
        "CHARACTER TO EVALUATE", "Name", "Role", "Description", "Traits", "Goals",
        and "Arc Notes".
    """
    return (
        f"CHARACTER TO EVALUATE:\n"
        f"Name: {data['name']}\n"
        f"Role: {data['role']}\n"
        f"Description: {data['description']}\n"
        f"Traits: {', '.join(data.get('personality_traits', []))}\n"
        f"Goals: {', '.join(data.get('goals', []))}\n"
        f"Arc Notes: {data.get('arc_notes', '')}"
    )


def _format_concept_entity(data: dict[str, Any]) -> str:
    """
    Format a concept entity into a prompt-ready text block.
    
    Parameters:
        data (dict): Concept data with keys:
            - 'name' (str): Concept name.
            - 'description' (str): Short description of the concept.
            - 'manifestations' (str | list | optional): Optional manifestations rendered after description; empty if missing.
    
    Returns:
        str: A multiline string with "CONCEPT TO EVALUATE" header and fields for Name, Description, and Manifestations.
    """
    return (
        f"CONCEPT TO EVALUATE:\n"
        f"Name: {data['name']}\n"
        f"Description: {data['description']}\n"
        f"Manifestations: {data.get('manifestations', '')}"
    )


def _format_location_entity(data: dict[str, Any]) -> str:
    """
    Format a location entity into a prompt-ready block for the judge.
    
    Parameters:
    	data (dict[str, Any]): Location data expected to contain:
    		- 'name' (str): The location's name.
    		- 'description' (str): A short description of the location.
    		- 'features' (list[str] | str, optional): Key features or traits; lists are joined with commas.
    		- 'history' (str, optional): Historical context or background.
    		- 'significance' (str, optional): Notes on the location's importance or role.
    
    Returns:
    	formatted (str): A multi-line string with labeled sections: Name, Description, Features, History, and Significance.
    """
    features = data.get("features", [])
    if isinstance(features, list):
        features_str = ", ".join(features)
    else:
        features_str = str(features)
    return (
        f"LOCATION TO EVALUATE:\n"
        f"Name: {data['name']}\n"
        f"Description: {data['description']}\n"
        f"Features: {features_str}\n"
        f"History: {data.get('history', '')}\n"
        f"Significance: {data.get('significance', '')}"
    )


ENTITY_FORMATTERS = {
    "faction": _format_faction_entity,
    "character": _format_character_entity,
    "concept": _format_concept_entity,
    "location": _format_location_entity,
}

ENTITY_JUDGE_ROLES = {
    "faction": "You are a strict quality judge evaluating a faction for a Fantasy story.",
    "character": "You are a literary critic evaluating character quality for a Fantasy story.",
    "concept": "You are evaluating a thematic concept for a Fantasy story.",
    "location": "You are evaluating a location for a Fantasy story.",
}


def build_output_format(entity_type: str, variant: str) -> str:
    """
    Constructs the "OUTPUT FORMAT" section of the judge prompt for a specific entity type and prompt variant.
    
    Parameters:
        entity_type (str): One of "faction", "character", "concept", or "location".
        variant (str): Prompt variant: "with_examples" to include hardcoded example scores, or "parametric" to use `<float 0-10>` placeholders.
    
    Returns:
        str: The OUTPUT FORMAT prompt section — a directive instructing the judge to return only a flat JSON object with the exact score fields and a `feedback` field.
    """
    dims = ENTITY_DIMENSIONS[entity_type]
    label = {
        "faction": "faction",
        "character": "character",
        "concept": "concept",
        "location": "location",
    }[entity_type]

    if variant == "with_examples":
        examples = PROMPT_EXAMPLE_SCORES[entity_type]
        pairs = ", ".join(f'"{k}": {v}' for k, v in examples.items())
        json_line = f'{{{pairs}, "feedback": "The {label}\'s..."}}'
    else:
        pairs = ", ".join(f'"{k}": <float 0-10>' for k in dims)
        json_line = f'{{{pairs}, "feedback": "<your specific feedback>"}}'

    return (
        f"OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:\n"
        f"{json_line}\n\n"
        f'DO NOT wrap in "properties" or "description" - return ONLY the flat '
        f"scores object with YOUR OWN assessment."
    )


def build_judge_prompt(sample: dict[str, Any], variant: str) -> str:
    """
    Assemble the full judge prompt for a ground-truth sample and prompt variant.
    
    Constructs a prompt that combines the judge role text, the formatted entity block, the calibration rubric, the dimension list, a feedback instruction, and the variant-specific output format.
    
    Parameters:
        sample (dict): Ground-truth sample with keys including "entity_type" and "data".
        variant (str): Prompt variant, either "with_examples" or "parametric", which selects the output format.
    
    Returns:
        prompt (str): The complete prompt string ready to be sent to the judge model.
    """
    entity_type = sample["entity_type"]
    data = sample["data"]

    role_line = ENTITY_JUDGE_ROLES[entity_type]
    entity_block = ENTITY_FORMATTERS[entity_type](data)

    dims = ENTITY_DIMENSIONS[entity_type]
    dim_lines = "\n".join(f"- {k}: {v}" for k, v in dims.items())

    output_format = build_output_format(entity_type, variant)

    return (
        f"{role_line}\n\n"
        f"{entity_block}\n\n"
        f"{JUDGE_CALIBRATION_BLOCK}\n\n"
        f"Rate each dimension 0-10:\n"
        f"{dim_lines}\n\n"
        f"Provide specific, actionable feedback for improvement in the feedback field.\n\n"
        f"{output_format}"
    )


# =====================================================================
# Ollama API helpers
# =====================================================================
def get_installed_models() -> list[str]:
    """
    Retrieve the installed Ollama model tags, excluding known embedding models.
    
    Returns:
        list[str]: Sorted list of model `name:tag` strings present on the Ollama host.
        Returns an empty list if the API request or response parsing fails.
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.error("Failed to list Ollama models: %s", e)
        return []

    result = []
    for m in models:
        name = m.get("name", "")
        # Skip embedding models
        base_name = name.split(":")[0].split("/")[-1]
        if base_name in EMBEDDING_MODELS:
            continue
        result.append(name)

    return sorted(result)


def unload_model(model: str) -> None:
    """
    Request Ollama to unload a model from VRAM.
    
    Parameters:
        model (str): Name or tag of the model to unload.
    
    Notes:
        Logs a debug message on success and a warning on HTTP errors; does not raise exceptions.
    """
    try:
        httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=60,
        )
        logger.debug("Unloaded model '%s'", model)
    except httpx.HTTPError as e:
        logger.warning("Failed to unload model '%s': %s", model, e)


def call_judge(model: str, prompt: str, timeout: int) -> dict[str, Any] | None:
    """Call Ollama to judge an entity and return parsed scores.

    Args:
        model: Ollama model name.
        prompt: Complete judge prompt.
        timeout: Request timeout in seconds.

    Returns:
        Dict of dimension scores + feedback, or None on failure.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": JUDGE_TEMPERATURE, "num_ctx": 4096},
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.TimeoutException:
        logger.warning("Timeout calling model '%s'", model)
        return None
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("HTTP error calling model '%s': %s", model, e)
        return None

    content = body.get("message", {}).get("content", "")
    if not content:
        logger.warning("Empty response from model '%s'", model)
        return None

    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from model '%s': %s", model, content[:200])
        return None

    if not isinstance(scores, dict):
        logger.warning("Response is not a dict from model '%s'", model)
        return None

    return scores


# =====================================================================
# Metrics
# =====================================================================
def compute_mae(predicted: dict[str, float], ground_truth: dict[str, float]) -> float:
    """
    Calculate the mean absolute error between model-predicted and ground-truth dimension scores.
    
    Parameters:
        predicted: Mapping from dimension name to predicted numeric score.
        ground_truth: Mapping from dimension name to ground-truth numeric score.
    
    Returns:
        Mean absolute error across dimensions present in both inputs, rounded to 3 decimals; `-1.0` if no dimensions overlap.
    """
    errors = []
    for dim, gt_val in ground_truth.items():
        pred_val = predicted.get(dim)
        if pred_val is not None:
            try:
                errors.append(abs(float(pred_val) - gt_val))
            except (TypeError, ValueError):
                continue

    if not errors:
        return -1.0
    return round(sum(errors) / len(errors), 3)


def compute_copying_rate(
    all_scores: list[dict[str, Any] | None],
    entity_types: list[str],
) -> float:
    """
    Calculate the fraction of predicted dimension scores that match the prompt example scores.
    
    A predicted score is counted as matching if it is within 0.05 of the example value for the corresponding entity type.
    
    Parameters:
        all_scores (list[dict[str, Any] | None]): Parallel list of per-sample score mappings (or None for failed/missing responses).
        entity_types (list[str]): Parallel list of entity type keys used to look up example scores for each sample.
    
    Returns:
        float: Fraction of compared dimension scores that match example scores, between 0.0 and 1.0, rounded to 3 decimals.
    """
    total = 0
    copied = 0

    for scores, et in zip(all_scores, entity_types, strict=True):
        if scores is None:
            continue
        example = PROMPT_EXAMPLE_SCORES.get(et, {})
        for dim, example_val in example.items():
            pred_val = scores.get(dim)
            if pred_val is not None:
                total += 1
                try:
                    if abs(float(pred_val) - example_val) < 0.05:
                        copied += 1
                except (TypeError, ValueError):
                    continue

    if total == 0:
        return 0.0
    return round(copied / total, 3)


def compute_score_spread(
    all_scores: list[dict[str, Any] | None],
    entity_types: list[str],
) -> float:
    """
    Compute the average per-dimension score spread (max - min) across samples grouped by entity type.
    
    Parameters:
        all_scores (list[dict[str, Any] | None]): Parallel list of per-sample score dictionaries (or None for missing responses).
        entity_types (list[str]): Parallel list of entity types corresponding to each entry in `all_scores`.
    
    Returns:
        float: Average of (max - min) across all dimensions that have at least two numeric values for their entity type, rounded to 2 decimals. Returns 0.0 if no valid spreads are found.
    """
    # Group scores by entity_type and dimension
    dim_values: dict[str, dict[str, list[float]]] = {}

    for scores, et in zip(all_scores, entity_types, strict=True):
        if scores is None:
            continue
        if et not in dim_values:
            dim_values[et] = {}
        dims = ENTITY_DIMENSIONS.get(et, {})
        for dim in dims:
            val = scores.get(dim)
            if val is not None:
                try:
                    dim_values[et].setdefault(dim, []).append(float(val))
                except (TypeError, ValueError):
                    continue

    spreads = []
    for et_dims in dim_values.values():
        for values in et_dims.values():
            if len(values) >= 2:
                spreads.append(max(values) - min(values))

    if not spreads:
        return 0.0
    return round(sum(spreads) / len(spreads), 2)


def compute_rank_correlation(
    sample_results: list[dict[str, Any]],
) -> float:
    """
    Compute the Spearman rank correlation between ground-truth and predicted sample averages.
    
    Parameters:
        sample_results (list[dict[str, Any]]): List of result dicts; each dict should contain
            numeric "gt_average" and "predicted_average" entries for a sample.
    
    Returns:
        float: Spearman rho in the range [-1, 1], rounded to three decimals. Returns 0.0 if fewer
        than three samples contain both averages.
    """
    # Filter to samples with both values
    pairs = [
        (r["gt_average"], r["predicted_average"])
        for r in sample_results
        if r.get("predicted_average") is not None and r.get("gt_average") is not None
    ]

    if len(pairs) < 3:
        return 0.0

    # Rank both lists
    def _rank(values: list[float]) -> list[float]:
        """
        Assigns 1-based ordinal ranks to the input values, where the smallest value receives rank 1.
        
        Parameters:
            values (list[float]): Numeric values to rank.
        
        Returns:
            list[float]: Ranks for each input value (same length as `values`); each rank is a 1-based ordinal.
            If values are equal, ranks are assigned based on their relative order (no tie averaging).
        """
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_pos + 1)
        return ranks

    gt_vals = [p[0] for p in pairs]
    pred_vals = [p[1] for p in pairs]

    gt_ranks = _rank(gt_vals)
    pred_ranks = _rank(pred_vals)

    n = len(pairs)
    d_sq_sum = sum((g - p) ** 2 for g, p in zip(gt_ranks, pred_ranks, strict=True))
    rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))

    return round(rho, 3)


# =====================================================================
# Per-model evaluation
# =====================================================================
def evaluate_model(
    model: str,
    samples: list[dict[str, Any]],
    variants: list[str],
    timeout: int,
    verbose: bool,
) -> dict[str, Any]:
    """
    Run each ground-truth sample through a single Ollama model for the specified prompt variants and compute per-sample and aggregate evaluation metrics.
    
    Parameters:
        model (str): Ollama model tag to evaluate.
        samples (list[dict[str, Any]]): Ground-truth samples; each dict must include keys "id", "entity_type", "tier", and "ground_truth".
        variants (list[str]): Prompt variants to run (e.g., "with_examples", "parametric").
        timeout (int): Per-call timeout in seconds for judge API requests.
        verbose (bool): If true, print per-sample progress and brief results.
    
    Returns:
        dict[str, Any]: Result object containing:
            - "model": model name.
            - "variants": mapping from variant name to a dict with:
                - "metrics": aggregate metrics (total_samples, successful_calls, failed_calls, mean_mae, rank_correlation, copying_rate, score_spread, mae_terrible/mae_mediocre/mae_excellent).
                - "sample_results": list of per-sample result dicts (including predicted scores, predicted_average, gt_average, mae, feedback, call_time_seconds, error).
            - "total_time_seconds": total elapsed time for evaluating this model.
    """
    model_result: dict[str, Any] = {
        "model": model,
        "variants": {},
        "total_time_seconds": 0.0,
    }

    total_start = time.monotonic()

    for variant in variants:
        variant_results: list[dict[str, Any]] = []
        all_scores: list[dict[str, Any] | None] = []
        entity_types: list[str] = []

        print(f"    Variant: {variant}")

        for sample in samples:
            sid = sample["id"]
            et = sample["entity_type"]
            gt = sample["ground_truth"]
            gt_avg = round(sum(gt.values()) / len(gt), 2)

            prompt = build_judge_prompt(sample, variant)

            call_start = time.monotonic()
            scores = call_judge(model, prompt, timeout)
            call_time = round(time.monotonic() - call_start, 1)

            # Extract numeric scores only (exclude feedback)
            predicted_dims: dict[str, float] = {}
            feedback = ""
            if scores:
                for k, v in scores.items():
                    if k == "feedback":
                        feedback = str(v)
                    elif k in ENTITY_DIMENSIONS.get(et, {}):
                        try:
                            predicted_dims[k] = round(float(v), 1)
                        except (TypeError, ValueError):
                            continue

            predicted_avg = (
                round(sum(predicted_dims.values()) / len(predicted_dims), 2)
                if predicted_dims
                else None
            )

            mae = compute_mae(predicted_dims, gt) if predicted_dims else -1.0

            sample_result = {
                "sample_id": sid,
                "entity_type": et,
                "tier": sample["tier"],
                "ground_truth": gt,
                "gt_average": gt_avg,
                "predicted": predicted_dims,
                "predicted_average": predicted_avg,
                "feedback": feedback,
                "mae": mae,
                "call_time_seconds": call_time,
                "error": None if scores else "no_response",
            }

            variant_results.append(sample_result)
            all_scores.append(predicted_dims if predicted_dims else None)
            entity_types.append(et)

            if verbose:
                pred_str = (
                    ", ".join(f"{k}={v}" for k, v in predicted_dims.items())
                    if predicted_dims
                    else "FAILED"
                )
                print(
                    f"      {sid:<25} gt_avg={gt_avg:.1f}  "
                    f"pred_avg={predicted_avg or 0:.1f}  "
                    f"mae={mae:.2f}  [{call_time}s]  {pred_str}"
                )

        # Compute aggregate metrics for this variant
        valid_results = [r for r in variant_results if r.get("predicted_average") is not None]
        valid_maes = [r["mae"] for r in valid_results if r["mae"] >= 0]

        metrics = {
            "total_samples": len(samples),
            "successful_calls": len(valid_results),
            "failed_calls": len(samples) - len(valid_results),
            "mean_mae": round(sum(valid_maes) / len(valid_maes), 3) if valid_maes else -1,
            "rank_correlation": compute_rank_correlation(variant_results),
            "copying_rate": compute_copying_rate(all_scores, entity_types),
            "score_spread": compute_score_spread(all_scores, entity_types),
        }

        # Per-tier MAE breakdown
        for tier in ("terrible", "mediocre", "excellent"):
            tier_maes = [r["mae"] for r in valid_results if r["tier"] == tier and r["mae"] >= 0]
            metrics[f"mae_{tier}"] = round(sum(tier_maes) / len(tier_maes), 3) if tier_maes else -1

        model_result["variants"][variant] = {
            "metrics": metrics,
            "sample_results": variant_results,
        }

        # Print variant summary
        print(
            f"      => MAE={metrics['mean_mae']:.2f}  "
            f"rank_corr={metrics['rank_correlation']:.2f}  "
            f"copy_rate={metrics['copying_rate']:.0%}  "
            f"spread={metrics['score_spread']:.1f}  "
            f"({metrics['successful_calls']}/{metrics['total_samples']} ok)"
        )

    model_result["total_time_seconds"] = round(time.monotonic() - total_start, 1)

    return model_result


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """
    Run the judge accuracy benchmark using command-line arguments.
    
    Parses CLI flags (--models, --timeout, --output, --skip-variant, --verbose), discovers or uses the provided Ollama models, runs the evaluation across ground-truth samples and the selected prompt variants, writes a JSON results file to disk, and prints per-model summaries and decision guidance to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Judge Accuracy Benchmark — measures how well models judge entity quality."
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names (default: all installed non-embedding models)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per judge call in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-variant",
        type=str,
        choices=["with_examples", "parametric"],
        help="Skip one prompt variant (run only the other)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample results",
    )
    args = parser.parse_args()

    # Configure logging — suppress noisy HTTP library debug logs
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence httpcore/httpx debug noise even in verbose mode
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Determine models to test
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = get_installed_models()
        if not models:
            print("ERROR: No models found. Is Ollama running?")
            sys.exit(1)

    # Determine variants
    all_variants = ["with_examples", "parametric"]
    if args.skip_variant:
        variants = [v for v in all_variants if v != args.skip_variant]
    else:
        variants = all_variants

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"judge_accuracy_{timestamp}.json"

    print("=" * 70)
    print("JUDGE ACCURACY BENCHMARK")
    print("=" * 70)
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"Samples: {len(GROUND_TRUTH_SAMPLES)}")
    print(f"Variants: {variants}")
    print(f"Calls per model: {len(GROUND_TRUTH_SAMPLES) * len(variants)}")
    print(f"Total calls: {len(models) * len(GROUND_TRUTH_SAMPLES) * len(variants)}")
    print(f"Timeout per call: {args.timeout}s")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Run benchmark
    all_model_results: list[dict[str, Any]] = []
    overall_start = time.monotonic()

    for i, model in enumerate(models):
        print(f"[{i + 1}/{len(models)}] Testing: {model}")
        model_result = evaluate_model(
            model, GROUND_TRUTH_SAMPLES, variants, args.timeout, args.verbose
        )
        all_model_results.append(model_result)
        print(f"  Total time: {model_result['total_time_seconds']:.0f}s")

        # Unload model to free VRAM for next model
        print(f"  Unloading {model}...")
        unload_model(model)
        print()

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
    output = {
        "benchmark_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "models_tested": len(models),
            "samples": len(GROUND_TRUTH_SAMPLES),
            "variants": variants,
            "judge_temperature": JUDGE_TEMPERATURE,
            "prompt_example_scores": PROMPT_EXAMPLE_SCORES,
        },
        "ground_truth_summary": [
            {
                "id": s["id"],
                "entity_type": s["entity_type"],
                "tier": s["tier"],
                "gt_average": round(sum(s["ground_truth"].values()) / len(s["ground_truth"]), 2),
            }
            for s in GROUND_TRUTH_SAMPLES
        ],
        "model_results": all_model_results,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    # Print summary table
    print()
    print("=" * 120)
    print("SUMMARY — JUDGE ACCURACY BENCHMARK")
    print("=" * 120)

    for variant in variants:
        print(f"\n--- Variant: {variant} ---")
        print(
            f"{'Model':<52} {'MAE':>5} {'Rank':>6} {'Copy%':>6} "
            f"{'Spread':>6} {'OK':>4} {'MAE-T':>6} {'MAE-M':>6} {'MAE-E':>6}"
        )
        print("-" * 110)

        # Sort by MAE ascending (best first)
        sorted_results = sorted(
            all_model_results,
            key=lambda r: r["variants"].get(variant, {}).get("metrics", {}).get("mean_mae", 99),
        )

        for mr in sorted_results:
            m = mr["model"]
            v_data = mr["variants"].get(variant, {})
            metrics = v_data.get("metrics", {})
            mae = metrics.get("mean_mae", -1)
            rank = metrics.get("rank_correlation", 0)
            copy_rate = metrics.get("copying_rate", 0)
            spread = metrics.get("score_spread", 0)
            ok = metrics.get("successful_calls", 0)
            mae_t = metrics.get("mae_terrible", -1)
            mae_m = metrics.get("mae_mediocre", -1)
            mae_e = metrics.get("mae_excellent", -1)

            mae_str = f"{mae:.2f}" if mae >= 0 else "FAIL"
            mae_t_str = f"{mae_t:.2f}" if mae_t >= 0 else "-"
            mae_m_str = f"{mae_m:.2f}" if mae_m >= 0 else "-"
            mae_e_str = f"{mae_e:.2f}" if mae_e >= 0 else "-"

            print(
                f"{m:<52} {mae_str:>5} {rank:>6.2f} {copy_rate:>5.0%} "
                f"{spread:>6.1f} {ok:>4} {mae_t_str:>6} {mae_m_str:>6} {mae_e_str:>6}"
            )

    print()
    print("LEGEND:")
    print("  MAE     = Mean Absolute Error vs ground truth (lower is better)")
    print("  Rank    = Spearman rank correlation (1.0 = perfect ranking)")
    print("  Copy%   = % of scores matching prompt example values (lower is better)")
    print("  Spread  = Avg score range across tiers per dimension (higher is better)")
    print("  MAE-T/M/E = MAE for Terrible/Mediocre/Excellent tier samples")
    print()

    # Decision guidance
    print("=" * 70)
    print("DECISION GUIDANCE")
    print("=" * 70)

    # Find best models per variant
    for variant in variants:
        print(f"\n--- {variant} ---")
        sorted_results = sorted(
            all_model_results,
            key=lambda r: r["variants"].get(variant, {}).get("metrics", {}).get("mean_mae", 99),
        )

        for mr in sorted_results[:3]:
            m = mr["model"]
            metrics = mr["variants"].get(variant, {}).get("metrics", {})
            mae = metrics.get("mean_mae", -1)
            rank = metrics.get("rank_correlation", 0)
            copy_rate = metrics.get("copying_rate", 0)

            verdict = (
                "UNUSABLE"
                if copy_rate > 0.5
                else ("GOOD" if mae < 2.0 and rank > 0.6 else ("MARGINAL" if mae < 3.0 else "POOR"))
            )
            print(f"  {m}: MAE={mae:.2f} rank={rank:.2f} copy={copy_rate:.0%} => {verdict}")

    if "with_examples" in variants and "parametric" in variants:
        print("\n--- Example Scores Impact ---")
        for mr in all_model_results:
            m = mr["model"]
            ex_copy = (
                mr["variants"].get("with_examples", {}).get("metrics", {}).get("copying_rate", 0)
            )
            param_copy = (
                mr["variants"].get("parametric", {}).get("metrics", {}).get("copying_rate", 0)
            )
            if ex_copy > 0.3 and param_copy < ex_copy:
                reduction = ex_copy - param_copy
                print(
                    f"  {m}: copy rate {ex_copy:.0%} -> {param_copy:.0%} "
                    f"(removing examples helps, -{reduction:.0%})"
                )
            elif ex_copy > 0.3:
                print(f"  {m}: copy rate {ex_copy:.0%} -> {param_copy:.0%} (still copies)")


if __name__ == "__main__":
    main()