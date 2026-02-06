#!/usr/bin/env python3
"""Retry Deduplication Investigation — are repeated calls byte-identical?

For model/schema pairs that FAIL (from Script 1 output), issues multiple independent
structured-output /api/chat calls and compares raw JSON responses pairwise. This
investigates issue #267 (C2): no learning from repeated failures.

Does NOT hook instructor or observe instructor-managed retries. Instead, measures
variability across standalone /api/chat requests made with format="json".
Compares: byte-identical check, Jaccard token similarity, error escalation.
Tests temperature escalation (0.1 -> 0.3 -> 0.5 -> 0.7) across calls.

Usage:
    python scripts/investigate_retry_deduplication.py [options]
      --compliance-report script1_output.json  (required: Script 1 output)
      --models model1,model2    (override: test specific models instead of using report)
      --schemas character,faction  (override: test specific schemas)
      --attempts 5              (default: 5 retry attempts to capture)
      --timeout 120             (seconds per call, default: 120)
      --output results.json     (default: output/diagnostics/retry_dedup_<ts>.json)
      --verbose
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from pydantic import BaseModel

from scripts._ollama_helpers import (
    OLLAMA_BASE,
    unload_model,
)
from src.memory.story_state import (
    Character,
    Concept,
    Faction,
    Item,
    Location,
    Relationship,
)
from src.memory.world_quality._entity_scores import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Temperature escalation sequence for retry investigation
TEMPERATURE_ESCALATION = [0.1, 0.3, 0.5, 0.7]

# Schema class lookup — must match Script 1's SCHEMA_REGISTRY
SCHEMA_CLASS_MAP: dict[str, type[BaseModel]] = {
    "character": Character,
    "faction": Faction,
    "location": Location,
    "item": Item,
    "concept": Concept,
    "relationship": Relationship,
    "character_quality": CharacterQualityScores,
    "location_quality": LocationQualityScores,
    "faction_quality": FactionQualityScores,
    "item_quality": ItemQualityScores,
    "concept_quality": ConceptQualityScores,
}

# Canonical prompts matching Script 1 (reused for consistency)
CANONICAL_BRIEF = (
    "In a crumbling empire where magic is fueled by memory, a disgraced archivist "
    "discovers that the ruling council has been erasing collective memories to maintain "
    "power. She must navigate rival factions, ancient artifacts, and forbidden knowledge "
    "to restore what was lost before the empire collapses into civil war. "
    "Genre: Fantasy with Political Intrigue. Tone: Dark and suspenseful."
)

SCHEMA_PROMPTS: dict[str, str] = {
    "character": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        "Create a complex, morally grey CHARACTER for this story. "
        'Return a JSON object with: "name", "role", "description", '
        '"personality_traits" (list), "goals" (list), "relationships" (dict), "arc_notes".'
    ),
    "faction": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        "Create a FACTION. Return JSON with: "
        '"name", "type": "faction", "description", "leader", "goals" (list), '
        '"values" (list), "base_location".'
    ),
    "location": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        'Create a LOCATION. Return JSON with: "name", "type": "location", '
        '"description" (with sensory details), "significance".'
    ),
    "item": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        'Create an ITEM. Return JSON with: "name", "type": "item", '
        '"description", "significance", "properties" (list).'
    ),
    "concept": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        'Create a CONCEPT. Return JSON with: "name", "type": "concept", '
        '"description", "manifestations".'
    ),
    "relationship": (
        f"Story context: {CANONICAL_BRIEF}\n\n"
        'Create a RELATIONSHIP. Return JSON with: "source", "target", '
        '"relation_type", "description".'
    ),
    "character_quality": (
        "Rate a character named Vesper (memory-absorbing healer) on 0-10 scale. "
        'Return JSON: "depth", "goal_clarity", "flaws", "uniqueness", "arc_potential", "feedback".'
    ),
    "location_quality": (
        "Rate a location called The Whispering Archive (crystal memory library). "
        'Return JSON: "atmosphere", "narrative_significance", "story_relevance", '
        '"distinctiveness", "feedback".'
    ),
    "faction_quality": (
        "Rate a faction called The Veilkeepers (memory mage order). "
        'Return JSON: "coherence", "influence", "conflict_potential", "distinctiveness", "feedback".'
    ),
    "item_quality": (
        "Rate an item called The Mnemorian Codex (memory-recording crystal book). "
        'Return JSON: "story_significance", "uniqueness", "narrative_potential", '
        '"integration", "feedback".'
    ),
    "concept_quality": (
        "Rate a concept called Weight of Forgotten Truth (calcified memory stones). "
        'Return JSON: "relevance", "depth", "manifestation", "resonance", "feedback".'
    ),
}


# =====================================================================
# Similarity metrics
# =====================================================================
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercased tokens.
    """
    return re.findall(r"\w+", text.lower())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0).
    """
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


def byte_identical(text_a: str, text_b: str) -> bool:
    """Check if two strings are byte-identical.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        True if byte-identical.
    """
    return text_a == text_b


# =====================================================================
# Raw response capture via Ollama chat API (bypasses instructor)
# =====================================================================
def capture_raw_response(
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    """Make a raw Ollama chat call and return the response text.

    Uses format="json" to match instructor's behavior but does NOT validate.

    Args:
        model: Ollama model name.
        prompt: User prompt.
        temperature: Generation temperature.
        timeout: Request timeout in seconds.

    Returns:
        Dict with raw_text, response_time, error.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }

    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        elapsed = round(time.monotonic() - start, 2)

        raw_text = body.get("message", {}).get("content", "")
        return {
            "raw_text": raw_text,
            "response_time": elapsed,
            "error": None,
        }
    except httpx.TimeoutException:
        elapsed = round(time.monotonic() - start, 2)
        logger.warning("Timeout on raw capture for %s", model)
        return {"raw_text": "", "response_time": elapsed, "error": "timeout"}
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        elapsed = round(time.monotonic() - start, 2)
        logger.warning("Error on raw capture for %s: %s", model, e)
        return {"raw_text": "", "response_time": elapsed, "error": str(e)[:200]}


def validate_against_schema(raw_text: str, schema_class: type[BaseModel]) -> dict[str, Any]:
    """Try to parse raw text against a Pydantic schema.

    Args:
        raw_text: JSON text from LLM.
        schema_class: Pydantic model to validate against.

    Returns:
        Dict with valid (bool), error_count (int), error_types (list).
    """
    try:
        data = json.loads(raw_text)
        schema_class.model_validate(data)
        return {"valid": True, "error_count": 0, "error_types": []}
    except json.JSONDecodeError:
        return {"valid": False, "error_count": 1, "error_types": ["json_parse_error"]}
    except Exception as e:
        error_types = []
        if hasattr(e, "errors"):
            error_types = [err.get("type", "unknown") for err in e.errors()]
        return {
            "valid": False,
            "error_count": len(error_types) if error_types else 1,
            "error_types": error_types if error_types else [type(e).__name__],
        }


# =====================================================================
# Core investigation
# =====================================================================
def investigate_retry_pair(
    model: str,
    schema_name: str,
    attempts: int,
    timeout: int,
    verbose: bool,
) -> dict[str, Any]:
    """Investigate retry deduplication for a single model/schema pair.

    Captures multiple raw responses at escalating temperatures and compares them.

    Args:
        model: Ollama model name.
        schema_name: Schema identifier.
        attempts: Number of responses to capture per temperature.
        timeout: Request timeout per call.
        verbose: Print detailed output.

    Returns:
        Dict with pairwise comparisons, identical rates, error escalation.
    """
    schema_class = SCHEMA_CLASS_MAP[schema_name]
    prompt = SCHEMA_PROMPTS[schema_name]

    result: dict[str, Any] = {
        "model": model,
        "schema": schema_name,
        "temperature_results": {},
        "summary": {},
    }

    all_identical_rates: list[float] = []
    all_jaccard_scores: list[float] = []

    for temp in TEMPERATURE_ESCALATION:
        temp_key = f"temp_{temp}"
        responses: list[dict[str, Any]] = []

        if verbose:
            print(f"    temp={temp}: capturing {attempts} responses...")

        for attempt_idx in range(attempts):
            raw = capture_raw_response(model, prompt, temp, timeout)
            validation = validate_against_schema(raw["raw_text"], schema_class)
            responses.append(
                {
                    "attempt": attempt_idx + 1,
                    "raw_text": raw["raw_text"],
                    "response_time": raw["response_time"],
                    "error": raw["error"],
                    "validation": validation,
                }
            )

        # Pairwise comparisons
        pairwise: list[dict[str, Any]] = []
        identical_count = 0
        total_pairs = 0

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                text_a = responses[i]["raw_text"]
                text_b = responses[j]["raw_text"]

                if not text_a or not text_b:
                    continue

                is_identical = byte_identical(text_a, text_b)
                jaccard = jaccard_similarity(text_a, text_b)

                pairwise.append(
                    {
                        "pair": [i + 1, j + 1],
                        "byte_identical": is_identical,
                        "jaccard_similarity": jaccard,
                    }
                )

                total_pairs += 1
                if is_identical:
                    identical_count += 1
                all_jaccard_scores.append(jaccard)

        identical_rate = round(identical_count / total_pairs, 3) if total_pairs > 0 else 0.0
        all_identical_rates.append(identical_rate)

        # Error escalation: do errors get worse across attempts?
        error_counts = [r["validation"]["error_count"] for r in responses]
        error_escalating = len(error_counts) >= 2 and error_counts[-1] > error_counts[0]

        # Collect all error types across attempts
        all_error_types: list[str] = []
        for r in responses:
            all_error_types.extend(r["validation"]["error_types"])
        error_type_counts = dict(Counter(all_error_types))

        temp_result = {
            "temperature": temp,
            "attempts": attempts,
            "responses": [
                {
                    "attempt": r["attempt"],
                    "response_time": r["response_time"],
                    "error": r["error"],
                    "valid": r["validation"]["valid"],
                    "error_count": r["validation"]["error_count"],
                    "error_types": r["validation"]["error_types"],
                    "text_length": len(r["raw_text"]),
                    # Store first 200 chars of raw text for debugging
                    "text_preview": r["raw_text"][:200],
                }
                for r in responses
            ],
            "pairwise_comparisons": pairwise,
            "identical_rate": identical_rate,
            "mean_jaccard": (
                round(sum(j["jaccard_similarity"] for j in pairwise) / len(pairwise), 4)
                if pairwise
                else 0.0
            ),
            "error_escalating": error_escalating,
            "error_type_distribution": error_type_counts,
            "valid_count": sum(1 for r in responses if r["validation"]["valid"]),
        }

        result["temperature_results"][temp_key] = temp_result

        if verbose:
            print(
                f"      identical={identical_rate:.0%} "
                f"jaccard={temp_result['mean_jaccard']:.2f} "
                f"valid={temp_result['valid_count']}/{attempts} "
                f"escalating={'YES' if error_escalating else 'no'}"
            )

    # Summary across all temperatures
    result["summary"] = {
        "avg_identical_rate": (
            round(sum(all_identical_rates) / len(all_identical_rates), 3)
            if all_identical_rates
            else 0.0
        ),
        "avg_jaccard": (
            round(sum(all_jaccard_scores) / len(all_jaccard_scores), 4)
            if all_jaccard_scores
            else 0.0
        ),
        "temperature_helps": (
            all_identical_rates[-1] < all_identical_rates[0]
            if len(all_identical_rates) >= 2
            else False
        ),
        "temperatures_tested": TEMPERATURE_ESCALATION,
    }

    return result


# =====================================================================
# Compliance report loader
# =====================================================================
def load_failing_pairs(report_path: str) -> list[tuple[str, str]]:
    """Load model/schema pairs that failed from Script 1 output.

    Args:
        report_path: Path to Script 1 JSON output.

    Returns:
        List of (model, schema_name) tuples where raw_pass_rate < 1.0.
        Empty list if the file is missing or malformed.
    """
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        logger.error("Compliance report not found: %s", report_path)
        return []
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in compliance report %s: %s", report_path, e)
        return []

    failing_pairs: list[tuple[str, str]] = []

    for model_result in report.get("model_results", []):
        model_name = model_result.get("model")
        if not model_name:
            logger.warning("Skipping model_result entry missing 'model' key")
            continue
        for schema_name, schema_result in model_result.get("schema_results", {}).items():
            if "raw_pass_rate" not in schema_result:
                logger.warning(
                    "Missing raw_pass_rate for %s/%s, treating as failure", model_name, schema_name
                )
                failing_pairs.append((model_name, schema_name))
                continue
            if schema_result["raw_pass_rate"] < 1.0:
                failing_pairs.append((model_name, schema_name))

    logger.info("Loaded %d failing model/schema pairs from %s", len(failing_pairs), report_path)
    return failing_pairs


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the retry deduplication investigation."""
    parser = argparse.ArgumentParser(
        description=(
            "Retry Deduplication Investigation — are retry responses byte-identical? "
            "Is retry feedback wasted on some models?"
        )
    )
    parser.add_argument(
        "--compliance-report",
        type=str,
        help="Path to Script 1 (investigate_schema_compliance.py) output JSON",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Override: comma-separated model names to test",
    )
    parser.add_argument(
        "--schemas",
        type=str,
        help="Override: comma-separated schema names to test",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Number of responses to capture per temperature (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per call in seconds (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine which model/schema pairs to test
    if args.compliance_report:
        failing_pairs = load_failing_pairs(args.compliance_report)
        if not failing_pairs:
            print("No failing pairs found in compliance report. Nothing to investigate.")
            sys.exit(0)
    elif args.models and args.schemas:
        models = [m.strip() for m in args.models.split(",")]
        schemas = [s.strip() for s in args.schemas.split(",")]
        invalid = [s for s in schemas if s not in SCHEMA_CLASS_MAP]
        if invalid:
            print(f"ERROR: Unknown schemas: {invalid}")
            sys.exit(1)
        failing_pairs = [(m, s) for m in models for s in schemas]
    else:
        print(
            "ERROR: Must provide either --compliance-report (Script 1 output) "
            "or both --models and --schemas"
        )
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"retry_dedup_{timestamp}.json"

    # Print header
    unique_models = sorted({m for m, _ in failing_pairs})
    unique_schemas = sorted({s for _, s in failing_pairs})

    print("=" * 70)
    print("RETRY DEDUPLICATION INVESTIGATION")
    print("=" * 70)
    print(f"Failing pairs to investigate: {len(failing_pairs)}")
    print(f"Unique models: {len(unique_models)}")
    for m in unique_models:
        print(f"  - {m}")
    print(f"Unique schemas: {len(unique_schemas)}")
    for s in unique_schemas:
        print(f"  - {s}")
    print(f"Attempts per temperature: {args.attempts}")
    print(f"Temperature escalation: {TEMPERATURE_ESCALATION}")
    calls_per_pair = args.attempts * len(TEMPERATURE_ESCALATION)
    print(f"Calls per pair: {calls_per_pair}")
    print(f"Total calls: {len(failing_pairs) * calls_per_pair}")
    print(f"Timeout per call: {args.timeout}s")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Run investigation
    all_results: list[dict[str, Any]] = []
    overall_start = time.monotonic()
    current_model = None

    for i, (model, schema) in enumerate(failing_pairs):
        # Unload previous model if switching
        if current_model and current_model != model:
            print(f"  Unloading {current_model}...")
            unload_model(current_model)

        current_model = model
        print(f"[{i + 1}/{len(failing_pairs)}] {model} / {schema}")

        pair_result = investigate_retry_pair(
            model, schema, args.attempts, args.timeout, args.verbose
        )
        all_results.append(pair_result)

        summary = pair_result["summary"]
        print(
            f"  => identical={summary['avg_identical_rate']:.0%} "
            f"jaccard={summary['avg_jaccard']:.2f} "
            f"temp_helps={'YES' if summary['temperature_helps'] else 'no'}"
        )

    # Unload final model
    if current_model:
        print(f"\nUnloading {current_model}...")
        unload_model(current_model)

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
    output = {
        "investigation_metadata": {
            "script": "investigate_retry_deduplication.py",
            "issue": "#267 — C2: Retry Response Deduplication",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "pairs_investigated": len(failing_pairs),
            "attempts_per_temperature": args.attempts,
            "temperature_escalation": TEMPERATURE_ESCALATION,
            "compliance_report": args.compliance_report,
        },
        "pair_results": all_results,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")

    # Print summary
    print()
    print("=" * 90)
    print("RETRY DEDUPLICATION SUMMARY")
    print("=" * 90)

    print(f"{'Model':<40} {'Schema':<20} {'Identical%':>10} {'Jaccard':>8} {'Temp Helps':>10}")
    print("-" * 90)

    total_identical = 0.0
    total_jaccard = 0.0
    count = 0

    for r in all_results:
        s = r["summary"]
        temp_helps_str = "YES" if s["temperature_helps"] else "no"
        print(
            f"{r['model']:<40} {r['schema']:<20} "
            f"{s['avg_identical_rate']:>9.0%} {s['avg_jaccard']:>8.2f} "
            f"{temp_helps_str:>10}"
        )
        total_identical += s["avg_identical_rate"]
        total_jaccard += s["avg_jaccard"]
        count += 1

    if count > 0:
        print("-" * 90)
        print(
            f"{'AVERAGE':<40} {'':<20} "
            f"{total_identical / count:>9.0%} {total_jaccard / count:>8.2f}"
        )

    # Key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    high_identical = [r for r in all_results if r["summary"]["avg_identical_rate"] > 0.5]
    if high_identical:
        print(f"\nHIGH IDENTICAL RATE (>50%) — retries are wasted for {len(high_identical)} pairs:")
        for r in high_identical:
            print(
                f"  {r['model']} / {r['schema']}: "
                f"{r['summary']['avg_identical_rate']:.0%} identical"
            )

    temp_helpful = [r for r in all_results if r["summary"]["temperature_helps"]]
    if temp_helpful:
        print(f"\nTEMPERATURE ESCALATION HELPS for {len(temp_helpful)} pairs:")
        for r in temp_helpful:
            print(f"  {r['model']} / {r['schema']}")

    if count > 0:
        avg_identical = total_identical / count
        print(f"\nOVERALL: {avg_identical:.0%} of retry responses are byte-identical on average.")
        if avg_identical > 0.3:
            print(
                "  RECOMMENDATION: Implement error deduplication in circuit breaker. "
                "Many retries produce identical output, wasting tokens."
            )
        else:
            print(
                "  Retries generally produce diverse output. Current retry strategy is reasonable."
            )

    print()


if __name__ == "__main__":
    main()
