#!/usr/bin/env python3
"""Schema Compliance Investigation — tests which models can produce valid structured output.

For each installed model x each entity/quality schema, attempts structured output
via instructor and records pass/fail/error details. This investigates issue #267 (C1):
models failing structured output for complex schemas.

Test matrix: 6 entity schemas + 5 quality score schemas = 11 schemas per model,
3 trials each = 33 calls per model. Tests with max_retries=0 (raw pass/fail) and
max_retries=3 (can instructor fix it?).

Usage:
    python scripts/investigate_schema_compliance.py [options]
      --models model1,model2    (default: all non-embedding installed models)
      --schemas character,faction  (default: all 11)
      --trials 3                (default: 3)
      --timeout 120             (seconds per call, default: 120)
      --output results.json     (default: output/diagnostics/schema_compliance_<ts>.json)
      --temperature 0.1         (default: 0.1)
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from scripts._ollama_helpers import (
    CANONICAL_BRIEF,
    OLLAMA_BASE,
    get_installed_models,
    get_model_info,
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

# All testable schemas mapped to their Pydantic model class and a generation prompt
SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "character": {
        "model_class": Character,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a complex, morally grey CHARACTER for this story. "
            "Return a JSON object with these fields:\n"
            '- "name": string\n'
            '- "role": string (protagonist, antagonist, supporting, etc.)\n'
            '- "description": string (detailed description)\n'
            '- "personality_traits": list of strings\n'
            '- "goals": list of strings\n'
            '- "relationships": object mapping character names to relationship descriptions\n'
            '- "arc_notes": string describing character development\n'
            "Make the character unique and compelling."
        ),
    },
    "faction": {
        "model_class": Faction,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a FACTION for this story. Return a JSON object with these fields:\n"
            '- "name": string\n'
            '- "type": "faction"\n'
            '- "description": string\n'
            '- "leader": string\n'
            '- "goals": list of strings\n'
            '- "values": list of strings\n'
            '- "base_location": string\n'
            "Make the faction unique with internal tensions."
        ),
    },
    "location": {
        "model_class": Location,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a LOCATION for this story. Return a JSON object with these fields:\n"
            '- "name": string\n'
            '- "type": "location"\n'
            '- "description": string (with rich sensory details)\n'
            '- "significance": string\n'
            "Make the location atmospheric and memorable."
        ),
    },
    "item": {
        "model_class": Item,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a significant ITEM for this story. Return a JSON object with these fields:\n"
            '- "name": string\n'
            '- "type": "item"\n'
            '- "description": string\n'
            '- "significance": string\n'
            '- "properties": list of strings\n'
            "Make the item unique and plot-relevant."
        ),
    },
    "concept": {
        "model_class": Concept,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a thematic CONCEPT for this story. Return a JSON object with these fields:\n"
            '- "name": string\n'
            '- "type": "concept"\n'
            '- "description": string\n'
            '- "manifestations": string\n'
            "Make the concept thematically rich."
        ),
    },
    "relationship": {
        "model_class": Relationship,
        "category": "entity",
        "prompt": (
            f"Story context: {CANONICAL_BRIEF}\n\n"
            "Create a RELATIONSHIP between two characters. Return a JSON object with:\n"
            '- "source": string (source character name)\n'
            '- "target": string (target character name)\n'
            '- "relation_type": string (e.g., allies_with, enemies_with, mentors)\n'
            '- "description": string (history, dynamics, tension)\n'
            "Make the relationship complex and conflict-rich."
        ),
    },
    "character_quality": {
        "model_class": CharacterQualityScores,
        "category": "quality_score",
        "prompt": (
            "You are a quality judge evaluating a character.\n\n"
            "Character: Vesper, a former healer who absorbs others' traumatic memories to cure "
            "suffering, but each absorbed memory overwrites her own identity.\n\n"
            "Rate this character on a 0-10 scale. Return ONLY a JSON object with:\n"
            '- "depth": float (psychological complexity)\n'
            '- "goal_clarity": float (clarity and story relevance of goals)\n'
            '- "flaws": float (meaningful vulnerabilities)\n'
            '- "uniqueness": float (distinctiveness)\n'
            '- "arc_potential": float (room for transformation)\n'
            '- "feedback": string (specific improvement suggestions)\n'
            "Use decimals (e.g., 7.3, not 7)."
        ),
    },
    "location_quality": {
        "model_class": LocationQualityScores,
        "category": "quality_score",
        "prompt": (
            "You are a quality judge evaluating a location.\n\n"
            "Location: The Whispering Archive, a vast underground library carved into living "
            "crystal that resonates with stored memories.\n\n"
            "Rate this location on a 0-10 scale. Return ONLY a JSON object with:\n"
            '- "atmosphere": float (sensory richness, mood)\n'
            '- "narrative_significance": float (plot/symbolic meaning)\n'
            '- "story_relevance": float (theme/character links)\n'
            '- "distinctiveness": float (memorable qualities)\n'
            '- "feedback": string\n'
            "Use decimals."
        ),
    },
    "faction_quality": {
        "model_class": FactionQualityScores,
        "category": "quality_score",
        "prompt": (
            "You are a quality judge evaluating a faction.\n\n"
            "Faction: The Veilkeepers, a fractured order of memory mages who guard the boundary "
            "between remembered and forgotten truths.\n\n"
            "Rate this faction on a 0-10 scale. Return ONLY a JSON object with:\n"
            '- "coherence": float (internal consistency)\n'
            '- "influence": float (world impact)\n'
            '- "conflict_potential": float (story conflict opportunities)\n'
            '- "distinctiveness": float (memorable qualities)\n'
            '- "feedback": string\n'
            "Use decimals."
        ),
    },
    "item_quality": {
        "model_class": ItemQualityScores,
        "category": "quality_score",
        "prompt": (
            "You are a quality judge evaluating an item.\n\n"
            "Item: The Mnemorian Codex, a crystallized book that records memories of anyone who "
            "touches it, slowly extracting their most painful recollections.\n\n"
            "Rate this item on a 0-10 scale. Return ONLY a JSON object with:\n"
            '- "story_significance": float (story importance)\n'
            '- "uniqueness": float (distinctive qualities)\n'
            '- "narrative_potential": float (plot opportunities)\n'
            '- "integration": float (fits world)\n'
            '- "feedback": string\n'
            "Use decimals."
        ),
    },
    "concept_quality": {
        "model_class": ConceptQualityScores,
        "category": "quality_score",
        "prompt": (
            "You are a quality judge evaluating a thematic concept.\n\n"
            "Concept: The Weight of Forgotten Truth — erased memories calcify into physical "
            "'memory stones' that accumulate in the empire's foundations.\n\n"
            "Rate this concept on a 0-10 scale. Return ONLY a JSON object with:\n"
            '- "relevance": float (theme alignment)\n'
            '- "depth": float (philosophical richness)\n'
            '- "manifestation": float (how it appears in story)\n'
            '- "resonance": float (emotional impact)\n'
            '- "feedback": string\n'
            "Use decimals."
        ),
    },
}

ALL_SCHEMA_NAMES = sorted(SCHEMA_REGISTRY.keys())


# =====================================================================
# Schema field count helper
# =====================================================================
def get_schema_field_count(model_class: type[BaseModel]) -> int:
    """Count the number of fields in a Pydantic model.

    Args:
        model_class: Pydantic model class.

    Returns:
        Number of fields.
    """
    return len(model_class.model_fields)


def get_schema_complexity(model_class: type[BaseModel]) -> dict[str, Any]:
    """Analyze schema complexity for reporting.

    Args:
        model_class: Pydantic model class to analyze.

    Returns:
        Dict with field_count, nested_types, constrained_fields, list_fields.
    """
    fields = model_class.model_fields
    nested_types = 0
    constrained_fields = 0
    list_fields = 0

    for field_info in fields.values():
        annotation = field_info.annotation
        annotation_str = str(annotation)

        if "list" in annotation_str.lower():
            list_fields += 1
        if "dict" in annotation_str.lower():
            nested_types += 1
        if field_info.metadata:
            constrained_fields += 1

    return {
        "field_count": len(fields),
        "nested_types": nested_types,
        "constrained_fields": constrained_fields,
        "list_fields": list_fields,
    }


# =====================================================================
# Core test function
# =====================================================================
def test_schema_compliance(
    model: str,
    schema_name: str,
    schema_info: dict[str, Any],
    temperature: float,
    timeout: int,
    max_retries: int,
    verbose: bool,
) -> dict[str, Any]:
    """Test a single model against a single schema.

    Args:
        model: Ollama model name.
        schema_name: Schema identifier.
        schema_info: Schema registry entry with model_class and prompt.
        temperature: Generation temperature.
        timeout: Request timeout in seconds.
        max_retries: Number of instructor retries (0 for raw, 3 for recovery).
        verbose: Print detailed output.

    Returns:
        Dict with pass/fail, timing, error details, raw response capture.
    """
    model_class = schema_info["model_class"]
    prompt = schema_info["prompt"]

    # Capture raw responses on parse errors
    parse_error_messages: list[str] = []

    def on_parse_error(e: Any) -> None:
        """Capture exception message when Pydantic validation fails."""
        error_str = str(e)[:500] if e else ""
        parse_error_messages.append(error_str)
        logger.debug("Parse error captured for %s/%s: %s", model, schema_name, error_str[:100])

    # Create a fresh instructor client for this test
    openai_client = OpenAI(
        base_url=f"{OLLAMA_BASE}/v1",
        api_key="ollama",
        timeout=float(timeout),
    )
    client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
    client.on("parse:error", on_parse_error)

    result: dict[str, Any] = {
        "schema": schema_name,
        "max_retries": max_retries,
        "passed": False,
        "error_type": None,
        "error_detail": None,
        "parse_error_messages": [],
        "validation_errors": [],
        "response_time_seconds": 0.0,
        "parsed_data": None,
    }

    start = time.monotonic()

    try:
        parsed = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=model_class,
            max_retries=max_retries,
            temperature=temperature,
        )
        elapsed = round(time.monotonic() - start, 2)
        result["passed"] = True
        result["response_time_seconds"] = elapsed

        # Serialize the parsed result for the report
        if hasattr(parsed, "model_dump"):
            result["parsed_data"] = parsed.model_dump(mode="json")
        else:
            result["parsed_data"] = str(parsed)

        if verbose:
            logger.info("  PASS %s (retries=%d) in %.1fs", schema_name, max_retries, elapsed)

    except ValidationError as e:
        elapsed = round(time.monotonic() - start, 2)
        result["response_time_seconds"] = elapsed
        result["error_type"] = "validation_error"
        result["error_detail"] = str(e)[:500]
        result["validation_errors"] = [
            {"type": err["type"], "loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()
        ]
        if verbose:
            logger.warning(
                "  FAIL %s (retries=%d) validation: %s",
                schema_name,
                max_retries,
                str(e)[:100],
            )

    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        result["response_time_seconds"] = elapsed
        result["error_type"] = type(e).__name__
        result["error_detail"] = str(e)[:500]
        if verbose:
            logger.warning(
                "  FAIL %s (retries=%d) %s: %s",
                schema_name,
                max_retries,
                type(e).__name__,
                str(e)[:100],
            )

    result["parse_error_messages"] = parse_error_messages
    return result


# =====================================================================
# Per-model evaluation
# =====================================================================
def evaluate_model(
    model: str,
    schemas: list[str],
    trials: int,
    temperature: float,
    timeout: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run all schema compliance tests for a single model.

    Tests each schema with both max_retries=0 (raw) and max_retries=3 (recovery).

    Args:
        model: Ollama model name.
        schemas: List of schema names to test.
        trials: Number of trials per schema per retry mode.
        temperature: Generation temperature.
        timeout: Request timeout per call in seconds.
        verbose: Print detailed output.

    Returns:
        Result dict with per-schema pass rates, error summaries, timing.
    """
    model_start = time.monotonic()
    model_result: dict[str, Any] = {
        "model": model,
        "schema_results": {},
        "summary": {},
        "total_time_seconds": 0.0,
    }

    # Get model details for context
    info = get_model_info(model)
    if info:
        model_result["model_info"] = {
            "parameters": info.get("parameter_size", "unknown"),
            "quantization": info.get("quantization", "unknown"),
            "family": info.get("family", "unknown"),
        }

    total_pass_raw = 0
    total_pass_retry = 0
    total_tests = 0

    for schema_name in schemas:
        schema_info = SCHEMA_REGISTRY[schema_name]
        schema_complexity = get_schema_complexity(schema_info["model_class"])

        schema_result: dict[str, Any] = {
            "category": schema_info["category"],
            "complexity": schema_complexity,
            "raw_trials": [],
            "retry_trials": [],
            "raw_pass_rate": 0.0,
            "retry_pass_rate": 0.0,
            "avg_raw_time": 0.0,
            "avg_retry_time": 0.0,
            "common_errors": [],
        }

        error_types: list[str] = []

        def run_trials(
            max_retries: int,
            label: str,
            _schema_name: str = schema_name,
            _schema_info: dict[str, Any] = schema_info,
            _schema_result: dict[str, Any] = schema_result,
            _error_types: list[str] = error_types,
        ) -> tuple[int, list[float]]:
            """Run trials for a given retry mode, returning pass count and times.

            Args:
                max_retries: Instructor max_retries setting.
                label: Label for verbose output (e.g., "raw", "retry").
                _schema_name: Bound from outer loop to satisfy B023.
                _schema_info: Bound from outer loop to satisfy B023.
                _schema_result: Bound from outer loop to satisfy B023.
                _error_types: Bound from outer loop to satisfy B023.

            Returns:
                Tuple of (pass_count, list of response times).
            """
            passes = 0
            times: list[float] = []
            for trial_idx in range(trials):
                if verbose:
                    print(f"    {_schema_name} {label} trial {trial_idx + 1}/{trials}")
                trial_result = test_schema_compliance(
                    model, _schema_name, _schema_info, temperature, timeout, max_retries, verbose
                )
                _schema_result[f"{label}_trials"].append(trial_result)
                times.append(trial_result["response_time_seconds"])
                if trial_result["passed"]:
                    passes += 1
                elif trial_result["error_type"]:
                    _error_types.append(trial_result["error_type"])
            return passes, times

        raw_passes, raw_times = run_trials(0, "raw")
        retry_passes, retry_times = run_trials(3, "retry")

        # Compute summary for this schema
        schema_result["raw_pass_rate"] = round(raw_passes / trials, 3) if trials > 0 else 0.0
        schema_result["retry_pass_rate"] = round(retry_passes / trials, 3) if trials > 0 else 0.0
        schema_result["avg_raw_time"] = (
            round(sum(raw_times) / len(raw_times), 2) if raw_times else 0.0
        )
        schema_result["avg_retry_time"] = (
            round(sum(retry_times) / len(retry_times), 2) if retry_times else 0.0
        )

        # Deduplicate error types
        error_counts: dict[str, int] = {}
        for et in error_types:
            error_counts[et] = error_counts.get(et, 0) + 1
        schema_result["common_errors"] = [
            {"type": k, "count": v} for k, v in sorted(error_counts.items(), key=lambda x: -x[1])
        ]

        model_result["schema_results"][schema_name] = schema_result
        total_pass_raw += raw_passes
        total_pass_retry += retry_passes
        total_tests += trials

        # Print schema summary
        print(
            f"    {schema_name:<22} raw={raw_passes}/{trials} "
            f"retry={retry_passes}/{trials}  "
            f"t_raw={schema_result['avg_raw_time']:.1f}s "
            f"t_retry={schema_result['avg_retry_time']:.1f}s"
        )

    # Model-level summary
    model_result["summary"] = {
        "total_schemas_tested": len(schemas),
        "total_trials_per_schema": trials,
        "overall_raw_pass_rate": round(total_pass_raw / total_tests, 3) if total_tests > 0 else 0.0,
        "overall_retry_pass_rate": (
            round(total_pass_retry / total_tests, 3) if total_tests > 0 else 0.0
        ),
        "instructor_recovery_rate": (
            round(
                (total_pass_retry - total_pass_raw) / max(total_tests - total_pass_raw, 1),
                3,
            )
            if total_pass_retry > total_pass_raw
            else 0.0
        ),
    }

    model_result["total_time_seconds"] = round(time.monotonic() - model_start, 1)
    return model_result


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the schema compliance investigation."""
    parser = argparse.ArgumentParser(
        description=(
            "Schema Compliance Investigation — tests which models can produce "
            "valid structured output for which Pydantic schemas."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names (default: all installed non-embedding models)",
    )
    parser.add_argument(
        "--schemas",
        type=str,
        help=f"Comma-separated schema names (default: all). Available: {', '.join(ALL_SCHEMA_NAMES)}",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per schema per retry mode (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per call in seconds (default: 120)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-trial results",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = get_installed_models()
        if not models:
            print("ERROR: No models found. Is Ollama running?")
            sys.exit(1)

    # Determine schemas
    if args.schemas:
        schemas = [s.strip() for s in args.schemas.split(",")]
        invalid = [s for s in schemas if s not in SCHEMA_REGISTRY]
        if invalid:
            print(f"ERROR: Unknown schemas: {invalid}")
            print(f"Available: {ALL_SCHEMA_NAMES}")
            sys.exit(1)
    else:
        schemas = ALL_SCHEMA_NAMES

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"schema_compliance_{timestamp}.json"

    # Print header
    calls_per_model = len(schemas) * args.trials * 2  # 2 retry modes
    print("=" * 70)
    print("SCHEMA COMPLIANCE INVESTIGATION")
    print("=" * 70)
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"Schemas: {len(schemas)}")
    for s in schemas:
        complexity = get_schema_complexity(SCHEMA_REGISTRY[s]["model_class"])
        print(f"  - {s} ({complexity['field_count']} fields)")
    print(f"Trials per schema per mode: {args.trials}")
    print("Retry modes: raw (retries=0), recovery (retries=3)")
    print(f"Calls per model: {calls_per_model}")
    print(f"Total calls: {len(models) * calls_per_model}")
    print(f"Temperature: {args.temperature}")
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
            model, schemas, args.trials, args.temperature, args.timeout, args.verbose
        )
        all_model_results.append(model_result)

        summary = model_result["summary"]
        print(
            f"  Summary: raw={summary['overall_raw_pass_rate']:.0%} "
            f"retry={summary['overall_retry_pass_rate']:.0%} "
            f"recovery={summary['instructor_recovery_rate']:.0%} "
            f"({model_result['total_time_seconds']:.0f}s)"
        )

        # Unload model to free VRAM
        print(f"  Unloading {model}...")
        unload_model(model)
        print()

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build schema complexity report
    schema_complexity_report = {}
    for s_name in schemas:
        s_info = SCHEMA_REGISTRY[s_name]
        schema_complexity_report[s_name] = {
            "category": s_info["category"],
            "complexity": get_schema_complexity(s_info["model_class"]),
        }

    # Build output
    output = {
        "investigation_metadata": {
            "script": "investigate_schema_compliance.py",
            "issue": "#267 — C1: Model Structured-Output Compatibility",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "models_tested": len(models),
            "schemas_tested": len(schemas),
            "trials_per_schema": args.trials,
            "temperature": args.temperature,
            "timeout": args.timeout,
        },
        "schema_complexity": schema_complexity_report,
        "model_results": all_model_results,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    # Print compatibility matrix
    print()
    print("=" * 120)
    print("COMPATIBILITY MATRIX — Raw Pass Rate (retries=0)")
    print("=" * 120)

    header = f"{'Model':<45}"
    for s in schemas:
        header += f" {s[:12]:>12}"
    print(header)
    print("-" * (45 + 13 * len(schemas)))

    for mr in all_model_results:
        row = f"{mr['model']:<45}"
        for s in schemas:
            sr = mr["schema_results"].get(s, {})
            rate = sr.get("raw_pass_rate", 0)
            cell = f"{rate:.0%}" if rate > 0 else "FAIL"
            row += f" {cell:>12}"
        print(row)

    print()
    print("=" * 120)
    print("COMPATIBILITY MATRIX — With Instructor Recovery (retries=3)")
    print("=" * 120)

    print(header)
    print("-" * (45 + 13 * len(schemas)))

    for mr in all_model_results:
        row = f"{mr['model']:<45}"
        for s in schemas:
            sr = mr["schema_results"].get(s, {})
            rate = sr.get("retry_pass_rate", 0)
            cell = f"{rate:.0%}" if rate > 0 else "FAIL"
            row += f" {cell:>12}"
        print(row)

    # Print summary recommendations
    print()
    print("=" * 70)
    print("FINDINGS")
    print("=" * 70)

    # Identify problematic schemas (low pass rate across models)
    schema_avg_raw: dict[str, float] = {}
    for s in schemas:
        rates = []
        for mr in all_model_results:
            sr = mr["schema_results"].get(s, {})
            rates.append(sr.get("raw_pass_rate", 0))
        schema_avg_raw[s] = sum(rates) / len(rates) if rates else 0

    problem_schemas = [(s, r) for s, r in schema_avg_raw.items() if r < 0.5]
    if problem_schemas:
        print("\nPROBLEMATIC SCHEMAS (avg raw pass rate < 50%):")
        for s, r in sorted(problem_schemas, key=lambda x: x[1]):
            complexity = get_schema_complexity(SCHEMA_REGISTRY[s]["model_class"])
            print(f"  {s}: {r:.0%} avg pass rate ({complexity['field_count']} fields)")

    # Identify problematic models
    for mr in all_model_results:
        summary = mr["summary"]
        if summary["overall_raw_pass_rate"] < 0.5:
            print(
                f"\nPROBLEMATIC MODEL: {mr['model']} — "
                f"raw={summary['overall_raw_pass_rate']:.0%} "
                f"retry={summary['overall_retry_pass_rate']:.0%}"
            )

    # Recovery effectiveness
    print("\nINSTRUCTOR RECOVERY EFFECTIVENESS:")
    for mr in all_model_results:
        summary = mr["summary"]
        raw = summary["overall_raw_pass_rate"]
        retry = summary["overall_retry_pass_rate"]
        recovery = summary["instructor_recovery_rate"]
        if recovery > 0:
            print(f"  {mr['model']}: {raw:.0%} -> {retry:.0%} (recovery={recovery:.0%})")
        else:
            print(f"  {mr['model']}: {raw:.0%} -> {retry:.0%} (no recovery needed or possible)")

    print()


if __name__ == "__main__":
    main()
