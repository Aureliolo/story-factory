#!/usr/bin/env python3
"""Validator Effectiveness Investigation — does the AI validator add value?

Tests the ValidatorAgent against a corpus of 20 known-quality responses to
determine if smollm2:1.7b adds value beyond regex/rule-based checks.
This investigates issue #267 (A3): questionable validator usefulness.

3 validation paths compared:
  1) Rule-based only (regex CJK check + printable ratio)
  2) AI with smollm2:1.7b (production default)
  3) AI with a larger reference model (e.g., gemma3:4b)

Test corpus: 5 valid English, 5 with CJK mixed in, 5 garbage/non-printable,
5 wrong language = 20 test cases with known ground truth.

Usage:
    python scripts/investigate_validator_effectiveness.py [options]
      --validator-model smollm2:1.7b     (default: smollm2:1.7b)
      --reference-model gemma3:4b        (default: first installed model != validator)
      --timeout 30                       (seconds per call, default: 30)
      --output results.json              (default: output/diagnostics/validator_<ts>.json)
      --verbose
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Ollama endpoint
OLLAMA_BASE = "http://localhost:11434"

# Embedding models to exclude
EMBEDDING_MODELS = {"bge-m3", "snowflake-arctic-embed", "mxbai-embed-large"}

# Production validator thresholds (from src/settings.py defaults)
DEFAULT_CJK_THRESHOLD = 5
DEFAULT_PRINTABLE_RATIO = 0.85
DEFAULT_AI_CHECK_MIN_LENGTH = 200

# Validator system prompt (matches src/agents/validator.py)
VALIDATOR_SYSTEM_PROMPT = """You are a response validator. Your ONLY job is to answer TRUE or FALSE.

You check if AI-generated content:
1. Is written in the CORRECT language (most important!)
2. Is relevant to the task (not random gibberish)
3. Does not contain obvious errors like wrong character encoding

Be strict about language - if the expected language is English but you see Chinese/Japanese/Korean characters, that's FALSE.
Be lenient about content quality - you're just checking basic sanity, not quality.

ALWAYS respond with ONLY one word: TRUE or FALSE. Nothing else."""


# =====================================================================
# Test Corpus — 20 samples with known ground truth
# =====================================================================
TEST_CORPUS: list[dict[str, Any]] = [
    # --- Category 1: Valid English (should PASS all checks) ---
    {
        "id": "valid_english_1",
        "category": "valid_english",
        "text": (
            "The ancient library stood at the heart of the crumbling empire, its shelves "
            "lined with crystallized memories that hummed softly in the dim light. Each "
            "crystal held the recollections of a different citizen, preserved against the "
            "council's decree to erase all traces of the old world."
        ),
        "expected_valid": True,
        "description": "Clean English prose, story-relevant",
    },
    {
        "id": "valid_english_2",
        "category": "valid_english",
        "text": (
            "Commander Thessa surveyed the rubble-strewn courtyard, her jaw tight with "
            "barely contained fury. The explosion had destroyed the eastern wing entirely, "
            "scattering fragments of priceless memory crystals across the flagstones like "
            "glittering tears. Whoever had done this knew exactly what they were targeting."
        ),
        "expected_valid": True,
        "description": "Longer narrative with action",
    },
    {
        "id": "valid_english_3",
        "category": "valid_english",
        "text": (
            '{"name": "The Veilkeepers", "description": "A fractured order of memory mages '
            'who guard the boundary between remembered and forgotten truths.", "goals": '
            '["Protect the Veil of Remembrance", "Resolve internal schism"], '
            '"leader": "The Divided Council"}'
        ),
        "expected_valid": True,
        "description": "Valid JSON entity (structured output)",
    },
    {
        "id": "valid_english_4",
        "category": "valid_english",
        "text": (
            "The concept of memory as currency had transformed the empire's economy. "
            "Citizens could trade vivid recollections for goods, and the wealthy hoarded "
            "rare memories like precious gems. But the real power lay with those who could "
            "forge memories — creating experiences that never happened."
        ),
        "expected_valid": True,
        "description": "World-building exposition",
    },
    {
        "id": "valid_english_5",
        "category": "valid_english",
        "text": (
            "Chapter 3: The Descent\n\n"
            "Mira descended the spiral staircase, each step taking her deeper into the "
            "archive's forbidden eighth level. The whispers grew louder here, coalescing "
            "into half-formed sentences that tugged at her consciousness. She pressed her "
            "palms against her temples and kept walking."
        ),
        "expected_valid": True,
        "description": "Chapter opening with formatting",
    },
    # --- Category 2: CJK Mixed In (should FAIL — wrong language) ---
    {
        "id": "cjk_mixed_1",
        "category": "cjk_mixed",
        "text": (
            "The ancient library 图书馆 stood at the heart of the crumbling empire. "
            "Its shelves 书架 were lined with 水晶 crystallized memories 记忆 that hummed "
            "softly 轻轻地 in the dim 昏暗 light."
        ),
        "expected_valid": False,
        "description": "English with many CJK characters interspersed",
    },
    {
        "id": "cjk_mixed_2",
        "category": "cjk_mixed",
        "text": (
            "Commander Thessa は廃墟の中庭を調査した。爆発は東翼を完全に破壊し、"
            "貴重な記憶の結晶の破片を石畳の上に散らしていた。"
        ),
        "expected_valid": False,
        "description": "Mostly Japanese with English name",
    },
    {
        "id": "cjk_mixed_3",
        "category": "cjk_mixed",
        "text": (
            "在崩溃的帝国中心，古老的图书馆矗立着。The Veilkeepers guard the boundary "
            "between 记住 and 忘记 truths. 他们的领导者是分裂的议会。"
        ),
        "expected_valid": False,
        "description": "Chinese with some English phrases",
    },
    {
        "id": "cjk_mixed_4",
        "category": "cjk_mixed",
        "text": (
            "기억의 힘으로 움직이는 제국에서 한 기록관리사가 지배 위원회의 비밀을 발견한다. "
            "She must navigate rival factions to restore what was lost."
        ),
        "expected_valid": False,
        "description": "Korean with English ending",
    },
    {
        "id": "cjk_mixed_5",
        "category": "cjk_mixed",
        "text": (
            '{"name": "记忆守护者", "description": "一个破碎的记忆法师秩序", '
            '"goals": ["保护记忆之幕", "解决内部分裂"], "leader": "分裂议会"}'
        ),
        "expected_valid": False,
        "description": "JSON with CJK values",
    },
    # --- Category 3: Garbage/Non-printable (should FAIL) ---
    {
        "id": "garbage_1",
        "category": "garbage",
        "text": (
            "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0e\x0f\x10\x11\x12\x13\x14\x15"
            "\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" * 10
        ),
        "expected_valid": False,
        "description": "Pure control characters",
    },
    {
        "id": "garbage_2",
        "category": "garbage",
        "text": "a" * 50 + "\x00" * 200 + "b" * 50,
        "expected_valid": False,
        "description": "Some text with large null block",
    },
    {
        "id": "garbage_3",
        "category": "garbage",
        "text": (
            "jkdf89w3 nmd,./;'[] 2309ru sdlkfj RANDOM sdklfj 329 "
            "asd;lfkj !@#$%^&*() qwerty 98765 zxcvbn poiuy "
            "lkjhgf mnbvcx asdfghjkl qwertyuiop zxcvbnm " * 3
        ),
        "expected_valid": False,
        "description": "Random keyboard mashing (no coherent content)",
    },
    {
        "id": "garbage_4",
        "category": "garbage",
        "text": "\ufffd" * 100 + "\ufffe" * 50 + "\uffff" * 50,
        "expected_valid": False,
        "description": "Unicode replacement and non-characters",
    },
    {
        "id": "garbage_5",
        "category": "garbage",
        "text": (
            "the the the the the the the the the the the the the the "
            "the the the the the the the the the the the the the the "
            "the the the the the the the the the the the the the the " * 3
        ),
        "expected_valid": False,
        "description": "Repetitive single word (degenerate output)",
    },
    # --- Category 4: Wrong Language (coherent but not English) ---
    {
        "id": "wrong_lang_1",
        "category": "wrong_language",
        "text": (
            "La antigua biblioteca se alzaba en el corazón del imperio en ruinas, "
            "sus estantes llenos de recuerdos cristalizados que zumbaban suavemente "
            "en la tenue luz. Cada cristal contenía los recuerdos de un ciudadano diferente."
        ),
        "expected_valid": False,
        "description": "Spanish (coherent but wrong language)",
    },
    {
        "id": "wrong_lang_2",
        "category": "wrong_language",
        "text": (
            "Die alte Bibliothek stand im Herzen des zerfallenden Imperiums. "
            "Ihre Regale waren mit kristallisierten Erinnerungen gefüllt, die "
            "leise im schwachen Licht summten."
        ),
        "expected_valid": False,
        "description": "German (coherent but wrong language)",
    },
    {
        "id": "wrong_lang_3",
        "category": "wrong_language",
        "text": (
            "L'ancienne bibliothèque se dressait au cœur de l'empire en ruine, "
            "ses étagères remplies de souvenirs cristallisés qui bourdonnaient "
            "doucement dans la lumière tamisée."
        ),
        "expected_valid": False,
        "description": "French (coherent but wrong language)",
    },
    {
        "id": "wrong_lang_4",
        "category": "wrong_language",
        "text": (
            "Древняя библиотека стояла в самом сердце рушащейся империи. "
            "Её полки были заполнены кристаллизованными воспоминаниями, "
            "которые тихо гудели в тусклом свете."
        ),
        "expected_valid": False,
        "description": "Russian (coherent but wrong language)",
    },
    {
        "id": "wrong_lang_5",
        "category": "wrong_language",
        "text": (
            "古い図書館は崩壊する帝国の中心に立っていた。その棚には結晶化された"
            "記憶が並び、薄暗い光の中で静かにハミングしていた。それぞれの結晶には"
            "異なる市民の回想が保存されていた。"
        ),
        "expected_valid": False,
        "description": "Japanese (coherent but wrong language)",
    },
]


# =====================================================================
# Ollama API helpers
# =====================================================================
def get_installed_models() -> list[str]:
    """Retrieve installed Ollama model tags, excluding known embedding models.

    Returns:
        Sorted list of model name:tag strings. Empty list on failure.
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
        base_name = name.split(":")[0].split("/")[-1]
        if base_name in EMBEDDING_MODELS:
            continue
        result.append(name)

    return sorted(result)


def unload_model(model: str) -> None:
    """Request Ollama to unload a model from VRAM.

    Args:
        model: Name or tag of the model to unload.
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


def get_model_info(model: str) -> dict[str, Any]:
    """Get model info from Ollama (size, parameters, etc.).

    Args:
        model: Model name to query.

    Returns:
        Dict with model details.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/show",
            json={"name": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        # Calculate file size from model blob sizes
        model_info = data.get("model_info", {})
        return {
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization": details.get("quantization_level", "unknown"),
            "family": details.get("family", "unknown"),
            "total_parameters": model_info.get("general.parameter_count", 0),
        }
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Failed to get info for model '%s': %s", model, e)
        return {}


def call_validator_model(
    model: str,
    text_sample: str,
    expected_language: str,
    task_description: str,
    timeout: int,
) -> dict[str, Any]:
    """Call a model with the validator prompt and return TRUE/FALSE result.

    Mirrors the logic in ValidatorAgent._ai_validate().

    Args:
        model: Ollama model name.
        text_sample: Text to validate (truncated to 1000 chars).
        expected_language: Expected language.
        task_description: Task description.
        timeout: Request timeout.

    Returns:
        Dict with result (bool|None), raw_response, response_time, error.
    """
    sample = text_sample[:1000] + ("..." if len(text_sample) > 1000 else "")

    prompt = f"""Check this AI response:

EXPECTED LANGUAGE: {expected_language}
TASK: {task_description}

RESPONSE SAMPLE:
---
{sample}
---

Is this response:
1. Written in {expected_language}? (CRITICAL)
2. Relevant to the task?
3. Not gibberish or corrupted text?

Answer only TRUE or FALSE."""

    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        elapsed = round(time.monotonic() - start, 2)

        raw = body.get("message", {}).get("content", "").strip()
        raw_upper = raw.upper()

        if "TRUE" in raw_upper:
            result = True
        elif "FALSE" in raw_upper:
            result = False
        else:
            result = None  # Ambiguous

        return {
            "result": result,
            "raw_response": raw[:200],
            "response_time": elapsed,
            "error": None,
        }
    except httpx.TimeoutException:
        elapsed = round(time.monotonic() - start, 2)
        return {"result": None, "raw_response": "", "response_time": elapsed, "error": "timeout"}
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        elapsed = round(time.monotonic() - start, 2)
        return {"result": None, "raw_response": "", "response_time": elapsed, "error": str(e)[:200]}


# =====================================================================
# Validation paths
# =====================================================================
def validate_rule_based(
    text: str,
    expected_language: str = "English",
    cjk_threshold: int = DEFAULT_CJK_THRESHOLD,
    printable_ratio: float = DEFAULT_PRINTABLE_RATIO,
) -> dict[str, Any]:
    """Apply rule-based validation (regex CJK check + printable ratio).

    Mirrors the rule-based checks in ValidatorAgent.validate_response().

    Args:
        text: Text to validate.
        expected_language: Expected language.
        cjk_threshold: Max CJK characters before failure.
        printable_ratio: Min ratio of printable characters.

    Returns:
        Dict with valid (bool), reason, checks_applied.
    """
    start = time.monotonic()

    if not text:
        return {
            "valid": False,
            "reason": "empty_text",
            "checks_applied": ["empty_check"],
            "response_time": 0.0,
        }

    checks_applied = []
    reasons = []

    # CJK check (only for English)
    if expected_language == "English":
        cjk_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
        cjk_matches = cjk_pattern.findall(text)
        checks_applied.append("cjk_check")
        if len(cjk_matches) > cjk_threshold:
            reasons.append(f"cjk_chars={len(cjk_matches)}")

    # Printable ratio check
    printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
    actual_ratio = printable_count / len(text) if text else 0.0
    checks_applied.append("printable_ratio_check")
    if actual_ratio < printable_ratio:
        reasons.append(f"printable_ratio={actual_ratio:.2f}")

    elapsed = round(time.monotonic() - start, 4)

    return {
        "valid": len(reasons) == 0,
        "reason": ", ".join(reasons) if reasons else "passed",
        "checks_applied": checks_applied,
        "response_time": elapsed,
        "cjk_count": len(cjk_matches) if expected_language == "English" else 0,
        "printable_ratio": round(actual_ratio, 3),
    }


def validate_with_ai_model(
    text: str,
    model: str,
    expected_language: str,
    timeout: int,
) -> dict[str, Any]:
    """Apply AI-based validation using a specified model.

    First runs rule-based checks (same as production), then AI check if text is long enough.

    Args:
        text: Text to validate.
        model: Ollama model name.
        expected_language: Expected language.
        timeout: Request timeout.

    Returns:
        Dict with valid (bool), rule_result, ai_result, combined_result.
    """
    # Run rule-based checks first
    rule_result = validate_rule_based(text, expected_language)

    # If rule-based already fails, skip AI (matches production behavior)
    if not rule_result["valid"]:
        return {
            "valid": False,
            "method": "rule_based_only",
            "rule_result": rule_result,
            "ai_result": None,
            "response_time": rule_result["response_time"],
        }

    # Run AI check if text is long enough
    if len(text) <= DEFAULT_AI_CHECK_MIN_LENGTH:
        return {
            "valid": True,
            "method": "rule_based_only_short_text",
            "rule_result": rule_result,
            "ai_result": None,
            "response_time": rule_result["response_time"],
        }

    ai_result = call_validator_model(
        model, text, expected_language, "Generate story content", timeout
    )

    combined_valid = ai_result["result"] is not False  # Fail open on ambiguous
    total_time = round(rule_result["response_time"] + ai_result["response_time"], 3)

    return {
        "valid": combined_valid,
        "method": "rule_based_plus_ai",
        "rule_result": rule_result,
        "ai_result": ai_result,
        "response_time": total_time,
    }


# =====================================================================
# Confusion matrix helpers
# =====================================================================
def compute_confusion_matrix(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute confusion matrix metrics from test results.

    Args:
        results: List of result dicts with 'expected_valid' and 'actual_valid'.

    Returns:
        Dict with tp, fp, tn, fn, accuracy, precision, recall, f1, false_positive_rate.
    """
    tp = sum(1 for r in results if r["expected_valid"] and r["actual_valid"])
    fp = sum(1 for r in results if not r["expected_valid"] and r["actual_valid"])
    tn = sum(1 for r in results if not r["expected_valid"] and not r["actual_valid"])
    fn = sum(1 for r in results if r["expected_valid"] and not r["actual_valid"])

    total = tp + fp + tn + fn
    accuracy = round((tp + tn) / total, 3) if total > 0 else 0.0
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1 = (
        round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    )
    fpr = round(fp / (fp + tn), 3) if (fp + tn) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the validator effectiveness investigation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validator Effectiveness Investigation — tests whether AI validation "
            "adds value beyond rule-based checks."
        )
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="smollm2:1.7b",
        help="Model used for AI validation (default: smollm2:1.7b)",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        help="Reference model for comparison (default: first installed model != validator)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per validation call in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample results",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine reference model
    if args.reference_model:
        reference_model = args.reference_model
    else:
        installed = get_installed_models()
        candidates = [m for m in installed if not m.startswith("smollm")]
        if candidates:
            reference_model = candidates[0]
        elif installed:
            reference_model = installed[0]
        else:
            print("ERROR: No models found. Is Ollama running?")
            sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"validator_effectiveness_{timestamp}.json"

    # Get model info for size comparison
    validator_info = get_model_info(args.validator_model)
    reference_info = get_model_info(reference_model)

    # Print header
    print("=" * 70)
    print("VALIDATOR EFFECTIVENESS INVESTIGATION")
    print("=" * 70)
    print(f"Validator model: {args.validator_model}")
    if validator_info:
        print(
            f"  Size: {validator_info.get('parameter_size', '?')}, "
            f"Quant: {validator_info.get('quantization', '?')}"
        )
    print(f"Reference model: {reference_model}")
    if reference_info:
        print(
            f"  Size: {reference_info.get('parameter_size', '?')}, "
            f"Quant: {reference_info.get('quantization', '?')}"
        )
    print(f"Test corpus: {len(TEST_CORPUS)} samples")
    categories: dict[str, int] = {}
    for tc in TEST_CORPUS:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    print(f"Timeout per call: {args.timeout}s")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Run all 3 validation paths
    overall_start = time.monotonic()

    # Path 1: Rule-based only
    print("Path 1: Rule-based validation only")
    print("-" * 50)
    rule_results: list[dict[str, Any]] = []
    for sample in TEST_CORPUS:
        result = validate_rule_based(sample["text"])
        rule_results.append(
            {
                "sample_id": sample["id"],
                "category": sample["category"],
                "expected_valid": sample["expected_valid"],
                "actual_valid": result["valid"],
                "correct": sample["expected_valid"] == result["valid"],
                "result": result,
            }
        )
        if args.verbose:
            status = "OK" if sample["expected_valid"] == result["valid"] else "WRONG"
            print(
                f"  {sample['id']:<22} expected={sample['expected_valid']} "
                f"actual={result['valid']} {status}"
            )

    rule_matrix = compute_confusion_matrix(rule_results)
    print(f"  Accuracy: {rule_matrix['accuracy']:.0%}")
    print(f"  False positive rate: {rule_matrix['false_positive_rate']:.0%}")
    print()

    # Path 2: AI with validator model
    print(f"Path 2: AI validation with {args.validator_model}")
    print("-" * 50)
    ai_validator_results: list[dict[str, Any]] = []
    for sample in TEST_CORPUS:
        result = validate_with_ai_model(
            sample["text"], args.validator_model, "English", args.timeout
        )
        ai_validator_results.append(
            {
                "sample_id": sample["id"],
                "category": sample["category"],
                "expected_valid": sample["expected_valid"],
                "actual_valid": result["valid"],
                "correct": sample["expected_valid"] == result["valid"],
                "method": result["method"],
                "response_time": result["response_time"],
                "result": result,
            }
        )
        if args.verbose:
            status = "OK" if sample["expected_valid"] == result["valid"] else "WRONG"
            print(
                f"  {sample['id']:<22} expected={sample['expected_valid']} "
                f"actual={result['valid']} method={result['method']} "
                f"t={result['response_time']:.2f}s {status}"
            )

    ai_validator_matrix = compute_confusion_matrix(ai_validator_results)
    avg_ai_time = (
        sum(r["response_time"] for r in ai_validator_results) / len(ai_validator_results)
        if ai_validator_results
        else 0
    )
    print(f"  Accuracy: {ai_validator_matrix['accuracy']:.0%}")
    print(f"  False positive rate: {ai_validator_matrix['false_positive_rate']:.0%}")
    print(f"  Avg response time: {avg_ai_time:.2f}s")

    # Unload validator model
    unload_model(args.validator_model)
    print()

    # Path 3: AI with reference model
    print(f"Path 3: AI validation with {reference_model} (reference)")
    print("-" * 50)
    ai_reference_results: list[dict[str, Any]] = []
    for sample in TEST_CORPUS:
        result = validate_with_ai_model(sample["text"], reference_model, "English", args.timeout)
        ai_reference_results.append(
            {
                "sample_id": sample["id"],
                "category": sample["category"],
                "expected_valid": sample["expected_valid"],
                "actual_valid": result["valid"],
                "correct": sample["expected_valid"] == result["valid"],
                "method": result["method"],
                "response_time": result["response_time"],
                "result": result,
            }
        )
        if args.verbose:
            status = "OK" if sample["expected_valid"] == result["valid"] else "WRONG"
            print(
                f"  {sample['id']:<22} expected={sample['expected_valid']} "
                f"actual={result['valid']} method={result['method']} "
                f"t={result['response_time']:.2f}s {status}"
            )

    ai_reference_matrix = compute_confusion_matrix(ai_reference_results)
    avg_ref_time = (
        sum(r["response_time"] for r in ai_reference_results) / len(ai_reference_results)
        if ai_reference_results
        else 0
    )
    print(f"  Accuracy: {ai_reference_matrix['accuracy']:.0%}")
    print(f"  False positive rate: {ai_reference_matrix['false_positive_rate']:.0%}")
    print(f"  Avg response time: {avg_ref_time:.2f}s")

    # Unload reference model
    unload_model(reference_model)

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
    output = {
        "investigation_metadata": {
            "script": "investigate_validator_effectiveness.py",
            "issue": "#267 — A3: Validator Usefulness",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "validator_model": args.validator_model,
            "validator_info": validator_info,
            "reference_model": reference_model,
            "reference_info": reference_info,
            "test_corpus_size": len(TEST_CORPUS),
            "production_thresholds": {
                "cjk_threshold": DEFAULT_CJK_THRESHOLD,
                "printable_ratio": DEFAULT_PRINTABLE_RATIO,
                "ai_check_min_length": DEFAULT_AI_CHECK_MIN_LENGTH,
            },
        },
        "path_results": {
            "rule_based": {
                "description": "Rule-based only (regex CJK + printable ratio)",
                "confusion_matrix": rule_matrix,
                "avg_response_time": round(
                    sum(r["result"]["response_time"] for r in rule_results) / len(rule_results), 4
                ),
                "per_category": _per_category_accuracy(rule_results),
                "sample_results": rule_results,
            },
            "ai_validator": {
                "description": f"AI validation with {args.validator_model}",
                "model": args.validator_model,
                "confusion_matrix": ai_validator_matrix,
                "avg_response_time": round(avg_ai_time, 3),
                "per_category": _per_category_accuracy(ai_validator_results),
                "sample_results": ai_validator_results,
            },
            "ai_reference": {
                "description": f"AI validation with {reference_model}",
                "model": reference_model,
                "confusion_matrix": ai_reference_matrix,
                "avg_response_time": round(avg_ref_time, 3),
                "per_category": _per_category_accuracy(ai_reference_results),
                "sample_results": ai_reference_results,
            },
        },
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")

    # Print comparison summary
    print()
    print("=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)

    print(f"{'Path':<45} {'Accuracy':>9} {'FPR':>6} {'F1':>6} {'Avg Time':>9}")
    print("-" * 78)
    print(
        f"{'1. Rule-based only':<45} "
        f"{rule_matrix['accuracy']:>8.0%} {rule_matrix['false_positive_rate']:>5.0%} "
        f"{rule_matrix['f1']:>5.2f} {'<0.001s':>9}"
    )
    print(
        f"{'2. AI + ' + args.validator_model:<45} "
        f"{ai_validator_matrix['accuracy']:>8.0%} {ai_validator_matrix['false_positive_rate']:>5.0%} "
        f"{ai_validator_matrix['f1']:>5.2f} {avg_ai_time:>8.2f}s"
    )
    print(
        f"{'3. AI + ' + reference_model:<45} "
        f"{ai_reference_matrix['accuracy']:>8.0%} {ai_reference_matrix['false_positive_rate']:>5.0%} "
        f"{ai_reference_matrix['f1']:>5.2f} {avg_ref_time:>8.2f}s"
    )

    # Per-category breakdown
    print()
    print("PER-CATEGORY ACCURACY:")
    for cat in sorted(categories):
        rule_acc = _category_accuracy(rule_results, cat)
        ai_val_acc = _category_accuracy(ai_validator_results, cat)
        ai_ref_acc = _category_accuracy(ai_reference_results, cat)
        print(f"  {cat:<20} rule={rule_acc:.0%}  ai_val={ai_val_acc:.0%}  ai_ref={ai_ref_acc:.0%}")

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    rule_acc = rule_matrix["accuracy"]
    ai_val_acc = ai_validator_matrix["accuracy"]
    ai_ref_acc = ai_reference_matrix["accuracy"]

    if ai_val_acc > rule_acc + 0.1:
        print(
            f"AI validator ({args.validator_model}) significantly improves accuracy: "
            f"{rule_acc:.0%} -> {ai_val_acc:.0%}"
        )
        print("RECOMMENDATION: Keep AI validator.")
    elif ai_val_acc > rule_acc:
        print(
            f"AI validator ({args.validator_model}) marginally improves accuracy: "
            f"{rule_acc:.0%} -> {ai_val_acc:.0%}"
        )
        print(
            f"But adds {avg_ai_time:.1f}s latency per validation. "
            "Consider if the improvement justifies the cost."
        )
    else:
        print(
            f"AI validator ({args.validator_model}) does NOT improve over rule-based: "
            f"rule={rule_acc:.0%} vs ai={ai_val_acc:.0%}"
        )
        print("RECOMMENDATION: Remove AI validator, use rule-based checks only.")

    if ai_ref_acc > ai_val_acc + 0.1:
        print(
            f"\nReference model ({reference_model}) is significantly better: "
            f"{ai_val_acc:.0%} -> {ai_ref_acc:.0%}"
        )
        print("If keeping AI validation, consider using a larger model.")

    print()


def _per_category_accuracy(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute accuracy per category.

    Args:
        results: List of result dicts with 'category' and 'correct' keys.

    Returns:
        Dict mapping category to accuracy and count.
    """
    categories: dict[str, list[bool]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["correct"])

    return {
        cat: {
            "accuracy": round(sum(vals) / len(vals), 3),
            "correct": sum(vals),
            "total": len(vals),
        }
        for cat, vals in sorted(categories.items())
    }


def _category_accuracy(results: list[dict[str, Any]], category: str) -> float:
    """Get accuracy for a specific category.

    Args:
        results: List of result dicts.
        category: Category to filter.

    Returns:
        Accuracy as float (0-1).
    """
    cat_results = [r for r in results if r["category"] == category]
    if not cat_results:
        return 0.0
    return sum(1 for r in cat_results if r["correct"]) / len(cat_results)


if __name__ == "__main__":
    main()
