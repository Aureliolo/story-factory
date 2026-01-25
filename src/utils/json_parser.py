"""JSON extraction utilities for parsing LLM responses."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.exceptions import JSONParseError

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Structured result from JSON parsing with diagnostic information.

    Provides detailed information about the parsing process, including
    which repair strategies were attempted and which succeeded.
    """

    data: dict[str, Any] | list[Any] | None = None
    success: bool = False
    repair_applied: str | None = None
    strategies_tried: list[str] = field(default_factory=list)
    original_error: str | None = None

    def __bool__(self) -> bool:
        """Allow ParseResult to be used in boolean context."""
        return self.success


def convert_list_to_dict(
    data: list[Any],
    context_hint: str | None = None,
) -> dict[str, Any]:
    """Convert a list to a dict structure when LLM returns wrong type.

    LLMs sometimes return a list of properties instead of a proper entity dict.
    This function attempts to convert it to a usable dict structure.

    Args:
        data: List data from LLM.
        context_hint: Optional hint about expected structure (e.g., "item", "character").

    Returns:
        Dict with appropriate structure based on list contents.
    """
    if not data:
        logger.debug("convert_list_to_dict: empty list, returning empty dict")
        return {}

    # If list contains dicts, it might be a list of entities - return first
    if all(isinstance(item, dict) for item in data):
        context_info = f" for context '{context_hint}'" if context_hint else ""
        logger.warning(
            f"List contains {len(data)} dict items, returning first as entity{context_info}"
        )
        first_item: dict[str, Any] = data[0]
        return first_item

    # If list contains strings, treat as properties/features
    if all(isinstance(item, str) for item in data):
        context_info = f" for context '{context_hint}'" if context_hint else ""
        logger.warning(f"Converting list of {len(data)} strings to properties dict{context_info}")
        return {
            "properties": data,
            "description": "; ".join(str(x) for x in data[:3]) + ("..." if len(data) > 3 else ""),
        }

    # Mixed types - convert to generic structure
    context_info = f" for context '{context_hint}'" if context_hint else ""
    logger.warning(f"Converting mixed-type list of {len(data)} items to dict{context_info}")
    return {"items": data}


def _iterate_json_chars(text: str) -> list[tuple[int, str, bool]]:
    """Iterate through JSON string characters, tracking string context.

    This helper function handles escape sequences properly and returns
    character positions with their string context.

    Args:
        text: JSON string to iterate.

    Yields:
        Tuples of (index, character, in_string) for each non-escaped character.
    """
    result: list[tuple[int, str, bool]] = []
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append((i, char, not in_string))  # Quote itself is at boundary
            continue
        result.append((i, char, in_string))

    return result


def _count_unescaped_quotes(text: str) -> int:
    """Count unescaped double quotes in text.

    Args:
        text: String to analyze.

    Returns:
        Number of unescaped quote characters.
    """
    return sum(1 for _, char, _ in _iterate_json_chars(text) if char == '"')


def _repair_truncated_json(text: str) -> ParseResult:
    """Attempt multiple strategies to repair truncated JSON.

    Args:
        text: Potentially truncated JSON string.

    Returns:
        ParseResult with repair information and parsed data if successful.
    """
    strategies_tried: list[str] = []

    # Strategy 1: Simple brace/bracket closing
    try:
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0 or open_brackets > 0:
            repaired = text.rstrip(",: \n\t\"'")
            # Close any open string (escape-aware)
            if _count_unescaped_quotes(repaired) % 2 == 1:
                repaired += '"'
            repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            result = _try_parse_json(repaired)
            if result is not None:
                logger.info("Repaired truncated JSON with brace closing")
                strategies_tried.append("brace_closing")
                return ParseResult(
                    data=result,
                    success=True,
                    repair_applied="brace_closing",
                    strategies_tried=strategies_tried,
                )
            strategies_tried.append("brace_closing")
    except Exception:  # pragma: no cover
        strategies_tried.append("brace_closing_failed")

    # Strategy 2: Find last complete object/array and parse that
    try:
        # Find the last complete JSON structure using helper
        brace_count = 0
        bracket_count = 0
        last_complete_idx = -1

        for i, char, in_string in _iterate_json_chars(text):
            if in_string:
                continue

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    last_complete_idx = i + 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    last_complete_idx = i + 1

        if last_complete_idx > 0:
            partial = text[:last_complete_idx]
            result = _try_parse_json(partial)
            if result is not None:
                logger.info("Repaired truncated JSON by extracting complete structure")
                strategies_tried.append("extract_complete")
                return ParseResult(
                    data=result,
                    success=True,
                    repair_applied="extract_complete",
                    strategies_tried=strategies_tried,
                )
            strategies_tried.append("extract_complete")
    except Exception:  # pragma: no cover
        strategies_tried.append("extract_complete_failed")

    # Strategy 3: Truncate at last comma and close
    try:
        # Find last comma outside of strings using helper
        last_comma = -1

        for i, char, in_string in _iterate_json_chars(text):
            if not in_string and char == ",":
                last_comma = i

        if last_comma > 0:
            truncated = text[:last_comma]
            open_braces = truncated.count("{") - truncated.count("}")
            open_brackets = truncated.count("[") - truncated.count("]")
            truncated += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            result = _try_parse_json(truncated)
            if result is not None:
                logger.info("Repaired truncated JSON by truncating at last comma")
                strategies_tried.append("truncate_comma")
                return ParseResult(
                    data=result,
                    success=True,
                    repair_applied="truncate_comma",
                    strategies_tried=strategies_tried,
                )
            strategies_tried.append("truncate_comma")
    except Exception:  # pragma: no cover
        strategies_tried.append("truncate_comma_failed")

    logger.debug(f"All JSON repair strategies failed: {strategies_tried}")
    return ParseResult(
        data=None,
        success=False,
        repair_applied=None,
        strategies_tried=strategies_tried,
        original_error="All repair strategies failed",
    )


def clean_llm_text(text: str) -> str:
    """Clean LLM output text by removing thinking tags and other artifacts.

    Args:
        text: Raw text from LLM output.

    Returns:
        Cleaned text suitable for display.
    """
    if not text:
        return text

    # Remove <think>...</think> blocks (including content)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove orphan opening/closing tags
    cleaned = re.sub(r"</?think>", "", cleaned)

    # Remove other common LLM artifacts
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)  # Special tokens like <|endoftext|>

    # Clean up excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def _try_parse_json(json_str: str) -> dict[str, Any] | list[Any] | None:
    """Try to parse a string as JSON.

    Args:
        json_str: String to parse.

    Returns:
        Parsed JSON or None if parsing fails.
    """
    try:
        parsed: dict[str, Any] | list[Any] = json.loads(json_str.strip())
        return parsed
    except json.JSONDecodeError as e:
        # %.100s format specifier already limits to 100 chars, no need for manual slicing
        logger.debug("try_parse failed: %s (input preview: %.100s...)", e, json_str)
        return None


def extract_json(
    response: str,
    fallback_pattern: str | None = None,
    strict: bool = True,
) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON object or array from an LLM response.

    Tries multiple extraction strategies in order:
    1. ```json code block (markdown standard)
    2. ``` code block (without language marker)
    3. Raw JSON object {...} or array [...]
    4. Custom fallback pattern (if provided)

    Args:
        response: The LLM response text
        fallback_pattern: Optional regex pattern to try if other methods fail
        strict: If True (default), raises JSONParseError on failure.
                If False, returns None on failure (legacy behavior).

    Returns:
        Parsed JSON (dict or list), or None only if strict=False and parsing fails.

    Raises:
        JSONParseError: If strict=True and no valid JSON could be extracted.
    """
    # Strip <think>...</think> tags (some models output reasoning this way)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"</?think>", "", response)  # orphan opening or closing tags

    # Strategy 1: Try ```json code block
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        result = _try_parse_json(json_match.group(1))
        if result is not None:
            return result
        logger.debug("Found ```json block but failed to parse")

    # Strategy 2: Try ``` code block (no language marker)
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        result = _try_parse_json(code_match.group(1))
        if result is not None:
            return result
        logger.debug("Found ``` block but failed to parse")

    # Strategy 3: Try raw JSON object or array
    # Look for outermost { } or [ ]
    json_obj_match = re.search(r"(\{[\s\S]*\})", response)
    if json_obj_match:
        result = _try_parse_json(json_obj_match.group(1))
        if result is not None:
            return result
        logger.debug("Found raw JSON object but failed to parse")

    json_arr_match = re.search(r"(\[[\s\S]*\])", response)
    if json_arr_match:
        result = _try_parse_json(json_arr_match.group(1))
        if result is not None:
            return result
        logger.debug("Found raw JSON array but failed to parse")

    # Strategy 4: Try to repair truncated JSON (common with token limits)
    # Check if response has unbalanced braces/brackets
    stripped = response.strip()
    open_braces = stripped.count("{") - stripped.count("}")
    open_brackets = stripped.count("[") - stripped.count("]")

    if (stripped.startswith("{") or stripped.startswith("[")) and (
        open_braces > 0 or open_brackets > 0
    ):
        logger.warning(
            f"Detected truncated JSON (unbalanced: {open_braces} braces, "
            f"{open_brackets} brackets) - likely token limit reached"
        )
        repair_result = _repair_truncated_json(stripped)
        if repair_result.success:
            return repair_result.data

    # Strategy 5: Try custom fallback pattern
    if fallback_pattern:
        fallback_match = re.search(fallback_pattern, response, re.DOTALL)
        if fallback_match:
            result = _try_parse_json(fallback_match.group(0))
            if result is not None:
                return result
            logger.debug("Found fallback match but failed to parse")

    # Failure path
    error_msg = f"No valid JSON found in response. Response preview: {response[:200]}..."
    if strict:
        logger.error(error_msg)
        raise JSONParseError(
            error_msg,
            response_preview=response[:500],
            expected_type="dict or list",
        )
    # When strict=False, caller expects JSON to possibly not exist (optional parsing)
    logger.debug(error_msg)
    return None


def extract_json_with_info(
    response: str,
    fallback_pattern: str | None = None,
) -> ParseResult:
    """Extract JSON with detailed diagnostic information.

    Similar to extract_json, but returns a ParseResult with information about
    which repair strategies were attempted and which succeeded.

    Args:
        response: The LLM response text
        fallback_pattern: Optional regex pattern to try if other methods fail

    Returns:
        ParseResult with data, success status, and repair information.
    """
    strategies_tried: list[str] = []

    # Strip <think>...</think> tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"</?think>", "", response)

    # Strategy 1: Try ```json code block
    strategies_tried.append("json_code_block")
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        result = _try_parse_json(json_match.group(1))
        if result is not None:
            return ParseResult(
                data=result,
                success=True,
                repair_applied=None,
                strategies_tried=strategies_tried,
            )

    # Strategy 2: Try ``` code block
    strategies_tried.append("plain_code_block")
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        result = _try_parse_json(code_match.group(1))
        if result is not None:
            return ParseResult(
                data=result,
                success=True,
                repair_applied=None,
                strategies_tried=strategies_tried,
            )

    # Strategy 3: Try raw JSON
    strategies_tried.append("raw_json")
    json_obj_match = re.search(r"(\{[\s\S]*\})", response)
    if json_obj_match:
        result = _try_parse_json(json_obj_match.group(1))
        if result is not None:
            return ParseResult(
                data=result,
                success=True,
                repair_applied=None,
                strategies_tried=strategies_tried,
            )

    json_arr_match = re.search(r"(\[[\s\S]*\])", response)
    if json_arr_match:
        result = _try_parse_json(json_arr_match.group(1))
        if result is not None:
            return ParseResult(
                data=result,
                success=True,
                repair_applied=None,
                strategies_tried=strategies_tried,
            )

    # Strategy 4: Truncation repair
    stripped = response.strip()
    open_braces = stripped.count("{") - stripped.count("}")
    open_brackets = stripped.count("[") - stripped.count("]")

    if (stripped.startswith("{") or stripped.startswith("[")) and (
        open_braces > 0 or open_brackets > 0
    ):
        strategies_tried.append("truncation_repair")
        repair_result = _repair_truncated_json(stripped)
        if repair_result.success:
            return ParseResult(
                data=repair_result.data,
                success=True,
                repair_applied=repair_result.repair_applied,
                strategies_tried=strategies_tried + repair_result.strategies_tried,
            )
        strategies_tried.extend(repair_result.strategies_tried)

    # Strategy 5: Fallback pattern
    if fallback_pattern:
        strategies_tried.append("fallback_pattern")
        fallback_match = re.search(fallback_pattern, response, re.DOTALL)
        if fallback_match:
            result = _try_parse_json(fallback_match.group(0))
            if result is not None:
                return ParseResult(
                    data=result,
                    success=True,
                    repair_applied="fallback_pattern",
                    strategies_tried=strategies_tried,
                )

    # Failure
    return ParseResult(
        data=None,
        success=False,
        repair_applied=None,
        strategies_tried=strategies_tried,
        original_error=f"No valid JSON found. Preview: {response[:200]}...",
    )


def extract_json_list(response: str, strict: bool = True) -> list[Any] | None:
    """Extract a JSON array from an LLM response.

    Args:
        response: The LLM response text
        strict: If True (default), raises JSONParseError on failure.
                If False, returns None on failure (legacy behavior).

    Returns:
        Parsed list, or None only if strict=False and parsing fails.

    Raises:
        JSONParseError: If strict=True and no valid JSON array could be extracted.
    """
    # Try code block first, then fall back to raw JSON array
    result = extract_json(response, fallback_pattern=r"\[[\s\S]*\]", strict=strict)
    if result is None:
        return None

    if isinstance(result, list):
        return result

    # LLMs sometimes return a single object when they should return a list
    # Wrap it in a list as a fallback (extract_json only returns dict, list, or None)
    logger.warning("Expected JSON array but got dict - wrapping in list")
    return [result]


def parse_json_to_model[T](
    response: str,
    model_class: type[T],
    fallback_pattern: str | None = None,
    strict: bool = True,
) -> T | None:
    """Extract JSON and parse into a Pydantic model.

    Args:
        response: The LLM response text
        model_class: The Pydantic model class to instantiate
        fallback_pattern: Optional regex pattern to try if no code block found
        strict: If True (default), raises JSONParseError on failure.
                If False, returns None on failure (legacy behavior).

    Returns:
        Instance of model_class, or None only if strict=False and parsing fails.

    Raises:
        JSONParseError: If strict=True and parsing or model instantiation fails.
    """
    data = extract_json(response, fallback_pattern, strict=strict)
    if data is None:
        return None

    if isinstance(data, list):
        # LLM returned list instead of dict - try to convert
        logger.warning(
            f"Expected JSON object but got list for {model_class.__name__} - attempting conversion"
        )
        data = convert_list_to_dict(data, context_hint=model_class.__name__)
        if not data:
            error_msg = f"Expected JSON object but got empty list for {model_class.__name__}"
            if strict:
                logger.error(error_msg)
                raise JSONParseError(
                    error_msg,
                    response_preview=response[:500],
                    expected_type=model_class.__name__,
                )
            return None

    try:
        return model_class(**data)
    except (TypeError, ValueError) as e:
        error_msg = f"Failed to create {model_class.__name__}: {e}"
        if strict:
            logger.error(error_msg)
            raise JSONParseError(
                error_msg,
                response_preview=response[:500],
                expected_type=model_class.__name__,
            ) from e
        logger.debug(error_msg)
        return None


def parse_json_list_to_models[T](
    response: str,
    model_class: type[T],
    strict: bool = True,
    min_count: int = 1,
) -> list[T]:
    """Extract JSON array and parse into a list of Pydantic models.

    Args:
        response: The LLM response text
        model_class: The Pydantic model class to instantiate for each item
        strict: If True (default), raises JSONParseError on failure.
                If False, returns empty list on failure (legacy behavior).
        min_count: Minimum number of successfully parsed models required.
                   Defaults to 1. Only enforced when strict=True.

    Returns:
        List of model instances. Empty list only if strict=False.

    Raises:
        JSONParseError: If strict=True and extraction fails or min_count not met.
    """
    data = extract_json_list(response, strict=strict)
    if data is None:
        return []

    results = []
    failed_items = []
    for i, item in enumerate(data):
        try:
            results.append(model_class(**item))
        except (TypeError, ValueError) as e:
            error_msg = f"Failed to create {model_class.__name__} at index {i}: {e}"
            # Log at debug for individual item failures since we continue processing
            logger.debug(error_msg)
            failed_items.append((i, str(e)))

    # Check if we met minimum count requirement
    if strict and len(results) < min_count:
        error_msg = (
            f"Failed to parse enough {model_class.__name__} items. "
            f"Got {len(results)}, need at least {min_count}. "
            f"Failed items: {failed_items}"
        )
        logger.error(error_msg)
        raise JSONParseError(
            error_msg,
            response_preview=response[:500],
            expected_type=f"list[{model_class.__name__}]",
        )

    return results
