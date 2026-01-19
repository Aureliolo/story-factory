"""JSON extraction utilities for parsing LLM responses."""

import json
import logging
import re
from typing import Any

from utils.exceptions import JSONParseError

logger = logging.getLogger(__name__)


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
    except json.JSONDecodeError:
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

    # Strategy 4: Try custom fallback pattern
    if fallback_pattern:
        fallback_match = re.search(fallback_pattern, response, re.DOTALL)
        if fallback_match:
            result = _try_parse_json(fallback_match.group(0))
            if result is not None:
                return result
            logger.debug("Found fallback match but failed to parse")

    # Failure path
    error_msg = f"No valid JSON found in response. Response preview: {response[:200]}..."
    logger.error(error_msg)
    if strict:
        raise JSONParseError(
            error_msg,
            response_preview=response[:500],
            expected_type="dict or list",
        )
    return None


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

    if not isinstance(data, dict):
        error_msg = f"Expected JSON object but got {type(data).__name__}"
        logger.error(error_msg)
        if strict:
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
        logger.error(error_msg)
        if strict:
            raise JSONParseError(
                error_msg,
                response_preview=response[:500],
                expected_type=model_class.__name__,
            ) from e
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
            logger.error(error_msg)
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
