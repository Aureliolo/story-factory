"""JSON extraction utilities for parsing LLM responses."""

import json
import logging
import re
from typing import Any

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

    Returns:
        Parsed JSON (dict or list) or None if extraction/parsing fails
    """
    # Strip <think>...</think> tags (some models output reasoning this way)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"</think>", "", response)  # orphan closing tags

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

    logger.warning(f"No valid JSON found in response. Response preview: {response[:200]}...")
    return None


def extract_json_list(response: str) -> list[Any] | None:
    """Extract a JSON array from an LLM response.

    Args:
        response: The LLM response text

    Returns:
        Parsed list or None if extraction/parsing fails
    """
    # Try code block first, then fall back to raw JSON array
    result = extract_json(response, fallback_pattern=r"\[[\s\S]*\]")
    if result is None:
        return None

    if isinstance(result, list):
        return result

    logger.warning(f"Expected JSON array but got {type(result).__name__}")
    return None


def parse_json_to_model[T](
    response: str,
    model_class: type[T],
    fallback_pattern: str | None = None,
) -> T | None:
    """Extract JSON and parse into a Pydantic model.

    Args:
        response: The LLM response text
        model_class: The Pydantic model class to instantiate
        fallback_pattern: Optional regex pattern to try if no code block found

    Returns:
        Instance of model_class or None if extraction/parsing fails
    """
    data = extract_json(response, fallback_pattern)
    if data is None or not isinstance(data, dict):
        if data is not None:
            logger.warning(f"Expected JSON object but got {type(data).__name__}")
        return None

    try:
        return model_class(**data)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to create {model_class.__name__}: {e}")
        return None


def parse_json_list_to_models[T](
    response: str,
    model_class: type[T],
) -> list[T]:
    """Extract JSON array and parse into a list of Pydantic models.

    Args:
        response: The LLM response text
        model_class: The Pydantic model class to instantiate for each item

    Returns:
        List of model instances (empty list on failure)
    """
    data = extract_json_list(response)
    if data is None:
        return []

    results = []
    for i, item in enumerate(data):
        try:
            results.append(model_class(**item))
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to create {model_class.__name__} at index {i}: {e}")

    return results
