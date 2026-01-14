"""JSON extraction utilities for parsing LLM responses."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(
    response: str,
    fallback_pattern: str | None = None,
) -> dict | None:
    """Extract a JSON object from an LLM response.

    Looks for JSON in markdown code blocks first, then tries fallback pattern.

    Args:
        response: The LLM response text
        fallback_pattern: Optional regex pattern to try if no code block found

    Returns:
        Parsed JSON dict or None if extraction/parsing fails
    """
    # Try markdown code block first
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    elif fallback_pattern:
        # Try fallback pattern
        fallback_match = re.search(fallback_pattern, response, re.DOTALL)
        if fallback_match:
            json_str = fallback_match.group(0)
        else:
            logger.debug("No JSON found in response (tried code block and fallback)")
            return None
    else:
        logger.debug("No JSON code block found in response")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}. Content: {json_str[:200]}...")
        return None


def extract_json_list(response: str) -> list[dict] | None:
    """Extract a JSON array from an LLM response.

    Args:
        response: The LLM response text

    Returns:
        Parsed list of dicts or None if extraction/parsing fails
    """
    result = extract_json(response)
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
    if data is None:
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
