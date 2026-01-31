"""Shared helpers for world quality service entity modules."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_TEMP_INCREMENT = 0.15
_RETRY_TEMP_MAX = 1.5


def retry_temperature(config: Any, creation_retries: int) -> float:
    """Calculate escalating temperature for creation retries.

    Each retry increases temperature by 0.15 above the configured creator
    temperature, capped at 1.5, to encourage the model to produce different names.

    Args:
        config: RefinementConfig with creator_temperature.
        creation_retries: Number of retries so far.

    Returns:
        Temperature value for the next creation attempt.
    """
    computed = config.creator_temperature + (creation_retries * _RETRY_TEMP_INCREMENT)
    capped = computed > _RETRY_TEMP_MAX
    result = float(min(computed, _RETRY_TEMP_MAX))
    logger.debug(
        "retry_temperature: base=%.2f, retries=%d, computed=%.2f, capped=%s",
        config.creator_temperature,
        creation_retries,
        result,
        capped,
    )
    return result
