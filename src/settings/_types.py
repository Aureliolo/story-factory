"""Type definitions and constants for Story Factory settings."""

import logging
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class ModelInfo(TypedDict):
    """Type definition for model information."""

    name: str
    size_gb: float
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str
    # Tags for role suitability - list of agent roles this model is good for
    tags: list[str]
    # Prompt prefix required by some embedding models before input text.
    # Only relevant for models tagged "embedding". Omit for models that work raw.
    embedding_prefix: NotRequired[str]


class AgentRoleInfo(TypedDict):
    """Type definition for agent role information."""

    name: str
    description: str


# Temperature decay curve options for world quality refinement
REFINEMENT_TEMP_DECAY_CURVES: dict[str, str] = {
    "linear": "Linear",
    "exponential": "Exponential",
    "step": "Step",
}

# Log level options for settings UI
LOG_LEVELS: dict[str, str] = {
    "DEBUG": "Debug",
    "INFO": "Info",
    "WARNING": "Warning",
    "ERROR": "Error",
}


# Agent role definitions
AGENT_ROLES: dict[str, AgentRoleInfo] = {
    "interviewer": {
        "name": "Interviewer",
        "description": "Gathers story requirements",
    },
    "architect": {
        "name": "Architect",
        "description": "Designs story structure",
    },
    "writer": {
        "name": "Writer",
        "description": "Writes prose",
    },
    "editor": {
        "name": "Editor",
        "description": "Polishes prose",
    },
    "continuity": {
        "name": "Continuity Checker",
        "description": "Checks for plot holes",
    },
    "judge": {
        "name": "Judge",
        "description": "Evaluates entity quality for refinement",
    },
    "suggestion": {
        "name": "Suggestion Assistant",
        "description": "Generates writing prompts and suggestions",
    },
    "embedding": {
        "name": "Embedding",
        "description": "Generates text embeddings for semantic similarity",
    },
}


# Minimum quality rating for roles requiring structured output (calibrated numeric
# scores in JSON, structured reasoning, quality prose).  Models below this threshold
# trigger a warning during auto-selection.  Roles not listed here have no minimum.
# Determined empirically via scripts/evaluate_judge_accuracy.py (Issue #228):
# models below these thresholds produce unreliable structured output (high MAE,
# low rank correlation, or excessive copying of prompt example scores).
MINIMUM_ROLE_QUALITY: dict[str, int] = {
    "judge": 7,  # Needs calibrated numeric scores in JSON
    "architect": 6,  # Needs structured story outlines
    "writer": 6,  # Needs quality prose
    "editor": 6,  # Needs quality editing
    "continuity": 6,  # Needs structured analysis
    "suggestion": 6,  # Needs coherent suggestions
}


def check_minimum_quality(model_id: str, quality: float, agent_role: str) -> None:
    """
    Warns when an auto-selected model's quality is below the minimum required for the specified agent role.

    Called during auto-selection only — not when the user explicitly sets a model.

    Parameters:
        model_id (str): The Ollama model identifier.
        quality (float): The model's quality rating from the registry.
        agent_role (str): The agent role being assigned (e.g., "judge").
    """
    min_quality = MINIMUM_ROLE_QUALITY.get(agent_role)
    if min_quality is not None and quality < min_quality:
        logger.warning(
            "Auto-selected model '%s' (quality=%.1f) is below minimum quality %d "
            "for role '%s'. Structured output may be unreliable. "
            "Consider installing a higher-quality model tagged for '%s'.",
            model_id,
            quality,
            min_quality,
            agent_role,
            agent_role,
        )
