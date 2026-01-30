"""Type definitions and constants for Story Factory settings."""

from typing import NotRequired, TypedDict


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
    # Prompt prefix required by embedding models (e.g., "search_document: " for nomic).
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
    "validator": {
        "name": "Validator",
        "description": "Validates AI responses",
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
