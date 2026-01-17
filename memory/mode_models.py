"""Pydantic models for generation modes and scoring.

These models define the structure for:
- Generation modes (presets and custom)
- Quality scores
- Tuning recommendations
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VramStrategy(str, Enum):
    """VRAM management strategy for model loading."""

    SEQUENTIAL = "sequential"  # Fully unload between agents
    PARALLEL = "parallel"  # Keep multiple models loaded
    ADAPTIVE = "adaptive"  # Smart loading based on available VRAM


class LearningTrigger(str, Enum):
    """When to trigger learning/tuning analysis."""

    OFF = "off"  # No automatic analysis
    AFTER_PROJECT = "after_project"  # After story completion
    PERIODIC = "periodic"  # Every N chapters
    CONTINUOUS = "continuous"  # Background analysis


class AutonomyLevel(str, Enum):
    """How autonomous the tuning system should be."""

    MANUAL = "manual"  # All changes require approval
    CAUTIOUS = "cautious"  # Auto-apply temp changes, prompt for model swaps
    BALANCED = "balanced"  # Auto-apply when confidence > 80%
    AGGRESSIVE = "aggressive"  # Auto-apply all, just notify
    EXPERIMENTAL = "experimental"  # Try variations to gather data


class GenerationMode(BaseModel):
    """A generation mode defines model assignments and settings per agent.

    Modes can be presets (built-in) or custom (user-created).
    """

    id: str = Field(description="Unique identifier for the mode")
    name: str = Field(description="Display name")
    description: str = Field(default="", description="User-facing description")

    agent_models: dict[str, str] = Field(description="Mapping of agent_role to model_id")
    agent_temperatures: dict[str, float] = Field(description="Mapping of agent_role to temperature")

    vram_strategy: VramStrategy = Field(
        default=VramStrategy.ADAPTIVE,
        description="How to manage VRAM when switching models",
    )

    is_preset: bool = Field(
        default=False,
        description="Whether this is a built-in preset",
    )
    is_experimental: bool = Field(
        default=False,
        description="Whether this mode tries variations to gather data",
    )

    model_config = ConfigDict(use_enum_values=True)


class QualityScores(BaseModel):
    """Quality scores from LLM judge and derived metrics."""

    prose_quality: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="LLM-judged prose quality (0-10)",
    )
    instruction_following: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="How well the output followed the brief (0-10)",
    )
    consistency_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Derived from continuity issues (0-10)",
    )


class PerformanceMetrics(BaseModel):
    """Performance metrics from a generation."""

    tokens_generated: int | None = None
    time_seconds: float | None = None
    tokens_per_second: float | None = None
    vram_used_gb: float | None = None


class ImplicitSignals(BaseModel):
    """Implicit quality signals from user behavior."""

    was_regenerated: bool = Field(
        default=False,
        description="User clicked regenerate (negative signal)",
    )
    edit_distance: int | None = Field(
        default=None,
        description="Levenshtein distance after user edits",
    )
    user_rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional 1-5 star rating from user",
    )


class GenerationScore(BaseModel):
    """Complete scoring record for a single generation."""

    # Context
    project_id: str
    chapter_id: str | None = None
    agent_role: str
    model_id: str
    mode_name: str
    genre: str | None = None

    # Scores
    quality: QualityScores = Field(default_factory=QualityScores)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    signals: ImplicitSignals = Field(default_factory=ImplicitSignals)

    # For A/B comparisons
    prompt_hash: str | None = None

    timestamp: datetime = Field(default_factory=datetime.now)


class RecommendationType(str, Enum):
    """Types of tuning recommendations."""

    MODEL_SWAP = "model_swap"  # Change model for a role
    TEMP_ADJUST = "temp_adjust"  # Adjust temperature
    MODE_CHANGE = "mode_change"  # Switch to different mode
    VRAM_STRATEGY = "vram_strategy"  # Change loading strategy


class TuningRecommendation(BaseModel):
    """A recommendation from the learning/tuning system."""

    id: int | None = Field(default=None, description="Database ID after saving")
    timestamp: datetime = Field(default_factory=datetime.now)

    recommendation_type: RecommendationType = Field(description="Type of change being recommended")
    current_value: str = Field(description="Current setting value")
    suggested_value: str = Field(description="Recommended new value")
    affected_role: str | None = Field(
        default=None,
        description="Agent role affected (for model/temp changes)",
    )

    reason: str = Field(description="Human-readable explanation")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in recommendation (0-1)",
    )
    evidence: dict[str, Any] | None = Field(
        default=None,
        description="Supporting statistics",
    )
    expected_improvement: str | None = Field(
        default=None,
        description="Expected improvement description",
    )

    # Outcome tracking
    was_applied: bool = Field(default=False)
    user_feedback: Literal["accepted", "rejected", "ignored"] | None = Field(default=None)

    model_config = ConfigDict(use_enum_values=True)


class ModelPerformanceSummary(BaseModel):
    """Aggregated performance summary for a model."""

    model_id: str
    agent_role: str
    genre: str | None = None

    avg_prose_quality: float | None = None
    avg_instruction_following: float | None = None
    avg_consistency: float | None = None
    avg_tokens_per_second: float | None = None

    sample_count: int = 0
    last_updated: datetime | None = None


class LearningSettings(BaseModel):
    """Settings for the learning/tuning system."""

    triggers: list[LearningTrigger] = Field(
        default_factory=lambda: [LearningTrigger.AFTER_PROJECT],
        description="When to trigger analysis",
    )
    autonomy: AutonomyLevel = Field(
        default=AutonomyLevel.BALANCED,
        description="How autonomous the tuning should be",
    )
    periodic_interval: int = Field(
        default=5,
        ge=1,
        description="Chapters between periodic analysis",
    )
    min_samples_for_recommendation: int = Field(
        default=5,
        ge=1,
        description="Minimum samples before making recommendations",
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Confidence threshold for auto-applying (balanced mode)",
    )

    model_config = ConfigDict(use_enum_values=True)


# === Preset Mode Definitions ===

PRESET_MODES: dict[str, GenerationMode] = {
    "quality_max": GenerationMode(
        id="quality_max",
        name="Maximum Quality",
        description="Sequential big models, full VRAM utilization",
        agent_models={
            "architect": "huihui_ai/qwen3-abliterated:30b",
            "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "continuity": "deepseek-r1-14b",
            "interviewer": "huihui_ai/dolphin3-abliterated:8b",
            "validator": "smollm2:1.7b",
        },
        agent_temperatures={
            "architect": 0.3,
            "writer": 0.9,
            "editor": 0.5,
            "continuity": 0.1,
            "interviewer": 0.5,
            "validator": 0.1,
        },
        vram_strategy=VramStrategy.SEQUENTIAL,
        is_preset=True,
    ),
    "quality_creative": GenerationMode(
        id="quality_creative",
        name="Creative Focus",
        description="30B reasoning + premium creative writers",
        agent_models={
            "architect": "huihui_ai/qwen3-abliterated:30b",
            "writer": "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit",
            "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "continuity": "huihui_ai/dolphin3-abliterated:8b",
            "interviewer": "huihui_ai/dolphin3-abliterated:8b",
            "validator": "smollm2:1.7b",
        },
        agent_temperatures={
            "architect": 0.3,
            "writer": 1.0,
            "editor": 0.6,
            "continuity": 0.2,
            "interviewer": 0.5,
            "validator": 0.1,
        },
        vram_strategy=VramStrategy.SEQUENTIAL,
        is_preset=True,
    ),
    "balanced": GenerationMode(
        id="balanced",
        name="Balanced",
        description="Good quality with reasonable speed",
        agent_models={
            "architect": "qwen3:14b",
            "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "continuity": "huihui_ai/qwen3-abliterated:8b",
            "interviewer": "huihui_ai/dolphin3-abliterated:8b",
            "validator": "qwen3:0.6b",
        },
        agent_temperatures={
            "architect": 0.4,
            "writer": 0.8,
            "editor": 0.5,
            "continuity": 0.2,
            "interviewer": 0.5,
            "validator": 0.1,
        },
        vram_strategy=VramStrategy.ADAPTIVE,
        is_preset=True,
    ),
    "draft_fast": GenerationMode(
        id="draft_fast",
        name="Fast Draft",
        description="Quick iteration with smaller models",
        agent_models={
            "architect": "huihui_ai/qwen3-abliterated:8b",
            "writer": "huihui_ai/dolphin3-abliterated:8b",
            "editor": "huihui_ai/dolphin3-abliterated:8b",
            "continuity": "qwen3:4b",
            "interviewer": "huihui_ai/dolphin3-abliterated:8b",
            "validator": "qwen3:0.6b",
        },
        agent_temperatures={
            "architect": 0.4,
            "writer": 0.8,
            "editor": 0.5,
            "continuity": 0.2,
            "interviewer": 0.5,
            "validator": 0.1,
        },
        vram_strategy=VramStrategy.PARALLEL,
        is_preset=True,
    ),
    "experimental": GenerationMode(
        id="experimental",
        name="Experimental",
        description="Varies models to gather comparative data",
        agent_models={
            "architect": "huihui_ai/qwen3-abliterated:30b",
            "writer": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "editor": "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
            "continuity": "huihui_ai/qwen3-abliterated:8b",
            "interviewer": "huihui_ai/dolphin3-abliterated:8b",
            "validator": "qwen3:0.6b",
        },
        agent_temperatures={
            "architect": 0.4,
            "writer": 0.9,
            "editor": 0.5,
            "continuity": 0.2,
            "interviewer": 0.5,
            "validator": 0.1,
        },
        vram_strategy=VramStrategy.ADAPTIVE,
        is_preset=True,
        is_experimental=True,
    ),
}


def get_preset_mode(mode_id: str) -> GenerationMode | None:
    """Get a preset mode by ID."""
    return PRESET_MODES.get(mode_id)


def list_preset_modes() -> list[GenerationMode]:
    """List all preset modes."""
    return list(PRESET_MODES.values())
