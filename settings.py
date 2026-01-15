"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from memory.mode_models import LearningTrigger

# Configure module logger
logger = logging.getLogger(__name__)


SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Centralized paths for story and world output files
STORIES_DIR = Path(__file__).parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent / "output" / "worlds"


class ModelInfo(TypedDict):
    """Type definition for model information."""

    name: str
    release: str
    size_gb: int
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str


class AgentRoleInfo(TypedDict):
    """Type definition for agent role information."""

    name: str
    description: str
    recommended_quality: int


# Available models registry - curated for creative writing (uncensored)
# Organized by use case: creative specialists, general purpose, high-end
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # === CREATIVE WRITING SPECIALISTS ===
    # These models are specifically fine-tuned for fiction and prose
    "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": {
        "name": "Celeste V1.9 12B",
        "release": "2025",
        "size_gb": 13,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Purpose-built for fiction writing, excellent prose quality",
    },
    "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit": {
        "name": "Dark Champion 18B MOE",
        "release": "2025",
        "size_gb": 11,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Exceptional fiction/RP, outstanding prose quality",
    },
    # === GENERAL PURPOSE (FAST) ===
    # Good for interviewer, editor, continuity - fast and reliable
    "huihui_ai/dolphin3-abliterated:8b": {
        "name": "Dolphin 3.0 8B Abliterated",
        "release": "2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7.5,
        "speed": 9,
        "uncensored": True,
        "description": "Eric Hartford's latest, highly compliant, no Chinese output",
    },
    "CognitiveComputations/dolphin-mistral-nemo:12b": {
        "name": "Dolphin Mistral Nemo 12B",
        "release": "2025",
        "size_gb": 7,
        "vram_required": 10,
        "quality": 8,
        "speed": 8,
        "uncensored": True,
        "description": "128K context, excellent for editing and refinement",
    },
    # === HIGH-END (LARGE VRAM / QUANTIZED) ===
    # Best reasoning for story architecture
    "huihui_ai/llama3.3-abliterated:70b": {
        "name": "Llama 3.3 70B Abliterated",
        "release": "December 2024",
        "size_gb": 40,
        "vram_required": 48,
        "quality": 9.5,
        "speed": 5,
        "uncensored": True,
        "description": "Best reasoning, excellent for story architecture",
    },
    "huihui_ai/llama3.3-abliterated:70b-q4_K_M": {
        "name": "Llama 3.3 70B Q4_K_M",
        "release": "December 2024",
        "size_gb": 40,
        "vram_required": 24,
        "quality": 9,
        "speed": 4,
        "uncensored": True,
        "description": "Quantized 70B, fits 24GB VRAM, great for architect role",
    },
    # === LEGACY (kept for compatibility) ===
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated (v1)",
        "release": "April 2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Fast but may output Chinese characters - use Dolphin instead",
    },
}

# Agent role definitions
AGENT_ROLES: dict[str, AgentRoleInfo] = {
    "interviewer": {
        "name": "Interviewer",
        "description": "Gathers story requirements",
        "recommended_quality": 7,  # Doesn't need highest quality
    },
    "architect": {
        "name": "Architect",
        "description": "Designs story structure",
        "recommended_quality": 8,  # Needs good reasoning
    },
    "writer": {
        "name": "Writer",
        "description": "Writes prose",
        "recommended_quality": 9,  # Needs best quality
    },
    "editor": {
        "name": "Editor",
        "description": "Polishes prose",
        "recommended_quality": 8,
    },
    "continuity": {
        "name": "Continuity Checker",
        "description": "Checks for plot holes",
        "recommended_quality": 7,
    },
    "validator": {
        "name": "Validator",
        "description": "Validates AI responses",
        "recommended_quality": 3,  # Uses small/fast model for basic sanity checks
    },
}


@dataclass
class AgentSettings:
    """Settings for a single agent."""

    model: str = "auto"  # "auto" means auto-select based on role
    temperature: float = 0.8


@dataclass
class Settings:
    """Application settings, stored as JSON."""

    # General
    ollama_url: str = "http://localhost:11434"
    context_size: int = 32768
    max_tokens: int = 8192  # Increased to support longer chapters (2000+ words)

    # Default model for all agents (if not using per-agent)
    default_model: str = "huihui_ai/dolphin3-abliterated:8b"

    # Per-agent model settings
    use_per_agent_models: bool = True
    agent_models: dict = field(
        default_factory=lambda: {
            "interviewer": "auto",
            "architect": "auto",
            "writer": "auto",
            "editor": "auto",
            "continuity": "auto",
            "validator": "qwen2.5:0.5b",  # Small, fast model for validation
        }
    )

    # Agent temperatures
    agent_temperatures: dict = field(
        default_factory=lambda: {
            "interviewer": 0.7,
            "architect": 0.85,
            "writer": 0.9,
            "editor": 0.6,
            "continuity": 0.3,
            "validator": 0.1,  # Very low for consistent validation
        }
    )

    # Revision settings
    revision_temperature: float = 0.7  # Lower temperature for more focused revisions

    # Context truncation limits (characters)
    # These control how much context is sent to the LLM to stay within limits
    previous_chapter_context_chars: int = 2000  # End of previous chapter for continuity
    chapter_analysis_chars: int = 4000  # Chapter content for analysis
    full_text_preview_chars: int = 3000  # Text preview for editing suggestions

    # Interaction settings
    interaction_mode: str = "checkpoint"
    chapters_between_checkpoints: int = 3
    max_revision_iterations: int = 3

    # Comparison mode
    comparison_models: list = field(default_factory=list)

    # UI settings
    dark_mode: bool = True
    last_project_id: str | None = None  # Remember last opened project

    # Generation mode settings
    current_mode: str = "balanced"  # ID of active generation mode
    use_mode_system: bool = True  # Whether to use mode-based model selection

    # Learning/tuning settings
    learning_triggers: list = field(
        default_factory=lambda: [
            LearningTrigger.AFTER_PROJECT.value
        ]  # off, after_project, periodic, continuous
    )
    learning_autonomy: str = "balanced"  # manual, cautious, balanced, aggressive, experimental
    learning_periodic_interval: int = 5  # Chapters between periodic analysis
    learning_min_samples: int = 5  # Minimum samples before making recommendations
    learning_confidence_threshold: float = 0.8  # For auto-applying in balanced mode

    def save(self):
        """Save settings to JSON file."""
        # Validate before saving
        self.validate()
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> None:
        """Validate settings values.

        Raises:
            ValueError: If any setting value is invalid.
        """
        # Validate URL format
        try:
            parsed = urlparse(self.ollama_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme in ollama_url: {self.ollama_url}")
            if not parsed.netloc:
                raise ValueError(f"Invalid URL (missing host) in ollama_url: {self.ollama_url}")
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid ollama_url: {self.ollama_url} - {e}") from e

        # Validate numeric ranges
        if not 1024 <= self.context_size <= 128000:
            raise ValueError(
                f"context_size must be between 1024 and 128000, got {self.context_size}"
            )

        if not 256 <= self.max_tokens <= 32000:
            raise ValueError(f"max_tokens must be between 256 and 32000, got {self.max_tokens}")

        if not 1 <= self.chapters_between_checkpoints <= 20:
            raise ValueError(
                f"chapters_between_checkpoints must be between 1 and 20, got {self.chapters_between_checkpoints}"
            )

        if not 0 <= self.max_revision_iterations <= 10:
            raise ValueError(
                f"max_revision_iterations must be between 0 and 10, got {self.max_revision_iterations}"
            )

        # Validate interaction mode
        valid_modes = ["minimal", "checkpoint", "interactive", "collaborative"]
        if self.interaction_mode not in valid_modes:
            raise ValueError(
                f"interaction_mode must be one of {valid_modes}, got {self.interaction_mode}"
            )

        # Validate temperatures
        expected_agents = set(AGENT_ROLES)

        unknown_temp_agents = set(self.agent_temperatures) - expected_agents
        if unknown_temp_agents:
            raise ValueError(
                f"Unknown agent(s) in agent_temperatures: {sorted(unknown_temp_agents)}; "
                f"expected only: {sorted(expected_agents)}"
            )

        for agent, temp in self.agent_temperatures.items():
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Temperature for {agent} must be between 0.0 and 2.0, got {temp}")

        # Validate learning settings
        valid_autonomy = ["manual", "cautious", "balanced", "aggressive", "experimental"]
        if self.learning_autonomy not in valid_autonomy:
            raise ValueError(
                f"learning_autonomy must be one of {valid_autonomy}, got {self.learning_autonomy}"
            )

        valid_triggers = ["off", "after_project", "periodic", "continuous"]
        for trigger in self.learning_triggers:
            if trigger not in valid_triggers:
                raise ValueError(
                    f"Invalid learning trigger: {trigger}; must be one of {valid_triggers}"
                )

        if not 1 <= self.learning_periodic_interval <= 20:
            raise ValueError(
                f"learning_periodic_interval must be between 1 and 20, got {self.learning_periodic_interval}"
            )

        if not 0.0 <= self.learning_confidence_threshold <= 1.0:
            raise ValueError(
                f"learning_confidence_threshold must be between 0.0 and 1.0, got {self.learning_confidence_threshold}"
            )

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from JSON file, or create defaults."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
                settings = cls(**data)
                # Validate loaded settings
                settings.validate()
                return settings
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in settings file, using defaults: {e}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Settings file has invalid values, using defaults: {e}")

        # Create default settings
        settings = cls()
        settings.save()
        return settings

    def get_model_for_agent(self, agent_role: str, available_vram: int = 24) -> str:
        """Get the appropriate model for an agent role.

        If set to 'auto', selects based on agent's quality requirements and available VRAM.
        Special handling for writer role to prefer creative writing specialists.
        """
        if not self.use_per_agent_models:
            return self.default_model

        model_setting: str = self.agent_models.get(agent_role, "auto")

        if model_setting != "auto":
            return model_setting

        # Special case: Writer role prefers creative writing specialists
        if agent_role == "writer":
            creative_models = [
                "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
                "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit",
            ]
            for model_id in creative_models:
                info = AVAILABLE_MODELS.get(model_id)
                if info and info["vram_required"] <= available_vram:
                    return model_id

        # Special case: Architect role prefers high-reasoning models
        if agent_role == "architect":
            architect_models = [
                "huihui_ai/llama3.3-abliterated:70b-q4_K_M",  # Fits 24GB
                "huihui_ai/llama3.3-abliterated:70b",  # Needs 48GB
            ]
            for model_id in architect_models:
                info = AVAILABLE_MODELS.get(model_id)
                if info and info["vram_required"] <= available_vram:
                    return model_id

        # Auto-select based on agent role and VRAM
        role_info: AgentRoleInfo | None = AGENT_ROLES.get(agent_role)
        required_quality: int = role_info["recommended_quality"] if role_info else 7

        # Filter models that fit VRAM and meet quality requirement
        candidates = []
        for model_id, info in AVAILABLE_MODELS.items():
            if info["vram_required"] <= available_vram and info["quality"] >= required_quality:
                candidates.append((model_id, info))

        if not candidates:
            return self.default_model

        # For high quality roles (9+), prioritize quality
        # For lower quality roles, prioritize speed while meeting quality threshold
        if required_quality >= 9:
            # Writer needs best quality - sort by quality desc
            candidates.sort(key=lambda x: x[1]["quality"], reverse=True)
        else:
            # Other roles - prefer speed while meeting quality threshold
            # Find candidates closest to required quality with best speed
            candidates.sort(key=lambda x: (-x[1]["speed"], x[1]["quality"]))

        return candidates[0][0]

    def get_temperature_for_agent(self, agent_role: str) -> float:
        """Get temperature setting for an agent."""
        temp: float = self.agent_temperatures.get(agent_role, 0.8)
        return temp


def get_installed_models() -> list[str]:
    """Get list of models currently installed in Ollama."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except FileNotFoundError:
        logger.warning("Ollama not found. Please ensure Ollama is installed and in PATH.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("Ollama list command timed out.")
        return []
    except (OSError, ValueError) as e:
        logger.warning(f"Error listing Ollama models: {e}")
        return []


def get_available_vram() -> int:
    """Detect available VRAM in GB. Returns 8GB default if detection fails."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        return vram_mb // 1024
    except FileNotFoundError:
        logger.info("nvidia-smi not found. Using default VRAM assumption of 8GB.")
        return 8
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out. Using default VRAM assumption of 8GB.")
        return 8
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse VRAM from nvidia-smi output: {e}. Using default 8GB.")
        return 8
    except OSError as e:
        logger.info(f"Could not detect VRAM: {e}. Using default 8GB.")
        return 8


def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a model."""
    return AVAILABLE_MODELS.get(
        model_id,
        ModelInfo(
            name=model_id,
            release="Unknown",
            size_gb=0,
            vram_required=0,
            quality=5,
            speed=5,
            uncensored=True,
            description="Unknown model",
        ),
    )
