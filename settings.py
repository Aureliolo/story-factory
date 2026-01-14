"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
import subprocess

# Configure module logger
logger = logging.getLogger(__name__)


SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Available models registry - models released in 2025 or late 2024
AVAILABLE_MODELS = {
    # === 2025 Models (Newest) ===
    "huihui_ai/qwen3-abliterated:32b": {
        "name": "Qwen3 32B Abliterated",
        "release": "April 2025",
        "size_gb": 20,
        "vram_required": 22,
        "quality": 9,
        "speed": 6,
        "nsfw": True,
        "description": "Newest Qwen3, excellent reasoning and creative writing",
    },
    "huihui_ai/qwen3-abliterated:14b": {
        "name": "Qwen3 14B Abliterated",
        "release": "April 2025",
        "size_gb": 9,
        "vram_required": 12,
        "quality": 8,
        "speed": 8,
        "nsfw": True,
        "description": "Fast Qwen3 variant, good quality/speed balance",
    },
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated",
        "release": "April 2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "nsfw": True,
        "description": "Compact Qwen3, fast inference",
    },
    "huihui_ai/qwen3-next-abliterated": {
        "name": "Qwen3-Next 70B Abliterated",
        "release": "2025",
        "size_gb": 48,
        "vram_required": 24,
        "quality": 9.5,
        "speed": 5,
        "nsfw": True,
        "description": "Next-gen Qwen3 70B, excellent creative writing",
    },
    "huihui_ai/llama3.3-abliterated": {
        "name": "Llama 3.3 70B Abliterated",
        "release": "December 2024",
        "size_gb": 40,
        "vram_required": 24,
        "quality": 9.5,
        "speed": 5,
        "nsfw": True,
        "description": "Meta's newest 70B, state-of-art performance",
    },
    "goekdenizguelmez/JOSIEFIED-Qwen3:8b": {
        "name": "JOSIEFIED Qwen3 8B",
        "release": "2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7.5,
        "speed": 9,
        "nsfw": True,
        "description": "Fine-tuned Qwen3 for maximum uncensored behavior",
    },
    # === Older but quality models ===
    "tohur/natsumura-storytelling-rp-llama-3.1": {
        "name": "Natsumura 8B (Storytelling)",
        "release": "2024",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 6,
        "speed": 9,
        "nsfw": True,
        "description": "Tuned specifically for storytelling/RP",
    },
}

# Agent role definitions
AGENT_ROLES = {
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
    default_model: str = "huihui_ai/qwen3-abliterated:32b"

    # Per-agent model settings
    use_per_agent_models: bool = True
    agent_models: dict = field(default_factory=lambda: {
        "interviewer": "auto",
        "architect": "auto",
        "writer": "auto",
        "editor": "auto",
        "continuity": "auto",
    })

    # Agent temperatures
    agent_temperatures: dict = field(default_factory=lambda: {
        "interviewer": 0.7,
        "architect": 0.85,
        "writer": 0.9,
        "editor": 0.6,
        "continuity": 0.3,
    })

    # Interaction settings
    interaction_mode: str = "checkpoint"
    chapters_between_checkpoints: int = 3
    max_revision_iterations: int = 3

    # Comparison mode
    comparison_models: list = field(default_factory=list)

    def save(self):
        """Save settings to JSON file."""
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> 'Settings':
        """Load settings from JSON file, or create defaults."""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                
                # Create settings instance
                settings = cls(**data)
                
                # Validate settings
                settings.validate()
                
                return settings
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in settings file, using defaults: {e}")
            except TypeError as e:
                logger.warning(f"Settings file has invalid structure, using defaults: {e}")
            except ValueError as e:
                logger.warning(f"Settings validation failed, using defaults: {e}")

        # Create default settings
        settings = cls()
        settings.save()
        return settings
    
    def validate(self) -> None:
        """Validate settings values.
        
        Raises:
            ValueError: If any setting is invalid
        """
        from utils.validators import validate_model_name, validate_temperature
        
        # Validate context size
        if self.context_size < 2048:
            raise ValueError(f"context_size must be at least 2048, got {self.context_size}")
        if self.context_size > 128000:
            logger.warning(f"context_size {self.context_size} is very large, may cause issues")
        
        # Validate max_tokens
        if self.max_tokens < 512:
            raise ValueError(f"max_tokens must be at least 512, got {self.max_tokens}")
        if self.max_tokens > self.context_size:
            raise ValueError(f"max_tokens ({self.max_tokens}) cannot exceed context_size ({self.context_size})")
        
        # Validate default model
        try:
            validate_model_name(self.default_model)
        except Exception as e:
            raise ValueError(f"Invalid default_model: {e}")
        
        # Validate agent models
        if self.use_per_agent_models:
            for role, model in self.agent_models.items():
                if role not in AGENT_ROLES:
                    logger.warning(f"Unknown agent role in settings: {role}")
                if model and model != "auto":
                    try:
                        validate_model_name(model)
                    except Exception as e:
                        raise ValueError(f"Invalid model for {role}: {e}")
        
        # Validate temperatures
        for role, temp in self.agent_temperatures.items():
            if role not in AGENT_ROLES:
                logger.warning(f"Unknown agent role in temperature settings: {role}")
            try:
                validate_temperature(temp)
            except Exception as e:
                raise ValueError(f"Invalid temperature for {role}: {e}")
        
        # Validate interaction mode
        valid_modes = ["minimal", "checkpoint", "interactive", "collaborative"]
        if self.interaction_mode not in valid_modes:
            raise ValueError(
                f"interaction_mode must be one of {valid_modes}, got {self.interaction_mode}"
            )
        
        # Validate chapters_between_checkpoints
        if self.chapters_between_checkpoints < 1:
            raise ValueError(
                f"chapters_between_checkpoints must be at least 1, got {self.chapters_between_checkpoints}"
            )
        
        # Validate max_revision_iterations
        if self.max_revision_iterations < 0:
            raise ValueError(
                f"max_revision_iterations cannot be negative, got {self.max_revision_iterations}"
            )
        if self.max_revision_iterations > 10:
            logger.warning(
                f"max_revision_iterations {self.max_revision_iterations} is very high, may slow generation"
            )

    def get_model_for_agent(self, agent_role: str, available_vram: int = 24) -> str:
        """Get the appropriate model for an agent role.

        If set to 'auto', selects based on agent's quality requirements and available VRAM.
        """
        if not self.use_per_agent_models:
            return self.default_model

        model_setting = self.agent_models.get(agent_role, "auto")

        if model_setting != "auto":
            return model_setting

        # Auto-select based on agent role and VRAM
        role_info = AGENT_ROLES.get(agent_role, {})
        required_quality = role_info.get("recommended_quality", 7)

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
        return self.agent_temperatures.get(agent_role, 0.8)


def get_installed_models() -> list[str]:
    """Get list of models currently installed in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
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
            timeout=10
        )
        vram_mb = int(result.stdout.strip().split('\n')[0])
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


def get_model_info(model_id: str) -> dict:
    """Get information about a model."""
    return AVAILABLE_MODELS.get(model_id, {
        "name": model_id,
        "release": "Unknown",
        "quality": 5,
        "speed": 5,
        "nsfw": True,
        "description": "Unknown model",
    })
