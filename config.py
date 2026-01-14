"""Configuration for the Story Factory."""

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"

# Model options (uncomment your preferred model):
# Starter model (8B) - fast, good for testing
DEFAULT_MODEL = "tohur/natsumura-storytelling-rp-llama-3.1"

# Best quality for RTX 4090 (70B) - uncomment to use:
# DEFAULT_MODEL = "vanilj/midnight-miqu-70b-v1.5:Q4_K_M"  # Best prose quality
# DEFAULT_MODEL = "huihui_ai/llama3.3-abliterated"  # Newest, excellent

# Generation settings
DEFAULT_TEMPERATURE = 0.8
MAX_TOKENS = 4096

# Context window - CRITICAL for story continuity
# Ollama defaults to 2048 which is way too low!
# Set based on your VRAM: 8GB=8192, 16GB=16384, 24GB=32768
CONTEXT_SIZE = 32768  # 32K tokens for RTX 4090

# Interaction modes
class InteractionMode:
    MINIMAL = "minimal"      # Only asks at start and shows final result
    CHECKPOINT = "checkpoint"  # Asks at outline + every N chapters + end
    INTERACTIVE = "interactive"  # Asks after each major decision/chapter
    COLLABORATIVE = "collaborative"  # Asks frequently, lets user steer mid-scene

DEFAULT_INTERACTION_MODE = InteractionMode.CHECKPOINT
CHAPTERS_BETWEEN_CHECKPOINTS = 3

# Revision settings
MAX_REVISION_ITERATIONS = 3

# Story length presets
STORY_LENGTHS = {
    "short_story": {"pages": (1, 10), "chapters": None},
    "novella": {"pages": (10, 50), "chapters": (5, 10)},
    "novel": {"pages": (50, 200), "chapters": (15, 40)},
}
