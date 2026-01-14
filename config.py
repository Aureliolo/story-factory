"""Configuration for the Story Factory."""

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tohur/natsumura-storytelling-rp-llama-3.1"

# Generation settings
DEFAULT_TEMPERATURE = 0.8
MAX_TOKENS = 4096

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
