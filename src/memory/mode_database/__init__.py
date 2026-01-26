"""Mode database package.

This package contains the ModeDatabase class for model scoring and mode management.
The module has been moved to a package structure to support future modular splitting.

Current structure:
- _database.py: Main ModeDatabase class with all methods

Future planned split (from Issue #198):
- _base.py: Schema, migrations, init
- _scoring.py: Score recording and retrieval
- _performance.py: Model performance tracking
- _recommendations.py: Recommendation CRUD
- _custom_modes.py: Custom mode management
- _world_entity.py: World entity scoring
- _prompt_metrics.py: Prompt analytics
- _refinement.py: Refinement effectiveness tracking
- _cost_tracking.py: Cost summaries and generation runs
"""

from src.memory.mode_database._database import DEFAULT_DB_PATH, ModeDatabase

__all__ = ["DEFAULT_DB_PATH", "ModeDatabase"]
