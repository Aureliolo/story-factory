# story-factory Development Patterns

> Auto-generated skill from repository analysis

## Overview

The story-factory is a Python application for generating and refining story worlds through an AI-powered quality loop system. The codebase follows a modular architecture with distinct layers for memory management, services, UI components, and quality assessment. The system uses LLM integration for content generation and implements sophisticated quality scoring and refinement workflows.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files and modules
- Private modules prefixed with underscore: `_quality_loop.py`, `_settings.py`
- Test files follow pattern: `test_*.py`

### Import Style
```python
# Standard library imports first
import json
from typing import List, Dict, Optional

# Third-party imports
from pydantic import BaseModel
import streamlit as st

# Local imports with relative paths
from ..memory.templates import StoryTemplate
from ._quality_loop import QualityLoop
```

### Module Structure
```python
# src/services/world_quality_service/_character.py
from pydantic import BaseModel
from ..llm_client import LLMClient

class CharacterScores(BaseModel):
    consistency: float
    depth: float

async def create_character(prompt: str) -> Character:
    """Create a new character."""
    pass

async def judge_character(character: Character) -> CharacterScores:
    """Evaluate character quality."""
    pass

async def refine_character(character: Character, scores: CharacterScores) -> Character:
    """Improve character based on scores."""
    pass
```

## Workflows

### Add New Entity Type
**Trigger:** When someone wants to add a new type of entity that can be generated, judged, and refined
**Command:** `/add-entity-type`

1. **Create Pydantic model** in `src/memory/templates.py` or `story_state.py`:
   ```python
   class NewEntity(BaseModel):
       id: str
       name: str
       description: str
       # Additional fields specific to entity
   ```

2. **Add quality scores model** in `src/memory/world_quality/_entity_scores.py`:
   ```python
   class NewEntityScores(BaseModel):
       consistency: float
       relevance: float
       quality: float
   ```

3. **Create quality service module** in `src/services/world_quality_service/_newentity.py` with create/judge/refine functions

4. **Add entity type to build pipeline** in `src/services/world_service/_build.py`

5. **Add UI components and pages** in `src/ui/pages/`

6. **Add comprehensive tests** for all modules in `tests/unit/`

### Quality Loop Improvements
**Trigger:** When quality metrics need improvement or loop behavior needs adjustment
**Command:** `/improve-quality-loop`

1. **Update quality loop logic** in `src/services/world_quality_service/_quality_loop.py`:
   ```python
   async def run_quality_loop(entity, max_iterations: int = 3):
       for i in range(max_iterations):
           scores = await judge_entity(entity)
           if scores.overall_quality > threshold:
               break
           entity = await refine_entity(entity, scores)
   ```

2. **Modify scoring models** in `src/memory/world_quality/_models.py`

3. **Add analytics** in `src/services/world_quality_service/_analytics.py`

4. **Update settings and validation** in `src/settings/`

5. **Add UI controls** in `src/ui/pages/settings/`

6. **Add comprehensive tests** for new behaviors

### LLM Integration Changes
**Trigger:** When LLM API changes, structured output needs improvement, or model support needs updates
**Command:** `/update-llm-integration`

1. **Update LLM client** in `src/services/llm_client.py`:
   ```python
   async def generate_structured(self, prompt: str, model_class: BaseModel):
       response = await self.client.generate(
           model=self.model_name,
           prompt=prompt,
           format=model_class.model_json_schema()
       )
       return model_class.model_validate_json(response.response)
   ```

2. **Modify base agent structured output** in `src/agents/base.py`

3. **Update model registry** in `src/settings/_model_registry.py`

4. **Fix streaming utilities** in `src/utils/streaming.py` if needed

5. **Add investigation scripts** in `scripts/` for debugging

6. **Update tests and mocks** in `tests/shared/mock_ollama.py`

### Settings Migration and Validation
**Trigger:** When new configurable behavior needs to be added with user controls
**Command:** `/add-setting`

1. **Add setting fields** to `src/settings/_settings.py`:
   ```python
   class Settings(BaseModel):
       new_feature_enabled: bool = True
       new_feature_threshold: float = 0.7
   ```

2. **Add validation rules** in `src/settings/_validation.py`:
   ```python
   def validate_new_feature_threshold(value: float) -> float:
       if not 0.0 <= value <= 1.0:
           raise ValueError("Threshold must be between 0.0 and 1.0")
       return value
   ```

3. **Create UI controls** in `src/ui/pages/settings/`

4. **Add migration logic** for existing settings files

5. **Update tests** for settings validation and UI

6. **Test integration behavior**

### Database Schema Migration
**Trigger:** When new data needs to be stored or existing schema needs modification
**Command:** `/migrate-schema`

1. **Update schema version** and add migration in `src/memory/world_database/_schema.py`:
   ```python
   CURRENT_VERSION = 3
   
   def migrate_v2_to_v3(conn):
       conn.execute("ALTER TABLE entities ADD COLUMN new_field TEXT")
   ```

2. **Add new database operations** in `src/memory/world_database/__init__.py`

3. **Create helper modules** if needed (like `_cycles.py`)

4. **Update related services** that use the new data

5. **Add migration tests** and integration tests

### UI Component Enhancement
**Trigger:** When UI needs new functionality, performance improvements, or bug fixes
**Command:** `/enhance-ui`

1. **Update component** in `src/ui/components/` or `src/ui/pages/`

2. **Modify related services** for data fetching optimization

3. **Add caching or performance improvements**

4. **Update graph renderer** if visualization changes needed

5. **Add component tests** if testable

### Investigation Script Creation
**Trigger:** When production issues need investigation or model behavior needs analysis
**Command:** `/investigate-issue`

1. **Create investigation script** in `scripts/investigate_*.py`:
   ```python
   # scripts/investigate_model_performance.py
   import asyncio
   from src.services.llm_client import LLMClient
   
   async def main():
       # Investigation logic
       pass
   
   if __name__ == "__main__":
       asyncio.run(main())
   ```

2. **Add helper utilities** in `scripts/_*.py` if needed

3. **Include comprehensive unit tests** in `tests/unit/test_investigate_*.py`

4. **Document findings** and implement fixes in main codebase

5. **Update model registry** or settings based on findings

### World Building Pipeline Improvements
**Trigger:** When world building needs better entity quality, relationship generation, or pipeline reliability
**Command:** `/improve-world-building`

1. **Update build pipeline** in `src/services/world_service/_build.py`

2. **Modify entity generation** in world quality service modules

3. **Update relationship generation** in `src/services/world_quality_service/_relationship.py`

4. **Add health metrics improvements** in `src/services/world_service/_health.py`

5. **Add comprehensive build tests**

## Testing Patterns

### Test Structure
```python
# tests/unit/test_services/test_quality_loop.py
import pytest
from unittest.mock import AsyncMock, patch

from src.services.world_quality_service._quality_loop import QualityLoop

class TestQualityLoop:
    @pytest.mark.asyncio
    async def test_quality_loop_improvement(self):
        # Arrange
        mock_entity = create_mock_entity()
        
        # Act
        result = await QualityLoop.run(mock_entity)
        
        # Assert
        assert result.quality_score > 0.8
```

### Mock Patterns
```python
# tests/shared/mock_ollama.py
class MockOllamaClient:
    async def generate(self, **kwargs):
        return MockResponse(response='{"field": "value"}')
```

## Commands

| Command | Purpose |
|---------|---------|
| `/add-entity-type` | Add support for a new entity type with full quality pipeline |
| `/improve-quality-loop` | Enhance quality refinement loop behavior and metrics |
| `/update-llm-integration` | Modify LLM client integration or model compatibility |
| `/add-setting` | Add new settings with validation and UI controls |
| `/migrate-schema` | Update database schema with migration path |
| `/enhance-ui` | Improve UI components with new features or fixes |
| `/investigate-issue` | Create diagnostic scripts for analysis |
| `/improve-world-building` | Enhance world building pipeline and reliability |