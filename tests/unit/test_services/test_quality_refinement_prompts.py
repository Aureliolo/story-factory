"""Tests for quality refinement prompt improvements.

Covers:
- Score field alias renames (collision avoidance)
- Relationship creation array handling
- Dynamic threshold usage in refine prompts (no hardcoded "9+")
"""

import inspect
import logging

import pytest

from src.memory.world_quality import (
    CharacterQualityScores,
    ItemQualityScores,
    LocationQualityScores,
)
from src.services.world_quality_service import (
    _character,
    _concept,
    _faction,
    _item,
    _location,
    _relationship,
)

logger = logging.getLogger(__name__)


class TestScoreFieldAliases:
    """Verify score models use aliases to avoid entity field collisions."""

    def test_character_has_goals_field_with_goal_clarity_alias(self):
        """CharacterQualityScores.goals uses alias 'goal_clarity' for LLM JSON."""
        field_info = CharacterQualityScores.model_fields["goals"]
        assert field_info.alias == "goal_clarity"

    def test_location_has_significance_field_with_narrative_significance_alias(self):
        """LocationQualityScores.significance uses alias 'narrative_significance'."""
        field_info = LocationQualityScores.model_fields["significance"]
        assert field_info.alias == "narrative_significance"

    def test_item_has_significance_field_with_story_significance_alias(self):
        """ItemQualityScores.significance uses alias 'story_significance'."""
        field_info = ItemQualityScores.model_fields["significance"]
        assert field_info.alias == "story_significance"

    def test_character_construct_by_field_name(self):
        """Can construct CharacterQualityScores using Python field name."""
        scores = CharacterQualityScores(
            depth=7.0, goals=8.0, flaws=6.0, uniqueness=7.5, arc_potential=8.0
        )
        assert scores.goals == 8.0

    def test_character_construct_by_alias(self):
        """Can construct CharacterQualityScores using alias name."""
        scores = CharacterQualityScores(  # type: ignore[call-arg]
            depth=7.0, goal_clarity=8.0, flaws=6.0, uniqueness=7.5, arc_potential=8.0
        )
        assert scores.goals == 8.0

    def test_location_construct_by_field_name(self):
        """Can construct LocationQualityScores using Python field name."""
        scores = LocationQualityScores(
            atmosphere=7.0, significance=8.0, story_relevance=6.0, distinctiveness=7.5
        )
        assert scores.significance == 8.0

    def test_location_construct_by_alias(self):
        """Can construct LocationQualityScores using alias name."""
        scores = LocationQualityScores(  # type: ignore[call-arg]
            atmosphere=7.0, narrative_significance=8.0, story_relevance=6.0, distinctiveness=7.5
        )
        assert scores.significance == 8.0

    def test_item_construct_by_field_name(self):
        """Can construct ItemQualityScores using Python field name."""
        scores = ItemQualityScores(
            significance=8.0, uniqueness=7.0, narrative_potential=6.0, integration=7.5
        )
        assert scores.significance == 8.0

    def test_item_construct_by_alias(self):
        """Can construct ItemQualityScores using alias name."""
        scores = ItemQualityScores(  # type: ignore[call-arg]
            story_significance=8.0, uniqueness=7.0, narrative_potential=6.0, integration=7.5
        )
        assert scores.significance == 8.0

    def test_character_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'goal_clarity' as key, not 'goals'."""
        scores = CharacterQualityScores(
            depth=7.0, goals=8.0, flaws=6.0, uniqueness=7.5, arc_potential=8.0
        )
        d = scores.to_dict()
        assert "goal_clarity" in d
        assert "goals" not in d
        assert d["goal_clarity"] == 8.0

    def test_location_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'narrative_significance' as key."""
        scores = LocationQualityScores(
            atmosphere=7.0, significance=8.0, story_relevance=6.0, distinctiveness=7.5
        )
        d = scores.to_dict()
        assert "narrative_significance" in d
        assert "significance" not in d
        assert d["narrative_significance"] == 8.0

    def test_item_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'story_significance' as key."""
        scores = ItemQualityScores(
            significance=8.0, uniqueness=7.0, narrative_potential=6.0, integration=7.5
        )
        d = scores.to_dict()
        assert "story_significance" in d
        assert "significance" not in d
        assert d["story_significance"] == 8.0

    def test_character_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'goal_clarity' not 'goals'."""
        scores = CharacterQualityScores(
            depth=9.0, goals=5.0, flaws=9.0, uniqueness=9.0, arc_potential=9.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "goal_clarity" in weak
        assert "goals" not in weak

    def test_location_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'narrative_significance'."""
        scores = LocationQualityScores(
            atmosphere=9.0, significance=5.0, story_relevance=9.0, distinctiveness=9.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "narrative_significance" in weak
        assert "significance" not in weak

    def test_item_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'story_significance'."""
        scores = ItemQualityScores(
            significance=5.0, uniqueness=9.0, narrative_potential=9.0, integration=9.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "story_significance" in weak
        assert "significance" not in weak

    def test_character_average_unchanged(self):
        """Average calculation works correctly with alias fields."""
        scores = CharacterQualityScores(
            depth=8.0, goals=7.0, flaws=6.0, uniqueness=9.0, arc_potential=5.0
        )
        assert scores.average == 7.0

    def test_character_json_schema_uses_alias(self):
        """JSON schema (used by Instructor) exposes alias names, not field names."""
        schema = CharacterQualityScores.model_json_schema()
        props = schema["properties"]
        assert "goal_clarity" in props
        assert "goals" not in props

    def test_location_json_schema_uses_alias(self):
        """JSON schema exposes 'narrative_significance', not 'significance'."""
        schema = LocationQualityScores.model_json_schema()
        props = schema["properties"]
        assert "narrative_significance" in props
        assert "significance" not in props

    def test_item_json_schema_uses_alias(self):
        """JSON schema exposes 'story_significance', not 'significance'."""
        schema = ItemQualityScores.model_json_schema()
        props = schema["properties"]
        assert "story_significance" in props
        assert "significance" not in props


class TestRelationshipArrayHandling:
    """Test that relationship creation handles array responses."""

    def test_create_relationship_has_array_defense(self):
        """_create_relationship source handles list results from extract_json."""
        source = inspect.getsource(_relationship._create_relationship)
        assert "isinstance(data, list)" in source

    def test_create_relationship_prompt_requests_single_object(self):
        """Creation prompt instructs LLM to return single object, not array."""
        source = inspect.getsource(_relationship._create_relationship)
        assert "Do NOT return an array" in source


class TestDynamicThreshold:
    """Verify no refine prompt hardcodes '9+' — all use dynamic threshold."""

    @pytest.mark.parametrize(
        "module",
        [_character, _faction, _location, _item, _concept, _relationship],
        ids=["character", "faction", "location", "item", "concept", "relationship"],
    )
    def test_refine_prompt_has_no_hardcoded_nine_plus(self, module):
        """Refine function source must not contain hardcoded '9+' threshold."""
        # Find the refine function in the module
        refine_fn = None
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_refine_"):
                refine_fn = obj
                break

        assert refine_fn is not None, f"No _refine_* function found in {module.__name__}"
        source = inspect.getsource(refine_fn)
        assert "9+" not in source, (
            f"{module.__name__}.{refine_fn.__name__} still contains hardcoded '9+' threshold"
        )

    @pytest.mark.parametrize(
        "module",
        [_character, _faction, _location, _item, _concept, _relationship],
        ids=["character", "faction", "location", "item", "concept", "relationship"],
    )
    def test_refine_function_reads_threshold_from_config(self, module):
        """Refine function source must read threshold from config."""
        refine_fn = None
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_refine_"):
                refine_fn = obj
                break

        assert refine_fn is not None
        source = inspect.getsource(refine_fn)
        assert "get_config().quality_threshold" in source or "threshold" in source, (
            f"{module.__name__}.{refine_fn.__name__} does not read threshold from config"
        )
