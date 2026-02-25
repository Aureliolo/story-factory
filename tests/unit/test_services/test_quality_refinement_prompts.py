"""Tests for quality refinement prompt improvements.

Covers:
- Score field alias renames (collision avoidance)
- Relationship creation array handling
- Dynamic threshold usage in refine prompts (no hardcoded "9+")
- Temporal fields + calendar context in judge and refinement prompts (#385)
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import (
    Concept,
    Faction,
    Item,
    Location,
    StoryBrief,
    StoryState,
)
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RelationshipQualityScores,
)
from src.services.world_quality_service import (
    _character,
    _concept,
    _faction,
    _item,
    _location,
    _relationship,
)


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
            depth=7.0,
            goals=8.0,
            flaws=6.0,
            uniqueness=7.5,
            arc_potential=8.0,
            temporal_plausibility=7.0,
        )
        assert scores.goals == 8.0

    def test_character_construct_by_alias(self):
        """Can construct CharacterQualityScores using alias name."""
        scores = CharacterQualityScores(  # type: ignore[call-arg]
            depth=7.0,
            goal_clarity=8.0,
            flaws=6.0,
            uniqueness=7.5,
            arc_potential=8.0,
            temporal_plausibility=7.0,
        )
        assert scores.goals == 8.0

    def test_location_construct_by_field_name(self):
        """Can construct LocationQualityScores using Python field name."""
        scores = LocationQualityScores(
            atmosphere=7.0,
            significance=8.0,
            story_relevance=6.0,
            distinctiveness=7.5,
            temporal_plausibility=7.0,
        )
        assert scores.significance == 8.0

    def test_location_construct_by_alias(self):
        """Can construct LocationQualityScores using alias name."""
        scores = LocationQualityScores(  # type: ignore[call-arg]
            atmosphere=7.0,
            narrative_significance=8.0,
            story_relevance=6.0,
            distinctiveness=7.5,
            temporal_plausibility=7.0,
        )
        assert scores.significance == 8.0

    def test_item_construct_by_field_name(self):
        """Can construct ItemQualityScores using Python field name."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=7.5,
            temporal_plausibility=7.0,
        )
        assert scores.significance == 8.0

    def test_item_construct_by_alias(self):
        """Can construct ItemQualityScores using alias name."""
        scores = ItemQualityScores(  # type: ignore[call-arg]
            story_significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=7.5,
            temporal_plausibility=7.0,
        )
        assert scores.significance == 8.0

    def test_character_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'goal_clarity' as key, not 'goals'."""
        scores = CharacterQualityScores(
            depth=7.0,
            goals=8.0,
            flaws=6.0,
            uniqueness=7.5,
            arc_potential=8.0,
            temporal_plausibility=7.0,
        )
        d = scores.to_dict()
        assert "goal_clarity" in d
        assert "goals" not in d
        assert d["goal_clarity"] == 8.0

    def test_location_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'narrative_significance' as key."""
        scores = LocationQualityScores(
            atmosphere=7.0,
            significance=8.0,
            story_relevance=6.0,
            distinctiveness=7.5,
            temporal_plausibility=7.0,
        )
        d = scores.to_dict()
        assert "narrative_significance" in d
        assert "significance" not in d
        assert d["narrative_significance"] == 8.0

    def test_item_to_dict_uses_alias_key(self):
        """to_dict() uses alias name 'story_significance' as key."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=7.5,
            temporal_plausibility=7.0,
        )
        d = scores.to_dict()
        assert "story_significance" in d
        assert "significance" not in d
        assert d["story_significance"] == 8.0

    def test_character_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'goal_clarity' not 'goals'."""
        scores = CharacterQualityScores(
            depth=9.0,
            goals=5.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "goal_clarity" in weak
        assert "goals" not in weak

    def test_location_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'narrative_significance'."""
        scores = LocationQualityScores(
            atmosphere=9.0,
            significance=5.0,
            story_relevance=9.0,
            distinctiveness=9.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "narrative_significance" in weak
        assert "significance" not in weak

    def test_item_weak_dimensions_uses_alias(self):
        """weak_dimensions() returns alias name 'story_significance'."""
        scores = ItemQualityScores(
            significance=5.0,
            uniqueness=9.0,
            narrative_potential=9.0,
            integration=9.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "story_significance" in weak
        assert "significance" not in weak

    def test_character_average_unchanged(self):
        """Average calculation works correctly with alias fields."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=7.0,
            flaws=6.0,
            uniqueness=9.0,
            arc_potential=5.0,
            temporal_plausibility=7.0,
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
    """Test that relationship creation uses schema-constrained generation."""

    def test_create_relationship_uses_schema_constrained_generation(self):
        """_create_relationship uses json.loads with grammar-constrained format=schema."""
        source = inspect.getsource(_relationship._create_relationship)
        assert "json.loads" in source
        assert "format=schema" in source

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


class TestWeakDimensionsBranchCoverage:
    """Ensure every dimension branch in weak_dimensions is exercised."""

    def test_character_depth_and_uniqueness_weak(self):
        """Cover depth and uniqueness branches in CharacterQualityScores.weak_dimensions."""
        scores = CharacterQualityScores(
            depth=3.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=4.0,
            arc_potential=9.0,
            temporal_plausibility=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "depth" in weak
        assert "uniqueness" in weak
        assert "goal_clarity" not in weak

    def test_relationship_dynamics_and_authenticity_weak(self):
        """Cover dynamics and authenticity branches in RelationshipQualityScores."""
        scores = RelationshipQualityScores(
            tension=9.0, dynamics=3.0, story_potential=9.0, authenticity=4.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "dynamics" in weak
        assert "authenticity" in weak
        assert "tension" not in weak


class TestRelationshipSchemaDefense:
    """Test that relationship creation guards against non-dict JSON responses."""

    def test_create_relationship_has_non_dict_guard(self):
        """_create_relationship validates json.loads result is a dict."""
        source = inspect.getsource(_relationship._create_relationship)
        assert "isinstance(data, dict)" in source

    def test_refine_relationship_has_non_dict_guard(self):
        """_refine_relationship validates json.loads result is a dict."""
        source = inspect.getsource(_relationship._refine_relationship)
        assert "isinstance(data, dict)" in source


@pytest.fixture
def make_mock_svc():
    """Factory fixture: build a mock WorldQualityService with calendar context.

    Returns a callable that accepts an optional ``calendar_text`` override.
    """

    def _factory(calendar_text: str = "Era of Flames: 0-500 AF") -> MagicMock:
        """Build a mock WorldQualityService with the given calendar text."""
        svc = MagicMock()
        svc.get_calendar_context.return_value = f"\nCALENDAR & TIMELINE:\n{calendar_text}\n"
        svc._get_judge_model.return_value = "test-judge:8b"
        svc.settings = MagicMock()

        # Judge config with multi-call disabled (simplest path)
        judge_config = MagicMock()
        judge_config.enabled = False
        judge_config.multi_call_enabled = False
        svc.get_judge_config.return_value = judge_config

        # Refinement config with threshold
        config = MagicMock()
        config.get_threshold.return_value = 7.5
        svc.get_config.return_value = config

        # Creator model for refine functions
        svc._get_creator_model.return_value = "test-creator:8b"
        svc._format_properties.return_value = "magical, ancient"

        return svc

    return _factory


@pytest.fixture
def story_state() -> StoryState:
    """Minimal StoryState for testing."""
    brief = StoryBrief(
        premise="A kingdom falls",
        genre="fantasy",
        tone="dark",
        themes=["power", "betrayal"],
        setting_place="Kingdom",
        setting_time="Medieval",
        target_length="short_story",
        content_rating="none",
        language="English",
    )
    return StoryState(id="test-story", brief=brief)


class TestJudgePromptTemporalFields:
    """Verify judge prompts include temporal fields and calendar context (#385)."""

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_faction_judge_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Faction judge prompt must contain Founding Era, Dissolution Year, Temporal Notes."""
        mock_gen.return_value = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        faction = {
            "name": "Iron Covenant",
            "description": "A secretive guild",
            "leader": "Lord Vex",
            "goals": ["domination"],
            "values": ["loyalty"],
            "founding_year": 200,
            "dissolution_year": 450,
            "founding_era": "Era of Flames",
            "temporal_notes": "Founded during the Great War",
        }

        _faction._judge_faction_quality(svc, faction, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Founding Era: Era of Flames" in prompt
        assert "Dissolution Year: 450" in prompt
        assert "Temporal Notes: Founded during the Great War" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_item_judge_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Item judge prompt must contain Creation Era, Temporal Notes."""
        mock_gen.return_value = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        item = {
            "name": "Flame Sword",
            "description": "A burning blade",
            "significance": "Key weapon",
            "properties": ["fire damage"],
            "creation_year": 100,
            "creation_era": "Age of Forging",
            "temporal_notes": "Forged in the first furnace",
        }

        _item._judge_item_quality(svc, item, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Creation Era: Age of Forging" in prompt
        assert "Temporal Notes: Forged in the first furnace" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_concept_judge_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Concept judge prompt must contain Emergence Era, Temporal Notes."""
        mock_gen.return_value = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        concept = {
            "name": "The Binding",
            "description": "A cosmic force",
            "manifestations": "Appears as golden chains",
            "emergence_year": 50,
            "emergence_era": "Dawn Era",
            "temporal_notes": "Emerged with the first gods",
        }

        _concept._judge_concept_quality(svc, concept, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Emergence Era: Dawn Era" in prompt
        assert "Temporal Notes: Emerged with the first gods" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_location_judge_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Location judge prompt must contain Destruction Year, Founding Era, Temporal Notes."""
        mock_gen.return_value = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        location = {
            "name": "Ashhold Citadel",
            "description": "A fortress of black stone",
            "significance": "Seat of power",
            "founding_year": 300,
            "destruction_year": 480,
            "founding_era": "Era of Flames",
            "temporal_notes": "Built on ancient ruins",
        }

        _location._judge_location_quality(svc, location, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Destruction Year: 480" in prompt
        assert "Founding Era: Era of Flames" in prompt
        assert "Temporal Notes: Built on ancient ruins" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_faction_judge_prompt_uses_na_for_missing_fields(
        self, mock_gen, make_mock_svc, story_state
    ):
        """Missing temporal fields should render as N/A, not empty."""
        mock_gen.return_value = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        faction = {
            "name": "Bare Guild",
            "description": "Test",
            "leader": "None",
            "goals": [],
            "values": [],
        }

        _faction._judge_faction_quality(svc, faction, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Founding Era: N/A" in prompt
        assert "Dissolution Year: N/A" in prompt
        assert "Temporal Notes: N/A" in prompt

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_item_judge_prompt_uses_na_for_missing_fields(
        self, mock_gen, make_mock_svc, story_state
    ):
        """Missing temporal fields on items should render as N/A, not empty."""
        mock_gen.return_value = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        item = {
            "name": "Plain Dagger",
            "description": "Test",
            "significance": "None",
            "properties": [],
        }

        _item._judge_item_quality(svc, item, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Creation Era: N/A" in prompt
        assert "Temporal Notes: N/A" in prompt

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_concept_judge_prompt_uses_na_for_missing_fields(
        self, mock_gen, make_mock_svc, story_state
    ):
        """Missing temporal fields on concepts should render as N/A, not empty."""
        mock_gen.return_value = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        concept = {
            "name": "Bare Concept",
            "description": "Test",
            "manifestations": "None",
        }

        _concept._judge_concept_quality(svc, concept, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Emergence Era: N/A" in prompt
        assert "Temporal Notes: N/A" in prompt

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_location_judge_prompt_uses_na_for_missing_fields(
        self, mock_gen, make_mock_svc, story_state
    ):
        """Missing temporal fields on locations should render as N/A, not empty."""
        mock_gen.return_value = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        svc = make_mock_svc()

        location = {
            "name": "Bare Ruins",
            "description": "Test",
            "significance": "None",
        }

        _location._judge_location_quality(svc, location, story_state, 0.1)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Founding Era: N/A" in prompt
        assert "Destruction Year: N/A" in prompt
        assert "Temporal Notes: N/A" in prompt


class TestRefinePromptTemporalFields:
    """Verify refinement prompts include temporal fields and calendar context (#385)."""

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_faction_refine_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Faction refine prompt must contain temporal fields and calendar context."""
        mock_gen.return_value = Faction(name="Iron Covenant", description="Improved")
        svc = make_mock_svc()

        faction = {
            "name": "Iron Covenant",
            "description": "A secretive guild",
            "leader": "Lord Vex",
            "goals": ["domination"],
            "values": ["loyalty"],
            "founding_year": 200,
            "dissolution_year": 450,
            "founding_era": "Era of Flames",
            "temporal_notes": "Founded during the Great War",
        }
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=3.0,
        )

        _faction._refine_faction(svc, faction, scores, story_state, 0.7)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Dissolution Year: 450" in prompt
        assert "Founding Era: Era of Flames" in prompt
        assert "Temporal Notes: Founded during the Great War" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_item_refine_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Item refine prompt must contain Creation Era, Temporal Notes, calendar context."""
        mock_gen.return_value = Item(name="Flame Sword", description="Improved")
        svc = make_mock_svc()

        item = {
            "name": "Flame Sword",
            "description": "A burning blade",
            "significance": "Key weapon",
            "properties": ["fire damage"],
            "creation_year": 100,
            "creation_era": "Age of Forging",
            "temporal_notes": "Forged in the first furnace",
        }
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=3.0,
        )

        _item._refine_item(svc, item, scores, story_state, 0.7)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Creation Era: Age of Forging" in prompt
        assert "Temporal Notes: Forged in the first furnace" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_concept_refine_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Concept refine prompt must contain Emergence Era, Temporal Notes, calendar context."""
        mock_gen.return_value = Concept(name="The Binding", description="Improved")
        svc = make_mock_svc()

        concept = {
            "name": "The Binding",
            "description": "A cosmic force",
            "manifestations": "Appears as golden chains",
            "emergence_year": 50,
            "emergence_era": "Dawn Era",
            "temporal_notes": "Emerged with the first gods",
        }
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=3.0,
        )

        _concept._refine_concept(svc, concept, scores, story_state, 0.7)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Emergence Era: Dawn Era" in prompt
        assert "Temporal Notes: Emerged with the first gods" in prompt
        assert "CALENDAR & TIMELINE" in prompt

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_location_refine_prompt_has_temporal_fields(self, mock_gen, make_mock_svc, story_state):
        """Location refine prompt must contain temporal fields and calendar context."""
        mock_gen.return_value = Location(name="Ashhold Citadel", description="Improved")
        svc = make_mock_svc()

        location = {
            "name": "Ashhold Citadel",
            "description": "A fortress of black stone",
            "significance": "Seat of power",
            "founding_year": 300,
            "destruction_year": 480,
            "founding_era": "Era of Flames",
            "temporal_notes": "Built on ancient ruins",
        }
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=3.0,
        )

        _location._refine_location(svc, location, scores, story_state, 0.7)

        prompt = mock_gen.call_args.kwargs["prompt"]
        assert "Destruction Year: 480" in prompt
        assert "Founding Era: Era of Flames" in prompt
        assert "Temporal Notes: Built on ancient ruins" in prompt
        assert "CALENDAR & TIMELINE" in prompt
