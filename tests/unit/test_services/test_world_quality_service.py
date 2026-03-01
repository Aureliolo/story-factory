"""Tests for WorldQualityService - multi-model iteration for world building quality."""

import json
import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import ollama
import pytest

from src.memory.story_state import (
    Chapter,
    Character,
    CharacterCreation,
    Concept,
    Faction,
    Item,
    Location,
    PlotOutline,
    StoryBrief,
    StoryState,
)
from src.memory.world_database import Entity
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RelationshipQualityScores,
)
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError
from tests.shared.mock_ollama import MockStreamChunk


@pytest.fixture
def settings():
    """Create settings with test values including world-health defaults."""
    return Settings(
        ollama_url="http://localhost:11434",
        ollama_timeout=60,
        world_quality_enabled=True,
        world_quality_max_iterations=3,
        world_quality_threshold=7.0,
        world_quality_thresholds={
            "character": 7.0,
            "location": 7.0,
            "faction": 7.0,
            "item": 7.0,
            "concept": 7.0,
            "event": 7.0,
            "relationship": 7.0,
            "plot": 7.0,
            "chapter": 7.0,
        },
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        mini_description_words_max=15,
        # World-health settings
        relationship_minimums={"character": {"default": 2}, "location": {"default": 1}},
        fuzzy_match_threshold=0.8,
        max_relationships_per_entity=10,
    )


@pytest.fixture
def mock_mode_service():
    """Create mock mode service."""
    mode_service = MagicMock()
    mode_service.get_model_for_agent.return_value = "test-model"
    # prepare_model() reads vram_strategy from settings — provide a valid value
    # so that _make_model_preparers callbacks don't crash when anti-self-judging
    # swaps the judge model (making creator != judge).
    mode_service.settings.vram_strategy = "sequential"
    return mode_service


@pytest.fixture
def service(settings, mock_mode_service):
    """Create WorldQualityService with mocked dependencies."""
    svc = WorldQualityService(settings, mock_mode_service)
    # Mock analytics_db to prevent tests from writing to real database
    svc._analytics_db = MagicMock()
    return svc


@pytest.fixture
def story_state():
    """Create story state with brief for testing."""
    state = StoryState(id="test-story-id")
    state.brief = StoryBrief(
        premise="A detective solves mysteries in a haunted mansion",
        genre="mystery",
        subgenres=["gothic", "horror"],
        tone="dark and atmospheric",
        themes=["truth", "fear", "redemption"],
        setting_time="Victorian era",
        setting_place="English countryside",
        target_length="novella",
        language="English",
        content_rating="mild",
    )
    return state


@pytest.fixture
def mock_ollama_client():
    """Create a mock ollama client."""
    return MagicMock()


class TestFormatProperties:
    """Tests for _format_properties static method."""

    @pytest.mark.parametrize(
        "input_props, expected",
        [
            # String list
            pytest.param(
                ["sharp blade", "magical", "ancient"],
                "sharp blade, magical, ancient",
                id="string_list",
            ),
            # Dict with name
            pytest.param(
                [{"name": "glowing"}, {"name": "heavy"}],
                "glowing, heavy",
                id="dict_with_name",
            ),
            # Dict with description
            pytest.param(
                [{"description": "emits light"}, {"description": "very heavy"}],
                "emits light, very heavy",
                id="dict_with_description",
            ),
            # Mixed string and dict
            pytest.param(
                ["magical", {"name": "ancient"}, {"description": "glowing"}],
                "magical, ancient, glowing",
                id="mixed_types",
            ),
            # Empty list
            pytest.param([], "", id="empty_list"),
            # Other types (int, bool, None)
            pytest.param([123, True, None], "123, True, None", id="other_types"),
            # None input
            pytest.param(None, "", id="none_input"),
            # Single string value (non-list)
            pytest.param("single property", "single property", id="single_value"),
            # Empty string name (key exists)
            pytest.param([{"name": "", "description": "fallback"}], "", id="empty_string_name"),
            # None name/description values - filtered out to avoid empty strings
            pytest.param([{"name": None}, {"description": None}], "", id="none_name"),
            # Non-string dict values
            pytest.param(
                [{"name": 123}, {"description": True}], "123, True", id="non_string_values"
            ),
        ],
    )
    def test_format_properties(self, service, input_props, expected):
        """Test _format_properties with various input types."""
        result = service._format_properties(input_props)
        assert result == expected

    def test_format_dict_without_name_or_description(self, service):
        """Should fall back to str(dict) when neither name nor description present."""
        props = [{"other_key": "value"}, {"foo": "bar"}]
        result = service._format_properties(props)
        # Falls back to str(dict) representation, need to check contains
        assert "other_key" in result
        assert "foo" in result


class TestCharacterQualityScores:
    """Tests for CharacterQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=7.0,
            flaws=6.0,
            uniqueness=9.0,
            arc_potential=5.0,
            temporal_plausibility=7.0,
        )
        assert scores.average == 7.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=7.0,
            flaws=6.0,
            uniqueness=9.0,
            arc_potential=5.0,
            temporal_plausibility=7.0,
            feedback="Good character, needs more flaws",
        )
        result = scores.to_dict()
        assert result["depth"] == 8.0
        assert result["goal_clarity"] == 7.0
        assert result["temporal_plausibility"] == 7.0
        assert result["average"] == 7.0
        assert result["feedback"] == "Good character, needs more flaws"

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=6.5,
            flaws=5.0,
            uniqueness=9.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "goal_clarity" in weak
        assert "flaws" in weak
        assert "arc_potential" in weak
        assert "temporal_plausibility" in weak
        assert "depth" not in weak
        assert "uniqueness" not in weak

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        # Creating without required fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CharacterQualityScores()  # type: ignore[call-arg]

        # Should have errors for all 6 score fields
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {
            "depth",
            "goal_clarity",
            "flaws",
            "uniqueness",
            "arc_potential",
            "temporal_plausibility",
        }


class TestLocationQualityScores:
    """Tests for LocationQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.0,
            story_relevance=6.0,
            distinctiveness=9.0,
            temporal_plausibility=7.5,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=5.0,
            story_relevance=6.0,
            distinctiveness=9.0,
            temporal_plausibility=6.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "narrative_significance" in weak
        assert "story_relevance" in weak
        assert "temporal_plausibility" in weak
        assert "atmosphere" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.0,
            story_relevance=6.0,
            distinctiveness=9.0,
            temporal_plausibility=7.5,
            feedback="Add more sensory details",
        )
        result = scores.to_dict()
        assert result["atmosphere"] == 8.0
        assert result["temporal_plausibility"] == 7.5
        assert result["average"] == 7.5
        assert result["feedback"] == "Add more sensory details"

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            LocationQualityScores()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {
            "atmosphere",
            "narrative_significance",
            "story_relevance",
            "distinctiveness",
            "temporal_plausibility",
        }


class TestRelationshipQualityScores:
    """Tests for RelationshipQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = RelationshipQualityScores(
            tension=8.0,
            dynamics=7.0,
            story_potential=6.0,
            authenticity=9.0,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = RelationshipQualityScores(
            tension=6.0,
            dynamics=7.0,
            story_potential=5.0,
            authenticity=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "tension" in weak
        assert "story_potential" in weak
        assert "dynamics" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = RelationshipQualityScores(
            tension=8.0,
            dynamics=7.0,
            story_potential=6.0,
            authenticity=9.0,
            feedback="More conflict needed",
        )
        result = scores.to_dict()
        assert result["tension"] == 8.0
        assert result["average"] == 7.5
        assert result["feedback"] == "More conflict needed"


class TestFactionQualityScores:
    """Tests for FactionQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=7.0,
            conflict_potential=6.0,
            distinctiveness=9.0,
            temporal_plausibility=7.5,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = FactionQualityScores(
            coherence=6.0,
            influence=5.0,
            conflict_potential=8.0,
            distinctiveness=9.0,
            temporal_plausibility=6.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "coherence" in weak
        assert "influence" in weak
        assert "temporal_plausibility" in weak
        assert "conflict_potential" not in weak
        assert "distinctiveness" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=7.0,
            conflict_potential=6.0,
            distinctiveness=9.0,
            temporal_plausibility=7.5,
            feedback="More internal structure needed",
        )
        result = scores.to_dict()
        assert result["coherence"] == 8.0
        assert result["temporal_plausibility"] == 7.5
        assert result["average"] == 7.5
        assert result["feedback"] == "More internal structure needed"

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            FactionQualityScores()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {
            "coherence",
            "influence",
            "conflict_potential",
            "distinctiveness",
            "temporal_plausibility",
        }


class TestItemQualityScores:
    """Tests for ItemQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=9.0,
            temporal_plausibility=7.5,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=5.0,
            narrative_potential=8.0,
            integration=9.0,
            temporal_plausibility=6.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "story_significance" in weak
        assert "uniqueness" in weak
        assert "temporal_plausibility" in weak
        assert "narrative_potential" not in weak
        assert "integration" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=9.0,
            temporal_plausibility=7.5,
            feedback="More history needed",
        )
        result = scores.to_dict()
        assert result["story_significance"] == 8.0
        assert result["temporal_plausibility"] == 7.5
        assert result["average"] == 7.5
        assert result["feedback"] == "More history needed"

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ItemQualityScores()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {
            "story_significance",
            "uniqueness",
            "narrative_potential",
            "integration",
            "temporal_plausibility",
        }


class TestConceptQualityScores:
    """Tests for ConceptQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=7.0,
            manifestation=6.0,
            resonance=9.0,
            temporal_plausibility=7.5,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=5.0,
            manifestation=8.0,
            resonance=9.0,
            temporal_plausibility=6.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "relevance" in weak
        assert "depth" in weak
        assert "temporal_plausibility" in weak
        assert "manifestation" not in weak
        assert "resonance" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=7.0,
            manifestation=6.0,
            resonance=9.0,
            temporal_plausibility=7.5,
            feedback="More philosophical depth",
        )
        result = scores.to_dict()
        assert result["relevance"] == 8.0
        assert result["temporal_plausibility"] == 7.5
        assert result["average"] == 7.5
        assert result["feedback"] == "More philosophical depth"

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ConceptQualityScores()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {
            "relevance",
            "depth",
            "manifestation",
            "resonance",
            "temporal_plausibility",
        }


class TestRefinementConfig:
    """Tests for RefinementConfig model."""

    def test_from_settings(self, settings):
        """Test creating config from src.settings."""
        config = RefinementConfig.from_settings(settings)
        assert config.max_iterations == 3
        assert config.quality_threshold == 7.0
        assert config.creator_temperature == 0.9
        assert config.judge_temperature == 0.1
        assert config.refinement_temperature == 0.7

    def test_default_values(self):
        """Test default config values."""
        config = RefinementConfig()
        assert config.max_iterations == 3
        assert config.quality_threshold == 7.5

    def test_validation(self):
        """Test validation constraints."""
        with pytest.raises(ValueError):
            RefinementConfig(max_iterations=0)

        with pytest.raises(ValueError):
            RefinementConfig(quality_threshold=11.0)


class TestWorldQualityServiceInit:
    """Tests for WorldQualityService initialization and properties."""

    def test_get_config(self, service, settings):
        """Test getting refinement config."""
        config = service.get_config()
        assert config.max_iterations == settings.world_quality_max_iterations
        assert config.quality_threshold == settings.world_quality_threshold

    def test_client_creation(self, service, settings):
        """Test lazy client creation."""
        assert service._client is None
        with patch("ollama.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()
            client = service.client
            assert client is not None
            # Second access returns same client
            assert service.client is client

    def test_client_uses_scaled_timeout(self, settings, mock_mode_service):
        """Test that client uses scaled timeout based on writer model size."""
        # Create fresh service without cached client
        svc = WorldQualityService(settings, mock_mode_service)
        assert svc._client is None

        with (
            patch("ollama.Client") as mock_client_class,
            patch.object(settings, "get_model_for_agent", return_value="test-writer:40b"),
            patch.object(settings, "get_scaled_timeout", return_value=360.0),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = svc.client

            # Verify get_model_for_agent was called with "writer"
            settings.get_model_for_agent.assert_called_once_with("writer")
            # Verify get_scaled_timeout was called with the writer model
            settings.get_scaled_timeout.assert_called_once_with("test-writer:40b")
            # Verify Client was created with scaled timeout
            mock_client_class.assert_called_once_with(
                host=settings.ollama_url,
                timeout=360.0,
            )
            assert client is mock_client

    def test_analytics_db_creation(self, settings, mock_mode_service):
        """Test lazy analytics database creation."""
        # Create service without the fixture's analytics_db mock
        svc = WorldQualityService(settings, mock_mode_service)
        assert svc._analytics_db is None
        with patch("src.services.world_quality_service.ModeDatabase") as mock_db_class:
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db
            db = svc.analytics_db
            assert db is mock_db
            # Second access returns same instance
            assert svc.analytics_db is mock_db

    def test_get_creator_model(self, service, mock_mode_service):
        """Test getting creator model."""
        model = service._get_creator_model()
        mock_mode_service.get_model_for_agent.assert_called_with("writer")
        assert model == "test-model"

    def test_get_judge_model(self, service, mock_mode_service):
        """Test getting judge model."""
        model = service._get_judge_model()
        mock_mode_service.get_model_for_agent.assert_called_with("judge")
        assert model == "test-model"


class TestJudgeConfigCaching:
    """Tests for JudgeConsistencyConfig caching in WorldQualityService."""

    def test_judge_config_cached_on_second_call(self, service):
        """get_judge_config() returns the same cached instance on repeated calls."""
        config1 = service.get_judge_config()
        config2 = service.get_judge_config()
        assert config1 is config2

    def test_judge_config_cleared_on_invalidate(self, service):
        """invalidate_model_cache() clears the cached JudgeConsistencyConfig."""
        config1 = service.get_judge_config()
        assert service._judge_config is not None

        service.invalidate_model_cache()
        assert service._judge_config is None

        # Next call creates a fresh config
        config2 = service.get_judge_config()
        assert config2 is not config1

    def test_invalidate_before_any_get_judge_config(self, service):
        """invalidate_model_cache() is a no-op when _judge_config was never populated."""
        assert service._judge_config is None  # initial state
        service.invalidate_model_cache()  # must not raise
        assert service._judge_config is None

        # Subsequent get_judge_config still creates a fresh config
        config = service.get_judge_config()
        assert config is not None

    def test_invalidate_also_clears_client(self, service):
        """invalidate_model_cache() also clears the cached Ollama client."""
        # Populate the client cache
        _ = service.client
        assert service._client is not None

        service.invalidate_model_cache()
        assert service._client is None

    def test_client_thread_safe_single_instance(self, service):
        """Concurrent client property access should all observe the same cached instance."""
        import threading

        barrier = threading.Barrier(8)
        ids: list[int] = []
        ids_lock = threading.Lock()

        def worker() -> None:
            """Access client from a thread and record its id."""
            barrier.wait()
            c = service.client
            with ids_lock:
                ids.append(id(c))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(ids)) == 1


class TestRefinementConfigCaching:
    """Tests for RefinementConfig caching in WorldQualityService."""

    def test_refinement_config_cached_on_second_call(self, service):
        """get_config() returns the same cached instance on repeated calls."""
        config1 = service.get_config()
        config2 = service.get_config()
        assert config1 is config2

    def test_refinement_config_cleared_on_invalidate(self, service):
        """invalidate_model_cache() clears the cached RefinementConfig."""
        config1 = service.get_config()
        assert service._refinement_config is not None

        service.invalidate_model_cache()
        assert service._refinement_config is None

        # Next call creates a fresh config
        config2 = service.get_config()
        assert config2 is not config1

    def test_invalidate_before_any_get_config(self, service):
        """invalidate_model_cache() is a no-op when _refinement_config was never populated."""
        assert service._refinement_config is None  # initial state
        service.invalidate_model_cache()  # must not raise
        assert service._refinement_config is None

        # Subsequent get_config still creates a fresh config
        config = service.get_config()
        assert config is not None

    def test_refinement_config_thread_safe_single_instance(self, service):
        """Concurrent get_config() calls should all observe the same cached instance."""
        import threading

        barrier = threading.Barrier(8)
        ids: list[int] = []
        ids_lock = threading.Lock()

        def worker() -> None:
            """Access config from a thread and record its id."""
            barrier.wait()
            cfg = service.get_config()
            with ids_lock:
                ids.append(id(cfg))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(ids)) == 1


class TestRecordEntityQuality:
    """Tests for record_entity_quality method."""

    def test_record_entity_quality_success(self, service, mock_mode_service):
        """Test successful recording of entity quality."""
        mock_analytics_db = MagicMock()
        service._analytics_db = mock_analytics_db
        mock_mode_service.get_model_for_agent.return_value = "creator-model"

        scores = {
            "depth": 8.0,
            "average": 7.5,
            "feedback": "Good character",
        }

        service.record_entity_quality(
            project_id="test-project",
            entity_type="character",
            entity_name="John Doe",
            scores=scores,
            iterations=2,
            generation_time=5.5,
        )

        mock_analytics_db.record_world_entity_score.assert_called_once_with(
            project_id="test-project",
            entity_type="character",
            entity_name="John Doe",
            model_id="creator-model",
            scores=scores,
            iterations_used=2,
            generation_time_seconds=5.5,
            feedback="Good character",
            early_stop_triggered=False,
            threshold_met=False,
            peak_score=None,
            final_score=None,
            score_progression=None,
            consecutive_degradations=0,
            best_iteration=0,
            quality_threshold=None,
            max_iterations=None,
            below_threshold_admitted=False,
        )

    def test_record_entity_quality_handles_error(self, service, mock_mode_service):
        """Test that recording failures are logged but don't raise."""
        mock_analytics_db = MagicMock()
        mock_analytics_db.record_world_entity_score.side_effect = Exception("DB error")
        service._analytics_db = mock_analytics_db

        # Should not raise, just log warning
        service.record_entity_quality(
            project_id="test-project",
            entity_type="character",
            entity_name="John Doe",
            scores={"average": 7.0},
            iterations=1,
            generation_time=3.0,
        )

    def test_record_entity_quality_validates_inputs(self, service):
        """Test validation of required inputs."""
        with pytest.raises(ValueError, match="project_id"):
            service.record_entity_quality(
                project_id="",
                entity_type="character",
                entity_name="John",
                scores={},
                iterations=1,
                generation_time=1.0,
            )

        with pytest.raises(ValueError, match="entity_type"):
            service.record_entity_quality(
                project_id="test",
                entity_type="",
                entity_name="John",
                scores={},
                iterations=1,
                generation_time=1.0,
            )

        with pytest.raises(ValueError, match="entity_name"):
            service.record_entity_quality(
                project_id="test",
                entity_type="character",
                entity_name="",
                scores={},
                iterations=1,
                generation_time=1.0,
            )


class TestCreateCharacter:
    """Tests for _create_character method."""

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_success(self, mock_generate_structured, service, story_state):
        """Test successful character creation."""
        mock_creation = CharacterCreation(
            name="Dr. Eleanor Grey",
            role="protagonist",
            description="A brilliant detective haunted by past failures",
            personality_traits=["observant", "determined", "secretive"],
            goals=["solve the case", "find redemption"],
            arc_notes="Will learn to trust others",
        )
        mock_generate_structured.return_value = mock_creation

        character = service._create_character(story_state, existing_names=[], temperature=0.9)

        assert character is not None
        assert character.name == "Dr. Eleanor Grey"
        assert character.role == "protagonist"
        assert "observant" in character.trait_names
        assert character.arc_progress == {}
        assert character.arc_type is None

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_with_existing_names(
        self, mock_generate_structured, service, story_state
    ):
        """Test character creation avoids existing names."""
        mock_creation = CharacterCreation(
            name="New Character",
            role="supporting",
            description="A mysterious stranger",
            personality_traits=["quiet"],
            goals=["unknown"],
            arc_notes="Will reveal secrets",
        )
        mock_generate_structured.return_value = mock_creation

        character = service._create_character(
            story_state, existing_names=["John Doe", "Jane Doe"], temperature=0.9
        )

        assert character is not None
        # Verify prompt includes existing names
        call_args = mock_generate_structured.call_args
        assert "John Doe" in call_args.kwargs["prompt"]
        assert "Jane Doe" in call_args.kwargs["prompt"]

    def test_create_character_no_brief_returns_none(self, service):
        """Test character creation returns None without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_character(state, existing_names=[], temperature=0.9)
        assert result is None

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_validation_error_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test character creation raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Character", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_ollama_error(self, mock_generate_structured, service, story_state):
        """Test character creation handles Ollama errors."""
        mock_generate_structured.side_effect = ollama.ResponseError("Model not found")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_connection_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test character creation handles connection errors."""
        mock_generate_structured.side_effect = ConnectionError("Connection refused")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_timeout_error(self, mock_generate_structured, service, story_state):
        """Test character creation handles timeout errors."""
        mock_generate_structured.side_effect = TimeoutError("Request timed out")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_unexpected_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test character creation handles unexpected errors."""
        mock_generate_structured.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)


class TestJudgeCharacterQuality:
    """Tests for _judge_character_quality method."""

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_judge_character_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful character quality judgment."""
        mock_generate_structured.return_value = CharacterQualityScores(
            depth=8.0,
            goals=7.5,
            flaws=7.0,
            uniqueness=8.5,
            arc_potential=8.0,
            temporal_plausibility=7.8,
            feedback="Strong character with good depth",
        )

        character = Character(
            name="Test Character",
            role="protagonist",
            description="A test character",
            personality_traits=["brave"],
            goals=["win"],
            arc_notes="Will grow",
        )

        scores = service._judge_character_quality(character, story_state, temperature=0.1)

        assert scores.depth == 8.0
        assert scores.goals == 7.5
        assert scores.feedback == "Strong character with good depth"
        assert scores.average == pytest.approx(7.8, rel=0.01)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_judge_character_quality_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on validation failure."""
        mock_generate_structured.side_effect = Exception("Validation failed")

        character = Character(name="Test", role="supporting", description="Test")

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_character_quality(character, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_judge_character_quality_no_brief_uses_default_genre(
        self, mock_generate_structured, service
    ):
        """Test judge uses default genre when brief is missing."""
        mock_generate_structured.return_value = CharacterQualityScores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=7.0,
            arc_potential=7.0,
            temporal_plausibility=7.0,
            feedback="Decent character",
        )

        state = StoryState(id="test-id")
        state.brief = None
        character = Character(name="Test", role="supporting", description="Test")

        scores = service._judge_character_quality(character, state, temperature=0.1)

        assert scores.average == 7.0
        # Verify prompt uses "fiction" as default genre
        call_args = mock_generate_structured.call_args
        assert "fiction" in call_args.kwargs["prompt"]

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_judge_character_quality_exception_reraises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge reraises exceptions as WorldGenerationError."""
        mock_generate_structured.side_effect = RuntimeError("Some error")

        character = Character(name="Test", role="supporting", description="Test")

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_character_quality(character, story_state, temperature=0.1)


class TestRefineCharacter:
    """Tests for _refine_character method."""

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_success(self, mock_generate_structured, service, story_state):
        """Test successful character refinement."""
        mock_generate_structured.return_value = CharacterCreation(
            name="John Doe",
            role="protagonist",
            description="A more complex description with deeper psychology",
            personality_traits=["brave", "conflicted", "hopeful"],
            goals=["save the world", "overcome inner demons"],
            arc_notes="Will transform from bitter loner to trusting friend",
        )

        original_char = Character(
            name="John Doe",
            role="protagonist",
            description="A simple description",
            personality_traits=["brave"],
            goals=["save the world"],
            relationships={"Mary": "ally"},
            arc_notes="Basic arc",
            arc_progress={1: "Ordinary world", 2: "Call to adventure"},
            arc_type="hero_journey",
        )
        scores = CharacterQualityScores(
            depth=5.0,
            goals=6.0,
            flaws=4.0,
            uniqueness=5.5,
            arc_potential=5.0,
            temporal_plausibility=5.0,
            feedback="Needs more depth",
        )

        refined = service._refine_character(original_char, scores, story_state, temperature=0.7)

        assert refined.name == "John Doe"
        assert "deeper psychology" in refined.description
        assert len(refined.personality_traits) > 1
        # arc_progress, arc_type, and relationships should be preserved from original
        assert refined.relationships == {"Mary": "ally"}
        assert refined.arc_progress == {1: "Ordinary world", 2: "Call to adventure"}
        assert refined.arc_type == "hero_journey"

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_includes_weak_dimensions_in_prompt(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement prompt includes weak dimensions."""
        mock_generate_structured.return_value = CharacterCreation(
            name="Test",
            role="supporting",
            description="Refined",
            personality_traits=[],
            goals=[],
            arc_notes="",
        )

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=5.0,  # Below 7.0 threshold
            goals=8.0,
            flaws=4.0,  # Below 7.0 threshold
            uniqueness=9.0,
            arc_potential=5.0,  # Below 7.0 threshold
            temporal_plausibility=8.0,
        )

        service._refine_character(original_char, scores, story_state, temperature=0.7)

        call_args = mock_generate_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "depth" in prompt.lower()
        assert "flaws" in prompt.lower()
        assert "arc potential" in prompt.lower()

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_error_raises(self, mock_generate_structured, service, story_state):
        """Test refinement raises error on failure."""
        mock_generate_structured.side_effect = Exception("Generation failed")

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="refinement failed"):
            service._refine_character(original_char, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_with_no_brief_uses_english(self, mock_generate_structured, service):
        """Test refinement uses English when brief is missing."""
        mock_generate_structured.return_value = CharacterCreation(
            name="Test",
            role="supporting",
            description="Refined",
            personality_traits=[],
            goals=[],
            arc_notes="",
        )

        state = StoryState(id="test-id")
        state.brief = None
        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )

        service._refine_character(original_char, scores, state, temperature=0.7)

        call_args = mock_generate_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "English" in prompt

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_uses_character_creation_model(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement uses CharacterCreation to exclude arc_progress from schema (#305)."""
        mock_generate_structured.return_value = CharacterCreation(
            name="Test",
            role="supporting",
            description="Refined",
        )

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )

        service._refine_character(original_char, scores, story_state, temperature=0.7)

        call_args = mock_generate_structured.call_args
        assert call_args.kwargs["response_model"] is CharacterCreation

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_uses_character_creation_model(
        self, mock_generate_structured, service, story_state
    ):
        """Test creation uses CharacterCreation to exclude arc_progress from schema (#305)."""
        mock_generate_structured.return_value = CharacterCreation(
            name="Test",
            role="supporting",
            description="Created",
        )

        service._create_character(story_state, existing_names=[], temperature=0.9)

        call_args = mock_generate_structured.call_args
        assert call_args.kwargs["response_model"] is CharacterCreation


class TestGenerateCharacterWithQuality:
    """Tests for generate_character_with_quality method."""

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    def test_generate_character_meets_threshold_first_try(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test character generation that meets quality threshold on first try."""
        test_char = Character(
            name="Dr. Eleanor Grey",
            role="protagonist",
            description="A brilliant detective",
            personality_traits=["observant", "haunted", "determined"],
            goals=["solve the case", "confront her past"],
            arc_notes="Will learn to trust others",
        )
        mock_create.return_value = test_char

        high_scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=7.5,
            uniqueness=8.0,
            arc_potential=8.5,
            temporal_plausibility=8.0,
        )
        mock_judge.return_value = high_scores

        char, scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        assert char.name == "Dr. Eleanor Grey"
        assert scores.average >= 7.0
        assert iterations == 1
        mock_create.assert_called_once()
        mock_judge.assert_called_once()

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_generate_character_needs_refinement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test character generation that needs refinement."""
        initial_char = Character(
            name="John Doe",
            role="supporting",
            description="A mysterious stranger",
            personality_traits=["quiet"],
            goals=["unknown"],
            arc_notes="May change",
        )
        mock_create.return_value = initial_char

        refined_char = Character(
            name="John Doe",
            role="supporting",
            description="A mysterious stranger with a dark past",
            personality_traits=["quiet", "observant", "haunted"],
            goals=["protect the mansion's secrets", "atone for past mistakes"],
            arc_notes="Will reveal his true allegiance",
        )
        mock_refine.return_value = refined_char

        # First judgment returns low scores, second returns high scores
        low_scores = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=4.0,
            uniqueness=6.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
        )
        high_scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=7.5,
            uniqueness=8.0,
            arc_potential=8.5,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [low_scores, high_scores]

        char, scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        assert char.name == "John Doe"
        assert scores.average >= 7.0
        assert iterations == 2
        mock_create.assert_called_once()
        mock_refine.assert_called_once()
        assert mock_judge.call_count == 2

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_generate_character_returns_below_threshold_after_max_iterations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test character returns below threshold if max iterations exceeded.

        With best-iteration tracking, returns iteration 1 when all scores are equal
        (no improvement detected).
        """
        test_char = Character(name="Low Quality", role="supporting", description="Basic")
        mock_create.return_value = test_char
        mock_refine.return_value = test_char  # Mock refinement to prevent errors

        low_scores = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.return_value = low_scores

        char, scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        assert char.name == "Low Quality"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1

    @patch.object(WorldQualityService, "_create_character")
    def test_generate_character_raises_error_when_creation_fails(
        self, mock_create, service, story_state
    ):
        """Test that error is raised when character creation fails repeatedly."""
        mock_create.return_value = None

        with pytest.raises(WorldGenerationError, match="Failed to generate character"):
            service.generate_character_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    def test_generate_character_handles_judge_error(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test character generation handles judge errors gracefully."""
        test_char = Character(name="Test", role="supporting", description="Test")
        mock_create.return_value = test_char
        mock_judge.side_effect = WorldGenerationError("Judge failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate character"):
            service.generate_character_with_quality(story_state, existing_names=[])

    def test_generate_character_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_character_with_quality(state, existing_names=[])


class TestCreateLocation:
    """Tests for _create_location method."""

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_success(self, mock_generate_structured, service, story_state):
        """Test successful location creation."""
        mock_location = Location(
            name="Thornwood Manor",
            type="location",
            description="A crumbling Victorian mansion shrouded in mist",
            significance="Central setting where the mystery unfolds",
        )
        mock_generate_structured.return_value = mock_location

        location = service._create_location(story_state, existing_names=[], temperature=0.9)

        assert location["name"] == "Thornwood Manor"
        assert location["type"] == "location"
        assert "Victorian" in location["description"]

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_duplicate_name_returns_empty(
        self, mock_generate_structured, service, story_state
    ):
        """Test location creation returns empty dict when name is duplicate."""
        mock_location = Location(
            name="Thornwood Manor",
            type="location",
            description="A crumbling Victorian mansion",
            significance="Central setting",
        )
        mock_generate_structured.return_value = mock_location

        # Pass existing name - should return empty to force retry
        result = service._create_location(
            story_state, existing_names=["Thornwood Manor"], temperature=0.9
        )
        assert result == {}

    def test_create_location_no_brief_returns_empty(self, service):
        """Test location creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_location(state, existing_names=[], temperature=0.9)
        assert result == {}

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test location creation raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Location", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Location creation failed"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_ollama_error(self, mock_generate_structured, service, story_state):
        """Test location creation handles Ollama errors."""
        mock_generate_structured.side_effect = ollama.ResponseError("Error")

        with pytest.raises(WorldGenerationError, match="Location creation failed"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_non_dict_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test location creation raises error when result cannot be converted."""
        # Simulate a validation error (e.g., LLM returned invalid structure)
        mock_generate_structured.side_effect = ValueError("Invalid location data")

        with pytest.raises(WorldGenerationError, match="Location creation failed"):
            service._create_location(story_state, existing_names=[], temperature=0.9)


class TestJudgeLocationQuality:
    """Tests for _judge_location_quality method."""

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_judge_location_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful location quality judgment."""
        mock_generate_structured.return_value = LocationQualityScores(
            atmosphere=8.0,
            significance=7.5,
            story_relevance=8.0,
            distinctiveness=8.5,
            temporal_plausibility=8.0,
            feedback="Rich atmosphere, could be more distinctive",
        )

        location = {
            "name": "Dark Forest",
            "description": "A mysterious forest",
            "significance": "Important for plot",
        }

        scores = service._judge_location_quality(location, story_state, temperature=0.1)

        assert scores.atmosphere == 8.0
        assert scores.average == 8.0

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_judge_location_quality_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on failure."""
        mock_generate_structured.side_effect = Exception("Generation failed")

        location = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_location_quality(location, story_state, temperature=0.1)


class TestRefineLocation:
    """Tests for _refine_location method."""

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_success(self, mock_generate_structured, service, story_state):
        """Test successful location refinement."""
        mock_location = Location(
            name="Dark Forest",
            type="location",
            description="A deeply atmospheric ancient forest",
            significance="Central to the story's themes",
        )
        mock_generate_structured.return_value = mock_location

        original = {"name": "Dark Forest", "description": "A forest", "significance": "Unknown"}
        scores = LocationQualityScores(
            atmosphere=5.0,
            significance=5.0,
            story_relevance=6.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_location(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Dark Forest"
        assert "atmospheric" in refined["description"]

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_preserves_temporal_fields(
        self, mock_generate_structured, service, story_state
    ):
        """Test that refinement preserves temporal fields from original when LLM omits them."""
        mock_location = Location(
            name="Dark Forest",
            type="location",
            description="An improved forest",
            significance="Very important",
        )
        mock_generate_structured.return_value = mock_location

        original = {
            "name": "Dark Forest",
            "description": "A forest",
            "significance": "Unknown",
            "founding_year": 500,
            "founding_era": "Golden Age",
            "temporal_notes": "Ancient place",
        }
        scores = LocationQualityScores(
            atmosphere=5.0,
            significance=5.0,
            story_relevance=6.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_location(original, scores, story_state, temperature=0.7)

        assert refined["founding_year"] == 500
        assert refined["founding_era"] == "Golden Age"
        assert refined["temporal_notes"] == "Ancient place"

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Location", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Location refinement failed"):
            service._refine_location(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_ollama_error(self, mock_generate_structured, service, story_state):
        """Test refinement handles Ollama errors."""
        mock_generate_structured.side_effect = ConnectionError("Connection refused")

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Location refinement failed"):
            service._refine_location(original, scores, story_state, temperature=0.7)


class TestGenerateLocationWithQuality:
    """Tests for generate_location_with_quality method."""

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    def test_generate_location_meets_threshold(self, mock_judge, mock_create, service, story_state):
        """Test location generation with quality."""
        test_loc = {
            "name": "Thornwood Manor",
            "type": "location",
            "description": "A crumbling Victorian mansion shrouded in mist and shadow at the edge of town",
            "significance": "Central setting where the mystery unfolds",
        }
        mock_create.return_value = test_loc

        high_scores = LocationQualityScores(
            atmosphere=9.0,
            significance=8.0,
            story_relevance=8.5,
            distinctiveness=8.0,
            temporal_plausibility=8.5,
        )
        mock_judge.return_value = high_scores

        loc, scores, iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert loc["name"] == "Thornwood Manor"
        assert scores.average >= 7.0
        assert iterations == 1

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    @patch.object(WorldQualityService, "_refine_location")
    def test_generate_location_needs_refinement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test location generation that needs refinement."""
        initial_loc = {
            "name": "Basic Place",
            "description": "A vast and featureless plain stretching endlessly toward the horizon",
            "significance": "None",
        }
        mock_create.return_value = initial_loc

        refined_loc = {
            "name": "Basic Place",
            "description": "A richly detailed location with hidden depths and atmospheric wonder",
            "significance": "Important",
        }
        mock_refine.return_value = refined_loc

        low_scores = LocationQualityScores(
            atmosphere=5.0,
            significance=5.0,
            story_relevance=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        high_scores = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [low_scores, high_scores]

        loc, scores, iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert loc["name"] == "Basic Place"
        assert scores.average >= 7.0
        assert iterations == 2

    @patch.object(WorldQualityService, "_create_location")
    def test_generate_location_empty_creation_fails(self, mock_create, service, story_state):
        """Test that empty location creation causes failure."""
        mock_create.return_value = {}  # No name

        with pytest.raises(WorldGenerationError, match="Failed to generate location"):
            service.generate_location_with_quality(story_state, existing_names=[])

    def test_generate_location_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_location_with_quality(state, existing_names=[])

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    @patch.object(WorldQualityService, "_refine_location")
    def test_generate_location_error_during_iteration(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test location generation handles errors during iteration."""
        test_loc = {
            "name": "Test Location",
            "description": "A test location used to exercise error handling during iteration",
            "significance": "Testing errors",
        }
        mock_create.return_value = test_loc
        mock_refine.return_value = test_loc
        # First judgment succeeds with low score, second raises error, third returns low score
        # Service continues after error so needs values for all iterations
        low_scores = LocationQualityScores(
            atmosphere=5.0,
            significance=5.0,
            story_relevance=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.side_effect = [
            low_scores,
            WorldGenerationError("Judge failed"),
            low_scores,
        ]

        # Should still return location despite error on 2nd iteration
        # because we have valid results from 1st iteration
        loc, scores, _iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert loc["name"] == "Test Location"
        assert scores.average == low_scores.average


class TestIsDuplicateRelationship:
    """Tests for _is_duplicate_relationship method."""

    def test_duplicate_same_direction(self, service):
        """Test detecting duplicate in same direction."""
        existing = [("Alice", "Bob", "knows"), ("Charlie", "Diana", "loves")]
        assert service._is_duplicate_relationship("Alice", "Bob", existing)

    def test_duplicate_reverse_direction(self, service):
        """Test detecting duplicate in reverse direction."""
        existing = [("Alice", "Bob", "knows")]
        assert service._is_duplicate_relationship("Bob", "Alice", existing)

    def test_not_duplicate(self, service):
        """Test that unrelated pairs are not duplicates."""
        existing = [("Alice", "Bob", "knows")]
        assert not service._is_duplicate_relationship("Charlie", "Diana", existing)

    def test_empty_existing_list(self, service):
        """Test with empty existing list."""
        assert not service._is_duplicate_relationship("Alice", "Bob", [])


class TestCreateRelationship:
    """Tests for _create_relationship method."""

    def test_create_relationship_success(self, service, story_state, mock_ollama_client):
        """Test successful relationship creation."""
        rel_json = json.dumps(
            {
                "source": "Dr. Eleanor Grey",
                "target": "Lord Blackwood",
                "relation_type": "suspects",
                "description": "Eleanor suspects Lord Blackwood knows more than he's telling",
            }
        )
        mock_ollama_client.generate.return_value = {"response": rel_json}
        service._client = mock_ollama_client

        rel = service._create_relationship(
            story_state,
            entity_names=["Dr. Eleanor Grey", "Lord Blackwood"],
            existing_rels=[],
            temperature=0.9,
        )

        assert rel["source"] == "Dr. Eleanor Grey"
        assert rel["target"] == "Lord Blackwood"
        assert rel["relation_type"] == "suspects"

    def test_create_relationship_no_brief_returns_empty(self, service):
        """Test relationship creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_relationship(state, ["A", "B"], [], temperature=0.9)
        assert result == {}

    def test_create_relationship_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test relationship creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid relationship"):
            service._create_relationship(
                story_state, entity_names=["A", "B"], existing_rels=[], temperature=0.9
            )

    def test_create_relationship_ollama_error(self, service, story_state, mock_ollama_client):
        """Test relationship creation handles Ollama errors."""
        mock_ollama_client.generate.side_effect = ollama.ResponseError("Error")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_relationship(
                story_state, entity_names=["A", "B"], existing_rels=[], temperature=0.9
            )

    def test_create_relationship_logs_info_after_llm_call(
        self, service, story_state, mock_ollama_client, caplog
    ):
        """Relationship creation logs INFO with model, timing, and token counts (I5 fix)."""
        rel_json = json.dumps(
            {
                "source": "Dr. Eleanor Grey",
                "target": "Lord Blackwood",
                "relation_type": "suspects",
                "description": "A suspicion",
            }
        )
        mock_ollama_client.generate.return_value = {
            "response": rel_json,
            "prompt_eval_count": 120,
            "eval_count": 45,
            "total_duration": 2_500_000_000,  # 2.5 seconds in nanoseconds
        }
        service._client = mock_ollama_client

        with caplog.at_level(logging.INFO):
            service._create_relationship(
                story_state,
                entity_names=["Dr. Eleanor Grey", "Lord Blackwood"],
                existing_rels=[],
                temperature=0.9,
            )

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Relationship creation LLM call" in m for m in info_messages)
        assert any("model=" in m for m in info_messages)
        assert any("s, tokens=" in m for m in info_messages)
        assert any("tokens=120+45=165" in m for m in info_messages)


class TestJudgeRelationshipQuality:
    """Tests for _judge_relationship_quality method."""

    @patch("src.services.world_quality_service._relationship.generate_structured")
    def test_judge_relationship_quality_success(
        self, mock_generate_structured, service, story_state
    ):
        """Test successful relationship quality judgment."""
        mock_generate_structured.return_value = RelationshipQualityScores(
            tension=8.0,
            dynamics=7.5,
            story_potential=8.0,
            authenticity=8.5,
            feedback="Strong dynamic, more conflict potential",
        )

        relationship = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "rivals",
            "description": "Long-standing rivalry",
        }

        scores = service._judge_relationship_quality(relationship, story_state, temperature=0.1)

        assert scores.tension == 8.0
        assert scores.average == 8.0

    @patch("src.services.world_quality_service._relationship.generate_structured")
    def test_judge_relationship_quality_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on validation failure."""
        mock_generate_structured.side_effect = Exception("Validation failed")

        relationship = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_relationship_quality(relationship, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._relationship.generate_structured")
    def test_judge_relationship_quality_no_brief_uses_default_genre(
        self, mock_generate_structured, service
    ):
        """Test judge uses default genre when brief is missing."""
        mock_generate_structured.return_value = RelationshipQualityScores(
            tension=7.0,
            dynamics=7.0,
            story_potential=7.0,
            authenticity=7.0,
            feedback="Decent relationship",
        )

        state = StoryState(id="test-id")
        state.brief = None
        relationship = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}

        scores = service._judge_relationship_quality(relationship, state, temperature=0.1)

        assert scores.average == 7.0
        call_args = mock_generate_structured.call_args
        assert "fiction" in call_args.kwargs["prompt"]


class TestRefineRelationship:
    """Tests for _refine_relationship method."""

    def test_refine_relationship_success(self, service, story_state, mock_ollama_client):
        """Test successful relationship refinement."""
        refined_json = json.dumps(
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "rivals",
                "description": "A deep and complex rivalry with years of history",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "rivals",
            "description": "They don't like each other",
        }
        scores = RelationshipQualityScores(
            tension=5.0, dynamics=5.0, story_potential=6.0, authenticity=5.0
        )

        refined = service._refine_relationship(original, scores, story_state, temperature=0.7)

        assert refined["source"] == "Alice"
        assert "complex rivalry" in refined["description"]

    def test_refine_relationship_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        original = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}
        scores = RelationshipQualityScores(
            tension=6.0, dynamics=6.0, story_potential=6.0, authenticity=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid relationship refinement"):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

    def test_refine_relationship_logs_info_after_llm_call(
        self, service, story_state, mock_ollama_client, caplog
    ):
        """Relationship refinement logs INFO with model, timing, and token counts (I5 fix)."""
        refined_json = json.dumps(
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "rivals",
                "description": "Improved rivalry description",
            }
        )
        mock_ollama_client.generate.return_value = {
            "response": refined_json,
            "prompt_eval_count": 200,
            "eval_count": 80,
            "total_duration": 3_000_000_000,
        }
        service._client = mock_ollama_client

        original = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "rivals",
            "description": "Simple rivalry",
        }
        scores = RelationshipQualityScores(
            tension=5.0, dynamics=5.0, story_potential=5.0, authenticity=5.0
        )

        with caplog.at_level(logging.INFO):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Relationship refinement LLM call" in m for m in info_messages)
        assert any("model=" in m for m in info_messages)
        assert any("s, tokens=" in m for m in info_messages)
        assert any("tokens=200+80=280" in m for m in info_messages)


class TestGenerateRelationshipWithQuality:
    """Tests for generate_relationship_with_quality method."""

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    def test_generate_relationship_with_quality(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test relationship generation with quality."""
        test_rel = {
            "source": "Dr. Eleanor Grey",
            "target": "Lord Blackwood",
            "relation_type": "suspects",
            "description": "Eleanor suspects Lord Blackwood knows more than he's telling",
        }
        mock_create.return_value = test_rel

        high_scores = RelationshipQualityScores(
            tension=9.0, dynamics=8.0, story_potential=8.5, authenticity=8.0
        )
        mock_judge.return_value = high_scores

        rel, scores, iterations = service.generate_relationship_with_quality(
            story_state,
            entity_names=["Dr. Eleanor Grey", "Lord Blackwood"],
            existing_rels=[],
        )

        assert rel["source"] == "Dr. Eleanor Grey"
        assert rel["target"] == "Lord Blackwood"
        assert scores.average >= 7.0
        assert iterations == 1

    def test_generate_relationship_requires_two_entities(self, service, story_state):
        """Test that relationship generation requires at least 2 entities."""
        with pytest.raises(ValueError, match="Need at least 2 entities"):
            service.generate_relationship_with_quality(
                story_state,
                entity_names=["Only One Entity"],
                existing_rels=[],
            )

    @patch.object(WorldQualityService, "_create_relationship")
    def test_generate_relationship_skips_duplicates(self, mock_create, service, story_state):
        """Test that duplicate relationships are skipped."""
        # Return duplicate relationship
        mock_create.return_value = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "They know each other",
        }

        with pytest.raises(WorldGenerationError, match="Failed to generate relationship"):
            service.generate_relationship_with_quality(
                story_state,
                entity_names=["Alice", "Bob", "Charlie"],
                existing_rels=[("Alice", "Bob", "knows")],  # Already exists
            )

    @patch.object(WorldQualityService, "_create_relationship")
    def test_generate_relationship_incomplete_creation(self, mock_create, service, story_state):
        """Test that incomplete relationship creation causes retry."""
        mock_create.return_value = {"source": "Alice"}  # Missing target

        with pytest.raises(WorldGenerationError, match="Failed to generate relationship"):
            service.generate_relationship_with_quality(
                story_state,
                entity_names=["Alice", "Bob"],
                existing_rels=[],
            )

    def test_generate_relationship_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_relationship_with_quality(
                state, entity_names=["A", "B"], existing_rels=[]
            )


class TestCreateFaction:
    """Tests for _create_faction method."""

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_success(self, mock_generate_structured, service, story_state):
        """Test successful faction creation."""
        mock_faction = Faction(
            name="The Shadow Council",
            type="faction",
            description="A secret society manipulating events from the shadows",
            leader="The Grand Master",
            goals=["control the kingdom", "gather ancient artifacts"],
            values=["secrecy", "power"],
        )
        mock_generate_structured.return_value = mock_faction

        faction = service._create_faction(story_state, existing_names=[], temperature=0.9)

        assert faction["name"] == "The Shadow Council"
        assert faction["leader"] == "The Grand Master"
        assert len(faction["goals"]) == 2

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_with_existing_locations(
        self, mock_generate_structured, service, story_state
    ):
        """Test faction creation with existing locations for spatial grounding."""
        mock_faction = Faction(
            name="The Shadow Council",
            type="faction",
            description="A secret society based in the old castle",
            leader="The Grand Master",
            goals=["control the kingdom"],
            values=["secrecy"],
            base_location="The Dark Castle",
        )
        mock_generate_structured.return_value = mock_faction

        faction = service._create_faction(
            story_state,
            existing_names=[],
            temperature=0.9,
            existing_locations=["The Dark Castle", "The Royal Palace"],
        )

        assert faction["name"] == "The Shadow Council"
        assert faction["base_location"] == "The Dark Castle"

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_duplicate_name_returns_empty(
        self, mock_generate_structured, service, story_state
    ):
        """Test faction creation returns empty dict when name is duplicate."""
        mock_faction = Faction(
            name="The Shadow Council",
            type="faction",
            description="A secret society",
            leader="The Grand Master",
            goals=["control"],
            values=["power"],
        )
        mock_generate_structured.return_value = mock_faction

        # Pass existing name - should return empty to force retry
        result = service._create_faction(
            story_state, existing_names=["The Shadow Council"], temperature=0.9
        )
        assert result == {}

    def test_create_faction_no_brief_returns_empty(self, service):
        """Test faction creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_faction(state, existing_names=[], temperature=0.9)
        assert result == {}

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_error_raises(self, mock_generate_structured, service, story_state):
        """Test faction creation raises error on generation failure."""
        mock_generate_structured.side_effect = Exception("Generation failed")

        with pytest.raises(WorldGenerationError, match="Faction creation failed"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_ollama_error(self, mock_generate_structured, service, story_state):
        """Test faction creation handles Ollama errors."""
        mock_generate_structured.side_effect = TimeoutError("Timeout")

        with pytest.raises(WorldGenerationError, match="Faction creation failed"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)


class TestJudgeFactionQuality:
    """Tests for _judge_faction_quality method."""

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_judge_faction_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful faction quality judgment."""
        mock_generate_structured.return_value = FactionQualityScores(
            coherence=8.0,
            influence=7.5,
            conflict_potential=8.0,
            distinctiveness=8.5,
            temporal_plausibility=8.0,
            feedback="Strong faction with clear identity",
        )

        faction = {
            "name": "Test Guild",
            "description": "A powerful guild",
            "leader": "The Boss",
            "goals": ["dominate trade"],
            "values": ["profit"],
        }

        scores = service._judge_faction_quality(faction, story_state, temperature=0.1)

        assert scores.coherence == 8.0
        assert scores.average == 8.0

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_judge_faction_quality_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on validation failure."""
        mock_generate_structured.side_effect = Exception("Validation failed")

        faction = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_faction_quality(faction, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_judge_faction_quality_no_brief_uses_default_genre(
        self, mock_generate_structured, service
    ):
        """Test judge uses default genre when brief is missing."""
        mock_generate_structured.return_value = FactionQualityScores(
            coherence=7.0,
            influence=7.0,
            conflict_potential=7.0,
            distinctiveness=7.0,
            temporal_plausibility=7.0,
            feedback="Decent faction",
        )

        state = StoryState(id="test-id")
        state.brief = None
        faction = {"name": "Test", "description": "Test"}

        scores = service._judge_faction_quality(faction, state, temperature=0.1)

        assert scores.average == 7.0
        call_args = mock_generate_structured.call_args
        assert "fiction" in call_args.kwargs["prompt"]


class TestRefineFaction:
    """Tests for _refine_faction method."""

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_success(self, mock_generate_structured, service, story_state):
        """Test successful faction refinement."""
        mock_faction = Faction(
            name="Ignored - should use original",
            type="faction",
            description="A deeply influential guild with rich history",
            leader="The Grand Master",
            goals=["dominate trade", "expand influence"],
            values=["profit", "loyalty"],
        )
        mock_generate_structured.return_value = mock_faction

        original = {
            "name": "Test Guild",
            "description": "A guild",
            "leader": "Boss",
            "goals": ["make money"],
            "values": ["money"],
        }
        scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=6.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_faction(original, scores, story_state, temperature=0.7)

        # Name should be preserved from original
        assert refined["name"] == "Test Guild"
        assert "influential" in refined["description"]

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_preserves_temporal_fields(
        self, mock_generate_structured, service, story_state
    ):
        """Test that refinement preserves temporal fields from original when LLM omits them."""
        mock_faction = Faction(
            name="Ignored",
            type="faction",
            description="An improved guild",
            leader="Grand Master",
        )
        mock_generate_structured.return_value = mock_faction

        original = {
            "name": "Test Guild",
            "description": "A guild",
            "leader": "Boss",
            "goals": ["power"],
            "values": ["honor"],
            "founding_year": 200,
            "founding_era": "Iron Age",
            "dissolution_year": 800,
            "temporal_notes": "Rose and fell",
        }
        scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=6.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_faction(original, scores, story_state, temperature=0.7)

        assert refined["founding_year"] == 200
        assert refined["founding_era"] == "Iron Age"
        assert refined["dissolution_year"] == 800
        assert refined["temporal_notes"] == "Rose and fell"

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_error_raises(self, mock_generate_structured, service, story_state):
        """Test refinement raises error on generation failure."""
        mock_generate_structured.side_effect = Exception("Generation failed")

        original = {"name": "Test", "description": "Test", "leader": "X", "goals": [], "values": []}
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Faction refinement failed"):
            service._refine_faction(original, scores, story_state, temperature=0.7)


class TestGenerateFactionWithQuality:
    """Tests for generate_faction_with_quality method."""

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    def test_generate_faction_meets_threshold(self, mock_judge, mock_create, service, story_state):
        """Test faction generation that meets threshold."""
        test_faction = {
            "name": "The Order",
            "type": "faction",
            "description": "A powerful and secretive society operating in the shadows of the realm",
            "leader": "Grand Master",
            "goals": ["control"],
            "values": ["power"],
        }
        mock_create.return_value = test_faction

        high_scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.return_value = high_scores

        faction, scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[]
        )

        assert faction["name"] == "The Order"
        assert scores.average >= 7.0
        assert iterations == 1

    def test_generate_faction_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_faction_with_quality(state, existing_names=[])


class TestCreateItem:
    """Tests for _create_item method."""

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_success(self, mock_generate_structured, service, story_state):
        """Test successful item creation."""
        mock_item = Item(
            name="The Crimson Amulet",
            type="item",
            description="An ancient amulet that glows with inner light",
            significance="Key to unlocking the mansion's secrets",
            properties=["glows in darkness", "warm to touch"],
        )
        mock_generate_structured.return_value = mock_item

        item = service._create_item(story_state, existing_names=[], temperature=0.9)

        assert item["name"] == "The Crimson Amulet"
        assert len(item["properties"]) == 2

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_duplicate_name_returns_empty(
        self, mock_generate_structured, service, story_state
    ):
        """Test item creation returns empty dict when name is duplicate."""
        mock_item = Item(
            name="The Crimson Amulet",
            type="item",
            description="An ancient amulet",
            significance="Key item",
            properties=["glows"],
        )
        mock_generate_structured.return_value = mock_item

        # Pass existing name - should return empty to force retry
        result = service._create_item(
            story_state, existing_names=["The Crimson Amulet"], temperature=0.9
        )
        assert result == {}

    def test_create_item_no_brief_returns_empty(self, service):
        """Test item creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_item(state, existing_names=[], temperature=0.9)
        assert result == {}

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test item creation raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Item", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Item creation failed"):
            service._create_item(story_state, existing_names=[], temperature=0.9)


class TestJudgeItemQuality:
    """Tests for _judge_item_quality method."""

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_judge_item_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful item quality judgment."""
        mock_generate_structured.return_value = ItemQualityScores(
            significance=8.0,
            uniqueness=7.5,
            narrative_potential=8.0,
            integration=8.5,
            temporal_plausibility=8.0,
            feedback="Strong item with good story potential",
        )

        item = {
            "name": "Magic Sword",
            "description": "A powerful weapon",
            "significance": "Key to victory",
            "properties": ["sharp"],
        }

        scores = service._judge_item_quality(item, story_state, temperature=0.1)

        assert scores.significance == 8.0
        assert scores.average == 8.0

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_judge_item_quality_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on validation failure."""
        mock_generate_structured.side_effect = Exception("Validation failed")

        item = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_item_quality(item, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_judge_item_quality_no_brief_uses_default_genre(
        self, mock_generate_structured, service
    ):
        """Test judge uses default genre when brief is missing."""
        mock_generate_structured.return_value = ItemQualityScores(
            significance=7.0,
            uniqueness=7.0,
            narrative_potential=7.0,
            integration=7.0,
            temporal_plausibility=7.0,
            feedback="Decent item",
        )

        state = StoryState(id="test-id")
        state.brief = None
        item = {"name": "Test", "description": "Test"}

        scores = service._judge_item_quality(item, state, temperature=0.1)

        assert scores.average == 7.0
        call_args = mock_generate_structured.call_args
        assert "fiction" in call_args.kwargs["prompt"]


class TestRefineItem:
    """Tests for _refine_item method."""

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_success(self, mock_generate_structured, service, story_state):
        """Test successful item refinement."""
        mock_item = Item(
            name="Magic Sword",
            type="item",
            description="A legendary blade with a storied past",
            significance="Central to the hero's journey",
            properties=["cuts through anything", "glows in battle"],
        )
        mock_generate_structured.return_value = mock_item

        original = {
            "name": "Magic Sword",
            "description": "A sword",
            "significance": "Important",
            "properties": ["sharp"],
        }
        scores = ItemQualityScores(
            significance=5.0,
            uniqueness=5.0,
            narrative_potential=6.0,
            integration=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_item(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Magic Sword"
        assert "legendary" in refined["description"]

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_preserves_temporal_fields(
        self, mock_generate_structured, service, story_state
    ):
        """Test that refinement preserves temporal fields from original when LLM omits them."""
        mock_item = Item(
            name="Magic Sword",
            type="item",
            description="A much improved legendary blade",
            significance="Critical to the quest",
        )
        mock_generate_structured.return_value = mock_item

        original = {
            "name": "Magic Sword",
            "description": "A sword",
            "significance": "Important",
            "properties": ["sharp"],
            "creation_year": 150,
            "creation_era": "Bronze Age",
            "temporal_notes": "Forged in ancient times",
        }
        scores = ItemQualityScores(
            significance=5.0,
            uniqueness=5.0,
            narrative_potential=6.0,
            integration=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_item(original, scores, story_state, temperature=0.7)

        assert refined["creation_year"] == 150
        assert refined["creation_era"] == "Bronze Age"
        assert refined["temporal_notes"] == "Forged in ancient times"

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Item", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "significance": "X", "properties": []}
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Item refinement failed"):
            service._refine_item(original, scores, story_state, temperature=0.7)


class TestGenerateItemWithQuality:
    """Tests for generate_item_with_quality method."""

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    def test_generate_item_meets_threshold(self, mock_judge, mock_create, service, story_state):
        """Test item generation that meets threshold."""
        test_item = {
            "name": "Crystal Key",
            "type": "item",
            "description": "A key forged from pure enchanted crystal with a faintly glowing interior",
            "significance": "Opens the final door",
            "properties": ["unbreakable"],
        }
        mock_create.return_value = test_item

        high_scores = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.return_value = high_scores

        item, scores, iterations = service.generate_item_with_quality(
            story_state, existing_names=[]
        )

        assert item["name"] == "Crystal Key"
        assert scores.average >= 7.0
        assert iterations == 1

    def test_generate_item_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_item_with_quality(state, existing_names=[])


class TestCreateConcept:
    """Tests for _create_concept method."""

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_success(self, mock_generate_structured, service, story_state):
        """Test successful concept creation."""
        mock_concept = Concept(
            name="The Price of Truth",
            type="concept",
            description="Truth always comes with consequences that challenge the seeker",
            manifestations="Characters face moral dilemmas when uncovering secrets",
        )
        mock_generate_structured.return_value = mock_concept

        concept = service._create_concept(story_state, existing_names=[], temperature=0.9)

        assert concept["name"] == "The Price of Truth"
        assert concept["type"] == "concept"

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_duplicate_name_returns_empty(
        self, mock_generate_structured, service, story_state
    ):
        """Test concept creation returns empty dict when name is duplicate."""
        mock_concept = Concept(
            name="The Price of Truth",
            type="concept",
            description="Truth always comes with consequences",
            manifestations="Moral dilemmas",
        )
        mock_generate_structured.return_value = mock_concept

        # Pass existing name - should return empty to force retry
        result = service._create_concept(
            story_state, existing_names=["The Price of Truth"], temperature=0.9
        )
        assert result == {}

    def test_create_concept_no_brief_returns_empty(self, service):
        """Test concept creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_concept(state, existing_names=[], temperature=0.9)
        assert result == {}

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test concept creation raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Concept", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Concept creation failed"):
            service._create_concept(story_state, existing_names=[], temperature=0.9)


class TestJudgeConceptQuality:
    """Tests for _judge_concept_quality method."""

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_judge_concept_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful concept quality judgment."""
        mock_generate_structured.return_value = ConceptQualityScores(
            relevance=8.0,
            depth=7.5,
            manifestation=8.0,
            resonance=8.5,
            temporal_plausibility=8.0,
            feedback="Strong thematic concept",
        )

        concept = {
            "name": "Redemption",
            "description": "The journey from darkness to light",
            "manifestations": "Through character arcs",
        }

        scores = service._judge_concept_quality(concept, story_state, temperature=0.1)

        assert scores.relevance == 8.0
        assert scores.average == 8.0

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_judge_concept_quality_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge raises error on validation failure."""
        mock_generate_structured.side_effect = Exception("Validation failed")

        concept = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_concept_quality(concept, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_judge_concept_quality_no_brief_uses_default_genre(
        self, mock_generate_structured, service
    ):
        """Test judge uses default genre when brief is missing."""
        mock_generate_structured.return_value = ConceptQualityScores(
            relevance=7.0,
            depth=7.0,
            manifestation=7.0,
            resonance=7.0,
            temporal_plausibility=7.0,
            feedback="Decent concept",
        )

        state = StoryState(id="test-id")
        state.brief = None
        concept = {"name": "Test", "description": "Test"}

        scores = service._judge_concept_quality(concept, state, temperature=0.1)

        assert scores.average == 7.0
        call_args = mock_generate_structured.call_args
        assert "fiction" in call_args.kwargs["prompt"]


class TestRefineConcept:
    """Tests for _refine_concept method."""

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_success(self, mock_generate_structured, service, story_state):
        """Test successful concept refinement."""
        mock_concept = Concept(
            name="Redemption",
            type="concept",
            description="A profound journey through moral complexity",
            manifestations="Evident in every character's transformation",
        )
        mock_generate_structured.return_value = mock_concept

        original = {
            "name": "Redemption",
            "description": "Getting better",
            "manifestations": "Characters change",
        }
        scores = ConceptQualityScores(
            relevance=5.0,
            depth=5.0,
            manifestation=6.0,
            resonance=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_concept(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Redemption"
        assert "profound" in refined["description"]

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_preserves_temporal_fields(
        self, mock_generate_structured, service, story_state
    ):
        """Test that refinement preserves temporal fields from original when LLM omits them."""
        mock_concept = Concept(
            name="Redemption",
            type="concept",
            description="An improved concept of moral growth",
            manifestations="Deep character transformations",
        )
        mock_generate_structured.return_value = mock_concept

        original = {
            "name": "Redemption",
            "description": "Getting better",
            "manifestations": "Changes",
            "emergence_year": 1,
            "emergence_era": "Dawn Era",
            "temporal_notes": "As old as time",
        }
        scores = ConceptQualityScores(
            relevance=5.0,
            depth=5.0,
            manifestation=6.0,
            resonance=5.0,
            temporal_plausibility=5.0,
        )

        refined = service._refine_concept(original, scores, story_state, temperature=0.7)

        assert refined["emergence_year"] == 1
        assert refined["emergence_era"] == "Dawn Era"
        assert refined["temporal_notes"] == "As old as time"

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_invalid_json_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Concept", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "manifestations": "X"}
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Concept refinement failed"):
            service._refine_concept(original, scores, story_state, temperature=0.7)


class TestGenerateConceptWithQuality:
    """Tests for generate_concept_with_quality method."""

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    def test_generate_concept_meets_threshold(self, mock_judge, mock_create, service, story_state):
        """Test concept generation that meets threshold."""
        test_concept = {
            "name": "Truth vs Loyalty",
            "type": "concept",
            "description": "The profound tension between absolute honesty and unwavering personal allegiance",
            "manifestations": "Characters must choose between truth and friends",
        }
        mock_create.return_value = test_concept

        high_scores = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.return_value = high_scores

        concept, scores, iterations = service.generate_concept_with_quality(
            story_state, existing_names=[]
        )

        assert concept["name"] == "Truth vs Loyalty"
        assert scores.average >= 7.0
        assert iterations == 1

    def test_generate_concept_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_concept_with_quality(state, existing_names=[])


class TestBatchOperations:
    """Tests for batch generation methods."""

    @patch.object(WorldQualityService, "generate_character_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_characters_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch character generation."""
        char1 = Character(name="Character One", role="protagonist", description="First")
        char2 = Character(name="Character Two", role="antagonist", description="Second")
        scores1 = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=7.5,
            uniqueness=8.0,
            arc_potential=8.5,
            temporal_plausibility=8.0,
        )
        scores2 = CharacterQualityScores(
            depth=7.5,
            goals=8.0,
            flaws=8.0,
            uniqueness=7.5,
            arc_potential=8.0,
            temporal_plausibility=7.5,
        )

        mock_gen.side_effect = [
            (char1, scores1, 1),
            (char2, scores2, 2),
        ]

        results = service.generate_characters_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 2
        assert results[0][0].name == "Character One"
        assert results[1][0].name == "Character Two"
        assert mock_gen.call_count == 2
        assert results[0][1].average == scores1.average
        assert results[1][1].average == scores2.average

    @patch.object(WorldQualityService, "generate_character_with_quality")
    def test_generate_characters_partial_failure(self, mock_gen, service, story_state):
        """Test batch character generation with some failures."""
        char1 = Character(name="Character One", role="protagonist", description="First")
        scores1 = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=7.5,
            uniqueness=8.0,
            arc_potential=8.5,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (char1, scores1, 1),
            WorldGenerationError("Failed"),
        ]

        results = service.generate_characters_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 1
        assert results[0][0].name == "Character One"

    @patch.object(WorldQualityService, "generate_character_with_quality")
    def test_generate_characters_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch character generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any characters"):
            service.generate_characters_with_quality(story_state, name_provider=lambda: [], count=2)

    @patch.object(WorldQualityService, "generate_location_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_locations_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch location generation."""
        loc1 = {"name": "Location One", "description": "First"}
        loc2 = {"name": "Location Two", "description": "Second"}
        scores1 = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        scores2 = LocationQualityScores(
            atmosphere=7.5,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=7.5,
            temporal_plausibility=7.5,
        )

        mock_gen.side_effect = [
            (loc1, scores1, 1),
            (loc2, scores2, 2),
        ]

        results = service.generate_locations_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 2
        assert results[0][0]["name"] == "Location One"
        assert results[1][0]["name"] == "Location Two"

    @patch.object(WorldQualityService, "generate_location_with_quality")
    def test_generate_locations_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch location generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any locations"):
            service.generate_locations_with_quality(story_state, name_provider=lambda: [], count=2)

    @patch.object(WorldQualityService, "generate_faction_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_factions_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch faction generation."""
        faction1 = {"name": "Faction One", "description": "First"}
        faction2 = {"name": "Faction Two", "description": "Second"}
        scores1 = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        scores2 = FactionQualityScores(
            coherence=7.5,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=7.5,
            temporal_plausibility=7.5,
        )

        mock_gen.side_effect = [
            (faction1, scores1, 1),
            (faction2, scores2, 2),
        ]

        results = service.generate_factions_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 2
        assert results[0][0]["name"] == "Faction One"
        assert results[1][0]["name"] == "Faction Two"

    @patch.object(WorldQualityService, "generate_faction_with_quality")
    def test_generate_factions_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch faction generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any factions"):
            service.generate_factions_with_quality(story_state, name_provider=lambda: [], count=2)

    @patch.object(WorldQualityService, "generate_item_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_items_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch item generation."""
        item1 = {"name": "Item One", "description": "First"}
        item2 = {"name": "Item Two", "description": "Second"}
        scores1 = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )
        scores2 = ItemQualityScores(
            significance=7.5,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=7.5,
            temporal_plausibility=7.5,
        )

        mock_gen.side_effect = [
            (item1, scores1, 1),
            (item2, scores2, 2),
        ]

        results = service.generate_items_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 2
        assert results[0][0]["name"] == "Item One"
        assert results[1][0]["name"] == "Item Two"

    @patch.object(WorldQualityService, "generate_item_with_quality")
    def test_generate_items_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch item generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any items"):
            service.generate_items_with_quality(story_state, name_provider=lambda: [], count=2)

    @patch.object(WorldQualityService, "generate_concept_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_concepts_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch concept generation."""
        concept1 = {"name": "Concept One", "description": "First"}
        concept2 = {"name": "Concept Two", "description": "Second"}
        scores1 = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )
        scores2 = ConceptQualityScores(
            relevance=7.5,
            depth=8.0,
            manifestation=8.0,
            resonance=7.5,
            temporal_plausibility=7.5,
        )

        mock_gen.side_effect = [
            (concept1, scores1, 1),
            (concept2, scores2, 2),
        ]

        results = service.generate_concepts_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 2
        assert results[0][0]["name"] == "Concept One"
        assert results[1][0]["name"] == "Concept Two"

    @patch.object(WorldQualityService, "generate_concept_with_quality")
    def test_generate_concepts_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch concept generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any concepts"):
            service.generate_concepts_with_quality(story_state, name_provider=lambda: [], count=2)

    @patch.object(WorldQualityService, "_make_model_preparers", return_value=(None, None))
    @patch.object(WorldQualityService, "generate_relationship_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_relationships_with_quality(
        self, mock_record, mock_gen, _mock_preparers, service, story_state
    ):
        """Test batch relationship generation."""
        rel1 = {"source": "A", "target": "B", "relation_type": "knows", "description": "First"}
        rel2 = {"source": "B", "target": "C", "relation_type": "loves", "description": "Second"}
        scores1 = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0
        )
        scores2 = RelationshipQualityScores(
            tension=7.5, dynamics=8.0, story_potential=8.0, authenticity=7.5
        )

        mock_gen.side_effect = [
            (rel1, scores1, 1),
            (rel2, scores2, 2),
        ]

        results = service.generate_relationships_with_quality(
            story_state, entity_names_provider=lambda: ["A", "B", "C"], existing_rels=[], count=2
        )

        assert len(results) == 2
        assert results[0][0]["source"] == "A"
        assert results[1][0]["source"] == "B"

    @patch.object(WorldQualityService, "_make_model_preparers", return_value=(None, None))
    @patch.object(WorldQualityService, "generate_relationship_with_quality")
    def test_generate_relationships_all_fail_raises_error(
        self, mock_gen, _mock_preparers, service, story_state
    ):
        """Test batch relationship generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any relationships"):
            service.generate_relationships_with_quality(
                story_state, entity_names_provider=lambda: ["A", "B"], existing_rels=[], count=2
            )


class TestMiniDescriptions:
    """Tests for mini description generation."""

    def test_generate_mini_description_short_text_returned_as_is(self, service, settings):
        """Test short descriptions are returned as-is."""
        result = service.generate_mini_description(
            name="Test Entity",
            entity_type="character",
            full_description="A simple short description",
        )

        # Short enough, should be returned as-is or trimmed
        assert len(result.split()) <= settings.mini_description_words_max

    def _make_mini_desc_response(
        self, summary: str, prompt_eval_count: int = 10, eval_count: int = 5
    ) -> Iterator[MockStreamChunk]:
        """Create a mock streaming chat response for structured mini description output.

        Args:
            summary: The summary text for the MiniDescription model.
            prompt_eval_count: Simulated prompt token count.
            eval_count: Simulated completion token count.

        Returns:
            Iterator of MockStreamChunk compatible with consume_stream().
        """
        return iter(
            [
                MockStreamChunk(
                    content=json.dumps({"summary": summary}),
                    done=True,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                ),
            ]
        )

    def test_generate_mini_description_llm_called_for_long_text(self, service, mock_ollama_client):
        """Test that LLM is called for long descriptions via structured output."""
        long_description = " ".join(["word"] * 50)  # 50 words
        mock_ollama_client.chat.return_value = self._make_mini_desc_response(
            "A concise summary of the entity"
        )
        service._client = mock_ollama_client

        result = service.generate_mini_description(
            name="Test Entity",
            entity_type="character",
            full_description=long_description,
        )

        assert result == "A concise summary of the entity"
        mock_ollama_client.chat.assert_called_once()

    def test_generate_mini_description_structured_output_parses_cleanly(
        self, service, mock_ollama_client
    ):
        """Test that structured output parses JSON cleanly without quote stripping."""
        mock_ollama_client.chat.return_value = self._make_mini_desc_response("A clean summary")
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        assert result == "A clean summary"

    def test_generate_mini_description_structured_output_no_preambles(
        self, service, mock_ollama_client
    ):
        """Test that structured output prevents conversational preambles."""
        mock_ollama_client.chat.return_value = self._make_mini_desc_response("A cunning warrior")
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        assert result == "A cunning warrior"
        assert "<think>" not in result

    def test_generate_mini_description_empty_summary_fallback(
        self, service, settings, mock_ollama_client, caplog
    ):
        """Test fallback to truncation when structured output returns empty summary."""
        mock_ollama_client.chat.return_value = self._make_mini_desc_response("")
        service._client = mock_ollama_client

        # Set max_words to 5 for predictable output
        settings.mini_description_words_max = 5

        long_description = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        result = service.generate_mini_description(
            name="Test Entity",
            entity_type="character",
            full_description=long_description,
        )

        # Should fall back to truncated description
        assert result == "word1 word2 word3 word4 word5..."
        assert "empty for" in caplog.text

    def test_generate_mini_description_truncates_long_response(
        self, service, settings, mock_ollama_client
    ):
        """Test that overly long responses are truncated."""
        # Response with more words than max + 3
        long_response = " ".join(["word"] * 50)
        mock_ollama_client.chat.return_value = self._make_mini_desc_response(long_response)
        service._client = mock_ollama_client

        long_description = " ".join(["description"] * 50)
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        # Should be truncated with ellipsis
        assert result.endswith("...")
        assert len(result.split()) <= settings.mini_description_words_max + 1

    def test_generate_mini_description_handles_error(self, service, settings, mock_ollama_client):
        """Test fallback when LLM fails."""
        mock_ollama_client.chat.side_effect = Exception("LLM error")
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        # Should fallback to truncated description
        assert len(result.split()) <= settings.mini_description_words_max + 1
        assert result.endswith("...")

    def test_generate_mini_description_passes_json_schema_format(self, service, mock_ollama_client):
        """Test that chat call includes format=MiniDescription.model_json_schema()."""
        from src.services.world_quality_service._validation import MiniDescription

        mock_ollama_client.chat.return_value = self._make_mini_desc_response("A test summary")
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        assert call_kwargs["format"] == MiniDescription.model_json_schema()

    def test_generate_mini_descriptions_batch(self, service, mock_ollama_client):
        """Test batch mini description generation."""
        # Use side_effect callable for fresh iterator on each call (batch makes 2 calls)
        mock_ollama_client.chat.side_effect = lambda *a, **kw: self._make_mini_desc_response(
            "Short summary"
        )
        service._client = mock_ollama_client

        entities = [
            {"name": "Entity One", "type": "character", "description": " ".join(["word"] * 50)},
            {"name": "Entity Two", "type": "location", "description": " ".join(["word"] * 50)},
            {"name": "Entity Three", "type": "item", "description": ""},  # Empty description
        ]

        results = service.generate_mini_descriptions_batch(entities)

        # Only entities with descriptions should be processed
        assert "Entity One" in results
        assert "Entity Two" in results
        assert "Entity Three" not in results
        assert mock_ollama_client.chat.call_count == 2

    def test_generate_mini_description_logs_token_counts(self, service, mock_ollama_client, caplog):
        """Test that mini description logs model name and token counts at INFO (#304)."""
        mock_ollama_client.chat.return_value = self._make_mini_desc_response(
            "A brave warrior", prompt_eval_count=80, eval_count=20
        )
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._validation"):
            service.generate_mini_description(
                name="Test Hero",
                entity_type="character",
                full_description=long_description,
            )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "Mini description LLM call" in r.message and "tokens:" in r.message.lower()
            for r in info_records
        )

    def test_generate_mini_descriptions_batch_logs_summary(
        self, service, mock_ollama_client, caplog
    ):
        """Test that batch mini descriptions logs aggregate timing (#304)."""
        # Use side_effect callable for fresh iterator on each call (batch makes 2 calls)
        mock_ollama_client.chat.side_effect = lambda *a, **kw: self._make_mini_desc_response(
            "Short summary", prompt_eval_count=50, eval_count=10
        )
        service._client = mock_ollama_client

        entities = [
            {"name": "Entity One", "type": "character", "description": " ".join(["word"] * 50)},
            {"name": "Entity Two", "type": "location", "description": " ".join(["word"] * 50)},
        ]

        with caplog.at_level(logging.INFO, logger="src.services.world_quality_service._validation"):
            service.generate_mini_descriptions_batch(entities)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("Completed mini description batch" in r.message for r in info_records)


class TestSettingsValidation:
    """Tests for world quality settings validation."""

    def test_valid_settings(self):
        """Test valid settings are accepted."""
        settings = Settings(
            world_quality_enabled=True,
            world_quality_max_iterations=5,
            world_quality_threshold=8.0,
            world_quality_creator_temp=0.8,
            world_quality_judge_temp=0.2,
            world_quality_refinement_temp=0.6,
        )
        settings.validate()  # Should not raise
        assert settings.world_quality_max_iterations == 5
        assert settings.world_quality_threshold == 8.0

    def test_invalid_max_iterations(self):
        """Test max_iterations validation."""
        settings = Settings(world_quality_max_iterations=0)
        with pytest.raises(ValueError, match="world_quality_max_iterations"):
            settings.validate()

        settings = Settings(world_quality_max_iterations=11)
        with pytest.raises(ValueError, match="world_quality_max_iterations"):
            settings.validate()

    def test_invalid_threshold(self):
        """Test threshold validation."""
        settings = Settings(world_quality_threshold=-1.0)
        with pytest.raises(ValueError, match="world_quality_threshold"):
            settings.validate()

        settings = Settings(world_quality_threshold=11.0)
        with pytest.raises(ValueError, match="world_quality_threshold"):
            settings.validate()

    def test_invalid_temperatures(self):
        """Test temperature validation."""
        settings = Settings(world_quality_creator_temp=3.0)
        with pytest.raises(ValueError, match="world_quality_creator_temp"):
            settings.validate()

        settings = Settings(world_quality_judge_temp=-0.1)
        with pytest.raises(ValueError, match="world_quality_judge_temp"):
            settings.validate()

        settings = Settings(world_quality_refinement_temp=2.5)
        with pytest.raises(ValueError, match="world_quality_refinement_temp"):
            settings.validate()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_character_refinement_loop_with_multiple_iterations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test character goes through multiple refinement iterations."""
        initial_char = Character(name="Test", role="supporting", description="Basic")
        refined_char1 = Character(name="Test", role="supporting", description="Better")
        refined_char2 = Character(name="Test", role="supporting", description="Best")

        mock_create.return_value = initial_char
        mock_refine.side_effect = [refined_char1, refined_char2]

        # Scores improve over iterations
        scores_iter1 = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
        )
        scores_iter2 = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )
        scores_iter3 = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [scores_iter1, scores_iter2, scores_iter3]

        char, scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        assert char.description == "Best"
        assert scores.average >= 7.0
        assert iterations == 3
        assert mock_refine.call_count == 2

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    @patch.object(WorldQualityService, "_refine_location")
    def test_location_generation_returns_below_threshold_after_max(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test location returned even below threshold after max iterations.

        With best-iteration tracking, returns iteration 1 when all scores are equal.
        """
        test_loc = {
            "name": "Basic",
            "description": "A simple and unremarkable location lacking in atmosphere and detail",
        }
        mock_create.return_value = test_loc
        mock_refine.return_value = test_loc  # Mock refinement to prevent errors

        low_scores = LocationQualityScores(
            atmosphere=5.0,
            significance=5.0,
            story_relevance=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.return_value = low_scores

        loc, scores, iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert loc["name"] == "Basic"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1

    def test_empty_description_mini_description(self, service):
        """Test mini description with empty full description."""
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description="",
        )
        assert result == ""

    @patch.object(WorldQualityService, "generate_character_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_batch_updates_existing_names(self, mock_record, mock_gen, service, story_state):
        """Test batch generation properly updates existing names between iterations."""
        char1 = Character(name="Alice", role="protagonist", description="First")
        char2 = Character(name="Bob", role="antagonist", description="Second")
        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (char1, scores, 1),
            (char2, scores, 1),
        ]

        service.generate_characters_with_quality(
            story_state, name_provider=lambda: ["Existing"], count=2
        )

        # Second call should include both Existing and Alice
        # Args are passed positionally: (story_state, existing_names)
        second_call_args = mock_gen.call_args_list[1]
        existing_names_arg = second_call_args[0][1]  # Second positional argument
        assert "Existing" in existing_names_arg
        assert "Alice" in existing_names_arg

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    @patch.object(WorldQualityService, "_refine_relationship")
    def test_relationship_refinement_loop(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test relationship goes through refinement when below threshold."""
        initial_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "Basic relationship",
        }
        refined_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "Rich, complex relationship",
        }

        mock_create.return_value = initial_rel
        mock_refine.return_value = refined_rel

        low_scores = RelationshipQualityScores(
            tension=5.0, dynamics=5.0, story_potential=5.0, authenticity=5.0
        )
        high_scores = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0
        )
        mock_judge.side_effect = [low_scores, high_scores]

        rel, scores, iterations = service.generate_relationship_with_quality(
            story_state,
            entity_names=["Alice", "Bob"],
            existing_rels=[],
        )

        assert rel["description"] == "Rich, complex relationship"
        assert scores.average >= 7.0
        assert iterations == 2
        mock_refine.assert_called_once()


class TestExceptionHandlingPaths:
    """Tests for exception handling paths that weren't covered."""

    # ========== Character Creation Exception Paths ==========

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_value_error(self, mock_generate_structured, service, story_state):
        """Test character creation handles ValueError from generate_structured."""
        mock_generate_structured.side_effect = ValueError("Cannot convert to float")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_key_error(self, mock_generate_structured, service, story_state):
        """Test character creation handles KeyError from generate_structured."""
        mock_generate_structured.side_effect = KeyError("missing key")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_type_error(self, mock_generate_structured, service, story_state):
        """Test character creation handles TypeError from generate_structured."""
        mock_generate_structured.side_effect = TypeError("wrong type")

        with pytest.raises(WorldGenerationError, match="Character creation failed"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    # ========== Character Refinement Exception Paths ==========

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_value_error(self, mock_generate_structured, service, story_state):
        """Test character refinement handles ValueError from generate_structured."""
        mock_generate_structured.side_effect = ValueError("Cannot parse")

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Character refinement failed"):
            service._refine_character(original_char, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_refine_character_unexpected_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test character refinement handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected attribute error")

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Character refinement failed"):
            service._refine_character(original_char, scores, story_state, temperature=0.7)

    # ========== Location Creation Exception Paths ==========

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_json_parsing_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test location creation handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Location", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Location creation failed"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_create_location_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test location creation handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected error")

        with pytest.raises(WorldGenerationError, match="Location creation failed"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    # ========== Location Judge Exception Paths ==========

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_judge_location_quality_unexpected_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test location judge handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        location = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="Location quality judgment failed"):
            service._judge_location_quality(location, story_state, temperature=0.1)

    # ========== Location Refinement Exception Paths ==========

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_json_parsing_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test location refinement handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Location", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Location refinement failed"):
            service._refine_location(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_refine_location_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test location refinement handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0,
            significance=6.0,
            story_relevance=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Location refinement failed"):
            service._refine_location(original, scores, story_state, temperature=0.7)

    # ========== Relationship Creation Exception Paths ==========

    def test_create_relationship_json_parsing_error(self, service, story_state, mock_ollama_client):
        """Test relationship creation handles JSON parsing errors."""
        mock_ollama_client.generate.return_value = {"response": "not valid json!!!"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid relationship response format"):
            service._create_relationship(
                story_state, entity_names=["A", "B"], existing_rels=[], temperature=0.9
            )

    def test_create_relationship_non_dict_json_response(
        self, service, story_state, mock_ollama_client
    ):
        """Test relationship creation rejects non-dict JSON responses (e.g., lists)."""
        mock_ollama_client.generate.return_value = {"response": "[]"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid relationship JSON structure"):
            service._create_relationship(
                story_state, entity_names=["A", "B"], existing_rels=[], temperature=0.9
            )

    def test_create_relationship_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test relationship creation handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Unexpected relationship creation error"):
            service._create_relationship(
                story_state, entity_names=["A", "B"], existing_rels=[], temperature=0.9
            )

    # ========== Relationship Judge Exception Paths ==========

    def test_judge_relationship_quality_unexpected_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test relationship judge handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        relationship = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}

        with pytest.raises(WorldGenerationError, match="Relationship quality judgment failed"):
            service._judge_relationship_quality(relationship, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._relationship.generate_structured")
    def test_judge_relationship_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test relationship judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        relationship = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Relationship quality judgment failed"):
                service._judge_relationship_quality(relationship, story_state, temperature=0.1)

        # The _relationship module should log WARNING (not ERROR) for individual call failures
        rel_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_relationship" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in rel_warnings)
        # No ERROR from _relationship module (only _common logs ERROR for the aggregate fallback)
        rel_errors = [
            r for r in caplog.records if r.levelno == logging.ERROR and "_relationship" in r.name
        ]
        assert len(rel_errors) == 0

    # ========== Multi-Call Warning Paths (other entity types) ==========

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_judge_character_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test character judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        character = Character(name="Test", role="supporting", description="Test")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Character quality judgment failed"):
                service._judge_character_quality(character, story_state, temperature=0.1)

        char_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_character" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in char_warnings)

    @patch("src.services.world_quality_service._location.generate_structured")
    def test_judge_location_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test location judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        location = {"name": "Test", "description": "Test"}

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Location quality judgment failed"):
                service._judge_location_quality(location, story_state, temperature=0.1)

        loc_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_location" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in loc_warnings)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_judge_faction_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test faction judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        faction = {"name": "Test", "description": "Test"}

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Faction quality judgment failed"):
                service._judge_faction_quality(faction, story_state, temperature=0.1)

        fac_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_faction" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in fac_warnings)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_judge_item_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test item judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        item = {"name": "Test", "description": "Test"}

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Item quality judgment failed"):
                service._judge_item_quality(item, story_state, temperature=0.1)

        item_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_item" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in item_warnings)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_judge_concept_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test concept judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        concept = {"name": "Test", "description": "Test"}

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Concept quality judgment failed"):
                service._judge_concept_quality(concept, story_state, temperature=0.1)

        con_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_concept" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in con_warnings)

    @patch("src.services.world_quality_service._plot.generate_structured")
    def test_judge_plot_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test plot judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        plot_outline = PlotOutline(plot_summary="A mystery unfolds", plot_points=[])

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Plot quality judgment failed"):
                service._judge_plot_quality(plot_outline, story_state, temperature=0.1)

        plot_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_plot" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in plot_warnings)

    @patch("src.services.world_quality_service._chapter_quality.generate_structured")
    def test_judge_chapter_quality_error_with_multi_call_logs_warning(
        self, mock_generate_structured, service, story_state, caplog
    ):
        """Test chapter judge logs warning (not error) when multi-call is enabled."""
        service.settings.judge_multi_call_enabled = True
        mock_generate_structured.side_effect = Exception("LLM timeout")

        chapter = Chapter(number=1, title="The Beginning", outline="A mystery begins")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(WorldGenerationError, match="Chapter quality judgment failed"):
                service._judge_chapter_quality(chapter, story_state, temperature=0.1)

        ch_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "_chapter" in r.name
        ]
        assert any("judgment failed" in msg.message for msg in ch_warnings)

    # ========== Relationship Refinement Exception Paths ==========

    def test_refine_relationship_llm_error(self, service, story_state, mock_ollama_client):
        """Test relationship refinement handles LLM errors."""
        mock_ollama_client.generate.side_effect = ollama.ResponseError("LLM error")
        service._client = mock_ollama_client

        original = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}
        scores = RelationshipQualityScores(
            tension=6.0, dynamics=6.0, story_potential=6.0, authenticity=6.0
        )

        with pytest.raises(WorldGenerationError, match="LLM error during relationship refinement"):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

    def test_refine_relationship_json_parsing_error(self, service, story_state, mock_ollama_client):
        """Test relationship refinement handles JSON parsing errors."""
        mock_ollama_client.generate.return_value = {"response": "not valid json!!!"}
        service._client = mock_ollama_client

        original = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}
        scores = RelationshipQualityScores(
            tension=6.0, dynamics=6.0, story_potential=6.0, authenticity=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid relationship refinement"):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

    def test_refine_relationship_non_dict_json_response(
        self, service, story_state, mock_ollama_client
    ):
        """Test relationship refinement rejects non-dict JSON responses."""
        mock_ollama_client.generate.return_value = {"response": "[]"}
        service._client = mock_ollama_client

        original = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}
        scores = RelationshipQualityScores(
            tension=6.0, dynamics=6.0, story_potential=6.0, authenticity=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid relationship refinement JSON"):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

    def test_refine_relationship_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test relationship refinement handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        original = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}
        scores = RelationshipQualityScores(
            tension=6.0, dynamics=6.0, story_potential=6.0, authenticity=6.0
        )

        with pytest.raises(WorldGenerationError, match="Unexpected relationship refinement error"):
            service._refine_relationship(original, scores, story_state, temperature=0.7)

    # ========== Faction Creation Exception Paths ==========

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_generation_error(self, mock_generate_structured, service, story_state):
        """Test faction creation handles generation errors."""
        mock_generate_structured.side_effect = ValueError("validation error")

        with pytest.raises(WorldGenerationError, match="Faction creation failed"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_create_faction_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test faction creation handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        with pytest.raises(WorldGenerationError, match="Faction creation failed"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)

    # ========== Faction Judge Exception Paths ==========

    def test_judge_faction_quality_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test faction judge handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        faction = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="Faction quality judgment failed"):
            service._judge_faction_quality(faction, story_state, temperature=0.1)

    # ========== Faction Refinement Exception Paths ==========

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_llm_error(self, mock_generate_structured, service, story_state):
        """Test faction refinement handles LLM errors."""
        mock_generate_structured.side_effect = ConnectionError("Connection lost")

        original = {"name": "Test", "description": "Test", "leader": "X", "goals": [], "values": []}
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Faction refinement failed"):
            service._refine_faction(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_generation_error(self, mock_generate_structured, service, story_state):
        """Test faction refinement handles generation errors."""
        mock_generate_structured.side_effect = ValueError("validation error")

        original = {"name": "Test", "description": "Test", "leader": "X", "goals": [], "values": []}
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Faction refinement failed"):
            service._refine_faction(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._faction.generate_structured")
    def test_refine_faction_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test faction refinement handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        original = {"name": "Test", "description": "Test", "leader": "X", "goals": [], "values": []}
        scores = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Faction refinement failed"):
            service._refine_faction(original, scores, story_state, temperature=0.7)

    # ========== Item Creation Exception Paths ==========

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_llm_error(self, mock_generate_structured, service, story_state):
        """Test item creation handles LLM errors."""
        mock_generate_structured.side_effect = ollama.ResponseError("Model error")

        with pytest.raises(WorldGenerationError, match="Item creation failed"):
            service._create_item(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_json_parsing_error(self, mock_generate_structured, service, story_state):
        """Test item creation handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Item", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Item creation failed"):
            service._create_item(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_create_item_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test item creation handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        with pytest.raises(WorldGenerationError, match="Item creation failed"):
            service._create_item(story_state, existing_names=[], temperature=0.9)

    # ========== Item Judge Exception Paths ==========

    def test_judge_item_quality_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test item judge handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        item = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="Item quality judgment failed"):
            service._judge_item_quality(item, story_state, temperature=0.1)

    # ========== Item Refinement Exception Paths ==========

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_llm_error(self, mock_generate_structured, service, story_state):
        """Test item refinement handles LLM errors."""
        mock_generate_structured.side_effect = TimeoutError("Timeout")

        original = {"name": "Test", "description": "Test", "significance": "X", "properties": []}
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Item refinement failed"):
            service._refine_item(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_json_parsing_error(self, mock_generate_structured, service, story_state):
        """Test item refinement handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Item", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "significance": "X", "properties": []}
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Item refinement failed"):
            service._refine_item(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._item.generate_structured")
    def test_refine_item_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test item refinement handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        original = {"name": "Test", "description": "Test", "significance": "X", "properties": []}
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=6.0,
            narrative_potential=6.0,
            integration=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Item refinement failed"):
            service._refine_item(original, scores, story_state, temperature=0.7)

    # ========== Concept Creation Exception Paths ==========

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_llm_error(self, mock_generate_structured, service, story_state):
        """Test concept creation handles LLM errors."""
        mock_generate_structured.side_effect = ConnectionError("Connection refused")

        with pytest.raises(WorldGenerationError, match="Concept creation failed"):
            service._create_concept(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_json_parsing_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test concept creation handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Concept", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Concept creation failed"):
            service._create_concept(story_state, existing_names=[], temperature=0.9)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_create_concept_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test concept creation handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        with pytest.raises(WorldGenerationError, match="Concept creation failed"):
            service._create_concept(story_state, existing_names=[], temperature=0.9)

    # ========== Concept Judge Exception Paths ==========

    def test_judge_concept_quality_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test concept judge handles unexpected errors."""
        mock_ollama_client.generate.side_effect = AttributeError("Unexpected")
        service._client = mock_ollama_client

        concept = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="Concept quality judgment failed"):
            service._judge_concept_quality(concept, story_state, temperature=0.1)

    # ========== Concept Refinement Exception Paths ==========

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_llm_error(self, mock_generate_structured, service, story_state):
        """Test concept refinement handles LLM errors."""
        mock_generate_structured.side_effect = ollama.ResponseError("LLM error")

        original = {"name": "Test", "description": "Test", "manifestations": "X"}
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Concept refinement failed"):
            service._refine_concept(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_json_parsing_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test concept refinement handles validation errors."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "Concept", [{"type": "missing", "loc": ("name",), "input": {}}]
        )

        original = {"name": "Test", "description": "Test", "manifestations": "X"}
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Concept refinement failed"):
            service._refine_concept(original, scores, story_state, temperature=0.7)

    @patch("src.services.world_quality_service._concept.generate_structured")
    def test_refine_concept_unexpected_error(self, mock_generate_structured, service, story_state):
        """Test concept refinement handles unexpected errors."""
        mock_generate_structured.side_effect = AttributeError("Unexpected")

        original = {"name": "Test", "description": "Test", "manifestations": "X"}
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=6.0,
            manifestation=6.0,
            resonance=6.0,
            temporal_plausibility=6.0,
        )

        with pytest.raises(WorldGenerationError, match="Concept refinement failed"):
            service._refine_concept(original, scores, story_state, temperature=0.7)


class TestRefinementLoopEdgeCases:
    """Tests for refinement loop edge cases that weren't covered."""

    # ========== Relationship Loop Edge Cases ==========

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    def test_generate_relationship_error_during_iteration(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test relationship generation handles errors during iteration."""
        test_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "They know each other",
        }
        mock_create.return_value = test_rel
        # First judgment succeeds but returns low score, second raises error
        low_scores = RelationshipQualityScores(
            tension=5.0, dynamics=5.0, story_potential=5.0, authenticity=5.0
        )
        mock_judge.side_effect = [low_scores, WorldGenerationError("Judge failed")]

        # Should still return relationship despite error on 2nd iteration
        # because we have valid results from 1st iteration
        rel, scores, _iterations = service.generate_relationship_with_quality(
            story_state, entity_names=["Alice", "Bob", "Charlie"], existing_rels=[]
        )

        # We should have the result from before the error
        assert rel["source"] == "Alice"
        assert scores.average < 7.0

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    @patch.object(WorldQualityService, "_refine_relationship")
    def test_generate_relationship_below_threshold_after_max(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test relationship generation returns below threshold after max iterations.

        With early stopping, if scores don't improve or degrade, the loop
        runs until max_iterations (no early stop on plateau). Returns best iteration.
        """
        test_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "Basic",
        }
        mock_create.return_value = test_rel
        mock_refine.return_value = test_rel  # Mock refinement to prevent errors

        low_scores = RelationshipQualityScores(
            tension=5.0, dynamics=5.0, story_potential=5.0, authenticity=5.0
        )
        mock_judge.return_value = low_scores

        rel, scores, iterations = service.generate_relationship_with_quality(
            story_state, entity_names=["Alice", "Bob"], existing_rels=[]
        )

        assert rel["source"] == "Alice"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1

    # ========== Faction Loop Edge Cases ==========

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    @patch.object(WorldQualityService, "_refine_faction")
    def test_generate_faction_needs_refinement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test faction generation goes through refinement."""
        initial_faction = {
            "name": "Basic Guild",
            "description": "A simple trade guild with few members and little political influence",
        }
        mock_create.return_value = initial_faction

        refined_faction = {
            "name": "Basic Guild",
            "description": "A complex guild with deep historical roots and far-reaching political influence",
        }
        mock_refine.return_value = refined_faction

        low_scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        high_scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [low_scores, high_scores]

        faction, scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[]
        )

        assert (
            faction["description"]
            == "A complex guild with deep historical roots and far-reaching political influence"
        )
        assert scores.average >= 7.0
        assert iterations == 2
        mock_refine.assert_called_once()

    @patch.object(WorldQualityService, "_create_faction")
    def test_generate_faction_empty_name_fails(self, mock_create, service, story_state):
        """Test faction generation fails when creation returns empty name."""
        mock_create.return_value = {"description": "No name"}

        with pytest.raises(WorldGenerationError, match="Failed to generate faction"):
            service.generate_faction_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    def test_generate_faction_error_during_iteration(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test faction generation handles errors during iteration."""
        test_faction = {
            "name": "Test Guild",
            "description": "A test guild used to verify error handling during quality iteration",
        }
        mock_create.return_value = test_faction
        mock_judge.side_effect = WorldGenerationError("Judge failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate faction"):
            service.generate_faction_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    def test_generate_faction_below_threshold_after_max(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test faction generation returns below threshold after max iterations."""
        test_faction = {
            "name": "Test Guild",
            "description": "A basic and underdeveloped guild with minimal presence in the region",
        }
        mock_create.return_value = test_faction

        low_scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.return_value = low_scores

        faction, scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[]
        )

        assert faction["name"] == "Test Guild"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1

    # ========== Item Loop Edge Cases ==========

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    @patch.object(WorldQualityService, "_refine_item")
    def test_generate_item_needs_refinement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test item generation goes through refinement."""
        initial_item = {
            "name": "Basic Sword",
            "description": "A simple iron sword with no distinguishing features or special history",
        }
        mock_create.return_value = initial_item

        refined_item = {
            "name": "Basic Sword",
            "description": "A legendary blade with centuries of history forged in an ancient tradition",
        }
        mock_refine.return_value = refined_item

        low_scores = ItemQualityScores(
            significance=5.0,
            uniqueness=5.0,
            narrative_potential=5.0,
            integration=5.0,
            temporal_plausibility=5.0,
        )
        high_scores = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [low_scores, high_scores]

        item, scores, iterations = service.generate_item_with_quality(
            story_state, existing_names=[]
        )

        assert (
            item["description"]
            == "A legendary blade with centuries of history forged in an ancient tradition"
        )
        assert scores.average >= 7.0
        assert iterations == 2
        mock_refine.assert_called_once()

    @patch.object(WorldQualityService, "_create_item")
    def test_generate_item_empty_name_fails(self, mock_create, service, story_state):
        """Test item generation fails when creation returns empty name."""
        mock_create.return_value = {"description": "No name"}

        with pytest.raises(WorldGenerationError, match="Failed to generate item"):
            service.generate_item_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    def test_generate_item_error_during_iteration(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test item generation handles errors during iteration."""
        test_item = {
            "name": "Test Item",
            "description": "A test item used to verify error handling behavior during quality iteration",
        }
        mock_create.return_value = test_item
        mock_judge.side_effect = WorldGenerationError("Judge failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate item"):
            service.generate_item_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    @patch.object(WorldQualityService, "_refine_item")
    def test_generate_item_below_threshold_after_max(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test item generation returns below threshold after max iterations.

        With best-iteration tracking, returns iteration 1 when all scores are equal.
        """
        test_item = {
            "name": "Test Item",
            "description": "A basic and unremarkable item with little narrative significance or lore",
        }
        mock_create.return_value = test_item
        mock_refine.return_value = test_item  # Mock refinement to prevent errors

        low_scores = ItemQualityScores(
            significance=5.0,
            uniqueness=5.0,
            narrative_potential=5.0,
            integration=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.return_value = low_scores

        item, scores, iterations = service.generate_item_with_quality(
            story_state, existing_names=[]
        )

        assert item["name"] == "Test Item"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1

    # ========== Concept Loop Edge Cases ==========

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    @patch.object(WorldQualityService, "_refine_concept")
    def test_generate_concept_needs_refinement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test concept generation goes through refinement."""
        initial_concept = {
            "name": "Basic Theme",
            "description": "A simple and underdeveloped theme lacking philosophical depth and nuance",
        }
        mock_create.return_value = initial_concept

        refined_concept = {
            "name": "Basic Theme",
            "description": "A profound philosophical concept exploring the deepest human contradictions",
        }
        mock_refine.return_value = refined_concept

        low_scores = ConceptQualityScores(
            relevance=5.0,
            depth=5.0,
            manifestation=5.0,
            resonance=5.0,
            temporal_plausibility=5.0,
        )
        high_scores = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )
        mock_judge.side_effect = [low_scores, high_scores]

        concept, scores, iterations = service.generate_concept_with_quality(
            story_state, existing_names=[]
        )

        assert (
            concept["description"]
            == "A profound philosophical concept exploring the deepest human contradictions"
        )
        assert scores.average >= 7.0
        assert iterations == 2
        mock_refine.assert_called_once()

    @patch.object(WorldQualityService, "_create_concept")
    def test_generate_concept_empty_name_fails(self, mock_create, service, story_state):
        """Test concept generation fails when creation returns empty name."""
        mock_create.return_value = {"description": "No name"}

        with pytest.raises(WorldGenerationError, match="Failed to generate concept"):
            service.generate_concept_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    def test_generate_concept_error_during_iteration(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test concept generation handles errors during iteration."""
        test_concept = {
            "name": "Test Concept",
            "description": "A test concept used to verify error handling behavior during quality iteration",
        }
        mock_create.return_value = test_concept
        mock_judge.side_effect = WorldGenerationError("Judge failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate concept"):
            service.generate_concept_with_quality(story_state, existing_names=[])

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    @patch.object(WorldQualityService, "_refine_concept")
    def test_generate_concept_below_threshold_after_max(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test concept generation returns below threshold after max iterations.

        With best-iteration tracking, returns iteration 1 when all scores are equal.
        """
        test_concept = {
            "name": "Test Concept",
            "description": "A basic and underdeveloped concept lacking in philosophical depth and resonance",
        }
        mock_create.return_value = test_concept
        mock_refine.return_value = test_concept  # Mock refinement to prevent errors

        low_scores = ConceptQualityScores(
            relevance=5.0,
            depth=5.0,
            manifestation=5.0,
            resonance=5.0,
            temporal_plausibility=5.0,
        )
        mock_judge.return_value = low_scores

        concept, scores, iterations = service.generate_concept_with_quality(
            story_state, existing_names=[]
        )

        assert concept["name"] == "Test Concept"
        assert scores.average < 7.0
        # mock_refine returns same entity → unchanged detection breaks loop;
        # Hail-mary creates same entity as best → M3 identical output skip (no extra judge call)
        assert iterations == 1


class TestBatchOperationsPartialFailure:
    """Tests for batch operations with partial failures (warning logs)."""

    @patch.object(WorldQualityService, "generate_faction_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_factions_partial_failure_logs_warning(
        self, mock_record, mock_gen, service, story_state
    ):
        """Test batch faction generation logs warning on partial failure."""
        faction1 = {"name": "Faction One", "description": "First"}
        scores1 = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (faction1, scores1, 1),
            WorldGenerationError("Second failed"),
        ]

        # Should succeed with partial results
        results = service.generate_factions_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 1
        assert results[0][0]["name"] == "Faction One"

    @patch.object(WorldQualityService, "generate_item_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_items_partial_failure_logs_warning(
        self, mock_record, mock_gen, service, story_state
    ):
        """Test batch item generation logs warning on partial failure."""
        item1 = {"name": "Item One", "description": "First"}
        scores1 = ItemQualityScores(
            significance=8.0,
            uniqueness=8.0,
            narrative_potential=8.0,
            integration=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (item1, scores1, 1),
            WorldGenerationError("Second failed"),
            WorldGenerationError("Third failed"),
        ]

        results = service.generate_items_with_quality(
            story_state, name_provider=lambda: [], count=3
        )

        assert len(results) == 1
        assert results[0][0]["name"] == "Item One"

    @patch.object(WorldQualityService, "generate_concept_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_concepts_partial_failure_logs_warning(
        self, mock_record, mock_gen, service, story_state
    ):
        """Test batch concept generation logs warning on partial failure."""
        concept1 = {"name": "Concept One", "description": "First"}
        scores1 = ConceptQualityScores(
            relevance=8.0,
            depth=8.0,
            manifestation=8.0,
            resonance=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (concept1, scores1, 1),
            WorldGenerationError("Second failed"),
        ]

        results = service.generate_concepts_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 1
        assert results[0][0]["name"] == "Concept One"

    @patch.object(WorldQualityService, "generate_location_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_locations_partial_failure_logs_warning(
        self, mock_record, mock_gen, service, story_state
    ):
        """Test batch location generation logs warning on partial failure."""
        loc1 = {"name": "Location One", "description": "First"}
        scores1 = LocationQualityScores(
            atmosphere=8.0,
            significance=8.0,
            story_relevance=8.0,
            distinctiveness=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (loc1, scores1, 1),
            WorldGenerationError("Second failed"),
        ]

        results = service.generate_locations_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 1
        assert results[0][0]["name"] == "Location One"

    @patch.object(WorldQualityService, "_make_model_preparers", return_value=(None, None))
    @patch.object(WorldQualityService, "generate_relationship_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_relationships_partial_failure_logs_warning(
        self, mock_record, mock_gen, _mock_preparers, service, story_state
    ):
        """Test batch relationship generation logs warning on partial failure."""
        rel1 = {"source": "A", "target": "B", "relation_type": "knows", "description": "First"}
        scores1 = RelationshipQualityScores(
            tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0
        )

        mock_gen.side_effect = [
            (rel1, scores1, 1),
            WorldGenerationError("Second failed"),
        ]

        results = service.generate_relationships_with_quality(
            story_state, entity_names_provider=lambda: ["A", "B", "C"], existing_rels=[], count=2
        )

        assert len(results) == 1
        assert results[0][0]["source"] == "A"

    @patch.object(WorldQualityService, "generate_character_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_characters_partial_failure_logs_warning(
        self, mock_record, mock_gen, service, story_state
    ):
        """Test batch character generation logs warning on partial failure."""
        char1 = Character(name="Character One", role="protagonist", description="First")
        scores1 = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
        )

        mock_gen.side_effect = [
            (char1, scores1, 1),
            WorldGenerationError("Second failed"),
        ]

        results = service.generate_characters_with_quality(
            story_state, name_provider=lambda: [], count=2
        )

        assert len(results) == 1
        assert results[0][0].name == "Character One"


class TestCustomInstructions:
    """Tests for custom_instructions parameter coverage."""

    @patch("src.services.world_quality_service._character.generate_structured")
    def test_create_character_prompt_with_custom_instructions(
        self, mock_generate_structured, service, story_state
    ):
        """Test character creation with custom_instructions parameter."""
        mock_creation = CharacterCreation(
            name="Custom Char",
            role="supporting",
            description="A custom character",
            personality_traits=["unique"],
            goals=["custom goal"],
            arc_notes="Custom arc",
        )
        mock_generate_structured.return_value = mock_creation

        result = service._create_character(
            story_state,
            existing_names=[],
            temperature=0.9,
            custom_instructions="Make this character a mysterious wizard.",
        )

        assert result is not None
        assert result.name == "Custom Char"
        # Verify the prompt contained the custom instructions
        call_args = mock_generate_structured.call_args
        assert "SPECIFIC REQUIREMENTS" in call_args[1]["prompt"]
        assert "mysterious wizard" in call_args[1]["prompt"]


class TestFactionIterationRegression:
    """Tests for faction iteration regression path coverage."""

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    @patch.object(WorldQualityService, "_refine_faction")
    def test_faction_iteration_worse_after_peak_returns_best(
        self, mock_refine, mock_judge, mock_create, settings, mock_mode_service, story_state
    ):
        """Test faction returns best iteration when later iterations get worse."""
        # Create service with very high threshold to ensure it never meets
        settings.world_quality_threshold = 9.5
        settings.world_quality_max_iterations = 3
        service = WorldQualityService(settings, mock_mode_service)

        test_faction = {
            "name": "Peak Faction",
            "description": "A powerful and well-organized faction controlling the northern territories of the realm",
            "leader": "Leader",
            "goals": ["goal"],
            "values": ["value"],
            "base_location": "",
        }
        mock_create.return_value = test_faction

        # First judge call - high score (but below threshold)
        high_scores = FactionQualityScores(
            coherence=8.5,
            influence=8.5,
            conflict_potential=8.5,
            distinctiveness=8.5,
            temporal_plausibility=8.5,
        )
        # Second judge call - lower scores (regression)
        low_scores1 = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )
        # Third judge call - even lower scores
        low_scores2 = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=5.0,
            distinctiveness=5.0,
            temporal_plausibility=5.0,
        )

        mock_judge.side_effect = [high_scores, low_scores1, low_scores2]
        mock_refine.return_value = test_faction  # Return same faction for simplicity

        faction, scores, iteration = service.generate_faction_with_quality(
            story_state,
            existing_names=[],
            existing_locations=[],
        )

        # Should return the best iteration (iteration 1) not the last
        assert faction is not None
        assert iteration == 1  # Best iteration was the first one
        assert scores.average >= 8.0  # Should have scores from best iteration

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    def test_faction_scores_none_reconstructed_from_last(
        self, mock_judge, mock_create, settings, mock_mode_service, story_state
    ):
        """Test faction scores are reconstructed when None at end of loop."""
        # Set threshold impossibly high with only 1 iteration
        settings.world_quality_threshold = 10.0  # Impossible
        settings.world_quality_max_iterations = 1
        service = WorldQualityService(settings, mock_mode_service)

        test_faction = {
            "name": "Single Faction",
            "description": "A faction representing the sole iteration through the quality refinement loop",
            "leader": "Leader",
            "goals": ["goal"],
            "values": ["value"],
            "base_location": "",
        }
        mock_create.return_value = test_faction

        # Single judge call with scores below threshold
        scores_below_threshold = FactionQualityScores(
            coherence=6.0,
            influence=6.0,
            conflict_potential=6.0,
            distinctiveness=6.0,
            temporal_plausibility=6.0,
        )
        mock_judge.return_value = scores_below_threshold

        faction, scores, iteration = service.generate_faction_with_quality(
            story_state,
            existing_names=[],
            existing_locations=[],
        )

        # Should return valid faction with reconstructed scores from last iteration
        assert faction is not None
        assert scores is not None
        assert scores.coherence == 6.0
        assert iteration == 1


class TestSuggestRelationshipsForEntity:
    """Tests for suggest_relationships_for_entity async method."""

    @pytest.fixture
    def entity(self):
        """Create test entity."""

        return Entity(
            id="char-001",
            type="character",
            name="Alice",
            description="A brave warrior seeking justice",
            attributes={"role": "protagonist"},
        )

    @pytest.fixture
    def available_entities(self):
        """Create available target entities."""

        return [
            Entity(id="char-001", type="character", name="Alice", description="A brave warrior"),
            Entity(id="char-002", type="character", name="Bob", description="A cunning thief"),
            Entity(id="loc-001", type="location", name="Castle", description="A grand fortress"),
        ]

    @pytest.mark.asyncio
    async def test_suggest_relationships_success(self, service, entity, available_entities):
        """Test successful relationship suggestions."""
        # Mock the LLM response
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_name": "Bob",
                        "relation_type": "ally_of",
                        "description": "They work together",
                        "confidence": 0.85,
                        "bidirectional": True,
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert len(result) == 1
        assert result[0]["target_entity_name"] == "Bob"
        assert result[0]["target_entity_id"] == "char-002"
        assert result[0]["relation_type"] == "ally_of"

    @pytest.mark.asyncio
    async def test_suggest_relationships_no_targets(self, service, entity):
        """Test suggestion when no target entities available."""
        # Only the source entity exists
        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=[entity],  # Only self
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_empty_response(self, service, entity, available_entities):
        """Test suggestion with empty LLM response."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": ""}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_invalid_json(self, service, entity, available_entities):
        """Test suggestion with invalid JSON response."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not valid json"}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_json_returns_list_not_dict(
        self, service, entity, available_entities
    ):
        """Test suggestion when JSON parses to a list instead of dict."""
        # Return valid JSON that's a list, not a dict
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": '["item1", "item2"]'}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Should return empty list because result is not a dict
        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_exception_handled(
        self, service, entity, available_entities
    ):
        """Test suggestion handles exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("LLM error")
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Should return empty list on error
        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_entity_at_max_relationships(
        self, service, entity, available_entities
    ):
        """Test suggestion returns empty when entity already at max relationships."""
        # Set max_relationships_per_entity to 2
        service.settings.max_relationships_per_entity = 2

        # Entity already has 2 relationships
        existing_relationships = [
            {"source_name": "Alice", "target_name": "Bob", "relation_type": "ally_of"},
            {"source_name": "Alice", "target_name": "Castle", "relation_type": "lives_in"},
        ]

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=existing_relationships,
            story_brief=None,
            num_suggestions=3,
        )

        # Should return empty list - entity already at max
        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_relationships_target_resolution_by_id(
        self, service, entity, available_entities
    ):
        """Test suggestion resolves target by ID when provided."""
        # LLM returns target_entity_id
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_id": "loc-001",  # ID provided
                        "target_entity_name": "The Castle",  # Name doesn't match exactly
                        "relation_type": "lives_in",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert len(result) == 1
        assert result[0]["target_entity_id"] == "loc-001"
        assert result[0]["target_entity_name"] == "Castle"  # Should be resolved to actual name

    @pytest.mark.asyncio
    async def test_suggest_relationships_fuzzy_matching(self, service, entity, available_entities):
        """Test suggestion uses fuzzy matching when exact match fails."""
        # LLM returns misspelled name
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_name": "Bobb",  # Misspelled
                        "relation_type": "ally_of",
                        "confidence": 0.8,
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        # Set fuzzy threshold low enough to match "Bobb" -> "Bob"
        service.settings.fuzzy_match_threshold = 0.7

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        assert len(result) == 1
        assert result[0]["target_entity_id"] == "char-002"  # Bob's ID
        assert result[0]["target_entity_name"] == "Bob"

    @pytest.mark.asyncio
    async def test_suggest_relationships_unresolved_target_warning(
        self, service, entity, available_entities, caplog
    ):
        """Test suggestion logs warning for unresolved targets."""
        import logging

        # LLM returns non-existent entity
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_name": "Charlie",  # Doesn't exist
                        "relation_type": "enemy_of",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        with caplog.at_level(logging.WARNING):
            result = await service.suggest_relationships_for_entity(
                entity=entity,
                available_entities=available_entities,
                existing_relationships=[],
                story_brief=None,
                num_suggestions=3,
            )

        # Should return empty - target couldn't be resolved
        assert result == []
        # Warning should be logged
        assert "Could not resolve target entity" in caplog.text

    @pytest.mark.asyncio
    async def test_suggest_relationships_logs_warning_for_unconfigured_entity_type(
        self, service, entity, available_entities, caplog
    ):
        """Test that warning is logged when entity type not in relationship_minimums."""
        import logging

        # Clear the existing minimums to trigger the warning
        service.settings.relationship_minimums = {}

        response_json = json.dumps({"suggestions": []})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        with caplog.at_level(logging.WARNING):
            await service.suggest_relationships_for_entity(
                entity=entity,
                available_entities=available_entities,
                existing_relationships=[],
                story_brief=None,
                num_suggestions=3,
            )

        # Warning should be logged for unconfigured entity type
        assert "No relationship_minimums configured for entity type" in caplog.text

    @pytest.mark.asyncio
    async def test_suggest_relationships_uses_default_role_when_specific_role_missing(
        self, service, entity, available_entities, caplog
    ):
        """Test that 'default' role is used when specific role not in minimums."""
        import logging

        # Set up minimums with only 'default' for character type
        service.settings.relationship_minimums = {"character": {"default": 3}}

        response_json = json.dumps({"suggestions": []})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        with caplog.at_level(logging.DEBUG):
            await service.suggest_relationships_for_entity(
                entity=entity,
                available_entities=available_entities,
                existing_relationships=[],
                story_brief=None,
                num_suggestions=3,
            )

        # Debug message should show default was used
        assert "using default" in caplog.text

    @pytest.mark.asyncio
    async def test_suggest_relationships_uses_fallback_when_no_config(
        self, service, entity, available_entities, caplog
    ):
        """Test that fallback value of 2 is used when no config at all."""
        import logging

        # Set up minimums with entity type but no matching role or default
        service.settings.relationship_minimums = {"character": {"protagonist": 5}}
        # Entity has role="protagonist" but we'll override to something else
        entity.attributes["role"] = "sidekick"  # Not configured

        response_json = json.dumps({"suggestions": []})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        with caplog.at_level(logging.WARNING):
            await service.suggest_relationships_for_entity(
                entity=entity,
                available_entities=available_entities,
                existing_relationships=[],
                story_brief=None,
                num_suggestions=3,
            )

        # Warning should be logged about using fallback
        assert "using fallback value of 2" in caplog.text

    @pytest.mark.asyncio
    async def test_suggest_relationships_uses_specific_role_minimums(
        self, service, entity, available_entities
    ):
        """Test that specific role minimum is used when configured."""
        # Set up minimums with specific role for protagonist
        service.settings.relationship_minimums = {"character": {"protagonist": 5, "default": 2}}
        # Entity has role="protagonist" which matches
        entity.attributes["role"] = "protagonist"

        response_json = json.dumps({"suggestions": []})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        # Should use protagonist minimum (5) not default (2)
        await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Verify prompt was sent - no warning about missing role
        assert mock_client.generate.called

    @pytest.mark.asyncio
    async def test_suggest_relationships_coerces_invalid_confidence_to_default(
        self, service, entity, available_entities
    ):
        """Test that invalid confidence values are coerced to 0.5."""
        # LLM returns invalid confidence that can't be parsed as float
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_id": "char-002",
                        "target_entity_name": "Bob",
                        "relation_type": "ally_of",
                        "confidence": "not_a_number",  # Invalid
                        "bidirectional": True,
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Should have result with confidence coerced to 0.5
        assert len(result) == 1
        assert result[0]["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_suggest_relationships_coerces_string_bidirectional(
        self, service, entity, available_entities
    ):
        """Test that string bidirectional values are coerced to boolean."""
        # LLM returns bidirectional as string "true"
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_id": "char-002",
                        "target_entity_name": "Bob",
                        "relation_type": "ally_of",
                        "confidence": 0.9,
                        "bidirectional": "true",  # String instead of bool
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Should have result with bidirectional coerced to True
        assert len(result) == 1
        assert result[0]["bidirectional"] is True

    @pytest.mark.asyncio
    async def test_suggest_relationships_coerces_unexpected_bidirectional_type(
        self, service, entity, available_entities
    ):
        """Test that unexpected bidirectional types default to False."""
        # LLM returns bidirectional as a list (unexpected type)
        response_json = json.dumps(
            {
                "suggestions": [
                    {
                        "target_entity_id": "char-002",
                        "target_entity_name": "Bob",
                        "relation_type": "ally_of",
                        "confidence": 0.9,
                        "bidirectional": ["unexpected", "list"],  # Unexpected type
                    }
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.suggest_relationships_for_entity(
            entity=entity,
            available_entities=available_entities,
            existing_relationships=[],
            story_brief=None,
            num_suggestions=3,
        )

        # Should have result with bidirectional defaulting to False
        assert len(result) == 1
        assert result[0]["bidirectional"] is False


class TestExtractEntityClaims:
    """Tests for extract_entity_claims async method."""

    @pytest.fixture
    def entity(self):
        """Create test entity."""

        return Entity(
            id="char-001",
            type="character",
            name="Alice",
            description="Alice is 25 years old and has blue eyes. She is the queen of the realm.",
            attributes={"role": "protagonist"},
        )

    @pytest.mark.asyncio
    async def test_extract_claims_success(self, service, entity):
        """Test successful claim extraction."""
        response_json = json.dumps(
            {
                "claims": [
                    "Alice is 25 years old",
                    "Alice has blue eyes",
                    "Alice is the queen of the realm",
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.extract_entity_claims(entity)

        assert len(result) == 3
        assert result[0]["entity_id"] == "char-001"
        assert result[0]["entity_name"] == "Alice"
        assert "25 years old" in result[0]["claim"]

    @pytest.mark.asyncio
    async def test_extract_claims_invalid_json(self, service, entity):
        """Test extraction with invalid JSON response."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not valid json"}
        service._client = mock_client

        result = await service.extract_entity_claims(entity)

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_claims_json_returns_list_not_dict(self, service, entity):
        """Test extraction when JSON parses to a list instead of dict."""
        # Return valid JSON that's a list, not a dict
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": '["claim1", "claim2"]'}
        service._client = mock_client

        result = await service.extract_entity_claims(entity)

        # Should return empty list because result is not a dict
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_claims_exception_handled(self, service, entity):
        """Test extraction handles exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("LLM error")
        service._client = mock_client

        result = await service.extract_entity_claims(entity)

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_claims_filters_non_strings(self, service, entity):
        """Test that non-string claims are filtered out."""
        response_json = json.dumps(
            {
                "claims": [
                    "Valid claim",
                    123,  # Not a string
                    None,  # Not a string
                    "Another valid claim",
                ]
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.extract_entity_claims(entity)

        # Should only have 2 valid string claims
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_extract_claims_handles_non_list_claims(self, service, entity, caplog):
        """Test that non-list claims value is handled with warning."""
        import logging

        # LLM returns claims as a string instead of a list
        response_json = json.dumps({"claims": "This is not a list"})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        with caplog.at_level(logging.WARNING):
            result = await service.extract_entity_claims(entity)

        # Should return empty list
        assert result == []
        # Warning should be logged
        assert "Expected claims to be a list" in caplog.text


class TestCheckContradiction:
    """Tests for check_contradiction async method."""

    @pytest.fixture
    def claim_a(self):
        """Create first test claim."""
        return {
            "entity_id": "char-001",
            "entity_name": "Alice",
            "entity_type": "character",
            "claim": "Alice has blue eyes",
        }

    @pytest.fixture
    def claim_b(self):
        """Create second test claim."""
        return {
            "entity_id": "char-001",
            "entity_name": "Alice",
            "entity_type": "character",
            "claim": "Alice has green eyes",
        }

    @pytest.mark.asyncio
    async def test_check_contradiction_found(self, service, claim_a, claim_b):
        """Test detecting a contradiction."""
        response_json = json.dumps(
            {
                "is_contradiction": True,
                "severity": "high",
                "explanation": "Eye color cannot be both blue and green",
                "resolution_suggestion": "Choose one eye color",
                "confidence": 0.95,
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is not None
        assert result["severity"] == "high"
        assert result["claim_a"] == claim_a
        assert result["claim_b"] == claim_b

    @pytest.mark.asyncio
    async def test_check_contradiction_not_found(self, service, claim_a, claim_b):
        """Test when claims don't contradict."""
        response_json = json.dumps(
            {
                "is_contradiction": False,
                "explanation": "No conflict found",
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_invalid_json(self, service, claim_a, claim_b):
        """Test with invalid JSON response."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not valid json"}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_json_returns_list_not_dict(self, service, claim_a, claim_b):
        """Test when JSON parses to a list instead of dict."""
        # Return valid JSON that's a list, not a dict
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": '["item1", "item2"]'}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        # Should return None because result is not a dict
        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_exception_handled(self, service, claim_a, claim_b):
        """Test exception handling in contradiction check."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("LLM error")
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_coerces_string_is_contradiction(
        self, service, claim_a, claim_b
    ):
        """Test that string is_contradiction values are coerced to boolean."""
        # LLM returns is_contradiction as string "true"
        response_json = json.dumps(
            {
                "is_contradiction": "true",  # String instead of bool
                "severity": "medium",
                "explanation": "Eye color conflict",
                "resolution_suggestion": "Pick one",
                "confidence": 0.8,
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is not None
        assert result["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_check_contradiction_string_false_not_truthy(self, service, claim_a, claim_b):
        """Test that string 'false' is correctly treated as False."""
        # LLM returns is_contradiction as string "false" which would be truthy in Python
        response_json = json.dumps(
            {
                "is_contradiction": "false",  # String "false" should be False
                "explanation": "No conflict",
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        # Should return None because "false" should be coerced to False
        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_coerces_unexpected_type_to_false(
        self, service, claim_a, claim_b
    ):
        """Test that unexpected is_contradiction types default to False."""
        # LLM returns is_contradiction as a list (unexpected type)
        response_json = json.dumps(
            {
                "is_contradiction": ["unexpected"],  # Unexpected type
                "explanation": "Something",
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        # Should return None because unexpected type defaults to False
        assert result is None

    @pytest.mark.asyncio
    async def test_check_contradiction_coerces_invalid_confidence(self, service, claim_a, claim_b):
        """Test that invalid confidence values are coerced to 0.5."""
        # LLM returns invalid confidence that can't be parsed as float
        response_json = json.dumps(
            {
                "is_contradiction": True,
                "severity": "high",
                "explanation": "Conflict",
                "resolution_suggestion": "Fix it",
                "confidence": "not_a_number",  # Invalid
            }
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": response_json}
        service._client = mock_client

        result = await service.check_contradiction(claim_a, claim_b)

        assert result is not None
        assert result["confidence"] == 0.5  # Should be coerced to default


class TestValidateEntityConsistency:
    """Tests for validate_entity_consistency async method."""

    @pytest.fixture
    def entities(self):
        """Create test entities."""

        return [
            Entity(
                id="char-001",
                type="character",
                name="Alice",
                description="Alice is 25 years old",
            ),
            Entity(
                id="char-002",
                type="character",
                name="Bob",
                description="Bob is Alice's brother who is 30 years old",
            ),
        ]

    @pytest.mark.asyncio
    async def test_validate_consistency_finds_contradictions(self, service, entities):
        """Test finding contradictions across entities."""
        # Mock extract_entity_claims to return claims
        extract_response = json.dumps({"claims": ["Claim from entity"]})
        # Mock check_contradiction to return a contradiction
        check_response = json.dumps(
            {
                "is_contradiction": True,
                "severity": "medium",
                "explanation": "Test contradiction",
                "resolution_suggestion": "Fix it",
                "confidence": 0.8,
            }
        )

        mock_client = MagicMock()
        # First two calls are for extract_entity_claims (one per entity)
        # Third call is for check_contradiction
        mock_client.generate.side_effect = [
            {"response": extract_response},
            {"response": extract_response},
            {"response": check_response},
        ]
        service._client = mock_client

        result = await service.validate_entity_consistency(entities, max_comparisons=10)

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_validate_consistency_single_entity(self, service):
        """Test with single entity returns no contradictions."""

        single_entity = [Entity(id="1", type="character", name="A", description="Test")]

        # Mock extract to return 1 claim
        extract_response = json.dumps({"claims": ["Single claim"]})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": extract_response}
        service._client = mock_client

        result = await service.validate_entity_consistency(single_entity, max_comparisons=10)

        # Single claim can't contradict itself
        assert result == []

    @pytest.mark.asyncio
    async def test_validate_consistency_respects_max_comparisons(self, service, entities):
        """Test that max_comparisons limit is respected."""
        # Mock extract to return many claims
        many_claims = json.dumps({"claims": [f"Claim {i}" for i in range(10)]})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": many_claims}
        service._client = mock_client

        # Set max_comparisons very low
        result = await service.validate_entity_consistency(entities, max_comparisons=2)

        # Should complete without error
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_validate_consistency_skips_same_entity_claims(self, service):
        """Test that claims from the same entity are not compared."""

        single_entity = [
            Entity(id="1", type="character", name="A", description="Test A"),
        ]

        # Mock extract to return multiple claims from same entity
        claims = json.dumps({"claims": ["Claim 1", "Claim 2", "Claim 3"]})
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": claims}
        service._client = mock_client

        result = await service.validate_entity_consistency(single_entity, max_comparisons=100)

        # Claims from same entity should not be compared
        # So check_contradiction should never be called
        assert result == []


class TestMakeModelPreparers:
    """Tests for WorldQualityService._make_model_preparers()."""

    def test_same_model_returns_none_pair(self, service):
        """When creator and judge resolve to the same model, return (None, None)."""
        service._model_cache.invalidate()
        with patch(
            "src.services.world_quality_service._model_resolver.resolve_model_pair",
            return_value=("test-model:8b", "test-model:8b"),
        ):
            prep_c, prep_j = service._make_model_preparers("character")

        assert prep_c is None
        assert prep_j is None

    def test_different_models_returns_callables(self, service):
        """When creator and judge differ, return callable preparers."""
        with patch(
            "src.services.world_quality_service._model_resolver.resolve_model_pair",
            return_value=("creator-model:8b", "judge-model:8b"),
        ):
            prep_c, prep_j = service._make_model_preparers("character")

        assert callable(prep_c)
        assert callable(prep_j)

    def test_preparers_call_prepare_model(self, service):
        """Returned preparers delegate to prepare_model with correct model IDs."""
        with (
            patch(
                "src.services.world_quality_service._model_resolver.resolve_model_pair",
                return_value=("creator-model:8b", "judge-model:8b"),
            ),
            patch(
                "src.services.world_quality_service._model_resolver.prepare_model",
            ) as mock_prepare,
        ):
            prep_c, prep_j = service._make_model_preparers("location")

            prep_c()
            mock_prepare.assert_called_once_with(
                service.mode_service, "creator-model:8b", role="creator"
            )

            mock_prepare.reset_mock()
            prep_j()
            mock_prepare.assert_called_once_with(
                service.mode_service, "judge-model:8b", role="judge"
            )

    def test_prepare_creator_graceful_on_failure(self, service):
        """prepare_creator logs warning and continues when prepare_model raises."""
        with (
            patch(
                "src.services.world_quality_service._model_resolver.resolve_model_pair",
                return_value=("creator-model:8b", "judge-model:8b"),
            ),
            patch(
                "src.services.world_quality_service._model_resolver.prepare_model",
                side_effect=ConnectionError("Ollama unreachable"),
            ),
        ):
            prep_c, _ = service._make_model_preparers("character")
            # Should not raise — degrades gracefully with a warning
            prep_c()

    def test_prepare_judge_graceful_on_failure(self, service):
        """prepare_judge logs warning and continues when prepare_model raises."""
        with (
            patch(
                "src.services.world_quality_service._model_resolver.resolve_model_pair",
                return_value=("creator-model:8b", "judge-model:8b"),
            ),
            patch(
                "src.services.world_quality_service._model_resolver.prepare_model",
                side_effect=ValueError("Invalid vram_strategy"),
            ),
        ):
            _, prep_j = service._make_model_preparers("character")
            # Should not raise — degrades gracefully with a warning
            prep_j()

    def test_prepare_creator_graceful_on_vram_allocation_error(self, service):
        """prepare_creator logs warning and continues when VRAMAllocationError is raised."""
        from src.utils.exceptions import VRAMAllocationError

        with (
            patch(
                "src.services.world_quality_service._model_resolver.resolve_model_pair",
                return_value=("creator-model:8b", "judge-model:8b"),
            ),
            patch(
                "src.services.world_quality_service._model_resolver.prepare_model",
                side_effect=VRAMAllocationError("Not enough VRAM for creator"),
            ),
        ):
            prep_c, _ = service._make_model_preparers("character")
            # Should not raise — degrades gracefully with a warning
            prep_c()

    def test_prepare_judge_graceful_on_vram_allocation_error(self, service):
        """prepare_judge logs warning and continues when VRAMAllocationError is raised."""
        from src.utils.exceptions import VRAMAllocationError

        with (
            patch(
                "src.services.world_quality_service._model_resolver.resolve_model_pair",
                return_value=("creator-model:8b", "judge-model:8b"),
            ),
            patch(
                "src.services.world_quality_service._model_resolver.prepare_model",
                side_effect=VRAMAllocationError("Not enough VRAM for judge"),
            ),
        ):
            _, prep_j = service._make_model_preparers("character")
            # Should not raise — degrades gracefully with a warning
            prep_j()
