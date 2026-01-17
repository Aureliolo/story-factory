"""Tests for WorldQualityService - multi-model iteration for world building quality."""

import json
from unittest.mock import MagicMock, patch

import ollama
import pytest

from memory.story_state import Character, StoryBrief, StoryState
from memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RelationshipQualityScores,
)
from services.world_quality_service import WorldQualityService
from settings import Settings
from utils.exceptions import WorldGenerationError


@pytest.fixture
def settings():
    """Create settings with test values."""
    return Settings(
        ollama_url="http://localhost:11434",
        ollama_timeout=60,
        world_quality_enabled=True,
        world_quality_max_iterations=3,
        world_quality_threshold=7.0,
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        llm_tokens_character_create=500,
        llm_tokens_character_judge=300,
        llm_tokens_character_refine=500,
        llm_tokens_location_create=400,
        llm_tokens_location_judge=300,
        llm_tokens_location_refine=400,
        llm_tokens_faction_create=400,
        llm_tokens_faction_judge=300,
        llm_tokens_faction_refine=400,
        llm_tokens_item_create=400,
        llm_tokens_item_judge=300,
        llm_tokens_item_refine=400,
        llm_tokens_concept_create=400,
        llm_tokens_concept_judge=300,
        llm_tokens_concept_refine=400,
        llm_tokens_relationship_create=400,
        llm_tokens_relationship_judge=300,
        llm_tokens_relationship_refine=400,
        llm_tokens_mini_description=100,
        mini_description_words_max=15,
    )


@pytest.fixture
def mock_mode_service():
    """Create mock mode service."""
    mode_service = MagicMock()
    mode_service.get_model_for_agent.return_value = "test-model"
    return mode_service


@pytest.fixture
def service(settings, mock_mode_service):
    """Create WorldQualityService with mocked dependencies."""
    return WorldQualityService(settings, mock_mode_service)


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
            feedback="Good character, needs more flaws",
        )
        result = scores.to_dict()
        assert result["depth"] == 8.0
        assert result["goals"] == 7.0
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
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "goals" in weak
        assert "flaws" in weak
        assert "arc_potential" in weak
        assert "depth" not in weak
        assert "uniqueness" not in weak

    def test_fields_are_required(self):
        """Test that all score fields are required (no defaults)."""
        from pydantic import ValidationError

        # Creating without required fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CharacterQualityScores()  # type: ignore[call-arg]

        # Should have errors for all 5 score fields
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert missing_fields == {"depth", "goals", "flaws", "uniqueness", "arc_potential"}


class TestLocationQualityScores:
    """Tests for LocationQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.0,
            story_relevance=6.0,
            distinctiveness=9.0,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=5.0,
            story_relevance=6.0,
            distinctiveness=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "significance" in weak
        assert "story_relevance" in weak
        assert "atmosphere" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = LocationQualityScores(
            atmosphere=8.0,
            significance=7.0,
            story_relevance=6.0,
            distinctiveness=9.0,
            feedback="Add more sensory details",
        )
        result = scores.to_dict()
        assert result["atmosphere"] == 8.0
        assert result["average"] == 7.5
        assert result["feedback"] == "Add more sensory details"


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
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = FactionQualityScores(
            coherence=6.0,
            influence=5.0,
            conflict_potential=8.0,
            distinctiveness=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "coherence" in weak
        assert "influence" in weak
        assert "conflict_potential" not in weak
        assert "distinctiveness" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=7.0,
            conflict_potential=6.0,
            distinctiveness=9.0,
            feedback="More internal structure needed",
        )
        result = scores.to_dict()
        assert result["coherence"] == 8.0
        assert result["average"] == 7.5
        assert result["feedback"] == "More internal structure needed"


class TestItemQualityScores:
    """Tests for ItemQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=9.0,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = ItemQualityScores(
            significance=6.0,
            uniqueness=5.0,
            narrative_potential=8.0,
            integration=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "significance" in weak
        assert "uniqueness" in weak
        assert "narrative_potential" not in weak
        assert "integration" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = ItemQualityScores(
            significance=8.0,
            uniqueness=7.0,
            narrative_potential=6.0,
            integration=9.0,
            feedback="More history needed",
        )
        result = scores.to_dict()
        assert result["significance"] == 8.0
        assert result["average"] == 7.5
        assert result["feedback"] == "More history needed"


class TestConceptQualityScores:
    """Tests for ConceptQualityScores model."""

    def test_average_calculation(self):
        """Test average score calculation."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=7.0,
            manifestation=6.0,
            resonance=9.0,
        )
        assert scores.average == 7.5

    def test_weak_dimensions(self):
        """Test finding dimensions below threshold."""
        scores = ConceptQualityScores(
            relevance=6.0,
            depth=5.0,
            manifestation=8.0,
            resonance=9.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "relevance" in weak
        assert "depth" in weak
        assert "manifestation" not in weak
        assert "resonance" not in weak

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = ConceptQualityScores(
            relevance=8.0,
            depth=7.0,
            manifestation=6.0,
            resonance=9.0,
            feedback="More philosophical depth",
        )
        result = scores.to_dict()
        assert result["relevance"] == 8.0
        assert result["average"] == 7.5
        assert result["feedback"] == "More philosophical depth"


class TestRefinementConfig:
    """Tests for RefinementConfig model."""

    def test_from_settings(self, settings):
        """Test creating config from settings."""
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
        assert config.quality_threshold == 7.0

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
        client = service.client
        assert client is not None
        # Second access returns same client
        assert service.client is client

    def test_analytics_db_creation(self, service):
        """Test lazy analytics database creation."""
        assert service._analytics_db is None
        with patch("services.world_quality_service.ModeDatabase") as mock_db_class:
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db
            db = service.analytics_db
            assert db is mock_db
            # Second access returns same instance
            assert service.analytics_db is mock_db

    def test_get_creator_model(self, service, mock_mode_service):
        """Test getting creator model."""
        model = service._get_creator_model()
        mock_mode_service.get_model_for_agent.assert_called_with("writer")
        assert model == "test-model"

    def test_get_judge_model(self, service, mock_mode_service):
        """Test getting judge model."""
        model = service._get_judge_model()
        mock_mode_service.get_model_for_agent.assert_called_with("validator")
        assert model == "test-model"


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

    def test_create_character_success(self, service, story_state, mock_ollama_client):
        """Test successful character creation."""
        character_json = json.dumps(
            {
                "name": "Dr. Eleanor Grey",
                "role": "protagonist",
                "description": "A brilliant detective haunted by past failures",
                "personality_traits": ["observant", "determined", "secretive"],
                "goals": ["solve the case", "find redemption"],
                "relationships": {},
                "arc_notes": "Will learn to trust others",
            }
        )
        mock_ollama_client.generate.return_value = {"response": character_json}
        service._client = mock_ollama_client

        character = service._create_character(story_state, existing_names=[], temperature=0.9)

        assert character is not None
        assert character.name == "Dr. Eleanor Grey"
        assert character.role == "protagonist"
        assert "observant" in character.personality_traits

    def test_create_character_with_existing_names(self, service, story_state, mock_ollama_client):
        """Test character creation avoids existing names."""
        character_json = json.dumps(
            {
                "name": "New Character",
                "role": "supporting",
                "description": "A mysterious stranger",
                "personality_traits": ["quiet"],
                "goals": ["unknown"],
                "relationships": {},
                "arc_notes": "Will reveal secrets",
            }
        )
        mock_ollama_client.generate.return_value = {"response": character_json}
        service._client = mock_ollama_client

        character = service._create_character(
            story_state, existing_names=["John Doe", "Jane Doe"], temperature=0.9
        )

        assert character is not None
        # Verify prompt includes existing names
        call_args = mock_ollama_client.generate.call_args
        assert "John Doe" in call_args.kwargs["prompt"]
        assert "Jane Doe" in call_args.kwargs["prompt"]

    def test_create_character_no_brief_returns_none(self, service):
        """Test character creation returns None without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_character(state, existing_names=[], temperature=0.9)
        assert result is None

    def test_create_character_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test character creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not valid json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid character"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    def test_create_character_ollama_error(self, service, story_state, mock_ollama_client):
        """Test character creation handles Ollama errors."""
        mock_ollama_client.generate.side_effect = ollama.ResponseError("Model not found")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    def test_create_character_connection_error(self, service, story_state, mock_ollama_client):
        """Test character creation handles connection errors."""
        mock_ollama_client.generate.side_effect = ConnectionError("Connection refused")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    def test_create_character_timeout_error(self, service, story_state, mock_ollama_client):
        """Test character creation handles timeout errors."""
        mock_ollama_client.generate.side_effect = TimeoutError("Request timed out")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    def test_create_character_unexpected_error(self, service, story_state, mock_ollama_client):
        """Test character creation handles unexpected errors."""
        mock_ollama_client.generate.side_effect = RuntimeError("Unexpected error")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Unexpected"):
            service._create_character(story_state, existing_names=[], temperature=0.9)

    def test_create_character_non_dict_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test character creation raises error when JSON is not a dict."""
        mock_ollama_client.generate.return_value = {"response": '["not", "a", "dict"]'}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid character JSON"):
            service._create_character(story_state, existing_names=[], temperature=0.9)


class TestJudgeCharacterQuality:
    """Tests for _judge_character_quality method."""

    def test_judge_character_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful character quality judgment."""
        scores_json = json.dumps(
            {
                "depth": 8.0,
                "goals": 7.5,
                "flaws": 7.0,
                "uniqueness": 8.5,
                "arc_potential": 8.0,
                "feedback": "Strong character with good depth",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

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

    def test_judge_character_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "depth": 8.0,
                "goals": 7.5,
                # Missing: flaws, uniqueness, arc_potential
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        character = Character(name="Test", role="supporting", description="Test")

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_character_quality(character, story_state, temperature=0.1)

    def test_judge_character_quality_no_brief_uses_default_genre(self, service, mock_ollama_client):
        """Test judge uses default genre when brief is missing."""
        scores_json = json.dumps(
            {
                "depth": 7.0,
                "goals": 7.0,
                "flaws": 7.0,
                "uniqueness": 7.0,
                "arc_potential": 7.0,
                "feedback": "Decent character",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

        state = StoryState(id="test-id")
        state.brief = None
        character = Character(name="Test", role="supporting", description="Test")

        scores = service._judge_character_quality(character, state, temperature=0.1)

        assert scores.average == 7.0
        # Verify prompt uses "fiction" as default genre
        call_args = mock_ollama_client.generate.call_args
        assert "fiction" in call_args.kwargs["prompt"]

    def test_judge_character_quality_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error on invalid JSON response."""
        mock_ollama_client.generate.return_value = {"response": "not json at all"}
        service._client = mock_ollama_client

        character = Character(name="Test", role="supporting", description="Test")

        with pytest.raises(WorldGenerationError, match="Failed to extract JSON"):
            service._judge_character_quality(character, story_state, temperature=0.1)

    def test_judge_character_quality_exception_reraises(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge reraises WorldGenerationError as-is."""
        mock_ollama_client.generate.side_effect = RuntimeError("Some error")
        service._client = mock_ollama_client

        character = Character(name="Test", role="supporting", description="Test")

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_character_quality(character, story_state, temperature=0.1)


class TestRefineCharacter:
    """Tests for _refine_character method."""

    def test_refine_character_success(self, service, story_state, mock_ollama_client):
        """Test successful character refinement."""
        refined_json = json.dumps(
            {
                "name": "John Doe",
                "role": "protagonist",
                "description": "A more complex description with deeper psychology",
                "personality_traits": ["brave", "conflicted", "hopeful"],
                "goals": ["save the world", "overcome inner demons"],
                "relationships": {"Mary": "ally"},
                "arc_notes": "Will transform from bitter loner to trusting friend",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original_char = Character(
            name="John Doe",
            role="protagonist",
            description="A simple description",
            personality_traits=["brave"],
            goals=["save the world"],
            arc_notes="Basic arc",
        )
        scores = CharacterQualityScores(
            depth=5.0,
            goals=6.0,
            flaws=4.0,
            uniqueness=5.5,
            arc_potential=5.0,
            feedback="Needs more depth",
        )

        refined = service._refine_character(original_char, scores, story_state, temperature=0.7)

        assert refined.name == "John Doe"
        assert "deeper psychology" in refined.description
        assert len(refined.personality_traits) > 1

    def test_refine_character_keeps_original_on_missing_fields(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement keeps original values when new response is missing fields."""
        # JSON missing some optional fields
        partial_json = json.dumps(
            {
                "name": "John Doe",
                "role": "protagonist",
                "description": "New description",
                # Missing personality_traits, goals, relationships, arc_notes
            }
        )
        mock_ollama_client.generate.return_value = {"response": partial_json}
        service._client = mock_ollama_client

        original_char = Character(
            name="John Doe",
            role="protagonist",
            description="Old description",
            personality_traits=["brave", "kind"],
            goals=["original goal"],
            relationships={"Alice": "friend"},
            arc_notes="Original arc",
        )
        scores = CharacterQualityScores(
            depth=6.0, goals=6.0, flaws=6.0, uniqueness=6.0, arc_potential=6.0
        )

        refined = service._refine_character(original_char, scores, story_state, temperature=0.7)

        assert refined.name == "John Doe"
        assert refined.description == "New description"
        # Original values should be preserved for missing fields
        assert refined.personality_traits == ["brave", "kind"]
        assert refined.goals == ["original goal"]
        assert refined.relationships == {"Alice": "friend"}
        assert refined.arc_notes == "Original arc"

    def test_refine_character_includes_weak_dimensions_in_prompt(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement prompt includes weak dimensions."""
        refined_json = json.dumps(
            {
                "name": "Test",
                "role": "supporting",
                "description": "Refined",
                "personality_traits": [],
                "goals": [],
                "relationships": {},
                "arc_notes": "",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=5.0,  # Below 7.0 threshold
            goals=8.0,
            flaws=4.0,  # Below 7.0 threshold
            uniqueness=9.0,
            arc_potential=5.0,  # Below 7.0 threshold
        )

        service._refine_character(original_char, scores, story_state, temperature=0.7)

        call_args = mock_ollama_client.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "depth" in prompt
        assert "flaws" in prompt
        assert "arc_potential" in prompt

    def test_refine_character_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not valid json"}
        service._client = mock_ollama_client

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0, goals=6.0, flaws=6.0, uniqueness=6.0, arc_potential=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid character refinement"):
            service._refine_character(original_char, scores, story_state, temperature=0.7)

    def test_refine_character_ollama_error(self, service, story_state, mock_ollama_client):
        """Test refinement handles Ollama errors."""
        mock_ollama_client.generate.side_effect = ollama.ResponseError("Error")
        service._client = mock_ollama_client

        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0, goals=6.0, flaws=6.0, uniqueness=6.0, arc_potential=6.0
        )

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._refine_character(original_char, scores, story_state, temperature=0.7)

    def test_refine_character_with_no_brief_uses_english(self, service, mock_ollama_client):
        """Test refinement uses English when brief is missing."""
        refined_json = json.dumps(
            {
                "name": "Test",
                "role": "supporting",
                "description": "Refined",
                "personality_traits": [],
                "goals": [],
                "relationships": {},
                "arc_notes": "",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        state = StoryState(id="test-id")
        state.brief = None
        original_char = Character(name="Test", role="supporting", description="Test")
        scores = CharacterQualityScores(
            depth=6.0, goals=6.0, flaws=6.0, uniqueness=6.0, arc_potential=6.0
        )

        service._refine_character(original_char, scores, state, temperature=0.7)

        call_args = mock_ollama_client.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "English" in prompt


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
            depth=8.0, goals=8.0, flaws=7.5, uniqueness=8.0, arc_potential=8.5
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
            depth=5.0, goals=5.0, flaws=4.0, uniqueness=6.0, arc_potential=5.0
        )
        high_scores = CharacterQualityScores(
            depth=8.0, goals=8.0, flaws=7.5, uniqueness=8.0, arc_potential=8.5
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
    def test_generate_character_returns_below_threshold_after_max_iterations(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test character returns below threshold if max iterations exceeded."""
        test_char = Character(name="Low Quality", role="supporting", description="Basic")
        mock_create.return_value = test_char

        low_scores = CharacterQualityScores(
            depth=5.0, goals=5.0, flaws=5.0, uniqueness=5.0, arc_potential=5.0
        )
        mock_judge.return_value = low_scores

        char, scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        assert char.name == "Low Quality"
        assert scores.average < 7.0
        assert iterations == 3  # max_iterations

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

    def test_create_location_success(self, service, story_state, mock_ollama_client):
        """Test successful location creation."""
        location_json = json.dumps(
            {
                "name": "Thornwood Manor",
                "type": "location",
                "description": "A crumbling Victorian mansion shrouded in mist",
                "significance": "Central setting where the mystery unfolds",
            }
        )
        mock_ollama_client.generate.return_value = {"response": location_json}
        service._client = mock_ollama_client

        location = service._create_location(story_state, existing_names=[], temperature=0.9)

        assert location["name"] == "Thornwood Manor"
        assert location["type"] == "location"
        assert "Victorian" in location["description"]

    def test_create_location_no_brief_returns_empty(self, service):
        """Test location creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_location(state, existing_names=[], temperature=0.9)
        assert result == {}

    def test_create_location_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test location creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid location"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    def test_create_location_ollama_error(self, service, story_state, mock_ollama_client):
        """Test location creation handles Ollama errors."""
        mock_ollama_client.generate.side_effect = ollama.ResponseError("Error")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_location(story_state, existing_names=[], temperature=0.9)

    def test_create_location_non_dict_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test location creation raises error when JSON is not a dict."""
        mock_ollama_client.generate.return_value = {"response": '["a", "list"]'}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid location JSON"):
            service._create_location(story_state, existing_names=[], temperature=0.9)


class TestJudgeLocationQuality:
    """Tests for _judge_location_quality method."""

    def test_judge_location_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful location quality judgment."""
        scores_json = json.dumps(
            {
                "atmosphere": 8.0,
                "significance": 7.5,
                "story_relevance": 8.0,
                "distinctiveness": 8.5,
                "feedback": "Rich atmosphere, could be more distinctive",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

        location = {
            "name": "Dark Forest",
            "description": "A mysterious forest",
            "significance": "Important for plot",
        }

        scores = service._judge_location_quality(location, story_state, temperature=0.1)

        assert scores.atmosphere == 8.0
        assert scores.average == 8.0

    def test_judge_location_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "atmosphere": 8.0,
                # Missing other fields
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        location = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_location_quality(location, story_state, temperature=0.1)

    def test_judge_location_quality_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        location = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="Failed to extract JSON"):
            service._judge_location_quality(location, story_state, temperature=0.1)


class TestRefineLocation:
    """Tests for _refine_location method."""

    def test_refine_location_success(self, service, story_state, mock_ollama_client):
        """Test successful location refinement."""
        refined_json = json.dumps(
            {
                "name": "Dark Forest",
                "type": "location",
                "description": "A deeply atmospheric ancient forest",
                "significance": "Central to the story's themes",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original = {"name": "Dark Forest", "description": "A forest", "significance": "Unknown"}
        scores = LocationQualityScores(
            atmosphere=5.0, significance=5.0, story_relevance=6.0, distinctiveness=5.0
        )

        refined = service._refine_location(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Dark Forest"
        assert "atmospheric" in refined["description"]

    def test_refine_location_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0, significance=6.0, story_relevance=6.0, distinctiveness=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid location refinement"):
            service._refine_location(original, scores, story_state, temperature=0.7)

    def test_refine_location_ollama_error(self, service, story_state, mock_ollama_client):
        """Test refinement handles Ollama errors."""
        mock_ollama_client.generate.side_effect = ConnectionError("Connection refused")
        service._client = mock_ollama_client

        original = {"name": "Test", "description": "Test", "significance": "Test"}
        scores = LocationQualityScores(
            atmosphere=6.0, significance=6.0, story_relevance=6.0, distinctiveness=6.0
        )

        with pytest.raises(WorldGenerationError, match="LLM error"):
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
            "description": "A crumbling Victorian mansion shrouded in mist",
            "significance": "Central setting where the mystery unfolds",
        }
        mock_create.return_value = test_loc

        high_scores = LocationQualityScores(
            atmosphere=9.0, significance=8.0, story_relevance=8.5, distinctiveness=8.0
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
        initial_loc = {"name": "Basic Place", "description": "Plain", "significance": "None"}
        mock_create.return_value = initial_loc

        refined_loc = {
            "name": "Basic Place",
            "description": "Richly detailed location",
            "significance": "Important",
        }
        mock_refine.return_value = refined_loc

        low_scores = LocationQualityScores(
            atmosphere=5.0, significance=5.0, story_relevance=5.0, distinctiveness=5.0
        )
        high_scores = LocationQualityScores(
            atmosphere=8.0, significance=8.0, story_relevance=8.0, distinctiveness=8.0
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


class TestIsDuplicateRelationship:
    """Tests for _is_duplicate_relationship method."""

    def test_duplicate_same_direction(self, service):
        """Test detecting duplicate in same direction."""
        existing = [("Alice", "Bob"), ("Charlie", "Diana")]
        assert service._is_duplicate_relationship("Alice", "Bob", "knows", existing)

    def test_duplicate_reverse_direction(self, service):
        """Test detecting duplicate in reverse direction."""
        existing = [("Alice", "Bob")]
        assert service._is_duplicate_relationship("Bob", "Alice", "knows", existing)

    def test_not_duplicate(self, service):
        """Test that unrelated pairs are not duplicates."""
        existing = [("Alice", "Bob")]
        assert not service._is_duplicate_relationship("Charlie", "Diana", "knows", existing)

    def test_empty_existing_list(self, service):
        """Test with empty existing list."""
        assert not service._is_duplicate_relationship("Alice", "Bob", "knows", [])


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


class TestJudgeRelationshipQuality:
    """Tests for _judge_relationship_quality method."""

    def test_judge_relationship_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful relationship quality judgment."""
        scores_json = json.dumps(
            {
                "tension": 8.0,
                "dynamics": 7.5,
                "story_potential": 8.0,
                "authenticity": 8.5,
                "feedback": "Strong dynamic, more conflict potential",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

        relationship = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "rivals",
            "description": "Long-standing rivalry",
        }

        scores = service._judge_relationship_quality(relationship, story_state, temperature=0.1)

        assert scores.tension == 8.0
        assert scores.average == 8.0

    def test_judge_relationship_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "tension": 8.0,
                # Missing other fields
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        relationship = {"source": "A", "target": "B", "relation_type": "knows", "description": "X"}

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_relationship_quality(relationship, story_state, temperature=0.1)


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
                existing_rels=[("Alice", "Bob")],  # Already exists
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

    def test_create_faction_success(self, service, story_state, mock_ollama_client):
        """Test successful faction creation."""
        faction_json = json.dumps(
            {
                "name": "The Shadow Council",
                "type": "faction",
                "description": "A secret society manipulating events from the shadows",
                "leader": "The Grand Master",
                "goals": ["control the kingdom", "gather ancient artifacts"],
                "values": ["secrecy", "power"],
            }
        )
        mock_ollama_client.generate.return_value = {"response": faction_json}
        service._client = mock_ollama_client

        faction = service._create_faction(story_state, existing_names=[], temperature=0.9)

        assert faction["name"] == "The Shadow Council"
        assert faction["leader"] == "The Grand Master"
        assert len(faction["goals"]) == 2

    def test_create_faction_no_brief_returns_empty(self, service):
        """Test faction creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_faction(state, existing_names=[], temperature=0.9)
        assert result == {}

    def test_create_faction_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test faction creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid faction"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)

    def test_create_faction_ollama_error(self, service, story_state, mock_ollama_client):
        """Test faction creation handles Ollama errors."""
        mock_ollama_client.generate.side_effect = TimeoutError("Timeout")
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="LLM error"):
            service._create_faction(story_state, existing_names=[], temperature=0.9)


class TestJudgeFactionQuality:
    """Tests for _judge_faction_quality method."""

    def test_judge_faction_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful faction quality judgment."""
        scores_json = json.dumps(
            {
                "coherence": 8.0,
                "influence": 7.5,
                "conflict_potential": 8.0,
                "distinctiveness": 8.5,
                "feedback": "Strong faction with clear identity",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

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

    def test_judge_faction_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "coherence": 8.0,
                # Missing other fields
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        faction = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_faction_quality(faction, story_state, temperature=0.1)


class TestRefineFaction:
    """Tests for _refine_faction method."""

    def test_refine_faction_success(self, service, story_state, mock_ollama_client):
        """Test successful faction refinement."""
        refined_json = json.dumps(
            {
                "name": "Test Guild",
                "type": "faction",
                "description": "A deeply influential guild with rich history",
                "leader": "The Grand Master",
                "goals": ["dominate trade", "expand influence"],
                "values": ["profit", "loyalty"],
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original = {
            "name": "Test Guild",
            "description": "A guild",
            "leader": "Boss",
            "goals": ["make money"],
            "values": ["money"],
        }
        scores = FactionQualityScores(
            coherence=5.0, influence=5.0, conflict_potential=6.0, distinctiveness=5.0
        )

        refined = service._refine_faction(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Test Guild"
        assert "influential" in refined["description"]

    def test_refine_faction_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        original = {"name": "Test", "description": "Test", "leader": "X", "goals": [], "values": []}
        scores = FactionQualityScores(
            coherence=6.0, influence=6.0, conflict_potential=6.0, distinctiveness=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid faction refinement"):
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
            "description": "A powerful secret society",
            "leader": "Grand Master",
            "goals": ["control"],
            "values": ["power"],
        }
        mock_create.return_value = test_faction

        high_scores = FactionQualityScores(
            coherence=8.0, influence=8.0, conflict_potential=8.0, distinctiveness=8.0
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

    def test_create_item_success(self, service, story_state, mock_ollama_client):
        """Test successful item creation."""
        item_json = json.dumps(
            {
                "name": "The Crimson Amulet",
                "type": "item",
                "description": "An ancient amulet that glows with inner light",
                "significance": "Key to unlocking the mansion's secrets",
                "properties": ["glows in darkness", "warm to touch"],
            }
        )
        mock_ollama_client.generate.return_value = {"response": item_json}
        service._client = mock_ollama_client

        item = service._create_item(story_state, existing_names=[], temperature=0.9)

        assert item["name"] == "The Crimson Amulet"
        assert len(item["properties"]) == 2

    def test_create_item_no_brief_returns_empty(self, service):
        """Test item creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_item(state, existing_names=[], temperature=0.9)
        assert result == {}

    def test_create_item_invalid_json_raises_error(self, service, story_state, mock_ollama_client):
        """Test item creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid item"):
            service._create_item(story_state, existing_names=[], temperature=0.9)


class TestJudgeItemQuality:
    """Tests for _judge_item_quality method."""

    def test_judge_item_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful item quality judgment."""
        scores_json = json.dumps(
            {
                "significance": 8.0,
                "uniqueness": 7.5,
                "narrative_potential": 8.0,
                "integration": 8.5,
                "feedback": "Strong item with good story potential",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

        item = {
            "name": "Magic Sword",
            "description": "A powerful weapon",
            "significance": "Key to victory",
            "properties": ["sharp"],
        }

        scores = service._judge_item_quality(item, story_state, temperature=0.1)

        assert scores.significance == 8.0
        assert scores.average == 8.0

    def test_judge_item_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "significance": 8.0,
                # Missing other fields
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        item = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_item_quality(item, story_state, temperature=0.1)


class TestRefineItem:
    """Tests for _refine_item method."""

    def test_refine_item_success(self, service, story_state, mock_ollama_client):
        """Test successful item refinement."""
        refined_json = json.dumps(
            {
                "name": "Magic Sword",
                "type": "item",
                "description": "A legendary blade with a storied past",
                "significance": "Central to the hero's journey",
                "properties": ["cuts through anything", "glows in battle"],
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original = {
            "name": "Magic Sword",
            "description": "A sword",
            "significance": "Important",
            "properties": ["sharp"],
        }
        scores = ItemQualityScores(
            significance=5.0, uniqueness=5.0, narrative_potential=6.0, integration=5.0
        )

        refined = service._refine_item(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Magic Sword"
        assert "legendary" in refined["description"]

    def test_refine_item_invalid_json_raises_error(self, service, story_state, mock_ollama_client):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        original = {"name": "Test", "description": "Test", "significance": "X", "properties": []}
        scores = ItemQualityScores(
            significance=6.0, uniqueness=6.0, narrative_potential=6.0, integration=6.0
        )

        with pytest.raises(WorldGenerationError, match="Invalid item refinement"):
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
            "description": "A key made of pure crystal",
            "significance": "Opens the final door",
            "properties": ["unbreakable"],
        }
        mock_create.return_value = test_item

        high_scores = ItemQualityScores(
            significance=8.0, uniqueness=8.0, narrative_potential=8.0, integration=8.0
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

    def test_create_concept_success(self, service, story_state, mock_ollama_client):
        """Test successful concept creation."""
        concept_json = json.dumps(
            {
                "name": "The Price of Truth",
                "type": "concept",
                "description": "Truth always comes with consequences that challenge the seeker",
                "manifestations": "Characters face moral dilemmas when uncovering secrets",
            }
        )
        mock_ollama_client.generate.return_value = {"response": concept_json}
        service._client = mock_ollama_client

        concept = service._create_concept(story_state, existing_names=[], temperature=0.9)

        assert concept["name"] == "The Price of Truth"
        assert concept["type"] == "concept"

    def test_create_concept_no_brief_returns_empty(self, service):
        """Test concept creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_concept(state, existing_names=[], temperature=0.9)
        assert result == {}

    def test_create_concept_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test concept creation raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        with pytest.raises(WorldGenerationError, match="Invalid concept"):
            service._create_concept(story_state, existing_names=[], temperature=0.9)


class TestJudgeConceptQuality:
    """Tests for _judge_concept_quality method."""

    def test_judge_concept_quality_success(self, service, story_state, mock_ollama_client):
        """Test successful concept quality judgment."""
        scores_json = json.dumps(
            {
                "relevance": 8.0,
                "depth": 7.5,
                "manifestation": 8.0,
                "resonance": 8.5,
                "feedback": "Strong thematic concept",
            }
        )
        mock_ollama_client.generate.return_value = {"response": scores_json}
        service._client = mock_ollama_client

        concept = {
            "name": "Redemption",
            "description": "The journey from darkness to light",
            "manifestations": "Through character arcs",
        }

        scores = service._judge_concept_quality(concept, story_state, temperature=0.1)

        assert scores.relevance == 8.0
        assert scores.average == 8.0

    def test_judge_concept_quality_missing_fields_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test judge raises error when response is missing required fields."""
        incomplete_json = json.dumps(
            {
                "relevance": 8.0,
                # Missing other fields
            }
        )
        mock_ollama_client.generate.return_value = {"response": incomplete_json}
        service._client = mock_ollama_client

        concept = {"name": "Test", "description": "Test"}

        with pytest.raises(WorldGenerationError, match="missing required fields"):
            service._judge_concept_quality(concept, story_state, temperature=0.1)


class TestRefineConcept:
    """Tests for _refine_concept method."""

    def test_refine_concept_success(self, service, story_state, mock_ollama_client):
        """Test successful concept refinement."""
        refined_json = json.dumps(
            {
                "name": "Redemption",
                "type": "concept",
                "description": "A profound journey through moral complexity",
                "manifestations": "Evident in every character's transformation",
            }
        )
        mock_ollama_client.generate.return_value = {"response": refined_json}
        service._client = mock_ollama_client

        original = {
            "name": "Redemption",
            "description": "Getting better",
            "manifestations": "Characters change",
        }
        scores = ConceptQualityScores(relevance=5.0, depth=5.0, manifestation=6.0, resonance=5.0)

        refined = service._refine_concept(original, scores, story_state, temperature=0.7)

        assert refined["name"] == "Redemption"
        assert "profound" in refined["description"]

    def test_refine_concept_invalid_json_raises_error(
        self, service, story_state, mock_ollama_client
    ):
        """Test refinement raises error on invalid JSON."""
        mock_ollama_client.generate.return_value = {"response": "not json"}
        service._client = mock_ollama_client

        original = {"name": "Test", "description": "Test", "manifestations": "X"}
        scores = ConceptQualityScores(relevance=6.0, depth=6.0, manifestation=6.0, resonance=6.0)

        with pytest.raises(WorldGenerationError, match="Invalid concept refinement"):
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
            "description": "The tension between honesty and allegiance",
            "manifestations": "Characters must choose between truth and friends",
        }
        mock_create.return_value = test_concept

        high_scores = ConceptQualityScores(
            relevance=8.0, depth=8.0, manifestation=8.0, resonance=8.0
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
            depth=8.0, goals=8.0, flaws=7.5, uniqueness=8.0, arc_potential=8.5
        )
        scores2 = CharacterQualityScores(
            depth=7.5, goals=8.0, flaws=8.0, uniqueness=7.5, arc_potential=8.0
        )

        mock_gen.side_effect = [
            (char1, scores1, 1),
            (char2, scores2, 2),
        ]

        results = service.generate_characters_with_quality(story_state, existing_names=[], count=2)

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
            depth=8.0, goals=8.0, flaws=7.5, uniqueness=8.0, arc_potential=8.5
        )

        mock_gen.side_effect = [
            (char1, scores1, 1),
            WorldGenerationError("Failed"),
        ]

        results = service.generate_characters_with_quality(story_state, existing_names=[], count=2)

        assert len(results) == 1
        assert results[0][0].name == "Character One"

    @patch.object(WorldQualityService, "generate_character_with_quality")
    def test_generate_characters_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch character generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any characters"):
            service.generate_characters_with_quality(story_state, existing_names=[], count=2)

    @patch.object(WorldQualityService, "generate_location_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_locations_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch location generation."""
        loc1 = {"name": "Location One", "description": "First"}
        loc2 = {"name": "Location Two", "description": "Second"}
        scores1 = LocationQualityScores(
            atmosphere=8.0, significance=8.0, story_relevance=8.0, distinctiveness=8.0
        )
        scores2 = LocationQualityScores(
            atmosphere=7.5, significance=8.0, story_relevance=8.0, distinctiveness=7.5
        )

        mock_gen.side_effect = [
            (loc1, scores1, 1),
            (loc2, scores2, 2),
        ]

        results = service.generate_locations_with_quality(story_state, existing_names=[], count=2)

        assert len(results) == 2
        assert results[0][0]["name"] == "Location One"
        assert results[1][0]["name"] == "Location Two"

    @patch.object(WorldQualityService, "generate_location_with_quality")
    def test_generate_locations_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch location generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any locations"):
            service.generate_locations_with_quality(story_state, existing_names=[], count=2)

    @patch.object(WorldQualityService, "generate_faction_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_factions_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch faction generation."""
        faction1 = {"name": "Faction One", "description": "First"}
        faction2 = {"name": "Faction Two", "description": "Second"}
        scores1 = FactionQualityScores(
            coherence=8.0, influence=8.0, conflict_potential=8.0, distinctiveness=8.0
        )
        scores2 = FactionQualityScores(
            coherence=7.5, influence=8.0, conflict_potential=8.0, distinctiveness=7.5
        )

        mock_gen.side_effect = [
            (faction1, scores1, 1),
            (faction2, scores2, 2),
        ]

        results = service.generate_factions_with_quality(story_state, existing_names=[], count=2)

        assert len(results) == 2
        assert results[0][0]["name"] == "Faction One"
        assert results[1][0]["name"] == "Faction Two"

    @patch.object(WorldQualityService, "generate_faction_with_quality")
    def test_generate_factions_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch faction generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any factions"):
            service.generate_factions_with_quality(story_state, existing_names=[], count=2)

    @patch.object(WorldQualityService, "generate_item_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_items_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch item generation."""
        item1 = {"name": "Item One", "description": "First"}
        item2 = {"name": "Item Two", "description": "Second"}
        scores1 = ItemQualityScores(
            significance=8.0, uniqueness=8.0, narrative_potential=8.0, integration=8.0
        )
        scores2 = ItemQualityScores(
            significance=7.5, uniqueness=8.0, narrative_potential=8.0, integration=7.5
        )

        mock_gen.side_effect = [
            (item1, scores1, 1),
            (item2, scores2, 2),
        ]

        results = service.generate_items_with_quality(story_state, existing_names=[], count=2)

        assert len(results) == 2
        assert results[0][0]["name"] == "Item One"
        assert results[1][0]["name"] == "Item Two"

    @patch.object(WorldQualityService, "generate_item_with_quality")
    def test_generate_items_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch item generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any items"):
            service.generate_items_with_quality(story_state, existing_names=[], count=2)

    @patch.object(WorldQualityService, "generate_concept_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_concepts_with_quality(self, mock_record, mock_gen, service, story_state):
        """Test batch concept generation."""
        concept1 = {"name": "Concept One", "description": "First"}
        concept2 = {"name": "Concept Two", "description": "Second"}
        scores1 = ConceptQualityScores(relevance=8.0, depth=8.0, manifestation=8.0, resonance=8.0)
        scores2 = ConceptQualityScores(relevance=7.5, depth=8.0, manifestation=8.0, resonance=7.5)

        mock_gen.side_effect = [
            (concept1, scores1, 1),
            (concept2, scores2, 2),
        ]

        results = service.generate_concepts_with_quality(story_state, existing_names=[], count=2)

        assert len(results) == 2
        assert results[0][0]["name"] == "Concept One"
        assert results[1][0]["name"] == "Concept Two"

    @patch.object(WorldQualityService, "generate_concept_with_quality")
    def test_generate_concepts_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch concept generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any concepts"):
            service.generate_concepts_with_quality(story_state, existing_names=[], count=2)

    @patch.object(WorldQualityService, "generate_relationship_with_quality")
    @patch.object(WorldQualityService, "record_entity_quality")
    def test_generate_relationships_with_quality(self, mock_record, mock_gen, service, story_state):
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
            story_state, entity_names=["A", "B", "C"], existing_rels=[], count=2
        )

        assert len(results) == 2
        assert results[0][0]["source"] == "A"
        assert results[1][0]["source"] == "B"

    @patch.object(WorldQualityService, "generate_relationship_with_quality")
    def test_generate_relationships_all_fail_raises_error(self, mock_gen, service, story_state):
        """Test batch relationship generation raises error when all fail."""
        mock_gen.side_effect = WorldGenerationError("All failed")

        with pytest.raises(WorldGenerationError, match="Failed to generate any relationships"):
            service.generate_relationships_with_quality(
                story_state, entity_names=["A", "B"], existing_rels=[], count=2
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

    def test_generate_mini_description_llm_called_for_long_text(self, service, mock_ollama_client):
        """Test that LLM is called for long descriptions."""
        long_description = " ".join(["word"] * 50)  # 50 words
        mock_ollama_client.generate.return_value = {"response": "A concise summary of the entity"}
        service._client = mock_ollama_client

        result = service.generate_mini_description(
            name="Test Entity",
            entity_type="character",
            full_description=long_description,
        )

        assert result == "A concise summary of the entity"
        mock_ollama_client.generate.assert_called_once()

    def test_generate_mini_description_strips_quotes(self, service, mock_ollama_client):
        """Test that quotes are stripped from response."""
        mock_ollama_client.generate.return_value = {"response": '"A quoted summary"'}
        service._client = mock_ollama_client

        long_description = " ".join(["word"] * 50)
        result = service.generate_mini_description(
            name="Test",
            entity_type="character",
            full_description=long_description,
        )

        assert result == "A quoted summary"

    def test_generate_mini_description_truncates_long_response(
        self, service, settings, mock_ollama_client
    ):
        """Test that overly long responses are truncated."""
        # Response with more words than max + 3
        long_response = " ".join(["word"] * 50)
        mock_ollama_client.generate.return_value = {"response": long_response}
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
        mock_ollama_client.generate.side_effect = Exception("LLM error")
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

    def test_generate_mini_descriptions_batch(self, service, mock_ollama_client):
        """Test batch mini description generation."""
        mock_ollama_client.generate.return_value = {"response": "Short summary"}
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
        assert mock_ollama_client.generate.call_count == 2


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
            depth=5.0, goals=5.0, flaws=5.0, uniqueness=5.0, arc_potential=5.0
        )
        scores_iter2 = CharacterQualityScores(
            depth=6.0, goals=6.0, flaws=6.0, uniqueness=6.0, arc_potential=6.0
        )
        scores_iter3 = CharacterQualityScores(
            depth=8.0, goals=8.0, flaws=8.0, uniqueness=8.0, arc_potential=8.0
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
    def test_location_generation_returns_below_threshold_after_max(
        self, mock_judge, mock_create, service, story_state
    ):
        """Test location returned even below threshold after max iterations."""
        test_loc = {"name": "Basic", "description": "Simple location"}
        mock_create.return_value = test_loc

        low_scores = LocationQualityScores(
            atmosphere=5.0, significance=5.0, story_relevance=5.0, distinctiveness=5.0
        )
        mock_judge.return_value = low_scores

        loc, scores, iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert loc["name"] == "Basic"
        assert scores.average < 7.0
        assert iterations == 3  # max_iterations

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
            depth=8.0, goals=8.0, flaws=8.0, uniqueness=8.0, arc_potential=8.0
        )

        mock_gen.side_effect = [
            (char1, scores, 1),
            (char2, scores, 1),
        ]

        service.generate_characters_with_quality(story_state, existing_names=["Existing"], count=2)

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
