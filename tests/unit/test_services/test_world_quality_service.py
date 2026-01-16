"""Tests for WorldQualityService - multi-model iteration for world building quality."""

from unittest.mock import MagicMock, patch

import pytest

from memory.story_state import Character, StoryBrief, StoryState
from memory.world_quality import (
    CharacterQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RelationshipQualityScores,
)
from services.world_quality_service import WorldQualityService
from settings import Settings


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
        import pytest
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


class TestWorldQualityService:
    """Tests for WorldQualityService."""

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

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    def test_generate_character_with_quality_meets_threshold(
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
    def test_generate_character_with_quality_needs_refinement(
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

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    def test_generate_location_with_quality(self, mock_judge, mock_create, service, story_state):
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

    def test_generate_without_brief_raises_error(self, service):
        """Test that generation without brief raises error."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="must have a brief"):
            service.generate_character_with_quality(state, existing_names=[])

    @patch.object(WorldQualityService, "generate_character_with_quality")
    def test_batch_character_generation(self, mock_gen, service, story_state):
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
        # Verify both calls were made
        assert mock_gen.call_count == 2
        # Verify quality scores are returned correctly
        assert results[0][1].average == scores1.average
        assert results[1][1].average == scores2.average


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
