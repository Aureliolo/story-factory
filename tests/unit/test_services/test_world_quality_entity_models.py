"""Tests for entity-type-specific model selection in WorldQualityService."""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings


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


class TestEntityTypeModelMapping:
    """Test that different entity types use appropriate models."""

    def test_character_uses_writer_model(self, settings):
        """Test that character generation uses writer model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-writer-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for character entity type
        model = service._get_creator_model(entity_type="character")

        # Should call get_model_for_agent with "writer"
        mock_mode_service.get_model_for_agent.assert_called_once_with("writer")
        assert model == "test-writer-model"

    def test_faction_uses_architect_model(self, settings):
        """Test that faction generation uses architect model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-architect-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for faction entity type
        model = service._get_creator_model(entity_type="faction")

        # Should call get_model_for_agent with "architect"
        mock_mode_service.get_model_for_agent.assert_called_once_with("architect")
        assert model == "test-architect-model"

    def test_location_uses_writer_model(self, settings):
        """Test that location generation uses writer model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-writer-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for location entity type
        model = service._get_creator_model(entity_type="location")

        # Should call get_model_for_agent with "writer"
        mock_mode_service.get_model_for_agent.assert_called_once_with("writer")
        assert model == "test-writer-model"

    def test_item_uses_writer_model(self, settings):
        """Test that item generation uses writer model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-writer-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for item entity type
        model = service._get_creator_model(entity_type="item")

        # Should call get_model_for_agent with "writer"
        mock_mode_service.get_model_for_agent.assert_called_once_with("writer")
        assert model == "test-writer-model"

    def test_concept_uses_architect_model(self, settings):
        """Test that concept generation uses architect model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-architect-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for concept entity type
        model = service._get_creator_model(entity_type="concept")

        # Should call get_model_for_agent with "architect"
        mock_mode_service.get_model_for_agent.assert_called_once_with("architect")
        assert model == "test-architect-model"

    def test_relationship_uses_editor_model(self, settings):
        """Test that relationship generation uses editor model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-editor-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for relationship entity type
        model = service._get_creator_model(entity_type="relationship")

        # Should call get_model_for_agent with "editor"
        mock_mode_service.get_model_for_agent.assert_called_once_with("editor")
        assert model == "test-editor-model"

    def test_default_uses_writer_model(self, settings):
        """Test that unknown entity types default to writer model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-writer-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model for unknown entity type
        model = service._get_creator_model(entity_type="unknown_type")

        # Should call get_model_for_agent with "writer" (default)
        mock_mode_service.get_model_for_agent.assert_called_once_with("writer")
        assert model == "test-writer-model"

    def test_no_entity_type_uses_writer_model(self, settings):
        """Test that no entity type defaults to writer model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-writer-model"

        service = WorldQualityService(settings, mock_mode_service)

        # Get model without entity type
        model = service._get_creator_model()

        # Should call get_model_for_agent with "writer" (default)
        mock_mode_service.get_model_for_agent.assert_called_once_with("writer")
        assert model == "test-writer-model"


class TestAnalyticsRecording:
    """Test that analytics correctly record model information."""

    def test_record_entity_quality_with_explicit_model(self, settings):
        """Test that record_entity_quality accepts explicit model_id."""
        mock_mode_service = MagicMock()
        service = WorldQualityService(settings, mock_mode_service)
        service._analytics_db = MagicMock()

        # Record with explicit model_id
        service.record_entity_quality(
            project_id="test-project",
            entity_type="character",
            entity_name="Test Character",
            scores={"average": 8.0, "feedback": "Good"},
            iterations=2,
            generation_time=5.0,
            model_id="explicit-model-id",
        )

        # Should use the explicit model_id
        service._analytics_db.record_world_entity_score.assert_called_once()
        call_args = service._analytics_db.record_world_entity_score.call_args
        assert call_args.kwargs["model_id"] == "explicit-model-id"

    def test_record_entity_quality_infers_model_from_entity_type(self, settings):
        """Test that record_entity_quality infers model from entity_type."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "inferred-faction-model"

        service = WorldQualityService(settings, mock_mode_service)
        service._analytics_db = MagicMock()

        # Record without explicit model_id
        service.record_entity_quality(
            project_id="test-project",
            entity_type="faction",
            entity_name="Test Faction",
            scores={"average": 7.5, "feedback": "Decent"},
            iterations=1,
            generation_time=3.0,
        )

        # Should infer model from entity_type (faction -> architect)
        mock_mode_service.get_model_for_agent.assert_called_once_with("architect")
        service._analytics_db.record_world_entity_score.assert_called_once()
        call_args = service._analytics_db.record_world_entity_score.call_args
        assert call_args.kwargs["model_id"] == "inferred-faction-model"

    def test_different_entity_types_record_different_models(self, settings):
        """Test that different entity types record their respective models."""
        mock_mode_service = MagicMock()

        # Configure different return values for different agent roles
        def mock_get_model(agent_role):
            """Return a mock model ID based on the agent role."""
            model_map = {
                "writer": "writer-model",
                "architect": "architect-model",
                "editor": "editor-model",
            }
            return model_map.get(agent_role, "default-model")

        mock_mode_service.get_model_for_agent.side_effect = mock_get_model

        service = WorldQualityService(settings, mock_mode_service)
        service._analytics_db = MagicMock()

        # Record different entity types
        entity_types_and_expected_models = [
            ("character", "writer-model"),
            ("faction", "architect-model"),
            ("location", "writer-model"),
            ("item", "writer-model"),
            ("concept", "architect-model"),
            ("relationship", "editor-model"),
        ]

        for entity_type, expected_model in entity_types_and_expected_models:
            service._analytics_db.reset_mock()
            mock_mode_service.get_model_for_agent.reset_mock()

            service.record_entity_quality(
                project_id="test-project",
                entity_type=entity_type,
                entity_name=f"Test {entity_type}",
                scores={"average": 8.0},
                iterations=1,
                generation_time=1.0,
            )

            # Verify correct model was recorded
            call_args = service._analytics_db.record_world_entity_score.call_args
            assert call_args.kwargs["model_id"] == expected_model, (
                f"Expected {expected_model} for {entity_type}, got {call_args.kwargs['model_id']}"
            )


class TestJudgeModelSelection:
    """Test that judge model selection works correctly for entity types."""

    def test_judge_model_uses_judge_for_character(self, settings):
        """Test that character judgment uses judge model."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-judge-model"

        service = WorldQualityService(settings, mock_mode_service)

        model = service._get_judge_model(entity_type="character")

        mock_mode_service.get_model_for_agent.assert_called_once_with("judge")
        assert model == "test-judge-model"

    def test_judge_model_uses_judge_for_all_entity_types(self, settings):
        """Test that all entity types use judge for quality evaluation."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-judge-model"

        service = WorldQualityService(settings, mock_mode_service)

        entity_types = ["character", "faction", "location", "item", "concept", "relationship"]
        for entity_type in entity_types:
            mock_mode_service.reset_mock()
            model = service._get_judge_model(entity_type=entity_type)

            mock_mode_service.get_model_for_agent.assert_called_once_with("judge")
            assert model == "test-judge-model"

    def test_judge_model_without_entity_type_uses_judge(self, settings):
        """Test that no entity type defaults to judge for quality evaluation."""
        mock_mode_service = MagicMock()
        mock_mode_service.get_model_for_agent.return_value = "test-judge-model"

        service = WorldQualityService(settings, mock_mode_service)

        model = service._get_judge_model()

        mock_mode_service.get_model_for_agent.assert_called_once_with("judge")
        assert model == "test-judge-model"


class TestClassConstants:
    """Test that class constants are properly defined."""

    def test_entity_creator_roles_defined(self, settings):
        """Test that ENTITY_CREATOR_ROLES constant is defined."""
        assert hasattr(WorldQualityService, "ENTITY_CREATOR_ROLES")
        roles = WorldQualityService.ENTITY_CREATOR_ROLES
        assert roles["character"] == "writer"
        assert roles["faction"] == "architect"
        assert roles["location"] == "writer"
        assert roles["item"] == "writer"
        assert roles["concept"] == "architect"
        assert roles["relationship"] == "editor"

    def test_entity_judge_roles_defined(self, settings):
        """Test that ENTITY_JUDGE_ROLES constant is defined."""
        assert hasattr(WorldQualityService, "ENTITY_JUDGE_ROLES")
        roles = WorldQualityService.ENTITY_JUDGE_ROLES
        # All entity types should use judge for quality evaluation
        for entity_type in ["character", "faction", "location", "item", "concept", "relationship"]:
            assert roles[entity_type] == "judge"
