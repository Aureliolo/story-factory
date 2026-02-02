"""Tests for world quality service entity regeneration methods."""

from unittest.mock import MagicMock

import pytest

from src.memory.entities import Entity
from src.memory.story_state import StoryBrief
from src.services.model_mode_service import ModelModeService
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for tests."""
    settings = MagicMock(spec=Settings)
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_timeout = 120
    settings.world_quality_threshold = 7.0
    settings.world_quality_max_iterations = 3
    settings.world_quality_creator_temp = 0.9
    settings.world_quality_judge_temp = 0.1
    settings.world_quality_refinement_temp = 0.7
    # RefinementConfig.from_settings() required attributes
    settings.world_quality_early_stopping_patience = 2
    settings.world_quality_refinement_temp_start = 0.9
    settings.world_quality_refinement_temp_end = 0.3
    settings.world_quality_refinement_temp_decay = "linear"
    settings.world_quality_early_stopping_min_iterations = 2
    settings.world_quality_early_stopping_variance_tolerance = 0.5
    # _resolve_model_for_role() required attributes
    settings.use_per_agent_models = True
    settings.default_model = "auto"
    settings.agent_models = {
        "writer": "auto",
        "validator": "auto",
        "architect": "auto",
        "judge": "auto",
    }
    return settings


@pytest.fixture
def mock_mode_service():
    """Create mock mode service."""
    service = MagicMock(spec=ModelModeService)
    service.get_model_for_agent.return_value = "test-model:latest"
    return service


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    return MagicMock()


@pytest.fixture
def world_quality_service(mock_settings, mock_mode_service, mock_ollama_client):
    """Create WorldQualityService for tests."""
    service = WorldQualityService(mock_settings, mock_mode_service)
    # Inject the mock client directly
    service._client = mock_ollama_client
    return service


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id="test-entity-1",
        name="Test Character",
        type="character",
        description="A test character for regeneration testing.",
        attributes={
            "role": "protagonist",
            "traits": ["brave", "curious"],
            "quality_scores": {"average": 6.5, "depth": 5.0, "consistency": 8.0},
        },
    )


@pytest.fixture
def sample_story_brief():
    """Create a sample story brief for testing."""
    brief = MagicMock(spec=StoryBrief)
    brief.title = "Test Story"
    brief.genre = "Fantasy"
    brief.themes = ["Adventure", "Friendship"]
    brief.setting = "Medieval Kingdom"
    return brief


class TestRefineEntity:
    """Tests for refine_entity method."""

    @pytest.mark.asyncio
    async def test_refine_entity_returns_none_when_entity_is_none(
        self, world_quality_service, sample_story_brief
    ):
        """Test that refine_entity returns None when entity is None."""
        result = await world_quality_service.refine_entity(
            entity=None, story_brief=sample_story_brief
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_returns_none_when_story_brief_is_none(
        self, world_quality_service, sample_entity
    ):
        """Test that refine_entity returns None when story_brief is None."""
        result = await world_quality_service.refine_entity(entity=sample_entity, story_brief=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_success(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test successful entity refinement."""
        mock_response = {
            "response": '{"name": "Improved Character", "description": "A better description", "attributes": {"role": "protagonist", "traits": ["brave", "wise"]}}'
        }
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.refine_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is not None
        assert result["name"] == "Improved Character"
        assert "description" in result

    @pytest.mark.asyncio
    async def test_refine_entity_handles_empty_response(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that refine_entity handles empty LLM response."""
        mock_response = {"response": ""}
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.refine_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_handles_invalid_json_response(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that refine_entity handles invalid JSON in response."""
        mock_response = {"response": "This is not valid JSON"}
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.refine_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_returns_none_for_list_response(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that refine_entity returns None when response is a list instead of dict."""
        mock_response = {"response": '[{"name": "Item1"}, {"name": "Item2"}]'}
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.refine_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        # Should return None because we expect a dict, not a list
        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_handles_exception(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that refine_entity handles exceptions gracefully."""
        world_quality_service._client.generate.side_effect = Exception("LLM error")

        result = await world_quality_service.refine_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_builds_feedback_from_low_scores(
        self, world_quality_service, sample_story_brief
    ):
        """Test that refine_entity builds feedback from low quality scores."""
        entity = Entity(
            id="test-entity",
            name="Test",
            type="character",
            description="Test description",
            attributes={
                "quality_scores": {
                    "average": 5.0,
                    "depth": 4.0,  # Low score - should be in feedback
                    "consistency": 8.0,  # High score - should not be in feedback
                }
            },
        )

        mock_response = {"response": '{"name": "Test", "description": "Better", "attributes": {}}'}
        world_quality_service._client.generate.return_value = mock_response

        await world_quality_service.refine_entity(
            entity=entity,
            story_brief=sample_story_brief,
        )

        # Check that generate was called with prompt containing feedback
        call_args = world_quality_service._client.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "depth" in prompt.lower()  # Low score mentioned


class TestRegenerateEntity:
    """Tests for regenerate_entity method."""

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_when_entity_is_none(
        self, world_quality_service, sample_story_brief
    ):
        """Test that regenerate_entity returns None when entity is None."""
        result = await world_quality_service.regenerate_entity(
            entity=None, story_brief=sample_story_brief
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_when_story_brief_is_none(
        self, world_quality_service, sample_entity
    ):
        """Test that regenerate_entity returns None when story_brief is None."""
        result = await world_quality_service.regenerate_entity(
            entity=sample_entity, story_brief=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_success(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test successful entity regeneration."""
        mock_response = {
            "response": '{"name": "New Character", "description": "A completely new description", "attributes": {"role": "hero", "traits": ["determined"]}}'
        }
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.regenerate_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is not None
        assert result["name"] == "New Character"

    @pytest.mark.asyncio
    async def test_regenerate_entity_with_custom_instructions(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test regeneration with custom instructions."""
        mock_response = {
            "response": '{"name": "Customized", "description": "Based on guidance", "attributes": {}}'
        }
        custom_guidance = "Make this character more mysterious"
        world_quality_service._client.generate.return_value = mock_response

        await world_quality_service.regenerate_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
            custom_instructions=custom_guidance,
        )

        # Check that custom instructions were included in prompt
        call_args = world_quality_service._client.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "mysterious" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_empty_response(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that regenerate_entity handles empty LLM response."""
        mock_response = {"response": ""}
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.regenerate_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_exception(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that regenerate_entity handles exceptions gracefully."""
        world_quality_service._client.generate.side_effect = Exception("LLM error")

        result = await world_quality_service.regenerate_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_entity_without_attributes(
        self, world_quality_service, sample_story_brief
    ):
        """Test regeneration of entity with no attributes."""
        entity = Entity(
            id="test-entity",
            name="Test",
            type="location",
            description="A test location",
            attributes={},
        )

        mock_response = {
            "response": '{"name": "New Location", "description": "A better place", "attributes": {"atmosphere": "mysterious"}}'
        }
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.regenerate_entity(
            entity=entity,
            story_brief=sample_story_brief,
        )

        assert result is not None
        assert result["name"] == "New Location"

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_for_list_response(
        self, world_quality_service, sample_entity, sample_story_brief
    ):
        """Test that regenerate_entity returns None when response is a list instead of dict."""
        mock_response = {"response": '[{"name": "Item1"}, {"name": "Item2"}]'}
        world_quality_service._client.generate.return_value = mock_response

        result = await world_quality_service.regenerate_entity(
            entity=sample_entity,
            story_brief=sample_story_brief,
        )

        # Should return None because we expect a dict, not a list
        assert result is None
