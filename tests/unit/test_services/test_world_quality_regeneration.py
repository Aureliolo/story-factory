"""Tests for world quality service entity regeneration methods."""

from unittest.mock import MagicMock, patch

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
    settings.world_quality_threshold = 7.0
    settings.world_quality_max_iterations = 3
    settings.world_quality_creator_temp = 0.9
    settings.world_quality_judge_temp = 0.1
    settings.world_quality_refinement_temp = 0.7
    return settings


@pytest.fixture
def mock_mode_service():
    """Create mock mode service."""
    return MagicMock(spec=ModelModeService)


@pytest.fixture
def world_quality_service(mock_settings, mock_mode_service):
    """Create WorldQualityService for tests."""
    return WorldQualityService(mock_settings, mock_mode_service)


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
        self, world_quality_service, sample_story_brief, mock_settings
    ):
        """Test that refine_entity returns None when entity is None."""
        result = await world_quality_service.refine_entity(
            entity=None, story_brief=sample_story_brief, settings=mock_settings
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_returns_none_when_story_brief_is_none(
        self, world_quality_service, sample_entity, mock_settings
    ):
        """Test that refine_entity returns None when story_brief is None."""
        result = await world_quality_service.refine_entity(
            entity=sample_entity, story_brief=None, settings=mock_settings
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_success(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test successful entity refinement."""
        mock_response = {
            "response": '{"name": "Improved Character", "description": "A better description", "attributes": {"role": "protagonist", "traits": ["brave", "wise"]}}'
        }

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.refine_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is not None
            assert result["name"] == "Improved Character"
            assert "description" in result

    @pytest.mark.asyncio
    async def test_refine_entity_handles_empty_response(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that refine_entity handles empty LLM response."""
        mock_response = {"response": ""}

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.refine_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_handles_invalid_json_response(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that refine_entity handles invalid JSON in response."""
        mock_response = {"response": "This is not valid JSON"}

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.refine_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_handles_exception(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that refine_entity handles exceptions gracefully."""
        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.side_effect = Exception("LLM error")

            result = await world_quality_service.refine_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_refine_entity_builds_feedback_from_low_scores(
        self, world_quality_service, sample_story_brief, mock_settings
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

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            await world_quality_service.refine_entity(
                entity=entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            # Check that generate was called with prompt containing feedback
            call_args = mock_client.generate.call_args
            prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
            assert "depth" in prompt.lower()  # Low score mentioned


class TestRegenerateEntity:
    """Tests for regenerate_entity method."""

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_when_entity_is_none(
        self, world_quality_service, sample_story_brief, mock_settings
    ):
        """Test that regenerate_entity returns None when entity is None."""
        result = await world_quality_service.regenerate_entity(
            entity=None, story_brief=sample_story_brief, settings=mock_settings
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_when_story_brief_is_none(
        self, world_quality_service, sample_entity, mock_settings
    ):
        """Test that regenerate_entity returns None when story_brief is None."""
        result = await world_quality_service.regenerate_entity(
            entity=sample_entity, story_brief=None, settings=mock_settings
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_success(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test successful entity regeneration."""
        mock_response = {
            "response": '{"name": "New Character", "description": "A completely new description", "attributes": {"role": "hero", "traits": ["determined"]}}'
        }

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.regenerate_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is not None
            assert result["name"] == "New Character"

    @pytest.mark.asyncio
    async def test_regenerate_entity_with_custom_instructions(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test regeneration with custom instructions."""
        mock_response = {
            "response": '{"name": "Customized", "description": "Based on guidance", "attributes": {}}'
        }
        custom_guidance = "Make this character more mysterious"

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            await world_quality_service.regenerate_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
                custom_instructions=custom_guidance,
            )

            # Check that custom instructions were included in prompt
            call_args = mock_client.generate.call_args
            prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
            assert "mysterious" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_empty_response(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that regenerate_entity handles empty LLM response."""
        mock_response = {"response": ""}

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.regenerate_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_exception(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that regenerate_entity handles exceptions gracefully."""
        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.side_effect = Exception("LLM error")

            result = await world_quality_service.regenerate_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_regenerate_entity_handles_entity_without_attributes(
        self, world_quality_service, sample_story_brief, mock_settings
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

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.regenerate_entity(
                entity=entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            assert result is not None
            assert result["name"] == "New Location"

    @pytest.mark.asyncio
    async def test_regenerate_entity_returns_none_for_list_response(
        self, world_quality_service, sample_entity, sample_story_brief, mock_settings
    ):
        """Test that regenerate_entity returns None when response is a list instead of dict."""
        mock_response = {"response": '[{"name": "Item1"}, {"name": "Item2"}]'}

        with patch.object(world_quality_service, "client", create=True) as mock_client:
            mock_client.generate.return_value = mock_response

            result = await world_quality_service.regenerate_entity(
                entity=sample_entity,
                story_brief=sample_story_brief,
                settings=mock_settings,
            )

            # Should return None because we expect a dict, not a list
            assert result is None
