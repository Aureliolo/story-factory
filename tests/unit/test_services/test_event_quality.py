"""Tests for event generation, judgment, and refinement quality loop.

Tests cover:
- _create_event: success, empty description, LLM failure, no brief
- _judge_event_quality: success, multi-call averaging, error handling
- _refine_event: success, temporal field preservation, participant preservation, error
- generate_event_with_quality: full quality loop wiring
"""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import (
    StoryBrief,
    StoryState,
    WorldEventCreation,
)
from src.memory.world_quality import EventQualityScores
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def settings():
    """Create settings with test values."""
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
    )


@pytest.fixture
def mock_mode_service():
    """Create mock mode service."""
    mode_service = MagicMock()
    mode_service.get_model_for_agent.return_value = "test-model:8b"
    return mode_service


@pytest.fixture
def service(settings, mock_mode_service):
    """Create WorldQualityService with mocked dependencies."""
    svc = WorldQualityService(settings, mock_mode_service)
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


class TestCreateEvent:
    """Tests for _create_event function."""

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_create_event_success(self, mock_generate_structured, service, story_state):
        """Test successful event creation returns a dict with description."""
        mock_event = WorldEventCreation(
            description="The Great Fire consumed the eastern wing of Thornwood Manor",
            year=1847,
            era_name="Victorian Era",
            participants=[
                {"entity_name": "Lord Thornwood", "role": "affected"},
                {"entity_name": "Thornwood Manor", "role": "location"},
            ],
            consequences=["The east wing was sealed off", "Insurance fraud suspected"],
        )
        mock_generate_structured.return_value = mock_event

        result = service._create_event(
            story_state, existing_descriptions=[], entity_context="Test context", temperature=0.9
        )

        assert (
            result["description"] == "The Great Fire consumed the eastern wing of Thornwood Manor"
        )
        assert result["year"] == 1847
        assert result["era_name"] == "Victorian Era"
        assert len(result["participants"]) == 2
        assert len(result["consequences"]) == 2

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_create_event_empty_description_returns_empty(
        self, mock_generate_structured, service, story_state
    ):
        """Test event creation returns empty dict when description is empty."""
        mock_event = WorldEventCreation(
            description="",
            year=None,
            era_name="",
            participants=[],
            consequences=[],
        )
        mock_generate_structured.return_value = mock_event

        result = service._create_event(
            story_state, existing_descriptions=[], entity_context="Test context", temperature=0.9
        )

        assert result == {}

    def test_create_event_no_brief_returns_empty(self, service):
        """Test event creation returns empty dict without brief."""
        state = StoryState(id="test-id")
        state.brief = None

        result = service._create_event(
            state, existing_descriptions=[], entity_context="Test context", temperature=0.9
        )
        assert result == {}

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_create_event_llm_failure_raises_error(
        self, mock_generate_structured, service, story_state
    ):
        """Test event creation raises WorldGenerationError on LLM failure."""
        mock_generate_structured.side_effect = ConnectionError("Connection refused")

        with pytest.raises(WorldGenerationError, match="Event creation failed"):
            service._create_event(
                story_state,
                existing_descriptions=[],
                entity_context="Test context",
                temperature=0.9,
            )

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_create_event_validation_error_raises(
        self, mock_generate_structured, service, story_state
    ):
        """Test event creation raises error on validation failure."""
        from pydantic import ValidationError

        mock_generate_structured.side_effect = ValidationError.from_exception_data(
            "WorldEventCreation", [{"type": "missing", "loc": ("description",), "input": {}}]
        )

        with pytest.raises(WorldGenerationError, match="Event creation failed"):
            service._create_event(
                story_state,
                existing_descriptions=[],
                entity_context="Test context",
                temperature=0.9,
            )

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_create_event_includes_existing_descriptions_in_prompt(
        self, mock_generate_structured, service, story_state
    ):
        """Test that existing descriptions are passed to prevent duplication."""
        mock_event = WorldEventCreation(
            description="A new unique event",
            year=1850,
        )
        mock_generate_structured.return_value = mock_event

        service._create_event(
            story_state,
            existing_descriptions=["The Great Fire", "The Flood"],
            entity_context="Test context",
            temperature=0.9,
        )

        # Verify the prompt includes existing descriptions
        call_args = mock_generate_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "The Great Fire" in prompt
        assert "The Flood" in prompt
        assert "DO NOT DUPLICATE" in prompt


class TestJudgeEventQuality:
    """Tests for _judge_event_quality function."""

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_judge_event_quality_success(self, mock_generate_structured, service, story_state):
        """Test successful event quality judgment."""
        mock_generate_structured.return_value = EventQualityScores(
            significance=8.0,
            temporal_plausibility=7.5,
            causal_coherence=8.0,
            narrative_potential=8.5,
            entity_integration=7.0,
            feedback="Strong event with good narrative potential",
        )

        event = {
            "description": "The Great Fire of Thornwood",
            "year": 1847,
            "era_name": "Victorian Era",
            "participants": [{"entity_name": "Lord Thornwood", "role": "affected"}],
            "consequences": ["Manor sealed off"],
        }

        scores = service._judge_event_quality(event, story_state, temperature=0.1)

        assert scores.significance == 8.0
        assert scores.temporal_plausibility == 7.5
        assert scores.causal_coherence == 8.0
        assert scores.narrative_potential == 8.5
        assert scores.entity_integration == 7.0
        assert scores.average == pytest.approx(7.8)

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_judge_event_quality_error_raises(self, mock_generate_structured, service, story_state):
        """Test judge raises error on failure."""
        mock_generate_structured.side_effect = Exception("Generation failed")

        event = {"description": "Test Event", "year": 1847}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            service._judge_event_quality(event, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_judge_event_quality_multi_call_warning(
        self, mock_generate_structured, settings, mock_mode_service, story_state
    ):
        """Test judge logs warning (not error) when multi_call is enabled and call fails."""
        settings.judge_consistency_enabled = True
        settings.judge_multi_call_enabled = True
        svc = WorldQualityService(settings, mock_mode_service)
        svc._analytics_db = MagicMock()

        mock_generate_structured.side_effect = Exception("LLM timeout")
        event = {"description": "Some Event", "year": 1900}

        with pytest.raises(WorldGenerationError, match="judgment failed"):
            svc._judge_event_quality(event, story_state, temperature=0.1)

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_judge_event_quality_handles_missing_participants(
        self, mock_generate_structured, service, story_state
    ):
        """Test judge handles events with no participants."""
        mock_generate_structured.return_value = EventQualityScores(
            significance=7.0,
            temporal_plausibility=7.0,
            causal_coherence=7.0,
            narrative_potential=7.0,
            entity_integration=5.0,
            feedback="No participants listed",
        )

        event = {"description": "An earthquake", "year": 1800}

        scores = service._judge_event_quality(event, story_state, temperature=0.1)
        assert scores.entity_integration == 5.0


class TestRefineEvent:
    """Tests for _refine_event function."""

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_refine_event_success(self, mock_generate_structured, service, story_state):
        """Test successful event refinement."""
        refined_event = WorldEventCreation(
            description="The Great Fire devastated the eastern wing of Thornwood Manor",
            year=1847,
            month=10,
            era_name="Victorian Era",
            participants=[
                {"entity_name": "Lord Thornwood", "role": "affected"},
                {"entity_name": "Mrs. Blackwood", "role": "witness"},
            ],
            consequences=[
                "The east wing was permanently sealed",
                "Insurance fraud investigation launched",
                "Family reputation tarnished",
            ],
        )
        mock_generate_structured.return_value = refined_event

        original = {
            "description": "The Great Fire at Thornwood",
            "year": 1847,
            "era_name": "Victorian Era",
            "participants": [{"entity_name": "Lord Thornwood", "role": "affected"}],
            "consequences": ["Manor damaged"],
        }

        scores = EventQualityScores(
            significance=6.0,
            temporal_plausibility=7.0,
            causal_coherence=5.0,
            narrative_potential=6.0,
            entity_integration=5.0,
            feedback="Needs more detail and consequences",
        )

        result = service._refine_event(original, scores, story_state, temperature=0.7)

        assert "devastated" in result["description"]
        assert result["year"] == 1847
        assert len(result["participants"]) == 2
        assert len(result["consequences"]) == 3

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_refine_event_preserves_temporal_fields(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement preserves temporal fields when refined version drops them."""
        refined_event = WorldEventCreation(
            description="Improved event description",
            year=None,  # Dropped temporal fields
            month=None,
            era_name="",
            participants=[],
            consequences=["Better consequence"],
        )
        mock_generate_structured.return_value = refined_event

        original = {
            "description": "Original event",
            "year": 1847,
            "month": 10,
            "era_name": "Victorian Era",
            "participants": [],
            "consequences": [],
        }

        scores = EventQualityScores(
            significance=5.0,
            temporal_plausibility=5.0,
            causal_coherence=5.0,
            narrative_potential=5.0,
            entity_integration=5.0,
        )

        result = service._refine_event(original, scores, story_state, temperature=0.7)

        # Temporal fields should be preserved from original
        assert result["year"] == 1847
        assert result["month"] == 10
        assert result["era_name"] == "Victorian Era"

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_refine_event_preserves_participants(
        self, mock_generate_structured, service, story_state
    ):
        """Test refinement preserves participants when refined version drops them."""
        refined_event = WorldEventCreation(
            description="Improved event description",
            year=1847,
            era_name="Victorian Era",
            participants=[],  # Empty participants
            consequences=["Better consequence"],
        )
        mock_generate_structured.return_value = refined_event

        original_participants = [
            {"entity_name": "Lord Thornwood", "role": "actor"},
            {"entity_name": "Thornwood Manor", "role": "location"},
        ]
        original = {
            "description": "Original event",
            "year": 1847,
            "era_name": "Victorian Era",
            "participants": original_participants,
            "consequences": [],
        }

        scores = EventQualityScores(
            significance=5.0,
            temporal_plausibility=5.0,
            causal_coherence=5.0,
            narrative_potential=5.0,
            entity_integration=5.0,
        )

        result = service._refine_event(original, scores, story_state, temperature=0.7)

        # Participants should be preserved from original
        assert result["participants"] == original_participants

    @patch("src.services.world_quality_service._event.generate_structured")
    def test_refine_event_error_raises(self, mock_generate_structured, service, story_state):
        """Test refinement raises WorldGenerationError on failure."""
        mock_generate_structured.side_effect = ConnectionError("Connection refused")

        original = {"description": "Test", "year": 1847}
        scores = EventQualityScores(
            significance=5.0,
            temporal_plausibility=5.0,
            causal_coherence=5.0,
            narrative_potential=5.0,
            entity_integration=5.0,
        )

        with pytest.raises(WorldGenerationError, match="Event refinement failed"):
            service._refine_event(original, scores, story_state, temperature=0.7)


class TestGenerateEventWithQuality:
    """Tests for the full quality loop integration."""

    @patch("src.services.world_quality_service._event._refine_event")
    @patch("src.services.world_quality_service._event._judge_event_quality")
    @patch("src.services.world_quality_service._event._create_event")
    def test_generate_event_passes_on_first_iteration(
        self, mock_create, mock_judge, mock_refine, service, story_state
    ):
        """Test event passes quality on first iteration without refinement."""
        mock_create.return_value = {
            "description": "The Great Fire",
            "year": 1847,
            "era_name": "Victorian Era",
            "participants": [{"entity_name": "Lord Thornwood", "role": "affected"}],
            "consequences": ["Manor damaged"],
        }
        mock_judge.return_value = EventQualityScores(
            significance=8.0,
            temporal_plausibility=8.0,
            causal_coherence=8.0,
            narrative_potential=8.0,
            entity_integration=8.0,
            feedback="Excellent event",
        )

        event, scores, iterations = service.generate_event_with_quality(
            story_state,
            existing_descriptions=[],
            entity_context="Test context",
        )

        assert event["description"] == "The Great Fire"
        assert scores.average == 8.0
        assert iterations == 1
        mock_refine.assert_not_called()

    @patch("src.services.world_quality_service._event._refine_event")
    @patch("src.services.world_quality_service._event._judge_event_quality")
    @patch("src.services.world_quality_service._event._create_event")
    def test_generate_event_refines_below_threshold(
        self, mock_create, mock_judge, mock_refine, service, story_state
    ):
        """Test event gets refined when below quality threshold."""
        mock_create.return_value = {
            "description": "A vague event happened",
            "year": 1847,
            "participants": [],
            "consequences": [],
        }
        # First judge: below threshold; second judge: above threshold
        mock_judge.side_effect = [
            EventQualityScores(
                significance=5.0,
                temporal_plausibility=5.0,
                causal_coherence=5.0,
                narrative_potential=5.0,
                entity_integration=5.0,
                feedback="Too vague",
            ),
            EventQualityScores(
                significance=8.0,
                temporal_plausibility=8.0,
                causal_coherence=8.0,
                narrative_potential=8.0,
                entity_integration=8.0,
                feedback="Much better",
            ),
        ]
        mock_refine.return_value = {
            "description": "A detailed and significant event",
            "year": 1847,
            "participants": [{"entity_name": "Hero", "role": "actor"}],
            "consequences": ["World changed"],
        }

        _event, scores, iterations = service.generate_event_with_quality(
            story_state,
            existing_descriptions=[],
            entity_context="Test context",
        )

        assert scores.average == 8.0
        assert iterations == 2
        mock_refine.assert_called_once()

    def test_generate_event_no_brief_raises_error(self, service):
        """Test event generation raises ValueError without a brief."""
        state = StoryState(id="test-id")
        state.brief = None

        with pytest.raises(ValueError, match="Story must have a brief"):
            service.generate_event_with_quality(
                state,
                existing_descriptions=[],
                entity_context="Test context",
            )

    @patch("src.services.world_quality_service._event._create_event")
    def test_generate_event_creation_failure_raises(self, mock_create, service, story_state):
        """Test that creation failure propagates as WorldGenerationError."""
        mock_create.side_effect = WorldGenerationError("Event creation failed: timeout")

        with pytest.raises(WorldGenerationError, match="Event creation failed"):
            service.generate_event_with_quality(
                story_state,
                existing_descriptions=[],
                entity_context="Test context",
            )


class TestGenerateEventsWithQualityBatch:
    """Tests for generate_events_with_quality batch wrapper."""

    @patch("src.services.world_quality_service._event._judge_event_quality")
    @patch("src.services.world_quality_service._event._create_event")
    def test_batch_generates_events(self, mock_create, mock_judge, service, story_state):
        """Test batch wrapper calls single-event generator and collects results."""
        mock_create.return_value = {
            "description": "A great battle",
            "year": 1200,
            "participants": [],
            "consequences": [],
        }
        mock_judge.return_value = EventQualityScores(
            significance=8.0,
            temporal_plausibility=8.0,
            causal_coherence=8.0,
            narrative_potential=8.0,
            entity_integration=8.0,
            feedback="Good",
        )

        results = service.generate_events_with_quality(
            story_state,
            existing_descriptions=["Existing event"],
            entity_context="Test context",
            count=2,
        )

        assert len(results) >= 1
        for event_dict, scores in results:
            assert event_dict["description"] == "A great battle"
            assert scores.average == 8.0
