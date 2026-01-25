"""Tests for early stopping in quality refinement loops."""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import Character, StoryBrief, StoryState
from src.memory.world_quality import (
    CharacterQualityScores,
    FactionQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RefinementHistory,
)
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings


@pytest.fixture
def settings():
    """Create settings with early stopping enabled.

    Uses threshold of 9.0 to ensure scores (8.0-8.5) don't meet it,
    allowing the early stopping logic to be properly tested.
    """
    return Settings(
        ollama_url="http://localhost:11434",
        ollama_timeout=60,
        world_quality_enabled=True,
        world_quality_max_iterations=10,
        world_quality_threshold=9.0,  # High threshold so 8.x scores don't exit early
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        world_quality_early_stopping_patience=2,
        llm_tokens_faction_create=400,
        llm_tokens_faction_judge=300,
        llm_tokens_faction_refine=400,
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


class TestRefinementHistoryEarlyStopping:
    """Tests for RefinementHistory early stopping logic."""

    def test_consecutive_degradations_tracked(self):
        """Test that consecutive degradations are tracked correctly."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # First iteration - becomes peak
        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        assert history.consecutive_degradations == 0
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

        # Second iteration - degrades
        history.add_iteration(2, {"name": "Test"}, {"score": 7.5}, 7.5)
        assert history.consecutive_degradations == 1
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

        # Third iteration - degrades again
        history.add_iteration(3, {"name": "Test"}, {"score": 7.0}, 7.0)
        assert history.consecutive_degradations == 2
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

    def test_consecutive_degradations_reset_on_new_peak(self):
        """Test that consecutive degradations reset when a new peak is reached."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Iteration 1 - peak
        history.add_iteration(1, {"name": "Test"}, {"score": 7.0}, 7.0)
        assert history.consecutive_degradations == 0

        # Iteration 2 - degrades
        history.add_iteration(2, {"name": "Test"}, {"score": 6.5}, 6.5)
        assert history.consecutive_degradations == 1

        # Iteration 3 - new peak (resets counter)
        history.add_iteration(3, {"name": "Test"}, {"score": 8.0}, 8.0)
        assert history.consecutive_degradations == 0
        assert history.peak_score == 8.0
        assert history.best_iteration == 3

        # Iteration 4 - degrades from new peak
        history.add_iteration(4, {"name": "Test"}, {"score": 7.5}, 7.5)
        assert history.consecutive_degradations == 1

    def test_should_stop_early_returns_false_initially(self):
        """Test that should_stop_early returns False initially."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")
        assert not history.should_stop_early(patience=2)

    def test_should_stop_early_returns_false_before_patience(self):
        """Test that should_stop_early returns False before patience threshold."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        history.add_iteration(2, {"name": "Test"}, {"score": 7.5}, 7.5)
        assert history.consecutive_degradations == 1
        assert not history.should_stop_early(patience=2)

    def test_should_stop_early_returns_true_at_patience(self):
        """Test that should_stop_early returns True when patience reached."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        history.add_iteration(2, {"name": "Test"}, {"score": 7.5}, 7.5)
        history.add_iteration(3, {"name": "Test"}, {"score": 7.0}, 7.0)
        assert history.consecutive_degradations == 2
        assert history.should_stop_early(patience=2)

    def test_should_stop_early_returns_true_after_patience(self):
        """Test that should_stop_early returns True when exceeding patience."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        history.add_iteration(2, {"name": "Test"}, {"score": 7.5}, 7.5)
        history.add_iteration(3, {"name": "Test"}, {"score": 7.0}, 7.0)
        history.add_iteration(4, {"name": "Test"}, {"score": 6.5}, 6.5)
        assert history.consecutive_degradations == 3
        assert history.should_stop_early(patience=2)

    def test_plateau_after_peak_increments_degradation_counter(self):
        """Test that plateauing (equal score) doesn't reset counter but doesn't increment."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Iteration 1 - peak
        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        assert history.consecutive_degradations == 0

        # Iteration 2 - same score (plateau)
        history.add_iteration(2, {"name": "Test"}, {"score": 8.0}, 8.0)
        # Plateau doesn't increment degradation counter
        assert history.consecutive_degradations == 0

    def test_analyze_improvement_includes_consecutive_degradations(self):
        """Test that analyze_improvement includes consecutive_degradations."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(1, {"name": "Test"}, {"score": 8.0}, 8.0)
        history.add_iteration(2, {"name": "Test"}, {"score": 7.5}, 7.5)
        history.add_iteration(3, {"name": "Test"}, {"score": 7.0}, 7.0)

        analysis = history.analyze_improvement()
        assert "consecutive_degradations" in analysis
        assert analysis["consecutive_degradations"] == 2


class TestRefinementConfig:
    """Tests for RefinementConfig early stopping settings."""

    def test_early_stopping_patience_from_settings(self):
        """Test that early_stopping_patience is loaded from settings."""
        settings = Settings(
            world_quality_early_stopping_patience=3,
        )
        config = RefinementConfig.from_settings(settings)
        assert config.early_stopping_patience == 3

    def test_early_stopping_patience_default(self):
        """Test default early_stopping_patience value."""
        config = RefinementConfig()
        assert config.early_stopping_patience == 2

    def test_early_stopping_patience_validation(self):
        """Test early_stopping_patience validation."""
        # Valid values
        RefinementConfig(early_stopping_patience=1)
        RefinementConfig(early_stopping_patience=10)

        # Invalid values
        with pytest.raises(ValueError):  # Pydantic validation error
            RefinementConfig(early_stopping_patience=0)

        with pytest.raises(ValueError):
            RefinementConfig(early_stopping_patience=11)


class TestFactionGenerationEarlyStopping:
    """Tests for faction generation with early stopping."""

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    @patch.object(WorldQualityService, "_refine_faction")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that faction generation stops early after consecutive degradations."""
        # Create faction that degrades after initial peak
        test_faction = {"name": "TestFaction", "description": "A test faction"}
        mock_create.return_value = test_faction
        mock_refine.return_value = test_faction

        # Score progression: 8.2 -> 7.9 -> 7.6 (should stop here with patience=2)
        scores = [
            FactionQualityScores(
                coherence=8.2, influence=8.2, conflict_potential=8.2, distinctiveness=8.2
            ),
            FactionQualityScores(
                coherence=7.9, influence=7.9, conflict_potential=7.9, distinctiveness=7.9
            ),
            FactionQualityScores(
                coherence=7.6, influence=7.6, conflict_potential=7.6, distinctiveness=7.6
            ),
        ]
        mock_judge.side_effect = scores

        _faction, final_scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[], existing_locations=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.2)
        assert iterations == 1
        assert final_scores.average == 8.2

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    @patch.object(WorldQualityService, "_refine_faction")
    def test_no_early_stopping_on_single_degradation(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that single degradation doesn't trigger early stopping.

        Score progression: 8.0 -> 7.5 -> 9.5 (improves and meets threshold)
        The single degradation (8.0->7.5) doesn't trigger early stopping,
        allowing the loop to continue and eventually meet threshold.
        """
        test_faction = {"name": "TestFaction", "description": "A test faction"}
        mock_create.return_value = test_faction
        mock_refine.return_value = test_faction

        # Score progression: 8.0 -> 7.5 -> 9.5 (meets threshold=9.0, stops)
        scores = [
            FactionQualityScores(
                coherence=8.0, influence=8.0, conflict_potential=8.0, distinctiveness=8.0
            ),
            FactionQualityScores(
                coherence=7.5, influence=7.5, conflict_potential=7.5, distinctiveness=7.5
            ),
            FactionQualityScores(
                coherence=9.5, influence=9.5, conflict_potential=9.5, distinctiveness=9.5
            ),
        ]
        mock_judge.side_effect = scores

        _faction, final_scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[], existing_locations=[]
        )

        # Should complete 3 iterations - single degradation doesn't trigger early stop
        assert mock_judge.call_count == 3
        # Returns iteration 3 since it met threshold (9.5 >= 9.0)
        assert iterations == 3
        assert final_scores.average == 9.5

    @patch.object(WorldQualityService, "_create_faction")
    @patch.object(WorldQualityService, "_judge_faction_quality")
    @patch.object(WorldQualityService, "_refine_faction")
    def test_early_stopping_saves_compute(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that early stopping prevents running all max_iterations.

        Scores degrade consistently after peak, triggering early stopping
        after patience (2) consecutive degradations, saving compute.
        """
        test_faction = {"name": "TestFaction", "description": "A test faction"}
        mock_create.return_value = test_faction
        mock_refine.return_value = test_faction

        # Scores degrade consistently after peak
        scores = [
            FactionQualityScores(
                coherence=8.0, influence=8.0, conflict_potential=8.0, distinctiveness=8.0
            ),
            FactionQualityScores(
                coherence=7.5, influence=7.5, conflict_potential=7.5, distinctiveness=7.5
            ),
            FactionQualityScores(
                coherence=7.0, influence=7.0, conflict_potential=7.0, distinctiveness=7.0
            ),
            # These should not be reached due to early stopping
            FactionQualityScores(
                coherence=6.5, influence=6.5, conflict_potential=6.5, distinctiveness=6.5
            ),
        ]
        mock_judge.side_effect = scores

        _faction, final_scores, iterations = service.generate_faction_with_quality(
            story_state, existing_names=[], existing_locations=[]
        )

        # Should only run 3 iterations (early stopping saves compute)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.0)
        assert iterations == 1
        assert final_scores.average == 8.0


class TestCharacterGenerationEarlyStopping:
    """Tests for character generation with early stopping."""

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that character generation stops early after consecutive degradations."""
        test_char = Character(name="TestChar", role="protagonist", description="A test character")
        mock_create.return_value = test_char
        mock_refine.return_value = test_char

        # Score progression: 8.2 -> 7.9 -> 7.6 (should stop here with patience=2)
        scores = [
            CharacterQualityScores(
                depth=8.2, goals=8.2, flaws=8.2, uniqueness=8.2, arc_potential=8.2
            ),
            CharacterQualityScores(
                depth=7.9, goals=7.9, flaws=7.9, uniqueness=7.9, arc_potential=7.9
            ),
            CharacterQualityScores(
                depth=7.6, goals=7.6, flaws=7.6, uniqueness=7.6, arc_potential=7.6
            ),
        ]
        mock_judge.side_effect = scores

        _char, final_scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.2)
        assert iterations == 1
        assert final_scores.average == 8.2

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_no_early_stopping_on_improvement(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that improvement resets degradation counter and allows threshold exit."""
        test_char = Character(name="TestChar", role="protagonist", description="A test character")
        mock_create.return_value = test_char
        mock_refine.return_value = test_char

        # Score progression: 8.0 -> 7.5 -> 9.5 (meets threshold=9.0, stops)
        scores = [
            CharacterQualityScores(
                depth=8.0, goals=8.0, flaws=8.0, uniqueness=8.0, arc_potential=8.0
            ),
            CharacterQualityScores(
                depth=7.5, goals=7.5, flaws=7.5, uniqueness=7.5, arc_potential=7.5
            ),
            CharacterQualityScores(
                depth=9.5, goals=9.5, flaws=9.5, uniqueness=9.5, arc_potential=9.5
            ),
        ]
        mock_judge.side_effect = scores

        _char, final_scores, iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        # Should complete 3 iterations - single degradation doesn't trigger early stop
        assert mock_judge.call_count == 3
        # Returns iteration 3 since it met threshold (9.5 >= 9.0)
        assert iterations == 3
        assert final_scores.average == 9.5


class TestLocationGenerationEarlyStopping:
    """Tests for location generation with early stopping."""

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    @patch.object(WorldQualityService, "_refine_location")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that location generation stops early after consecutive degradations."""
        test_loc = {"name": "TestLocation", "description": "A test location"}
        mock_create.return_value = test_loc
        mock_refine.return_value = test_loc

        # Score progression: 8.0 -> 7.5 -> 7.0 (should stop here with patience=2)
        scores = [
            LocationQualityScores(
                atmosphere=8.0, significance=8.0, story_relevance=8.0, distinctiveness=8.0
            ),
            LocationQualityScores(
                atmosphere=7.5, significance=7.5, story_relevance=7.5, distinctiveness=7.5
            ),
            LocationQualityScores(
                atmosphere=7.0, significance=7.0, story_relevance=7.0, distinctiveness=7.0
            ),
        ]
        mock_judge.side_effect = scores

        _loc, final_scores, iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.0)
        assert iterations == 1
        assert final_scores.average == 8.0
