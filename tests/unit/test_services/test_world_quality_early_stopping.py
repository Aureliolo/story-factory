"""Tests for early stopping in quality refinement loops."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import Character, StoryBrief, StoryState
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RefinementHistory,
    RelationshipQualityScores,
)
from src.services.world_quality_service import WorldQualityService
from src.settings import Settings
from tests.shared.mock_ollama import TEST_MODEL


def _make_unique_refine(factory: Callable[[int], Any]) -> Callable[..., Any]:
    """Create a refine side_effect that produces unique entities per call.

    Wraps a factory callable, passing an incrementing counter so each call
    returns a distinct entity. This prevents unchanged-output detection
    from triggering during tests.

    Args:
        factory: Callable that takes a version number and returns an entity.

    Returns:
        Side-effect function suitable for mock_refine.side_effect.
    """
    counter = {"n": 0}

    def _refine(*args: Any, **kwargs: Any) -> Any:
        """Return entity with unique version number."""
        counter["n"] += 1
        return factory(counter["n"])

    return _refine


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
    mode_service.get_model_for_agent.return_value = TEST_MODEL
    return mode_service


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client to avoid network calls in tests."""
    return MagicMock()


@pytest.fixture
def service(settings, mock_mode_service, mock_ollama_client):
    """Create WorldQualityService with mocked dependencies."""
    svc = WorldQualityService(settings, mock_mode_service)
    svc._client = mock_ollama_client
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
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        assert history.consecutive_degradations == 0
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

        # Second iteration - degrades
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        assert history.consecutive_degradations == 1
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

        # Third iteration - degrades again
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )
        assert history.consecutive_degradations == 2
        assert history.peak_score == 8.0
        assert history.best_iteration == 1

    def test_consecutive_degradations_reset_on_new_peak(self):
        """Test that consecutive degradations reset when a new peak is reached."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Iteration 1 - peak
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )
        assert history.consecutive_degradations == 0

        # Iteration 2 - degrades
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 6.5}, average_score=6.5
        )
        assert history.consecutive_degradations == 1

        # Iteration 3 - new peak (resets counter)
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        assert history.consecutive_degradations == 0
        assert history.peak_score == 8.0
        assert history.best_iteration == 3

        # Iteration 4 - degrades from new peak
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        assert history.consecutive_degradations == 1

    def test_should_stop_early_returns_false_initially(self):
        """Test that should_stop_early returns False initially."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")
        assert not history.should_stop_early(patience=2, min_iterations=1)

    def test_should_stop_early_returns_false_before_patience(self):
        """Test that should_stop_early returns False before patience threshold."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        assert history.consecutive_degradations == 1
        assert not history.should_stop_early(patience=2, min_iterations=1)

    def test_should_stop_early_returns_true_at_patience(self):
        """Test that should_stop_early returns True when patience reached."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )
        assert history.consecutive_degradations == 2
        assert history.should_stop_early(patience=2, min_iterations=1)

    def test_should_stop_early_returns_true_after_patience(self):
        """Test that should_stop_early returns True when exceeding patience."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 6.5}, average_score=6.5
        )
        assert history.consecutive_degradations == 3
        assert history.should_stop_early(patience=2, min_iterations=1)

    def test_plateau_after_peak_does_not_increment_degradation_counter(self):
        """Test that plateauing (equal score) doesn't reset counter but doesn't increment."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Iteration 1 - peak
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        assert history.consecutive_degradations == 0

        # Iteration 2 - same score (plateau)
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        # Plateau doesn't increment degradation counter
        assert history.consecutive_degradations == 0

    def test_analyze_improvement_includes_consecutive_degradations(self):
        """Test that analyze_improvement includes consecutive_degradations."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.5}, average_score=7.5
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )

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
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestFaction", "description": f"A test faction v{n}"}
        )

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
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestFaction", "description": f"A test faction v{n}"}
        )

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
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestFaction", "description": f"A test faction v{n}"}
        )

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
    def test_zero_scores_fallback_path(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test character generation with zero scores falls back to last iteration.

        When all scores are 0, peak_score is never exceeded, so best_character
        is never set. The function falls back to returning the last character.
        """
        test_char = Character(name="ZeroChar", role="protagonist", description="Zero scores")
        mock_create.return_value = test_char
        mock_refine.side_effect = _make_unique_refine(
            lambda n: Character(
                name="ZeroChar", role="protagonist", description=f"Zero scores v{n}"
            )
        )

        # All zero scores - peak_score (0.0) is never exceeded
        zero_scores = CharacterQualityScores(
            depth=0, goals=0, flaws=0, uniqueness=0, arc_potential=0
        )
        mock_judge.return_value = zero_scores

        char, final_scores, _iterations = service.generate_character_with_quality(
            story_state, existing_names=[]
        )

        # Should run all max_iterations (10) since no threshold met and no early stopping on zeros
        assert mock_judge.call_count == 10
        # Returns last iteration since best_character was never set (peak_score never exceeded)
        assert char.name == "ZeroChar"
        assert final_scores.average == 0

    @patch.object(WorldQualityService, "_create_character")
    @patch.object(WorldQualityService, "_judge_character_quality")
    @patch.object(WorldQualityService, "_refine_character")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that character generation stops early after consecutive degradations."""
        test_char = Character(name="TestChar", role="protagonist", description="A test character")
        mock_create.return_value = test_char
        mock_refine.side_effect = _make_unique_refine(
            lambda n: Character(
                name="TestChar", role="protagonist", description=f"A test character v{n}"
            )
        )

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
        mock_refine.side_effect = _make_unique_refine(
            lambda n: Character(
                name="TestChar", role="protagonist", description=f"A test character v{n}"
            )
        )

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
    def test_zero_scores_fallback_path(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test location generation with zero scores falls back to last iteration."""
        test_loc = {"name": "ZeroLoc", "description": "Zero scores"}
        mock_create.return_value = test_loc
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "ZeroLoc", "description": f"Zero scores v{n}"}
        )

        zero_scores = LocationQualityScores(
            atmosphere=0, significance=0, story_relevance=0, distinctiveness=0
        )
        mock_judge.return_value = zero_scores

        loc, final_scores, _iterations = service.generate_location_with_quality(
            story_state, existing_names=[]
        )

        assert mock_judge.call_count == 10
        assert loc["name"] == "ZeroLoc"
        assert final_scores.average == 0

    @patch.object(WorldQualityService, "_create_location")
    @patch.object(WorldQualityService, "_judge_location_quality")
    @patch.object(WorldQualityService, "_refine_location")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that location generation stops early after consecutive degradations."""
        test_loc = {"name": "TestLocation", "description": "A test location"}
        mock_create.return_value = test_loc
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestLocation", "description": f"A test location v{n}"}
        )

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


class TestRelationshipGenerationEarlyStopping:
    """Tests for relationship generation with early stopping."""

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    @patch.object(WorldQualityService, "_refine_relationship")
    def test_zero_scores_fallback_path(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test relationship generation with zero scores falls back to last iteration."""
        test_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "knows",
            "description": "Zero",
        }
        mock_create.return_value = test_rel
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "knows",
                "description": f"Zero v{n}",
            }
        )

        zero_scores = RelationshipQualityScores(
            tension=0, dynamics=0, story_potential=0, authenticity=0
        )
        mock_judge.return_value = zero_scores

        rel, final_scores, _iterations = service.generate_relationship_with_quality(
            story_state, entity_names=["Alice", "Bob"], existing_rels=[]
        )

        assert mock_judge.call_count == 10
        assert rel["source"] == "Alice"
        assert final_scores.average == 0

    @patch.object(WorldQualityService, "_create_relationship")
    @patch.object(WorldQualityService, "_judge_relationship_quality")
    @patch.object(WorldQualityService, "_refine_relationship")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that relationship generation stops early after consecutive degradations."""
        test_rel = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "allies",
            "description": "They work together",
        }
        mock_create.return_value = test_rel
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "allies",
                "description": f"They work together v{n}",
            }
        )

        # Score progression: 8.0 -> 7.5 -> 7.0 (should stop here with patience=2)
        scores = [
            RelationshipQualityScores(
                tension=8.0, dynamics=8.0, story_potential=8.0, authenticity=8.0
            ),
            RelationshipQualityScores(
                tension=7.5, dynamics=7.5, story_potential=7.5, authenticity=7.5
            ),
            RelationshipQualityScores(
                tension=7.0, dynamics=7.0, story_potential=7.0, authenticity=7.0
            ),
        ]
        mock_judge.side_effect = scores

        _rel, final_scores, iterations = service.generate_relationship_with_quality(
            story_state, entity_names=["Alice", "Bob"], existing_rels=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.0)
        assert iterations == 1
        assert final_scores.average == 8.0


class TestItemGenerationEarlyStopping:
    """Tests for item generation with early stopping."""

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    @patch.object(WorldQualityService, "_refine_item")
    def test_zero_scores_fallback_path(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test item generation with zero scores falls back to last iteration."""
        test_item = {"name": "ZeroItem", "description": "Zero scores"}
        mock_create.return_value = test_item
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "ZeroItem", "description": f"Zero scores v{n}"}
        )

        zero_scores = ItemQualityScores(
            significance=0, uniqueness=0, narrative_potential=0, integration=0
        )
        mock_judge.return_value = zero_scores

        item, final_scores, _iterations = service.generate_item_with_quality(
            story_state, existing_names=[]
        )

        assert mock_judge.call_count == 10
        assert item["name"] == "ZeroItem"
        assert final_scores.average == 0

    @patch.object(WorldQualityService, "_create_item")
    @patch.object(WorldQualityService, "_judge_item_quality")
    @patch.object(WorldQualityService, "_refine_item")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that item generation stops early after consecutive degradations."""
        test_item = {"name": "TestItem", "description": "A test item"}
        mock_create.return_value = test_item
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestItem", "description": f"A test item v{n}"}
        )

        # Score progression: 8.0 -> 7.5 -> 7.0 (should stop here with patience=2)
        scores = [
            ItemQualityScores(
                significance=8.0, uniqueness=8.0, narrative_potential=8.0, integration=8.0
            ),
            ItemQualityScores(
                significance=7.5, uniqueness=7.5, narrative_potential=7.5, integration=7.5
            ),
            ItemQualityScores(
                significance=7.0, uniqueness=7.0, narrative_potential=7.0, integration=7.0
            ),
        ]
        mock_judge.side_effect = scores

        _item, final_scores, iterations = service.generate_item_with_quality(
            story_state, existing_names=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.0)
        assert iterations == 1
        assert final_scores.average == 8.0


class TestConceptGenerationEarlyStopping:
    """Tests for concept generation with early stopping."""

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    @patch.object(WorldQualityService, "_refine_concept")
    def test_zero_scores_fallback_path(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test concept generation with zero scores falls back to last iteration."""
        test_concept = {"name": "ZeroConcept", "description": "Zero scores"}
        mock_create.return_value = test_concept
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "ZeroConcept", "description": f"Zero scores v{n}"}
        )

        zero_scores = ConceptQualityScores(relevance=0, depth=0, manifestation=0, resonance=0)
        mock_judge.return_value = zero_scores

        concept, final_scores, _iterations = service.generate_concept_with_quality(
            story_state, existing_names=[]
        )

        assert mock_judge.call_count == 10
        assert concept["name"] == "ZeroConcept"
        assert final_scores.average == 0

    @patch.object(WorldQualityService, "_create_concept")
    @patch.object(WorldQualityService, "_judge_concept_quality")
    @patch.object(WorldQualityService, "_refine_concept")
    def test_early_stopping_after_consecutive_degradations(
        self, mock_refine, mock_judge, mock_create, service, story_state
    ):
        """Test that concept generation stops early after consecutive degradations."""
        test_concept = {"name": "TestConcept", "description": "A test concept"}
        mock_create.return_value = test_concept
        mock_refine.side_effect = _make_unique_refine(
            lambda n: {"name": "TestConcept", "description": f"A test concept v{n}"}
        )

        # Score progression: 8.0 -> 7.5 -> 7.0 (should stop here with patience=2)
        scores = [
            ConceptQualityScores(relevance=8.0, depth=8.0, manifestation=8.0, resonance=8.0),
            ConceptQualityScores(relevance=7.5, depth=7.5, manifestation=7.5, resonance=7.5),
            ConceptQualityScores(relevance=7.0, depth=7.0, manifestation=7.0, resonance=7.0),
        ]
        mock_judge.side_effect = scores

        _concept, final_scores, iterations = service.generate_concept_with_quality(
            story_state, existing_names=[]
        )

        # Should have run 3 iterations (2 consecutive degradations from peak at iteration 1)
        assert mock_judge.call_count == 3
        # Returns best iteration number (1) and best scores (8.0)
        assert iterations == 1
        assert final_scores.average == 8.0


class TestEnhancedEarlyStopping:
    """Tests for enhanced early stopping with variance tolerance."""

    def test_should_stop_early_with_variance_tolerance(self):
        """Test that variance tolerance prevents stopping on noisy degradations."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Add iterations with small variance in scores
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.9}, average_score=7.9
        )  # -0.1
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.8}, average_score=7.8
        )  # -0.1

        # Without variance tolerance, this would trigger early stop (patience=2)
        assert history.consecutive_degradations == 2

        # With high variance tolerance, small drops are considered noise
        should_stop = history.should_stop_early(
            patience=2,
            min_iterations=2,  # We're past min_iterations
            variance_tolerance=0.5,  # High tolerance for noise
        )
        # Score drop is 0.2, variance is low, should NOT stop
        assert should_stop is False

    def test_should_stop_early_respects_min_iterations(self):
        """Test that should_stop_early respects min_iterations parameter."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Add degrading iterations
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 7.0}, average_score=7.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 6.0}, average_score=6.0
        )

        # With min_iterations=4, should not stop even with 2 degradations
        should_stop = history.should_stop_early(
            patience=2,
            min_iterations=4,
        )
        # Only 3 iterations run, min is 4, so should NOT stop
        assert should_stop is False

    def test_should_stop_early_large_degradation_ignores_variance(self):
        """Test that large degradations trigger early stop despite variance tolerance."""
        history = RefinementHistory(entity_type="faction", entity_name="Test")

        # Add iterations with large score drops
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 8.0}, average_score=8.0
        )
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 6.5}, average_score=6.5
        )  # -1.5
        history.add_iteration(
            entity_data={"name": "Test"}, scores={"score": 5.0}, average_score=5.0
        )  # -1.5

        assert history.consecutive_degradations == 2

        # Even with variance tolerance, large drops should trigger early stop
        should_stop = history.should_stop_early(
            patience=2,
            min_iterations=2,
            variance_tolerance=0.5,  # Tolerance is smaller than drops
        )
        # Score drop (3.0) exceeds variance tolerance, should stop
        assert should_stop is True


class TestDynamicTemperature:
    """Tests for dynamic temperature adjustment during refinement."""

    def test_first_iteration_uses_start_temperature(self):
        """Test that first iteration always uses start temperature."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="linear",
        )

        temp = config.get_refinement_temperature(iteration=1, max_iterations=5)
        assert temp == 0.8

    def test_last_iteration_uses_end_temperature(self):
        """Test that last iteration always uses end temperature."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="linear",
        )

        temp = config.get_refinement_temperature(iteration=5, max_iterations=5)
        assert temp == 0.3

    def test_linear_decay_midpoint(self):
        """Test linear decay gives expected midpoint temperature."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.4,
            refinement_temp_decay="linear",
        )

        # Iteration 3 of 5: progress = (3-1)/(5-1) = 0.5
        # Expected: 0.8 + 0.5 * (0.4 - 0.8) = 0.8 - 0.2 = 0.6
        temp = config.get_refinement_temperature(iteration=3, max_iterations=5)
        assert temp == pytest.approx(0.6)

    def test_exponential_decay(self):
        """Test exponential decay drops faster initially than linear."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="exponential",
        )

        # Exponential decay uses 1 - (1 - progress)^2 for faster initial drop
        # Iteration 3 of 5: progress = 0.5, decay_factor = 1 - (1-0.5)^2 = 0.75
        # Expected: 0.8 + 0.75 * (0.3 - 0.8) = 0.8 - 0.375 = 0.425
        temp = config.get_refinement_temperature(iteration=3, max_iterations=5)
        assert temp == pytest.approx(0.425)

    def test_step_decay_before_midpoint(self):
        """Test step decay stays at start temp before midpoint."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="step",
        )

        # Iteration 2 of 5: progress = 0.25 < 0.5, use start temp
        temp = config.get_refinement_temperature(iteration=2, max_iterations=5)
        assert temp == 0.8

    def test_step_decay_after_midpoint(self):
        """Test step decay drops to end temp after midpoint."""
        config = RefinementConfig(
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="step",
        )

        # Iteration 4 of 5: progress = 0.75 >= 0.5, use end temp
        temp = config.get_refinement_temperature(iteration=4, max_iterations=5)
        assert temp == 0.3

    def test_get_refinement_temperature_uses_config_max_iterations(self):
        """Test that max_iterations defaults to config value."""
        config = RefinementConfig(
            max_iterations=3,
            refinement_temp_start=0.9,
            refinement_temp_end=0.4,
            refinement_temp_decay="linear",
        )

        # Should use config.max_iterations (3) when not specified
        temp = config.get_refinement_temperature(iteration=2)
        # progress = (2-1)/(3-1) = 0.5, temp = 0.9 + 0.5*(0.4-0.9) = 0.65
        assert temp == pytest.approx(0.65)

    def test_get_refinement_temperature_handles_max_iterations_one(self):
        """Test that max_iterations=1 with iteration>1 returns end temperature.

        This guards against division by zero when max_iterations=1 but
        an iteration > 1 is somehow passed (defensive coding).
        """
        config = RefinementConfig(
            max_iterations=1,
            refinement_temp_start=0.8,
            refinement_temp_end=0.3,
            refinement_temp_decay="linear",
        )

        # Unusual case: iteration > 1 with max_iterations=1
        # The guard prevents division by zero: progress = (iter-1)/(max_iter-1)
        temp = config.get_refinement_temperature(iteration=2, max_iterations=1)
        assert temp == 0.3

    def test_temperature_rounded_to_3_decimal_places(self):
        """Test that temperatures are rounded to avoid float precision artifacts (#248).

        Float arithmetic can produce values like 0.5249999999999999 instead of 0.525.
        The rounding ensures clean log output and consistent behavior.
        """
        config = RefinementConfig(
            refinement_temp_start=0.7,
            refinement_temp_end=0.35,
            refinement_temp_decay="linear",
            max_iterations=3,
        )

        # Iteration 2 of 3: progress = (2-1)/(3-1) = 0.5
        # Expected: 0.7 + 0.5 * (0.35 - 0.7) = 0.7 - 0.175 = 0.525
        # Without rounding, this could be 0.5249999999999999
        temp = config.get_refinement_temperature(iteration=2, max_iterations=3)

        # The temperature should be exactly 0.525, not 0.5249999999999999
        assert temp == 0.525
        # Also verify it's a clean float representation
        assert str(temp) == "0.525"


class TestLocationQualityScoresWeakDimensions:
    """Tests for LocationQualityScores.weak_dimensions edge cases."""

    def test_weak_dimensions_identifies_atmosphere(self):
        """Test weak_dimensions identifies low atmosphere."""
        scores = LocationQualityScores(
            atmosphere=5.0, significance=8.0, story_relevance=8.0, distinctiveness=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "atmosphere" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_narrative_significance(self):
        """Test weak_dimensions identifies low narrative_significance."""
        scores = LocationQualityScores(
            atmosphere=8.0, significance=5.0, story_relevance=8.0, distinctiveness=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "narrative_significance" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_story_relevance(self):
        """Test weak_dimensions identifies low story_relevance."""
        scores = LocationQualityScores(
            atmosphere=8.0, significance=8.0, story_relevance=5.0, distinctiveness=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "story_relevance" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_distinctiveness(self):
        """Test weak_dimensions identifies low distinctiveness."""
        scores = LocationQualityScores(
            atmosphere=8.0, significance=8.0, story_relevance=8.0, distinctiveness=5.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "distinctiveness" in weak
        assert len(weak) == 1


class TestItemQualityScoresWeakDimensions:
    """Tests for ItemQualityScores.weak_dimensions edge cases."""

    def test_weak_dimensions_identifies_story_significance(self):
        """Test weak_dimensions identifies low story_significance."""
        scores = ItemQualityScores(
            significance=5.0, uniqueness=8.0, narrative_potential=8.0, integration=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "story_significance" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_uniqueness(self):
        """Test weak_dimensions identifies low uniqueness."""
        scores = ItemQualityScores(
            significance=8.0, uniqueness=5.0, narrative_potential=8.0, integration=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "uniqueness" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_narrative_potential(self):
        """Test weak_dimensions identifies low narrative_potential."""
        scores = ItemQualityScores(
            significance=8.0, uniqueness=8.0, narrative_potential=5.0, integration=8.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "narrative_potential" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_integration(self):
        """Test weak_dimensions identifies low integration."""
        scores = ItemQualityScores(
            significance=8.0, uniqueness=8.0, narrative_potential=8.0, integration=5.0
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "integration" in weak
        assert len(weak) == 1


class TestConceptQualityScoresWeakDimensions:
    """Tests for ConceptQualityScores.weak_dimensions edge cases."""

    def test_weak_dimensions_identifies_relevance(self):
        """Test weak_dimensions identifies low relevance."""
        scores = ConceptQualityScores(relevance=5.0, depth=8.0, manifestation=8.0, resonance=8.0)
        weak = scores.weak_dimensions(threshold=7.0)
        assert "relevance" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_depth(self):
        """Test weak_dimensions identifies low depth."""
        scores = ConceptQualityScores(relevance=8.0, depth=5.0, manifestation=8.0, resonance=8.0)
        weak = scores.weak_dimensions(threshold=7.0)
        assert "depth" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_manifestation(self):
        """Test weak_dimensions identifies low manifestation."""
        scores = ConceptQualityScores(relevance=8.0, depth=8.0, manifestation=5.0, resonance=8.0)
        weak = scores.weak_dimensions(threshold=7.0)
        assert "manifestation" in weak
        assert len(weak) == 1

    def test_weak_dimensions_identifies_resonance(self):
        """Test weak_dimensions identifies low resonance."""
        scores = ConceptQualityScores(relevance=8.0, depth=8.0, manifestation=8.0, resonance=5.0)
        weak = scores.weak_dimensions(threshold=7.0)
        assert "resonance" in weak
        assert len(weak) == 1
