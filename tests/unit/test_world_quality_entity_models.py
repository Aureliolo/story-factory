"""Tests for RefinementHistory plateau detection (Issue #231).

Tests cover:
- consecutive_plateaus field tracking
- Plateau resets on new peak
- should_stop_early() returns True after N plateaus
- Flat score sequences trigger early stop
"""

from src.memory.world_quality import RefinementHistory


class TestConsecutivePlateaus:
    """Test consecutive_plateaus tracking in RefinementHistory."""

    def test_plateau_increments_on_equal_scores(self):
        """consecutive_plateaus increments when score equals peak."""
        history = RefinementHistory(entity_type="location", entity_name="Test Place")

        # First iteration sets peak
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 0
        assert history.peak_score == 6.0

        # Second iteration matches peak — plateau
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 1

        # Third iteration matches peak again — plateau
        history.add_iteration(
            entity_data={"name": "Test Place"},
            scores={"atmosphere": 6.0, "significance": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 2

    def test_plateau_resets_on_new_peak(self):
        """consecutive_plateaus resets to 0 when a new peak is reached."""
        history = RefinementHistory(entity_type="character", entity_name="Hero")

        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 6.0},
            average_score=6.0,
        )
        assert history.consecutive_plateaus == 1

        # New peak resets plateaus
        history.add_iteration(
            entity_data={"name": "Hero"},
            scores={"depth": 7.0},
            average_score=7.0,
        )
        assert history.consecutive_plateaus == 0
        assert history.peak_score == 7.0

    def test_degradation_does_not_increment_plateaus(self):
        """consecutive_plateaus stays unchanged when score degrades."""
        history = RefinementHistory(entity_type="item", entity_name="Sword")

        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 7.0},
            average_score=7.0,
        )
        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 7.0},
            average_score=7.0,
        )
        assert history.consecutive_plateaus == 1

        # Degradation — plateaus should not change
        history.add_iteration(
            entity_data={"name": "Sword"},
            scores={"significance": 5.0},
            average_score=5.0,
        )
        assert history.consecutive_plateaus == 1
        assert history.consecutive_degradations == 1


class TestShouldStopEarlyPlateau:
    """Test should_stop_early() plateau detection."""

    def test_flat_scores_trigger_early_stop(self):
        """Flat scores [6.0, 6.0, 6.0] with patience=2 triggers early stop."""
        history = RefinementHistory(entity_type="location", entity_name="Tavern")

        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Tavern"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )

        assert history.consecutive_plateaus == 2
        assert history.should_stop_early(patience=2, min_iterations=2) is True

    def test_plateau_does_not_trigger_before_min_iterations(self):
        """Plateau should not trigger early stop before min_iterations."""
        history = RefinementHistory(entity_type="faction", entity_name="Guild")

        history.add_iteration(
            entity_data={"name": "Guild"},
            scores={"coherence": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Guild"},
            scores={"coherence": 6.0},
            average_score=6.0,
        )

        # 2 iterations, min_iterations=3 — should NOT trigger
        assert history.consecutive_plateaus == 1
        assert history.should_stop_early(patience=1, min_iterations=3) is False

    def test_plateau_below_patience_does_not_trigger(self):
        """Plateau count below patience should not trigger early stop."""
        history = RefinementHistory(entity_type="concept", entity_name="Theme")

        history.add_iteration(
            entity_data={"name": "Theme"},
            scores={"relevance": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Theme"},
            scores={"relevance": 6.0},
            average_score=6.0,
        )

        # 1 plateau, patience=2 — should NOT trigger
        assert history.consecutive_plateaus == 1
        assert history.should_stop_early(patience=2, min_iterations=2) is False

    def test_mixed_plateau_then_improvement(self):
        """Plateau followed by improvement should NOT trigger early stop."""
        history = RefinementHistory(entity_type="location", entity_name="Castle")

        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Castle"},
            scores={"atmosphere": 7.0},
            average_score=7.0,
        )

        assert history.consecutive_plateaus == 0
        assert history.should_stop_early(patience=2, min_iterations=2) is False

    def test_degradation_still_triggers_early_stop(self):
        """Degradation-based early stop still works alongside plateau detection."""
        history = RefinementHistory(entity_type="item", entity_name="Ring")

        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 8.0},
            average_score=8.0,
        )
        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 6.0},
            average_score=6.0,
        )
        history.add_iteration(
            entity_data={"name": "Ring"},
            scores={"significance": 5.0},
            average_score=5.0,
        )

        assert history.consecutive_degradations == 2
        assert history.consecutive_plateaus == 0
        # Score drop is 3.0, well above variance tolerance of 0.3
        assert history.should_stop_early(patience=2, min_iterations=2) is True
