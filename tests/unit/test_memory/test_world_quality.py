"""Tests for world quality models."""

from src.memory.world_quality import (
    FactionQualityScores,
    RefinementHistory,
)


class TestRefinementHistory:
    """Tests for RefinementHistory model."""

    def test_get_best_entity_returns_none_when_best_iteration_is_zero(self):
        """Test get_best_entity returns None when best_iteration is 0."""
        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=0,  # No best iteration set
        )
        result = history.get_best_entity()
        assert result is None

    def test_get_best_entity_returns_none_when_no_iterations(self):
        """Test get_best_entity returns None with empty iterations."""
        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=1,  # Set to 1 but no iterations
            iterations=[],
        )
        result = history.get_best_entity()
        assert result is None

    def test_get_best_entity_returns_none_when_iteration_not_found(self):
        """Test get_best_entity returns None when best_iteration not in list."""
        from src.memory.world_quality import IterationRecord

        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=5,  # Points to iteration 5
            iterations=[
                IterationRecord(
                    iteration=1,  # Only iteration 1 exists
                    entity_data={"name": "Test"},
                    scores={"depth": 5.0},
                    average_score=5.0,
                ),
                IterationRecord(
                    iteration=2,
                    entity_data={"name": "Test2"},
                    scores={"depth": 6.0},
                    average_score=6.0,
                ),
            ],
        )
        # best_iteration=5 but no iteration with that number exists
        result = history.get_best_entity()
        assert result is None

    def test_get_best_entity_returns_correct_entity(self):
        """Test get_best_entity returns entity from best iteration."""
        from src.memory.world_quality import IterationRecord

        history = RefinementHistory(
            entity_type="character",
            entity_name="Test Character",
            best_iteration=2,
            iterations=[
                IterationRecord(
                    iteration=1,
                    entity_data={"name": "First"},
                    scores={"depth": 5.0},
                    average_score=5.0,
                ),
                IterationRecord(
                    iteration=2,
                    entity_data={"name": "Second"},
                    scores={"depth": 8.0},
                    average_score=8.0,
                ),
            ],
        )
        result = history.get_best_entity()
        assert result == {"name": "Second"}


class TestFactionQualityScores:
    """Tests for FactionQualityScores model."""

    def test_weak_dimensions_identifies_conflict_potential(self):
        """Test weak_dimensions identifies low conflict_potential."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=5.0,  # Below threshold
            distinctiveness=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "conflict_potential" in weak
        assert "coherence" not in weak
        assert "influence" not in weak
        assert "distinctiveness" not in weak

    def test_weak_dimensions_identifies_distinctiveness(self):
        """Test weak_dimensions identifies low distinctiveness."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=5.0,  # Below threshold
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "distinctiveness" in weak
        assert "coherence" not in weak
        assert "influence" not in weak
        assert "conflict_potential" not in weak

    def test_weak_dimensions_identifies_multiple(self):
        """Test weak_dimensions identifies multiple low dimensions."""
        scores = FactionQualityScores(
            coherence=5.0,
            influence=5.0,
            conflict_potential=5.0,
            distinctiveness=5.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert len(weak) == 4
        assert "coherence" in weak
        assert "influence" in weak
        assert "conflict_potential" in weak
        assert "distinctiveness" in weak

    def test_weak_dimensions_returns_empty_when_all_above_threshold(self):
        """Test weak_dimensions returns empty list when all scores high."""
        scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert weak == []
