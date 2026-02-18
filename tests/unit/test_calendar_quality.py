"""Tests for CalendarQualityScores model."""

import pytest

from src.memory.world_quality import CalendarQualityScores


class TestCalendarQualityScores:
    """Tests for CalendarQualityScores Pydantic model."""

    def test_construction_with_valid_scores(self):
        """Test construction with valid score values."""
        scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=7.5,
            completeness=9.0,
            uniqueness=6.5,
            feedback="Good calendar system.",
        )
        assert scores.internal_consistency == 8.0
        assert scores.thematic_fit == 7.5
        assert scores.completeness == 9.0
        assert scores.uniqueness == 6.5
        assert scores.feedback == "Good calendar system."

    def test_average_calculation(self):
        """Test average property computes mean of all four dimensions."""
        scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=6.0,
            completeness=10.0,
            uniqueness=4.0,
            feedback="Test",
        )
        # (8 + 6 + 10 + 4) / 4 = 7.0
        assert scores.average == 7.0

    def test_average_all_same(self):
        """Test average when all dimensions are the same."""
        scores = CalendarQualityScores(
            internal_consistency=7.5,
            thematic_fit=7.5,
            completeness=7.5,
            uniqueness=7.5,
            feedback="Uniform",
        )
        assert scores.average == 7.5

    def test_to_dict(self):
        """Test to_dict serialization includes all fields."""
        scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=7.0,
            completeness=9.0,
            uniqueness=6.0,
            feedback="Detailed feedback.",
        )
        d = scores.to_dict()
        assert d["internal_consistency"] == 8.0
        assert d["thematic_fit"] == 7.0
        assert d["completeness"] == 9.0
        assert d["uniqueness"] == 6.0
        assert d["average"] == 7.5
        assert d["feedback"] == "Detailed feedback."

    def test_weak_dimensions_below_threshold(self):
        """Test weak_dimensions returns dimensions below threshold."""
        scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=5.0,
            completeness=9.0,
            uniqueness=3.0,
            feedback="Needs work.",
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert "thematic_fit" in weak
        assert "uniqueness" in weak
        assert "internal_consistency" not in weak
        assert "completeness" not in weak

    def test_weak_dimensions_all_above_threshold(self):
        """Test weak_dimensions returns empty when all above threshold."""
        scores = CalendarQualityScores(
            internal_consistency=8.0,
            thematic_fit=8.0,
            completeness=8.0,
            uniqueness=8.0,
            feedback="All good.",
        )
        assert scores.weak_dimensions(threshold=7.0) == []

    def test_weak_dimensions_all_below_threshold(self):
        """Test weak_dimensions returns all dimensions when all below threshold."""
        scores = CalendarQualityScores(
            internal_consistency=3.0,
            thematic_fit=4.0,
            completeness=5.0,
            uniqueness=2.0,
            feedback="All weak.",
        )
        weak = scores.weak_dimensions(threshold=7.0)
        assert len(weak) == 4
        assert set(weak) == {
            "internal_consistency",
            "thematic_fit",
            "completeness",
            "uniqueness",
        }

    def test_weak_dimensions_custom_threshold(self):
        """Test weak_dimensions with a custom threshold."""
        scores = CalendarQualityScores(
            internal_consistency=8.5,
            thematic_fit=8.0,
            completeness=9.0,
            uniqueness=7.0,
            feedback="Good.",
        )
        # With threshold 8.5, only internal_consistency and completeness pass
        weak = scores.weak_dimensions(threshold=8.5)
        assert "thematic_fit" in weak
        assert "uniqueness" in weak
        assert "internal_consistency" not in weak
        assert "completeness" not in weak

    def test_boundary_score_values(self):
        """Test scores at boundary values (0 and 10)."""
        scores = CalendarQualityScores(
            internal_consistency=0.0,
            thematic_fit=10.0,
            completeness=0.0,
            uniqueness=10.0,
            feedback="Extreme.",
        )
        assert scores.average == 5.0
        weak = scores.weak_dimensions(threshold=5.0)
        assert "internal_consistency" in weak
        assert "completeness" in weak

    def test_validation_rejects_out_of_range(self):
        """Test that scores outside 0-10 are rejected by Pydantic."""
        with pytest.raises(ValueError):
            CalendarQualityScores(
                internal_consistency=11.0,
                thematic_fit=7.0,
                completeness=7.0,
                uniqueness=7.0,
                feedback="Invalid.",
            )

    def test_validation_rejects_negative(self):
        """Test that negative scores are rejected by Pydantic."""
        with pytest.raises(ValueError):
            CalendarQualityScores(
                internal_consistency=-1.0,
                thematic_fit=7.0,
                completeness=7.0,
                uniqueness=7.0,
                feedback="Invalid.",
            )
