"""Unit tests for refinement diagnostic utility functions.

Tests the helper functions used by the diagnostic scripts in scripts/.
These tests do NOT require a running Ollama instance.
"""

from scripts.evaluate_ab_prompts import DEFAULT_AB_TYPES, IMPROVED_REFINERS
from scripts.evaluate_judge_consistency import (
    CONSISTENT_STD_THRESHOLD,
    NOISY_STD_THRESHOLD,
    compute_feedback_similarity,
    compute_statistics,
    determine_verdict,
)
from scripts.evaluate_refinement import (
    ALL_ENTITY_TYPES,
    SCORE_DIMENSIONS,
    compute_description_diff,
    compute_feedback_specificity,
    compute_summary,
    extract_description,
    make_canonical_brief,
    make_story_state,
)


class TestComputeDescriptionDiff:
    """Tests for compute_description_diff()."""

    def test_identical_strings_return_zero(self):
        """Identical strings should have zero diff."""
        assert compute_description_diff("hello world", "hello world") == 0.0

    def test_completely_different_strings_return_high(self):
        """Completely different strings should have high diff ratio."""
        diff = compute_description_diff("aaaa", "zzzz")
        assert diff > 0.8

    def test_empty_strings_return_zero(self):
        """Two empty strings should have zero diff."""
        assert compute_description_diff("", "") == 0.0

    def test_one_empty_returns_one(self):
        """One empty string vs non-empty should return 1.0."""
        diff = compute_description_diff("", "some text here")
        assert diff == 1.0

    def test_similar_strings_return_low(self):
        """Similar strings with minor changes should return low diff."""
        diff = compute_description_diff(
            "The dark fortress stands on the hill",
            "The dark fortress stands on the mountain",
        )
        assert 0.0 < diff < 0.5

    def test_return_is_rounded(self):
        """Diff ratio should be rounded to 4 decimal places."""
        diff = compute_description_diff("abc", "abd")
        assert diff == round(diff, 4)

    def test_both_directions_similar(self):
        """SequenceMatcher is not perfectly symmetric, but results should be close."""
        a = "The ancient library holds many secrets"
        b = "The library was rebuilt with new technology"
        diff_ab = compute_description_diff(a, b)
        diff_ba = compute_description_diff(b, a)
        assert abs(diff_ab - diff_ba) < 0.1
        assert diff_ab > 0.3
        assert diff_ba > 0.3


class TestExtractDescription:
    """Tests for extract_description()."""

    def test_returns_description_from_dict(self):
        """Should extract description field from entity dict."""
        data = {"name": "Test", "description": "A great place"}
        assert extract_description(data, "location") == "A great place"

    def test_returns_empty_for_none(self):
        """Should return empty string for None entity data."""
        assert extract_description(None, "faction") == ""

    def test_returns_empty_for_missing_key(self):
        """Should return empty string when description key is absent."""
        assert extract_description({"name": "Test"}, "faction") == ""

    def test_relationship_includes_source_target(self):
        """Relationship extraction should include source, target, and relation type."""
        data = {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "allies_with",
            "description": "Old friends",
        }
        result = extract_description(data, "relationship")
        assert "Alice" in result
        assert "Bob" in result
        assert "allies_with" in result
        assert "Old friends" in result


class TestComputeFeedbackSpecificity:
    """Tests for compute_feedback_specificity()."""

    def test_empty_feedback_returns_zero(self):
        """Empty feedback string should return 0.0 specificity."""
        assert compute_feedback_specificity("", {"name": "Test"}) == 0.0

    def test_generic_feedback_returns_low(self):
        """Generic feedback without entity-specific words should score low."""
        feedback = "This is good but could be better overall"
        entity = {"name": "Iron Covenant", "description": "A secretive military order"}
        specificity = compute_feedback_specificity(feedback, entity)
        assert specificity < 0.5

    def test_specific_feedback_returns_higher(self):
        """Feedback referencing entity-specific words should score higher."""
        entity = {
            "name": "Iron Covenant",
            "description": "A secretive military order protecting ancient secrets",
            "goals": ["Protect the archive", "Eliminate memory thieves"],
        }
        feedback = (
            "The Iron Covenant lacks detail about its military structure and "
            "secretive practices. The archive protection goal needs more specificity."
        )
        specificity = compute_feedback_specificity(feedback, entity)
        assert specificity > 0.0

    def test_returns_rounded(self):
        """Specificity should be rounded to 4 decimal places."""
        result = compute_feedback_specificity(
            "interesting concept", {"name": "Test", "description": "A concept"}
        )
        assert result == round(result, 4)


class TestComputeSummary:
    """Tests for compute_summary()."""

    def test_empty_results(self):
        """Empty results list should produce empty summary dicts."""
        summary = compute_summary([])
        assert summary["pass_rate_by_type"] == {}

    def test_single_type_all_pass(self):
        """All entities passing should give 100% pass rate."""
        results = [
            {
                "entity_type": "faction",
                "threshold_met": True,
                "score_progression": [6.5, 7.5, 8.0],
                "description_diff_ratios": [None, 0.3, 0.2],
                "feedback_strings": ["Needs work", "Better", "Good"],
                "feedback_specificity_scores": [0.1, 0.2, 0.3],
            },
            {
                "entity_type": "faction",
                "threshold_met": True,
                "score_progression": [7.0, 8.0],
                "description_diff_ratios": [None, 0.25],
                "feedback_strings": ["OK", "Great"],
                "feedback_specificity_scores": [0.15, 0.25],
            },
        ]
        summary = compute_summary(results)
        assert summary["pass_rate_by_type"]["faction"] == 1.0

    def test_mixed_pass_fail(self):
        """Mix of pass and fail should give correct pass rate."""
        results = [
            {
                "entity_type": "concept",
                "threshold_met": True,
                "score_progression": [7.5],
                "description_diff_ratios": [None],
                "feedback_strings": ["Good"],
                "feedback_specificity_scores": [0.2],
            },
            {
                "entity_type": "concept",
                "threshold_met": False,
                "score_progression": [6.0, 6.5],
                "description_diff_ratios": [None, 0.1],
                "feedback_strings": ["Weak", "Still weak"],
                "feedback_specificity_scores": [0.1, 0.1],
            },
        ]
        summary = compute_summary(results)
        assert summary["pass_rate_by_type"]["concept"] == 0.5

    def test_plateau_rate_computation(self):
        """Score changes below 0.3 should count as plateaus."""
        results = [
            {
                "entity_type": "item",
                "threshold_met": False,
                "score_progression": [6.5, 6.5, 6.6],
                "description_diff_ratios": [None, 0.1, 0.05],
                "feedback_strings": ["OK", "Same", "Same"],
                "feedback_specificity_scores": [0.1, 0.1, 0.1],
            },
        ]
        summary = compute_summary(results)
        # delta[0] = 0.0 (< 0.3 -> plateau), delta[1] = 0.1 (< 0.3 -> plateau)
        # 2 plateaus out of 2 refinement iterations = 1.0
        assert summary["plateau_rate"]["item"] == 1.0

    def test_regression_rate_computation(self):
        """Negative score deltas should count as regressions."""
        results = [
            {
                "entity_type": "location",
                "threshold_met": False,
                "score_progression": [7.0, 6.5, 6.0],
                "description_diff_ratios": [None, 0.3, 0.2],
                "feedback_strings": ["OK", "Worse", "Bad"],
                "feedback_specificity_scores": [0.1, 0.1, 0.1],
            },
        ]
        summary = compute_summary(results)
        assert summary["regression_rate"]["location"] == 1.0

    def test_multiple_types_in_results(self):
        """Summary should separate metrics per entity type."""
        results = [
            {
                "entity_type": "faction",
                "threshold_met": True,
                "score_progression": [8.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Great"],
                "feedback_specificity_scores": [0.3],
            },
            {
                "entity_type": "concept",
                "threshold_met": False,
                "score_progression": [5.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Weak"],
                "feedback_specificity_scores": [0.1],
            },
        ]
        summary = compute_summary(results)
        assert "faction" in summary["pass_rate_by_type"]
        assert "concept" in summary["pass_rate_by_type"]
        assert summary["pass_rate_by_type"]["faction"] == 1.0
        assert summary["pass_rate_by_type"]["concept"] == 0.0


class TestStatistics:
    """Tests for compute_statistics() from judge consistency script."""

    def test_empty_values(self):
        """Empty list should return zeroed statistics."""
        stats = compute_statistics([])
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_single_value(self):
        """Single value should return it as mean with zero std."""
        stats = compute_statistics([7.5])
        assert stats["mean"] == 7.5
        assert stats["std"] == 0.0
        assert stats["cv"] == 0.0

    def test_identical_values(self):
        """Identical values should have zero std and cv."""
        stats = compute_statistics([6.0, 6.0, 6.0])
        assert stats["mean"] == 6.0
        assert stats["std"] == 0.0
        assert stats["cv"] == 0.0

    def test_varied_values(self):
        """Varied values should have non-zero std, correct min/max."""
        stats = compute_statistics([5.0, 7.0, 9.0])
        assert stats["mean"] == 7.0
        assert stats["std"] > 0
        assert stats["min"] == 5.0
        assert stats["max"] == 9.0
        assert stats["cv"] > 0

    def test_cv_computation(self):
        """CV should be zero when all values are identical."""
        stats = compute_statistics([10.0, 10.0, 10.0])
        assert stats["cv"] == 0.0


class TestFeedbackSimilarity:
    """Tests for compute_feedback_similarity()."""

    def test_single_feedback(self):
        """Single feedback should return perfect similarity."""
        assert compute_feedback_similarity(["hello"]) == 1.0

    def test_identical_feedbacks(self):
        """Identical feedback strings should return 1.0."""
        result = compute_feedback_similarity(["hello world", "hello world"])
        assert result == 1.0

    def test_completely_different(self):
        """Completely different word sets should return 0.0."""
        result = compute_feedback_similarity(["alpha beta gamma", "delta epsilon zeta"])
        assert result == 0.0

    def test_partially_overlapping(self):
        """Partially overlapping word sets should return between 0 and 1."""
        result = compute_feedback_similarity(
            [
                "the faction lacks coherence",
                "the faction needs more distinctiveness",
            ]
        )
        assert 0.0 < result < 1.0

    def test_empty_list(self):
        """Empty feedback list should return 1.0."""
        result = compute_feedback_similarity([])
        assert result == 1.0


class TestDetermineVerdict:
    """Tests for determine_verdict()."""

    def test_consistent_when_low_std(self):
        """Low std across all dimensions should yield consistent verdict."""
        per_dim = {
            "coherence": {"mean": 7.0, "std": 0.1, "min": 6.9, "max": 7.1, "cv": 0.014},
            "influence": {"mean": 6.5, "std": 0.15, "min": 6.3, "max": 6.7, "cv": 0.023},
        }
        assert determine_verdict(per_dim) == "consistent"

    def test_noisy_when_high_std(self):
        """High std should yield noisy verdict."""
        per_dim = {
            "coherence": {"mean": 7.0, "std": 0.8, "min": 5.5, "max": 8.5, "cv": 0.114},
            "influence": {"mean": 6.5, "std": 0.6, "min": 5.5, "max": 7.5, "cv": 0.092},
        }
        assert determine_verdict(per_dim) == "noisy"

    def test_borderline_when_moderate_std(self):
        """Moderate std should yield borderline verdict."""
        per_dim = {
            "coherence": {"mean": 7.0, "std": 0.3, "min": 6.5, "max": 7.5, "cv": 0.043},
            "influence": {"mean": 6.5, "std": 0.35, "min": 6.0, "max": 7.0, "cv": 0.054},
        }
        assert determine_verdict(per_dim) == "borderline"

    def test_empty_dimensions(self):
        """Empty dimensions dict should default to consistent."""
        assert determine_verdict({}) == "consistent"

    def test_single_noisy_dimension_makes_noisy(self):
        """One noisy dimension should make the overall verdict noisy."""
        per_dim = {
            "coherence": {"mean": 7.0, "std": 0.1, "min": 6.9, "max": 7.1, "cv": 0.014},
            "influence": {"mean": 6.5, "std": 0.6, "min": 5.5, "max": 7.5, "cv": 0.092},
        }
        assert determine_verdict(per_dim) == "noisy"


class TestCanonicalBrief:
    """Tests for canonical brief and story state creation."""

    def test_make_canonical_brief_has_all_fields(self):
        """Canonical brief should have all required StoryBrief fields."""
        brief = make_canonical_brief()
        assert brief.premise
        assert brief.genre == "Fantasy"
        assert len(brief.themes) >= 2
        assert brief.setting_place
        assert brief.setting_time
        assert brief.language == "English"

    def test_make_story_state_wraps_brief(self):
        """Story state should wrap the brief and use diagnostic ID."""
        brief = make_canonical_brief()
        state = make_story_state(brief)
        assert state.brief is brief
        assert state.id == "diagnostic-run"

    def test_all_entity_types_list(self):
        """ALL_ENTITY_TYPES should contain all 6 entity types."""
        expected = {"character", "faction", "location", "item", "concept", "relationship"}
        assert set(ALL_ENTITY_TYPES) == expected

    def test_score_dimensions_covers_all_types(self):
        """Every entity type should have at least 4 score dimensions defined."""
        for et in ALL_ENTITY_TYPES:
            assert et in SCORE_DIMENSIONS, f"Missing score dimensions for {et}"
            assert len(SCORE_DIMENSIONS[et]) >= 4, f"Too few dimensions for {et}"


class TestComputeSummaryEdgeCases:
    """Edge case tests for summary computation."""

    def test_no_refinement_iterations(self):
        """When all entities pass on first try, refinement metrics should handle gracefully."""
        results = [
            {
                "entity_type": "character",
                "threshold_met": True,
                "score_progression": [8.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Excellent"],
                "feedback_specificity_scores": [0.5],
            },
        ]
        summary = compute_summary(results)
        assert summary["avg_score_improvement_per_iteration"]["character"] == 0.0
        assert summary["plateau_rate"]["character"] == 0.0
        assert summary["regression_rate"]["character"] == 0.0

    def test_all_none_diff_ratios(self):
        """When all diff ratios are None (single iterations), average should be 0."""
        results = [
            {
                "entity_type": "item",
                "threshold_met": True,
                "score_progression": [8.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Good"],
                "feedback_specificity_scores": [0.3],
            },
        ]
        summary = compute_summary(results)
        assert summary["avg_description_diff_ratio"]["item"] == 0.0


class TestABPromptConstants:
    """Tests for A/B prompt script constants and structure."""

    def test_default_ab_types_are_valid(self):
        """All default A/B types must be in the global entity type list."""
        for et in DEFAULT_AB_TYPES:
            assert et in ALL_ENTITY_TYPES, f"{et} not in ALL_ENTITY_TYPES"

    def test_improved_refiners_cover_default_types(self):
        """Every default A/B type must have an improved refiner function."""
        for et in DEFAULT_AB_TYPES:
            assert et in IMPROVED_REFINERS, f"Missing improved refiner for {et}"

    def test_improved_refiners_are_callable(self):
        """All improved refiners must be callable functions."""
        for et, func in IMPROVED_REFINERS.items():
            assert callable(func), f"Refiner for {et} is not callable"

    def test_relationship_excluded_from_ab(self):
        """Relationship type should not be in A/B defaults (no improved refiner)."""
        assert "relationship" not in DEFAULT_AB_TYPES
        assert "relationship" not in IMPROVED_REFINERS


class TestStatisticsEdgeCases:
    """Additional edge case tests for compute_statistics()."""

    def test_two_values_uses_sample_std(self):
        """With 2 values, should use sample std (n-1 denominator)."""
        stats = compute_statistics([4.0, 6.0])
        assert stats["mean"] == 5.0
        # Sample std of [4, 6] = sqrt(((4-5)^2 + (6-5)^2) / 1) = sqrt(2) ~ 1.414
        assert abs(stats["std"] - 1.414) < 0.01

    def test_cv_with_zero_mean(self):
        """Coefficient of variation should be 0 when mean is 0."""
        stats = compute_statistics([0.0, 0.0])
        assert stats["cv"] == 0.0

    def test_all_fields_present(self):
        """Statistics dict should always contain mean, std, min, max, cv."""
        stats = compute_statistics([3.0, 5.0, 7.0])
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "cv" in stats

    def test_values_are_rounded(self):
        """Statistics should be rounded to 3 decimal places."""
        stats = compute_statistics([1.0, 2.0, 3.0])
        assert stats["mean"] == round(stats["mean"], 3)
        assert stats["std"] == round(stats["std"], 3)
        assert stats["cv"] == round(stats["cv"], 3)


class TestFeedbackSimilarityEdgeCases:
    """Additional edge case tests for compute_feedback_similarity()."""

    def test_three_feedbacks_all_identical(self):
        """Three identical feedbacks should return 1.0."""
        result = compute_feedback_similarity(
            ["same words here", "same words here", "same words here"]
        )
        assert result == 1.0

    def test_three_feedbacks_mixed(self):
        """Mixed feedbacks with partial overlap should return intermediate value."""
        result = compute_feedback_similarity(
            [
                "the faction needs coherence",
                "the faction lacks depth",
                "completely unrelated words about nothing",
            ]
        )
        assert 0.0 < result < 1.0

    def test_empty_string_feedbacks(self):
        """Empty strings should be filtered out, returning 1.0."""
        result = compute_feedback_similarity(["", ""])
        assert result == 1.0

    def test_result_is_rounded(self):
        """Similarity result should be rounded to 3 decimal places."""
        result = compute_feedback_similarity(["alpha beta gamma delta", "alpha beta epsilon zeta"])
        assert result == round(result, 3)


class TestComputeFeedbackSpecificityEdgeCases:
    """Additional edge case tests for compute_feedback_specificity()."""

    def test_entity_with_list_fields(self):
        """Specificity should use list fields like goals and values."""
        entity = {
            "name": "Test",
            "description": "A faction",
            "goals": ["dominate trade", "control ports"],
            "values": ["loyalty", "discipline"],
        }
        feedback = "The faction should clarify its trade domination goals and loyalty values"
        specificity = compute_feedback_specificity(feedback, entity)
        assert specificity > 0.0

    def test_all_stop_words_feedback(self):
        """Feedback with only stop words should return 0.0 after filtering."""
        entity = {"name": "X", "description": "Y"}
        feedback = "the a an is are was were be been being"
        specificity = compute_feedback_specificity(feedback, entity)
        assert specificity == 0.0

    def test_entity_with_no_matching_fields(self):
        """When entity has no text-extractable keys, specificity should be 0.0."""
        entity = {"id": 123, "type": "custom"}
        feedback = "interesting concept with good potential"
        specificity = compute_feedback_specificity(feedback, entity)
        assert specificity == 0.0


class TestExtractDescriptionEdgeCases:
    """Additional edge case tests for extract_description()."""

    def test_relationship_with_missing_fields(self):
        """Relationship with only partial fields should still work."""
        data = {"source": "Alice", "description": "Rivals"}
        result = extract_description(data, "relationship")
        assert "Alice" in result
        assert "Rivals" in result

    def test_empty_description(self):
        """Entity with empty description returns empty string."""
        assert extract_description({"name": "Test", "description": ""}, "faction") == ""

    def test_non_relationship_ignores_source_target(self):
        """Non-relationship entities should just return description."""
        data = {"name": "X", "description": "A place", "source": "should_not_appear"}
        result = extract_description(data, "location")
        assert result == "A place"
        assert "should_not_appear" not in result


class TestComputeSummaryMetrics:
    """Tests for summary metrics beyond pass rate."""

    def test_avg_score_improvement(self):
        """Average score improvement per iteration should be computed correctly."""
        results = [
            {
                "entity_type": "faction",
                "threshold_met": True,
                "score_progression": [5.0, 7.0, 8.0],
                "description_diff_ratios": [None, 0.3, 0.2],
                "feedback_strings": ["Bad", "Better", "Good"],
                "feedback_specificity_scores": [0.1, 0.2, 0.3],
            },
        ]
        summary = compute_summary(results)
        # Improvements: 7.0-5.0=2.0, 8.0-7.0=1.0. Average = 1.5
        assert summary["avg_score_improvement_per_iteration"]["faction"] == 1.5

    def test_avg_feedback_length(self):
        """Average feedback length should count words correctly."""
        results = [
            {
                "entity_type": "item",
                "threshold_met": True,
                "score_progression": [7.5],
                "description_diff_ratios": [None],
                "feedback_strings": ["one two three four"],
                "feedback_specificity_scores": [0.2],
            },
        ]
        summary = compute_summary(results)
        assert summary["avg_feedback_length"]["item"] == 4.0

    def test_avg_description_diff_ratio(self):
        """Average description diff ratio should skip None values."""
        results = [
            {
                "entity_type": "concept",
                "threshold_met": False,
                "score_progression": [5.0, 6.0, 6.5],
                "description_diff_ratios": [None, 0.4, 0.2],
                "feedback_strings": ["Bad", "OK", "Better"],
                "feedback_specificity_scores": [0.1, 0.1, 0.1],
            },
        ]
        summary = compute_summary(results)
        # Average of [0.4, 0.2] = 0.3
        assert summary["avg_description_diff_ratio"]["concept"] == 0.3

    def test_avg_feedback_specificity(self):
        """Average feedback specificity should average across all entities of a type."""
        results = [
            {
                "entity_type": "location",
                "threshold_met": True,
                "score_progression": [8.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Good"],
                "feedback_specificity_scores": [0.4],
            },
            {
                "entity_type": "location",
                "threshold_met": False,
                "score_progression": [5.0],
                "description_diff_ratios": [None],
                "feedback_strings": ["Weak"],
                "feedback_specificity_scores": [0.2],
            },
        ]
        summary = compute_summary(results)
        # Average of [0.4, 0.2] = 0.3
        assert summary["avg_feedback_specificity"]["location"] == 0.3


class TestVerdictThresholds:
    """Tests verifying that verdict thresholds are correctly defined."""

    def test_noisy_threshold_is_greater_than_consistent(self):
        """Noisy threshold must be greater than consistent threshold."""
        assert NOISY_STD_THRESHOLD > CONSISTENT_STD_THRESHOLD

    def test_consistent_threshold_is_positive(self):
        """Consistent threshold must be positive."""
        assert CONSISTENT_STD_THRESHOLD > 0

    def test_verdict_at_exact_consistent_boundary(self):
        """At exactly the consistent threshold, verdict should be consistent."""
        per_dim = {
            "dim1": {"mean": 7.0, "std": CONSISTENT_STD_THRESHOLD - 0.01},
        }
        assert determine_verdict(per_dim) == "consistent"

    def test_verdict_at_exact_noisy_boundary(self):
        """At exactly the noisy threshold, verdict should be noisy."""
        per_dim = {
            "dim1": {"mean": 7.0, "std": NOISY_STD_THRESHOLD + 0.01},
        }
        assert determine_verdict(per_dim) == "noisy"
