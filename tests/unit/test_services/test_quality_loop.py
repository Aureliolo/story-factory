"""Tests for the generic quality refinement loop.

Tests cover:
- Basic create-judge-return flow
- Refinement when below threshold
- Best iteration tracking and return
- Early stopping (plateau and degradation)
- Review mode (initial_entity provided)
- Error handling (WorldGenerationError during iterations)
- Unchanged output detection (#246)
"""

import logging
from unittest.mock import MagicMock

import pytest

from src.memory.world_quality import (
    CharacterQualityScores,
    RefinementConfig,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def mock_svc():
    """Create a mock WorldQualityService with analytics logging."""
    svc = MagicMock()
    svc._log_refinement_analytics = MagicMock()
    return svc


@pytest.fixture
def config():
    """Create a RefinementConfig with test defaults."""
    return RefinementConfig(
        quality_threshold=8.0,
        max_iterations=5,
        creator_temperature=0.9,
        judge_temperature=0.1,
        refinement_temperature=0.7,
        early_stopping_patience=2,
        early_stopping_min_iterations=2,
        early_stopping_variance_tolerance=0.3,
    )


def _make_scores(avg: float, feedback: str = "Test") -> CharacterQualityScores:
    """Create CharacterQualityScores where all dimensions equal avg."""
    return CharacterQualityScores(
        depth=avg,
        goals=avg,
        flaws=avg,
        uniqueness=avg,
        arc_potential=avg,
        feedback=feedback,
    )


class TestQualityLoopBasicFlow:
    """Test basic create-judge-return flow."""

    def test_creates_and_judges_above_threshold(self, mock_svc, config):
        """Create → judge → return immediately when above threshold."""
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        result_entity, result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entity,
            judge_fn=lambda e: scores,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        assert result_entity == entity
        assert result_scores is scores
        assert iterations == 1
        mock_svc._log_refinement_analytics.assert_called_once()

    def test_refines_below_threshold(self, mock_svc, config):
        """Below threshold → refine → re-judge → return when above."""
        original = {"name": "Hero"}
        refined = {"name": "Hero v2"}
        low_scores = _make_scores(6.0)
        high_scores = _make_scores(8.5)

        call_count = 0

        def judge_fn(entity):
            """Return low scores on first call, high scores on second."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return low_scores
            return high_scores

        result_entity, result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: original,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: refined,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        assert result_entity == refined
        assert result_scores is high_scores
        assert iterations == 2

    def test_returns_best_iteration_when_degrading(self, mock_svc, config):
        """Three iterations with peak at iteration 2, should return best."""
        config.max_iterations = 3
        config.early_stopping_patience = 5  # Disable early stopping

        entities = [{"name": "v1"}, {"name": "v2"}, {"name": "v3"}]
        scores_list = [_make_scores(6.0), _make_scores(7.5), _make_scores(5.0)]
        iteration_idx = 0

        def create_fn(retries):
            """Return the first entity."""
            return entities[0]

        def judge_fn(entity):
            """Return scores in sequence: 6.0, 7.5, 5.0."""
            nonlocal iteration_idx
            result = scores_list[iteration_idx]
            iteration_idx += 1
            return result

        refine_idx = 0

        def refine_fn(entity, scores, iteration):
            """Return the next entity version from the list."""
            nonlocal refine_idx
            refine_idx += 1
            return entities[refine_idx]

        result_entity, _result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Best iteration was iteration 2 (score 7.5)
        assert result_entity == {"name": "v2"}
        assert _result_scores.average == 7.5
        assert iterations == 2

    def test_returns_best_when_all_scores_equal(self, mock_svc, config):
        """When all scores are equal, returns earliest (best_iteration=1)."""
        config.max_iterations = 3
        config.early_stopping_patience = 5

        entities = [{"name": "v1"}, {"name": "v2"}, {"name": "v3"}]
        iter_idx = 0

        def create_fn(retries):
            """Return the first entity."""
            return entities[0]

        def judge_fn(entity):
            """Return constant score of 7.0 for every entity."""
            return _make_scores(7.0)

        def refine_fn(entity, scores, iteration):
            """Return the next entity version from the list."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        result_entity, _result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Best iteration is 1 (first is always best when equal)
        assert result_entity == {"name": "v1"}


class TestQualityLoopEarlyStopping:
    """Test early stopping behavior."""

    def test_early_stop_plateau(self, mock_svc, config):
        """Flat scores trigger plateau early stop."""
        config.max_iterations = 5
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def create_fn(retries):
            """Return the first entity."""
            return entities[0]

        def judge_fn(entity):
            """Return constant score of 6.0 to simulate plateau."""
            return _make_scores(6.0)

        def refine_fn(entity, scores, iteration):
            """Return the next entity version from the list."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        _result_entity, _result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Should stop early, not run all 5 iterations
        assert iterations <= 3  # 3 iterations: [6.0, 6.0, 6.0] → 2 consecutive plateaus
        # Verify early_stop_triggered=True in analytics call
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True

    def test_early_stop_degradation(self, mock_svc, config):
        """Declining scores trigger degradation early stop."""
        config.max_iterations = 5
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 2

        scores_list = [_make_scores(7.0), _make_scores(6.0), _make_scores(5.0)]
        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0
        judge_idx = 0

        def create_fn(retries):
            """Return the first entity."""
            return entities[0]

        def judge_fn(entity):
            """Return declining scores: 7.0, 6.0, 5.0."""
            nonlocal judge_idx
            result = scores_list[min(judge_idx, len(scores_list) - 1)]
            judge_idx += 1
            return result

        def refine_fn(entity, scores, iteration):
            """Return the next entity version from the list."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        _result_entity, result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Best iteration should be 1 (score 7.0)
        assert result_scores.average == 7.0
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True


class TestQualityLoopReviewMode:
    """Test review mode (initial_entity provided)."""

    def test_review_mode_skips_creation(self, mock_svc, config):
        """When initial_entity is provided, creation is skipped."""
        entity = {"name": "Existing Hero"}
        scores = _make_scores(8.5)
        create_fn = MagicMock()

        result_entity, _result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=lambda e: scores,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            initial_entity=entity,
        )

        create_fn.assert_not_called()
        assert result_entity == entity
        assert iterations == 1

    def test_review_mode_refines_below_threshold(self, mock_svc, config):
        """Review mode: entity below threshold gets refined."""
        original = {"name": "Weak Hero"}
        refined = {"name": "Strong Hero"}
        low_scores = _make_scores(5.0)
        high_scores = _make_scores(9.0)
        judge_count = 0

        def judge_fn(entity):
            """Return low scores on first call, high scores on second."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return low_scores
            return high_scores

        result_entity, result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=MagicMock(),
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: refined,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            initial_entity=original,
        )

        assert result_entity == refined
        assert result_scores is high_scores
        assert iterations == 2


class TestQualityLoopErrorHandling:
    """Test error handling during iterations."""

    def test_handles_generation_errors_continues_loop(self, mock_svc, config):
        """WorldGenerationError during judge doesn't abort entire loop."""
        config.max_iterations = 3
        entity = {"name": "Hero"}
        good_scores = _make_scores(8.5)
        judge_count = 0

        def judge_fn(e):
            """Raise WorldGenerationError on first call, return good scores after."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                raise WorldGenerationError("First judge failed")
            return good_scores

        _result_entity, result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entity,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        assert result_scores is good_scores

    def test_all_iterations_fail_raises_error(self, mock_svc, config):
        """When every iteration fails, WorldGenerationError is raised."""
        config.max_iterations = 3

        def always_fail_judge(e):
            """Always raise WorldGenerationError."""
            raise WorldGenerationError("Always fails")

        with pytest.raises(WorldGenerationError, match="Failed to generate character"):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: {"name": "Hero"},
                judge_fn=always_fail_judge,
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

    def test_empty_creation_retries(self, mock_svc, config):
        """Empty creation results in retries until a valid entity is produced."""
        config.max_iterations = 4
        create_count = 0

        def create_fn(retries):
            """Return empty entity twice, then a valid entity on third call."""
            nonlocal create_count
            create_count += 1
            if create_count <= 2:
                return {"name": ""}  # Empty name → is_empty returns True
            return {"name": "Valid Hero"}

        result_entity, _result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=lambda e: _make_scores(8.5),
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        assert result_entity == {"name": "Valid Hero"}
        assert create_count == 3  # 2 empty + 1 valid

    def test_empty_refinement_retries(self, mock_svc, config):
        """When refine_fn returns an empty entity, the loop retries via creation."""
        config.max_iterations = 4
        config.early_stopping_patience = 10  # Disable early stopping

        create_count = 0

        def create_fn(retries):
            """Return a valid entity each time creation is called."""
            nonlocal create_count
            create_count += 1
            return {"name": f"Hero v{create_count}"}

        judge_count = 0

        def judge_fn(entity):
            """Return low scores first, then high scores to exit the loop."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return _make_scores(5.0, feedback="Needs improvement")
            return _make_scores(9.0, feedback="Excellent")

        def refine_fn(entity, scores, iteration):
            """Return an empty entity to trigger the retry-via-creation path."""
            return {"name": ""}

        _result_entity, result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # The loop should have: created Hero v1 -> judged low -> refined empty ->
        # retried creation as Hero v2 -> judged high -> returned
        assert result_scores.average == 9.0
        assert create_count >= 2  # At least initial create + retry after empty refine

    def test_judge_error_resets_scores_prevents_stale_refinement(self, mock_svc, config):
        """After a judge error, scores are reset so next iteration re-judges instead of refining."""
        config.max_iterations = 4
        config.early_stopping_patience = 10  # Disable early stopping

        judge_count = 0
        refine_calls = []

        def judge_fn(entity):
            """Succeed, then fail, then succeed above threshold."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return _make_scores(5.0, feedback="Needs work")
            if judge_count == 2:
                raise WorldGenerationError("Judge temporarily unavailable")
            return _make_scores(9.0, feedback="Great")

        def refine_fn(entity, scores, iteration):
            """Track which scores are used for refinement."""
            refine_calls.append(scores.feedback)
            return {"name": "Refined Hero"}

        _result_entity, result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        assert result_scores.average == 9.0
        # Iteration 0: create → judge OK (5.0) → scores set
        # Iteration 1: refine with scores → judge FAILS → scores reset to None
        # Iteration 2: scores is None → skip refinement → re-judge same entity → OK (9.0)
        # Refine should only be called once (iteration 1), not on iteration 2
        assert len(refine_calls) == 1
        assert refine_calls[0] == "Needs work"


class TestQualityLoopUnchangedOutput:
    """Test unchanged refinement output detection (#246)."""

    def test_unchanged_output_breaks_loop_early(self, mock_svc, config):
        """When refine returns identical dict, loop breaks and judge is called only once."""
        config.max_iterations = 5
        config.early_stopping_patience = 10  # Disable normal early stopping

        entity = {"name": "Hero", "trait": "brave"}
        low_scores = _make_scores(6.0)
        judge_calls = 0

        def judge_fn(e):
            """Track judge calls."""
            nonlocal judge_calls
            judge_calls += 1
            return low_scores

        def refine_fn(e, s, i):
            """Return identical entity (unchanged output)."""
            return {"name": "Hero", "trait": "brave"}

        _result_entity, result_scores, _iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entity,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Judge should only be called once (iteration 0), then refine returns
        # unchanged output on iteration 1 and loop breaks before re-judging
        assert judge_calls == 1
        assert result_scores.average == 6.0
        # Analytics should record early_stop_triggered=True
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True

    def test_unchanged_output_does_not_trigger_on_first_iteration(self, mock_svc, config):
        """Unchanged detection only fires after iteration 0 has history."""
        config.max_iterations = 2

        entity = {"name": "Hero"}
        high_scores = _make_scores(9.0)

        _result_entity, result_scores, iterations = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entity,
            judge_fn=lambda e: high_scores,
            refine_fn=lambda e, s, i: entity,  # Same entity, but irrelevant
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Should return on first iteration via threshold, not via unchanged detection
        assert iterations == 1
        assert result_scores.average == 9.0

    def test_unchanged_output_different_entities_continues(self, mock_svc, config):
        """Different dicts from refine don't trigger unchanged detection."""
        config.max_iterations = 3
        config.early_stopping_patience = 10  # Disable normal early stopping

        entities = [{"name": "Hero v1"}, {"name": "Hero v2"}, {"name": "Hero v3"}]
        scores_list = [_make_scores(6.0), _make_scores(6.5), _make_scores(7.0)]
        iter_idx = 0
        judge_idx = 0

        def create_fn(retries):
            """Return first entity."""
            return entities[0]

        def judge_fn(e):
            """Return scores in sequence."""
            nonlocal judge_idx
            result = scores_list[min(judge_idx, len(scores_list) - 1)]
            judge_idx += 1
            return result

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[min(iter_idx, len(entities) - 1)]

        quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # All 3 iterations should run (no unchanged detection)
        assert judge_idx == 3

    def test_unchanged_output_logs_info(self, mock_svc, config, caplog):
        """Verify log message is emitted when unchanged output is detected."""
        config.max_iterations = 5
        config.early_stopping_patience = 10

        entity = {"name": "Hero", "trait": "brave"}
        low_scores = _make_scores(6.0)

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: entity,
                judge_fn=lambda e: low_scores,
                refine_fn=lambda e, s, i: {"name": "Hero", "trait": "brave"},
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("refinement produced unchanged output" in msg for msg in caplog.messages)
