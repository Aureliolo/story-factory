"""Tests for the generic quality refinement loop.

Tests cover:
- Basic create-judge-return flow
- Refinement when below threshold
- Best iteration tracking and return
- Early stopping (plateau and degradation)
- Review mode (initial_entity provided)
- Error handling (WorldGenerationError during iterations)
- Unchanged output detection (#246)
- Score rounding at threshold boundary (#303)
- WARNING for sub-threshold entities returned via best-iteration path (#303)
- Per-iteration timing instrumentation (#304)
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.memory.world_quality import (
    CalendarQualityScores,
    CharacterQualityScores,
    FactionQualityScores,
    RefinementConfig,
    RefinementHistory,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import StoryFactoryError, WorldGenerationError


def _all_thresholds(value: float) -> dict[str, float]:
    """Return a complete per-entity quality_thresholds dict with all types set to *value*."""
    return {
        "character": value,
        "location": value,
        "faction": value,
        "item": value,
        "concept": value,
        "event": value,
        "relationship": value,
        "plot": value,
        "chapter": value,
    }


@pytest.fixture
def mock_svc():
    """Create a mock WorldQualityService with analytics logging.

    Uses spec=[] so MagicMock does not auto-create arbitrary attributes.
    Tests that need analytics_db must set it explicitly on the mock.
    """
    svc = MagicMock(spec=[])
    svc._log_refinement_analytics = MagicMock()
    return svc


@pytest.fixture
def config():
    """Create a RefinementConfig with test defaults."""
    return RefinementConfig(
        quality_threshold=8.0,
        quality_thresholds=_all_thresholds(8.0),
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
        temporal_plausibility=avg,
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
        # 4th score is for the hail-mary judge call (below peak, so best stays v2)
        scores_list = [_make_scores(6.0), _make_scores(7.5), _make_scores(5.0), _make_scores(4.0)]
        iteration_idx = 0

        def create_fn(retries):
            """Return the first entity."""
            return entities[0]

        def judge_fn(entity):
            """Return scores in sequence: 6.0, 7.5, 5.0, 4.0 (hail-mary)."""
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

        # Best iteration was iteration 2 (score 7.5), total iterations = 4
        # (3 regular iterations + 1 hail-mary attempt that scored below peak)
        assert result_entity == {"name": "v2"}
        assert _result_scores.average == 7.5
        assert iterations == 4

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

        # Judge should only be called once in the loop (iteration 0), then refine
        # returns unchanged output on iteration 1 and loop breaks before re-judging.
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert judge_calls == 2
        assert result_scores.average == 6.0
        # Analytics should record early_stop_triggered=True
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True

    def test_unchanged_detection_compares_against_previous_refine(self, mock_svc, config):
        """Unchanged detection triggers when refine echoes its OWN previous output."""
        config.max_iterations = 5
        config.early_stopping_patience = 10  # Disable normal early stopping

        # Refine produces a new entity once, then echoes it
        refine_results = [
            {"name": "Hero", "trait": "brave v2"},  # Different from creation
            {"name": "Hero", "trait": "brave v2"},  # Same as previous refine
        ]
        refine_idx = 0

        def refine_fn(e, s, i):
            """Return new entity once, then echo it."""
            nonlocal refine_idx
            result = refine_results[min(refine_idx, len(refine_results) - 1)]
            refine_idx += 1
            return result

        judge_calls = 0

        def judge_fn(e):
            """Count judge invocations."""
            nonlocal judge_calls
            judge_calls += 1
            return _make_scores(6.0)  # Always below threshold

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero", "trait": "brave v1"},
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

        # Iteration 0: create v1 → judge (6.0, below threshold)
        # Iteration 1: refine → v2 (different) → judge (6.0)
        # Iteration 2: refine → v2 (same as iteration 1) → unchanged detection → break
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert judge_calls == 3

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
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert judge_idx == 4

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


class TestQualityLoopB10SkipRedundantJudge:
    """Tests for B10: skip redundant judge calls after refinement errors (#266)."""

    def test_refinement_error_skips_judge_on_unchanged_entity(self, mock_svc, config):
        """After WorldGenerationError during refinement, judge is NOT called on unchanged entity."""
        config.max_iterations = 4
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 10  # Disable score-plateau early-stop

        judge_calls = 0
        refine_calls = 0

        def judge_fn(e):
            """Track judge calls and return low scores."""
            nonlocal judge_calls
            judge_calls += 1
            return _make_scores(5.0)

        def refine_fn(e, s, i):
            """Fail on first refinement, succeed on subsequent with unique entities."""
            nonlocal refine_calls
            refine_calls += 1
            if refine_calls == 1:
                raise WorldGenerationError("Refinement LLM error")
            return {"name": f"Refined Hero v{refine_calls}"}

        _result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
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

        # Iteration 0: create → judge (5.0) → judge_calls=1
        # Iteration 1: refine raises → scores preserved, no judge → refine_calls=1
        # Iteration 2: refine succeeds (v2) → judge (5.0) → judge_calls=2, refine_calls=2
        # Iteration 3: refine succeeds (v3) → judge (5.0) → judge_calls=3, refine_calls=3
        # +1 for hail-mary fresh creation judge call (threshold not met)
        # judge should NOT have been called after iteration 1's refinement error
        assert judge_calls == 4  # 3 in loop + 1 hail-mary
        assert refine_calls == 3
        assert scoring_rounds == 4

    def test_judge_error_allows_re_judge_on_next_iteration(self, mock_svc, config):
        """After judge error, the same entity IS re-judged (not skipped)."""
        config.max_iterations = 3

        judge_calls = 0

        def judge_fn(e):
            """Fail on first call, succeed on second."""
            nonlocal judge_calls
            judge_calls += 1
            if judge_calls == 1:
                raise WorldGenerationError("Judge LLM error")
            return _make_scores(9.0)

        _result_entity, result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
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

        # Iteration 0: create → judge raises → needs_judging stays True
        # Iteration 1: needs_judging=True → re-judge → succeeds (9.0) → threshold met
        assert judge_calls == 2
        assert result_scores.average == 9.0
        assert scoring_rounds == 1

    def test_failed_refinements_counter_incremented(self, mock_svc, config):
        """RefinementHistory.failed_refinements tracks error count."""
        config.max_iterations = 3
        config.early_stopping_patience = 10

        refine_count = 0

        def refine_fn(e, s, i):
            """Always fail refinement."""
            nonlocal refine_count
            refine_count += 1
            raise WorldGenerationError("Refinement failed")

        _result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=lambda e: _make_scores(5.0),
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Iteration 0: create → judge (5.0) → scoring_rounds=1
        # Iteration 1: refine raises → failed_refinements=1, scores preserved
        # Iteration 2: refine raises → failed_refinements=2, scores preserved
        # Post-loop: hail-mary fires (5.0 < 8.0, max_iterations=3 > 1) → scoring_rounds=2
        assert scoring_rounds == 2
        # Verify analytics was called with the history tracking failed refinements
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call[0][0]
        assert history.failed_refinements == 2


class TestQualityLoopStageTracking:
    """Tests for stage-based error classification (create vs judge vs refine)."""

    def test_create_fn_error_not_counted_as_failed_refinement(self, mock_svc, config):
        """WorldGenerationError during create_fn is NOT counted as failed_refinements."""
        config.max_iterations = 4

        create_count = 0

        def create_fn(retries):
            """Fail on first call, succeed on second."""
            nonlocal create_count
            create_count += 1
            if create_count == 1:
                raise WorldGenerationError("Create LLM error")
            return {"name": "Hero"}

        _result_entity, _result_scores, _scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=lambda e: _make_scores(9.0),
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Create error should NOT increment failed_refinements
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call[0][0]
        assert history.failed_refinements == 0

    def test_judge_error_not_counted_as_failed_refinement(self, mock_svc, config):
        """WorldGenerationError during judge_fn is NOT counted as failed_refinements."""
        config.max_iterations = 4

        judge_count = 0

        def judge_fn(e):
            """Fail on first call, succeed on second."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                raise WorldGenerationError("Judge LLM error")
            return _make_scores(9.0)

        _result_entity, _result_scores, _scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
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

        # Judge error should NOT increment failed_refinements
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call[0][0]
        assert history.failed_refinements == 0

    def test_refine_error_counted_as_failed_refinement(self, mock_svc, config):
        """WorldGenerationError during refine_fn IS counted as failed_refinements."""
        config.max_iterations = 4
        config.early_stopping_patience = 10

        refine_count = 0

        def refine_fn(e, s, i):
            """Fail on first refinement, succeed on second."""
            nonlocal refine_count
            refine_count += 1
            if refine_count == 1:
                raise WorldGenerationError("Refinement LLM error")
            return {"name": "Refined Hero"}

        _result_entity, _result_scores, _scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=lambda e: _make_scores(5.0),
            refine_fn=refine_fn,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Exactly 1 refinement error should be counted
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call[0][0]
        assert history.failed_refinements == 1

    def test_mixed_errors_only_counts_refine_failures(self, mock_svc, config):
        """Only refinement errors are counted in failed_refinements, not create or judge errors."""
        config.max_iterations = 6
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 10  # Disable score-plateau early-stop

        create_count = 0
        judge_count = 0
        refine_count = 0

        def create_fn(retries):
            """Fail on first call, succeed after."""
            nonlocal create_count
            create_count += 1
            if create_count == 1:
                raise WorldGenerationError("Create error")
            return {"name": "Hero"}

        def judge_fn(e):
            """Fail on second call (after first successful judge + refine attempt)."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 2:
                raise WorldGenerationError("Judge error")
            return _make_scores(5.0)

        def refine_fn(e, s, i):
            """Fail on second refinement attempt."""
            nonlocal refine_count
            refine_count += 1
            if refine_count == 2:
                raise WorldGenerationError("Refine error")
            return {"name": f"Refined v{refine_count}"}

        _result_entity, _result_scores, _scoring_rounds = quality_refinement_loop(
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

        # 1 create error + 1 judge error + 1 refine error, but only refine counts
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call[0][0]
        assert history.failed_refinements == 1


class TestQualityLoopC3ScoringRounds:
    """Tests for C3: scoring_rounds separates actual judge calls from loop iterations (#266)."""

    def test_scoring_rounds_excludes_failed_iterations(self, mock_svc, config):
        """scoring_rounds only counts successful judge calls, not total loop iterations."""
        config.max_iterations = 5
        config.early_stopping_patience = 10

        judge_calls = 0

        def judge_fn(e):
            """Return low scores."""
            nonlocal judge_calls
            judge_calls += 1
            return _make_scores(5.0)

        refine_count = 0

        def refine_fn(e, s, i):
            """Fail first, succeed after."""
            nonlocal refine_count
            refine_count += 1
            if refine_count == 1:
                raise WorldGenerationError("Temp error")
            return {"name": "Refined"}

        _entity, _scores, scoring_rounds = quality_refinement_loop(
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

        # scoring_rounds should equal actual judge_calls
        assert scoring_rounds == judge_calls
        # Loop ran 5 iterations but scoring_rounds < 5 because of failed refinement
        assert scoring_rounds < 5

    def test_scoring_rounds_returned_on_threshold_met(self, mock_svc, config):
        """When threshold is met, scoring_rounds (not iteration count) is returned."""
        config.max_iterations = 5

        # First creation fails (empty), second succeeds, judge passes threshold
        create_count = 0

        def create_fn(retries):
            """Return empty first, valid second."""
            nonlocal create_count
            create_count += 1
            if create_count == 1:
                return {"name": ""}  # Empty
            return {"name": "Valid"}

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=lambda e: _make_scores(9.0),
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Loop iteration 0: empty creation (no judge)
        # Loop iteration 1: valid creation → judge (9.0) → threshold met
        # scoring_rounds = 1 (only 1 actual judge call)
        assert scoring_rounds == 1

    def test_early_stopping_uses_scoring_rounds_in_log(self, mock_svc, config, caplog):
        """Early stopping log message includes scoring_rounds count."""
        config.max_iterations = 5
        config.early_stopping_patience = 2
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return next entity."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: entities[0],
                judge_fn=lambda e: _make_scores(6.0),
                refine_fn=refine_fn,
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("scoring round" in msg for msg in caplog.messages)


class TestQualityLoopScoreRounding:
    """Tests for score rounding at threshold boundary (#303)."""

    def test_score_rounds_up_to_meet_threshold(self, mock_svc, config):
        """A score of 7.46 rounds to 7.5 and should pass >= 7.5 threshold."""
        config.quality_threshold = 7.5
        config.quality_thresholds = _all_thresholds(7.5)

        # CharacterQualityScores: average = (d + g + f + u + a + tp) / 6
        # To get average ≈ 7.467: sum = 44.8, avg = 44.8/6 ≈ 7.467
        # Use: 7.0, 7.0, 7.0, 8.0, 8.3, 7.5 → sum = 44.8, avg ≈ 7.467
        scores = CharacterQualityScores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=8.0,
            arc_potential=8.3,
            temporal_plausibility=7.5,
        )
        # Verify the average is indeed ~7.46, which would display as "7.5"
        assert round(scores.average, 1) == 7.5
        assert scores.average < 7.5  # Raw value is below threshold

        entity = {"name": "Hero"}
        result_entity, _result_scores, iterations = quality_refinement_loop(
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
        assert iterations == 1
        # Should have met threshold (threshold_met=True in analytics)
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["threshold_met"] is True

    def test_score_rounds_down_misses_threshold(self, mock_svc, config):
        """A score of 7.44 rounds to 7.4 and should fail >= 7.5 threshold."""
        config.quality_threshold = 7.5
        config.quality_thresholds = _all_thresholds(7.5)
        config.max_iterations = 1

        # 6 * 7.433 ≈ 44.6 → 7.0, 7.0, 7.0, 8.0, 8.2, 7.4 → sum = 44.6, avg ≈ 7.433
        scores = CharacterQualityScores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=8.0,
            arc_potential=8.2,
            temporal_plausibility=7.4,
        )
        assert round(scores.average, 1) == 7.4
        assert scores.average < 7.5

        entity = {"name": "Hero"}
        _result_entity, _result_scores, _iterations = quality_refinement_loop(
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

        # Should NOT have met threshold
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["threshold_met"] is False

    def test_post_loop_threshold_met_uses_rounded_peak(self, mock_svc, config):
        """Post-loop threshold_met rounds peak_score — 7.44 rounds to 7.4, fails >= 7.5."""
        config.quality_threshold = 7.5
        config.quality_thresholds = _all_thresholds(7.5)
        config.max_iterations = 2
        config.early_stopping_patience = 10

        # avg ≈ 7.433 rounds to 7.4 → fails in-loop, enters post-loop path
        scores_below = CharacterQualityScores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=8.0,
            arc_potential=8.2,
            temporal_plausibility=7.4,
        )
        # Second iteration even worse → post-loop returns best (iteration 1)
        scores_worse = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
        )
        judge_count = 0

        def judge_fn(e):
            """Return sub-threshold scores, then worse, to reach post-loop path."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return scores_below
            return scores_worse

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: {"name": "Hero v2"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Peak was ~7.433, rounds to 7.4, so threshold_met should be False
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["threshold_met"] is False


class TestQualityLoopSubThresholdWarning:
    """Tests for WARNING when sub-threshold entities are returned via best-iteration path (#303)."""

    def test_warning_logged_when_best_entity_below_threshold(self, mock_svc, config, caplog):
        """WARNING should be logged when returning best entity that didn't meet threshold."""
        config.quality_threshold = 8.0
        config.max_iterations = 2
        config.early_stopping_patience = 10

        # Both iterations below threshold (7.0 then 6.0)
        scores_list = [_make_scores(7.0), _make_scores(6.0)]
        judge_idx = 0

        def judge_fn(e):
            """Return declining scores, both below threshold."""
            nonlocal judge_idx
            result = scores_list[min(judge_idx, len(scores_list) - 1)]
            judge_idx += 1
            return result

        with caplog.at_level(logging.WARNING):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: {"name": "Hero"},
                judge_fn=judge_fn,
                refine_fn=lambda e, s, i: {"name": "Hero v2"},
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("did not meet quality threshold after" in msg for msg in caplog.messages)

    def test_no_warning_when_best_entity_meets_threshold(self, mock_svc, config, caplog):
        """No sub-threshold WARNING when threshold is met in-loop (early return)."""
        config.quality_threshold = 7.0
        config.quality_thresholds = _all_thresholds(7.0)
        config.max_iterations = 2
        config.early_stopping_patience = 10

        # First score (7.5) meets threshold in-loop → early return, no post-loop path
        scores_list = [_make_scores(7.5), _make_scores(5.0)]
        judge_idx = 0

        def judge_fn(e):
            """Return 7.5 — meets threshold in-loop on first iteration."""
            nonlocal judge_idx
            result = scores_list[min(judge_idx, len(scores_list) - 1)]
            judge_idx += 1
            return result

        with caplog.at_level(logging.WARNING):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: {"name": "Hero"},
                judge_fn=judge_fn,
                refine_fn=lambda e, s, i: {"name": "Hero v2"},
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert not any("did not meet quality threshold after" in msg for msg in caplog.messages)


class TestQualityLoopTimingInstrumentation:
    """Tests for per-iteration timing instrumentation (#304)."""

    def test_creation_timing_logged(self, mock_svc, config, caplog):
        """Creation timing should be logged at INFO level."""
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
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

        assert any("creation took" in msg.lower() for msg in caplog.messages)

    def test_judge_timing_logged(self, mock_svc, config, caplog):
        """Judge call timing should be logged at INFO level."""
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
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

        assert any("judge call took" in msg.lower() for msg in caplog.messages)

    def test_refinement_timing_logged(self, mock_svc, config, caplog):
        """Refinement timing should be logged at INFO level when refine is called."""
        config.max_iterations = 2
        config.early_stopping_patience = 10

        judge_count = 0

        def judge_fn(e):
            """Return below threshold first, then above."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return _make_scores(5.0, feedback="Needs work")
            return _make_scores(9.0, feedback="Great")

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: {"name": "Hero"},
                judge_fn=judge_fn,
                refine_fn=lambda e, s, i: {"name": "Hero v2"},
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("refinement took" in msg.lower() for msg in caplog.messages)

    def test_judge_timing_includes_scoring_round(self, mock_svc, config, caplog):
        """Judge timing log should include the scoring round number."""
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
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

        assert any("scoring round 1" in msg.lower() for msg in caplog.messages)


class TestPerEntityThresholds:
    """Test per-entity quality threshold support in the quality loop."""

    def test_uses_per_entity_threshold_for_character(self, mock_svc):
        """Loop should use per-entity threshold (7.0) instead of fallback (8.0)."""
        thresholds = _all_thresholds(8.0)
        thresholds["character"] = 7.0
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=thresholds,
            max_iterations=5,
            early_stopping_patience=2,
            early_stopping_min_iterations=2,
        )
        entity = {"name": "Hero"}
        # Score 7.2 — above per-entity 7.0 but below legacy 8.0
        scores = _make_scores(7.2)

        result_entity, _result_scores, iterations = quality_refinement_loop(
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

        # Should pass on first iteration (7.2 >= 7.0)
        assert result_entity == entity
        assert iterations == 1

    def test_uses_per_entity_threshold_for_item(self, mock_svc):
        """Loop should use higher per-entity threshold for items."""
        thresholds = _all_thresholds(7.5)
        thresholds["item"] = 8.5
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=thresholds,
            max_iterations=3,
            early_stopping_patience=2,
            early_stopping_min_iterations=2,
        )
        entity = {"name": "Sword"}
        # Score 8.0 — above fallback 7.5 but below item-specific 8.5
        scores_low = _make_scores(8.0)
        scores_high = _make_scores(8.6)

        call_count = 0

        def judge_fn(e):
            """Return low scores first, then high scores."""
            nonlocal call_count
            call_count += 1
            return scores_low if call_count == 1 else scores_high

        _result_entity, _result_scores, iterations = quality_refinement_loop(
            entity_type="item",
            create_fn=lambda retries: entity,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: {"name": "Sword v2"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Should need refinement (8.0 < 8.5), then pass (8.6 >= 8.5)
        assert iterations == 2

    def test_missing_entity_type_raises(self, mock_svc):
        """Loop should propagate ValueError when entity_type is missing from thresholds."""
        config = RefinementConfig(
            quality_threshold=6.0,
            quality_thresholds={"character": 7.0},  # No "faction" entry
            max_iterations=5,
            early_stopping_patience=2,
            early_stopping_min_iterations=2,
        )
        entity = {"name": "Guild"}
        scores = _make_scores(6.5)

        with pytest.raises(ValueError, match="No quality threshold configured"):
            quality_refinement_loop(
                entity_type="faction",
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

    def test_none_quality_thresholds_raises(self, mock_svc):
        """When quality_thresholds is None, get_threshold should raise ValueError."""
        config = RefinementConfig(
            quality_threshold=7.0,
            quality_thresholds=None,
            max_iterations=5,
            early_stopping_patience=2,
            early_stopping_min_iterations=2,
        )
        entity = {"name": "Hero"}
        scores = _make_scores(7.5)

        with pytest.raises(ValueError, match="quality_thresholds is empty"):
            quality_refinement_loop(
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


class TestRefinementConfigGetThreshold:
    """Test RefinementConfig.get_threshold() method."""

    def test_returns_per_entity_threshold(self):
        """get_threshold should return per-entity value when available."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5) | {"character": 8.0, "item": 9.0},
        )
        assert config.get_threshold("character") == 8.0
        assert config.get_threshold("item") == 9.0

    def test_missing_type_raises(self):
        """get_threshold should raise ValueError for unlisted types."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds={"character": 8.0},
        )
        with pytest.raises(ValueError, match="No quality threshold configured"):
            config.get_threshold("faction")

    def test_none_thresholds_raises(self):
        """get_threshold with None thresholds should raise ValueError."""
        config = RefinementConfig(
            quality_threshold=6.0,
            quality_thresholds=None,
        )
        with pytest.raises(ValueError, match="quality_thresholds is empty"):
            config.get_threshold("character")

    def test_empty_thresholds_raises(self):
        """get_threshold with empty dict should raise ValueError."""
        config = RefinementConfig(
            quality_threshold=6.5,
            quality_thresholds={},
        )
        with pytest.raises(ValueError, match="quality_thresholds is empty"):
            config.get_threshold("character")

    def test_from_settings_populates_thresholds(self):
        """from_settings should populate quality_thresholds from settings."""
        mock_settings = MagicMock()
        mock_settings.world_quality_max_iterations = 3
        mock_settings.world_quality_threshold = 7.5
        mock_settings.world_quality_thresholds = _all_thresholds(7.5) | {"item": 8.0}
        mock_settings.world_quality_creator_temp = 0.9
        mock_settings.world_quality_judge_temp = 0.1
        mock_settings.world_quality_refinement_temp = 0.7
        mock_settings.world_quality_early_stopping_patience = 2
        mock_settings.world_quality_refinement_temp_start = 0.7
        mock_settings.world_quality_refinement_temp_end = 0.35
        mock_settings.world_quality_refinement_temp_decay = "linear"
        mock_settings.world_quality_early_stopping_min_iterations = 2
        mock_settings.world_quality_early_stopping_variance_tolerance = 0.3
        mock_settings.world_quality_score_plateau_tolerance = 0.2
        mock_settings.world_quality_dimension_minimum = 6.0

        config = RefinementConfig.from_settings(mock_settings)

        assert config.get_threshold("character") == 7.5
        assert config.get_threshold("item") == 8.0
        assert config.score_plateau_tolerance == 0.2
        assert config.dimension_minimum == 6.0


class TestScorePlateauEarlyStop:
    """Tests for score-plateau early-stop (#328)."""

    def test_identical_consecutive_scores_trigger_plateau(self, mock_svc, config):
        """Two consecutive identical scores trigger score-plateau early-stop."""
        config.max_iterations = 5
        config.early_stopping_patience = 10  # Disable normal early stopping
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        # Scores: 7.0, 7.0 → plateau on second scoring round
        judge_idx = 0

        def judge_fn(e):
            """Return 7.0 for all iterations."""
            nonlocal judge_idx
            judge_idx += 1
            return _make_scores(7.0)

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # Should stop after 2 scoring rounds in loop (7.0 → 7.0)
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert scoring_rounds == 3
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True

    def test_near_equal_scores_trigger_plateau(self, mock_svc, config):
        """Scores within 0.1 tolerance trigger score-plateau early-stop."""
        config.max_iterations = 5
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        # Scores: 7.0, 7.05 → within 0.1 tolerance
        score_values = [7.0, 7.05]
        judge_idx = 0

        def judge_fn(e):
            """Return scores that are near-equal."""
            nonlocal judge_idx
            result = _make_scores(score_values[min(judge_idx, len(score_values) - 1)])
            judge_idx += 1
            return result

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert scoring_rounds == 3
        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["early_stop_triggered"] is True

    def test_plateau_respects_min_iterations(self, mock_svc, config):
        """Score-plateau does not trigger before early_stopping_min_iterations."""
        config.max_iterations = 5
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 3  # Require at least 3 iterations

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        # Scores: 7.0, 7.0, 7.0 → plateau triggers on 3rd (>= min_iterations=3)
        def judge_fn(e):
            """Return constant 7.0."""
            return _make_scores(7.0)

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # With min_iterations=3, needs at least 3 scoring rounds
        # Scoring round 2 has only 2 iterations, doesn't meet min 3
        # Scoring round 3: len(history) == 3 >= max(2, 3), triggers
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert scoring_rounds == 4

    def test_improving_scores_do_not_trigger_plateau(self, mock_svc, config):
        """Steadily improving scores never trigger score-plateau early-stop."""
        config.max_iterations = 3
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        # Scores: 5.0, 6.0, 7.0 → always improving, never plateau
        score_values = [5.0, 6.0, 7.0]
        judge_idx = 0

        def judge_fn(e):
            """Return improving scores."""
            nonlocal judge_idx
            result = _make_scores(score_values[min(judge_idx, len(score_values) - 1)])
            judge_idx += 1
            return result

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # All 3 iterations should run (no plateau)
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert scoring_rounds == 4

    def test_plateau_logs_info_message(self, mock_svc, config, caplog):
        """Score-plateau early-stop logs an info message."""
        config.max_iterations = 5
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: entities[0],
                judge_fn=lambda e: _make_scores(7.0),
                refine_fn=refine_fn,
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any(
            "score plateaued at" in msg or "identical scores" in msg for msg in caplog.messages
        )

    def test_score_difference_above_tolerance_continues(self, mock_svc, config):
        """Scores differing by more than 0.1 do not trigger plateau."""
        config.max_iterations = 3
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 2

        entities = [{"name": f"v{i}"} for i in range(5)]
        iter_idx = 0

        def refine_fn(e, s, i):
            """Return a different entity each time."""
            nonlocal iter_idx
            iter_idx += 1
            return entities[iter_idx]

        # Scores: 7.0, 6.8 → difference 0.2 > tolerance 0.1
        score_values = [7.0, 6.8, 6.6]
        judge_idx = 0

        def judge_fn(e):
            """Return scores with 0.2 difference."""
            nonlocal judge_idx
            result = _make_scores(score_values[min(judge_idx, len(score_values) - 1)])
            judge_idx += 1
            return result

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # All 3 iterations should run (0.2 difference > 0.1 tolerance)
        # +1 for hail-mary fresh creation judge call (threshold not met)
        assert scoring_rounds == 4


class TestFallbackWhenBestEntityNotFound:
    """Test defensive fallback when get_best_entity returns None."""

    @patch.object(RefinementHistory, "get_best_entity", return_value=None)
    def test_returns_last_entity_when_best_not_found(self, _mock_get_best, mock_svc):
        """Fallback returns current entity/scores when best_entity_data is None."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=1,
            early_stopping_patience=2,
            early_stopping_min_iterations=2,
        )
        entity = {"name": "Hero"}
        scores = _make_scores(5.0, feedback="Below threshold")

        result, result_scores, scoring_rounds = quality_refinement_loop(
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

        # Fallback returns the current entity since best was not found
        assert result["name"] == "Hero"
        assert result_scores.average == 5.0
        assert scoring_rounds == 1


class TestHailMaryStructuralGating:
    """Tests for hail-mary gating when temporal_plausibility is a structural blocker (#385)."""

    def test_hail_mary_skipped_when_temporal_plausibility_below_4(self, mock_svc):
        """Hail-mary is skipped when temporal_plausibility is the lowest dim below 4.0."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        # Scores where temporal_plausibility is the structural bottleneck
        structural_scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=2.0,  # Structural deficit — below 4.0
            feedback="Temporal placement is completely wrong",
        )

        entities = [{"name": "v1"}, {"name": "v2"}]
        create_calls = 0

        def create_fn(retries):
            """Track creation calls."""
            nonlocal create_calls
            create_calls += 1
            return entities[0]

        judge_count = 0

        def judge_fn(entity):
            """Return structural deficit scores consistently."""
            nonlocal judge_count
            judge_count += 1
            return structural_scores

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="faction",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: entities[1],
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=FactionQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # With structural gating, hail-mary should be skipped:
        # - 2 in-loop iterations (create+judge, refine+judge)
        # - 0 hail-mary judge calls (skipped due to temporal_plausibility < 4.0)
        assert scoring_rounds == 2
        # create_fn called once for initial creation, NOT again for hail-mary
        assert create_calls == 1

    def test_hail_mary_proceeds_when_lowest_dim_is_not_temporal(self, mock_svc):
        """Hail-mary proceeds normally when lowest dim is not temporal_plausibility."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        # Scores where a non-temporal dimension is the bottleneck
        non_structural_scores = FactionQualityScores(
            coherence=3.0,  # Lowest, but not temporal_plausibility
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=5.0,
            feedback="Needs more coherence",
        )

        entities = [{"name": "v1"}, {"name": "v2"}]
        create_calls = 0

        def create_fn(retries):
            """Track creation calls."""
            nonlocal create_calls
            create_calls += 1
            return entities[0]

        judge_count = 0

        def judge_fn(entity):
            """Return non-structural deficit scores."""
            nonlocal judge_count
            judge_count += 1
            return non_structural_scores

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="faction",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: entities[1],
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=FactionQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Hail-mary should proceed: 2 in-loop + 1 hail-mary = 3 scoring rounds
        assert scoring_rounds == 3
        # create_fn called once for initial + once for hail-mary
        assert create_calls == 2

    def test_hail_mary_proceeds_when_temporal_above_4(self, mock_svc):
        """Hail-mary proceeds when temporal_plausibility is lowest but >= 4.0."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        # temporal_plausibility is lowest but at 4.0 (not below threshold)
        borderline_scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=4.0,  # At threshold, not below
            feedback="Temporal placement could be better",
        )

        create_calls = 0

        def create_fn(retries):
            """Track creation calls."""
            nonlocal create_calls
            create_calls += 1
            return {"name": "v1"}

        _entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="faction",
            create_fn=create_fn,
            judge_fn=lambda e: borderline_scores,
            refine_fn=lambda e, s, i: {"name": "v2"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=FactionQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Hail-mary should proceed: temporal_plausibility is 4.0 (not < 4.0)
        assert scoring_rounds == 3
        assert create_calls == 2

    def test_hail_mary_gating_logs_info(self, mock_svc, caplog):
        """Structural gating logs an info message when skipping hail-mary."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        structural_scores = FactionQualityScores(
            coherence=8.0,
            influence=8.0,
            conflict_potential=8.0,
            distinctiveness=8.0,
            temporal_plausibility=1.5,
            feedback="No temporal context",
        )

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="faction",
                create_fn=lambda retries: {"name": "v1"},
                judge_fn=lambda e: structural_scores,
                refine_fn=lambda e, s, i: {"name": "v2"},
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=FactionQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("skipping hail-mary" in msg for msg in caplog.messages)
        assert any("structural deficit" in msg for msg in caplog.messages)


class TestZeroScoreAnomalyDetection:
    """Tests for zero-score anomaly detection in quality loop (#395 Fix 8)."""

    def test_zero_dimension_triggers_rejudge(self, mock_svc):
        """Zero on any dimension discards scores and re-judges."""
        config = RefinementConfig(
            quality_threshold=7.0,
            quality_thresholds=_all_thresholds(7.0),
            max_iterations=3,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        entity = {"name": "Hero"}
        judge_call = 0

        def judge_fn(e):
            """Return zero-score on first call, valid scores on second."""
            nonlocal judge_call
            judge_call += 1
            if judge_call == 1:
                # First call: one dimension is 0.0 (parse failure)
                return CharacterQualityScores(
                    depth=0.0,  # Zero = likely parse failure
                    goals=8.0,
                    flaws=8.0,
                    uniqueness=8.0,
                    arc_potential=8.0,
                    temporal_plausibility=8.0,
                    feedback="Partial parse",
                )
            # Second call: all dimensions valid and above threshold
            return CharacterQualityScores(
                depth=8.0,
                goals=8.0,
                flaws=8.0,
                uniqueness=8.0,
                arc_potential=8.0,
                temporal_plausibility=8.0,
                feedback="Good character",
            )

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

        # The zero-score call should have been discarded, second call accepted
        assert result_scores.depth == 8.0
        assert judge_call == 2  # Called twice: once rejected, once accepted

    def test_zero_on_multiple_dimensions_triggers_rejudge(self, mock_svc):
        """Multiple zero dimensions also trigger re-judge."""
        config = RefinementConfig(
            quality_threshold=7.0,
            quality_thresholds=_all_thresholds(7.0),
            max_iterations=3,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        entity = {"name": "Hero"}
        judge_call = 0

        def judge_fn(e):
            """Return multi-zero scores on first call, valid scores on second."""
            nonlocal judge_call
            judge_call += 1
            if judge_call == 1:
                # Multiple zeroes = definitely a parse failure
                return CharacterQualityScores(
                    depth=0.0,
                    goals=0.0,
                    flaws=0.0,
                    uniqueness=8.0,
                    arc_potential=8.0,
                    temporal_plausibility=8.0,
                    feedback="Mostly zeroes",
                )
            return _make_scores(8.0, "Good")

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

        assert result_scores.average == 8.0
        assert judge_call == 2

    def test_half_point_score_accepted_normally(self, mock_svc):
        """Score of 0.5 (not 0.0) should be accepted without re-judging."""
        # Threshold is set below the average (6.75) so the loop exits after one
        # judge call — confirming 0.5 is NOT treated as a parse-failure anomaly.
        # dimension_minimum=0.0 disables the floor (0.5 is a valid score, not a floor concern).
        config = RefinementConfig(
            quality_threshold=5.0,
            quality_thresholds=_all_thresholds(5.0),
            max_iterations=3,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
            dimension_minimum=0.0,
        )

        entity = {"name": "Hero"}
        judge_call = 0

        def judge_fn(e):
            """Return low-but-nonzero scores that should be accepted."""
            nonlocal judge_call
            judge_call += 1
            return CharacterQualityScores(
                depth=0.5,  # Low but not zero — legitimate score
                goals=8.0,
                flaws=8.0,
                uniqueness=8.0,
                arc_potential=8.0,
                temporal_plausibility=8.0,
                feedback="Shallow but valid",
            )

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

        # 0.5 should NOT be rejected — only exactly 0.0 is treated as parse failure
        assert result_scores.depth == 0.5
        assert judge_call == 1  # Only called once, accepted

    def test_persistent_zero_scores_exhaust_max_iterations(self, mock_svc):
        """When judge always returns zero scores, loop exhausts iterations and raises."""
        config = RefinementConfig(
            quality_threshold=7.0,
            quality_thresholds=_all_thresholds(7.0),
            max_iterations=3,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        entity = {"name": "Hero"}

        def always_zero_judge(e):
            """Always return zero scores (simulating persistent parse failures)."""
            return CharacterQualityScores(
                depth=0.0,
                goals=0.0,
                flaws=0.0,
                uniqueness=0.0,
                arc_potential=0.0,
                temporal_plausibility=0.0,
                feedback="All zeroes",
            )

        with pytest.raises(WorldGenerationError, match="Failed to generate character"):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: entity,
                judge_fn=always_zero_judge,
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

    def test_hail_mary_zero_scores_discarded(self, mock_svc, caplog):
        """Hail-mary judge returning zero scores discards the fresh entity."""
        config = RefinementConfig(
            quality_threshold=9.0,
            quality_thresholds=_all_thresholds(9.0),
            max_iterations=2,
            early_stopping_patience=10,
            early_stopping_min_iterations=10,
        )

        entities = [{"name": "Original"}, {"name": "HailMary"}]
        judge_call = 0

        def judge_fn(entity):
            """Return sub-threshold scores in loop, zero scores for hail-mary."""
            nonlocal judge_call
            judge_call += 1
            if entity["name"] == "HailMary":
                # Hail-mary judge returns zero — parse failure
                return CharacterQualityScores(
                    depth=0.0,
                    goals=7.0,
                    flaws=7.0,
                    uniqueness=7.0,
                    arc_potential=7.0,
                    temporal_plausibility=7.0,
                    feedback="Partial parse on hail-mary",
                )
            # Normal sub-threshold scores for the main loop
            return CharacterQualityScores(
                depth=6.0,
                goals=6.0,
                flaws=6.0,
                uniqueness=6.0,
                arc_potential=6.0,
                temporal_plausibility=6.0,
                feedback="Below threshold",
            )

        create_call = 0

        def create_fn(retries):
            """Return Original first, then HailMary for hail-mary attempt."""
            nonlocal create_call
            create_call += 1
            if create_call <= 1:
                return entities[0]
            return entities[1]

        with caplog.at_level(logging.WARNING):
            result_entity, result_scores, scoring_rounds = quality_refinement_loop(
                entity_type="character",
                create_fn=create_fn,
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

        # Hail-mary zero scores should be discarded — original kept
        assert result_entity["name"] == "Original"
        assert result_scores.depth == 6.0
        # Hail-mary scoring round NOT counted (zero scores discarded);
        # main loop exits after 1 round due to unchanged-output detection
        assert scoring_rounds == 1
        # Warning logged about hail-mary zero scores
        assert any("hail-mary judge returned 0.0" in msg for msg in caplog.messages)


# ==========================================================================
# Phase 1: Sub-threshold entity flagging (#398 Bug 1)
# ==========================================================================


class TestBelowThresholdAdmitted:
    """Tests for below_threshold_admitted flag on RefinementHistory."""

    def test_flag_true_when_below_threshold(self, mock_svc, config):
        """Flag is True when entity is returned below quality threshold."""
        config.max_iterations = 2
        config.early_stopping_patience = 5

        entities = [{"name": "v1"}, {"name": "v2"}]
        # Scores below threshold (8.0) — entity will be admitted but flagged
        scores_list = [_make_scores(6.0), _make_scores(6.5), _make_scores(5.0)]
        judge_idx = 0
        refine_idx = 0

        def judge_fn(entity):
            """Return scores from list in sequence."""
            nonlocal judge_idx
            result = scores_list[min(judge_idx, len(scores_list) - 1)]
            judge_idx += 1
            return result

        def refine_fn(entity, scores, iteration):
            """Return next entity from list."""
            nonlocal refine_idx
            refine_idx += 1
            return entities[min(refine_idx, len(entities) - 1)]

        _result_entity, result_scores, _iters = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entities[0],
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

        # Entity was returned below threshold
        assert result_scores.average < 8.0
        # Analytics should have been called with history containing the flag
        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call.args[0]
        assert history.below_threshold_admitted is True

    def test_flag_false_when_threshold_met(self, mock_svc, config):
        """Flag is False when entity meets quality threshold."""
        entity = {"name": "Hero"}
        scores = _make_scores(9.0)

        quality_refinement_loop(
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

        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call.args[0]
        assert history.below_threshold_admitted is False


# ==========================================================================
# Phase 2: Monotonicity guard (#398 Bug 3)
# ==========================================================================


class TestMonotonicityGuard:
    """Tests for entity reversion on score regression."""

    def test_refine_receives_best_entity_after_regression(self, mock_svc, config):
        """After score regresses, refine_fn receives the best entity, not the degraded one."""
        config.max_iterations = 4
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 10

        entities_passed_to_refine: list[dict] = []

        v1 = {"name": "v1"}
        v2 = {"name": "v2"}
        v3 = {"name": "v3"}
        v4 = {"name": "v4"}

        # Score progression: 6.0, 7.5, 6.0, 7.0
        # After iteration 3 regresses (6.0 < 7.5), entity should revert to v2 (best)
        scores_seq = [
            _make_scores(6.0),
            _make_scores(7.5),
            _make_scores(6.0),
            _make_scores(7.0),
            _make_scores(5.0),  # hail-mary
        ]
        judge_idx = 0
        refine_counter = 0

        def judge_fn(entity):
            """Return scores from sequence."""
            nonlocal judge_idx
            result = scores_seq[min(judge_idx, len(scores_seq) - 1)]
            judge_idx += 1
            return result

        def refine_fn(entity, scores, iteration):
            """Track entities passed to refine and return next version."""
            nonlocal refine_counter
            entities_passed_to_refine.append(entity.copy())
            refine_counter += 1
            return [v2, v3, v4][min(refine_counter - 1, 2)]

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: v1,
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

        # refine_fn call 1: receives v1 (first entity, scored 6.0)
        assert entities_passed_to_refine[0] == v1
        # refine_fn call 2: receives v2 (scored 7.5 — best so far)
        assert entities_passed_to_refine[1] == v2
        # refine_fn call 3: after v3 scored 6.0 (regression from 7.5),
        # entity should have been reverted to v2 (best), so refine receives v2
        assert entities_passed_to_refine[2] == v2

    def test_history_records_all_iterations_including_degraded(self, mock_svc, config):
        """History still records all iterations, including degraded ones."""
        config.max_iterations = 3
        config.early_stopping_patience = 10
        config.early_stopping_min_iterations = 10

        scores_seq = [
            _make_scores(7.0),
            _make_scores(6.0),
            _make_scores(6.5),
            _make_scores(5.0),  # hail-mary
        ]
        judge_idx = 0

        def judge_fn(entity):
            """Return scores from sequence."""
            nonlocal judge_idx
            result = scores_seq[min(judge_idx, len(scores_seq) - 1)]
            judge_idx += 1
            return result

        refine_idx = 0

        def refine_fn(entity, scores, iteration):
            """Return next versioned entity."""
            nonlocal refine_idx
            refine_idx += 1
            return {"name": f"v{refine_idx + 1}"}

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "v1"},
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

        analytics_call = mock_svc._log_refinement_analytics.call_args
        history = analytics_call.args[0]
        # All 3 main-loop iterations recorded (plus hail-mary if it records)
        assert len(history.iterations) >= 3
        # Degraded score at iteration 2 is recorded
        assert history.iterations[1].average_score == 6.0


# ==========================================================================
# Phase 3: Hail-mary win rate tracking (#398 Bug 2)
# ==========================================================================


class TestHailMaryWinRateTracking:
    """Tests for hail-mary win rate gating and tracking."""

    def test_hail_mary_skipped_when_win_rate_low(self, mock_svc, config):
        """Hail-mary is skipped when historical win rate < 20%."""
        config.max_iterations = 2
        config.early_stopping_patience = 10

        # Add analytics_db with low win rate
        mock_svc.analytics_db = MagicMock()
        mock_svc.analytics_db.get_hail_mary_win_rate.return_value = 0.05  # 5%

        judge_calls = 0

        def judge_fn(entity):
            """Return constant low scores."""
            nonlocal judge_calls
            judge_calls += 1
            return _make_scores(6.0)

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: {"name": f"Hero v{i}"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Without hail-mary, only main loop judge calls happen
        # (2 iterations plateau → early stop, no hail-mary)
        mock_svc.analytics_db.get_hail_mary_win_rate.assert_called_once_with(
            entity_type="character", min_attempts=config.hail_mary_min_attempts
        )
        # Hail-mary was NOT attempted → record_hail_mary_attempt not called
        mock_svc.analytics_db.record_hail_mary_attempt.assert_not_called()

    def test_hail_mary_proceeds_with_insufficient_data(self, mock_svc, config):
        """Hail-mary proceeds when win rate is None (insufficient data)."""
        config.max_iterations = 2
        config.early_stopping_patience = 10

        mock_svc.analytics_db = MagicMock()
        mock_svc.analytics_db.get_hail_mary_win_rate.return_value = None  # insufficient

        judge_calls = 0

        def judge_fn(entity):
            """Return constant low scores."""
            nonlocal judge_calls
            judge_calls += 1
            return _make_scores(6.0)

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": "Hero"},
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: {"name": f"Hero v{i}"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Hail-mary attempted → record_hail_mary_attempt should be called
        mock_svc.analytics_db.record_hail_mary_attempt.assert_called_once()
        call_kwargs = mock_svc.analytics_db.record_hail_mary_attempt.call_args.kwargs
        assert call_kwargs["entity_type"] == "character"
        assert call_kwargs["won"] is False  # 6.0 doesn't beat 6.0

    def test_hail_mary_records_win(self, mock_svc, config):
        """Hail-mary records won=True when fresh entity beats best."""
        config.max_iterations = 2
        config.early_stopping_patience = 10

        mock_svc.analytics_db = MagicMock()
        mock_svc.analytics_db.get_hail_mary_win_rate.return_value = None

        judge_calls = 0

        def judge_fn(entity):
            """Return low scores first, then high score for hail-mary."""
            nonlocal judge_calls
            judge_calls += 1
            if judge_calls <= 2:
                return _make_scores(6.0)
            # Hail-mary judge returns higher score
            return _make_scores(7.5)

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: {"name": f"Hero v{retries}"},
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: {"name": f"Refined v{i}"},
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        mock_svc.analytics_db.record_hail_mary_attempt.assert_called_once()
        call_kwargs = mock_svc.analytics_db.record_hail_mary_attempt.call_args.kwargs
        assert call_kwargs["won"] is True
        assert call_kwargs["hail_mary_score"] == 7.5


# ==========================================================================
# Phase 4: Auto-pass for high first-pass rate (#398)
# ==========================================================================


class TestAutoPassScore:
    """Tests for auto_pass_score parameter on quality_refinement_loop."""

    def test_auto_pass_skips_judge(self, mock_svc, config):
        """When auto_pass_score is provided, judge and refine are never called."""
        judge_fn = MagicMock()
        refine_fn = MagicMock()
        auto_scores = _make_scores(8.0)

        result_entity, result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="relationship",
            create_fn=lambda retries: {"source": "A", "target": "B"},
            judge_fn=judge_fn,
            refine_fn=refine_fn,
            get_name=lambda e: f"{e.get('source', '?')} -> {e.get('target', '?')}",
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("source"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            auto_pass_score=auto_scores,
        )

        judge_fn.assert_not_called()
        refine_fn.assert_not_called()
        assert scoring_rounds == 0
        assert result_scores is auto_scores
        assert result_entity == {"source": "A", "target": "B"}
        # Analytics still recorded
        mock_svc._log_refinement_analytics.assert_called_once()

    def test_auto_pass_falls_through_on_empty_entity(self, mock_svc, config):
        """If auto_pass creation returns empty, falls through to full loop."""
        auto_scores = _make_scores(8.0)
        real_scores = _make_scores(9.0)

        create_calls = 0

        def create_fn(retries):
            """Return empty on first call, valid on second."""
            nonlocal create_calls
            create_calls += 1
            if create_calls == 1:
                return {}  # Empty — trigger fallthrough
            return {"name": "Hero"}

        result_entity, result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=lambda e: real_scores,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e.get("name", "?"),
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            auto_pass_score=auto_scores,
        )

        # Fell through to full loop which succeeded
        assert result_entity == {"name": "Hero"}
        assert result_scores is real_scores
        assert scoring_rounds == 1  # Real judge was called

    def test_auto_pass_records_analytics(self, mock_svc, config):
        """Auto-pass still records refinement analytics."""
        auto_scores = _make_scores(8.0)

        quality_refinement_loop(
            entity_type="relationship",
            create_fn=lambda retries: {"source": "A", "target": "B"},
            judge_fn=MagicMock(),
            refine_fn=MagicMock(),
            get_name=lambda e: f"{e.get('source', '?')} -> {e.get('target', '?')}",
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("source"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            auto_pass_score=auto_scores,
        )

        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["threshold_met"] is True
        history = analytics_call.args[0]
        assert history.final_score == auto_scores.average

    def test_auto_pass_below_threshold_flags_admitted(self, mock_svc, config):
        """Auto-pass with score below threshold sets below_threshold_admitted."""
        auto_scores = _make_scores(6.0)  # Below config threshold of 8.0

        quality_refinement_loop(
            entity_type="relationship",
            create_fn=lambda retries: {"source": "A", "target": "B"},
            judge_fn=MagicMock(),
            refine_fn=MagicMock(),
            get_name=lambda e: f"{e.get('source', '?')} -> {e.get('target', '?')}",
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("source"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            auto_pass_score=auto_scores,
        )

        analytics_call = mock_svc._log_refinement_analytics.call_args
        assert analytics_call.kwargs["threshold_met"] is False
        history = analytics_call.args[0]
        assert history.below_threshold_admitted is True

    def test_auto_pass_with_initial_entity_skips_create(self, mock_svc, config):
        """Auto-pass with initial_entity uses provided entity, does not call create_fn."""
        auto_scores = _make_scores(8.0)
        create_fn = MagicMock()

        result_entity, _scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=MagicMock(),
            refine_fn=MagicMock(),
            get_name=lambda e: e.get("name", "?"),
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            initial_entity={"name": "PreBuilt"},
            auto_pass_score=auto_scores,
        )

        create_fn.assert_not_called()
        assert result_entity == {"name": "PreBuilt"}
        assert scoring_rounds == 0


class TestAnalyticsDbAbsence:
    """Tests for graceful degradation when analytics_db is absent."""

    def test_hail_mary_win_rate_gate_without_analytics_db(self, mock_svc, config):
        """Hail-mary proceeds normally when analytics_db is absent (spec=[] mock)."""
        # mock_svc has spec=[] so analytics_db raises AttributeError
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
            early_stopping_patience=10,  # Disable degradation-based early stop
            early_stopping_min_iterations=1,
            early_stopping_variance_tolerance=0.3,
        )
        judge_scores = _make_scores(6.0)  # Below threshold to trigger hail-mary

        result_entity, result_scores, _rounds = quality_refinement_loop(
            entity_type="faction",
            create_fn=lambda retries: {"name": "TestFaction"},
            judge_fn=lambda e: judge_scores,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e.get("name", "?"),
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Should return something (not crash) even without analytics_db
        assert result_entity is not None
        assert result_scores is not None

    def test_hail_mary_recording_without_analytics_db(self, mock_svc, config):
        """Hail-mary recording silently skips when analytics_db is absent."""
        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
            early_stopping_patience=10,
            early_stopping_min_iterations=1,
            early_stopping_variance_tolerance=0.3,
        )
        call_count = 0

        def create_fn(retries):
            """Create a uniquely-named entity per call."""
            nonlocal call_count
            call_count += 1
            return {"name": f"Entity{call_count}"}

        judge_scores = _make_scores(6.0)  # Below threshold

        result_entity, _scores, _rounds = quality_refinement_loop(
            entity_type="faction",
            create_fn=create_fn,
            judge_fn=lambda e: judge_scores,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e.get("name", "?"),
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
        )

        # Should complete without crashing
        assert result_entity is not None


class TestMonotonicityGuardRevertFailure:
    """Test monotonicity guard when best entity data cannot be retrieved."""

    def test_warning_logged_when_best_entity_unavailable(self, mock_svc, caplog):
        """Log a warning when monotonicity guard cannot revert to best entity."""
        # Threshold 10.0 ensures no early exit on first iteration (score=9.0).
        high_threshold_config = RefinementConfig(
            quality_threshold=10.0,
            quality_thresholds=_all_thresholds(10.0),
            max_iterations=3,
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
            early_stopping_patience=10,
            early_stopping_min_iterations=1,
            early_stopping_variance_tolerance=0.3,
        )

        judge_call = 0

        def judge_fn(_entity):
            """Return escalating scores so the loop doesn't early-stop."""
            nonlocal judge_call
            judge_call += 1
            # First judge: high score, second judge: low → triggers monotonicity guard
            if judge_call == 1:
                return _make_scores(9.0)
            return _make_scores(5.0)

        refine_call = 0

        def refine_fn(entity, _scores, _iter):
            """Return a slightly modified entity to avoid unchanged-output detection."""
            nonlocal refine_call
            refine_call += 1
            # Return a different entity to avoid "unchanged refinement" early stop
            return {**entity, "version": refine_call}

        # Return None on the first call (monotonicity guard) but valid data on
        # subsequent calls (end-of-loop best-entity retrieval).
        call_count = 0
        original_get = RefinementHistory.get_best_entity

        def get_best_entity_side_effect(history_instance):
            """Return None on first call to simulate revert failure."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # Monotonicity guard call → cannot revert
            return original_get(history_instance)

        with patch.object(
            RefinementHistory,
            "get_best_entity",
            autospec=True,
            side_effect=get_best_entity_side_effect,
        ):
            with caplog.at_level(logging.WARNING):
                result_entity, _scores, _rounds = quality_refinement_loop(
                    entity_type="character",
                    create_fn=lambda retries: {"name": "Hero"},
                    judge_fn=judge_fn,
                    refine_fn=refine_fn,
                    get_name=lambda e: e.get("name", "?"),
                    serialize=lambda e: e.copy(),
                    is_empty=lambda e: not e.get("name"),
                    score_cls=CharacterQualityScores,
                    config=high_threshold_config,
                    svc=mock_svc,
                    story_id="test-story",
                )

        assert result_entity is not None
        assert any("could not retrieve best entity data" in r.message for r in caplog.records)


class TestAnalyticsDbUnexpectedErrors:
    """Tests for except-Exception handlers when analytics_db raises unexpected errors."""

    def test_win_rate_gate_unexpected_error_proceeds_with_hail_mary(self, mock_svc, config, caplog):
        """Win-rate gate logs warning and proceeds when analytics_db raises RuntimeError."""
        # Give mock_svc a real analytics_db that raises RuntimeError
        analytics_db = MagicMock()
        analytics_db.get_hail_mary_win_rate.side_effect = RuntimeError("DB connection lost")
        mock_svc.analytics_db = analytics_db

        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
            early_stopping_patience=10,
            early_stopping_min_iterations=1,
            early_stopping_variance_tolerance=0.3,
        )
        judge_scores = _make_scores(6.0)  # Below threshold to trigger hail-mary

        with caplog.at_level(logging.WARNING):
            result_entity, result_scores, _rounds = quality_refinement_loop(
                entity_type="faction",
                create_fn=lambda retries: {"name": "TestFaction"},
                judge_fn=lambda e: judge_scores,
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: e.get("name", "?"),
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert result_entity is not None
        assert result_scores is not None
        assert any("Hail-mary win-rate query failed" in msg for msg in caplog.messages)

    def test_hail_mary_recording_unexpected_error_continues(self, mock_svc, config, caplog):
        """Hail-mary recording logs warning and continues when analytics_db raises RuntimeError."""
        analytics_db = MagicMock()
        # Win-rate gate succeeds (returns high rate so hail-mary proceeds)
        analytics_db.get_hail_mary_win_rate.return_value = 0.50
        # Recording fails with unexpected error
        analytics_db.record_hail_mary_attempt.side_effect = RuntimeError("write failed")
        mock_svc.analytics_db = analytics_db

        config = RefinementConfig(
            quality_threshold=8.0,
            quality_thresholds=_all_thresholds(8.0),
            max_iterations=2,
            creator_temperature=0.9,
            judge_temperature=0.1,
            refinement_temperature=0.7,
            early_stopping_patience=10,
            early_stopping_min_iterations=1,
            early_stopping_variance_tolerance=0.3,
        )
        call_count = 0

        def create_fn(retries):
            """Create a uniquely-named entity per call."""
            nonlocal call_count
            call_count += 1
            return {"name": f"Entity{call_count}"}

        judge_scores = _make_scores(6.0)  # Below threshold

        with caplog.at_level(logging.WARNING):
            result_entity, _scores, _rounds = quality_refinement_loop(
                entity_type="faction",
                create_fn=create_fn,
                judge_fn=lambda e: judge_scores,
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: e.get("name", "?"),
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert result_entity is not None
        assert any("failed to record hail-mary analytics" in msg for msg in caplog.messages)


def _make_mixed_scores(
    *,
    depth: float = 8.0,
    goals: float = 8.0,
    flaws: float = 8.0,
    uniqueness: float = 8.0,
    arc_potential: float = 8.0,
    temporal_plausibility: float = 8.0,
    feedback: str = "Test",
) -> CharacterQualityScores:
    """Create CharacterQualityScores with individually controllable dimensions."""
    return CharacterQualityScores(
        depth=depth,
        goals=goals,
        flaws=flaws,
        uniqueness=uniqueness,
        arc_potential=arc_potential,
        temporal_plausibility=temporal_plausibility,
        feedback=feedback,
    )


class TestMinimumScoreProperty:
    """Tests for BaseQualityScores.minimum_score property."""

    def test_minimum_score_all_equal(self):
        """When all dimensions are equal, minimum_score returns that value."""
        scores = _make_scores(7.5)
        assert scores.minimum_score == 7.5

    def test_minimum_score_mixed_dimensions(self):
        """minimum_score returns the lowest dimension value."""
        scores = _make_mixed_scores(
            depth=9.0,
            goals=8.0,
            flaws=5.0,
            uniqueness=7.0,
            arc_potential=8.0,
            temporal_plausibility=6.0,
        )
        assert scores.minimum_score == 5.0

    def test_minimum_score_excludes_average_and_feedback(self):
        """minimum_score should not consider 'average' or 'feedback' keys."""
        scores = _make_mixed_scores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=7.0,
            arc_potential=7.0,
            temporal_plausibility=7.0,
        )
        # average is 7.0, feedback is "Test" — neither should affect minimum_score
        assert scores.minimum_score == 7.0

    def test_minimum_score_single_low_dimension(self):
        """A single low dimension should be returned as the minimum."""
        scores = _make_mixed_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=3.0,
        )
        assert scores.minimum_score == 3.0


class TestDimensionFloor:
    """Tests for per-dimension minimum floor in the quality refinement loop."""

    def test_dimension_floor_blocks_passing(self, mock_svc):
        """Average >= threshold but min dim < floor → does NOT pass on first judge.

        Entity should continue to refinement. The refine_fn produces a
        slightly different entity each time so the unchanged-output early
        stop doesn't kick in. With max_iterations=3, the loop should judge
        at least twice because the first judge is blocked by the floor.
        """
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5),
            max_iterations=3,
            early_stopping_patience=5,
            early_stopping_min_iterations=3,
            dimension_minimum=6.0,
        )

        # Average = (9+9+9+9+9+3)/6 = 8.0 >= 7.5, but min dim = 3.0 < 6.0
        low_dim_scores = _make_mixed_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=3.0,
        )

        entity = {"name": "FloorTest"}
        judge_calls = 0
        refine_call = 0

        def judge_fn(e):
            """Return scores with one dimension below floor."""
            nonlocal judge_calls
            judge_calls += 1
            return low_dim_scores

        def refine_fn(e, s, i):
            """Return a different entity each time to avoid unchanged-output detection."""
            nonlocal refine_call
            refine_call += 1
            return {"name": "FloorTest", "version": refine_call}

        _result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
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

        # Should have been judged more than once (first judge didn't pass due to floor)
        assert judge_calls >= 2
        assert scoring_rounds >= 2

    def test_dimension_floor_zero_disabled(self, mock_svc):
        """Floor=0.0 disables the check — passes with any dimension values."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5),
            max_iterations=5,
            early_stopping_patience=2,
            dimension_minimum=0.0,
        )

        # Average = (9+9+9+9+9+3)/6 = 8.0 >= 7.5, min dim = 3.0
        # With floor=0.0, the floor check is disabled → passes immediately
        scores = _make_mixed_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=3.0,
        )

        entity = {"name": "DisabledFloor"}

        result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
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
        assert scoring_rounds == 1  # Passed on first judge

    def test_dimension_floor_all_above(self, mock_svc):
        """All dims above floor + average above threshold → passes immediately."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5),
            max_iterations=5,
            early_stopping_patience=2,
            dimension_minimum=6.0,
        )

        # All dimensions >= 6.0, average = 8.0 >= 7.5 → passes
        scores = _make_mixed_scores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
        )

        entity = {"name": "AllAbove"}

        result_entity, result_scores, scoring_rounds = quality_refinement_loop(
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
        assert scoring_rounds == 1  # Passed on first judge

    def test_dimension_floor_exact_boundary(self, mock_svc):
        """Dims exactly at floor value should pass (strict < comparison)."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5),
            max_iterations=5,
            early_stopping_patience=2,
            dimension_minimum=6.0,
        )

        # Use scores that pass threshold AND are exactly at floor:
        scores_passing = _make_mixed_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=6.0,  # exactly at floor
        )

        entity = {"name": "BoundaryFloor"}

        result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda retries: entity,
            judge_fn=lambda e: scores_passing,
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
        assert scoring_rounds == 1  # Passed on first judge — 6.0 is NOT < 6.0

    def test_dimension_floor_log_message(self, mock_svc, caplog):
        """Floor violation should log 'dimension floor violated' message."""
        config = RefinementConfig(
            quality_threshold=7.5,
            quality_thresholds=_all_thresholds(7.5),
            max_iterations=2,
            early_stopping_patience=5,
            early_stopping_min_iterations=2,
            dimension_minimum=6.0,
        )

        scores = _make_mixed_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=3.0,
        )

        entity = {"name": "LogTest"}
        refine_call = 0

        def refine_fn(e, s, i):
            """Return different entity each time."""
            nonlocal refine_call
            refine_call += 1
            return {"name": "LogTest", "v": refine_call}

        with caplog.at_level(logging.INFO):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda retries: entity,
                judge_fn=lambda e: scores,
                refine_fn=refine_fn,
                get_name=lambda e: e["name"],
                serialize=lambda e: e.copy(),
                is_empty=lambda e: not e.get("name"),
                score_cls=CharacterQualityScores,
                config=config,
                svc=mock_svc,
                story_id="test-story",
            )

        assert any("dimension floor violated" in msg for msg in caplog.messages)


class TestMinimumScoreCalendar:
    """Test minimum_score with CalendarQualityScores to verify cross-subclass support."""

    def test_calendar_scores_minimum(self):
        """CalendarQualityScores.minimum_score returns the lowest of its 4 dimensions."""
        scores = CalendarQualityScores(
            internal_consistency=9.0,
            thematic_fit=4.0,
            completeness=8.0,
            uniqueness=7.0,
            feedback="Test",
        )
        assert scores.minimum_score == 4.0

    def test_calendar_scores_all_equal(self):
        """When all CalendarQualityScores dimensions are equal, returns that value."""
        scores = CalendarQualityScores(
            internal_consistency=7.5,
            thematic_fit=7.5,
            completeness=7.5,
            uniqueness=7.5,
            feedback="Test",
        )
        assert scores.minimum_score == 7.5

    def test_minimum_score_raises_on_broken_model(self):
        """minimum_score raises StoryFactoryError when to_dict() has no numeric dimensions."""
        scores = _make_scores(8.0)
        # Patch to_dict to return only metadata keys (simulating a broken model)
        with patch.object(
            type(scores), "to_dict", return_value={"average": 8.0, "feedback": "Test"}
        ):
            with pytest.raises(StoryFactoryError, match="no numeric scoring dimensions"):
                _ = scores.minimum_score


class TestPrepareModelCallbacks:
    """Test that prepare_creator/prepare_judge callbacks are called at the right points."""

    def test_prepare_callbacks_called_in_create_judge_flow(self, mock_svc, config):
        """Callbacks called: prepare_creator before create, prepare_judge before judge."""
        call_order: list[str] = []
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        def prep_creator():
            """Record prepare_creator call."""
            call_order.append("prepare_creator")

        def prep_judge():
            """Record prepare_judge call."""
            call_order.append("prepare_judge")

        def create_fn(retries):
            """Record create call and return entity."""
            call_order.append("create")
            return entity

        def judge_fn(e):
            """Record judge call and return scores."""
            call_order.append("judge")
            return scores

        quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            prepare_creator=prep_creator,
            prepare_judge=prep_judge,
        )

        assert call_order == ["prepare_creator", "create", "prepare_judge", "judge"]

    def test_prepare_callbacks_called_in_refine_flow(self, mock_svc, config):
        """Callbacks called before refine and before re-judge."""
        call_order: list[str] = []
        original = {"name": "Hero"}
        refined = {"name": "Hero v2"}
        low_scores = _make_scores(6.0)
        high_scores = _make_scores(8.5)
        judge_call_count = 0

        def prep_creator():
            """Record prepare_creator call."""
            call_order.append("prepare_creator")

        def prep_judge():
            """Record prepare_judge call."""
            call_order.append("prepare_judge")

        def create_fn(retries):
            """Record create call and return original entity."""
            call_order.append("create")
            return original

        def judge_fn(e):
            """Return low scores first, then high scores."""
            nonlocal judge_call_count
            call_order.append("judge")
            judge_call_count += 1
            return low_scores if judge_call_count == 1 else high_scores

        def refine_fn(e, s, i):
            """Record refine call and return refined entity."""
            call_order.append("refine")
            return refined

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
            prepare_creator=prep_creator,
            prepare_judge=prep_judge,
        )

        # Expected: create(prep_c, create) -> judge(prep_j, judge) -> refine(prep_c, refine) -> judge(prep_j, judge)
        assert call_order == [
            "prepare_creator",
            "create",
            "prepare_judge",
            "judge",
            "prepare_creator",
            "refine",
            "prepare_judge",
            "judge",
        ]

    def test_none_callbacks_are_no_ops(self, mock_svc, config):
        """When callbacks are None, the loop runs normally without errors."""
        entity = {"name": "Hero"}
        scores = _make_scores(8.5)

        result_entity, _result_scores, iterations = quality_refinement_loop(
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
            prepare_creator=None,
            prepare_judge=None,
        )

        assert result_entity == entity
        assert iterations == 1

    def test_prepare_callbacks_called_in_hail_mary(self, mock_svc, config):
        """Callbacks called before hail-mary create and judge."""
        # max_iterations must be > 1 for hail-mary to trigger
        config.max_iterations = 2
        config.early_stopping_patience = 1  # Early stop after 1 plateau
        call_order: list[str] = []
        entity = {"name": "Hero"}
        low_scores = _make_scores(6.0)

        # Mock analytics_db for hail-mary
        mock_svc.analytics_db = MagicMock()
        mock_svc.analytics_db.get_hail_mary_win_rate.return_value = 0.5
        mock_svc.analytics_db.record_hail_mary_attempt = MagicMock()

        judge_count = 0

        def prep_creator():
            """Record prepare_creator call."""
            call_order.append("prepare_creator")

        def prep_judge():
            """Record prepare_judge call."""
            call_order.append("prepare_judge")

        def create_fn(retries):
            """Record create call and return entity."""
            call_order.append("create")
            return entity

        def judge_fn(e):
            """Record judge call and always return low scores."""
            nonlocal judge_count
            call_order.append("judge")
            judge_count += 1
            return low_scores

        quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=judge_fn,
            refine_fn=lambda e, s, i: e,
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            prepare_creator=prep_creator,
            prepare_judge=prep_judge,
        )

        # Main loop iter 0: prep_creator, create, prep_judge, judge (below threshold)
        # Main loop iter 1: prep_creator before refine → unchanged detection → early stop
        # Hail-mary: prep_creator, create, prep_judge, judge
        assert call_order == [
            "prepare_creator",
            "create",
            "prepare_judge",
            "judge",
            "prepare_creator",  # refine attempt (before unchanged detection breaks)
            "prepare_creator",  # hail-mary create
            "create",
            "prepare_judge",
            "judge",
        ]

    def test_prepare_creator_called_in_auto_pass(self, mock_svc, config):
        """Auto-pass path calls prepare_creator before create_fn when entity is None."""
        call_order: list[str] = []
        auto_scores = _make_scores(8.5)

        def prep_creator():
            """Record prepare_creator call."""
            call_order.append("prepare_creator")

        def create_fn(retries):
            """Record create call and return entity."""
            call_order.append("create")
            return {"name": "Hero"}

        quality_refinement_loop(
            entity_type="relationship",
            create_fn=create_fn,
            judge_fn=MagicMock(),
            refine_fn=MagicMock(),
            get_name=lambda e: e["name"],
            serialize=lambda e: e.copy(),
            is_empty=lambda e: not e.get("name"),
            score_cls=CharacterQualityScores,
            config=config,
            svc=mock_svc,
            story_id="test-story",
            auto_pass_score=auto_scores,
            prepare_creator=prep_creator,
            prepare_judge=MagicMock(),
        )

        # prepare_creator should be called before create in auto-pass path
        assert call_order == ["prepare_creator", "create"]
