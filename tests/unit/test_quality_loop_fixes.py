"""Tests for specific fixes in the quality refinement loop.

Covers:
- C2: Floor check uses minimum_score_for_average (not minimum_score)
- M3: Hail-mary skip on identical output
- H2: iteration_callback parameter
- H3: Per-dimension regression logging
- minimum_score_for_average error when all dimensions excluded
- resolve_model_pair VRAM pair fit and exception fallbacks
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.memory.world_quality import CharacterQualityScores, RefinementConfig
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import StoryFactoryError, VRAMAllocationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_iterations: int = 3,
    dimension_minimum: float = 6.0,
    threshold: float = 7.5,
) -> RefinementConfig:
    """Build a RefinementConfig with per-entity thresholds."""
    return RefinementConfig(
        max_iterations=max_iterations,
        quality_threshold=threshold,
        quality_thresholds={"character": threshold},
        dimension_minimum=dimension_minimum,
        early_stopping_patience=5,
        early_stopping_min_iterations=2,
        score_plateau_tolerance=0.0,
    )


def _make_svc() -> MagicMock:
    """Build a mock WorldQualityService with analytics stubs."""
    svc = MagicMock()
    svc._log_refinement_analytics = MagicMock()
    svc.analytics_db.get_hail_mary_win_rate = MagicMock(return_value=0.5)
    svc.analytics_db.record_hail_mary_attempt = MagicMock()
    return svc


def _make_character_scores(
    *,
    depth: float = 8.0,
    goals: float = 8.0,
    flaws: float = 8.0,
    uniqueness: float = 8.0,
    arc_potential: float = 8.0,
    temporal_plausibility: float = 8.0,
    feedback: str = "Good",
) -> CharacterQualityScores:
    """Create a CharacterQualityScores object with customizable dimension values."""
    return CharacterQualityScores(
        depth=depth,
        goals=goals,
        flaws=flaws,
        uniqueness=uniqueness,
        arc_potential=arc_potential,
        temporal_plausibility=temporal_plausibility,
        feedback=feedback,
    )


# Common callables for dict-based entities
_serialize = lambda d: dict(d)  # noqa: E731
_get_name = lambda d: d.get("name", "")  # noqa: E731
_is_empty = lambda d: not d.get("name")  # noqa: E731


# ---------------------------------------------------------------------------
# C2: Floor check uses minimum_score_for_average
# ---------------------------------------------------------------------------


class TestC2FloorCheckUsesMinimumScoreForAverage:
    """The below_floor check uses scores.minimum_score_for_average so that
    excluded dimensions (like temporal_plausibility in CharacterQualityScores)
    do not trigger the floor check."""

    def test_low_temporal_plausibility_does_not_trigger_floor(self):
        """A character with temporal_plausibility=3.0 but all other
        dimensions >= 7.0 should NOT trigger the floor check (pass on
        iteration 1)."""
        entity = {"name": "Alice"}
        # temporal_plausibility is excluded from average and from floor check
        scores = _make_character_scores(temporal_plausibility=3.0)
        # Average of included dims (depth=8, goals=8, flaws=8, uniqueness=8, arc=8) = 8.0
        assert scores.average == 8.0
        # minimum_score_for_average ignores temporal_plausibility
        assert scores.minimum_score_for_average == 8.0
        # minimum_score includes temporal_plausibility
        assert scores.minimum_score == 3.0

        config = _make_config(max_iterations=1, dimension_minimum=6.0, threshold=7.5)
        svc = _make_svc()

        result_entity, result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: entity,
            judge_fn=lambda _e: scores,
            refine_fn=lambda e, s, i: e,
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
        )

        # Should pass on first iteration: average 8.0 >= 7.5, no floor violation
        assert result_entity == entity
        assert result_scores.average == 8.0
        assert scoring_rounds == 1

    def test_low_included_dimension_triggers_floor(self):
        """A character with depth=4.0 (included in average) should trigger
        the floor check, preventing early exit even if average meets threshold."""
        entity = {"name": "Bob"}
        # depth=4.0 is below the 6.0 dimension_minimum, and depth IS included in average
        scores = _make_character_scores(
            depth=4.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=9.0,
        )
        # Average of included dims: (4+9+9+9+9)/5 = 8.0
        assert scores.average == 8.0
        # minimum_score_for_average includes depth (4.0)
        assert scores.minimum_score_for_average == 4.0

        # With max_iterations=1, the loop cannot refine — it will return
        # the entity below threshold because it tried all iterations.
        config = _make_config(max_iterations=1, dimension_minimum=6.0, threshold=7.5)
        svc = _make_svc()

        result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: entity,
            judge_fn=lambda _e: scores,
            refine_fn=lambda e, s, i: e,
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
        )

        # Average meets threshold (8.0 >= 7.5) but floor is violated (4.0 < 6.0).
        # With only 1 iteration, the entity is admitted below threshold.
        assert result_entity == entity
        assert scoring_rounds == 1
        # Verify below_threshold_admitted was logged (threshold_met=False)
        analytics_call = svc._log_refinement_analytics.call_args
        assert analytics_call is not None
        assert analytics_call.kwargs.get("threshold_met") is False or (
            len(analytics_call.args) > 2 and analytics_call[1].get("threshold_met") is False
        )


# ---------------------------------------------------------------------------
# M3: Hail-mary skip on identical output
# ---------------------------------------------------------------------------


class TestM3HailMaryIdenticalOutputSkip:
    """When the hail-mary creates an entity identical to the best entity,
    the judge call is skipped."""

    def test_identical_hail_mary_skips_judge(self):
        """Hail-mary produces identical output -> judge is NOT called for
        the hail-mary entity."""
        entity = {"name": "Charlie", "trait": "brave"}
        # Score below threshold to trigger hail-mary
        low_scores = _make_character_scores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
        )
        # Average = 6.0 < 7.5 threshold

        judge_call_count = 0

        def counting_judge(e):
            """Count judge invocations and return low scores."""
            nonlocal judge_call_count
            judge_call_count += 1
            return low_scores

        # create_fn always returns the same entity (including hail-mary attempt)
        config = _make_config(max_iterations=1, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        _result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: dict(entity),
            judge_fn=counting_judge,
            refine_fn=lambda e, s, i: e,
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
        )

        # Judge should be called exactly once (for the main loop iteration).
        # The hail-mary should detect identical output and skip the judge call.
        assert judge_call_count == 1
        assert scoring_rounds == 1

    def test_different_hail_mary_calls_judge(self):
        """Hail-mary produces different output -> judge IS called."""
        original = {"name": "Diana", "trait": "cunning"}
        fresh = {"name": "Diana", "trait": "wise"}
        low_scores = _make_character_scores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
        )

        judge_call_count = 0
        judged_entities: list[dict] = []

        def counting_judge(e):
            """Count judge invocations, record judged entities, and return low scores."""
            nonlocal judge_call_count
            judge_call_count += 1
            judged_entities.append(e)
            return low_scores

        call_index = 0

        def create_fn(_retries):
            """Create entity, returning original on first call and fresh on hail-mary."""
            nonlocal call_index
            call_index += 1
            # First call returns original, second (hail-mary) returns fresh
            if call_index == 1:
                return dict(original)
            return dict(fresh)

        # max_iterations must be > 1 for hail-mary to trigger
        config = _make_config(max_iterations=2, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        quality_refinement_loop(
            entity_type="character",
            create_fn=create_fn,
            judge_fn=counting_judge,
            refine_fn=lambda e, s, i: dict(original),
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
        )

        # Judge called in main loop iterations + once for the hail-mary
        # The hail-mary entity is different so judge must be invoked for it
        assert judge_call_count >= 2
        # Prove the fresh hail-mary entity (trait="wise") actually reached judge_fn
        assert any(e.get("trait") == "wise" for e in judged_entities)


# ---------------------------------------------------------------------------
# H2: iteration_callback parameter
# ---------------------------------------------------------------------------


class TestH2IterationCallback:
    """quality_refinement_loop accepts an iteration_callback and fires it
    at the start of each iteration."""

    def test_callback_called_with_correct_args(self):
        """Callback is called with (current_iteration_1_indexed, max_iterations,
        entity_name) on each iteration."""
        entity = {"name": "Eve"}
        high_scores = _make_character_scores()  # avg=8.0, passes threshold
        callback = MagicMock()

        config = _make_config(max_iterations=3, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: dict(entity),
            judge_fn=lambda _e: high_scores,
            refine_fn=lambda e, s, i: e,
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
            iteration_callback=callback,
        )

        # Passes on iteration 1, so callback called once: (1, 3, "")
        # entity_name is "" because entity hasn't been created yet at callback time
        # on the first iteration (get_name called on None-check entity)
        assert callback.call_count == 1
        cb_args = callback.call_args[0]
        assert cb_args[0] == 1  # current iteration (1-indexed)
        assert cb_args[1] == 3  # max iterations
        # entity_name is "" on first iteration (entity is None before creation)
        assert cb_args[2] == ""

    def test_callback_receives_entity_name_on_subsequent_iterations(self):
        """After the entity is created, callback receives the entity name."""
        entity = {"name": "Frank"}
        iteration_count = 0

        def score_fn(_e):
            """Score function that returns low scores on first iteration, high on second."""
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 1:
                # Below threshold to force a second iteration
                return _make_character_scores(
                    depth=5.0,
                    goals=5.0,
                    flaws=5.0,
                    uniqueness=5.0,
                    arc_potential=5.0,
                )
            # Above threshold on second iteration
            return _make_character_scores()

        callback = MagicMock()
        config = _make_config(max_iterations=3, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: dict(entity),
            judge_fn=score_fn,
            refine_fn=lambda e, s, i: dict(entity),
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
            iteration_callback=callback,
        )

        # At least 2 calls: first with "" (entity not yet created), second with "Frank"
        assert callback.call_count >= 2
        # Second call should have entity name
        second_call_args = callback.call_args_list[1][0]
        assert second_call_args[0] == 2  # iteration 2
        assert second_call_args[2] == "Frank"  # entity name

    def test_no_callback_no_error(self):
        """When iteration_callback is None, no error occurs."""
        entity = {"name": "Grace"}
        high_scores = _make_character_scores()

        config = _make_config(max_iterations=1, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        # Should not raise
        result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
            entity_type="character",
            create_fn=lambda _retries: entity,
            judge_fn=lambda _e: high_scores,
            refine_fn=lambda e, s, i: e,
            get_name=_get_name,
            serialize=_serialize,
            is_empty=_is_empty,
            score_cls=CharacterQualityScores,
            config=config,
            svc=svc,
            story_id="test-story",
            iteration_callback=None,
        )

        assert result_entity == entity
        assert scoring_rounds == 1

    def test_callback_exception_is_swallowed(self, caplog):
        """When iteration_callback raises, the loop continues with a warning."""
        entity = {"name": "Grace"}
        high_scores = _make_character_scores()

        config = _make_config(max_iterations=1, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        def exploding_callback(_cur: int, _max: int, _name: str) -> None:
            """Raise to simulate callback failure."""
            raise RuntimeError("UI crashed")

        with caplog.at_level(logging.WARNING):
            result_entity, _result_scores, scoring_rounds = quality_refinement_loop(
                entity_type="character",
                create_fn=lambda _retries: entity,
                judge_fn=lambda _e: high_scores,
                refine_fn=lambda e, s, i: e,
                get_name=_get_name,
                serialize=_serialize,
                is_empty=_is_empty,
                score_cls=CharacterQualityScores,
                config=config,
                svc=svc,
                story_id="test-story",
                iteration_callback=exploding_callback,
            )

        assert result_entity == entity
        assert scoring_rounds == 1
        assert any("iteration callback failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# H3: Per-dimension regression logging
# ---------------------------------------------------------------------------


class TestH3PerDimensionRegressionLogging:
    """When a score regresses, per-dimension changes are logged via
    logger.info with dimension deltas."""

    def test_dimension_changes_logged_on_regression(self, caplog):
        """Verify logger.info is called with per-dimension change details
        when score regresses."""
        iteration_count = 0

        # First iteration: below threshold so loop continues
        first_scores = _make_character_scores(
            depth=7.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=7.0,
            arc_potential=7.0,
            temporal_plausibility=7.0,
            feedback="Below threshold",
        )
        # Second iteration: regression — depth drops from 7 to 4, arc rises from 7 to 8
        regressed_scores = _make_character_scores(
            depth=4.0,
            goals=7.0,
            flaws=7.0,
            uniqueness=7.0,
            arc_potential=8.0,
            temporal_plausibility=7.0,
            feedback="Regressed",
        )
        # Third iteration: pass to exit the loop
        passing_scores = _make_character_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
            temporal_plausibility=9.0,
            feedback="Pass",
        )

        def score_fn(_e):
            """Return different scores across iterations to test regression logging."""
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 1:
                return first_scores
            if iteration_count == 2:
                return regressed_scores
            return passing_scores

        # refine_fn must return data that differs from previous iteration
        # to avoid the "unchanged refinement output" early break
        refine_call = 0

        def refine_fn(e, s, i):
            """Increment version counter to create distinct refinements across iterations."""
            nonlocal refine_call
            refine_call += 1
            return {"name": "Hank", "version": refine_call}

        config = _make_config(
            max_iterations=5,
            dimension_minimum=0.0,
            threshold=7.5,
        )
        svc = _make_svc()

        with caplog.at_level(
            logging.INFO, logger="src.services.world_quality_service._quality_loop"
        ):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda _retries: {"name": "Hank", "version": 0},
                judge_fn=score_fn,
                refine_fn=refine_fn,
                get_name=_get_name,
                serialize=_serialize,
                is_empty=_is_empty,
                score_cls=CharacterQualityScores,
                config=config,
                svc=svc,
                story_id="test-story",
            )

        # Find the dimension-change log message
        dim_change_messages = [
            r.message for r in caplog.records if "dimension changes vs best" in r.message
        ]
        assert len(dim_change_messages) >= 1, (
            f"Expected at least one 'dimension changes vs best' log message, "
            f"got: {[r.message for r in caplog.records]}"
        )

        msg = dim_change_messages[0]
        # depth dropped from 7.0 to 4.0 => delta = -3.0
        assert "depth -3.0" in msg
        # arc_potential rose from 7.0 to 8.0 => delta = +1.0
        assert "arc_potential +1.0" in msg

    def test_no_dimension_log_when_no_regression(self, caplog):
        """When scores improve, no per-dimension regression log should appear."""
        entity = {"name": "Ivy"}
        iteration_count = 0

        first_scores = _make_character_scores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
        )
        second_scores = _make_character_scores(
            depth=9.0,
            goals=9.0,
            flaws=9.0,
            uniqueness=9.0,
            arc_potential=9.0,
        )

        def score_fn(_e):
            """Return improving scores to ensure no regression logging occurs."""
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 1:
                return first_scores
            return second_scores

        config = _make_config(max_iterations=3, dimension_minimum=0.0, threshold=7.5)
        svc = _make_svc()

        with caplog.at_level(
            logging.INFO, logger="src.services.world_quality_service._quality_loop"
        ):
            quality_refinement_loop(
                entity_type="character",
                create_fn=lambda _retries: dict(entity),
                judge_fn=score_fn,
                refine_fn=lambda e, s, i: dict(entity),
                get_name=_get_name,
                serialize=_serialize,
                is_empty=_is_empty,
                score_cls=CharacterQualityScores,
                config=config,
                svc=svc,
                story_id="test-story",
            )

        dim_change_messages = [
            r.message for r in caplog.records if "dimension changes vs best" in r.message
        ]
        assert len(dim_change_messages) == 0, (
            f"Expected no 'dimension changes vs best' log messages when scores improve, "
            f"got: {dim_change_messages}"
        )


# ---------------------------------------------------------------------------
# minimum_score_for_average: all dimensions excluded
# ---------------------------------------------------------------------------


class TestMinimumScoreForAverageAllExcluded:
    """Test minimum_score_for_average when all dimensions are excluded."""

    def test_raises_when_all_dimensions_excluded(self):
        """Should raise StoryFactoryError when every dimension is in _EXCLUDED_FROM_AVERAGE."""
        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
        )
        all_fields = frozenset(
            name
            for name, _field in CharacterQualityScores.model_fields.items()
            if name != "feedback"
        )
        with patch.object(type(scores), "_EXCLUDED_FROM_AVERAGE", all_fields):
            with pytest.raises(StoryFactoryError, match="no dimensions remain"):
                _ = scores.minimum_score_for_average


# ---------------------------------------------------------------------------
# resolve_model_pair: VRAM pair fit and exception fallbacks
# ---------------------------------------------------------------------------


class TestResolveModelPairVRAM:
    """Tests for resolve_model_pair VRAM pair fit and exception handling."""

    def _make_service(self):
        """Build a mock WorldQualityService for resolve_model_pair tests."""
        svc = MagicMock()
        svc.ENTITY_CREATOR_ROLES = {"character": "writer"}
        svc.ENTITY_JUDGE_ROLES = {"character": "judge"}
        svc._model_cache.get_creator_model.return_value = None
        svc._model_cache.get_judge_model.return_value = None
        # store methods return the stored model (race-safety canonicalization)
        svc._model_cache.store_creator_model.side_effect = lambda _role, model: model
        svc._model_cache.store_judge_model.side_effect = lambda _role, model, _creator: model
        svc.settings.use_per_agent_models = False
        svc.settings.default_model = "auto"
        svc.mode_service.get_model_for_agent.side_effect = lambda role: {
            "writer": "creator-model:8b",
            "judge": "judge-model:8b",
        }[role]
        return svc

    @patch("src.services.world_quality_service._model_resolver.pair_fits", return_value=False)
    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_pair_does_not_fit_falls_back_to_self_judging(self, mock_snapshot, mock_pair_fits):
        """When pair doesn't fit in VRAM, judge falls back to creator model."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        snapshot = MagicMock()
        snapshot.installed_models = {"creator-model:8b": 14.0, "judge-model:8b": 18.0}
        snapshot.available_vram_gb = 24.0
        mock_snapshot.return_value = snapshot

        svc = self._make_service()
        creator, judge = resolve_model_pair(svc, "character")

        assert creator == "creator-model:8b"
        assert judge == "creator-model:8b"

    @patch(
        "src.services.world_quality_service._model_resolver.get_vram_snapshot",
        side_effect=RuntimeError("nvidia-smi not found"),
    )
    def test_exception_during_vram_check_proceeds_with_both_models(self, mock_snapshot):
        """When VRAM snapshot raises, proceed with both resolved models (optimistic)."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        svc = self._make_service()
        creator, judge = resolve_model_pair(svc, "character")

        assert creator == "creator-model:8b"
        assert judge == "judge-model:8b"

    @patch(
        "src.services.world_quality_service._model_resolver.get_vram_snapshot",
        side_effect=TypeError("unexpected type from broken VRAM logic"),
    )
    def test_unexpected_exception_propagates_from_vram_check(self, mock_snapshot):
        """TypeError (not in narrowed catch list) propagates instead of being swallowed."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        svc = self._make_service()
        with pytest.raises(TypeError, match="unexpected type"):
            resolve_model_pair(svc, "character")

    @patch("src.services.world_quality_service._model_resolver.pair_fits", return_value=True)
    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_pair_fits_returns_both_models(self, mock_snapshot, mock_pair_fits):
        """When pair fits in VRAM, return both distinct models."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        snapshot = MagicMock()
        snapshot.installed_models = {"creator-model:8b": 5.0, "judge-model:8b": 4.0}
        snapshot.available_vram_gb = 20.0
        mock_snapshot.return_value = snapshot

        svc = self._make_service()
        creator, judge = resolve_model_pair(svc, "character")

        assert creator == "creator-model:8b"
        assert judge == "judge-model:8b"

    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_cached_pair_returned_without_vram_check(self, mock_snapshot):
        """When both models are in cache, return immediately without VRAM check."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        svc = self._make_service()
        svc._model_cache.get_creator_model.return_value = "cached-creator:8b"
        svc._model_cache.get_judge_model.return_value = "cached-judge:8b"

        creator, judge = resolve_model_pair(svc, "character")

        assert creator == "cached-creator:8b"
        assert judge == "cached-judge:8b"
        mock_snapshot.assert_not_called()

    def test_unknown_entity_type_raises_value_error(self):
        """resolve_model_pair rejects unknown entity types with ValueError."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        svc = self._make_service()

        with pytest.raises(ValueError, match="Unknown entity_type"):
            resolve_model_pair(svc, "nonexistent_type")

    def test_entity_missing_from_judge_roles_raises_value_error(self):
        """Entity in ENTITY_CREATOR_ROLES but missing from ENTITY_JUDGE_ROLES raises."""
        from src.services.world_quality_service._model_resolver import resolve_model_pair

        svc = self._make_service()
        # Add 'widget' to creator roles but not judge roles
        svc.ENTITY_CREATOR_ROLES["widget"] = "widget_creator"

        with pytest.raises(ValueError, match="Unknown entity_type 'widget'"):
            resolve_model_pair(svc, "widget")


class TestMakeModelPreparers:
    """Tests for make_model_preparers() factory function."""

    def _make_service(self, *, same_model: bool = False):
        """Build a mock WorldQualityService for make_model_preparers tests."""
        svc = MagicMock()
        svc.ENTITY_CREATOR_ROLES = {"character": "writer"}
        svc.ENTITY_JUDGE_ROLES = {"character": "judge"}
        svc._model_cache.get_creator_model.return_value = None
        svc._model_cache.get_judge_model.return_value = None
        svc._model_cache.store_creator_model.side_effect = lambda _role, model: model
        svc._model_cache.store_judge_model.side_effect = lambda _role, model, _creator: model
        svc.settings.use_per_agent_models = False
        svc.settings.default_model = "auto"
        if same_model:
            svc.mode_service.get_model_for_agent.return_value = "same-model:8b"
        else:
            svc.mode_service.get_model_for_agent.side_effect = lambda role: {
                "writer": "creator-model:8b",
                "judge": "judge-model:8b",
            }[role]
        return svc

    @patch("src.services.world_quality_service._model_resolver.pair_fits", return_value=True)
    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_same_model_returns_none_preparers(self, _snapshot, _pair_fits):
        """When creator == judge, both preparers should be None."""
        from src.services.world_quality_service._model_resolver import make_model_preparers

        svc = self._make_service(same_model=True)
        prep_creator, prep_judge = make_model_preparers(svc, "character")

        assert prep_creator is None
        assert prep_judge is None

    @patch("src.services.world_quality_service._model_resolver.pair_fits", return_value=True)
    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_different_models_return_callable_preparers(self, _snapshot, _pair_fits):
        """When creator != judge, both preparers should be callable."""
        from src.services.world_quality_service._model_resolver import make_model_preparers

        svc = self._make_service()
        snapshot = MagicMock()
        snapshot.installed_models = {"creator-model:8b": 5.0, "judge-model:8b": 4.0}
        snapshot.available_vram_gb = 20.0
        _snapshot.return_value = snapshot

        prep_creator, prep_judge = make_model_preparers(svc, "character")

        assert callable(prep_creator)
        assert callable(prep_judge)

    @patch("src.services.world_quality_service._model_resolver.prepare_model")
    @patch("src.services.world_quality_service._model_resolver.pair_fits", return_value=True)
    @patch("src.services.world_quality_service._model_resolver.get_vram_snapshot")
    def test_vram_error_in_preparer_propagates(self, _snapshot, _pair_fits, mock_prepare):
        """VRAMAllocationError in a preparer closure propagates (non-retryable)."""
        from src.services.world_quality_service._model_resolver import make_model_preparers

        snapshot = MagicMock()
        snapshot.installed_models = {"creator-model:8b": 5.0, "judge-model:8b": 4.0}
        snapshot.available_vram_gb = 20.0
        _snapshot.return_value = snapshot

        mock_prepare.side_effect = VRAMAllocationError("oom", model_id="creator-model:8b")

        svc = self._make_service()
        prep_creator, _prep_judge = make_model_preparers(svc, "character")

        # VRAMAllocationError must propagate — it's non-retryable
        assert prep_creator is not None
        with pytest.raises(VRAMAllocationError):
            prep_creator()
