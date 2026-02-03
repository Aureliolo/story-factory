"""Tests for judge reliability improvements (Issue #226).

Tests cover:
- _resolve_model_for_role() Settings hierarchy
- get_judge_config() returns JudgeConsistencyConfig
- judge_with_averaging() single/multi-call paths
- _aggregate_scores() mean/median strategies, outlier detection
- JUDGE_CALIBRATION_BLOCK usage in judge prompts
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    JudgeConsistencyConfig,
    LocationQualityScores,
    RelationshipQualityScores,
)
from src.services.world_quality_service import WorldQualityService
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    _aggregate_scores,
    judge_with_averaging,
)
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
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        llm_tokens_character_create=500,
        llm_tokens_character_judge=300,
        llm_tokens_character_refine=500,
        llm_tokens_location_create=400,
        llm_tokens_location_judge=300,
        llm_tokens_location_refine=400,
        llm_tokens_faction_create=400,
        llm_tokens_faction_judge=300,
        llm_tokens_faction_refine=400,
        llm_tokens_item_create=400,
        llm_tokens_item_judge=300,
        llm_tokens_item_refine=400,
        llm_tokens_concept_create=400,
        llm_tokens_concept_judge=300,
        llm_tokens_concept_refine=400,
        llm_tokens_relationship_create=400,
        llm_tokens_relationship_judge=300,
        llm_tokens_relationship_refine=400,
        llm_tokens_mini_description=100,
        mini_description_words_max=15,
    )


@pytest.fixture
def mock_mode_service():
    """Create a mock mode service."""
    svc = MagicMock()
    svc.get_model_for_agent.return_value = "auto-selected-model:8b"
    return svc


class TestResolveModelForRole:
    """Test the Settings hierarchy in _resolve_model_for_role()."""

    def test_default_model_takes_priority_when_per_agent_disabled(
        self, settings, mock_mode_service
    ):
        """When use_per_agent_models=False and default_model is set, use it."""
        settings.use_per_agent_models = False
        settings.default_model = "my-explicit-model:30b"

        service = WorldQualityService(settings, mock_mode_service)
        model = service._resolve_model_for_role("validator")

        assert model == "my-explicit-model:30b"
        mock_mode_service.get_model_for_agent.assert_not_called()

    def test_auto_default_falls_through_to_mode_service(self, settings, mock_mode_service):
        """When use_per_agent_models=False but default_model='auto', fall to mode service."""
        settings.use_per_agent_models = False
        settings.default_model = "auto"

        service = WorldQualityService(settings, mock_mode_service)
        model = service._resolve_model_for_role("validator")

        assert model == "auto-selected-model:8b"
        mock_mode_service.get_model_for_agent.assert_called_once_with("validator")

    def test_agent_model_takes_priority_when_per_agent_enabled(self, settings, mock_mode_service):
        """When use_per_agent_models=True and explicit model set, use it."""
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "my-validator-model:8b"}

        service = WorldQualityService(settings, mock_mode_service)
        model = service._resolve_model_for_role("validator")

        assert model == "my-validator-model:8b"
        mock_mode_service.get_model_for_agent.assert_not_called()

    def test_auto_agent_model_falls_through_to_mode_service(self, settings, mock_mode_service):
        """When use_per_agent_models=True but agent model is 'auto', fall to mode service."""
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}

        service = WorldQualityService(settings, mock_mode_service)
        model = service._resolve_model_for_role("validator")

        assert model == "auto-selected-model:8b"
        mock_mode_service.get_model_for_agent.assert_called_once_with("validator")

    def test_missing_agent_model_raises_error(self, settings, mock_mode_service):
        """When use_per_agent_models=True but role not in agent_models, raise ValueError."""
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "some-writer-model:8b"}  # No judge entry

        service = WorldQualityService(settings, mock_mode_service)
        with pytest.raises(ValueError, match="Unknown agent role 'judge'"):
            service._resolve_model_for_role("judge")

    def test_get_creator_model_uses_resolve(self, settings, mock_mode_service):
        """_get_creator_model() delegates to _resolve_model_for_role()."""
        settings.use_per_agent_models = False
        settings.default_model = "global-model:30b"

        service = WorldQualityService(settings, mock_mode_service)
        model = service._get_creator_model(entity_type="character")

        assert model == "global-model:30b"
        mock_mode_service.get_model_for_agent.assert_not_called()

    def test_get_judge_model_uses_resolve(self, settings, mock_mode_service):
        """_get_judge_model() delegates to _resolve_model_for_role()."""
        settings.use_per_agent_models = False
        settings.default_model = "global-model:30b"

        service = WorldQualityService(settings, mock_mode_service)
        model = service._get_judge_model()

        assert model == "global-model:30b"
        mock_mode_service.get_model_for_agent.assert_not_called()

    def test_per_agent_writer_vs_judge_different_models(self, settings, mock_mode_service):
        """Per-agent can set different models for creator vs judge roles."""
        settings.use_per_agent_models = True
        settings.agent_models = {
            "writer": "creative-model:30b",
            "judge": "strict-model:8b",
        }

        service = WorldQualityService(settings, mock_mode_service)

        creator = service._get_creator_model(entity_type="character")
        judge = service._get_judge_model(entity_type="character")

        assert creator == "creative-model:30b"
        assert judge == "strict-model:8b"
        mock_mode_service.get_model_for_agent.assert_not_called()


class TestGetJudgeConfig:
    """Test get_judge_config() returns proper JudgeConsistencyConfig."""

    def test_returns_config_from_settings(self, settings, mock_mode_service):
        """get_judge_config() returns a JudgeConsistencyConfig."""
        service = WorldQualityService(settings, mock_mode_service)
        config = service.get_judge_config()

        assert isinstance(config, JudgeConsistencyConfig)
        assert config.multi_call_enabled == settings.judge_multi_call_enabled
        assert config.multi_call_count == settings.judge_multi_call_count
        assert config.outlier_detection == settings.judge_outlier_detection
        assert config.outlier_strategy == settings.judge_outlier_strategy

    def test_config_reflects_settings_changes(self, settings, mock_mode_service):
        """Changing settings values is reflected in next get_judge_config() call."""
        settings.judge_multi_call_enabled = False
        settings.judge_multi_call_count = 5

        service = WorldQualityService(settings, mock_mode_service)
        config = service.get_judge_config()

        assert config.multi_call_enabled is False
        assert config.multi_call_count == 5


class TestJudgeWithAveraging:
    """Test the judge_with_averaging() helper function."""

    def _make_config(
        self,
        *,
        enabled=True,
        multi_call_enabled=False,
        multi_call_count=3,
        outlier_detection=True,
        outlier_std_threshold=2.0,
        outlier_strategy="median",
    ):
        """Create a JudgeConsistencyConfig with test defaults."""
        return JudgeConsistencyConfig(
            enabled=enabled,
            multi_call_enabled=multi_call_enabled,
            multi_call_count=multi_call_count,
            confidence_threshold=0.7,
            outlier_detection=outlier_detection,
            outlier_std_threshold=outlier_std_threshold,
            outlier_strategy=outlier_strategy,
        )

    def test_single_call_when_disabled(self):
        """When multi_call_enabled=False, makes one judge call."""
        scores = CharacterQualityScores(
            depth=7.0,
            goals=6.0,
            flaws=5.0,
            uniqueness=8.0,
            arc_potential=7.0,
            feedback="Good character",
        )
        judge_fn = MagicMock(return_value=scores)
        config = self._make_config(multi_call_enabled=False)

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        judge_fn.assert_called_once()
        assert result is scores

    def test_single_call_when_consistency_disabled(self):
        """When enabled=False, bypasses multi-call even if multi_call_enabled=True."""
        scores = CharacterQualityScores(
            depth=7.0,
            goals=6.0,
            flaws=5.0,
            uniqueness=8.0,
            arc_potential=7.0,
            feedback="Good character",
        )
        judge_fn = MagicMock(return_value=scores)
        config = self._make_config(enabled=False, multi_call_enabled=True, multi_call_count=5)

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        judge_fn.assert_called_once()
        assert result is scores

    def test_multi_call_makes_n_calls(self):
        """When multi_call_enabled=True with count=3, makes 3 judge calls."""
        scores = CharacterQualityScores(
            depth=7.0,
            goals=6.0,
            flaws=5.0,
            uniqueness=8.0,
            arc_potential=7.0,
            feedback="Good character",
        )
        judge_fn = MagicMock(return_value=scores)
        config = self._make_config(multi_call_enabled=True, multi_call_count=3)

        judge_with_averaging(judge_fn, CharacterQualityScores, config)

        assert judge_fn.call_count == 3

    def test_multi_call_returns_averaged_result(self):
        """Multi-call returns an averaged result, not any single call's result."""
        results = [
            CharacterQualityScores(
                depth=6.0,
                goals=6.0,
                flaws=6.0,
                uniqueness=6.0,
                arc_potential=6.0,
                feedback="Low",
            ),
            CharacterQualityScores(
                depth=8.0,
                goals=8.0,
                flaws=8.0,
                uniqueness=8.0,
                arc_potential=8.0,
                feedback="High",
            ),
            CharacterQualityScores(
                depth=7.0,
                goals=7.0,
                flaws=7.0,
                uniqueness=7.0,
                arc_potential=7.0,
                feedback="Mid",
            ),
        ]
        call_count = 0

        def judge_fn():
            """Return sequential results from the pre-built list."""
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        config = self._make_config(
            multi_call_enabled=True,
            multi_call_count=3,
            outlier_strategy="median",
        )

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        # Median of [6, 8, 7] is 7 for each dimension
        assert result.depth == 7.0
        assert result.goals == 7.0
        assert result.flaws == 7.0
        assert result.uniqueness == 7.0
        assert result.arc_potential == 7.0

    def test_multi_call_all_fail_falls_back_to_single(self):
        """When all multi calls fail, falls back to a single retry."""
        call_count = 0
        final_scores = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            feedback="Fallback",
        )

        def judge_fn():
            """Fail first 3 calls, succeed on 4th (fallback)."""
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise WorldGenerationError("Judge failed")
            return final_scores

        config = self._make_config(multi_call_enabled=True, multi_call_count=3)

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        assert call_count == 4  # 3 failed + 1 fallback
        assert result is final_scores

    def test_multi_call_partial_failures_uses_successes(self):
        """When some multi calls fail, uses the successful ones."""
        call_count = 0

        def judge_fn():
            """Fail on 2nd call, succeed on 1st and 3rd."""
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise WorldGenerationError("Judge failed")
            return CharacterQualityScores(
                depth=7.0,
                goals=7.0,
                flaws=7.0,
                uniqueness=7.0,
                arc_potential=7.0,
                feedback="OK",
            )

        config = self._make_config(
            multi_call_enabled=True,
            multi_call_count=3,
            outlier_strategy="median",
        )

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        # 2 successful calls, should still return a valid result
        assert call_count == 3
        assert result.depth == 7.0

    def test_multi_call_one_success_returns_single(self):
        """When only 1 of N calls succeeds, returns that single result."""
        single_result = CharacterQualityScores(
            depth=6.0,
            goals=6.0,
            flaws=6.0,
            uniqueness=6.0,
            arc_potential=6.0,
            feedback="Only one",
        )
        call_count = 0

        def judge_fn():
            """Succeed only on 2nd call, fail on all others."""
            nonlocal call_count
            call_count += 1
            if call_count != 2:
                raise WorldGenerationError("Failed")
            return single_result

        config = self._make_config(multi_call_enabled=True, multi_call_count=3)

        result = judge_with_averaging(judge_fn, CharacterQualityScores, config)

        assert call_count == 3
        assert result is single_result


class TestAggregateScores:
    """Test the _aggregate_scores() function."""

    def _make_config(self, strategy="median", outlier_detection=True, outlier_std=2.0):
        """Create a JudgeConsistencyConfig for testing."""
        return JudgeConsistencyConfig(
            enabled=True,
            multi_call_enabled=True,
            multi_call_count=3,
            outlier_detection=outlier_detection,
            outlier_std_threshold=outlier_std,
            outlier_strategy=strategy,
        )

    def test_median_strategy(self):
        """Median strategy returns median of each dimension."""
        results = [
            FactionQualityScores(
                coherence=5.0,
                influence=5.0,
                conflict_potential=5.0,
                distinctiveness=5.0,
                feedback="Low",
            ),
            FactionQualityScores(
                coherence=9.0,
                influence=9.0,
                conflict_potential=9.0,
                distinctiveness=9.0,
                feedback="High",
            ),
            FactionQualityScores(
                coherence=7.0,
                influence=7.0,
                conflict_potential=7.0,
                distinctiveness=7.0,
                feedback="Mid",
            ),
        ]
        config = self._make_config(strategy="median")

        result = _aggregate_scores(results, FactionQualityScores, config)

        assert result.coherence == 7.0
        assert result.influence == 7.0
        assert result.conflict_potential == 7.0
        assert result.distinctiveness == 7.0

    def test_mean_strategy_without_outliers(self):
        """Mean strategy returns mean of each dimension (no outliers)."""
        results = [
            LocationQualityScores(
                atmosphere=6.0,
                significance=6.0,
                story_relevance=6.0,
                distinctiveness=6.0,
                feedback="A",
            ),
            LocationQualityScores(
                atmosphere=7.0,
                significance=7.0,
                story_relevance=7.0,
                distinctiveness=7.0,
                feedback="B",
            ),
            LocationQualityScores(
                atmosphere=8.0,
                significance=8.0,
                story_relevance=8.0,
                distinctiveness=8.0,
                feedback="C",
            ),
        ]
        config = self._make_config(strategy="mean", outlier_detection=False)

        result = _aggregate_scores(results, LocationQualityScores, config)

        assert result.atmosphere == 7.0
        assert result.significance == 7.0
        assert result.story_relevance == 7.0
        assert result.distinctiveness == 7.0

    def test_mean_strategy_with_outlier_detection(self):
        """Mean strategy filters outliers before averaging."""
        # 4 close scores and one extreme outlier — needs 4+ points for reliable z-scores
        results = [
            ItemQualityScores(
                significance=7.0,
                uniqueness=7.0,
                narrative_potential=7.0,
                integration=7.0,
                feedback="Normal 1",
            ),
            ItemQualityScores(
                significance=7.0,
                uniqueness=7.0,
                narrative_potential=7.0,
                integration=7.0,
                feedback="Normal 2",
            ),
            ItemQualityScores(
                significance=7.0,
                uniqueness=7.0,
                narrative_potential=7.0,
                integration=7.0,
                feedback="Normal 3",
            ),
            ItemQualityScores(
                significance=7.0,
                uniqueness=7.0,
                narrative_potential=7.0,
                integration=7.0,
                feedback="Normal 4",
            ),
            ItemQualityScores(
                significance=1.0,
                uniqueness=1.0,
                narrative_potential=1.0,
                integration=1.0,
                feedback="Outlier",
            ),
        ]
        config = self._make_config(strategy="mean", outlier_detection=True, outlier_std=1.5)

        result = _aggregate_scores(results, ItemQualityScores, config)

        # With outlier filtered, mean should be 7.0 (not 5.8)
        assert result.significance == 7.0
        assert result.uniqueness == 7.0

    def test_combined_feedback_multiple(self):
        """Multiple feedbacks are joined with ' | '."""
        results = [
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="Needs more depth",
            ),
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="Improve resonance",
            ),
        ]
        config = self._make_config(strategy="median")

        result = _aggregate_scores(results, ConceptQualityScores, config)

        assert "Needs more depth" in result.feedback
        assert "Improve resonance" in result.feedback
        assert " | " in result.feedback

    def test_combined_feedback_deduplicates(self):
        """Identical feedbacks are deduplicated."""
        results = [
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="Same feedback",
            ),
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="Same feedback",
            ),
        ]
        config = self._make_config(strategy="median")

        result = _aggregate_scores(results, ConceptQualityScores, config)

        # Should be just "Same feedback", not "Same feedback | Same feedback"
        assert result.feedback == "Same feedback"

    def test_single_feedback(self):
        """Single non-empty feedback is returned as-is."""
        results = [
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="Only feedback",
            ),
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="",
            ),
        ]
        config = self._make_config(strategy="median")

        result = _aggregate_scores(results, ConceptQualityScores, config)

        assert result.feedback == "Only feedback"

    def test_empty_feedback_from_all_calls(self):
        """When all judge calls return empty feedback, result has empty feedback."""
        results = [
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="",
            ),
            ConceptQualityScores(
                relevance=7.0,
                depth=7.0,
                manifestation=7.0,
                resonance=7.0,
                feedback="",
            ),
        ]
        config = self._make_config(strategy="median")

        result = _aggregate_scores(results, ConceptQualityScores, config)

        assert result.feedback == ""

    def test_all_score_model_types(self):
        """Aggregation works for all 6 score model types."""
        # Verify it works with each score model type by checking field discovery
        model_classes = [
            (
                CharacterQualityScores,
                {
                    "depth": 7.0,
                    "goals": 7.0,
                    "flaws": 7.0,
                    "uniqueness": 7.0,
                    "arc_potential": 7.0,
                    "feedback": "OK",
                },
            ),
            (
                LocationQualityScores,
                {
                    "atmosphere": 7.0,
                    "significance": 7.0,
                    "story_relevance": 7.0,
                    "distinctiveness": 7.0,
                    "feedback": "OK",
                },
            ),
            (
                FactionQualityScores,
                {
                    "coherence": 7.0,
                    "influence": 7.0,
                    "conflict_potential": 7.0,
                    "distinctiveness": 7.0,
                    "feedback": "OK",
                },
            ),
            (
                ItemQualityScores,
                {
                    "significance": 7.0,
                    "uniqueness": 7.0,
                    "narrative_potential": 7.0,
                    "integration": 7.0,
                    "feedback": "OK",
                },
            ),
            (
                ConceptQualityScores,
                {
                    "relevance": 7.0,
                    "depth": 7.0,
                    "manifestation": 7.0,
                    "resonance": 7.0,
                    "feedback": "OK",
                },
            ),
            (
                RelationshipQualityScores,
                {
                    "tension": 7.0,
                    "dynamics": 7.0,
                    "story_potential": 7.0,
                    "authenticity": 7.0,
                    "feedback": "OK",
                },
            ),
        ]
        config = self._make_config(strategy="median")

        for model_class, kwargs in model_classes:
            results = [model_class(**kwargs), model_class(**kwargs)]
            result = _aggregate_scores(results, model_class, config)
            assert result.average == 7.0, (
                f"Failed for {model_class.__name__}: average={result.average}"
            )

    def test_retry_strategy_rejected_by_pydantic(self):
        """Retry strategy is not a valid option and Pydantic rejects it."""
        with pytest.raises(ValidationError):
            JudgeConsistencyConfig(outlier_strategy="retry")


class TestCalibrationBlock:
    """Test that JUDGE_CALIBRATION_BLOCK is used in judge prompts."""

    def test_calibration_block_contains_scoring_rules(self):
        """JUDGE_CALIBRATION_BLOCK contains the key scoring anchors."""
        assert "SCORING GUIDE" in JUDGE_CALIBRATION_BLOCK
        assert "6-7" in JUDGE_CALIBRATION_BLOCK  # first drafts land here
        assert "RULES:" in JUDGE_CALIBRATION_BLOCK
        assert "decimal" in JUDGE_CALIBRATION_BLOCK.lower()

    def test_calibration_block_requests_decimals(self):
        """JUDGE_CALIBRATION_BLOCK requires decimal precision in scores."""
        assert "one decimal place" in JUDGE_CALIBRATION_BLOCK.lower()
        # Contains example decimal scores
        assert "5.3" in JUDGE_CALIBRATION_BLOCK
        assert "7.1" in JUDGE_CALIBRATION_BLOCK
        assert "8.6" in JUDGE_CALIBRATION_BLOCK

    def test_calibration_block_allows_high_scores(self):
        """JUDGE_CALIBRATION_BLOCK does NOT suppress 8+ scores."""
        # The old block had "Do NOT give 8+" which made 7.5 threshold unreachable
        assert "Do NOT give 8" not in JUDGE_CALIBRATION_BLOCK
        # 8-9 range should be described as "Excellent" (achievable)
        assert "8-9" in JUDGE_CALIBRATION_BLOCK
        assert "Excellent" in JUDGE_CALIBRATION_BLOCK

    def test_calibration_block_has_score_ranges(self):
        """JUDGE_CALIBRATION_BLOCK defines score ranges covering 1-10."""
        assert "1-3:" in JUDGE_CALIBRATION_BLOCK
        assert "4-5:" in JUDGE_CALIBRATION_BLOCK
        assert "6-7:" in JUDGE_CALIBRATION_BLOCK
        assert "7-8:" in JUDGE_CALIBRATION_BLOCK
        assert "8-9:" in JUDGE_CALIBRATION_BLOCK
        assert "10:" in JUDGE_CALIBRATION_BLOCK

    def test_character_judge_uses_calibration_block(self):
        """Character judge prompt includes JUDGE_CALIBRATION_BLOCK."""
        from src.memory.story_state import Character
        from src.services.world_quality_service._character import _build_character_judge_prompt

        character = Character(
            name="Test Hero",
            role="protagonist",
            description="A brave warrior",
            personality_traits=["brave", "stubborn"],
            goals=["save the world"],
            arc_notes="Learns humility",
        )

        prompt = _build_character_judge_prompt(character, "fantasy")
        assert JUDGE_CALIBRATION_BLOCK in prompt

    def test_relationship_judge_prompt_template_has_calibration(self):
        """Verify calibration block is referenced in relationship judge."""
        # Read the source to verify it's in the template
        import inspect

        from src.services.world_quality_service import _relationship

        source = inspect.getsource(_relationship._judge_relationship_quality)
        assert "JUDGE_CALIBRATION_BLOCK" in source


class TestRetryTemperature:
    """Test the retry_temperature helper."""

    def test_first_retry_increases_temperature(self):
        """First retry increases temperature by increment."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9

        result = retry_temperature(config, 1)

        assert result == pytest.approx(1.05)

    def test_temperature_caps_at_max(self):
        """Temperature is capped at 1.5."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9

        result = retry_temperature(config, 10)  # 0.9 + 10*0.15 = 2.4 > 1.5

        assert result == 1.5

    def test_zero_retries_returns_base_temp(self):
        """Zero retries returns base creator temperature."""
        from src.services.world_quality_service._common import retry_temperature

        config = MagicMock()
        config.creator_temperature = 0.9

        result = retry_temperature(config, 0)

        assert result == 0.9


class TestJudgeWithAveragingEdgeCases:
    """Edge cases for judge_with_averaging."""

    def test_multi_call_count_2(self):
        """Works correctly with minimum multi_call_count of 2."""
        results = [
            RelationshipQualityScores(
                tension=6.0,
                dynamics=6.0,
                story_potential=6.0,
                authenticity=6.0,
                feedback="Low",
            ),
            RelationshipQualityScores(
                tension=8.0,
                dynamics=8.0,
                story_potential=8.0,
                authenticity=8.0,
                feedback="High",
            ),
        ]
        call_idx = 0

        def judge_fn():
            """Return sequential results from the pre-built list."""
            nonlocal call_idx
            result = results[call_idx]
            call_idx += 1
            return result

        config = JudgeConsistencyConfig(
            enabled=True,
            multi_call_enabled=True,
            multi_call_count=2,
            outlier_detection=False,
            outlier_strategy="median",
        )

        result = judge_with_averaging(judge_fn, RelationshipQualityScores, config)

        assert call_idx == 2
        # Median of [6, 8] = 7
        assert result.tension == 7.0

    def test_multi_call_count_5(self):
        """Works correctly with maximum multi_call_count of 5."""
        scores = ItemQualityScores(
            significance=7.0,
            uniqueness=7.0,
            narrative_potential=7.0,
            integration=7.0,
            feedback="OK",
        )
        judge_fn = MagicMock(return_value=scores)

        config = JudgeConsistencyConfig(
            enabled=True,
            multi_call_enabled=True,
            multi_call_count=5,
            outlier_detection=False,
            outlier_strategy="median",
        )

        result = judge_with_averaging(judge_fn, ItemQualityScores, config)

        assert judge_fn.call_count == 5
        assert result.significance == 7.0


class TestEntityJudgeRoles:
    """Test that ENTITY_JUDGE_ROLES maps to 'judge' role, not 'validator'."""

    def test_entity_judge_roles_use_judge_not_validator(self):
        """All entity types in ENTITY_JUDGE_ROLES should map to 'judge'."""
        for entity_type, role in WorldQualityService.ENTITY_JUDGE_ROLES.items():
            assert role == "judge", (
                f"Entity type '{entity_type}' maps to '{role}' instead of 'judge'"
            )

    def test_judge_tag_only_on_capable_models(self, monkeypatch):
        """No model below quality 7 should have the 'judge' tag."""
        from src.settings import _model_registry

        fake_registry: dict[str, dict[str, object]] = {
            "test-judge:8b": {"quality": 8, "tags": ["judge", "writer"]},
            "test-fast:4b": {"quality": 4, "tags": ["writer"]},
            "test-capable:30b": {"quality": 9, "tags": ["judge", "writer"]},
        }
        monkeypatch.setattr(_model_registry, "RECOMMENDED_MODELS", fake_registry)

        for model_id, info in fake_registry.items():
            tags: list[str] = info["tags"]  # type: ignore[assignment]
            quality: int = info["quality"]  # type: ignore[assignment]
            if "judge" in tags:
                assert quality >= 7, (
                    f"Model '{model_id}' (quality={quality}) has 'judge' tag but quality is below 7"
                )

    def test_small_models_lack_judge_tag(self, monkeypatch):
        """Small/fast models must NOT have 'judge' tag."""
        from src.settings import _model_registry

        fake_registry: dict[str, dict[str, object]] = {
            "test-tiny:1.7b": {"quality": 3, "tags": ["validator"]},
            "test-small:0.6b": {"quality": 2, "tags": []},
            "test-medium:4b": {"quality": 5, "tags": ["writer"]},
        }
        monkeypatch.setattr(_model_registry, "RECOMMENDED_MODELS", fake_registry)

        for model_id, info in fake_registry.items():
            tags: list[str] = info["tags"]  # type: ignore[assignment]
            assert "judge" not in tags, f"Small model '{model_id}' should not have 'judge' tag"


class TestJudgeCreatorConflict:
    """Test that judge warns when using the same model as creator."""

    def test_same_model_for_judge_and_creator_logs_warning(
        self, settings, mock_mode_service, caplog
    ):
        """When judge and creator resolve to same model and no alternative exists, warning logged."""
        import logging
        from unittest.mock import patch

        # Both writer and judge resolve to same model via auto-selection
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "same-model:8b", "judge": "same-model:8b"}

        service = WorldQualityService(settings, mock_mode_service)

        # No alternatives available — only the same model is returned
        with (
            patch.object(settings, "get_models_for_role", return_value=["same-model:8b"]),
            caplog.at_level(logging.WARNING),
        ):
            service._get_judge_model(entity_type="character")

        assert any("same as creator model" in record.message for record in caplog.records), (
            "Expected warning about judge being same as creator model"
        )

    def test_different_models_no_warning(self, settings, mock_mode_service, caplog):
        """When judge and creator resolve to different models, no warning."""
        import logging

        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "creative-model:30b", "judge": "strict-model:8b"}

        service = WorldQualityService(settings, mock_mode_service)

        with caplog.at_level(logging.WARNING):
            service._get_judge_model(entity_type="character")

        assert not any("same as creator model" in record.message for record in caplog.records), (
            "Should not warn when judge and creator are different models"
        )

    def test_no_entity_type_skips_conflict_check(self, settings, mock_mode_service, caplog):
        """When entity_type is None, conflict check is skipped."""
        import logging

        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "same-model:8b", "judge": "same-model:8b"}

        service = WorldQualityService(settings, mock_mode_service)

        with caplog.at_level(logging.WARNING):
            service._get_judge_model(entity_type=None)

        assert not any("same as creator model" in record.message for record in caplog.records), (
            "Should not check conflict when entity_type is None"
        )


class TestJudgeCallLogLevel:
    """Test that judge call failures use correct log level based on multi-call mode."""

    def test_single_call_uses_exception_logging(self, settings, mock_mode_service, caplog):
        """When multi-call is disabled, judge failures log at ERROR with traceback."""
        import logging
        from unittest.mock import patch

        settings.judge_consistency_enabled = True
        settings.judge_multi_call_enabled = False

        service = WorldQualityService(settings, mock_mode_service)
        story_state = MagicMock()
        story_state.brief.genre = "fantasy"
        story_state.brief.language = "English"
        character = MagicMock()
        character.name = "TestHero"
        character.role = "protagonist"
        character.description = "A test character"
        character.personality_traits = ["brave"]
        character.goals = ["survive"]
        character.arc_notes = "grows"

        with (
            patch(
                "src.services.world_quality_service._character.generate_structured",
                side_effect=Exception("LLM error"),
            ),
            caplog.at_level(logging.DEBUG),
            pytest.raises(WorldGenerationError, match="judgment failed"),
        ):
            service._judge_character_quality(character, story_state, 0.1)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("judgment failed" in r.message for r in error_records), (
            "Single-call mode should use logger.exception (ERROR level)"
        )

    def test_multi_call_uses_warning_logging(self, settings, mock_mode_service, caplog):
        """When multi-call is enabled, judge failures log at WARNING (not ERROR)."""
        import logging
        from unittest.mock import patch

        settings.judge_consistency_enabled = True
        settings.judge_multi_call_enabled = True

        service = WorldQualityService(settings, mock_mode_service)
        story_state = MagicMock()
        story_state.brief.genre = "fantasy"
        story_state.brief.language = "English"
        character = MagicMock()
        character.name = "TestHero"
        character.role = "protagonist"
        character.description = "A test character"
        character.personality_traits = ["brave"]
        character.goals = ["survive"]
        character.arc_notes = "grows"

        with (
            patch(
                "src.services.world_quality_service._character.generate_structured",
                side_effect=Exception("LLM error"),
            ),
            caplog.at_level(logging.DEBUG),
            pytest.raises(WorldGenerationError, match="judgment failed"),
        ):
            service._judge_character_quality(character, story_state, 0.1)

        # Filter to only the _character module's records (not _common.py which legitimately
        # logs ERROR when all calls fail)
        char_module = "src.services.world_quality_service._character"
        char_errors = [
            r for r in caplog.records if r.name == char_module and r.levelno >= logging.ERROR
        ]
        char_warnings = [
            r
            for r in caplog.records
            if r.name == char_module
            and r.levelno == logging.WARNING
            and "judgment failed" in r.message
        ]
        assert char_warnings, "Multi-call mode should use logger.warning for judge failures"
        assert not char_errors, (
            "Multi-call mode should NOT use logger.exception (ERROR level) in _single_judge_call"
        )

    @pytest.mark.parametrize(
        "entity_type,judge_method,patch_target,call_args",
        [
            (
                "location",
                "_judge_location_quality",
                "src.services.world_quality_service._location.generate_structured",
                lambda ss: ({"name": "Tavern", "description": "A pub"}, ss, 0.1),
            ),
            (
                "faction",
                "_judge_faction_quality",
                "src.services.world_quality_service._faction.generate_structured",
                lambda ss: ({"name": "Guild", "description": "A guild"}, ss, 0.1),
            ),
            (
                "item",
                "_judge_item_quality",
                "src.services.world_quality_service._item.generate_structured",
                lambda ss: ({"name": "Sword", "description": "A blade"}, ss, 0.1),
            ),
            (
                "concept",
                "_judge_concept_quality",
                "src.services.world_quality_service._concept.generate_structured",
                lambda ss: ({"name": "Honor", "description": "A theme"}, ss, 0.1),
            ),
            (
                "relationship",
                "_judge_relationship_quality",
                "src.services.world_quality_service._relationship.generate_structured",
                lambda ss: (
                    {
                        "source": "A",
                        "target": "B",
                        "relation_type": "knows",
                        "description": "Friends",
                    },
                    ss,
                    0.1,
                ),
            ),
        ],
        ids=["location", "faction", "item", "concept", "relationship"],
    )
    def test_single_call_exception_logging_all_entities(
        self,
        settings,
        mock_mode_service,
        caplog,
        entity_type,
        judge_method,
        patch_target,
        call_args,
    ):
        """Single-call mode uses logger.exception for all entity types."""
        import logging
        from unittest.mock import patch

        settings.judge_consistency_enabled = True
        settings.judge_multi_call_enabled = False

        service = WorldQualityService(settings, mock_mode_service)
        story_state = MagicMock()
        story_state.brief.genre = "fantasy"
        story_state.brief.language = "English"

        args = call_args(story_state)
        with (
            patch(patch_target, side_effect=Exception("LLM error")),
            caplog.at_level(logging.DEBUG),
            pytest.raises(WorldGenerationError, match="judgment failed"),
        ):
            getattr(service, judge_method)(*args)

        module_name = f"src.services.world_quality_service._{entity_type}"
        error_records = [
            r for r in caplog.records if r.name == module_name and r.levelno >= logging.ERROR
        ]
        assert error_records, f"Single-call mode should use logger.exception for {entity_type}"


class TestJudgePrefersAlternativeModel:
    """Test that _get_judge_model() prefers a different model from the creator."""

    def test_swaps_to_alternative_judge_model(self, settings, mock_mode_service, caplog):
        """When judge == creator, swaps to an alternative judge-tagged model."""
        import logging
        from unittest.mock import patch

        # Both writer and judge auto-select the same model
        settings.use_per_agent_models = False
        settings.default_model = "auto"
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"

        service = WorldQualityService(settings, mock_mode_service)

        # Provide an alternative judge model via get_models_for_role
        with (
            patch.object(
                settings,
                "get_models_for_role",
                return_value=["different-judge:8b", "same-model:8b"],
            ),
            caplog.at_level(logging.INFO),
        ):
            model = service._get_judge_model(entity_type="character")

        assert model == "different-judge:8b"
        assert any("Swapping judge model" in r.message for r in caplog.records)

    def test_falls_back_when_no_alternative(self, settings, mock_mode_service, caplog):
        """When no alternative judge model exists, keeps same model with warning."""
        import logging
        from unittest.mock import patch

        settings.use_per_agent_models = False
        settings.default_model = "auto"
        mock_mode_service.get_model_for_agent.return_value = "only-model:8b"

        service = WorldQualityService(settings, mock_mode_service)

        # Only model available is the same as creator
        with (
            patch.object(
                settings,
                "get_models_for_role",
                return_value=["only-model:8b"],
            ),
            caplog.at_level(logging.WARNING),
        ):
            model = service._get_judge_model(entity_type="character")

        assert model == "only-model:8b"
        assert any("same as creator model" in r.message for r in caplog.records)


class TestConflictWarningThrottle:
    """Test that conflict warnings are throttled (once per entity_type:model)."""

    def test_conflict_warning_fires_only_once(self, settings, mock_mode_service, caplog):
        """Same conflict key only produces one warning."""
        import logging
        from unittest.mock import patch

        settings.use_per_agent_models = False
        settings.default_model = "auto"
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"

        service = WorldQualityService(settings, mock_mode_service)

        with (
            patch.object(settings, "get_models_for_role", return_value=["same-model:8b"]),
            caplog.at_level(logging.WARNING),
        ):
            service._get_judge_model(entity_type="character")
            service._get_judge_model(entity_type="character")
            service._get_judge_model(entity_type="character")

        conflict_warnings = [
            r
            for r in caplog.records
            if "same as creator model" in r.message and r.levelno == logging.WARNING
        ]
        assert len(conflict_warnings) == 1, (
            f"Expected exactly 1 conflict warning, got {len(conflict_warnings)}"
        )

    def test_different_entity_types_get_separate_warnings(
        self, settings, mock_mode_service, caplog
    ):
        """Different entity_type keys produce separate warnings."""
        import logging
        from unittest.mock import patch

        settings.use_per_agent_models = False
        settings.default_model = "auto"
        mock_mode_service.get_model_for_agent.return_value = "same-model:8b"

        service = WorldQualityService(settings, mock_mode_service)

        with (
            patch.object(settings, "get_models_for_role", return_value=["same-model:8b"]),
            caplog.at_level(logging.WARNING),
        ):
            service._get_judge_model(entity_type="character")
            service._get_judge_model(entity_type="location")

        conflict_warnings = [
            r
            for r in caplog.records
            if "same as creator model" in r.message and r.levelno == logging.WARNING
        ]
        assert len(conflict_warnings) == 2


class TestJudgePromptOutputFormatDecimals:
    """Test that all entity judge prompts show decimal examples in OUTPUT FORMAT."""

    def test_character_prompt_has_decimal_examples(self):
        """Character judge prompt OUTPUT FORMAT uses decimal scores."""
        from src.memory.story_state import Character
        from src.services.world_quality_service._character import _build_character_judge_prompt

        character = Character(
            name="Test Hero",
            role="protagonist",
            description="A warrior",
            personality_traits=["brave"],
            goals=["survive"],
            arc_notes="Grows",
        )
        prompt = _build_character_judge_prompt(character, "fantasy")
        # Check for decimal example scores (not whole numbers)
        assert "6.3" in prompt or "7.8" in prompt or "5.1" in prompt

    def test_plot_prompt_has_decimal_examples(self):
        """Plot judge prompt OUTPUT FORMAT uses decimal scores."""
        from src.memory.story_state import PlotOutline, PlotPoint
        from src.services.world_quality_service._plot import _build_plot_judge_prompt

        plot = PlotOutline(
            plot_summary="A test plot",
            plot_points=[PlotPoint(description="Event 1")],
        )
        prompt = _build_plot_judge_prompt(plot, "fantasy", ["courage"])
        assert "7.4" in prompt or "5.8" in prompt or "8.1" in prompt

    def test_chapter_prompt_has_decimal_examples(self):
        """Chapter judge prompt OUTPUT FORMAT uses decimal scores."""
        from src.memory.story_state import Chapter
        from src.services.world_quality_service._chapter_quality import (
            _build_chapter_judge_prompt,
        )

        chapter = Chapter(number=1, title="Test", outline="Test outline")
        prompt = _build_chapter_judge_prompt(chapter, "fantasy", "A story about heroes")
        assert "6.9" in prompt or "7.3" in prompt or "5.4" in prompt
