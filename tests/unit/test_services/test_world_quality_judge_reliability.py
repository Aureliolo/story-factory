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
        settings.agent_models = {"writer": "some-writer-model:8b"}  # No validator entry

        service = WorldQualityService(settings, mock_mode_service)
        with pytest.raises(ValueError, match="Unknown agent role 'validator'"):
            service._resolve_model_for_role("validator")

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
        model = service._get_judge_model(entity_type="character")

        assert model == "global-model:30b"
        mock_mode_service.get_model_for_agent.assert_not_called()

    def test_per_agent_writer_vs_validator_different_models(self, settings, mock_mode_service):
        """Per-agent can set different models for creator vs judge roles."""
        settings.use_per_agent_models = True
        settings.agent_models = {
            "writer": "creative-model:30b",
            "validator": "strict-model:8b",
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

    def test_retry_strategy_falls_back_to_median(self):
        """Retry strategy is not implemented and falls back to median with a warning."""
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
        config = self._make_config(strategy="retry")

        result = _aggregate_scores(results, FactionQualityScores, config)

        # Should use median (fallback from retry) — median of [5, 7, 9] = 7
        assert result.coherence == 7.0
        assert result.influence == 7.0


class TestCalibrationBlock:
    """Test that JUDGE_CALIBRATION_BLOCK is used in judge prompts."""

    def test_calibration_block_contains_scoring_rules(self):
        """JUDGE_CALIBRATION_BLOCK contains the key scoring anchors."""
        assert "SCORING CALIBRATION" in JUDGE_CALIBRATION_BLOCK
        assert "5-6" in JUDGE_CALIBRATION_BLOCK  # first draft average
        assert "CRITICAL RULES" in JUDGE_CALIBRATION_BLOCK
        assert "score inflation" in JUDGE_CALIBRATION_BLOCK.lower()

    def test_calibration_block_has_all_score_levels(self):
        """JUDGE_CALIBRATION_BLOCK defines all score levels 1-10."""
        assert "1-2:" in JUDGE_CALIBRATION_BLOCK
        assert "3-4:" in JUDGE_CALIBRATION_BLOCK
        assert "5:" in JUDGE_CALIBRATION_BLOCK
        assert "6:" in JUDGE_CALIBRATION_BLOCK
        assert "7:" in JUDGE_CALIBRATION_BLOCK
        assert "8:" in JUDGE_CALIBRATION_BLOCK
        assert "9:" in JUDGE_CALIBRATION_BLOCK
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
