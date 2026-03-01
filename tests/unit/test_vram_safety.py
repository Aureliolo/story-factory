"""Tests for VRAM safety: residency guard, OOM detection, and non-retryable errors."""

import logging
from unittest.mock import ANY, MagicMock, patch

import ollama
import pytest

from src.memory.mode_models import VramStrategy
from src.services.model_mode_service._vram import (
    MIN_GPU_RESIDENCY,
    _last_prepared_model_lock,
    prepare_model,
    unload_all_except,
)
from src.settings import Settings
from src.utils.exceptions import (
    LLMError,
    VRAMAllocationError,
    WorldGenerationError,
)

# ──── Fixtures ────


@pytest.fixture
def mock_svc():
    """Create a mock ModelModeService with common defaults."""
    svc = MagicMock()
    svc.settings = MagicMock(spec=Settings)
    svc.settings.vram_strategy = VramStrategy.ADAPTIVE.value
    svc._loaded_models = set()
    svc._ollama_client = MagicMock()
    return svc


def _reset_prepare_model_cache() -> None:
    """Reset the module-level last-prepared cache between tests."""
    import src.services.model_mode_service._vram as vram_mod

    with _last_prepared_model_lock:
        vram_mod._last_prepared_model_key = None


# ──── Phase 1A: get_available_vram queries memory.free ────


class TestGetAvailableVram:
    """Tests for the nvidia-smi free VRAM query."""

    def test_get_available_vram_queries_memory_free(self):
        """Verify nvidia-smi is called with memory.free, not memory.total."""
        with patch("src.settings._utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="12000\n", returncode=0)

            from src.settings._utils import get_available_vram

            result = get_available_vram()

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "--query-gpu=memory.free" in cmd
            assert "--query-gpu=memory.total" not in cmd
            assert result == 11  # 12000 // 1024


# ──── Phase 1B: VRAMAllocationError hierarchy ────


class TestVRAMAllocationErrorHierarchy:
    """Tests for VRAMAllocationError exception class."""

    def test_vram_allocation_error_is_llm_error(self):
        """VRAMAllocationError should be a subclass of LLMError."""
        assert issubclass(VRAMAllocationError, LLMError)

    def test_vram_allocation_error_attributes(self):
        """VRAMAllocationError should store GPU context attributes."""
        err = VRAMAllocationError(
            "test",
            model_id="big-model:70b",
            model_size_gb=40.0,
            available_vram_gb=20.0,
            residency=0.5,
        )
        assert err.model_id == "big-model:70b"
        assert err.model_size_gb == 40.0
        assert err.available_vram_gb == 20.0
        assert err.residency == 0.5

    def test_vram_allocation_error_catchable_as_llm_error(self):
        """VRAMAllocationError should be catchable via except LLMError."""
        with pytest.raises(LLMError):
            raise VRAMAllocationError("oom", model_id="test:8b")


# ──── Phase 1C: prepare_model residency guard ────


class TestPrepareModelResidencyGuard:
    """Tests for the pre-load residency check in prepare_model()."""

    def setup_method(self):
        """Reset module cache before each test."""
        _reset_prepare_model_cache()

    @patch("src.settings.get_available_vram", return_value=5)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"huge-model:70b": 40.0},
    )
    @patch(
        "src.settings.get_model_info",
        return_value={"vram_required": 48},
    )
    def test_prepare_model_raises_vram_error_when_residency_below_threshold(
        self, _info, _sizes, _vram, mock_svc
    ):
        """prepare_model should raise VRAMAllocationError when residency is below 80%."""
        mock_svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value

        with pytest.raises(VRAMAllocationError) as exc_info:
            prepare_model(mock_svc, "huge-model:70b")

        assert exc_info.value.model_id == "huge-model:70b"
        assert exc_info.value.residency is not None
        assert exc_info.value.residency < MIN_GPU_RESIDENCY

    @patch("src.settings.get_available_vram", return_value=5)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"huge-model:70b": 40.0},
    )
    @patch(
        "src.settings.get_model_info",
        return_value={"vram_required": 48},
    )
    def test_prepare_model_does_not_add_to_loaded_on_vram_rejection(
        self, _info, _sizes, _vram, mock_svc
    ):
        """_loaded_models should not contain the rejected model."""
        mock_svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value

        with pytest.raises(VRAMAllocationError):
            prepare_model(mock_svc, "huge-model:70b")

        assert "huge-model:70b" not in mock_svc._loaded_models

    @patch("src.settings.get_available_vram", return_value=20)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"good-model:8b": 5.0},
    )
    @patch(
        "src.settings.get_model_info",
        return_value={"vram_required": 6},
    )
    def test_prepare_model_allows_load_when_residency_ok(self, _info, _sizes, _vram, mock_svc):
        """prepare_model should succeed and add to _loaded_models when residency is ok."""
        mock_svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value

        prepare_model(mock_svc, "good-model:8b")

        assert "good-model:8b" in mock_svc._loaded_models


# ──── Phase 1C: unload cache invalidation fix ────


class TestUnloadCacheInvalidation:
    """Tests for unconditional cache clear in unload_all_except."""

    def setup_method(self):
        """Reset module cache before each test."""
        _reset_prepare_model_cache()

    def test_unload_clears_cache_unconditionally(self, mock_svc):
        """Cache should be cleared even when the kept model differs from cached model."""
        import src.services.model_mode_service._vram as vram_mod

        # Simulate: model-A was prepared, now we unload model-B keeping model-C
        mock_svc._loaded_models = {"model-a", "model-b"}
        with _last_prepared_model_lock:
            vram_mod._last_prepared_model_key = ("model-a", "SEQUENTIAL")

        unload_all_except(mock_svc, "model-c")

        with _last_prepared_model_lock:
            assert vram_mod._last_prepared_model_key is None


# ──── Phase 1D: warmup calls prepare_model ────


class TestWarmupRouting:
    """Tests for warmup routing through prepare_model."""

    @patch("src.services.world_service._warmup.get_ollama_client")
    @patch("src.services.world_service._warmup.prepare_model")
    def test_warmup_calls_prepare_model(self, mock_prepare, mock_client):
        """_warm_models should call prepare_model before client.chat."""
        from src.services.world_service._warmup import _warm_models

        mock_services = MagicMock()
        mock_services.world_quality._get_creator_model.return_value = "creator:8b"
        mock_services.world_quality._get_judge_model.return_value = "judge:8b"
        mock_services.world_quality.settings = MagicMock(spec=Settings)

        _warm_models(mock_services)

        assert mock_prepare.call_count == 2
        mock_prepare.assert_any_call(ANY, "creator:8b")
        mock_prepare.assert_any_call(ANY, "judge:8b")

    @patch("src.services.world_service._warmup.get_ollama_client")
    @patch(
        "src.services.world_service._warmup.prepare_model",
        side_effect=VRAMAllocationError("oom", model_id="big:70b"),
    )
    def test_warmup_continues_on_vram_error(self, mock_prepare, mock_client):
        """Warmup should continue (non-fatal) when prepare_model raises VRAMAllocationError."""
        from src.services.world_service._warmup import _warm_models

        mock_services = MagicMock()
        mock_services.world_quality._get_creator_model.return_value = "big:70b"
        mock_services.world_quality._get_judge_model.return_value = "big:70b"
        mock_services.world_quality.settings = MagicMock(spec=Settings)

        # Should not raise
        _warm_models(mock_services)


# ──── Phase 1E: OOM detection in llm_client ────


class TestOOMDetection:
    """Tests for OOM detection in generate_structured."""

    @patch("src.services.llm_client.validate_context_size", return_value=4096)
    @patch("src.services.llm_client.get_ollama_client")
    def test_oom_response_raises_vram_allocation_error(self, mock_get_client, mock_ctx):
        """Ollama OOM response should raise VRAMAllocationError, not generic LLMError."""
        from pydantic import BaseModel

        from src.services.llm_client import generate_structured

        class DummyModel(BaseModel):
            """Minimal Pydantic model for structured generation test."""

            value: str = ""

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = ollama.ResponseError(
            "CUDA out of memory. Tried to allocate..."
        )

        settings = MagicMock(spec=Settings)
        settings.context_size = 4096

        with pytest.raises(VRAMAllocationError):
            generate_structured(
                settings=settings,
                model="test-model:8b",
                prompt="test",
                response_model=DummyModel,
            )

    @patch("src.services.llm_client.validate_context_size", return_value=4096)
    @patch("src.services.llm_client.get_ollama_client")
    def test_non_oom_response_raises_llm_error(self, mock_get_client, mock_ctx):
        """Non-OOM Ollama errors should still raise generic LLMError."""
        from pydantic import BaseModel

        from src.services.llm_client import generate_structured

        class DummyModel(BaseModel):
            """Minimal Pydantic model for structured generation test."""

            value: str = ""

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = ollama.ResponseError("model not found")

        settings = MagicMock(spec=Settings)
        settings.context_size = 4096

        with pytest.raises(LLMError) as exc_info:
            generate_structured(
                settings=settings,
                model="test-model:8b",
                prompt="test",
                response_model=DummyModel,
            )
        # Should be LLMError but NOT VRAMAllocationError
        assert not isinstance(exc_info.value, VRAMAllocationError)


# ──── Phase 1F: quality loop does not retry VRAM errors ────


class TestQualityLoopVRAMHandling:
    """Tests for non-retryable VRAMAllocationError in quality loop."""

    def test_quality_loop_does_not_retry_vram_errors(self):
        """quality_refinement_loop should re-raise VRAMAllocationError immediately."""
        from src.memory.world_quality import BaseQualityScores, RefinementConfig
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        # Create a create_fn that wraps VRAMAllocationError in WorldGenerationError
        # (as entity modules do when catching Exception)
        def create_fn(retries: int) -> dict:
            """Simulate entity creation that fails with VRAMAllocationError."""
            cause = VRAMAllocationError("oom", model_id="test:8b")
            raise WorldGenerationError("generation failed") from cause

        def judge_fn(_entity: object) -> BaseQualityScores:
            """Never reached — create_fn always raises first."""
            raise AssertionError("judge_fn should never be called")  # pragma: no cover

        config = MagicMock(spec=RefinementConfig)
        config.max_iterations = 3
        config.quality_threshold = 7.0
        config.quality_thresholds = {"test_entity": 7.0}
        config.get_threshold.return_value = 7.0
        config.dimension_minimum = 0.0

        svc = MagicMock()

        with pytest.raises(WorldGenerationError) as exc_info:
            quality_refinement_loop(
                entity_type="test_entity",
                create_fn=create_fn,
                judge_fn=judge_fn,
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: "test",
                serialize=lambda e: {},
                is_empty=lambda e: False,
                score_cls=BaseQualityScores,
                config=config,
                svc=svc,
                story_id="test-story",
            )

        # Should have VRAMAllocationError as the cause
        assert isinstance(exc_info.value.__cause__, VRAMAllocationError)


# ──── Phase 1G: _model_fits_in_vram helper ────


class TestModelFitsInVram:
    """Tests for the VRAM-aware judge alternative selection helper."""

    @patch("src.services.world_quality_service._model_resolver.get_available_vram", return_value=20)
    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={"good:8b": 5.0},
    )
    def test_model_fits_when_residency_ok(self, _sizes, _vram):
        """Model should fit when residency is above threshold."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("good:8b") is True

    @patch("src.services.world_quality_service._model_resolver.get_available_vram", return_value=5)
    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={"huge:70b": 40.0},
    )
    def test_model_does_not_fit_when_residency_below_threshold(self, _sizes, _vram):
        """Model should not fit when residency is below threshold."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("huge:70b") is False

    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={},
    )
    def test_unknown_model_assumed_to_fit(self, _sizes):
        """Unknown models (not in installed list) should be assumed to fit."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("unknown:8b") is True

    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={"zero:8b": 0.0},
    )
    def test_zero_size_model_assumed_to_fit(self, _sizes):
        """Models with zero size should be assumed to fit."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("zero:8b") is True

    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        side_effect=RuntimeError("nvidia-smi failed"),
    )
    def test_exception_assumes_fit(self, _sizes):
        """On exception, model should be assumed to fit (optimistic fallback)."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("any:8b") is True

    @patch("src.services.world_quality_service._model_resolver.get_available_vram", return_value=5)
    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={"big-alt:70b": 40.0, "creator:8b": 5.0},
    )
    def test_judge_skips_vram_violating_alternative(self, _sizes, _vram):
        """get_judge_model should skip alternatives that fail VRAM residency check."""
        from src.services.world_quality_service._model_resolver import get_judge_model

        svc = MagicMock()
        svc.settings.use_per_agent_models = False
        svc.settings.default_model = "creator:8b"
        svc.ENTITY_CREATOR_ROLES = {"character": "writer"}
        svc.ENTITY_JUDGE_ROLES = {"character": "judge"}
        # get_models_for_role returns alternatives: big-alt (won't fit)
        svc.settings.get_models_for_role.return_value = ["big-alt:70b"]
        svc._model_cache.get_creator_model.return_value = "creator:8b"
        svc._model_cache.store_creator_model.side_effect = lambda r, m: m
        svc._model_cache.get_judge_model.return_value = None
        svc._model_cache.store_judge_model.side_effect = lambda r, m, c: m
        svc._model_cache.has_warned_conflict.return_value = False
        svc._model_cache.mark_conflict_warned = MagicMock()

        # Judge should skip big-alt:70b (fails VRAM) and warn about self-judging
        model = get_judge_model(svc, entity_type="character")
        assert model == "creator:8b"  # Falls back to creator since alt won't fit

    @patch("src.services.world_quality_service._model_resolver.get_available_vram", return_value=20)
    @patch(
        "src.services.world_quality_service._model_resolver.get_installed_models_with_sizes",
        return_value={"boundary:8b": 25.0},  # 20/25 = 0.8 exactly
    )
    def test_model_fits_at_exact_80_percent_boundary(self, _sizes, _vram):
        """Model at exactly 80% residency should pass (uses strict < comparison)."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        assert _model_fits_in_vram("boundary:8b") is True


# ──── Phase 1H: Additional OOM patterns ────


class TestOOMPatternCoverage:
    """Tests for all three OOM detection patterns in generate_structured."""

    @pytest.fixture
    def _dummy_model(self):
        """Provide a minimal Pydantic model for structured generation tests."""
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            """Minimal Pydantic model for structured generation test."""

            value: str = ""

        return DummyModel

    @patch("src.services.llm_client.validate_context_size", return_value=4096)
    @patch("src.services.llm_client.get_ollama_client")
    def test_oom_memory_layout_pattern(self, mock_get_client, mock_ctx, _dummy_model):
        """'memory layout cannot be allocated' should trigger VRAMAllocationError."""
        from src.services.llm_client import generate_structured

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = ollama.ResponseError("memory layout cannot be allocated")

        settings = MagicMock(spec=Settings)
        settings.context_size = 4096

        with pytest.raises(VRAMAllocationError):
            generate_structured(
                settings=settings, model="test:8b", prompt="test", response_model=_dummy_model
            )

    @patch("src.services.llm_client.validate_context_size", return_value=4096)
    @patch("src.services.llm_client.get_ollama_client")
    def test_oom_unable_to_allocate_pattern(self, mock_get_client, mock_ctx, _dummy_model):
        """'unable to allocate' should trigger VRAMAllocationError."""
        from src.services.llm_client import generate_structured

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = ollama.ResponseError("unable to allocate 4GB buffer")

        settings = MagicMock(spec=Settings)
        settings.context_size = 4096

        with pytest.raises(VRAMAllocationError):
            generate_structured(
                settings=settings, model="test:8b", prompt="test", response_model=_dummy_model
            )


# ──── Phase 1I: prepare_model model-not-installed path ────


class TestPrepareModelNotInstalled:
    """Tests for prepare_model when model is not in the installed models list."""

    def setup_method(self):
        """Reset module cache before each test."""
        _reset_prepare_model_cache()

    @patch("src.settings.get_available_vram", return_value=20)
    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={},  # model NOT in installed list
    )
    @patch(
        "src.settings.get_model_info",
        return_value={"vram_required": 6},
    )
    def test_prepare_model_skips_residency_when_not_installed(self, _info, _sizes, _vram, mock_svc):
        """prepare_model should succeed when model is not in installed list (no residency check)."""
        mock_svc.settings.vram_strategy = VramStrategy.SEQUENTIAL.value

        prepare_model(mock_svc, "unknown-model:8b")

        assert "unknown-model:8b" in mock_svc._loaded_models


# ──── Phase 1J: quality loop no-retry assertion ────


class TestQualityLoopNoRetry:
    """Tests that VRAMAllocationError stops the loop after exactly 1 call."""

    def test_create_fn_called_exactly_once_on_vram_error(self):
        """create_fn should be called exactly once when VRAMAllocationError occurs."""
        from src.memory.world_quality import BaseQualityScores, RefinementConfig
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        call_count = 0

        def create_fn(retries: int) -> dict:
            """Track call count and raise VRAMAllocationError."""
            nonlocal call_count
            call_count += 1
            cause = VRAMAllocationError("oom", model_id="test:8b")
            raise WorldGenerationError("generation failed") from cause

        config = MagicMock(spec=RefinementConfig)
        config.max_iterations = 5  # Allow many iterations
        config.get_threshold.return_value = 7.0
        config.dimension_minimum = 0.0

        with pytest.raises(WorldGenerationError):
            quality_refinement_loop(
                entity_type="test_entity",
                create_fn=create_fn,
                judge_fn=lambda e: BaseQualityScores(),  # Never called; create_fn raises first
                refine_fn=lambda e, s, i: e,
                get_name=lambda e: "test",
                serialize=lambda e: {},
                is_empty=lambda e: False,
                score_cls=BaseQualityScores,
                config=config,
                svc=MagicMock(),
                story_id="test-story",
            )

        assert call_count == 1, f"create_fn called {call_count} times, expected 1"


# ──── Phase 1K: warmup mock argument style fix ────


class TestWarmupArgumentMatching:
    """Tests for warmup prepare_model call argument matching."""

    @patch("src.services.world_service._warmup.get_ollama_client")
    @patch("src.services.world_service._warmup.prepare_model")
    def test_warmup_calls_prepare_model_with_correct_args(self, mock_prepare, mock_client):
        """Verify prepare_model is called with correct model IDs using keyword matching."""
        from src.services.world_service._warmup import _warm_models

        mock_services = MagicMock()
        mock_services.world_quality._get_creator_model.return_value = "creator:8b"
        mock_services.world_quality._get_judge_model.return_value = "judge:8b"
        mock_services.world_quality.settings = MagicMock(spec=Settings)

        _warm_models(mock_services)

        assert mock_prepare.call_count == 2
        mock_prepare.assert_any_call(ANY, "creator:8b")
        mock_prepare.assert_any_call(ANY, "judge:8b")


# ──── Phase 1L: build_world VRAMAllocationError propagation ────


class TestBuildWorldVRAMPropagation:
    """Tests for VRAMAllocationError propagation through build_world except blocks."""

    @patch("src.services.world_service._build._warm_models")
    @patch("src.services.world_service._build.validate_type")
    def test_calendar_vram_error_propagates(self, _mock_validate, mock_warmup):
        """VRAMAllocationError during calendar generation should propagate, not be swallowed."""
        from src.services.world_service import WorldBuildOptions
        from src.services.world_service._build import build_world

        mock_svc = MagicMock()  # WorldService
        mock_services = MagicMock()  # ServiceContainer
        mock_services.settings.generate_calendar_on_world_build = True
        mock_services.world_quality.generate_calendar_with_quality.side_effect = (
            VRAMAllocationError("oom", model_id="test:8b")
        )
        mock_services.world_quality.get_config.return_value = MagicMock(
            max_iterations=3,
            quality_threshold=7.0,
            creator_temperature=0.9,
            judge_temperature=0.3,
            early_stopping_patience=1,
            quality_thresholds=None,
        )

        mock_story = MagicMock()
        mock_story.brief = MagicMock()
        mock_story.chapters = []
        mock_story.id = "test-story"

        options = WorldBuildOptions.full()

        with pytest.raises(VRAMAllocationError):
            build_world(mock_svc, mock_story, MagicMock(), mock_services, options)


# ──── Phase 1M: _generate_batch VRAMAllocationError propagation ────


class TestGenerateBatchVRAMPropagation:
    """Tests for VRAMAllocationError propagation through _generate_batch."""

    def test_generate_batch_propagates_vram_error(self):
        """VRAMAllocationError wrapped in WorldGenerationError should propagate from _generate_batch."""
        from src.services.world_quality_service._batch import _generate_batch

        mock_svc = MagicMock()

        def gen_fn(retries: int) -> tuple[dict, MagicMock, int]:
            """Raise WorldGenerationError wrapping VRAMAllocationError."""
            raise WorldGenerationError("gen failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="gen failed"):
            _generate_batch(
                mock_svc,
                count=1,
                entity_type="location",
                generate_fn=gen_fn,
                get_name=lambda e: e.get("name", ""),
            )

    def test_review_batch_propagates_vram_error(self):
        """VRAMAllocationError wrapped in WorldGenerationError should propagate from _review_batch."""
        from src.services.world_quality_service._batch import _review_batch

        mock_svc = MagicMock()

        def review_fn(entity: dict) -> tuple[dict, MagicMock, int]:
            """Raise WorldGenerationError wrapping VRAMAllocationError."""
            raise WorldGenerationError("review failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="review failed"):
            _review_batch(
                mock_svc,
                entities=[{"name": "Test Location"}],
                entity_type="location",
                review_fn=review_fn,
                get_name=lambda e: e.get("name", ""),
                zero_scores_fn=lambda msg: MagicMock(),
            )


# ──── Phase 1N: _generate_batch_parallel VRAMAllocationError propagation ────


class TestBatchParallelVRAMPropagation:
    """Tests for VRAMAllocationError propagation through _generate_batch_parallel."""

    def test_rolling_window_propagates_vram_error(self):
        """VRAMAllocationError in rolling-window path should propagate."""
        from src.services.world_quality_service._batch_parallel import (
            _generate_batch_parallel,
        )

        mock_svc = MagicMock()
        config = MagicMock()
        config.quality_threshold = 7.5
        config.get_threshold = MagicMock(return_value=7.5)
        mock_svc.get_config = MagicMock(return_value=config)

        def gen_fn(retries: int) -> tuple[dict, MagicMock, int]:
            """Raise WorldGenerationError wrapping VRAMAllocationError."""
            raise WorldGenerationError("create failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="create failed"):
            _generate_batch_parallel(
                mock_svc,
                count=1,
                entity_type="location",
                generate_fn=gen_fn,
                get_name=lambda e: e.get("name", ""),
                max_workers=1,
            )

    def _make_phased_svc(self):
        """Create a mock service that enters phased pipeline."""
        mock_svc = MagicMock()
        config = MagicMock()
        config.quality_threshold = 7.5
        config.get_threshold = MagicMock(return_value=7.5)
        mock_svc.get_config = MagicMock(return_value=config)
        # Different creator/judge models → triggers phased pipeline
        mock_svc._get_creator_model = MagicMock(return_value="creator:8b")
        mock_svc._get_judge_model = MagicMock(return_value="judge:8b")
        return mock_svc

    def test_phased_create_propagates_vram_error(self):
        """VRAMAllocationError in phased phase-1 create should propagate."""
        from src.services.world_quality_service._batch_parallel import (
            _generate_batch_parallel,
        )

        mock_svc = self._make_phased_svc()

        def create_only(retries: int):
            """Raise VRAMAllocationError wrapped in WorldGenerationError."""
            raise WorldGenerationError("create failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="create failed"):
            _generate_batch_parallel(
                mock_svc,
                count=1,
                entity_type="location",
                generate_fn=lambda r: ({"name": "X"}, MagicMock(average=8.0), 1),
                get_name=lambda e: e.get("name", ""),
                max_workers=2,
                create_only_fn=create_only,
                judge_only_fn=lambda e: MagicMock(average=8.0),
                is_empty_fn=lambda e: False,
                refine_with_initial_fn=lambda e: (e, MagicMock(average=8.0), 1),
            )

    def test_phased_judge_propagates_vram_error(self):
        """VRAMAllocationError in phased phase-2 judge should propagate."""
        from src.services.world_quality_service._batch_parallel import (
            _generate_batch_parallel,
        )

        mock_svc = self._make_phased_svc()

        def judge_only(e):
            """Raise VRAMAllocationError wrapped in WorldGenerationError."""
            raise WorldGenerationError("judge failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="judge failed"):
            _generate_batch_parallel(
                mock_svc,
                count=1,
                entity_type="location",
                generate_fn=lambda r: ({"name": "X"}, MagicMock(average=8.0), 1),
                get_name=lambda e: e.get("name", ""),
                max_workers=2,
                create_only_fn=lambda r: {"name": "X"},
                judge_only_fn=judge_only,
                is_empty_fn=lambda e: False,
                refine_with_initial_fn=lambda e: (e, MagicMock(average=8.0), 1),
            )

    def test_phased_on_success_propagates_vram_error(self):
        """VRAMAllocationError in phased phase-3 on_success should propagate."""
        from src.memory.world_quality import CharacterQualityScores
        from src.services.world_quality_service._batch_parallel import (
            _generate_batch_parallel,
        )

        mock_svc = self._make_phased_svc()
        mock_svc.get_config().get_threshold.return_value = 7.0

        scores = CharacterQualityScores(
            depth=8.0,
            goals=8.0,
            flaws=8.0,
            uniqueness=8.0,
            arc_potential=8.0,
            temporal_plausibility=8.0,
            feedback="Good",
        )

        def on_success(e):
            """Raise WorldGenerationError wrapping VRAMAllocationError."""
            raise WorldGenerationError("save failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="save failed"):
            _generate_batch_parallel(
                mock_svc,
                count=1,
                entity_type="character",
                generate_fn=lambda r: ({"name": "Hero"}, scores, 1),
                get_name=lambda e: e.get("name", ""),
                on_success=on_success,
                max_workers=2,
                create_only_fn=lambda r: {"name": "Hero"},
                judge_only_fn=lambda e: scores,
                is_empty_fn=lambda e: False,
                refine_with_initial_fn=lambda e: (e, scores, 1),
            )

    def test_phased_refine_propagates_vram_error(self):
        """VRAMAllocationError in phased phase-3b refine should propagate."""
        from src.memory.world_quality import CharacterQualityScores
        from src.services.world_quality_service._batch_parallel import (
            _generate_batch_parallel,
        )

        mock_svc = self._make_phased_svc()
        mock_svc.get_config().get_threshold.return_value = 9.0  # High → always fails

        low_scores = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
            feedback="Needs work",
        )

        def refine_fn(e) -> tuple[dict, CharacterQualityScores, int]:
            """Raise WorldGenerationError wrapping VRAMAllocationError."""
            raise WorldGenerationError("refine failed") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        with pytest.raises(WorldGenerationError, match="refine failed"):
            _generate_batch_parallel(
                mock_svc,
                count=1,
                entity_type="character",
                generate_fn=lambda r: ({"name": "X"}, low_scores, 1),
                get_name=lambda e: e.get("name", ""),
                max_workers=2,
                create_only_fn=lambda r: {"name": "X"},
                judge_only_fn=lambda e: low_scores,
                is_empty_fn=lambda e: False,
                refine_with_initial_fn=refine_fn,
            )


# ──── Phase 1O: quality_refinement_loop hail-mary VRAMAllocationError ────


class TestHailMaryVRAMPropagation:
    """Tests for VRAMAllocationError propagation through hail-mary retry."""

    def test_hail_mary_propagates_vram_error(self):
        """VRAMAllocationError in hail-mary create should propagate."""
        from src.memory.world_quality import BaseQualityScores, RefinementConfig
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        call_count = 0

        def create_fn(retries: int) -> dict:
            """First call succeeds, second (hail-mary) raises VRAM error."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"name": "Test"}
            raise WorldGenerationError("hail-mary oom") from VRAMAllocationError(
                "oom", model_id="test:8b"
            )

        def judge_fn(entity: dict) -> BaseQualityScores:
            """Return failing scores to trigger refinement loop exhaustion."""
            return BaseQualityScores(feedback="bad")

        def refine_fn(entity: dict, scores, iteration: int) -> dict:
            """Return unchanged entity (still fails quality)."""
            return entity

        config = RefinementConfig(
            max_iterations=2,  # >1 required for hail-mary, low to exhaust quickly
            quality_threshold=9.0,  # High threshold — always fails
            quality_thresholds={"test_entity": 9.0},
            dimension_minimum=0.0,
            early_stopping_patience=10,
        )

        with pytest.raises(WorldGenerationError, match="hail-mary oom"):
            quality_refinement_loop(
                entity_type="test_entity",
                create_fn=create_fn,
                judge_fn=judge_fn,
                refine_fn=refine_fn,
                get_name=lambda e: "test",
                serialize=lambda e: {},
                is_empty=lambda e: False,
                score_cls=BaseQualityScores,
                config=config,
                svc=MagicMock(),
                story_id="test-story",
            )


# ──── Phase 1P: _model_fits_in_vram unexpected exception path ────


class TestModelFitsUnexpectedException:
    """Tests for unexpected exception path in _model_fits_in_vram."""

    @patch("src.services.world_quality_service._model_resolver.get_available_vram")
    @patch("src.services.world_quality_service._model_resolver.get_installed_models_with_sizes")
    def test_unexpected_exception_logs_warning_and_returns_true(
        self, mock_installed, mock_vram, caplog
    ):
        """Unexpected exception should log a warning and return True (optimistic)."""
        from src.services.world_quality_service._model_resolver import _model_fits_in_vram

        mock_installed.side_effect = RuntimeError("unexpected GPU driver error")

        with caplog.at_level(logging.WARNING):
            result = _model_fits_in_vram("test-model:8b")

        assert result is True
        assert "Unexpected VRAM check failure" in caplog.text
