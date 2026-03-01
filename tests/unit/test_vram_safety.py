"""Tests for VRAM safety: residency guard, OOM detection, and non-retryable errors."""

import logging
from unittest.mock import MagicMock, patch

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

logger = logging.getLogger(__name__)


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
        assert mock_prepare.call_args_list[0].args[1] == "creator:8b"
        assert mock_prepare.call_args_list[1].args[1] == "judge:8b"

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
