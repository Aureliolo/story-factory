"""Tests for VRAM budget planning: pair_fits, plan_vram_budget, snapshot caching."""

import logging
from unittest.mock import patch

import pytest

from src.services.model_mode_service._vram_budget import (
    VRAMBudget,
    VRAMSnapshot,
    get_vram_snapshot,
    invalidate_vram_snapshot,
    pair_fits,
    plan_vram_budget,
)

# ──── Fixtures ────


@pytest.fixture(autouse=True)
def _clear_snapshot_cache():
    """Ensure each test starts and ends with a clean snapshot cache."""
    invalidate_vram_snapshot()
    yield
    invalidate_vram_snapshot()


# ──── pair_fits tests ────


class TestPairFits:
    """Tests for pair_fits() residency checks."""

    def test_both_models_fit_easily(self):
        """Both models well within VRAM should return True."""
        assert pair_fits(8.0, 10.0, 24.0) is True

    def test_creator_too_large(self):
        """Creator exceeding residency threshold should return False."""
        # 30GB creator on 24GB GPU → residency = 24/30 = 0.8, but
        # the individual check is >= 0.8, so 0.8 passes.
        # Use a size that clearly fails: 31GB → 24/31 ≈ 0.774
        assert pair_fits(31.0, 5.0, 24.0) is False

    def test_judge_too_large(self):
        """Judge exceeding residency threshold should return False."""
        assert pair_fits(5.0, 31.0, 24.0) is False

    def test_both_too_large(self):
        """Both models exceeding residency threshold should return False."""
        assert pair_fits(31.0, 31.0, 24.0) is False

    def test_unknown_creator_size_returns_true(self):
        """Zero creator size (unknown) should return True optimistically."""
        assert pair_fits(0.0, 10.0, 24.0) is True

    def test_unknown_judge_size_returns_true(self):
        """Zero judge size (unknown) should return True optimistically."""
        assert pair_fits(10.0, 0.0, 24.0) is True

    def test_zero_available_vram_returns_false(self):
        """Zero available VRAM should return False."""
        assert pair_fits(8.0, 10.0, 0.0) is False

    def test_exactly_at_combined_boundary(self):
        """Combined size exactly equal to available VRAM should pass."""
        # 12 + 12 = 24 == 24 → at boundary, still fits
        assert pair_fits(12.0, 12.0, 24.0) is True

    def test_combined_size_exceeds_vram(self):
        """Combined model size exceeding available VRAM should return False.

        This is the original C1 scenario: 14GB creator + 18GB judge = 32GB on 24GB GPU.
        """
        assert pair_fits(14.0, 18.0, 24.0) is False

    def test_combined_just_over(self):
        """Combined size just over available should fail."""
        assert pair_fits(12.1, 12.0, 24.0) is False

    def test_just_below_residency_threshold(self):
        """Models just below 80% residency should fail."""
        # 24GB VRAM / 30.1GB model ≈ 0.7973 < 0.8
        # Also combined = 35.1 > 24, fails combined check first
        assert pair_fits(30.1, 5.0, 24.0) is False

    def test_both_negative_sizes_return_true(self):
        """Negative sizes (defensive) should return True optimistically."""
        assert pair_fits(-1.0, -2.0, 24.0) is True

    def test_one_negative_one_positive_returns_true(self):
        """One negative size should trigger the optimistic early return."""
        assert pair_fits(-1.0, 10.0, 24.0) is True

    def test_negative_available_vram_returns_false(self):
        """Negative available VRAM should return False (treated like zero)."""
        assert pair_fits(8.0, 10.0, -1.0) is False

    def test_small_models_on_large_gpu(self):
        """Small models on large GPU should easily fit."""
        assert pair_fits(2.0, 3.0, 48.0) is True

    def test_pair_fits_logs_on_failure(self, caplog):
        """pair_fits should log debug message when pair does not fit."""
        with caplog.at_level(logging.DEBUG):
            result = pair_fits(40.0, 40.0, 10.0)

        assert result is False
        assert "Pair does not fit" in caplog.text


# ──── plan_vram_budget tests ────


class TestPlanVramBudget:
    """Tests for plan_vram_budget() allocation planning."""

    def test_normal_pair_that_fits(self):
        """Budget for models that fit should have fits=True and correct residencies."""
        budget = plan_vram_budget(8.0, 10.0, 24.0)

        assert isinstance(budget, VRAMBudget)
        assert budget.fits is True
        assert budget.total_vram_gb == 24.0
        assert budget.creator_size_gb == 8.0
        assert budget.judge_size_gb == 10.0
        assert budget.combined_gb == 18.0
        assert budget.residency_creator == 1.0  # 24/8 > 1 → clamped to 1.0
        assert budget.residency_judge == 1.0  # 24/10 > 1 → clamped to 1.0

    def test_pair_that_does_not_fit(self):
        """Budget for models that do not fit should have fits=False."""
        budget = plan_vram_budget(40.0, 40.0, 10.0)

        assert budget.fits is False
        assert budget.combined_gb == 80.0
        assert budget.residency_creator == pytest.approx(10.0 / 40.0)
        assert budget.residency_judge == pytest.approx(10.0 / 40.0)

    def test_zero_size_models_default_residency(self):
        """Zero-size models should get residency 1.0 (optimistic default)."""
        budget = plan_vram_budget(0.0, 0.0, 24.0)

        assert budget.residency_creator == 1.0
        assert budget.residency_judge == 1.0
        assert budget.fits is True  # pair_fits returns True for unknown sizes
        assert budget.combined_gb == 0.0

    def test_combined_gb_calculation(self):
        """combined_gb should equal creator_size_gb + judge_size_gb."""
        budget = plan_vram_budget(12.5, 7.3, 24.0)

        assert budget.combined_gb == pytest.approx(12.5 + 7.3)

    def test_zero_vram_with_positive_models(self):
        """Zero VRAM with positive model sizes: residency defaults to 1.0, fits=False."""
        budget = plan_vram_budget(8.0, 10.0, 0.0)

        # When available_vram <= 0, residency branch defaults to 1.0
        assert budget.residency_creator == 1.0
        assert budget.residency_judge == 1.0
        assert budget.fits is False  # pair_fits returns False for zero VRAM

    def test_residency_clamped_to_one(self):
        """Residency should never exceed 1.0 even for tiny models."""
        budget = plan_vram_budget(2.0, 3.0, 48.0)

        assert budget.residency_creator == 1.0
        assert budget.residency_judge == 1.0

    def test_partial_residency(self):
        """Models that partially fit should have fractional residency."""
        # 20 GB VRAM, 25 GB model → 20/25 = 0.8
        budget = plan_vram_budget(25.0, 20.0, 20.0)

        assert budget.residency_creator == pytest.approx(0.8)
        assert budget.residency_judge == 1.0  # 20/20 = 1.0

    def test_budget_is_frozen(self):
        """VRAMBudget should be immutable (frozen dataclass)."""
        budget = plan_vram_budget(8.0, 10.0, 24.0)

        with pytest.raises(AttributeError):
            budget.fits = False  # type: ignore[misc]

    def test_plan_vram_budget_logs_details(self, caplog):
        """plan_vram_budget should log budget details at debug level."""
        with caplog.at_level(logging.DEBUG):
            plan_vram_budget(8.0, 10.0, 24.0)

        assert "VRAM budget" in caplog.text


# ──── get_vram_snapshot tests ────


class TestGetVramSnapshot:
    """Tests for get_vram_snapshot() caching behavior."""

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"test-model:8b": 4.5, "test-judge:8b": 5.2},
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=20,
    )
    def test_returns_fresh_snapshot_on_first_call(self, mock_vram, mock_models):
        """First call should query VRAM and models, returning a fresh snapshot."""
        snapshot = get_vram_snapshot()

        assert isinstance(snapshot, VRAMSnapshot)
        assert snapshot.available_vram_gb == 20.0
        assert snapshot.installed_models == {"test-model:8b": 4.5, "test-judge:8b": 5.2}
        assert snapshot.timestamp > 0
        mock_vram.assert_called_once()
        mock_models.assert_called_once()

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"test-model:8b": 4.5},
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=20,
    )
    def test_returns_cached_snapshot_within_ttl(self, mock_vram, mock_models):
        """Second call within TTL should return cached snapshot without re-querying."""
        first = get_vram_snapshot()
        second = get_vram_snapshot()

        assert first is second
        # Only called once — second call used cache
        mock_vram.assert_called_once()
        mock_models.assert_called_once()

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"test-model:8b": 4.5},
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=20,
    )
    @patch("src.services.model_mode_service._vram_budget.time.monotonic")
    def test_refreshes_after_ttl_expires(self, mock_time, mock_vram, mock_models):
        """Snapshot should refresh when TTL has expired."""
        # First call at t=0
        mock_time.return_value = 100.0
        first = get_vram_snapshot()

        # Second call at t=31 (past 30s TTL)
        mock_time.return_value = 131.0
        second = get_vram_snapshot()

        assert first is not second
        assert mock_vram.call_count == 2
        assert mock_models.call_count == 2

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"test-model:8b": 4.5},
    )
    @patch(
        "src.settings.get_available_vram",
        side_effect=ConnectionError("nvidia-smi not found"),
    )
    def test_handles_connection_error_from_get_available_vram(self, mock_vram, mock_models):
        """ConnectionError from get_available_vram should result in 0.0 VRAM."""
        snapshot = get_vram_snapshot()

        assert snapshot.available_vram_gb == 0.0
        assert snapshot.installed_models == {"test-model:8b": 4.5}

    @patch(
        "src.settings.get_installed_models_with_sizes",
        side_effect=OSError("ollama list failed"),
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=20,
    )
    def test_handles_error_from_get_installed_models(self, mock_vram, mock_models):
        """OSError from get_installed_models_with_sizes should result in empty dict."""
        snapshot = get_vram_snapshot()

        assert snapshot.available_vram_gb == 20.0
        assert snapshot.installed_models == {}

    @patch(
        "src.settings.get_installed_models_with_sizes",
        side_effect=ConnectionError("offline"),
    )
    @patch(
        "src.settings.get_available_vram",
        side_effect=FileNotFoundError("nvidia-smi missing"),
    )
    def test_handles_both_errors_gracefully(self, mock_vram, mock_models):
        """Both queries failing should return a snapshot with 0.0 VRAM and empty models."""
        snapshot = get_vram_snapshot()

        assert snapshot.available_vram_gb == 0.0
        assert snapshot.installed_models == {}

    @patch(
        "src.settings.get_installed_models_with_sizes",
        side_effect=ValueError("bad output"),
    )
    @patch(
        "src.settings.get_available_vram",
        side_effect=TimeoutError("nvidia-smi timeout"),
    )
    def test_handles_timeout_and_value_errors(self, mock_vram, mock_models):
        """TimeoutError and ValueError should be caught and handled gracefully."""
        snapshot = get_vram_snapshot()

        assert snapshot.available_vram_gb == 0.0
        assert snapshot.installed_models == {}

    def test_snapshot_is_frozen(self):
        """VRAMSnapshot should be immutable (frozen dataclass)."""
        snapshot = VRAMSnapshot(available_vram_gb=20.0, installed_models={}, timestamp=1.0)

        with pytest.raises(AttributeError):
            snapshot.available_vram_gb = 10.0  # type: ignore[misc]


# ──── invalidate_vram_snapshot tests ────


class TestInvalidateVramSnapshot:
    """Tests for invalidate_vram_snapshot() cache clearing."""

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={"test-model:8b": 4.5},
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=20,
    )
    def test_invalidation_forces_refresh(self, mock_vram, mock_models):
        """After invalidation, next get_vram_snapshot should query fresh data."""
        first = get_vram_snapshot()

        invalidate_vram_snapshot()

        second = get_vram_snapshot()

        # Should have been called twice (once for each fresh query)
        assert mock_vram.call_count == 2
        assert mock_models.call_count == 2
        assert first is not second

    @patch(
        "src.settings.get_installed_models_with_sizes",
        return_value={},
    )
    @patch(
        "src.settings.get_available_vram",
        return_value=0,
    )
    def test_double_invalidation_is_safe(self, mock_vram, mock_models):
        """Calling invalidate twice without get_vram_snapshot should not error."""
        invalidate_vram_snapshot()
        invalidate_vram_snapshot()

        snapshot = get_vram_snapshot()
        assert isinstance(snapshot, VRAMSnapshot)

    def test_invalidation_logs_debug(self, caplog):
        """invalidate_vram_snapshot should log at debug level."""
        with caplog.at_level(logging.DEBUG):
            invalidate_vram_snapshot()

        assert "VRAM snapshot cache invalidated" in caplog.text
