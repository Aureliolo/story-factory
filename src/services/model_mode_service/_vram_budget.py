"""VRAM budget planning for pair-aware model selection.

When the world quality loop uses different models for creator and judge roles,
both models must fit together in GPU VRAM. This module plans VRAM allocation
so that model pairs are selected as a unit rather than independently — preventing
the scenario where a 14 GB creator + 18 GB judge = 32 GB on a 24 GB GPU.

The budget planner uses cached model-size data (no subprocess calls) and respects
the MIN_GPU_RESIDENCY threshold from ``_vram.py``.
"""

import logging
import threading
import time
from dataclasses import dataclass

from src.services.model_mode_service._vram import MIN_GPU_RESIDENCY

logger = logging.getLogger(__name__)

# Cache duration for VRAM snapshots (seconds).
# Avoids repeated subprocess calls to nvidia-smi and ``ollama list`` within
# the same quality loop iteration.
_VRAM_SNAPSHOT_TTL = 30.0

# Retry parameters for transient 0 GB VRAM readings during model eviction.
# nvidia-smi may report 0 GB available while Ollama is evicting a model;
# retrying avoids a pessimistic cascade into self-judging mode.
_VRAM_ZERO_RETRY_COUNT = 3
_VRAM_ZERO_RETRY_DELAY_S = 0.5


@dataclass(frozen=True)
class VRAMBudget:
    """Allocation plan for fitting two models in VRAM.

    All sizes are in GB (decimal, matching Ollama conventions).
    """

    total_vram_gb: float
    creator_size_gb: float
    judge_size_gb: float
    combined_gb: float
    fits: bool  # True when both models meet residency threshold together
    residency_creator: float  # Estimated GPU residency fraction for creator
    residency_judge: float  # Estimated GPU residency fraction for judge


@dataclass(frozen=True)
class VRAMSnapshot:
    """Point-in-time snapshot of GPU VRAM and loaded models.

    Cached for ``_VRAM_SNAPSHOT_TTL`` seconds to avoid repeated subprocess calls.
    """

    available_vram_gb: float  # Free VRAM from nvidia-smi (GiB converted to GB)
    installed_models: dict[str, float]  # model_id -> size_gb from ``ollama list``
    timestamp: float


# Module-level snapshot cache
_snapshot_lock = threading.Lock()
_cached_snapshot: VRAMSnapshot | None = None


def get_vram_snapshot() -> VRAMSnapshot:
    """Get a cached VRAM snapshot, refreshing if older than TTL.

    Returns a ``VRAMSnapshot`` with available VRAM and installed model sizes.
    The snapshot is cached for ``_VRAM_SNAPSHOT_TTL`` seconds to avoid the
    cost of 2+ subprocess calls (nvidia-smi + ollama list) per quality loop
    iteration.

    Returns:
        VRAMSnapshot with current VRAM and model data.
    """
    global _cached_snapshot
    now = time.monotonic()

    with _snapshot_lock:
        if _cached_snapshot is not None and (now - _cached_snapshot.timestamp) < _VRAM_SNAPSHOT_TTL:
            logger.debug(
                "VRAM snapshot cache hit (age=%.1fs)",
                now - _cached_snapshot.timestamp,
            )
            return _cached_snapshot

    # Read previous snapshot under lock before starting (used for zero-retry fallback)
    with _snapshot_lock:
        previous_snapshot = _cached_snapshot

    # Build fresh snapshot outside the lock (subprocess calls are slow)
    from src.settings import get_available_vram, get_installed_models_with_sizes

    try:
        available = float(get_available_vram())
    except (ConnectionError, TimeoutError, FileNotFoundError, OSError, ValueError) as e:
        logger.warning("Could not query available VRAM — pair-fit check will be pessimistic: %s", e)
        available = 0.0

    # Retry when nvidia-smi reports 0 GB but a previous snapshot had VRAM.
    # This happens transiently during model eviction.
    if (
        available == 0.0
        and previous_snapshot is not None
        and previous_snapshot.available_vram_gb > 0
    ):
        for attempt in range(1, _VRAM_ZERO_RETRY_COUNT + 1):
            logger.warning(
                "VRAM snapshot returned 0 GB (previous was %.1f GB) — retry %d/%d after %.1fs",
                previous_snapshot.available_vram_gb,
                attempt,
                _VRAM_ZERO_RETRY_COUNT,
                _VRAM_ZERO_RETRY_DELAY_S,
            )
            time.sleep(_VRAM_ZERO_RETRY_DELAY_S)
            try:
                available = float(get_available_vram())
            except ConnectionError, TimeoutError, FileNotFoundError, OSError, ValueError:
                continue
            if available > 0:
                logger.info(
                    "VRAM snapshot recovered to %.1f GB on retry %d",
                    available,
                    attempt,
                )
                break
        else:
            # All retries exhausted — use previous snapshot value with warning
            logger.warning(
                "VRAM snapshot still 0 GB after %d retries — "
                "using previous value %.1f GB to avoid self-judging cascade",
                _VRAM_ZERO_RETRY_COUNT,
                previous_snapshot.available_vram_gb,
            )
            available = previous_snapshot.available_vram_gb

    try:
        installed = get_installed_models_with_sizes()
    except (ConnectionError, TimeoutError, FileNotFoundError, OSError, ValueError) as e:
        logger.warning(
            "Could not query installed models — pair-fit check will use optimistic defaults: %s",
            e,
        )
        installed = {}

    snapshot = VRAMSnapshot(
        available_vram_gb=available,
        installed_models=dict(installed),
        timestamp=time.monotonic(),
    )

    with _snapshot_lock:
        _cached_snapshot = snapshot

    logger.debug(
        "VRAM snapshot refreshed: %.1f GB available, %d models installed",
        snapshot.available_vram_gb,
        len(snapshot.installed_models),
    )
    return snapshot


def invalidate_vram_snapshot() -> None:
    """Force the next ``get_vram_snapshot()`` call to refresh.

    Call after model eviction changes the VRAM landscape.
    """
    global _cached_snapshot
    with _snapshot_lock:
        _cached_snapshot = None
    logger.debug("VRAM snapshot cache invalidated")


def pair_fits(
    creator_size_gb: float,
    judge_size_gb: float,
    available_vram_gb: float,
) -> bool:
    """Check whether both models fit in VRAM with MIN_GPU_RESIDENCY.

    Two checks are applied:

    1. **Combined size**: the sum of both model sizes must not exceed available
       VRAM.  Even in a sequential loading strategy (load creator → run → evict
       → load judge), Ollama may keep both models resident simultaneously, so
       the combined budget is the safe baseline.
    2. **Individual residency**: each model must individually achieve at least
       ``MIN_GPU_RESIDENCY`` fraction GPU-resident.

    Args:
        creator_size_gb: Creator model size in GB.
        judge_size_gb: Judge model size in GB.
        available_vram_gb: Available GPU VRAM in GB.

    Returns:
        True if both models fit with adequate residency.
    """
    if creator_size_gb <= 0 or judge_size_gb <= 0:
        return True  # Unknown sizes — optimistic

    if available_vram_gb <= 0:
        return False

    # Combined check: both models must fit in total VRAM simultaneously.
    combined_gb = creator_size_gb + judge_size_gb
    if combined_gb > available_vram_gb:
        logger.debug(
            "Pair does not fit: combined=%.1fGB (creator=%.1fGB + judge=%.1fGB) > available=%.1fGB",
            combined_gb,
            creator_size_gb,
            judge_size_gb,
            available_vram_gb,
        )
        return False

    # Individual residency check: each model needs MIN_GPU_RESIDENCY.
    creator_residency = min(available_vram_gb / creator_size_gb, 1.0)
    judge_residency = min(available_vram_gb / judge_size_gb, 1.0)

    fits = creator_residency >= MIN_GPU_RESIDENCY and judge_residency >= MIN_GPU_RESIDENCY
    if not fits:  # pragma: no cover — defensive: combined check above catches this
        logger.debug(
            "Pair does not fit: creator=%.1fGB (%.0f%%), judge=%.1fGB (%.0f%%), "
            "available=%.1fGB, min_residency=%.0f%%",
            creator_size_gb,
            creator_residency * 100,
            judge_size_gb,
            judge_residency * 100,
            available_vram_gb,
            MIN_GPU_RESIDENCY * 100,
        )
    return fits


def plan_vram_budget(
    creator_size_gb: float,
    judge_size_gb: float,
    available_vram_gb: float,
) -> VRAMBudget:
    """Plan VRAM allocation for a creator+judge model pair.

    Args:
        creator_size_gb: Creator model size in GB.
        judge_size_gb: Judge model size in GB.
        available_vram_gb: Available GPU VRAM in GB.

    Returns:
        VRAMBudget with allocation details and fit assessment.
    """
    combined = creator_size_gb + judge_size_gb

    if creator_size_gb > 0 and available_vram_gb > 0:
        residency_creator = min(available_vram_gb / creator_size_gb, 1.0)
    else:
        residency_creator = 1.0

    if judge_size_gb > 0 and available_vram_gb > 0:
        residency_judge = min(available_vram_gb / judge_size_gb, 1.0)
    else:
        residency_judge = 1.0

    fits = pair_fits(creator_size_gb, judge_size_gb, available_vram_gb)

    budget = VRAMBudget(
        total_vram_gb=available_vram_gb,
        creator_size_gb=creator_size_gb,
        judge_size_gb=judge_size_gb,
        combined_gb=combined,
        fits=fits,
        residency_creator=residency_creator,
        residency_judge=residency_judge,
    )

    logger.debug(
        "VRAM budget: creator=%.1fGB (%.0f%%), judge=%.1fGB (%.0f%%), "
        "combined=%.1fGB, available=%.1fGB, fits=%s",
        creator_size_gb,
        residency_creator * 100,
        judge_size_gb,
        residency_judge * 100,
        combined,
        available_vram_gb,
        fits,
    )
    return budget
