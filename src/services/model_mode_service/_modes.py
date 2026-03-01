"""Mode management functions for ModelModeService."""

import logging
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from src.memory.mode_models import (
    PRESET_MODES,
    GenerationMode,
    SizePreference,
    VramStrategy,
    get_preset_mode,
    get_size_tier,
    list_preset_modes,
)
from src.settings._types import check_minimum_quality
from src.utils.validation import validate_not_empty, validate_not_none

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger(__name__)

_excluded_models_lock = threading.Lock()
_excluded_models_logged: set[str] = set()


def set_mode(svc: ModelModeService, mode_id: str) -> GenerationMode:
    """Activate the generation mode identified by `mode_id`.

    Sets it as the current mode, and synchronizes its VRAM strategy
    to application settings.

    Args:
        svc: The ModelModeService instance.
        mode_id: Identifier of a preset or custom generation mode.

    Returns:
        The activated generation mode object.

    Raises:
        ValueError: If no mode with `mode_id` exists or if a custom mode
            contains an invalid VRAM strategy.
    """
    validate_not_empty(mode_id, "mode_id")
    # Check presets first
    mode = get_preset_mode(mode_id)
    if mode:
        svc._current_mode = mode
        # Sync mode's VRAM strategy to settings so UI reflects mode default
        sync_vram_strategy_to_settings(svc, mode)
        logger.info(f"Activated preset mode: {mode.name}")
        return mode

    # Check custom modes
    custom = svc._db.get_custom_mode(mode_id)
    if custom:
        # Validate VRAM strategy - raise exception if invalid
        try:
            vram_strategy = VramStrategy(custom["vram_strategy"])
        except (ValueError, KeyError) as e:
            error_msg = (
                f"Invalid VRAM strategy '{custom.get('vram_strategy')}' in custom mode '{mode_id}'. "
                f"Valid options are: {', '.join([s.value for s in VramStrategy])}. "
                f"Please update the mode configuration."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Parse size preference (required - migration should have backfilled)
        size_pref_str = custom.get("size_preference")
        if size_pref_str is None:
            error_msg = (
                f"Missing size_preference in custom mode '{mode_id}'. "
                "Please run database migration to backfill this field."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            size_preference = SizePreference(size_pref_str)
        except ValueError as e:
            valid_options = ", ".join(s.value for s in SizePreference)
            error_msg = (
                f"Invalid size_preference '{size_pref_str}' in custom mode '{mode_id}'. "
                f"Valid options: {valid_options}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        mode = GenerationMode(
            id=custom["id"],
            name=custom["name"],
            description=custom["description"] or "",
            agent_models=custom["agent_models"],
            agent_temperatures=custom["agent_temperatures"],
            size_preference=size_preference,
            vram_strategy=vram_strategy,
            is_preset=False,
            is_experimental=bool(custom["is_experimental"]),
        )
        svc._current_mode = mode
        # Sync mode's VRAM strategy to settings so UI reflects mode default
        sync_vram_strategy_to_settings(svc, mode)
        logger.info(f"Activated custom mode: {mode.name}")
        return mode

    raise ValueError(f"Mode not found: {mode_id}")


def sync_vram_strategy_to_settings(svc: ModelModeService, mode: GenerationMode) -> None:
    """Sync the mode's VRAM strategy to settings.

    This ensures the Settings UI reflects the mode's default VRAM strategy,
    while still allowing the user to override it.

    Args:
        svc: The ModelModeService instance.
        mode: The mode whose VRAM strategy should be synced.
    """
    strategy_value = (
        mode.vram_strategy.value
        if isinstance(mode.vram_strategy, VramStrategy)
        else str(mode.vram_strategy)
    )
    if svc.settings.vram_strategy != strategy_value:
        logger.debug(f"Syncing VRAM strategy from mode '{mode.name}': {strategy_value}")
        svc.settings.vram_strategy = strategy_value


def list_modes(svc: ModelModeService) -> list[GenerationMode]:
    """Get all available generation modes by combining built-in presets with custom modes.

    Custom modes are converted into GenerationMode objects with strict validation:
    missing or invalid `size_preference` raises ValueError (migration/backfill required).
    Stored `vram_strategy` values are parsed into VramStrategy; invalid values raise.

    Args:
        svc: The ModelModeService instance.

    Returns:
        A list of GenerationMode instances for presets and custom modes.

    Raises:
        ValueError: If a custom mode has missing or invalid size_preference.
    """
    modes = list_preset_modes()

    # Add custom modes
    for custom in svc._db.list_custom_modes():
        mode_id = custom.get("id", "unknown")
        # Parse size preference (required - migration should have backfilled)
        size_pref_str = custom.get("size_preference")
        if size_pref_str is None:
            error_msg = (
                f"Missing size_preference in custom mode '{mode_id}'. "
                "Please run database migration to backfill this field."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            size_preference = SizePreference(size_pref_str)
        except ValueError as e:
            valid_options = ", ".join(s.value for s in SizePreference)
            error_msg = (
                f"Invalid size_preference '{size_pref_str}' in custom mode '{mode_id}'. "
                f"Valid options: {valid_options}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        modes.append(
            GenerationMode(
                id=custom["id"],
                name=custom["name"],
                description=custom["description"] or "",
                agent_models=custom["agent_models"],
                agent_temperatures=custom["agent_temperatures"],
                size_preference=size_preference,
                vram_strategy=VramStrategy(custom["vram_strategy"]),
                is_preset=False,
                is_experimental=bool(custom["is_experimental"]),
            )
        )

    return modes


def save_custom_mode(svc: ModelModeService, mode: GenerationMode) -> None:
    """Persist a custom GenerationMode to the application's mode database.

    Args:
        svc: The ModelModeService instance.
        mode: The custom mode to persist; must not be None.
    """
    validate_not_none(mode, "mode")
    svc._db.save_custom_mode(
        mode_id=mode.id,
        name=mode.name,
        agent_models=mode.agent_models,
        agent_temperatures=mode.agent_temperatures,
        size_preference=mode.size_preference.value
        if isinstance(mode.size_preference, SizePreference)
        else mode.size_preference,
        vram_strategy=mode.vram_strategy.value
        if isinstance(mode.vram_strategy, VramStrategy)
        else mode.vram_strategy,
        description=mode.description,
        is_experimental=mode.is_experimental,
    )
    logger.info(f"Saved custom mode: {mode.name}")


def delete_custom_mode(svc: ModelModeService, mode_id: str) -> bool:
    """Delete a custom mode.

    Args:
        svc: The ModelModeService instance.
        mode_id: The mode ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    validate_not_empty(mode_id, "mode_id")
    # Can't delete presets
    if mode_id in PRESET_MODES:
        logger.warning(f"Cannot delete preset mode: {mode_id}")
        return False

    return svc._db.delete_custom_mode(mode_id)


def get_model_for_agent(svc: ModelModeService, agent_role: str) -> str:
    """Get the model ID for an agent based on current mode.

    Model selection uses the mode's size_preference to steer toward
    larger or smaller models. The mode's agent_models is only used
    for explicit user overrides.

    Args:
        svc: The ModelModeService instance.
        agent_role: The agent role (writer, architect, etc.)

    Returns:
        Model ID selected based on size preference or from user override.
    """
    validate_not_empty(agent_role, "agent_role")
    mode = svc.get_current_mode()
    model_id = mode.agent_models.get(agent_role)

    if model_id:
        # User has explicitly overridden this role
        logger.debug(f"Using user-specified model {model_id} for {agent_role}")
        return model_id

    # Validate size_preference before model selection
    try:
        size_pref = SizePreference(mode.size_preference)
    except ValueError as e:
        valid_options = ", ".join(s.value for s in SizePreference)
        error_msg = (
            f"Invalid size_preference '{mode.size_preference}' in mode '{mode.id}'. "
            f"Valid options: {valid_options}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    # Try size-preference-aware selection
    try:
        from src.settings import get_available_vram

        vram = get_available_vram()
        selected = select_model_with_size_preference(svc, agent_role, size_pref, vram)
        logger.debug(
            f"Auto-selected {selected} for {agent_role} "
            f"(mode={mode.id}, size_pref={size_pref.value}, vram={vram}GB)"
        )
        return selected
    except ValueError:
        # No tagged models available - fall back to settings
        logger.debug(f"No tagged models for {agent_role}, falling back to settings")
        return svc.settings.get_model_for_agent(agent_role)


def select_model_with_size_preference(
    svc: ModelModeService,
    agent_role: str,
    size_pref: SizePreference,
    available_vram: int,
) -> str:
    """
    Select the best installed model ID for an agent role based on size preference and available VRAM.

    Filters installed models by the given agent_role tag, scores candidates by how well their size matches size_pref plus reported quality, prefers models that fit within available_vram, and enforces a minimum quality check before returning.

    Parameters:
        svc: ModelModeService instance used to access settings and helpers.
        agent_role: Agent role tag used to filter models (e.g., "assistant", "summarizer").
        size_pref: Desired model size preference (SizePreference).
        available_vram: Available VRAM in gigabytes used to prefer models that fit.

    Returns:
        The selected model ID as a string.

    Raises:
        ValueError: If no installed model is tagged for the given agent_role.
    """
    from src.services.model_mode_service._vram import MIN_GPU_RESIDENCY
    from src.settings import RECOMMENDED_MODELS, get_installed_models_with_sizes, get_model_info

    installed_models = get_installed_models_with_sizes()

    if not installed_models:
        # Return a recommended model as default for CI/testing
        default = next(iter(RECOMMENDED_MODELS.keys()))
        logger.warning(f"No models installed - returning default '{default}' for {agent_role}")
        return default

    # Find models tagged for this role
    candidates: list[dict] = []
    excluded_by_residency: list[str] = []

    # available_vram is in GiB (from nvidia-smi MiB // 1024).
    # size_gb from ollama list is in decimal GB (1 GB = 1000 MB).
    # Convert to same units: 1 GiB = 1024^3 / 1000^3 ≈ 1.0737 GB.
    available_vram_gb = available_vram * (1024**3 / 1000**3)

    for model_id, size_gb in installed_models.items():
        tags = svc.settings.get_model_tags(model_id)
        if agent_role in tags:
            # 80% GPU residency rule: skip models that can't be at least 80% GPU-resident.
            # Heavy GPU/CPU splitting causes 5-10x inference slowdown.
            if size_gb > 0 and available_vram_gb > 0:
                gpu_residency = available_vram_gb / size_gb
                if gpu_residency < MIN_GPU_RESIDENCY:
                    excluded_by_residency.append(model_id)
                    with _excluded_models_lock:
                        log_fn = (
                            logger.info if model_id not in _excluded_models_logged else logger.debug
                        )
                        _excluded_models_logged.add(model_id)
                    log_fn(
                        "Excluding %s for %s: %.0f%% GPU residency "
                        "(%.1fGB model, %.1fGB GPU). Minimum %.0f%% required.",
                        model_id,
                        agent_role,
                        gpu_residency * 100,
                        size_gb,
                        available_vram_gb,
                        MIN_GPU_RESIDENCY * 100,
                    )
                    continue

            # Get quality from RECOMMENDED_MODELS or estimate from model size
            model_info = get_model_info(model_id)
            quality: float = model_info["quality"]

            estimated_vram = int(size_gb * 1.2)
            fits_vram = estimated_vram <= available_vram
            tier = get_size_tier(size_gb)

            candidates.append(
                {
                    "model_id": model_id,
                    "size_gb": size_gb,
                    "quality": quality,
                    "tier": tier.value,
                    "fits_vram": fits_vram,
                }
            )

    if not candidates:
        if excluded_by_residency:
            excluded_list = ", ".join(excluded_by_residency)
            raise ValueError(
                f"All models tagged for role '{agent_role}' were excluded by GPU residency "
                f"({available_vram_gb:.0f} GB GPU, min {MIN_GPU_RESIDENCY * 100:.0f}%). "
                f"Excluded: [{excluded_list}]."
            )
        installed_list = ", ".join(installed_models.keys())
        raise ValueError(
            f"No model tagged for role '{agent_role}'. "
            f"Installed models: [{installed_list}]. "
            f"Configure model tags in Settings > Models tab."
        )

    # Calculate tier score based on size preference
    # Higher score = more preferred
    for c in candidates:
        c["tier_score"] = calculate_tier_score(c["size_gb"], size_pref)

    # Separate models that fit VRAM from those that don't
    fitting = [c for c in candidates if c["fits_vram"]]
    non_fitting = [c for c in candidates if not c["fits_vram"]]

    # Prefer models that fit VRAM
    pool = fitting if fitting else non_fitting

    # Sort by: tier_score (primary), quality (secondary), size based on preference (tertiary)
    if size_pref == SizePreference.LARGEST:
        # For largest preference: higher tier_score, higher quality, larger size
        pool.sort(key=lambda x: (x["tier_score"], x["quality"], x["size_gb"]), reverse=True)
    elif size_pref == SizePreference.SMALLEST:
        # For smallest preference: higher tier_score, higher quality, smaller size
        pool.sort(key=lambda x: (x["tier_score"], x["quality"], -x["size_gb"]), reverse=True)
    else:  # MEDIUM
        # For medium: higher tier_score, higher quality. Size is not a deciding factor.
        pool.sort(key=lambda x: (x["tier_score"], x["quality"]), reverse=True)

    best = pool[0]
    if not fitting:
        logger.warning(
            f"No model fits VRAM ({available_vram}GB) for {agent_role}. "
            f"Selected {best['model_id']} ({best['size_gb']:.1f}GB) anyway."
        )

    check_minimum_quality(str(best["model_id"]), best["quality"], agent_role)
    return str(best["model_id"])


@lru_cache(maxsize=64)
def calculate_tier_score(size_gb: float, size_pref: SizePreference) -> float:
    """Score how well a model size matches the given SizePreference on a 0-10 scale.

    Cached with ``@lru_cache`` — this pure function is called 28+ times per
    model selection with identical inputs during quality loops.

    Args:
        size_gb: Model size in gigabytes.
        size_pref: Desired size preference enum.

    Returns:
        A score between 0.0 and 10.0 where higher values indicate a closer match.
    """
    logger.debug("calculate_tier_score called: size_gb=%s, size_pref=%s", size_gb, size_pref)
    tier = get_size_tier(size_gb)

    # Define ideal sizes for each preference
    if size_pref == SizePreference.LARGEST:
        # Prefer large > medium > small > tiny
        tier_scores = {"large": 10, "medium": 7, "small": 4, "tiny": 1}
    elif size_pref == SizePreference.SMALLEST:
        # Prefer tiny > small > medium > large
        tier_scores = {"large": 1, "medium": 4, "small": 7, "tiny": 10}
    else:  # MEDIUM
        # Prefer medium > small > large > tiny (balanced)
        tier_scores = {"large": 5, "medium": 10, "small": 7, "tiny": 2}

    return tier_scores.get(tier.value, 5)


def select_model_pair(
    svc: ModelModeService,
    creator_role: str,
    judge_role: str,
) -> tuple[str, str]:
    """Resolve creator and judge models together, ensuring the pair fits in VRAM.

    Instead of resolving each model independently (which can produce a pair that
    exceeds GPU capacity), this function resolves the creator first, then checks
    whether the candidate judge model fits alongside it. If the initial pair
    doesn't fit, it falls back to using the same model for both roles (self-judging)
    to guarantee the pair fits.

    Args:
        svc: The ModelModeService instance.
        creator_role: Agent role for the creator (e.g. "writer", "architect").
        judge_role: Agent role for the judge (e.g. "judge").

    Returns:
        Tuple of (creator_model_id, judge_model_id).
    """
    from src.services.model_mode_service._vram_budget import get_vram_snapshot, pair_fits

    creator_model = get_model_for_agent(svc, creator_role)
    judge_model = get_model_for_agent(svc, judge_role)

    # If same model, no VRAM conflict possible
    if creator_model == judge_model:
        logger.debug(
            "Model pair resolved: same model %s for creator (%s) and judge (%s)",
            creator_model,
            creator_role,
            judge_role,
        )
        return creator_model, judge_model

    # Check if the pair fits in VRAM
    snapshot = get_vram_snapshot()
    creator_size = snapshot.installed_models.get(creator_model, 0.0)
    judge_size = snapshot.installed_models.get(judge_model, 0.0)

    if pair_fits(creator_size, judge_size, snapshot.available_vram_gb):
        logger.info(
            "Model pair resolved: creator=%s (%.1fGB), judge=%s (%.1fGB), "
            "available=%.1fGB — both fit",
            creator_model,
            creator_size,
            judge_model,
            judge_size,
            snapshot.available_vram_gb,
        )
        return creator_model, judge_model

    # Pair doesn't fit — fall back to self-judging with the creator model
    logger.warning(
        "Model pair does not fit in VRAM: creator=%s (%.1fGB) + judge=%s (%.1fGB) "
        "exceeds %.1fGB available. Falling back to self-judging with creator model.",
        creator_model,
        creator_size,
        judge_model,
        judge_size,
        snapshot.available_vram_gb,
    )
    return creator_model, creator_model


def get_temperature_for_agent(svc: ModelModeService, agent_role: str) -> float:
    """Get the temperature for an agent role according to the active generation mode.

    Falls back to application Settings if the mode does not specify a value.

    Args:
        svc: The ModelModeService instance.
        agent_role: The agent role.

    Returns:
        Temperature value for the specified agent role.
    """
    logger.debug("get_temperature_for_agent called: agent_role=%s", agent_role)
    validate_not_empty(agent_role, "agent_role")
    mode = svc.get_current_mode()
    temp = mode.agent_temperatures.get(agent_role)
    if temp is not None:
        return float(temp)
    # Fall back to settings (which validates agent role)
    return svc.settings.get_temperature_for_agent(agent_role)
