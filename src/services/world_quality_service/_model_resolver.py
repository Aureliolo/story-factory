"""Model resolution logic for WorldQualityService.

Handles the resolution of models for creator and judge roles, respecting
the Settings hierarchy and avoiding self-judging bias.
"""

import logging
from typing import TYPE_CHECKING

from src.services.model_mode_service._vram import MIN_GPU_RESIDENCY
from src.settings import get_available_vram, get_installed_models_with_sizes

if TYPE_CHECKING:
    from src.services.world_quality_service import WorldQualityService

logger = logging.getLogger(__name__)


def _model_fits_in_vram(model_id: str) -> bool:
    """Check whether a model meets the GPU residency threshold.

    Returns True if the model is not in the installed list (unknown size —
    optimistic) or if its residency ratio meets ``MIN_GPU_RESIDENCY``.
    """
    try:
        installed = get_installed_models_with_sizes()
        if model_id not in installed:
            return True  # Unknown size — let Ollama decide
        model_size = installed[model_id]
        if model_size <= 0:
            return True
        available = get_available_vram()
        residency = min(available / model_size, 1.0)
        if residency < MIN_GPU_RESIDENCY:
            logger.debug(
                "Model %s fails VRAM residency check (%.0f%% < %.0f%%)",
                model_id,
                residency * 100,
                MIN_GPU_RESIDENCY * 100,
            )
            return False
        return True
    except Exception as e:
        logger.debug("VRAM check failed for %s, assuming fits: %s", model_id, e)
        return True


def resolve_model_for_role(service: WorldQualityService, agent_role: str) -> str:
    """
    Determine the model ID to use for a given agent role following settings precedence.

    Resolves in this order:
    1) If settings.use_per_agent_models is False and settings.default_model != "auto", return settings.default_model.
    2) If settings.use_per_agent_models is True and settings.agent_models[agent_role] is set to a value other than "auto", return that explicit model.
    3) Otherwise, delegate to the mode service's auto-selection.

    Args:
        service: WorldQualityService instance whose settings and mode_service are consulted.
        agent_role: Agent role to resolve (e.g., "writer", "judge", "architect").

    Returns:
        The resolved model ID for the specified agent_role.

    """
    # 1. Per-agent disabled + explicit default → use default for everything
    if not service.settings.use_per_agent_models:
        if service.settings.default_model != "auto":
            logger.debug(
                "Using default_model '%s' for role '%s' (use_per_agent_models=False)",
                service.settings.default_model,
                agent_role,
            )
            return service.settings.default_model

    # 2. Per-agent enabled + explicit model for this role → use it
    if service.settings.use_per_agent_models:
        if agent_role not in service.settings.agent_models:
            raise ValueError(
                f"Unknown agent role '{agent_role}'. "
                f"Configured roles: {sorted(service.settings.agent_models.keys())}"
            )
        model_setting = service.settings.agent_models[agent_role]
        if model_setting != "auto":
            logger.debug(
                "Using explicit agent model '%s' for role '%s'",
                model_setting,
                agent_role,
            )
            return model_setting

    # 3. Fall through to mode service auto-selection
    model = service.mode_service.get_model_for_agent(agent_role)
    logger.debug(
        "Auto-selected model '%s' for role '%s' via mode service",
        model,
        agent_role,
    )
    return model


def get_creator_model(service: WorldQualityService, entity_type: str | None = None) -> str:
    """
    Determine the model ID to use for creative generation for a given entity type or the default writer role.

    Parameters:
        entity_type (str | None): Optional entity type (e.g., 'character', 'faction'). When provided, it must be a key in `ENTITY_CREATOR_ROLES` and the corresponding agent role will be used.

    Returns:
        str: Model ID to use for generation.
    """
    if entity_type:
        if entity_type not in service.ENTITY_CREATOR_ROLES:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. "
                f"Valid types: {sorted(service.ENTITY_CREATOR_ROLES.keys())}"
            )
        agent_role = service.ENTITY_CREATOR_ROLES[entity_type]
    else:
        agent_role = "writer"

    # Check cache (validates context automatically)
    cached = service._model_cache.get_creator_model(agent_role)
    if cached is not None:
        return cached

    # Resolve and store the model (double-check avoids duplicate log on race)
    model = resolve_model_for_role(service, agent_role)
    stored = service._model_cache.store_creator_model(agent_role, model)
    if stored == model:
        logger.info(
            "Resolved creator model '%s' for entity_type=%s (role=%s)",
            model,
            entity_type,
            agent_role,
        )
    return stored


def get_judge_model(service: WorldQualityService, entity_type: str | None = None) -> str:
    """
    Determine the model ID to use for judging quality for a given entity type or the default judge role.

    If an entity_type is provided it is mapped to a judge agent role via ENTITY_JUDGE_ROLES and validated. The selected model honors Settings precedence (explicit per-agent or default model) and falls back to automatic mode selection when appropriate. To avoid self-judging bias, when an entity_type is provided the function prefers a judge model different from the creator model for the same entity, searching across judge/architect/continuity role tags; if no alternative is available it emits a single throttled warning for that entity_type:model combination. Resolved models are cached per role+creator_model key so entity types with different creators get independent anti-self-judging decisions.

    Parameters:
        entity_type (str | None): Optional entity type to map to a judge agent role; if omitted the generic "judge" role is used.

    Returns:
        str: Model ID to use for judgment.
    """
    if entity_type:
        if entity_type not in service.ENTITY_JUDGE_ROLES:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. "
                f"Valid types: {sorted(service.ENTITY_JUDGE_ROLES.keys())}"
            )
        agent_role = service.ENTITY_JUDGE_ROLES[entity_type]
    else:
        agent_role = "judge"

    # Resolve the creator model first so we can use it in the cache key
    creator_model = get_creator_model(service, entity_type) if entity_type else None

    # Check cache using role+creator key for per-entity-type anti-self-judging
    cached = service._model_cache.get_judge_model(agent_role, creator_model)
    if cached is not None:
        return cached

    # Resolve the model
    model = resolve_model_for_role(service, agent_role)

    # Prefer a different model from the creator to avoid self-judging bias
    if entity_type and creator_model and model == creator_model:
        # Search for alternatives across multiple role tags, not just "judge".
        # Models tagged "architect" or "continuity" also have reasoning capability
        # suitable for quality judgment.
        search_roles = ["judge", "architect", "continuity"]
        alternative_found = False
        for search_role in search_roles:
            alternatives = service.settings.get_models_for_role(search_role)
            if not alternatives:
                logger.debug(
                    "No models found for tag '%s' while searching for judge alternative",
                    search_role,
                )
                continue
            for alt_model in alternatives:
                if alt_model != creator_model:
                    # Skip alternatives that would violate the VRAM residency threshold
                    if not _model_fits_in_vram(alt_model):
                        logger.debug(
                            "Skipping judge alternative '%s' — fails VRAM residency check",
                            alt_model,
                        )
                        continue
                    # Use INFO for non-judge tags so users see when a model is
                    # repurposed from another role as judge.
                    log_fn = logger.debug if search_role == "judge" else logger.info
                    log_fn(
                        "Swapping judge model from '%s' to '%s' for entity_type=%s "
                        "to avoid self-judging bias (creator='%s', found via '%s' tag)",
                        model,
                        alt_model,
                        entity_type,
                        creator_model,
                        search_role,
                    )
                    model = alt_model
                    alternative_found = True
                    break
            if alternative_found:
                break

        if not alternative_found:
            # Throttle warning: only warn once per entity_type:model combination
            conflict_key = f"{entity_type}:{model}"
            if not service._model_cache.has_warned_conflict(conflict_key):
                service._model_cache.mark_conflict_warned(conflict_key)
                logger.warning(
                    "Judge model '%s' is the same as creator model for entity_type=%s "
                    "and no alternative model is available across judge/architect/continuity "
                    "tags. A model judging its own output produces unreliable scores. "
                    "Install a second model or configure a different model for the 'judge' "
                    "role in Settings > Models.",
                    model,
                    entity_type,
                )

    # Store the resolved model keyed by role+creator for entity-type-aware caching
    stored = service._model_cache.store_judge_model(agent_role, model, creator_model)
    if stored == model:
        logger.info(
            "Resolved judge model '%s' for entity_type=%s (role=%s)",
            model,
            entity_type,
            agent_role,
        )
    return stored
