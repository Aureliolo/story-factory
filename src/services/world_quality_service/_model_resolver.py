"""Model resolution logic for WorldQualityService.

Handles the resolution of models for creator and judge roles, respecting
the Settings hierarchy and avoiding self-judging bias.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.world_quality_service import WorldQualityService

logger = logging.getLogger(__name__)


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

    If an entity_type is provided it is mapped to a judge agent role via ENTITY_JUDGE_ROLES and validated. The selected model honors Settings precedence (explicit per-agent or default model) and falls back to automatic mode selection when appropriate. To avoid self-judging bias, when an entity_type is provided the function prefers a judge model different from the creator model for the same entity; if no alternative judge model is available it emits a single throttled warning for that entity_type:model combination. Resolved models are cached per judge role to avoid repeated resolution.

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

    # Check cache (validates context automatically)
    cached = service._model_cache.get_judge_model(agent_role)
    if cached is not None:
        return cached

    # Resolve the model
    model = resolve_model_for_role(service, agent_role)

    # Prefer a different model from the creator to avoid self-judging bias
    if entity_type:
        creator_model = get_creator_model(service, entity_type)
        if model == creator_model:
            # Try to find an alternative judge model
            alternatives = service.settings.get_models_for_role("judge")
            alternative_found = False
            for alt_model in alternatives:
                if alt_model != creator_model:
                    logger.debug(
                        "Swapping judge model from '%s' to '%s' for entity_type=%s "
                        "to avoid self-judging bias (creator model is '%s')",
                        model,
                        alt_model,
                        entity_type,
                        creator_model,
                    )
                    model = alt_model
                    alternative_found = True
                    break

            if not alternative_found:
                # Throttle warning: only warn once per entity_type:model combination
                conflict_key = f"{entity_type}:{model}"
                if not service._model_cache.has_warned_conflict(conflict_key):
                    service._model_cache.mark_conflict_warned(conflict_key)
                    logger.warning(
                        "Judge model '%s' is the same as creator model for entity_type=%s "
                        "and no alternative judge model is available. "
                        "A model judging its own output produces unreliable scores. "
                        "Configure a different model for the 'judge' role in Settings > Models.",
                        model,
                        entity_type,
                    )

    # Store the resolved model (including any swapped alternative)
    stored = service._model_cache.store_judge_model(agent_role, model)
    if stored == model:
        logger.info(
            "Resolved judge model '%s' for entity_type=%s (role=%s)",
            model,
            entity_type,
            agent_role,
        )
    return stored
