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
    """Resolve the model for an agent role, respecting Settings hierarchy.

    The resolution order is:
    1. If use_per_agent_models is False and default_model is set → use default_model
    2. If use_per_agent_models is True and agent_models has an explicit model → use it
    3. Otherwise → delegate to ModelModeService auto-selection

    This ensures that Settings-level configuration (UI-visible) takes priority
    over the mode system's automatic selection, which was previously bypassed.

    Args:
        service: The WorldQualityService instance.
        agent_role: The agent role to resolve (writer, validator, etc.).

    Returns:
        Model ID to use for this role.
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
    """Get the model to use for creative generation.

    Respects the Settings hierarchy: explicit default_model or per-agent model
    takes priority over ModelModeService auto-selection.

    Uses resolved model storage to avoid redundant tier score calculations
    when the same role is requested multiple times.

    Args:
        service: The WorldQualityService instance.
        entity_type: Type of entity being created (character, faction, location, etc.).
                    If provided, uses entity-type-specific agent role for model selection.

    Returns:
        Model ID to use for generation.
    """
    agent_role = (
        service.ENTITY_CREATOR_ROLES.get(entity_type, "writer") if entity_type else "writer"
    )

    # Check cache (validates context automatically)
    cached = service._model_cache.get_creator_model(agent_role)
    if cached is not None:
        logger.debug("Using cached creator model '%s' for role=%s", cached, agent_role)
        return cached

    # Resolve and store the model
    model = resolve_model_for_role(service, agent_role)
    service._model_cache.store_creator_model(agent_role, model)
    logger.debug(
        "Resolved and cached creator model '%s' for entity_type=%s (role=%s)",
        model,
        entity_type,
        agent_role,
    )
    return model


def get_judge_model(service: WorldQualityService, entity_type: str | None = None) -> str:
    """Get the model to use for quality judgment.

    Respects the Settings hierarchy: explicit default_model or per-agent model
    takes priority over ModelModeService auto-selection.

    Uses resolved model storage to avoid redundant tier score calculations
    when the same role is requested multiple times.

    If the resolved judge model is the same as the creator model for the
    same entity type, attempts to pick a different model from the available
    judge-tagged models. Falls back to the same model with a throttled warning
    if no alternative is available.

    Args:
        service: The WorldQualityService instance.
        entity_type: Type of entity being judged. If provided, checks that
                    the judge model differs from the creator model.

    Returns:
        Model ID to use for judgment.
    """
    agent_role = service.ENTITY_JUDGE_ROLES.get(entity_type, "judge") if entity_type else "judge"

    # Check cache (validates context automatically)
    cached = service._model_cache.get_judge_model(agent_role)
    if cached is not None:
        logger.debug("Using cached judge model '%s' for role=%s", cached, agent_role)
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
    service._model_cache.store_judge_model(agent_role, model)
    logger.debug(
        "Resolved and cached judge model '%s' for entity_type=%s (role=%s)",
        model,
        entity_type,
        agent_role,
    )
    return model
