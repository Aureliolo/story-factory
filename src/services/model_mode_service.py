"""Model mode service - manages generation modes, scoring, and adaptive learning.

This service handles:
- Mode management (presets and custom modes)
- VRAM-aware model loading strategies
- Score recording and aggregation
- Learning/tuning recommendations
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.memory.mode_database import ModeDatabase
from src.memory.mode_models import (
    PRESET_MODES,
    AutonomyLevel,
    GenerationMode,
    LearningSettings,
    LearningTrigger,
    QualityScores,
    RecommendationType,
    SizePreference,
    TuningRecommendation,
    VramStrategy,
    get_preset_mode,
    get_size_tier,
    list_preset_modes,
)
from src.services.llm_client import generate_structured
from src.settings import Settings, get_available_vram
from src.utils.validation import validate_not_empty, validate_not_none, validate_positive

logger = logging.getLogger(__name__)


class ModelModeService:
    """Service for managing generation modes and adaptive learning.

    This service coordinates:
    - Mode selection and customization
    - VRAM-aware model loading/unloading
    - Quality scoring via LLM judge
    - Performance tracking and aggregation
    - Tuning recommendations based on historical data
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Path | str | None = None,
    ):
        """Initialize model mode service.

        Args:
            settings: Application settings.
            db_path: Path to scoring database. Defaults to output/model_scores.db
        """
        logger.debug(f"Initializing ModelModeService: db_path={db_path}")
        self.settings = settings
        # Default to output/model_scores.db at project root
        default_db = Path(__file__).parent.parent.parent / "output" / "model_scores.db"
        self._db_path = Path(db_path) if db_path else default_db
        self._db = ModeDatabase(self._db_path)

        # Current mode
        self._current_mode: GenerationMode | None = None

        # Learning settings
        self._learning_settings = LearningSettings()

        # Track chapters for periodic triggers
        self._chapters_since_analysis = 0

        # Loaded model tracking (for VRAM management)
        self._loaded_models: set[str] = set()
        logger.debug("ModelModeService initialized successfully")

    # === Mode Management ===

    def get_current_mode(self) -> GenerationMode:
        """Get the current generation mode.

        Returns preset 'balanced' if no mode is set.
        """
        if self._current_mode is None:
            self._current_mode = get_preset_mode("balanced") or list_preset_modes()[0]
        return self._current_mode

    def set_mode(self, mode_id: str) -> GenerationMode:
        """Set the current generation mode.

        Args:
            mode_id: ID of preset or custom mode.

        Returns:
            The activated mode.

        Raises:
            ValueError: If mode_id is not found.
        """
        validate_not_empty(mode_id, "mode_id")
        # Check presets first
        mode = get_preset_mode(mode_id)
        if mode:
            self._current_mode = mode
            # Sync mode's VRAM strategy to settings so UI reflects mode default
            self._sync_vram_strategy_to_settings(mode)
            logger.info(f"Activated preset mode: {mode.name}")
            return mode

        # Check custom modes
        custom = self._db.get_custom_mode(mode_id)
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
                description=custom.get("description", ""),
                agent_models=custom["agent_models"],
                agent_temperatures=custom["agent_temperatures"],
                size_preference=size_preference,
                vram_strategy=vram_strategy,
                is_preset=False,
                is_experimental=bool(custom.get("is_experimental", False)),
            )
            self._current_mode = mode
            # Sync mode's VRAM strategy to settings so UI reflects mode default
            self._sync_vram_strategy_to_settings(mode)
            logger.info(f"Activated custom mode: {mode.name}")
            return mode

        raise ValueError(f"Mode not found: {mode_id}")

    def _sync_vram_strategy_to_settings(self, mode: GenerationMode) -> None:
        """Sync the mode's VRAM strategy to settings.

        This ensures the Settings UI reflects the mode's default VRAM strategy,
        while still allowing the user to override it.

        Args:
            mode: The mode whose VRAM strategy should be synced.
        """
        strategy_value = (
            mode.vram_strategy.value
            if isinstance(mode.vram_strategy, VramStrategy)
            else str(mode.vram_strategy)
        )
        if self.settings.vram_strategy != strategy_value:
            logger.debug(f"Syncing VRAM strategy from mode '{mode.name}': {strategy_value}")
            self.settings.vram_strategy = strategy_value

    def list_modes(self) -> list[GenerationMode]:
        """List all available modes (presets + custom)."""
        modes = list_preset_modes()

        # Add custom modes
        for custom in self._db.list_custom_modes():
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
                    description=custom.get("description", ""),
                    agent_models=custom["agent_models"],
                    agent_temperatures=custom["agent_temperatures"],
                    size_preference=size_preference,
                    vram_strategy=VramStrategy(custom["vram_strategy"]),
                    is_preset=False,
                    is_experimental=bool(custom.get("is_experimental", False)),
                )
            )

        return modes

    def save_custom_mode(self, mode: GenerationMode) -> None:
        """Save a custom generation mode."""
        validate_not_none(mode, "mode")
        self._db.save_custom_mode(
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

    def delete_custom_mode(self, mode_id: str) -> bool:
        """Delete a custom mode.

        Returns:
            True if deleted, False if not found.
        """
        validate_not_empty(mode_id, "mode_id")
        # Can't delete presets
        if mode_id in PRESET_MODES:
            logger.warning(f"Cannot delete preset mode: {mode_id}")
            return False

        return self._db.delete_custom_mode(mode_id)

    def get_model_for_agent(self, agent_role: str) -> str:
        """Get the model ID for an agent based on current mode.

        Model selection uses the mode's size_preference to steer toward
        larger or smaller models. The mode's agent_models is only used
        for explicit user overrides.

        Args:
            agent_role: The agent role (writer, architect, etc.)

        Returns:
            Model ID selected based on size preference or from user override.
        """
        validate_not_empty(agent_role, "agent_role")
        mode = self.get_current_mode()
        model_id = mode.agent_models.get(agent_role)

        if model_id:
            # User has explicitly overridden this role
            logger.debug(f"Using user-specified model {model_id} for {agent_role}")
            return model_id

        # Try size-preference-aware selection
        try:
            vram = get_available_vram()
            size_pref = SizePreference(mode.size_preference)
            selected = self._select_model_with_size_preference(agent_role, size_pref, vram)
            logger.info(
                f"Auto-selected {selected} for {agent_role} "
                f"(mode={mode.id}, size_pref={size_pref.value}, vram={vram}GB)"
            )
            return selected
        except ValueError:
            # No tagged models available - fall back to settings
            logger.debug(f"No tagged models for {agent_role}, falling back to settings")
            return self.settings.get_model_for_agent(agent_role)

    def _select_model_with_size_preference(
        self,
        agent_role: str,
        size_pref: SizePreference,
        available_vram: int,
    ) -> str:
        """Select a model based on size preference.

        Args:
            agent_role: The agent role to select for.
            size_pref: Size preference (LARGEST, MEDIUM, SMALLEST).
            available_vram: Available VRAM in GB.

        Returns:
            Selected model ID.

        Raises:
            ValueError: If no tagged models are available.
        """
        from src.settings import RECOMMENDED_MODELS, get_installed_models_with_sizes

        installed_models = get_installed_models_with_sizes()

        if not installed_models:
            # Return a recommended model as default for CI/testing
            default = next(iter(RECOMMENDED_MODELS.keys()))
            logger.warning(f"No models installed - returning default '{default}' for {agent_role}")
            return default

        # Find models tagged for this role
        candidates: list[dict] = []
        for model_id, size_gb in installed_models.items():
            tags = self.settings.get_model_tags(model_id)
            if agent_role in tags:
                # Get quality from RECOMMENDED_MODELS if available
                quality = 5.0
                for rec_id, info in RECOMMENDED_MODELS.items():
                    if model_id == rec_id or model_id.startswith(rec_id.split(":")[0]):
                        quality = info.get("quality", 5.0)
                        break

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
            installed_list = ", ".join(installed_models.keys())
            raise ValueError(
                f"No model tagged for role '{agent_role}'. "
                f"Installed models: [{installed_list}]. "
                f"Configure model tags in Settings > Models tab."
            )

        # Calculate tier score based on size preference
        # Higher score = more preferred
        for c in candidates:
            c["tier_score"] = self._calculate_tier_score(c["size_gb"], size_pref)

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

        return str(best["model_id"])

    def _calculate_tier_score(self, size_gb: float, size_pref: SizePreference) -> float:
        """Calculate a preference score for a model based on size preference.

        Args:
            size_gb: Model size in GB.
            size_pref: The desired size preference.

        Returns:
            Score from 0-10, higher = more preferred.
        """
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

    def get_temperature_for_agent(self, agent_role: str) -> float:
        """Get the temperature for an agent based on current mode.

        Raises:
            ValueError: If agent_role is not configured in agent_temperatures.
        """
        validate_not_empty(agent_role, "agent_role")
        mode = self.get_current_mode()
        temp = mode.agent_temperatures.get(agent_role)
        if temp is not None:
            return float(temp)
        # Fall back to settings (which validates agent role)
        return self.settings.get_temperature_for_agent(agent_role)

    # === VRAM Management ===

    def prepare_model(self, model_id: str) -> None:
        """Prepare a model for use, respecting VRAM strategy.

        For sequential strategy, unloads other models first.
        For parallel, keeps models loaded.
        For adaptive, unloads if VRAM is constrained.

        The VRAM strategy is read from settings.vram_strategy, which can be
        configured by the user in the Settings UI to override the mode default.
        """
        from src.settings import get_installed_models_with_sizes, get_model_info

        validate_not_empty(model_id, "model_id")

        # Use settings.vram_strategy as the source of truth (user-configurable override)
        strategy_str = self.settings.vram_strategy
        try:
            strategy = VramStrategy(strategy_str)
        except ValueError as e:
            valid_options = ", ".join(s.value for s in VramStrategy)
            error_msg = (
                f"Invalid vram_strategy '{strategy_str}' in settings. "
                f"Valid options: {valid_options}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        logger.debug(f"Preparing model {model_id} with VRAM strategy: {strategy.value}")

        if strategy == VramStrategy.SEQUENTIAL:
            # Unload all other models
            self._unload_all_except(model_id)
        elif strategy == VramStrategy.ADAPTIVE:
            # Check if we need to free VRAM
            available = get_available_vram()

            # Get model size from installed models or RECOMMENDED_MODELS
            installed = get_installed_models_with_sizes()
            if model_id in installed:
                size_gb = installed[model_id]
                required = int(size_gb * 1.2)  # 20% overhead
            else:
                model_info = get_model_info(model_id)
                required = model_info["vram_required"]

            if available < required:
                # Need to free up space
                self._unload_all_except(model_id)

        # Model will be loaded on first use by Ollama
        self._loaded_models.add(model_id)

    def _unload_all_except(self, keep_model: str) -> None:
        """Unload all models except the specified one.

        Note: Ollama manages model lifecycle automatically via LRU caching.
        This method only updates our tracking. Actual VRAM freeing depends
        on Ollama's internal memory management, not explicit unload calls.

        For truly sequential model usage, consider using Ollama's --noprune
        flag with manual model loading/unloading via the API if available.
        """
        # For now, just clear our tracking
        # Ollama will unload based on its own LRU cache
        models_to_remove = self._loaded_models - {keep_model}
        if models_to_remove:
            logger.debug(
                f"Marking models for potential unload by Ollama: {models_to_remove} "
                f"(actual unloading depends on Ollama's memory management)"
            )
            self._loaded_models = {keep_model}

    # === Score Recording ===

    def record_generation(
        self,
        project_id: str,
        agent_role: str,
        model_id: str,
        *,
        chapter_id: str | None = None,
        genre: str | None = None,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        prompt_text: str | None = None,
    ) -> int:
        """Record a generation event.

        Returns:
            The score ID for later updates.
        """
        validate_not_empty(project_id, "project_id")
        validate_not_empty(agent_role, "agent_role")
        validate_not_empty(model_id, "model_id")
        mode = self.get_current_mode()

        # Calculate tokens/second
        tokens_per_second = None
        if tokens_generated and time_seconds and time_seconds > 0:
            tokens_per_second = tokens_generated / time_seconds

        # Generate prompt hash for A/B comparisons
        prompt_hash = None
        if prompt_text:
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]

        try:
            score_id = self._db.record_score(
                project_id=project_id,
                agent_role=agent_role,
                model_id=model_id,
                mode_name=mode.id,
                chapter_id=chapter_id,
                genre=genre,
                tokens_generated=tokens_generated,
                time_seconds=time_seconds,
                tokens_per_second=tokens_per_second,
                prompt_hash=prompt_hash,
            )

            speed_display = f"{tokens_per_second:.1f}" if tokens_per_second else "N/A"
            time_display = f"{time_seconds:.1f}" if time_seconds is not None else "N/A"

            logger.info(
                f"Recorded generation score {score_id}: {agent_role}/{model_id} "
                f"(mode={mode.id}, tokens={tokens_generated}, time={time_display}s, "
                f"speed={speed_display} t/s)"
            )
            return score_id

        except Exception as e:
            logger.error(
                f"Failed to record generation for {agent_role}/{model_id}: {e}",
                exc_info=True,
            )
            raise

    def update_quality_scores(
        self,
        score_id: int,
        quality: QualityScores,
    ) -> None:
        """Update a score record with quality scores."""
        validate_positive(score_id, "score_id")
        validate_not_none(quality, "quality")
        try:
            self._db.update_score(
                score_id,
                prose_quality=quality.prose_quality,
                instruction_following=quality.instruction_following,
                consistency_score=quality.consistency_score,
            )
            logger.debug(
                f"Updated quality scores for {score_id}: "
                f"prose={quality.prose_quality}, instruction={quality.instruction_following}, "
                f"consistency={quality.consistency_score}"
            )
        except Exception as e:
            logger.error(f"Failed to update quality scores for {score_id}: {e}", exc_info=True)
            raise

    def record_implicit_signal(
        self,
        score_id: int,
        *,
        was_regenerated: bool | None = None,
        edit_distance: int | None = None,
        user_rating: int | None = None,
    ) -> None:
        """Record an implicit quality signal."""
        validate_positive(score_id, "score_id")
        try:
            self._db.update_score(
                score_id,
                was_regenerated=was_regenerated,
                edit_distance=edit_distance,
                user_rating=user_rating,
            )
            signals = []
            if was_regenerated:
                signals.append("regenerated")
            if edit_distance is not None:
                signals.append(f"edited({edit_distance} chars)")
            if user_rating is not None:
                signals.append(f"rated({user_rating}/5)")

            logger.debug(f"Recorded signals for score {score_id}: {', '.join(signals)}")
        except Exception as e:
            logger.error(f"Failed to record implicit signal for {score_id}: {e}", exc_info=True)
            raise

    def update_performance_metrics(
        self,
        score_id: int,
        *,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        tokens_per_second: float | None = None,
        vram_used_gb: float | None = None,
    ) -> None:
        """Update a score record with performance metrics.

        Args:
            score_id: The score record ID.
            tokens_generated: Number of tokens generated.
            time_seconds: Generation time in seconds.
            tokens_per_second: Generation speed (calculated if not provided).
            vram_used_gb: VRAM used during generation.
        """
        validate_positive(score_id, "score_id")
        # Calculate tokens_per_second if not provided
        if tokens_per_second is None and tokens_generated and time_seconds and time_seconds > 0:
            tokens_per_second = tokens_generated / time_seconds

        try:
            self._db.update_performance_metrics(
                score_id,
                tokens_generated=tokens_generated,
                time_seconds=time_seconds,
                tokens_per_second=tokens_per_second,
                vram_used_gb=vram_used_gb,
            )
            time_display = f"{time_seconds:.1f}" if time_seconds is not None else "N/A"
            speed_display = f"{tokens_per_second:.1f}" if tokens_per_second else "N/A"
            logger.debug(
                f"Updated performance metrics for {score_id}: "
                f"tokens={tokens_generated}, time={time_display}s, "
                f"speed={speed_display} t/s"
            )
        except Exception as e:
            logger.error(f"Failed to update performance metrics for {score_id}: {e}", exc_info=True)
            raise

    # === LLM Quality Judge ===

    def judge_quality(
        self,
        content: str,
        genre: str,
        tone: str,
        themes: list[str],
    ) -> QualityScores:
        """Use LLM to judge content quality.

        Args:
            content: The generated content to evaluate.
            genre: Story genre.
            tone: Story tone.
            themes: Story themes.

        Returns:
            QualityScores with prose_quality and instruction_following.
        """
        validate_not_empty(content, "content")
        validate_not_empty(genre, "genre")
        validate_not_empty(tone, "tone")
        # Use validator model or smallest available
        judge_model = self.get_model_for_agent("validator")
        logger.debug(f"Using {judge_model} to judge quality for {genre}/{tone}")

        # Limit content size for faster judging
        truncated_content = content[: self.settings.content_truncation_for_judgment]

        prompt = f"""You are evaluating the quality of AI-generated story content.

**Story Brief:**
Genre: {genre}
Tone: {tone}
Themes: {", ".join(themes)}

**Content to evaluate:**
{truncated_content}

Rate each dimension from 0-10:

1. prose_quality: Creativity, flow, engagement, vocabulary variety
2. instruction_following: Adherence to genre, tone, themes"""

        try:
            scores = generate_structured(
                settings=self.settings,
                model=judge_model,
                prompt=prompt,
                response_model=QualityScores,
                temperature=self.settings.temp_capability_check,
            )
            logger.info(
                f"Quality judged: prose={scores.prose_quality:.1f}, "
                f"instruction={scores.instruction_following:.1f}"
            )
            return scores
        except Exception as e:
            logger.error(f"Quality judgment failed: {e}", exc_info=True)

        # Return neutral scores on failure
        logger.warning("Returning neutral quality scores (5.0) due to judgment failure")
        return QualityScores(prose_quality=5.0, instruction_following=5.0)

    def calculate_consistency_score(self, issues: list[dict[str, Any]]) -> float:
        """Calculate consistency score from continuity issues.

        Args:
            issues: List of ContinuityIssue-like dicts with 'severity'.

        Returns:
            Score from 0-10 (10 = no issues).
        """
        if not issues:
            return 10.0

        # Weight by severity
        penalty = 0.0
        for issue in issues:
            severity = issue.get("severity", "minor")
            if severity == "critical":
                penalty += 3.0
            elif severity == "moderate":
                penalty += 1.5
            else:  # minor
                penalty += 0.5

        return max(0.0, 10.0 - penalty)

    # === Learning/Tuning ===

    def set_learning_settings(self, settings: LearningSettings) -> None:
        """Update learning settings."""
        self._learning_settings = settings

    def get_learning_settings(self) -> LearningSettings:
        """Get current learning settings."""
        return self._learning_settings

    def should_tune(self) -> bool:
        """Check if tuning analysis should run based on triggers."""
        triggers = self._learning_settings.triggers

        if LearningTrigger.OFF in triggers:
            return False

        if LearningTrigger.CONTINUOUS in triggers:
            return True

        if LearningTrigger.PERIODIC in triggers:
            if self._chapters_since_analysis >= self._learning_settings.periodic_interval:
                return True

        return False

    def on_chapter_complete(self) -> None:
        """Called when a chapter is completed."""
        self._chapters_since_analysis += 1

    def on_project_complete(self) -> list[TuningRecommendation]:
        """Called when a project is completed.

        Returns recommendations if after_project trigger is enabled.
        """
        self._chapters_since_analysis = 0

        if LearningTrigger.AFTER_PROJECT in self._learning_settings.triggers:
            return self.get_recommendations()

        return []

    def get_recommendations(self) -> list[TuningRecommendation]:
        """Generate tuning recommendations based on historical data.

        Returns:
            List of recommendations, may be empty if insufficient data.
        """
        recommendations: list[TuningRecommendation] = []
        min_samples = self._learning_settings.min_samples_for_recommendation

        # Check if we have enough data
        total_scores = self._db.get_score_count()
        if total_scores < min_samples:
            logger.debug(f"Not enough samples for recommendations: {total_scores}")
            return []

        # Get current mode
        mode = self.get_current_mode()

        # Analyze each agent role
        for role in ["writer", "architect", "editor", "continuity"]:
            current_model = mode.agent_models.get(role)
            if not current_model:
                continue

            # Get top performers for this role
            top_models = self._db.get_top_models_for_role(role, limit=3, min_samples=min_samples)

            if not top_models:
                continue

            # Check if there's a better option
            best = top_models[0]
            if best["model_id"] != current_model:
                # Get current model's performance
                current_perf = self._db.get_model_performance(
                    model_id=current_model, agent_role=role
                )

                if current_perf:
                    current_quality = current_perf[0].get("avg_prose_quality", 0) or 0
                    best_quality = best.get("avg_prose_quality", 0) or 0

                    if best_quality > current_quality:
                        improvement = (
                            (best_quality - current_quality) / current_quality * 100
                            if current_quality > 0
                            else 0
                        )
                        confidence = min(
                            0.95,
                            best["sample_count"] / (min_samples * 3),
                        )

                        rec = TuningRecommendation(
                            recommendation_type=RecommendationType.MODEL_SWAP,
                            current_value=current_model,
                            suggested_value=best["model_id"],
                            affected_role=role,
                            reason=(
                                f"{best['model_id']} scores {best_quality:.1f} avg "
                                f"vs {current_quality:.1f} for {role}"
                            ),
                            confidence=confidence,
                            evidence={
                                "current_quality": current_quality,
                                "suggested_quality": best_quality,
                                "sample_count": best["sample_count"],
                            },
                            expected_improvement=f"+{improvement:.0f}% quality",
                        )
                        recommendations.append(rec)

        self._chapters_since_analysis = 0
        return recommendations

    def apply_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """Apply a tuning recommendation.

        Returns:
            True if successfully applied.
        """
        try:
            if recommendation.recommendation_type == "model_swap":
                if recommendation.affected_role and self._current_mode:
                    self._current_mode.agent_models[recommendation.affected_role] = (
                        recommendation.suggested_value
                    )
                    logger.info(
                        f"Applied recommendation: {recommendation.affected_role} "
                        f"now uses {recommendation.suggested_value}"
                    )

                    # Record outcome
                    if recommendation.id:
                        self._db.update_recommendation_outcome(
                            recommendation.id,
                            was_applied=True,
                            user_feedback="accepted",
                        )
                    return True

            elif recommendation.recommendation_type == "temp_adjust":
                if recommendation.affected_role and self._current_mode:
                    self._current_mode.agent_temperatures[recommendation.affected_role] = float(
                        recommendation.suggested_value
                    )
                    logger.info(
                        f"Applied recommendation: {recommendation.affected_role} "
                        f"temperature now {recommendation.suggested_value}"
                    )

                    # Record outcome (same as model_swap to prevent resurface)
                    if recommendation.id:
                        self._db.update_recommendation_outcome(
                            recommendation.id,
                            was_applied=True,
                            user_feedback="accepted",
                        )
                    return True

        except Exception as e:
            logger.error(f"Failed to apply recommendation: {e}")

        return False

    def handle_recommendations(
        self,
        recommendations: list[TuningRecommendation],
    ) -> list[TuningRecommendation]:
        """Handle recommendations based on autonomy level.

        Returns:
            Recommendations that were not auto-applied (need user approval).
        """
        autonomy = self._learning_settings.autonomy
        pending = []

        for rec in recommendations:
            # Save to database
            rec_id = self._db.record_recommendation(
                recommendation_type=rec.recommendation_type.value
                if hasattr(rec.recommendation_type, "value")
                else rec.recommendation_type,
                current_value=rec.current_value,
                suggested_value=rec.suggested_value,
                reason=rec.reason,
                confidence=rec.confidence,
                evidence=rec.evidence,
                affected_role=rec.affected_role,
                expected_improvement=rec.expected_improvement,
            )
            rec.id = rec_id

            # Decide whether to auto-apply
            should_apply = False

            if autonomy == AutonomyLevel.MANUAL:
                should_apply = False
            elif autonomy == AutonomyLevel.CAUTIOUS:
                should_apply = rec.recommendation_type == "temp_adjust"
            elif autonomy == AutonomyLevel.BALANCED:
                should_apply = rec.confidence >= self._learning_settings.confidence_threshold
            elif autonomy in (AutonomyLevel.AGGRESSIVE, AutonomyLevel.EXPERIMENTAL):
                should_apply = True

            if should_apply:
                self.apply_recommendation(rec)
            else:
                pending.append(rec)

        return pending

    # === Analytics ===

    def get_quality_vs_speed_data(
        self,
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get data for quality vs speed scatter plot."""
        return self._db.get_quality_vs_speed_data(agent_role)

    def get_model_performance(
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated model performance."""
        return self._db.get_model_performance(model_id, agent_role)

    def get_recommendation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recommendation history."""
        validate_positive(limit, "limit")
        return self._db.get_recommendation_history(limit)

    def export_scores_csv(self, output_path: Path | str) -> int:
        """Export all scores to CSV."""
        return self._db.export_scores_csv(output_path)

    def get_pending_recommendations(self) -> list[TuningRecommendation]:
        """Get recommendations awaiting user action as TuningRecommendation objects."""
        rows = self._db.get_pending_recommendations(limit=20)
        recommendations = []
        for row in rows:
            try:
                # Parse timestamp from string (SQLite stores as ISO format)
                timestamp_raw = row.get("timestamp")
                if timestamp_raw:
                    timestamp = datetime.fromisoformat(str(timestamp_raw))
                else:
                    timestamp = datetime.now()

                # Parse recommendation_type from string
                rec_type = RecommendationType(str(row.get("recommendation_type")))

                # Parse evidence from JSON (DB column is evidence_json)
                evidence = None
                evidence_json = row.get("evidence_json")
                if evidence_json:
                    try:
                        evidence = json.loads(evidence_json)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse evidence JSON for recommendation {row.get('id')}"
                        )

                rec = TuningRecommendation(
                    id=row.get("id"),
                    timestamp=timestamp,
                    recommendation_type=rec_type,
                    current_value=row.get("current_value", ""),
                    suggested_value=row.get("suggested_value", ""),
                    affected_role=row.get("affected_role"),
                    reason=row.get("reason", ""),
                    confidence=float(row.get("confidence", 0.5)),
                    evidence=evidence,
                    expected_improvement=row.get("expected_improvement"),
                )
                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Failed to parse recommendation {row.get('id')}: {e}")
        return recommendations

    def dismiss_recommendation(self, recommendation: TuningRecommendation) -> None:
        """Dismiss a recommendation so it won't resurface.

        Args:
            recommendation: The recommendation to dismiss.
        """
        if recommendation.id is None:
            logger.warning("Cannot dismiss recommendation without ID")
            return
        self._db.update_recommendation_outcome(
            recommendation_id=recommendation.id,
            was_applied=False,
            user_feedback="ignored",
        )
        logger.debug(f"Dismissed recommendation {recommendation.id}")

    def on_regenerate(self, project_id: str, chapter_id: str) -> None:
        """Record regeneration as a negative implicit signal.

        When a user regenerates a chapter, it indicates dissatisfaction with
        the previous output. This updates the most recent score for that
        chapter to mark it as regenerated.

        Args:
            project_id: The project ID.
            chapter_id: The chapter being regenerated.
        """
        try:
            # Find the most recent score for this project/chapter using efficient LIMIT 1 query
            score = self._db.get_latest_score_for_chapter(project_id, chapter_id)
            if score:
                score_id = score.get("id")
                if score_id:
                    self._db.update_score(score_id, was_regenerated=True)
                    logger.debug(
                        f"Marked score {score_id} as regenerated for "
                        f"project {project_id}, chapter {chapter_id}"
                    )
                    return
            logger.debug(
                f"No score found to mark as regenerated for "
                f"project {project_id}, chapter {chapter_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to record regeneration signal: {e}")
