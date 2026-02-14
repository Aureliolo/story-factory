#!/usr/bin/env python3
"""Build Performance Profiling — unified investigation for issues #324, #325, #327.

Runs a real entity generation build with full instrumentation to diagnose:
  1. Entity creation timing gaps (#327) — where time goes between LLM completion
     and "creation took" log entries
  2. VRAM swap latency (#324) — cost of creator↔judge model alternation
  3. World health regression (#325) — health metric changes during entity generation

Creates a single instrumented build that exercises VRAM swapping, exposes timing
gaps (semantic duplicate checks, DB writes), and produces health-comparable worlds.

Usage:
    python scripts/investigate_build_performance.py [options]
      --entity-types location,faction,concept  (default: location,faction,concept)
      --count-per-type 2                       (default: 2)
      --world-db output/worlds/existing.db     (optional: existing world for health comparison)
      --skip-vram                              (skip VRAM monitoring)
      --skip-health                            (skip health metric snapshots)
      --output results.json                    (default: output/diagnostics/build_perf_<ts>.json)
      --verbose
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._ollama_helpers import get_model_info
from scripts.evaluate_refinement import make_canonical_brief, make_story_state
from scripts.investigate_vram_usage import (
    get_gpu_vram_usage,
    get_ollama_loaded_models,
    nvidia_smi_available,
)
from src.services import ServiceContainer
from src.settings import Settings

logger = logging.getLogger(__name__)

# Suppress noisy HTTP loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

SUPPORTED_ENTITY_TYPES = ["location", "faction", "item", "concept", "character"]

# Diagnosis thresholds — centralised so they're easy to tune
COLD_START_THRESHOLD_S = 1.5
SEMANTIC_OVERHEAD_THRESHOLD_PCT = 15
UNACCOUNTED_OVERHEAD_THRESHOLD_PCT = 15
VRAM_SWAP_HIGH_THRESHOLD_PCT = 30
VRAM_SWAP_MEDIUM_THRESHOLD_PCT = 10
HEALTH_DROP_THRESHOLD = -5


# =====================================================================
# Timing Collector
# =====================================================================
@dataclass
class TimingEvent:
    """A single timed operation within entity generation."""

    operation: str
    duration_s: float
    model: str = ""
    entity_type: str = ""
    entity_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingCollector:
    """Accumulates per-operation timing events during a build."""

    events: list[TimingEvent] = field(default_factory=list)
    _current_entity_type: str = ""
    _current_entity_name: str = ""

    def set_current_entity(self, entity_type: str, entity_name: str = "") -> None:
        """Set the current entity context for subsequent events."""
        self._current_entity_type = entity_type
        self._current_entity_name = entity_name

    def record(
        self,
        operation: str,
        duration_s: float,
        model: str = "",
        **metadata: Any,
    ) -> None:
        """Record a timing event."""
        event = TimingEvent(
            operation=operation,
            duration_s=round(duration_s, 4),
            model=model,
            entity_type=self._current_entity_type,
            entity_name=self._current_entity_name,
            metadata=metadata,
        )
        self.events.append(event)
        logger.debug(
            "[TIMING] %s: %.4fs (model=%s, entity=%s/%s)",
            operation,
            duration_s,
            model,
            self._current_entity_type,
            self._current_entity_name,
        )

    def get_events_for_entity(self, entity_type: str, entity_name: str) -> list[TimingEvent]:
        """Get all events for a specific entity."""
        return [
            e for e in self.events if e.entity_type == entity_type and e.entity_name == entity_name
        ]

    def get_events_by_operation(self, operation: str) -> list[TimingEvent]:
        """Get all events of a given operation type."""
        return [e for e in self.events if e.operation == operation]


# =====================================================================
# VRAM Monitor
# =====================================================================
@dataclass
class VramSnapshot:
    """A single VRAM state snapshot."""

    timestamp: float
    gpu_vram: dict[str, Any]
    loaded_models: list[dict[str, Any]]
    context: str = ""


@dataclass
class VramTransition:
    """Records a model transition (swap) in VRAM."""

    from_model: str
    to_model: str
    swap_time_s: float
    entity_type: str = ""
    phase: str = ""


@dataclass
class VramMonitor:
    """Monitors GPU VRAM state and detects model transitions."""

    enabled: bool = True
    snapshots: list[VramSnapshot] = field(default_factory=list)
    transitions: list[VramTransition] = field(default_factory=list)
    _last_model: str = ""
    _last_response_time: float = 0.0
    gpu_total_mb: int = 0

    def __post_init__(self) -> None:
        """Detect GPU total VRAM on initialization."""
        if self.enabled and nvidia_smi_available():
            info = get_gpu_vram_usage()
            self.gpu_total_mb = info.get("total_mb", 0)
            logger.info("VramMonitor initialized: GPU total=%dMB", self.gpu_total_mb)
        elif self.enabled:
            logger.info("VramMonitor: nvidia-smi not available, VRAM data will be empty")

    def snapshot(self, context: str = "") -> VramSnapshot:
        """Take a VRAM snapshot now."""
        snap = VramSnapshot(
            timestamp=time.perf_counter(),
            gpu_vram=get_gpu_vram_usage(),
            loaded_models=get_ollama_loaded_models(),
            context=context,
        )
        self.snapshots.append(snap)
        logger.debug(
            "[VRAM] Snapshot: used=%dMB, loaded=%s, context=%s",
            snap.gpu_vram.get("used_mb", 0),
            [m["name"] for m in snap.loaded_models],
            context,
        )
        return snap

    def record_llm_call(
        self,
        model: str,
        call_start: float,
        call_end: float,
        entity_type: str = "",
        phase: str = "",
    ) -> None:
        """Record an LLM call and detect model transitions.

        Args:
            model: Model used for this call.
            call_start: perf_counter timestamp when the call started.
            call_end: perf_counter timestamp when the call finished.
            entity_type: Current entity type being generated.
            phase: Current phase (create, judge, refine).
        """
        if self._last_model and self._last_model != model:
            # Model transition detected — measure gap between previous response and current call start
            swap_time = max(0, call_start - self._last_response_time)
            transition = VramTransition(
                from_model=self._last_model,
                to_model=model,
                swap_time_s=round(swap_time, 3),
                entity_type=entity_type,
                phase=phase,
            )
            self.transitions.append(transition)
            logger.info(
                "[VRAM] Model transition: %s -> %s (%.2fs swap, phase=%s)",
                self._last_model,
                model,
                swap_time,
                phase,
            )

        self._last_model = model
        self._last_response_time = call_end


# =====================================================================
# Monkey-patching instrumentation
# =====================================================================
def install_patches(
    timing: TimingCollector,
    vram: VramMonitor,
) -> dict[tuple[Any, str], Any]:
    """Install timing instrumentation patches on target modules.

    Patches generate_structured and validate_unique_name at their import
    sites (entity modules bind names at import time via ``from ... import``).

    Args:
        timing: TimingCollector to record events to.
        vram: VramMonitor for VRAM snapshots around LLM calls.

    Returns:
        Dict mapping (object, attr_name) -> original function,
        for restoring later.
    """
    import src.services.llm_client as llm_module
    import src.utils.validation as validation_module

    originals: dict[tuple[Any, str], Any] = {}

    # --- Patch generate_structured ---
    original_generate = llm_module.generate_structured

    def instrumented_generate_structured(
        settings,
        model,
        prompt,
        response_model,
        system_prompt=None,
        temperature=0.1,
        max_retries=3,
    ):
        """Wrap generate_structured with timing and VRAM snapshots."""
        # Detect phase from response_model name
        model_name = response_model.__name__
        phase = "judge" if "Quality" in model_name or "Score" in model_name else "create_or_refine"

        if vram.enabled:
            vram.snapshot(f"pre_llm_{phase}_{model}")

        call_start = time.perf_counter()
        result = original_generate(
            settings=settings,
            model=model,
            prompt=prompt,
            response_model=response_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
        )
        call_end = time.perf_counter()
        duration = call_end - call_start

        timing.record(
            f"llm_{phase}",
            duration,
            model=model,
            response_model=model_name,
            temperature=temperature,
        )

        if vram.enabled:
            vram.record_llm_call(
                model=model,
                call_start=call_start,
                call_end=call_end,
                entity_type=timing._current_entity_type,
                phase=phase,
            )
            vram.snapshot(f"post_llm_{phase}_{model}")

        return result

    # Patch at source module
    originals[(llm_module, "generate_structured")] = original_generate
    llm_module.generate_structured = instrumented_generate_structured

    # Patch at all entity module import sites
    entity_modules = [
        "src.services.world_quality_service._location",
        "src.services.world_quality_service._faction",
        "src.services.world_quality_service._item",
        "src.services.world_quality_service._concept",
        "src.services.world_quality_service._character",
        "src.services.world_quality_service._relationship",
        "src.services.world_quality_service._common",
        "src.services.world_quality_service._chapter_quality",
        "src.services.world_quality_service._plot",
        "src.services.model_mode_service._scoring",
        "src.services.import_service",
    ]
    for mod_name in entity_modules:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "generate_structured"):
            originals[(mod, "generate_structured")] = mod.generate_structured
            mod.generate_structured = instrumented_generate_structured  # type: ignore[attr-defined]
        else:
            logger.warning(
                "Module %s not loaded or missing generate_structured — patch skipped", mod_name
            )

    # --- Patch validate_unique_name ---
    original_validate = validation_module.validate_unique_name

    def instrumented_validate_unique_name(
        name,
        existing_names,
        check_substring=True,
        min_substring_length=4,
        check_semantic=False,
        semantic_threshold=0.85,
        ollama_url=None,
        embedding_model=None,
    ):
        """Wrap validate_unique_name with timing instrumentation."""
        start = time.perf_counter()
        result = original_validate(
            name,
            existing_names,
            check_substring=check_substring,
            min_substring_length=min_substring_length,
            check_semantic=check_semantic,
            semantic_threshold=semantic_threshold,
            ollama_url=ollama_url,
            embedding_model=embedding_model,
        )
        duration = time.perf_counter() - start
        timing.record(
            "validate_unique_name",
            duration,
            check_semantic=check_semantic,
            existing_count=len(existing_names),
            is_unique=result[0],
            reason=result[2],
        )
        return result

    originals[(validation_module, "validate_unique_name")] = original_validate
    validation_module.validate_unique_name = instrumented_validate_unique_name

    # Patch at entity module import sites
    validation_modules = [
        "src.services.world_quality_service._location",
        "src.services.world_quality_service._faction",
        "src.services.world_quality_service._item",
        "src.services.world_quality_service._concept",
    ]
    for mod_name in validation_modules:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "validate_unique_name"):
            originals[(mod, "validate_unique_name")] = mod.validate_unique_name
            mod.validate_unique_name = instrumented_validate_unique_name  # type: ignore[attr-defined]
        else:
            logger.warning(
                "Module %s not loaded or missing validate_unique_name — patch skipped", mod_name
            )

    # --- Patch SemanticDuplicateChecker ---
    from src.utils.similarity import SemanticDuplicateChecker

    original_init = SemanticDuplicateChecker.__post_init__
    original_find = SemanticDuplicateChecker.find_semantic_duplicate

    def instrumented_post_init(self):
        """Wrap SemanticDuplicateChecker.__post_init__ with timing."""
        start = time.perf_counter()
        original_init(self)
        duration = time.perf_counter() - start
        timing.record(
            "semantic_init",
            duration,
            model=self.embedding_model,
            degraded=self._degraded,
        )

    def instrumented_find_semantic_duplicate(self, name, existing_names):
        """Wrap find_semantic_duplicate with timing instrumentation."""
        start = time.perf_counter()
        result = original_find(self, name, existing_names)
        duration = time.perf_counter() - start
        timing.record(
            "semantic_check",
            duration,
            candidate=name,
            existing_count=len(existing_names),
            is_duplicate=result[0],
            similarity=result[2],
        )
        return result

    originals[(SemanticDuplicateChecker, "__post_init__")] = original_init
    originals[(SemanticDuplicateChecker, "find_semantic_duplicate")] = original_find
    SemanticDuplicateChecker.__post_init__ = instrumented_post_init  # type: ignore[method-assign]
    SemanticDuplicateChecker.find_semantic_duplicate = instrumented_find_semantic_duplicate  # type: ignore[method-assign]

    # --- Patch WorldDatabase.add_entity ---
    from src.memory.world_database import _entities as entities_module

    original_add = entities_module.add_entity

    def instrumented_add_entity(db, entity_type, name, description="", attributes=None):
        """Wrap WorldDatabase.add_entity with timing instrumentation."""
        start = time.perf_counter()
        result = original_add(db, entity_type, name, description, attributes)
        duration = time.perf_counter() - start
        timing.record(
            "db_add_entity",
            duration,
            entity_type_db=entity_type,
            entity_name_db=name,
        )
        return result

    originals[(entities_module, "add_entity")] = original_add
    entities_module.add_entity = instrumented_add_entity

    logger.info("Installed %d instrumentation patches", len(originals))
    return originals


def uninstall_patches(originals: dict[tuple[Any, str], Any]) -> None:
    """Restore original functions from saved originals.

    Args:
        originals: Dict from install_patches() mapping (object, attr_name) to original callables.
    """
    for (obj, attr_name), original_fn in originals.items():
        setattr(obj, attr_name, original_fn)

    logger.info("Uninstalled %d instrumentation patches", len(originals))


# =====================================================================
# Health snapshot helpers
# =====================================================================
def take_health_snapshot(
    svc: ServiceContainer,
    world_db_path: Path | None = None,
) -> dict[str, Any]:
    """Take a world health snapshot using the real WorldService.

    Note: Health snapshots are only meaningful when ``world_db_path`` points to
    a real world database that entities are being written to.  When no path is
    provided, the script creates a throwaway diagnostic DB whose health metrics
    will remain at baseline (no entities are written there by the generation
    functions this script instruments).

    Args:
        svc: ServiceContainer with world service.
        world_db_path: Path to world DB file. If None, creates a temporary
            diagnostic DB (health deltas will be zero).

    Returns:
        Health metrics as a serializable dict.
    """
    from src.memory.world_database import WorldDatabase

    if world_db_path and world_db_path.exists():
        logger.info("Taking health snapshot from existing DB: %s", world_db_path)
        db_path = world_db_path
    else:
        # Create a per-run temporary DB so prior runs don't contaminate snapshots
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        db_path = diagnostics_dir / f"build_perf_diagnostic_{timestamp}.db"
        logger.info("Creating temporary diagnostic DB at %s", db_path)

    with WorldDatabase(db_path) as world_db:
        metrics = svc.world.get_world_health_metrics(world_db)
        return metrics.model_dump()


def compute_health_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute the delta between two health snapshots.

    Args:
        before: Health snapshot before generation.
        after: Health snapshot after generation.

    Returns:
        Dict with delta values for key metrics.
    """
    return {
        "health_delta": round(after["health_score"] - before["health_score"], 1),
        "entities_delta": after["total_entities"] - before["total_entities"],
        "relationships_delta": after["total_relationships"] - before["total_relationships"],
        "orphans_delta": after["orphan_count"] - before["orphan_count"],
        "circular_delta": after["circular_count"] - before["circular_count"],
        "avg_quality_delta": round(after["average_quality"] - before["average_quality"], 2),
        "density_delta": round(after["relationship_density"] - before["relationship_density"], 3),
    }


# =====================================================================
# Per-entity timing waterfall builder
# =====================================================================
def build_per_entity_waterfalls(
    timing: TimingCollector,
    entity_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-entity timing waterfall from collected events.

    Args:
        timing: TimingCollector with all events.
        entity_results: List of entity result dicts with entity_type, entity_name, total_s.

    Returns:
        List of per-entity timing breakdown dicts.
    """
    waterfalls: list[dict[str, Any]] = []

    for entity_info in entity_results:
        entity_type = entity_info["entity_type"]
        entity_name = entity_info["entity_name"]
        total_s = entity_info["total_s"]

        events = timing.get_events_for_entity(entity_type, entity_name)

        llm_create_s = sum(e.duration_s for e in events if e.operation == "llm_create_or_refine")
        llm_judge_s = sum(e.duration_s for e in events if e.operation == "llm_judge")
        semantic_init_s = sum(e.duration_s for e in events if e.operation == "semantic_init")
        semantic_check_s = sum(e.duration_s for e in events if e.operation == "semantic_check")
        validate_name_s = sum(e.duration_s for e in events if e.operation == "validate_unique_name")
        db_add_s = sum(e.duration_s for e in events if e.operation == "db_add_entity")

        # validate_unique_name includes both semantic_init and semantic_check time,
        # so subtract both to avoid double-counting
        validate_non_semantic_s = max(0, validate_name_s - semantic_init_s - semantic_check_s)
        accounted = (
            llm_create_s
            + llm_judge_s
            + semantic_init_s
            + semantic_check_s
            + validate_non_semantic_s
            + db_add_s
        )
        raw_unaccounted = total_s - accounted
        if raw_unaccounted < 0:
            logger.warning(
                "Timing mismatch for %s/%s: accounted %.2fs > total %.2fs (delta=%.2fs). "
                "Sub-operations may overlap or total_s was measured at a coarser granularity.",
                entity_type,
                entity_name,
                accounted,
                total_s,
                raw_unaccounted,
            )
        unaccounted_s = max(0, raw_unaccounted)

        waterfalls.append(
            {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "total_s": round(total_s, 2),
                "llm_create_refine_s": round(llm_create_s, 2),
                "llm_judge_s": round(llm_judge_s, 2),
                "semantic_init_s": round(semantic_init_s, 2),
                "semantic_check_s": round(semantic_check_s, 2),
                "validate_name_s": round(validate_name_s, 2),
                "db_add_entity_s": round(db_add_s, 4),
                "unaccounted_s": round(unaccounted_s, 2),
                "iterations": entity_info["iterations"],
                "final_score": entity_info["final_score"],
            }
        )

    return waterfalls


def build_timing_aggregate(waterfalls: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate timing statistics from per-entity waterfalls.

    Args:
        waterfalls: List of per-entity timing dicts from build_per_entity_waterfalls.

    Returns:
        Aggregate timing summary.
    """
    total_time = sum(w["total_s"] for w in waterfalls)
    total_llm = sum(w["llm_create_refine_s"] + w["llm_judge_s"] for w in waterfalls)
    total_semantic = sum(w["semantic_init_s"] + w["semantic_check_s"] for w in waterfalls)
    total_db = sum(w["db_add_entity_s"] for w in waterfalls)
    total_validate = sum(w["validate_name_s"] for w in waterfalls)
    total_unaccounted = sum(w["unaccounted_s"] for w in waterfalls)

    def pct(part: float, whole: float) -> float:
        """Compute percentage, returning 0.0 if whole is zero."""
        return round(part / whole * 100, 1) if whole > 0 else 0.0

    return {
        "total_time_s": round(total_time, 2),
        "total_llm_s": round(total_llm, 2),
        "total_semantic_s": round(total_semantic, 2),
        "total_validate_s": round(total_validate, 2),
        "total_db_s": round(total_db, 4),
        "total_unaccounted_s": round(total_unaccounted, 2),
        "pct_llm": pct(total_llm, total_time),
        "pct_semantic": pct(total_semantic, total_time),
        "pct_validate": pct(total_validate, total_time),
        "pct_db": pct(total_db, total_time),
        "pct_unaccounted": pct(total_unaccounted, total_time),
    }


# =====================================================================
# Diagnosis generator
# =====================================================================
def generate_diagnosis(
    waterfalls: list[dict[str, Any]],
    aggregate: dict[str, Any],
    vram: VramMonitor,
    health_diff: dict[str, Any] | None,
    total_time_s: float,
) -> dict[str, Any]:
    """Generate automated diagnosis from collected data.

    Args:
        waterfalls: Per-entity timing breakdowns.
        aggregate: Aggregate timing stats.
        vram: VramMonitor with transition data.
        health_diff: Health delta dict or None if skipped.
        total_time_s: Total wall-clock time for the whole run.

    Returns:
        Diagnosis dict with findings and recommendations.
    """
    diagnosis: dict[str, Any] = {}

    # --- Timing diagnosis ---
    # Check for embedding model cold-start
    semantic_inits = [w["semantic_init_s"] for w in waterfalls if w["semantic_init_s"] > 0]
    if semantic_inits and semantic_inits[0] > COLD_START_THRESHOLD_S:
        diagnosis["timing_gap_primary_cause"] = "semantic_duplicate_init"
        diagnosis["timing_detail"] = (
            f"First semantic init took {semantic_inits[0]:.1f}s (embedding model cold-start). "
            f"Subsequent inits: {[round(s, 1) for s in semantic_inits[1:]]}"
        )
    elif aggregate["pct_semantic"] > SEMANTIC_OVERHEAD_THRESHOLD_PCT:
        diagnosis["timing_gap_primary_cause"] = "semantic_checks_overhead"
        diagnosis["timing_detail"] = (
            f"Semantic checks consume {aggregate['pct_semantic']:.1f}% of total time"
        )
    elif aggregate["pct_unaccounted"] > UNACCOUNTED_OVERHEAD_THRESHOLD_PCT:
        diagnosis["timing_gap_primary_cause"] = "unaccounted_overhead"
        diagnosis["timing_detail"] = (
            f"Unaccounted time is {aggregate['pct_unaccounted']:.1f}% of total. "
            "Possible causes: Python overhead, GC, context switching"
        )
    else:
        diagnosis["timing_gap_primary_cause"] = "none_significant"
        diagnosis["timing_detail"] = "No significant timing gaps detected"

    # --- VRAM diagnosis ---
    if vram.enabled and vram.transitions:
        total_swap_s = sum(t.swap_time_s for t in vram.transitions)
        swap_pct = (total_swap_s / total_time_s * 100) if total_time_s > 0 else 0

        if swap_pct > VRAM_SWAP_HIGH_THRESHOLD_PCT:
            diagnosis["vram_swap_severity"] = "high"
            diagnosis["vram_recommendation"] = (
                f"Model swapping consumes {swap_pct:.0f}% of total time ({total_swap_s:.1f}s). "
                "Consider: (1) use same model for creator/judge, "
                "(2) batch all creates then all judges, "
                "(3) use smaller models that fit together in VRAM"
            )
        elif swap_pct > VRAM_SWAP_MEDIUM_THRESHOLD_PCT:
            diagnosis["vram_swap_severity"] = "medium"
            diagnosis["vram_recommendation"] = (
                f"Model swapping consumes {swap_pct:.0f}% of total time ({total_swap_s:.1f}s). "
                "Moderate overhead — batching could help"
            )
        else:
            diagnosis["vram_swap_severity"] = "low"
            diagnosis["vram_recommendation"] = (
                f"Model swapping consumes only {swap_pct:.0f}% of total time. "
                "Not a significant bottleneck"
            )
    elif vram.enabled:
        diagnosis["vram_swap_severity"] = "none"
        diagnosis["vram_recommendation"] = "No model transitions detected (single model mode?)"
    else:
        diagnosis["vram_swap_severity"] = "skipped"
        diagnosis["vram_recommendation"] = "VRAM monitoring was skipped"

    # --- Health diagnosis ---
    if health_diff:
        causes = []
        if health_diff["orphans_delta"] > 0:
            causes.append(
                f"Orphan count increased by {health_diff['orphans_delta']} "
                "(relationship generation not keeping up with entity generation)"
            )
        if health_diff["circular_delta"] > 0:
            causes.append(
                f"Circular dependencies increased by {health_diff['circular_delta']} "
                "(check for mutual A↔B or problematic A→B→C→A chains)"
            )
        if health_diff["health_delta"] < HEALTH_DROP_THRESHOLD:
            causes.append(f"Health score dropped by {abs(health_diff['health_delta']):.1f} points")

        diagnosis["health_regression_cause"] = (
            "; ".join(causes) if causes else "No significant health regression detected"
        )
    else:
        diagnosis["health_regression_cause"] = "Health monitoring was skipped"

    return diagnosis


# =====================================================================
# Entity generation runner
# =====================================================================
def run_entity_generation(
    svc: ServiceContainer,
    entity_types: list[str],
    count_per_type: int,
    timing: TimingCollector,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Run entity generation with instrumentation.

    Uses the real ServiceContainer and WorldQualityService to generate
    entities through the quality refinement loop.

    Args:
        svc: Initialized ServiceContainer.
        entity_types: List of entity types to generate.
        count_per_type: Number of entities per type.
        timing: TimingCollector for recording events.
        verbose: Print progress to console.

    Returns:
        List of entity result dicts with timing info.
    """
    brief = make_canonical_brief()
    story_state = make_story_state(brief)
    wqs = svc.world_quality
    results: list[dict[str, Any]] = []

    # Map entity types to their generation functions
    generate_fns: dict[str, Callable] = {
        "location": lambda names: wqs.generate_location_with_quality(story_state, names),
        "faction": lambda names: wqs.generate_faction_with_quality(story_state, names),
        "item": lambda names: wqs.generate_item_with_quality(story_state, names),
        "concept": lambda names: wqs.generate_concept_with_quality(story_state, names),
        "character": lambda names: wqs.generate_character_with_quality(story_state, names),
    }

    total = len(entity_types) * count_per_type
    completed = 0

    for entity_type in entity_types:
        existing_names: list[str] = []

        if entity_type not in generate_fns:
            logger.warning("Skipping unsupported entity type: %s", entity_type)
            continue

        gen_fn = generate_fns[entity_type]

        for i in range(count_per_type):
            completed += 1
            if verbose:
                print(f"  [{completed}/{total}] Generating {entity_type} #{i + 1}...")

            placeholder_name = f"{entity_type}_{i + 1}"
            timing.set_current_entity(entity_type, placeholder_name)
            events_before = len(timing.events)
            entity_start = time.perf_counter()

            try:
                entity, scores, iterations = gen_fn(existing_names)

                entity_elapsed = time.perf_counter() - entity_start

                # Extract name
                if isinstance(entity, dict):
                    name = entity.get("name", placeholder_name)
                else:
                    name = getattr(entity, "name", placeholder_name)

                # Update timing collector with actual name
                timing.set_current_entity(entity_type, name)
                # Re-tag only newly added events (avoids scanning all events)
                for event in timing.events[events_before:]:
                    if event.entity_type == entity_type and event.entity_name == placeholder_name:
                        event.entity_name = name

                existing_names.append(name)

                logger.info(
                    "Generated %s '%s': score=%.1f, iters=%d, time=%.1fs",
                    entity_type,
                    name,
                    scores.average,
                    iterations,
                    entity_elapsed,
                )

                result_info = {
                    "entity_type": entity_type,
                    "entity_name": name,
                    "total_s": round(entity_elapsed, 2),
                    "iterations": iterations,
                    "final_score": round(scores.average, 2),
                }
                results.append(result_info)

                if verbose:
                    print(
                        f"    OK: '{name}' score={scores.average:.1f} "
                        f"iters={iterations} time={entity_elapsed:.1f}s"
                    )

            except Exception as e:
                entity_elapsed = time.perf_counter() - entity_start
                logger.error("Failed to generate %s #%d: %s", entity_type, i + 1, e)
                results.append(
                    {
                        "entity_type": entity_type,
                        "entity_name": f"{entity_type}_{i + 1}_FAILED",
                        "total_s": round(entity_elapsed, 2),
                        "iterations": 0,
                        "final_score": 0,
                        "error": str(e)[:200],
                    }
                )
                if verbose:
                    print(f"    FAILED: {e}")

    return results


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the build performance investigation."""
    parser = argparse.ArgumentParser(
        description=(
            "Build Performance Profiling — unified investigation for issues #324, #325, #327. "
            "Instruments a real entity generation build to profile timing gaps, "
            "VRAM swap latency, and world health regression."
        )
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        default="location,faction,concept",
        help=(
            f"Comma-separated entity types to generate "
            f"(default: location,faction,concept). Options: {', '.join(SUPPORTED_ENTITY_TYPES)}"
        ),
    )
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=2,
        help="Number of entities to generate per type (default: 2)",
    )
    parser.add_argument(
        "--world-db",
        type=str,
        help="Path to existing world DB for health comparison (optional)",
    )
    parser.add_argument(
        "--skip-vram",
        action="store_true",
        help="Skip VRAM monitoring (faster, no nvidia-smi calls)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health metric snapshots",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress to console",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse entity types
    entity_types = [t.strip() for t in args.entity_types.split(",")]
    invalid = [t for t in entity_types if t not in SUPPORTED_ENTITY_TYPES]
    if invalid:
        print(f"ERROR: Invalid entity types: {invalid}")
        print(f"Valid types: {SUPPORTED_ENTITY_TYPES}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"build_perf_{timestamp}.json"

    # Initialize services
    print("Loading settings and initializing services...")
    settings = Settings.load()
    svc = ServiceContainer(settings)

    # Resolve models for display
    creator_model = svc.world_quality._get_creator_model()
    judge_model = svc.world_quality._get_judge_model()
    embedding_model = settings.embedding_model
    semantic_enabled = settings.semantic_duplicate_enabled

    # Print header
    print("=" * 70)
    print("BUILD PERFORMANCE PROFILING")
    print("Issues: #324 (VRAM swap), #325 (health regression), #327 (timing gaps)")
    print("=" * 70)
    print(f"Creator model:    {creator_model}")
    print(f"Judge model:      {judge_model}")
    print(f"Embedding model:  {embedding_model}")
    print(f"Semantic dedup:   {'enabled' if semantic_enabled else 'disabled'}")
    print(f"Entity types:     {entity_types}")
    print(f"Count per type:   {args.count_per_type}")
    print(f"Total entities:   {len(entity_types) * args.count_per_type}")
    print(f"VRAM monitoring:  {'enabled' if not args.skip_vram else 'SKIPPED'}")
    print(f"Health snapshots: {'enabled' if not args.skip_health else 'SKIPPED'}")
    print(f"Output:           {output_path}")
    print("=" * 70)
    print()

    # Get model info for metadata
    creator_info = get_model_info(creator_model)
    judge_info = get_model_info(judge_model)

    # Initialize collectors
    timing = TimingCollector()
    vram = VramMonitor(enabled=not args.skip_vram)

    # Resolve world DB path for health snapshots
    world_db_path = Path(args.world_db) if args.world_db else None

    # Health snapshot: before
    health_before: dict[str, Any] | None = None
    if not args.skip_health:
        print("Taking health snapshot (before)...")
        health_before = take_health_snapshot(svc, world_db_path)
        if args.verbose:
            print(f"  Health score: {health_before['health_score']:.1f}")
            print(f"  Entities: {health_before['total_entities']}")
            print(f"  Relationships: {health_before['total_relationships']}")
        print()

    # Install instrumentation
    print("Installing instrumentation patches...")
    originals = install_patches(timing, vram)
    print()

    # Run instrumented build
    print("Starting instrumented entity generation...")
    overall_start = time.perf_counter()

    try:
        entity_results = run_entity_generation(
            svc=svc,
            entity_types=entity_types,
            count_per_type=args.count_per_type,
            timing=timing,
            verbose=args.verbose,
        )
    finally:
        # Always uninstall patches
        uninstall_patches(originals)

    overall_time = round(time.perf_counter() - overall_start, 2)
    print()
    print(f"Entity generation complete in {overall_time:.1f}s")

    # Health snapshot: after
    health_after: dict[str, Any] | None = None
    health_diff: dict[str, Any] | None = None
    if not args.skip_health:
        print("Taking health snapshot (after)...")
        health_after = take_health_snapshot(svc, world_db_path)
        if health_before is not None:
            health_diff = compute_health_diff(health_before, health_after)
        if args.verbose and health_after:
            print(f"  Health score: {health_after['health_score']:.1f}")
            print(f"  Entities: {health_after['total_entities']}")
            print(f"  Relationships: {health_after['total_relationships']}")
            if health_diff:
                print(f"  Health delta: {health_diff['health_delta']:+.1f}")
        print()

    # Build timing waterfalls
    waterfalls = build_per_entity_waterfalls(timing, entity_results)
    aggregate = build_timing_aggregate(waterfalls)

    # Build VRAM summary
    vram_summary: dict[str, Any] = {}
    if vram.enabled:
        total_swap_s = sum(t.swap_time_s for t in vram.transitions)
        swap_pct = (total_swap_s / overall_time * 100) if overall_time > 0 else 0

        vram_summary = {
            "gpu_total_mb": vram.gpu_total_mb,
            "transitions": [
                {
                    "from_model": t.from_model,
                    "to_model": t.to_model,
                    "swap_time_s": t.swap_time_s,
                    "entity_type": t.entity_type,
                    "phase": t.phase,
                }
                for t in vram.transitions
            ],
            "total_transitions": len(vram.transitions),
            "total_swap_overhead_s": round(total_swap_s, 2),
            "pct_of_total_time": round(swap_pct, 1),
            "snapshots": [
                {
                    "context": s.context,
                    "gpu_used_mb": s.gpu_vram.get("used_mb", 0),
                    "gpu_free_mb": s.gpu_vram.get("free_mb", 0),
                    "loaded_models": [m["name"] for m in s.loaded_models],
                }
                for s in vram.snapshots
            ],
        }

    # Generate diagnosis
    diagnosis = generate_diagnosis(waterfalls, aggregate, vram, health_diff, overall_time)

    # Build output JSON
    output: dict[str, Any] = {
        "metadata": {
            "script": "investigate_build_performance.py",
            "issues": ["#324", "#325", "#327"],
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_s": overall_time,
            "models": {
                "creator": creator_model,
                "creator_info": creator_info,
                "judge": judge_model,
                "judge_info": judge_info,
                "embedding": embedding_model,
            },
            "entity_types": entity_types,
            "count_per_type": args.count_per_type,
            "semantic_duplicate_enabled": semantic_enabled,
        },
        "timing": {
            "per_entity": waterfalls,
            "aggregate": aggregate,
        },
    }

    if vram.enabled:
        output["vram"] = vram_summary

    if not args.skip_health:
        output["health"] = {
            "before": health_before,
            "after": health_after,
            "diff": health_diff,
        }

    output["diagnosis"] = diagnosis

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    # Print summary
    print()
    print("=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    print()
    print("TIMING (#327):")
    print(f"  Primary cause: {diagnosis['timing_gap_primary_cause']}")
    print(f"  Detail: {diagnosis['timing_detail']}")
    print(
        f"  Breakdown: LLM={aggregate['pct_llm']:.0f}%, "
        f"Semantic={aggregate['pct_semantic']:.0f}%, "
        f"Validation={aggregate['pct_validate']:.0f}%, "
        f"DB={aggregate['pct_db']:.0f}%, "
        f"Unaccounted={aggregate['pct_unaccounted']:.0f}%"
    )

    if vram.enabled:
        print()
        print("VRAM SWAP (#324):")
        print(f"  Severity: {diagnosis['vram_swap_severity']}")
        print(f"  Transitions: {len(vram.transitions)}")
        if vram.transitions:
            total_swap_s = sum(t.swap_time_s for t in vram.transitions)
            print(f"  Total swap overhead: {total_swap_s:.1f}s")
            print(f"  Avg swap time: {total_swap_s / len(vram.transitions):.1f}s")
        print(f"  Recommendation: {diagnosis['vram_recommendation']}")

    if health_diff:
        print()
        print("HEALTH (#325):")
        print(f"  Health delta: {health_diff['health_delta']:+.1f}")
        print(f"  Entities delta: {health_diff['entities_delta']:+d}")
        print(f"  Relationships delta: {health_diff['relationships_delta']:+d}")
        print(f"  Orphans delta: {health_diff['orphans_delta']:+d}")
        print(f"  Circular delta: {health_diff['circular_delta']:+d}")
        print(f"  Cause: {diagnosis['health_regression_cause']}")

    print()
    print("PER-ENTITY WATERFALL:")
    print(f"{'Entity':<30} {'Total':>6} {'LLM':>6} {'Sem':>6} {'DB':>6} {'Gap':>6} {'Score':>6}")
    print("-" * 72)
    for w in waterfalls:
        name = w["entity_name"][:28]
        print(
            f"{name:<30} {w['total_s']:>5.1f}s "
            f"{w['llm_create_refine_s'] + w['llm_judge_s']:>5.1f}s "
            f"{w['semantic_init_s'] + w['semantic_check_s']:>5.1f}s "
            f"{w['db_add_entity_s']:>5.2f}s "
            f"{w['unaccounted_s']:>5.1f}s "
            f"{w['final_score']:>5.1f}"
        )

    print()


if __name__ == "__main__":
    main()
