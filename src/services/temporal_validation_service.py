"""Temporal validation service - validates temporal consistency of world entities."""

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

from src.memory.entities import Entity
from src.memory.timeline_types import (
    EntityLifecycle,
    StoryTimestamp,
    extract_lifecycle_from_attributes,
)
from src.memory.world_calendar import WorldCalendar
from src.memory.world_database import WorldDatabase
from src.settings import Settings

logger = logging.getLogger(__name__)


class TemporalErrorType(StrEnum):
    """Types of temporal validation errors."""

    PREDATES_DEPENDENCY = "predates_dependency"  # Entity exists before its dependency
    INVALID_ERA = "invalid_era"  # Era reference doesn't match calendar
    ANACHRONISM = "anachronism"  # Technology/concept doesn't fit era
    POST_DESTRUCTION = "post_destruction"  # Event after location/faction destruction
    INVALID_DATE = "invalid_date"  # Date doesn't validate against calendar
    LIFESPAN_OVERLAP = "lifespan_overlap"  # Character lifespan inconsistent with events
    FOUNDING_ORDER = "founding_order"  # Faction founded before parent faction


class TemporalErrorSeverity(StrEnum):
    """Severity levels for temporal errors."""

    WARNING = "warning"  # Minor inconsistency, acceptable
    ERROR = "error"  # Significant inconsistency, should fix


class TemporalValidationIssue(BaseModel):
    """A detected temporal consistency issue (error or warning)."""

    entity_id: str = Field(description="ID of the entity with the error")
    entity_name: str = Field(description="Name of the entity")
    entity_type: str = Field(description="Type of entity (character, faction, etc.)")
    error_type: TemporalErrorType = Field(description="Type of temporal error")
    severity: TemporalErrorSeverity = Field(description="Severity of the error")
    message: str = Field(description="Human-readable error description")
    related_entity_id: str | None = Field(
        default=None, description="ID of related entity causing conflict"
    )
    related_entity_name: str | None = Field(default=None, description="Name of related entity")
    suggestion: str = Field(default="", description="Suggested fix")


class TemporalValidationResult(BaseModel):
    """Result of temporal validation for an entity or world."""

    errors: list[TemporalValidationIssue] = Field(default_factory=list)
    warnings: list[TemporalValidationIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings OK). Computed from errors list."""
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        """Get count of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get count of warnings."""
        return len(self.warnings)

    @property
    def total_issues(self) -> int:
        """Get total count of issues."""
        return self.error_count + self.warning_count


class TemporalValidationService:
    """Service for validating temporal consistency of world entities.

    Validates that:
    - Characters are born before joining factions
    - Locations exist before events occur there
    - Factions are founded before members join
    - Items are created within valid time periods
    - Era references match the calendar
    """

    def __init__(self, settings: Settings):
        """Initialize temporal validation service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        logger.debug("Initialized TemporalValidationService")

    def validate_entity(
        self,
        entity: Entity,
        calendar: WorldCalendar | None,
        all_entities: list[Entity],
        relationships: list[tuple[str, str, str]],
    ) -> TemporalValidationResult:
        """Validate temporal consistency of a single entity.

        Args:
            entity: Entity to validate.
            calendar: WorldCalendar for date validation (optional).
            all_entities: All entities in the world for cross-reference.
            relationships: List of (source_id, target_id, relation_type) tuples.

        Returns:
            TemporalValidationResult with any errors/warnings found.
        """
        logger.debug(f"Validating temporal consistency for {entity.type} '{entity.name}'")
        result = TemporalValidationResult()

        # Check if temporal validation is enabled in settings
        if not self.settings.validate_temporal_consistency:
            logger.debug("Temporal validation disabled in settings, skipping")
            return result

        # Extract lifecycle from entity attributes
        lifecycle = None
        if entity.attributes:
            lifecycle = extract_lifecycle_from_attributes(entity.attributes)

        # Build entity lookup for cross-reference
        entity_map = {e.id: e for e in all_entities}

        # Run validation rules based on entity type
        if entity.type == "character":
            self._validate_character(entity, lifecycle, calendar, entity_map, relationships, result)
        elif entity.type == "faction":
            self._validate_faction(entity, lifecycle, calendar, entity_map, relationships, result)
        elif entity.type == "location":
            self._validate_location(entity, lifecycle, calendar, entity_map, relationships, result)
        elif entity.type == "item":
            self._validate_item(entity, lifecycle, calendar, entity_map, relationships, result)

        # Validate dates against calendar if present
        if calendar and lifecycle:
            self._validate_dates_against_calendar(entity, lifecycle, calendar, result)

        logger.debug(
            f"Validation complete for '{entity.name}': "
            f"{len(result.errors)} errors, {len(result.warnings)} warnings"
        )
        return result

    def validate_world(self, world_db: WorldDatabase) -> TemporalValidationResult:
        """Validate temporal consistency of entire world.

        Args:
            world_db: WorldDatabase to validate.

        Returns:
            TemporalValidationResult with all errors/warnings found.
        """
        logger.info("Validating temporal consistency for world")
        result = TemporalValidationResult()

        # Check if temporal validation is enabled in settings
        if not self.settings.validate_temporal_consistency:
            logger.debug("Temporal validation disabled in settings, skipping world validation")
            return result

        # Get all entities
        all_entities = world_db.list_entities()

        # Get all relationships
        all_relationships = [
            (rel.source_id, rel.target_id, rel.relation_type)
            for rel in world_db.list_relationships()
        ]

        # Attempt to load calendar from world settings if available
        calendar = None
        try:
            world_settings = world_db.get_world_settings()
            if world_settings and world_settings.calendar:
                calendar = world_settings.calendar
                logger.debug(f"Loaded calendar for validation: {calendar.current_era_name}")
        except Exception as e:
            logger.debug(f"Could not load world calendar (table may not exist): {e}")

        # Validate each entity
        for entity in all_entities:
            entity_result = self.validate_entity(entity, calendar, all_entities, all_relationships)
            result.errors.extend(entity_result.errors)
            result.warnings.extend(entity_result.warnings)

        logger.info(
            f"World validation complete: {len(result.errors)} errors, "
            f"{len(result.warnings)} warnings across {len(all_entities)} entities"
        )
        return result

    def _validate_character(
        self,
        entity: Entity,
        lifecycle: EntityLifecycle | None,
        calendar: WorldCalendar | None,
        entity_map: dict[str, Entity],
        relationships: list[tuple[str, str, str]],
        result: TemporalValidationResult,
    ) -> None:
        """Validate temporal rules for characters."""
        if not lifecycle or not lifecycle.birth or lifecycle.birth.year is None:
            logger.debug(f"Character '{entity.name}' has no birth year, skipping birth checks")
            return

        birth_year = lifecycle.birth.year

        # Check faction memberships
        for source_id, target_id, rel_type in relationships:
            if source_id == entity.id and rel_type == "member_of":
                target = entity_map.get(target_id)
                if target and target.type == "faction":
                    target_lifecycle = None
                    if target.attributes:
                        target_lifecycle = extract_lifecycle_from_attributes(target.attributes)

                    founding_year = None
                    if target_lifecycle:
                        founding_year = target_lifecycle.founding_year

                    if founding_year is not None and birth_year < founding_year:
                        error = TemporalValidationIssue(
                            entity_id=entity.id,
                            entity_name=entity.name,
                            entity_type=entity.type,
                            error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                            severity=TemporalErrorSeverity.ERROR,
                            message=(
                                f"Character born in {birth_year} but faction "
                                f"'{target.name}' founded in {founding_year}"
                            ),
                            related_entity_id=target.id,
                            related_entity_name=target.name,
                            suggestion=(
                                f"Adjust birth year to {founding_year} or later, "
                                f"or adjust faction founding year"
                            ),
                        )
                        result.errors.append(error)
                        logger.warning(f"Temporal error: {error.message}")

    def _validate_faction(
        self,
        entity: Entity,
        lifecycle: EntityLifecycle | None,
        calendar: WorldCalendar | None,
        entity_map: dict[str, Entity],
        relationships: list[tuple[str, str, str]],
        result: TemporalValidationResult,
    ) -> None:
        """Validate temporal rules for factions."""
        founding_year = None

        if lifecycle:
            founding_year = lifecycle.founding_year
            # destruction_year could be used for future validation rules

        if founding_year is None:
            logger.debug(f"Faction '{entity.name}' has no founding year, skipping checks")
            return

        # Check parent faction relationships
        for source_id, target_id, rel_type in relationships:
            if source_id == entity.id and rel_type in ("child_of", "offshoot_of", "split_from"):
                target = entity_map.get(target_id)
                if target and target.type == "faction":
                    target_lifecycle = None
                    if target.attributes:
                        target_lifecycle = extract_lifecycle_from_attributes(target.attributes)

                    parent_founding = None
                    if target_lifecycle:
                        parent_founding = target_lifecycle.founding_year

                    if parent_founding is not None and founding_year < parent_founding:
                        error = TemporalValidationIssue(
                            entity_id=entity.id,
                            entity_name=entity.name,
                            entity_type=entity.type,
                            error_type=TemporalErrorType.FOUNDING_ORDER,
                            severity=TemporalErrorSeverity.ERROR,
                            message=(
                                f"Faction founded in {founding_year} but parent faction "
                                f"'{target.name}' founded in {parent_founding}"
                            ),
                            related_entity_id=target.id,
                            related_entity_name=target.name,
                            suggestion=(f"Adjust founding year to {parent_founding} or later"),
                        )
                        result.errors.append(error)
                        logger.warning(f"Temporal error: {error.message}")

    def _validate_location(
        self,
        entity: Entity,
        lifecycle: EntityLifecycle | None,
        calendar: WorldCalendar | None,
        entity_map: dict[str, Entity],
        relationships: list[tuple[str, str, str]],
        result: TemporalValidationResult,
    ) -> None:
        """Validate temporal rules for locations."""
        destruction_year = None
        if lifecycle:
            destruction_year = lifecycle.destruction_year

        if destruction_year is None:
            # No destruction year means location still exists, no post-destruction checks needed
            return

        # Check for events after destruction
        for source_id, target_id, rel_type in relationships:
            if target_id == entity.id and rel_type in ("located_in", "occurred_at"):
                source = entity_map.get(source_id)
                if source:
                    source_lifecycle = None
                    if source.attributes:
                        source_lifecycle = extract_lifecycle_from_attributes(source.attributes)

                    # Check if event/entity timestamp is after destruction
                    event_year = None
                    if source_lifecycle and source_lifecycle.birth:
                        event_year = source_lifecycle.birth.year

                    if event_year is not None and event_year > destruction_year:
                        error = TemporalValidationIssue(
                            entity_id=entity.id,
                            entity_name=entity.name,
                            entity_type=entity.type,
                            error_type=TemporalErrorType.POST_DESTRUCTION,
                            severity=TemporalErrorSeverity.ERROR,
                            message=(
                                f"Event '{source.name}' in {event_year} occurs at location "
                                f"'{entity.name}' destroyed in {destruction_year}"
                            ),
                            related_entity_id=source.id,
                            related_entity_name=source.name,
                            suggestion=(f"Adjust event year to before {destruction_year}"),
                        )
                        result.errors.append(error)
                        logger.warning(f"Temporal error: {error.message}")

    def _validate_item(
        self,
        entity: Entity,
        lifecycle: EntityLifecycle | None,
        calendar: WorldCalendar | None,
        entity_map: dict[str, Entity],
        relationships: list[tuple[str, str, str]],
        result: TemporalValidationResult,
    ) -> None:
        """Validate temporal rules for items."""
        # Items have simpler temporal validation
        # Check if creator existed when item was created
        if not lifecycle:
            return

        creation_year = None
        if lifecycle.birth:
            creation_year = lifecycle.birth.year

        if creation_year is None:
            return

        # Check creator relationship
        for source_id, target_id, rel_type in relationships:
            if target_id == entity.id and rel_type in ("created", "crafted", "forged"):
                creator = entity_map.get(source_id)
                if creator and creator.type == "character":
                    creator_lifecycle = None
                    if creator.attributes:
                        creator_lifecycle = extract_lifecycle_from_attributes(creator.attributes)

                    if creator_lifecycle and creator_lifecycle.birth:
                        creator_birth = creator_lifecycle.birth.year
                        if creator_birth is not None and creation_year < creator_birth:
                            error = TemporalValidationIssue(
                                entity_id=entity.id,
                                entity_name=entity.name,
                                entity_type=entity.type,
                                error_type=TemporalErrorType.PREDATES_DEPENDENCY,
                                severity=TemporalErrorSeverity.ERROR,
                                message=(
                                    f"Item created in {creation_year} but creator "
                                    f"'{creator.name}' born in {creator_birth}"
                                ),
                                related_entity_id=creator.id,
                                related_entity_name=creator.name,
                                suggestion=(f"Adjust creation year to {creator_birth} or later"),
                            )
                            result.errors.append(error)
                            logger.warning(f"Temporal error: {error.message}")

    def _validate_dates_against_calendar(
        self,
        entity: Entity,
        lifecycle: EntityLifecycle,
        calendar: WorldCalendar,
        result: TemporalValidationResult,
    ) -> None:
        """Validate that dates match the calendar rules and era names are consistent."""
        # Validate birth date
        if lifecycle.birth and lifecycle.birth.year is not None:
            is_valid, error_msg = calendar.validate_date(
                lifecycle.birth.year,
                lifecycle.birth.month,
                lifecycle.birth.day,
            )
            if not is_valid:
                warning = TemporalValidationIssue(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.type,
                    error_type=TemporalErrorType.INVALID_DATE,
                    severity=TemporalErrorSeverity.WARNING,
                    message=f"Birth date validation: {error_msg}",
                    suggestion="Adjust date to match calendar rules",
                )
                result.warnings.append(warning)
                logger.warning(f"Temporal warning: {warning.message}")

            # Cross-validate era name against calendar
            self._check_era_name_mismatch(entity, lifecycle.birth, calendar, "birth", result)

        # Validate death date
        if lifecycle.death and lifecycle.death.year is not None:
            is_valid, error_msg = calendar.validate_date(
                lifecycle.death.year,
                lifecycle.death.month,
                lifecycle.death.day,
            )
            if not is_valid:
                warning = TemporalValidationIssue(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.type,
                    error_type=TemporalErrorType.INVALID_DATE,
                    severity=TemporalErrorSeverity.WARNING,
                    message=f"Death date validation: {error_msg}",
                    suggestion="Adjust date to match calendar rules",
                )
                result.warnings.append(warning)
                logger.warning(f"Temporal warning: {warning.message}")

            # Cross-validate era name against calendar
            self._check_era_name_mismatch(entity, lifecycle.death, calendar, "death", result)

    def _check_era_name_mismatch(
        self,
        entity: Entity,
        timestamp: StoryTimestamp,
        calendar: WorldCalendar,
        date_label: str,
        result: TemporalValidationResult,
    ) -> None:
        """Compare an entity's stored era_name against the calendar-resolved era.

        Emits an INVALID_ERA warning when the entity carries an era name that
        does not match what the calendar says for that year.

        Args:
            entity: The entity being validated.
            timestamp: The StoryTimestamp with year and optional era_name.
            calendar: WorldCalendar used to resolve the correct era.
            date_label: Human-readable label for the date (e.g. "birth", "death").
            result: TemporalValidationResult to append warnings to.
        """
        if timestamp.year is None or not timestamp.era_name:
            return

        resolved_era = calendar.get_era_for_year(timestamp.year)
        if resolved_era is None:
            return

        if timestamp.era_name != resolved_era.name:
            warning = TemporalValidationIssue(
                entity_id=entity.id,
                entity_name=entity.name,
                entity_type=entity.type,
                error_type=TemporalErrorType.INVALID_ERA,
                severity=TemporalErrorSeverity.WARNING,
                message=(
                    f"Era name mismatch on {date_label} date: entity says "
                    f"'{timestamp.era_name}' but calendar resolves year "
                    f"{timestamp.year} to '{resolved_era.name}'"
                ),
                suggestion=(
                    f"Change era name from '{timestamp.era_name}' to "
                    f"'{resolved_era.name}' for year {timestamp.year}"
                ),
            )
            result.warnings.append(warning)
            logger.warning("Temporal warning: %s", warning.message)

    def calculate_temporal_consistency_score(self, result: TemporalValidationResult) -> float:
        """Calculate a temporal consistency score (0-10) from validation result.

        Args:
            result: Validation result to score.

        Returns:
            Score from 0 (many errors) to 10 (no issues).
        """
        if result.total_issues == 0:
            return 10.0

        # Each error reduces score by 2, each warning by 0.5
        penalty = (result.error_count * 2.0) + (result.warning_count * 0.5)
        score = max(0.0, 10.0 - penalty)

        logger.debug(
            f"Temporal consistency score: {score:.1f} "
            f"({result.error_count} errors, {result.warning_count} warnings)"
        )
        return score
