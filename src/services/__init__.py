"""Services layer - business logic separated from UI.

This module provides a clean interface between the UI and the underlying
business logic, agents, and data storage.
"""

import logging
import time
from dataclasses import dataclass

from src.memory.mode_database import ModeDatabase
from src.settings import Settings

from .backup_service import BackupService
from .calendar_service import CalendarService
from .comparison_service import ComparisonService
from .conflict_analysis_service import ConflictAnalysisService
from .content_guidelines_service import ContentGuidelinesService
from .context_retrieval_service import ContextRetrievalService
from .embedding_service import EmbeddingService
from .export_service import ExportService
from .import_service import ImportService
from .model_mode_service import ModelModeService
from .model_service import ModelService
from .project_service import ProjectService
from .scoring_service import ScoringService
from .story_service import StoryService
from .suggestion_service import SuggestionService
from .template_service import TemplateService
from .temporal_validation_service import TemporalValidationService
from .timeline_service import TimelineService
from .world_quality_service import WorldQualityService
from .world_service import WorldBuildOptions, WorldBuildProgress, WorldService
from .world_template_service import WorldTemplateService

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """Dependency injection container for all services.

    Usage:
        settings = Settings.load()
        services = ServiceContainer(settings)

        # Access services
        projects = services.project.list_projects()
        services.story.start_interview(state)
    """

    settings: Settings
    project: ProjectService
    story: StoryService
    world: WorldService
    model: ModelService
    export: ExportService
    mode: ModelModeService
    scoring: ScoringService
    world_quality: WorldQualityService
    suggestion: SuggestionService
    template: TemplateService
    backup: BackupService
    import_svc: ImportService
    comparison: ComparisonService
    timeline: TimelineService
    conflict_analysis: ConflictAnalysisService
    world_template: WorldTemplateService
    content_guidelines: ContentGuidelinesService
    calendar: CalendarService
    temporal_validation: TemporalValidationService
    embedding: EmbeddingService
    context_retrieval: ContextRetrievalService
    mode_db: ModeDatabase

    def __init__(self, settings: Settings | None = None):
        """
        Create and wire service instances that share a Settings object.

        Parameters:
            settings (Settings | None): Application settings to share across services. If omitted, settings are loaded via Settings.load().

        Notes:
            Some services are initialized with other service instances as dependencies (for example, `mode` is provided to `scoring`, `story`, `world_quality`, and `import_svc`; `timeline` is provided to `story`).
        """
        t0 = time.perf_counter()
        logger.info("Initializing ServiceContainer...")
        self.settings = settings or Settings.load()
        self.embedding = EmbeddingService(self.settings)
        self.project = ProjectService(self.settings, self.embedding)
        self.temporal_validation = TemporalValidationService(self.settings)
        self.world = WorldService(self.settings, temporal_validation=self.temporal_validation)
        self.model = ModelService(self.settings)
        self.export = ExportService(self.settings)
        self.mode = ModelModeService(self.settings)
        self.mode_db = self.mode.db  # shared ModeDatabase instance
        self.scoring = ScoringService(self.mode)
        self.context_retrieval = ContextRetrievalService(self.settings, self.embedding)
        self.timeline = TimelineService(self.settings)
        # StoryService needs mode_service for adaptive learning hooks,
        # context_retrieval for RAG-based prompt enrichment,
        # and timeline for temporal context in agent prompts
        self.story = StoryService(
            self.settings,
            mode_service=self.mode,
            context_retrieval=self.context_retrieval,
            timeline=self.timeline,
            mode_db=self.mode_db,
        )
        self.world_quality = WorldQualityService(self.settings, self.mode, self.mode_db)
        self.suggestion = SuggestionService(self.settings)
        self.template = TemplateService(self.settings)
        self.backup = BackupService(self.settings)
        self.import_svc = ImportService(self.settings, self.mode)
        self.comparison = ComparisonService(self.settings)
        self.conflict_analysis = ConflictAnalysisService(self.settings)
        self.world_template = WorldTemplateService(self.settings)
        self.content_guidelines = ContentGuidelinesService(self.settings)
        self.calendar = CalendarService(self.settings)
        service_count = len(self.__class__.__annotations__) - 2  # exclude 'settings' and 'mode_db'
        logger.info(
            "ServiceContainer initialized: %d services in %.2fs",
            service_count,
            time.perf_counter() - t0,
        )


__all__ = [
    "BackupService",
    "CalendarService",
    "ComparisonService",
    "ConflictAnalysisService",
    "ContentGuidelinesService",
    "ContextRetrievalService",
    "EmbeddingService",
    "ExportService",
    "ImportService",
    "ModelModeService",
    "ModelService",
    "ProjectService",
    "ScoringService",
    "ServiceContainer",
    "StoryService",
    "SuggestionService",
    "TemplateService",
    "TemporalValidationService",
    "TimelineService",
    "WorldBuildOptions",
    "WorldBuildProgress",
    "WorldQualityService",
    "WorldService",
    "WorldTemplateService",
]
