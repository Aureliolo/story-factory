"""Services layer - business logic separated from UI.

This module provides a clean interface between the UI and the underlying
business logic, agents, and data storage.
"""

from dataclasses import dataclass

from src.settings import Settings

from .backup_service import BackupService
from .comparison_service import ComparisonService
from .conflict_analysis_service import ConflictAnalysisService
from .export_service import ExportService
from .import_service import ImportService
from .model_mode_service import ModelModeService
from .model_service import ModelService
from .project_service import ProjectService
from .scoring_service import ScoringService
from .story_service import StoryService
from .suggestion_service import SuggestionService
from .template_service import TemplateService
from .timeline_service import TimelineService
from .world_quality_service import WorldQualityService
from .world_service import WorldBuildOptions, WorldBuildProgress, WorldService


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

    def __init__(self, settings: Settings | None = None):
        """
        Create and wire service instances that share a Settings object.

        Parameters:
            settings (Settings | None): Application settings to share across services. If omitted, settings are loaded via Settings.load().

        Notes:
            Some services are initialized with other service instances as dependencies (for example, `mode` is provided to `scoring`, `story`, `world_quality`, and `import_svc`).
        """
        self.settings = settings or Settings.load()
        self.project = ProjectService(self.settings)
        self.world = WorldService(self.settings)
        self.model = ModelService(self.settings)
        self.export = ExportService(self.settings)
        self.mode = ModelModeService(self.settings)
        self.scoring = ScoringService(self.mode)
        # StoryService needs mode_service for adaptive learning hooks
        self.story = StoryService(self.settings, mode_service=self.mode)
        self.world_quality = WorldQualityService(self.settings, self.mode)
        self.suggestion = SuggestionService(self.settings)
        self.template = TemplateService(self.settings)
        self.backup = BackupService(self.settings)
        self.import_svc = ImportService(self.settings, self.mode)
        self.comparison = ComparisonService(self.settings)
        self.timeline = TimelineService(self.settings)
        self.conflict_analysis = ConflictAnalysisService(self.settings)


__all__ = [
    "BackupService",
    "ComparisonService",
    "ConflictAnalysisService",
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
    "TimelineService",
    "WorldBuildOptions",
    "WorldBuildProgress",
    "WorldQualityService",
    "WorldService",
]
