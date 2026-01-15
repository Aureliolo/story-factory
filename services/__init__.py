"""Services layer - business logic separated from UI.

This module provides a clean interface between the UI and the underlying
business logic, agents, and data storage.
"""

from dataclasses import dataclass

from settings import Settings

from .export_service import ExportService
from .model_mode_service import ModelModeService
from .model_service import ModelService
from .project_service import ProjectService
from .scoring_service import ScoringService
from .story_service import StoryService
from .world_service import WorldService


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

    def __init__(self, settings: Settings | None = None):
        """Initialize all services with shared settings.

        Args:
            settings: Application settings. If None, loads from settings.json.
        """
        self.settings = settings or Settings.load()
        self.project = ProjectService(self.settings)
        self.story = StoryService(self.settings)
        self.world = WorldService(self.settings)
        self.model = ModelService(self.settings)
        self.export = ExportService(self.settings)
        self.mode = ModelModeService(self.settings)
        self.scoring = ScoringService(self.mode)


__all__ = [
    "ServiceContainer",
    "ProjectService",
    "StoryService",
    "WorldService",
    "ModelService",
    "ExportService",
    "ModelModeService",
    "ScoringService",
]
