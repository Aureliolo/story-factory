"""Models page package - Ollama model management."""

from src.services import ServiceContainer
from src.ui.state import AppState

from ._comparison_mixin import ComparisonMixin
from ._download_mixin import DownloadMixin, DownloadTask
from ._listing_mixin import ListingMixin
from ._operations_mixin import OperationsMixin
from ._page import ModelsPageBase

__all__ = ["DownloadTask", "ModelsPage"]


class ModelsPage(
    ListingMixin,
    OperationsMixin,
    DownloadMixin,
    ComparisonMixin,
    ModelsPageBase,
):
    """Models management page.

    Composed from mixins:
    - ModelsPageBase: Core functionality (__init__, build, cache management)
    - DownloadMixin: Download queue and progress tracking
    - ListingMixin: Model listing and display
    - OperationsMixin: Model operations (test, delete, update, tags)
    - ComparisonMixin: Model comparison functionality

    Features:
    - View installed models
    - Pull new models
    - Delete models
    - Model comparison
    - VRAM recommendations
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """
        Create and initialize a ModelsPage using the given application state and services.

        Parameters:
            state (AppState): The application state object used by the page.
            services (ServiceContainer): Container providing access to backend services.
        """
        super().__init__(state, services)
