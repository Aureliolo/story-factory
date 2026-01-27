"""Analytics page package - model performance dashboard.

This package provides the AnalyticsPage class composed from mixins for different
analytics sections.
"""

from src.services import ServiceContainer
from src.ui.pages.analytics._content_mixin import ContentMixin
from src.ui.pages.analytics._costs_mixin import CostsMixin
from src.ui.pages.analytics._export_mixin import ExportMixin
from src.ui.pages.analytics._model_mixin import ModelMixin
from src.ui.pages.analytics._page import AnalyticsPageBase
from src.ui.pages.analytics._recommendations_mixin import RecommendationsMixin
from src.ui.pages.analytics._summary_mixin import SummaryMixin
from src.ui.pages.analytics._trends_mixin import TrendsMixin
from src.ui.pages.analytics._world_quality_mixin import WorldQualityMixin
from src.ui.state import AppState


class AnalyticsPage(
    SummaryMixin,
    ModelMixin,
    ContentMixin,
    CostsMixin,
    TrendsMixin,
    WorldQualityMixin,
    RecommendationsMixin,
    ExportMixin,
    AnalyticsPageBase,
):
    """Analytics page for model performance tracking.

    Composed from mixins:
    - SummaryMixin: Summary cards section
    - ModelMixin: Model performance table and insights
    - ContentMixin: Content statistics section
    - CostsMixin: Generation costs section
    - TrendsMixin: Quality trends section
    - WorldQualityMixin: World quality metrics section
    - RecommendationsMixin: Recommendations history section
    - ExportMixin: CSV export functionality
    - AnalyticsPageBase: Core page structure and shared methods
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """
        Initialize the AnalyticsPage with the application state and available services.

        Parameters:
                state (AppState): Application state providing current project and UI context.
                services (ServiceContainer): Service container exposing dependencies used by the page.
        """
        super().__init__(state, services)


__all__ = ["AnalyticsPage"]
