"""Models page - Base class with core functionality."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui
from nicegui.elements.card import Card
from nicegui.elements.column import Column
from nicegui.elements.input import Input

from src.services import ServiceContainer
from src.ui.state import AppState

if TYPE_CHECKING:
    from ._download_mixin import DownloadTask

logger = logging.getLogger(__name__)


class ModelsPageBase:
    """Base class for ModelsPage with core functionality.

    Features:
    - View installed models
    - Pull new models
    - Delete models
    - Model comparison
    - VRAM recommendations
    """

    # Allow 10% VRAM tolerance - if you have 23GB and model needs 24GB, it should work
    VRAM_TOLERANCE = 0.10

    def __init__(self, state: AppState, services: ServiceContainer):
        """
        Create and initialize a ModelsPage using the given application state and services.

        Parameters:
            state (AppState): The application state object used by the page.
            services (ServiceContainer): Container providing access to backend services.
        """
        self.state = state
        self.services = services

        self._model_list: Column | None = None
        self._pull_progress: Column | None = None
        self._comparison_result: Column | None = None
        self._installed_section: Card | None = None
        self._custom_model_input: Input | None = None

        # Download queue - tracks all active/pending downloads
        self._download_queue: dict[str, DownloadTask] = {}
        self._download_lock = asyncio.Lock()
        self._max_concurrent_downloads = 3  # Allow 3 concurrent downloads
        self._background_tasks: set[asyncio.Task[Any]] = set()  # Prevent task GC

        # Filter state
        self._filter_fits_vram = True
        self._filter_quality_min = 0
        self._filter_uncensored_only = False
        self._filter_search = ""

        # Download all state
        self._download_all_btn: ui.button | None = None
        self._queue_status_label: ui.label | None = None

        # Cached values to avoid redundant service calls
        self._cached_vram: float | None = None
        self._cached_installed: set[str] | None = None

        # Comparison UI elements (set by comparison mixin)
        self._compare_models_select: ui.select | None = None
        self._compare_prompt: ui.textarea | None = None

    def _model_fits_vram(self, vram_required: float, available_vram: float) -> bool:
        """Check if model fits in VRAM with tolerance.

        Args:
            vram_required: VRAM required by model.
            available_vram: Available VRAM.

        Returns:
            True if model fits (with tolerance), False otherwise.
        """
        # Allow models that are within 10% of available VRAM
        return vram_required <= available_vram * (1 + self.VRAM_TOLERANCE)

    def _get_vram(self) -> float:
        """Get VRAM from cache or service."""
        if self._cached_vram is None:
            self._cached_vram = self.services.model.get_vram()
        return self._cached_vram

    def _get_installed(self) -> set[str]:
        """Get installed models from cache or service."""
        if self._cached_installed is None:
            self._cached_installed = set(self.services.model.list_installed())
        return self._cached_installed

    def _invalidate_cache(self) -> None:
        """Invalidate cached values - call after model installs/deletes."""
        self._cached_vram = None
        self._cached_installed = None

    def build(self) -> None:
        """Build the models page UI."""
        with ui.column().classes("w-full gap-6 p-4"):
            # Header with VRAM info
            self._build_header()

            # Installed models
            self._build_installed_section()

            # Available models
            self._build_available_section()

            # Model comparison
            self._build_comparison_section()

    def _build_header(self) -> None:
        """Build the header with VRAM info."""
        vram = self._get_vram()
        health = self.services.model.check_health()

        with ui.row().classes("w-full items-center"):
            ui.label("Models").classes("text-2xl font-bold")
            ui.space()

            # VRAM indicator
            with ui.row().classes("items-center gap-2 p-2 bg-gray-100 dark:bg-gray-700 rounded"):
                ui.icon("memory")
                ui.label(f"{vram} GB VRAM").classes("font-medium")

                if health.is_healthy:
                    ui.icon("check_circle", color="green")
                else:
                    ui.icon("error", color="red")

    def _refresh_all(self, notify: bool = True) -> None:
        """Refresh all model lists with visual feedback.

        Args:
            notify: Whether to show a notification. Set False when called from background.
        """
        self._invalidate_cache()
        self._refresh_installed_section()
        self._refresh_model_list()
        self._update_download_all_btn()
        if notify:
            ui.notify("Model lists refreshed", type="info")

    # Methods to be implemented by mixins
    def _build_installed_section(self) -> None:
        """Build the installed models section. Implemented by listing mixin."""
        raise NotImplementedError

    def _refresh_installed_section(self) -> None:
        """Refresh the installed models section. Implemented by listing mixin."""
        raise NotImplementedError

    def _build_available_section(self) -> None:
        """Build the available models section. Implemented by listing mixin."""
        raise NotImplementedError

    def _refresh_model_list(self) -> None:
        """Refresh the model list. Implemented by listing mixin."""
        raise NotImplementedError

    def _update_download_all_btn(self) -> None:
        """Update the download all button. Implemented by listing mixin."""
        raise NotImplementedError

    def _build_comparison_section(self) -> None:
        """Build the comparison section. Implemented by comparison mixin."""
        raise NotImplementedError
