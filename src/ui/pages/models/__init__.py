"""Models page - Ollama model management.

Package structure:
    __init__.py     - ModelsPage class, build(), header, cache helpers
    _download.py    - Download queue management and progress tracking
    _listing.py     - Available model listing, filtering, model cards
    _operations.py  - Installed section, test, delete, compare, updates
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from nicegui import ui
from nicegui.elements.card import Card
from nicegui.elements.column import Column
from nicegui.elements.input import Input

from src.services import ServiceContainer
from src.ui.state import AppState

logger = logging.getLogger(__name__)


@dataclass
class DownloadTask:
    """Tracks state of a single download."""

    model_id: str
    status: str = "queued"  # queued, downloading, completed, error
    progress: float = 0.0
    status_text: str = "Queued..."
    # UI elements (typed as Any to avoid NiceGUI import issues)
    card: Card | None = None
    progress_bar: object | None = None
    status_label: object | None = None
    cancel_requested: bool = False


class ModelsPage:
    """Models management page.

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
        from src.ui.pages.models._listing import build_available_section
        from src.ui.pages.models._operations import (
            build_comparison_section,
            build_installed_section,
        )

        with ui.column().classes("w-full gap-6 p-4"):
            # Header with VRAM info
            self._build_header()

            # Installed models
            build_installed_section(self)

            # Available models
            build_available_section(self)

            # Model comparison
            build_comparison_section(self)

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


__all__ = ["DownloadTask", "ModelsPage"]
