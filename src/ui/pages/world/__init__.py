"""World Builder page - entity and relationship management.

This package provides the WorldPage class for managing entities and relationships
in the story world. It is organized into mixins for maintainability:

- _page.py: Base class with __init__, build, and simple setup methods
- _generation.py: GenerationMixin with entity generation dialogs and methods
- _browser.py: BrowserMixin with entity browser methods
- _editor.py: EditorMixin with entity editor methods
- _graph.py: GraphMixin with graph interaction methods
- _analysis.py: AnalysisMixin with health and analysis methods
- _undo.py: UndoMixin with undo/redo methods
- _import.py: ImportMixin with import methods
"""

from src.services import ServiceContainer
from src.ui.pages.world._analysis import AnalysisMixin
from src.ui.pages.world._browser import BrowserMixin
from src.ui.pages.world._editor import EditorMixin
from src.ui.pages.world._generation import GenerationMixin
from src.ui.pages.world._graph import GraphMixin
from src.ui.pages.world._import import ImportMixin
from src.ui.pages.world._page import WorldPageBase
from src.ui.pages.world._undo import UndoMixin
from src.ui.state import AppState


class WorldPage(
    GenerationMixin,
    BrowserMixin,
    EditorMixin,
    GraphMixin,
    AnalysisMixin,
    UndoMixin,
    ImportMixin,
    WorldPageBase,
):
    """World Builder page for managing entities and relationships.

    This class composes all functionality from the various mixins:
    - WorldPageBase: Core initialization and build methods
    - GenerationMixin: Entity generation dialogs and methods
    - BrowserMixin: Entity browser with filtering and sorting
    - EditorMixin: Entity editor with version history
    - GraphMixin: Interactive graph visualization
    - AnalysisMixin: Health dashboard and analysis tools
    - UndoMixin: Undo/redo functionality
    - ImportMixin: Import entities from text

    Features:
    - Interactive graph visualization
    - Entity browser with filtering
    - Entity editor with version history
    - Relationship management
    - Graph analysis tools
    - World health dashboard
    - Import from text
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize world page.

        Args:
            state: Application state.
            services: Service container.
        """
        # Call the base class __init__ which sets up all common attributes
        super().__init__(state, services)


__all__ = ["WorldPage"]
