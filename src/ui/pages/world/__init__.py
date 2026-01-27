"""World Builder page - entity and relationship management.

This package provides the WorldPage class for managing entities and relationships
in the story world. It is organized into mixins for maintainability:

- _page.py: Base class with __init__, build, and simple setup methods
- _generation.py: GenerationMixin with toolbar, quality settings, and utility methods
- _gen_dialogs.py: GenDialogsMixin with entity preview and generation dialogs
- _gen_operations.py: GenOperationsMixin with the core _generate_more method
- _gen_world_ops.py: GenWorldOpsMixin with rebuild, clear, relationship generation, mini descriptions
- _browser.py: BrowserMixin with entity browser methods
- _editor.py: EditorMixin with entity editor methods
- _editor_ops.py: EditorOpsMixin with entity CRUD operations and regeneration
- _graph.py: GraphMixin with graph interaction methods
- _analysis.py: AnalysisMixin with health and analysis methods
- _undo.py: UndoMixin with undo/redo methods
- _import.py: ImportMixin with import methods
"""

from src.services import ServiceContainer
from src.ui.pages.world._analysis import AnalysisMixin
from src.ui.pages.world._browser import BrowserMixin
from src.ui.pages.world._editor import EditorMixin
from src.ui.pages.world._editor_ops import EditorOpsMixin
from src.ui.pages.world._gen_dialogs import GenDialogsMixin
from src.ui.pages.world._gen_operations import GenOperationsMixin
from src.ui.pages.world._gen_world_ops import GenWorldOpsMixin
from src.ui.pages.world._generation import GenerationMixin
from src.ui.pages.world._graph import GraphMixin
from src.ui.pages.world._import import ImportMixin
from src.ui.pages.world._page import WorldPageBase
from src.ui.pages.world._undo import UndoMixin
from src.ui.state import AppState


class WorldPage(
    GenerationMixin,
    GenDialogsMixin,
    GenOperationsMixin,
    GenWorldOpsMixin,
    BrowserMixin,
    EditorMixin,
    EditorOpsMixin,
    GraphMixin,
    AnalysisMixin,
    UndoMixin,
    ImportMixin,
    WorldPageBase,
):
    """World Builder page for managing entities and relationships.

    This class composes all functionality from the various mixins:
    - WorldPageBase: Core initialization and build methods
    - GenerationMixin: Toolbar, quality settings, and utility methods
    - GenDialogsMixin: Entity preview and generation dialogs
    - GenOperationsMixin: Core _generate_more method
    - GenWorldOpsMixin: Rebuild, clear, relationship generation, mini descriptions
    - BrowserMixin: Entity browser with filtering and sorting
    - EditorMixin: Entity editor with version history
    - EditorOpsMixin: Entity CRUD operations and regeneration
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
