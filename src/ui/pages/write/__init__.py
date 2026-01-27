"""Write Story page package.

This package splits the WritePage into focused mixins:
- WritePageBase: Core initialization, build method, notifications
- InterviewMixin: Interview phase UI and handlers
- StructureMixin: Fundamentals tab, world overview, story structure
- WritingMixin: Live writing display, chapter navigation, controls
- GenerationMixin: Chapter generation, feedback, learning recommendations
- VersionMixin: Version history display and rollback
- ExportMixin: Export formats and AI suggestions
"""

from src.services import ServiceContainer
from src.ui.state import AppState

from ._export_mixin import ExportMixin
from ._generation_mixin import GenerationMixin
from ._interview_mixin import InterviewMixin
from ._page import WritePageBase
from ._structure_mixin import StructureMixin
from ._version_mixin import VersionMixin
from ._writing_mixin import WritingMixin


class WritePage(
    ExportMixin,
    GenerationMixin,
    VersionMixin,
    WritingMixin,
    StructureMixin,
    InterviewMixin,
    WritePageBase,
):
    """Write Story page with Fundamentals and Live Writing tabs.

    Fundamentals tab:
    - Interview chat
    - World overview
    - Story structure
    - Reviews

    Live Writing tab:
    - Chapter navigator
    - Writing display with streaming
    - Feedback controls

    This class composes all mixins to provide full functionality.
    The MRO ensures proper method resolution:
    1. ExportMixin - export and suggestions
    2. GenerationMixin - chapter generation
    3. VersionMixin - version history
    4. WritingMixin - live writing display
    5. StructureMixin - fundamentals tab
    6. InterviewMixin - interview handling
    7. WritePageBase - core methods and __init__
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """
        Create a WritePage instance with all mixins initialized.

        Parameters:
            state: Application state used by the page.
            services: Service container providing access to all services.
        """
        super().__init__(state, services)


__all__ = ["WritePage"]
