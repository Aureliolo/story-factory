"""Export service package - handles exporting stories to various formats.

This package provides the ExportService class composed from specialized mixins:
- ExportServiceBase: Core functionality, template management
- TextExportMixin: Markdown and plain text export
- EpubExportMixin: EPUB e-book export
- PdfExportMixin: PDF export
- DocxExportMixin: DOCX and HTML export
- IOExportMixin: File I/O operations
"""

from ._base import (
    EBOOK_TEMPLATE,
    EXPORT_TEMPLATES,
    MANUSCRIPT_TEMPLATE,
    WEB_SERIAL_TEMPLATE,
    ExportOptions,
    ExportServiceBase,
    ExportTemplate,
    _validate_export_path,
)
from ._docx import DocxExportMixin
from ._epub import EpubExportMixin
from ._io import IOExportMixin
from ._pdf import PdfExportMixin
from ._text import TextExportMixin


class ExportService(
    TextExportMixin,
    EpubExportMixin,
    PdfExportMixin,
    DocxExportMixin,
    IOExportMixin,
    ExportServiceBase,
):
    """Export stories to various formats.

    Supports markdown, plain text, HTML, EPUB, PDF, and DOCX export with customizable templates.

    Composed from:
    - ExportServiceBase: Core functionality and template management
    - TextExportMixin: to_markdown(), to_text()
    - EpubExportMixin: to_epub()
    - PdfExportMixin: to_pdf()
    - DocxExportMixin: to_html(), to_docx()
    - IOExportMixin: save_to_file()
    """

    pass


# Default export path for backward compatibility
DEFAULT_EXPORT_PATH = "output/"

__all__ = [
    "EBOOK_TEMPLATE",
    "EXPORT_TEMPLATES",
    "MANUSCRIPT_TEMPLATE",
    "WEB_SERIAL_TEMPLATE",
    "ExportOptions",
    "ExportService",
    "ExportTemplate",
    "_validate_export_path",
]
