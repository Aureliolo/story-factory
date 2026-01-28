"""Export service - handles exporting stories to various formats."""

import logging
from pathlib import Path

from src.memory.story_state import StoryState
from src.settings import Settings
from src.utils.validation import (
    validate_not_none,
    validate_string_in_choices,
    validate_type,
)

from ._docx import to_docx as _to_docx
from ._epub import to_epub as _to_epub
from ._pdf import to_pdf as _to_pdf
from ._text import to_html as _to_html
from ._text import to_markdown as _to_markdown
from ._text import to_text as _to_text
from ._types import (
    EBOOK_TEMPLATE,
    EXPORT_TEMPLATES,
    MANUSCRIPT_TEMPLATE,
    WEB_SERIAL_TEMPLATE,
    ExportOptions,
    ExportTemplate,
    validate_export_path,
)

logger = logging.getLogger(__name__)


class ExportService:
    """Export stories to various formats.

    Supports markdown, plain text, HTML, EPUB, PDF, and DOCX export with customizable templates.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize ExportService.

        Args:
            settings: Application settings. If None, loads from src/settings.json.
        """
        logger.debug("Initializing ExportService")
        self.settings = settings or Settings.load()
        logger.debug("ExportService initialized successfully")

    def get_template(self, template_name: str | None = None) -> ExportTemplate:
        """Get export template by name.

        Args:
            template_name: Name of the template. If None, returns ebook template.

        Returns:
            ExportTemplate configuration.

        Raises:
            ValueError: If template name is not found.
        """
        if template_name is None:
            template_name = "ebook"
            logger.debug("No template specified, using default 'ebook' template")

        if template_name not in EXPORT_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available templates: {list(EXPORT_TEMPLATES.keys())}"
            )

        logger.debug(f"Retrieved export template: {template_name}")
        return EXPORT_TEMPLATES[template_name]

    def _format_chapter_header(
        self, chapter_number: int, chapter_title: str, options: ExportOptions
    ) -> str:
        """Format chapter header according to options.

        Args:
            chapter_number: Chapter number.
            chapter_title: Chapter title.
            options: Export options.

        Returns:
            Formatted chapter header.
        """
        if not options.include_chapter_numbers:
            logger.debug(f"Formatting chapter header without number: {chapter_title}")
            return chapter_title

        number_part = options.chapter_number_format.format(number=chapter_number)
        header = f"{number_part}{options.chapter_separator}{chapter_title}"
        logger.debug(f"Formatted chapter header: {header}")
        return header

    def to_markdown(self, state: StoryState) -> str:
        """Export story as markdown.

        Args:
            state: The story state to export.

        Returns:
            Markdown formatted string.
        """
        return _to_markdown(self, state)

    def to_text(self, state: StoryState) -> str:
        """Export story as plain text.

        Args:
            state: The story state to export.

        Returns:
            Plain text formatted string.
        """
        return _to_text(self, state)

    def to_epub(
        self,
        state: StoryState,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export the story as an EPUB e-book.

        Args:
            state: The story state to export.
            template: Template name to use; if None the "ebook" template is used.
            options: Export options to override template defaults; if None the template's
                options are used.

        Returns:
            The EPUB file contents as bytes.
        """
        return _to_epub(self, state, template, options)

    def to_pdf(
        self,
        state: StoryState,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export story as PDF.

        Args:
            state: The story state to export.
            template: Template name to use. If None, uses ebook template.
            options: Custom export options. If None, uses template defaults.

        Returns:
            PDF file as bytes.
        """
        return _to_pdf(self, state, template, options)

    def to_html(
        self,
        state: StoryState,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> str:
        """Export story as standalone HTML.

        Args:
            state: The story state to export.
            template: Template name to use. If None, uses ebook template.
            options: Custom export options. If None, uses template defaults.

        Returns:
            HTML formatted string.
        """
        return _to_html(self, state, template, options)

    def to_docx(
        self,
        state: StoryState,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export story as DOCX (Microsoft Word).

        Args:
            state: The story state to export.
            template: Template name to use. If None, uses ebook template.
            options: Custom export options. If None, uses template defaults.

        Returns:
            DOCX file as bytes.
        """
        return _to_docx(self, state, template, options)

    def save_to_file(
        self,
        state: StoryState,
        format: str,
        filepath: str | Path,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> Path:
        """Save exported story to a file.

        Args:
            state: The story state to export.
            format: Export format ('markdown', 'text', 'epub', 'pdf', 'html', 'docx').
            filepath: Output file path.
            template: Template name to use. If None, uses ebook template.
            options: Custom export options. If None, uses template defaults.

        Returns:
            Path where the file was saved.

        Raises:
            ValueError: If format is not supported or path is invalid.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_string_in_choices(
            format,
            "format",
            ["markdown", "text", "epub", "pdf", "html", "docx"],
        )
        validate_not_none(filepath, "filepath")
        logger.debug(
            f"save_to_file called: story_id={state.id}, format={format}, filepath={filepath}"
        )
        filepath = Path(filepath)

        try:
            # Validate export path
            filepath = validate_export_path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format == "markdown":
                text_content = self.to_markdown(state)
                filepath.write_text(text_content, encoding="utf-8")
            elif format == "text":
                text_content = self.to_text(state)
                filepath.write_text(text_content, encoding="utf-8")
            elif format == "html":
                text_content = self.to_html(state, template, options)
                filepath.write_text(text_content, encoding="utf-8")
            elif format == "epub":
                bytes_content = self.to_epub(state, template, options)
                filepath.write_bytes(bytes_content)
            elif format == "pdf":
                bytes_content = self.to_pdf(state, template, options)
                filepath.write_bytes(bytes_content)
            elif format == "docx":
                bytes_content = self.to_docx(state, template, options)
                filepath.write_bytes(bytes_content)
            # Note: No else branch needed - validate_string_in_choices ensures only valid formats reach here

            logger.info(f"Exported story to {filepath} ({format} format)")
            return filepath
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"save_to_file failed for story {state.id} to {filepath}: {e}", exc_info=True
            )
            raise

    def get_file_extension(self, format: str) -> str:
        """Get file extension for export format.

        Args:
            format: Export format name.

        Returns:
            File extension including dot.
        """
        logger.debug(f"get_file_extension called: format={format}")
        extensions = {
            "markdown": ".md",
            "text": ".txt",
            "html": ".html",
            "epub": ".epub",
            "pdf": ".pdf",
            "docx": ".docx",
        }
        if format not in extensions:
            raise ValueError(
                f"Unknown export format '{format}'. Supported formats: {list(extensions.keys())}"
            )
        return extensions[format]


__all__ = [
    "EBOOK_TEMPLATE",
    "EXPORT_TEMPLATES",
    "MANUSCRIPT_TEMPLATE",
    "WEB_SERIAL_TEMPLATE",
    "ExportOptions",
    "ExportService",
    "ExportTemplate",
    "validate_export_path",
]
