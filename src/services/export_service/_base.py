"""Base classes and templates for export service.

Contains ExportOptions, ExportTemplate, built-in templates, and validation.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import src.settings as _settings
from src.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Formatting options for exports."""

    # Font settings
    font_family: str = "Georgia, serif"
    font_size: int = 12  # in points for PDF/DOCX, pixels for HTML/EPUB
    line_height: float = 1.6

    # Spacing settings
    paragraph_spacing: float = 1.0  # em units
    chapter_spacing: float = 2.0  # em units

    # Chapter header settings
    chapter_number_format: str = "Chapter {number}"  # e.g., "Chapter 1" or "Ch. 1"
    include_chapter_numbers: bool = True
    chapter_separator: str = ": "  # separator between number and title

    # Custom CSS (for HTML/EPUB)
    custom_css: str = ""

    # Page settings (for PDF/DOCX)
    page_margin_inches: float = 1.0
    double_spaced: bool = False  # Override line_height to 2.0 if True


@dataclass
class ExportTemplate:
    """Template configuration for exports."""

    name: str
    description: str
    options: ExportOptions


# Built-in export templates
MANUSCRIPT_TEMPLATE = ExportTemplate(
    name="manuscript",
    description="Professional manuscript format - Courier New, 12pt, double-spaced, 1-inch margins",
    options=ExportOptions(
        font_family="Courier New, monospace",
        font_size=12,
        line_height=2.0,
        double_spaced=True,
        paragraph_spacing=0.0,  # No extra spacing in manuscript format
        chapter_spacing=3.0,
        page_margin_inches=1.0,
        chapter_number_format="Chapter {number}",
        include_chapter_numbers=True,
        chapter_separator=": ",
    ),
)

EBOOK_TEMPLATE = ExportTemplate(
    name="ebook",
    description="Reader-friendly ebook format - Georgia, readable spacing, clean headers",
    options=ExportOptions(
        font_family="Georgia, serif",
        font_size=14,
        line_height=1.8,
        double_spaced=False,
        paragraph_spacing=1.2,
        chapter_spacing=2.5,
        page_margin_inches=0.75,
        chapter_number_format="Chapter {number}",
        include_chapter_numbers=True,
        chapter_separator=": ",
        custom_css="""
            body { text-align: justify; }
            h2 { page-break-before: always; }
            p { margin-bottom: 1.2em; text-indent: 1.5em; }
            p:first-of-type { text-indent: 0; }
        """,
    ),
)

WEB_SERIAL_TEMPLATE = ExportTemplate(
    name="web_serial",
    description="Modern web serial format - clean, responsive, optimized for online reading",
    options=ExportOptions(
        font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        font_size=16,
        line_height=1.7,
        double_spaced=False,
        paragraph_spacing=1.0,
        chapter_spacing=2.0,
        page_margin_inches=1.0,
        chapter_number_format="Chapter {number}",
        include_chapter_numbers=True,
        chapter_separator=" - ",
        custom_css="""
            body {
                max-width: 650px;
                margin: 0 auto;
                padding: 20px;
                background: #fafafa;
                color: #333;
            }
            h1 {
                font-size: 2.5em;
                border-bottom: 3px solid #333;
                padding-bottom: 15px;
                margin-bottom: 30px;
            }
            h2 {
                font-size: 1.8em;
                margin-top: 50px;
                margin-bottom: 25px;
                color: #222;
            }
            .meta {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 40px;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 5px;
            }
            p {
                margin-bottom: 1em;
                line-height: 1.7;
            }
            @media (prefers-color-scheme: dark) {
                body { background: #1a1a1a; color: #e0e0e0; }
                h1 { border-bottom-color: #e0e0e0; }
                h2 { color: #f0f0f0; }
                .meta { background: #2a2a2a; color: #b0b0b0; }
            }
        """,
    ),
)

# Template registry
EXPORT_TEMPLATES: dict[str, ExportTemplate] = {
    "manuscript": MANUSCRIPT_TEMPLATE,
    "ebook": EBOOK_TEMPLATE,
    "web_serial": WEB_SERIAL_TEMPLATE,
}


def _validate_export_path(path: Path, base_dir: Path | None = None) -> Path:
    """Validate that an export path is safe to write to.

    Args:
        path: Path to validate
        base_dir: Base directory to constrain exports to (default: output/)

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path escapes base directory or is otherwise unsafe
    """
    if base_dir is None:
        base_dir = _settings.STORIES_DIR.parent
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()

        # Allow system temp directory for testing (cross-platform)
        temp_dir = Path(tempfile.gettempdir()).resolve()
        if resolved.is_relative_to(temp_dir):
            return resolved

        # Check if path is within base directory
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Export path {resolved} is outside allowed directory {base_resolved}"
            ) from None

        return resolved
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid export path: {path}") from e


class ExportServiceBase:
    """Base class for export service with template management."""

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
        return extensions.get(format, ".txt")
