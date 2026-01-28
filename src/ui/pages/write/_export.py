"""Export functions for story output."""

import logging
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from . import WritePage

logger = logging.getLogger(__name__)


async def export(page: WritePage, fmt: str) -> None:
    """Export the story in the given format.

    Args:
        page: The WritePage instance.
        fmt: Export format - one of "markdown", "text", "epub", "pdf".
    """
    if not page.state.project:
        return

    try:
        if fmt == "markdown":
            text_content = page.services.export.to_markdown(page.state.project)
            ui.download(text_content.encode(), f"{page.state.project.project_name}.md")
        elif fmt == "text":
            text_content = page.services.export.to_text(page.state.project)
            ui.download(text_content.encode(), f"{page.state.project.project_name}.txt")
        elif fmt == "epub":
            bytes_content = page.services.export.to_epub(page.state.project)
            ui.download(bytes_content, f"{page.state.project.project_name}.epub")
        elif fmt == "pdf":
            bytes_content = page.services.export.to_pdf(page.state.project)
            ui.download(bytes_content, f"{page.state.project.project_name}.pdf")

        page._notify(f"Exported as {fmt.upper()}", type="positive")
    except PermissionError:
        logger.exception(f"Permission denied exporting as {fmt}")
        page._notify("Export failed: Permission denied. Check file permissions.", type="negative")
    except OSError as e:
        logger.exception(f"OS error exporting as {fmt}")
        if "No space" in str(e) or "disk full" in str(e).lower():
            page._notify("Export failed: Insufficient disk space.", type="negative")
        else:
            page._notify(f"Export failed: {e}", type="negative")
    except ValueError as e:
        logger.exception(f"Value error exporting as {fmt}")
        page._notify(f"Export failed: {e}. Check chapter content.", type="negative")
    except Exception as e:
        logger.exception(f"Failed to export as {fmt}")
        page._notify(f"Export failed: {e}", type="negative")
