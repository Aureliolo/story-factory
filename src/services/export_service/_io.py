"""File I/O mixin for ExportService."""

import logging
from pathlib import Path

from src.memory.story_state import StoryState
from src.utils.validation import (
    validate_not_none,
    validate_string_in_choices,
    validate_type,
)

from ._base import ExportOptions, ExportServiceBase, _validate_export_path

logger = logging.getLogger(__name__)


class IOExportMixin(ExportServiceBase):
    """Mixin providing file I/O operations for export."""

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
            filepath = _validate_export_path(filepath)
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
