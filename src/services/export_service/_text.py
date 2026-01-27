"""Text and markdown export mixin for ExportService."""

import logging

from src.memory.story_state import StoryState
from src.utils.validation import validate_not_none, validate_type

from ._base import ExportServiceBase

logger = logging.getLogger(__name__)


class TextExportMixin(ExportServiceBase):
    """Mixin providing markdown and plain text export."""

    def to_markdown(self, state: StoryState) -> str:
        """Export story as markdown.

        Args:
            state: The story state to export.

        Returns:
            Markdown formatted string.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"Exporting story to markdown: {state.id}")
        brief = state.brief
        md_parts = []

        # Title and metadata
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        md_parts.append(f"# {title}\n")

        if brief:
            md_parts.append(f"*Genre: {brief.genre} | Tone: {brief.tone}*\n")
            md_parts.append(f"*Setting: {brief.setting_place}, {brief.setting_time}*\n")
            if brief.themes:
                md_parts.append(f"*Themes: {', '.join(brief.themes)}*\n")
            md_parts.append("\n---\n")

        # Chapters
        for chapter in state.chapters:
            if chapter.content:
                md_parts.append(f"\n## Chapter {chapter.number}: {chapter.title}\n\n")
                md_parts.append(chapter.content)
                md_parts.append("\n")

        return "\n".join(md_parts)

    def to_text(self, state: StoryState) -> str:
        """Export story as plain text.

        Args:
            state: The story state to export.

        Returns:
            Plain text formatted string.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"Exporting story to plain text: {state.id}")
        brief = state.brief
        text_parts = []

        # Title
        title = state.project_name or (brief.premise[:80] if brief else "Untitled Story")
        text_parts.append(title.upper())
        text_parts.append("=" * len(title))
        text_parts.append("")

        if brief:
            text_parts.append(f"Genre: {brief.genre} | Tone: {brief.tone}")
            text_parts.append(f"Setting: {brief.setting_place}, {brief.setting_time}")
            text_parts.append("")
            text_parts.append("-" * 60)
            text_parts.append("")

        # Chapters
        for chapter in state.chapters:
            if chapter.content:
                text_parts.append(f"CHAPTER {chapter.number}: {chapter.title.upper()}")
                text_parts.append("")
                text_parts.append(chapter.content)
                text_parts.append("")
                text_parts.append("-" * 40)
                text_parts.append("")

        return "\n".join(text_parts)
