"""Persistence and export functions for StoryOrchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from src.memory.story_state import StoryState
from src.settings import STORIES_DIR
from src.utils.constants import get_language_code

if TYPE_CHECKING:
    from . import StoryOrchestrator

logger = logging.getLogger(__name__)


def autosave(orc: StoryOrchestrator) -> str | None:
    """Auto-save current state with timestamp update.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The path where saved, or None if no story to save.
    """
    if not orc.story_state:
        return None

    try:
        orc.story_state.last_saved = datetime.now()
        orc.story_state.updated_at = datetime.now()
        path = orc.save_story()
        logger.debug(f"Autosaved story to {path}")
        return path
    except Exception as e:
        logger.warning(f"Autosave failed: {e}")
        return None


def save_story(orc: StoryOrchestrator, filepath: str | None = None) -> str:
    """Save the current story state to a JSON file.

    Args:
        orc: StoryOrchestrator instance.
        filepath: Optional custom path. Defaults to output/stories/<story_id>.json

    Returns:
        The path where the story was saved.
    """
    if not orc.story_state:
        raise ValueError("No story to save")

    # Update timestamps
    orc.story_state.updated_at = datetime.now()
    if not orc.story_state.last_saved:
        orc.story_state.last_saved = datetime.now()

    # Default save location
    output_path: Path
    if not filepath:
        output_dir = STORIES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{orc.story_state.id}.json"
    else:
        output_path = Path(filepath)

    # Convert to dict for JSON serialization
    story_data = orc.story_state.model_dump(mode="json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(story_data, f, indent=2, default=str)

    logger.info(f"Story saved to {output_path}")
    return str(output_path)


def load_story(orc: StoryOrchestrator, filepath: str) -> StoryState:
    """Load a story state from a JSON file.

    Args:
        orc: StoryOrchestrator instance.
        filepath: Path to the story JSON file.

    Returns:
        The loaded StoryState.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Story file not found: {path}")

    with open(path, encoding="utf-8") as f:
        story_data = json.load(f)

    orc.story_state = StoryState.model_validate(story_data)
    # Set correlation ID for event tracking
    orc._correlation_id = orc.story_state.id[:8]
    logger.info(f"Story loaded from {path}")
    return orc.story_state


def list_saved_stories() -> list[dict[str, str | None]]:
    """List all saved stories in the output directory.

    Returns:
        List of dicts with story metadata (id, path, created_at, status, etc.)
    """
    output_dir = STORIES_DIR
    stories: list[dict[str, str | None]] = []

    if not output_dir.exists():
        return stories

    for filepath in output_dir.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            stories.append(
                {
                    "id": data.get("id"),
                    "path": str(filepath),
                    "created_at": data.get("created_at"),
                    "status": data.get("status"),
                    "premise": (
                        data.get("brief", {}).get("premise", "")[:100] if data.get("brief") else ""
                    ),
                }
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not read story file {filepath}: {e}")

    return sorted(stories, key=lambda x: x.get("created_at") or "", reverse=True)


def get_full_story(orc: StoryOrchestrator) -> str:
    """Get the complete story text.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The full story as formatted text.
    """
    if not orc.story_state:
        raise ValueError("No story state available.")

    parts = []
    for chapter in orc.story_state.chapters:
        if chapter.content:
            parts.append(f"# Chapter {chapter.number}: {chapter.title}\n\n{chapter.content}")
    return "\n\n---\n\n".join(parts)


def export_to_markdown(orc: StoryOrchestrator) -> str:
    """Export the story as markdown.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The story in markdown format.
    """
    if not orc.story_state:
        raise ValueError("No story state available.")

    brief = orc.story_state.brief
    md_parts = []

    if brief:
        md_parts.extend(
            [
                f"# {brief.premise[:50]}...\n",
                f"*Genre: {brief.genre} | Tone: {brief.tone}*\n",
                f"*Setting: {brief.setting_place}, {brief.setting_time}*\n",
                "---\n",
            ]
        )
    else:
        md_parts.append("# Untitled Story\n\n---\n")

    for chapter in orc.story_state.chapters:
        if chapter.content:
            md_parts.append(f"\n## Chapter {chapter.number}: {chapter.title}\n\n")
            md_parts.append(chapter.content)

    return "\n".join(md_parts)


def export_to_text(orc: StoryOrchestrator) -> str:
    """Export the story as plain text.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The story in plain text format.
    """
    if not orc.story_state:
        raise ValueError("No story state available.")

    brief = orc.story_state.brief
    text_parts = []

    if brief:
        text_parts.extend(
            [
                brief.premise[:80],
                f"Genre: {brief.genre} | Tone: {brief.tone}",
                f"Setting: {brief.setting_place}, {brief.setting_time}",
                "=" * 60,
                "",
            ]
        )
    else:
        text_parts.extend(["Untitled Story", "=" * 60, ""])

    for chapter in orc.story_state.chapters:
        if chapter.content:
            text_parts.append(f"CHAPTER {chapter.number}: {chapter.title.upper()}")
            text_parts.append("")
            text_parts.append(chapter.content)
            text_parts.append("")
            text_parts.append("-" * 40)
            text_parts.append("")

    return "\n".join(text_parts)


def export_to_epub(orc: StoryOrchestrator) -> bytes:
    """Build an EPUB e-book from the current StoryState and return the file bytes.

    Uses the story's project name or brief to populate metadata (title, description,
    genre) and language, and includes each chapter that has content as an EPUB chapter
    in order.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The EPUB file contents as bytes.
    """
    if not orc.story_state:
        raise ValueError("No story state available.")

    from ebooklib import epub

    book = epub.EpubBook()

    # Set metadata
    brief = orc.story_state.brief
    title = orc.story_state.project_name or (brief.premise[:50] if brief else "Untitled Story")
    lang_code = get_language_code(brief.language) if brief else "en"
    book.set_identifier(orc.story_state.id)
    book.set_title(title)
    book.set_language(lang_code)

    if brief:
        book.add_metadata("DC", "description", brief.premise)
        book.add_metadata("DC", "subject", brief.genre)

    # Create chapters
    chapters = []
    for ch in orc.story_state.chapters:
        if ch.content:
            epub_chapter = epub.EpubHtml(
                title=f"Chapter {ch.number}: {ch.title}",
                file_name=f"chapter_{ch.number}.xhtml",
                lang=lang_code,
            )
            # Convert content to HTML (simple paragraph wrapping)
            html_content = "<br/><br/>".join(
                f"<p>{para}</p>" for para in ch.content.split("\n\n") if para.strip()
            )
            epub_chapter.content = f"<h1>Chapter {ch.number}: {ch.title}</h1>{html_content}"
            book.add_item(epub_chapter)
            chapters.append(epub_chapter)

    # Add navigation
    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define spine
    book.spine = ["nav", *chapters]

    # Write to bytes
    output = BytesIO()
    epub.write_epub(output, book)
    return output.getvalue()


def export_to_pdf(orc: StoryOrchestrator) -> bytes:
    """Export the story as PDF format.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        The PDF file contents as bytes.
    """
    if not orc.story_state:
        raise ValueError("No story state available.")

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
    )
    chapter_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading2"],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=30,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=12,
    )

    story_elements: list[Flowable] = []

    # Title page
    brief = orc.story_state.brief
    title = orc.story_state.project_name or (brief.premise[:50] if brief else "Untitled Story")
    story_elements.append(Paragraph(title, title_style))

    if brief:
        story_elements.append(Paragraph(f"<i>{brief.genre} | {brief.tone}</i>", styles["Normal"]))
        story_elements.append(Spacer(1, 0.5 * inch))

    story_elements.append(PageBreak())

    # Chapters
    for ch in orc.story_state.chapters:
        if ch.content:
            story_elements.append(Paragraph(f"Chapter {ch.number}: {ch.title}", chapter_style))

            # Split content into paragraphs
            for para in ch.content.split("\n\n"):
                if para.strip():
                    # Escape special characters for reportlab
                    safe_para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story_elements.append(Paragraph(safe_para, body_style))

            story_elements.append(PageBreak())

    doc.build(story_elements)
    return buffer.getvalue()


def export_story_to_file(
    orc: StoryOrchestrator, format: str = "markdown", filepath: str | None = None
) -> str:
    """Export the story to a file.

    Args:
        orc: StoryOrchestrator instance.
        format: Export format ('markdown', 'text', 'json', 'epub', 'pdf')
        filepath: Optional custom path. Defaults to output/stories/<story_id>.<ext>

    Returns:
        The path where the story was exported.
    """
    if not orc.story_state:
        raise ValueError("No story to export")

    # Determine file extension and content
    content: str | bytes
    if format == "markdown":
        ext = ".md"
        content = orc.export_to_markdown()
    elif format == "text":
        ext = ".txt"
        content = orc.export_to_text()
    elif format == "json":
        # JSON export is handled by save_story
        return orc.save_story(filepath)
    elif format == "epub":
        ext = ".epub"
        content = orc.export_to_epub()
    elif format == "pdf":
        ext = ".pdf"
        content = orc.export_to_pdf()
    else:
        raise ValueError(f"Unsupported export format: {format}")

    # Default export location
    output_path: Path
    if not filepath:
        output_dir = STORIES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{orc.story_state.id}{ext}"
    else:
        output_path = Path(filepath)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use isinstance for proper type narrowing that mypy understands
    if isinstance(content, bytes):
        with open(output_path, "wb") as f:
            f.write(content)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(f"Story exported to {output_path} ({format} format)")
    return str(output_path)
