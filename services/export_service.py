"""Export service - handles exporting stories to various formats."""

import html
import logging
from io import BytesIO
from pathlib import Path

from memory.story_state import StoryState
from settings import STORIES_DIR, Settings

logger = logging.getLogger(__name__)


def _validate_export_path(path: Path, base_dir: Path = STORIES_DIR.parent) -> Path:
    """Validate that an export path is safe to write to.

    Args:
        path: Path to validate
        base_dir: Base directory to constrain exports to (default: output/)

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path escapes base directory or is otherwise unsafe
    """
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()

        # Check if path is within base directory
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Export path {resolved} is outside allowed directory {base_resolved}"
            ) from None

        return resolved
    except Exception as e:
        raise ValueError(f"Invalid export path: {path}") from e


class ExportService:
    """Export stories to various formats.

    Supports markdown, plain text, EPUB, and PDF export.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize ExportService.

        Args:
            settings: Application settings. If None, loads from settings.json.
        """
        self.settings = settings or Settings.load()

    def to_markdown(self, state: StoryState) -> str:
        """Export story as markdown.

        Args:
            state: The story state to export.

        Returns:
            Markdown formatted string.
        """
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

    def to_epub(self, state: StoryState) -> bytes:
        """Export story as EPUB e-book.

        Args:
            state: The story state to export.

        Returns:
            EPUB file as bytes.
        """
        from ebooklib import epub

        book = epub.EpubBook()

        # Metadata
        brief = state.brief
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")

        # Language code mapping
        lang_map = {
            "English": "en",
            "German": "de",
            "Spanish": "es",
            "French": "fr",
            "Italian": "it",
            "Portuguese": "pt",
            "Dutch": "nl",
            "Russian": "ru",
            "Japanese": "ja",
            "Chinese": "zh",
            "Korean": "ko",
            "Arabic": "ar",
        }
        lang_code = lang_map.get(brief.language, "en") if brief else "en"

        book.set_identifier(state.id)
        book.set_title(title)
        book.set_language(lang_code)

        if brief:
            book.add_metadata("DC", "description", brief.premise)
            book.add_metadata("DC", "subject", brief.genre)

        # Create chapters
        chapters = []
        for ch in state.chapters:
            if ch.content:
                epub_chapter = epub.EpubHtml(
                    title=f"Chapter {ch.number}: {ch.title}",
                    file_name=f"chapter_{ch.number}.xhtml",
                    lang=lang_code,
                )

                # Convert to HTML (paragraph wrapping)
                paragraphs = ch.content.split("\n\n")
                html_paragraphs = [f"<p>{para}</p>" for para in paragraphs if para.strip()]
                html_content = "<br/><br/>".join(html_paragraphs)

                epub_chapter.content = f"""
                <html>
                <head><title>Chapter {ch.number}</title></head>
                <body>
                <h1>Chapter {ch.number}: {ch.title}</h1>
                {html_content}
                </body>
                </html>
                """

                book.add_item(epub_chapter)
                chapters.append(epub_chapter)

        # Navigation
        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Spine
        book.spine = ["nav"] + chapters

        # Write to bytes
        output = BytesIO()
        epub.write_epub(output, book)
        return output.getvalue()

    def to_pdf(self, state: StoryState) -> bytes:
        """Export story as PDF.

        Args:
            state: The story state to export.

        Returns:
            PDF file as bytes.
        """
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
        brief = state.brief
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        story_elements.append(Paragraph(title, title_style))

        if brief:
            story_elements.append(
                Paragraph(f"<i>{brief.genre} | {brief.tone}</i>", styles["Normal"])
            )
            story_elements.append(Spacer(1, 0.5 * inch))

        story_elements.append(PageBreak())

        # Chapters
        for ch in state.chapters:
            if ch.content:
                story_elements.append(Paragraph(f"Chapter {ch.number}: {ch.title}", chapter_style))

                # Split into paragraphs
                for para in ch.content.split("\n\n"):
                    if para.strip():
                        # Escape special characters using standard library
                        safe_para = html.escape(para.strip())
                        story_elements.append(Paragraph(safe_para, body_style))

                story_elements.append(PageBreak())

        doc.build(story_elements)
        return buffer.getvalue()

    def to_html(self, state: StoryState) -> str:
        """Export story as standalone HTML.

        Args:
            state: The story state to export.

        Returns:
            HTML formatted string.
        """
        brief = state.brief
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{html.escape(title)}</title>",
            "<meta charset='utf-8'>",
            "<style>",
            "body { font-family: Georgia, serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }",
            "h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }",
            "h2 { color: #555; margin-top: 40px; }",
            ".meta { color: #666; font-style: italic; margin-bottom: 30px; }",
            "p { text-align: justify; margin-bottom: 1em; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
        ]

        if brief:
            html_parts.append(
                f"<p class='meta'>Genre: {html.escape(brief.genre)} | Tone: {html.escape(brief.tone)}<br>"
                f"Setting: {html.escape(brief.setting_place)}, {html.escape(brief.setting_time)}</p>"
            )

        for chapter in state.chapters:
            if chapter.content:
                html_parts.append(
                    f"<h2>Chapter {chapter.number}: {html.escape(chapter.title)}</h2>"
                )
                for para in chapter.content.split("\n\n"):
                    if para.strip():
                        safe_para = html.escape(para.strip())
                        html_parts.append(f"<p>{safe_para}</p>")

        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def save_to_file(
        self,
        state: StoryState,
        format: str,
        filepath: str | Path,
    ) -> Path:
        """Save exported story to a file.

        Args:
            state: The story state to export.
            format: Export format ('markdown', 'text', 'epub', 'pdf', 'html').
            filepath: Output file path.

        Returns:
            Path where the file was saved.

        Raises:
            ValueError: If format is not supported or path is invalid.
        """
        filepath = Path(filepath)
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
            text_content = self.to_html(state)
            filepath.write_text(text_content, encoding="utf-8")
        elif format == "epub":
            bytes_content = self.to_epub(state)
            filepath.write_bytes(bytes_content)
        elif format == "pdf":
            bytes_content = self.to_pdf(state)
            filepath.write_bytes(bytes_content)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported story to {filepath} ({format} format)")
        return filepath

    def get_file_extension(self, format: str) -> str:
        """Get file extension for export format.

        Args:
            format: Export format name.

        Returns:
            File extension including dot.
        """
        extensions = {
            "markdown": ".md",
            "text": ".txt",
            "html": ".html",
            "epub": ".epub",
            "pdf": ".pdf",
        }
        return extensions.get(format, ".txt")
