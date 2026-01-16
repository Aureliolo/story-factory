"""Export service - handles exporting stories to various formats."""

import html
import logging
import tempfile
from io import BytesIO
from pathlib import Path

from memory.story_state import StoryState
from settings import STORIES_DIR, Settings
from utils.constants import get_language_code

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


class ExportService:
    """Export stories to various formats.

    Supports markdown, plain text, HTML, EPUB, PDF, and DOCX export.
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

    def to_epub(self, state: StoryState) -> bytes:
        """Export story as EPUB e-book.

        Args:
            state: The story state to export.

        Returns:
            EPUB file as bytes.
        """
        logger.info(f"Exporting story to EPUB: {state.id}")
        from ebooklib import epub

        book = epub.EpubBook()

        # Metadata
        brief = state.brief
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        lang_code = get_language_code(brief.language) if brief else "en"

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
                # Escape title for HTML (XSS prevention)
                safe_title = html.escape(ch.title)

                epub_chapter = epub.EpubHtml(
                    title=f"Chapter {ch.number}: {safe_title}",
                    file_name=f"chapter_{ch.number}.xhtml",
                    lang=lang_code,
                )

                # Convert to HTML with escaped content (XSS prevention)
                paragraphs = ch.content.split("\n\n")
                html_paragraphs = [
                    f"<p>{html.escape(para)}</p>" for para in paragraphs if para.strip()
                ]
                html_content = "<br/><br/>".join(html_paragraphs)

                epub_chapter.content = f"""
                <html>
                <head><title>Chapter {ch.number}</title></head>
                <body>
                <h1>Chapter {ch.number}: {safe_title}</h1>
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
        logger.debug(f"EPUB export complete: {len(chapters)} chapters")
        return output.getvalue()

    def to_pdf(self, state: StoryState) -> bytes:
        """Export story as PDF.

        Args:
            state: The story state to export.

        Returns:
            PDF file as bytes.
        """
        logger.info(f"Exporting story to PDF: {state.id}")
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
        logger.debug("PDF export complete")
        return buffer.getvalue()

    def to_html(self, state: StoryState) -> str:
        """Export story as standalone HTML.

        Args:
            state: The story state to export.

        Returns:
            HTML formatted string.
        """
        logger.debug(f"Exporting story to HTML: {state.id}")
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

    def to_docx(self, state: StoryState) -> bytes:
        """Export story as DOCX (Microsoft Word).

        Args:
            state: The story state to export.

        Returns:
            DOCX file as bytes.
        """
        logger.info(f"Exporting story to DOCX: {state.id}")
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt

        doc = Document()

        # Set document margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        # Title
        brief = state.brief
        title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        title_para = doc.add_paragraph()
        title_run = title_para.add_run(title)
        title_run.font.size = Pt(24)
        title_run.bold = True
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Metadata
        if brief:
            meta_para = doc.add_paragraph()
            meta_run = meta_para.add_run(
                f"Genre: {brief.genre} | Tone: {brief.tone}\n"
                f"Setting: {brief.setting_place}, {brief.setting_time}"
            )
            meta_run.font.size = Pt(10)
            meta_run.italic = True
            meta_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_paragraph()  # Spacing

        # Chapters
        for chapter in state.chapters:
            if chapter.content:
                # Chapter heading
                chapter_heading = doc.add_paragraph()
                chapter_run = chapter_heading.add_run(f"Chapter {chapter.number}: {chapter.title}")
                chapter_run.font.size = Pt(18)
                chapter_run.bold = True

                # Chapter content
                for para in chapter.content.split("\n\n"):
                    if para.strip():
                        p = doc.add_paragraph(para.strip())
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                        p_format = p.paragraph_format
                        p_format.space_after = Pt(12)
                        p_format.line_spacing = 1.5

                # Page break after each chapter
                doc.add_page_break()

        # Write to bytes using context manager for proper cleanup
        with BytesIO() as output:
            doc.save(output)
            logger.debug("DOCX export complete")
            return output.getvalue()

    def save_to_file(
        self,
        state: StoryState,
        format: str,
        filepath: str | Path,
    ) -> Path:
        """Save exported story to a file.

        Args:
            state: The story state to export.
            format: Export format ('markdown', 'text', 'epub', 'pdf', 'html', 'docx').
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
        elif format == "docx":
            bytes_content = self.to_docx(state)
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
            "docx": ".docx",
        }
        return extensions.get(format, ".txt")
