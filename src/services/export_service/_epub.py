"""EPUB export mixin for ExportService."""

import html
import logging
from io import BytesIO

from src.memory.story_state import StoryState
from src.utils.constants import get_language_code
from src.utils.validation import validate_not_none, validate_type

from ._base import ExportOptions, ExportServiceBase

logger = logging.getLogger(__name__)


class EpubExportMixin(ExportServiceBase):
    """Mixin providing EPUB export."""

    def to_epub(
        self,
        state: StoryState,
        template: str | None = None,
        options: ExportOptions | None = None,
    ) -> bytes:
        """
        Export the story as an EPUB e-book.

        Uses the specified template (defaults to the built-in "ebook" template when None) and the provided options (defaults to the template's options when None). Includes brief metadata (title, language, description, subject) when available; converts each chapter with content into an individual XHTML item with escaped HTML and a bundled stylesheet.

        Parameters:
            state (StoryState): The story state to export.
            template (str | None): Template name to use; if None the "ebook" template is used.
            options (ExportOptions | None): Export options to override template defaults; if None the template's options are used.

        Returns:
            bytes: The EPUB file contents.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.info(f"Exporting story to EPUB: {state.id}")
        from ebooklib import epub

        # Get template and options
        tmpl = self.get_template(template)
        opts = options or tmpl.options

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

        # Create custom CSS for EPUB
        custom_style = f"""
            @namespace epub "http://www.idpf.org/2007/ops";

            body {{
                font-family: {opts.font_family};
                font-size: {opts.font_size}px;
                line-height: {opts.line_height};
                margin: 1em;
            }}

            h1 {{
                margin-top: {opts.chapter_spacing}em;
                margin-bottom: 1em;
                font-size: 1.5em;
            }}

            p {{
                margin-bottom: {opts.paragraph_spacing}em;
                text-align: justify;
            }}

            {opts.custom_css}
        """

        # Add CSS
        css = epub.EpubItem(
            uid="style",
            file_name="style.css",
            media_type="text/css",
            content=custom_style,
        )
        book.add_item(css)

        # Create chapters
        chapters = []
        for ch in state.chapters:
            if ch.content:
                # Format chapter header with options
                chapter_header = self._format_chapter_header(ch.number, ch.title, opts)
                safe_header = html.escape(chapter_header)

                epub_chapter = epub.EpubHtml(
                    title=chapter_header,
                    file_name=f"chapter_{ch.number}.xhtml",
                    lang=lang_code,
                )

                # Convert to HTML with escaped content (XSS prevention)
                paragraphs = ch.content.split("\n\n")
                html_paragraphs = [
                    f"<p>{html.escape(para)}</p>" for para in paragraphs if para.strip()
                ]
                html_content = "\n".join(html_paragraphs)

                # Use simple HTML structure without XML declaration (ebooklib handles that)
                epub_chapter.content = f"""
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{safe_header}</title>
    <link rel="stylesheet" href="style.css" type="text/css"/>
</head>
<body>
    <h1>{safe_header}</h1>
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
        book.spine = ["nav", *chapters]

        # Write to bytes
        output = BytesIO()
        epub.write_epub(output, book)
        logger.debug(f"EPUB export complete: {len(chapters)} chapters")
        return output.getvalue()
