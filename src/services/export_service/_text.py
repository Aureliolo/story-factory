"""Text-based export functions (markdown, plain text, HTML)."""

import html
import logging

from src.memory.story_state import StoryState
from src.utils.validation import validate_not_none, validate_type

from ._types import ExportOptions

logger = logging.getLogger("src.services.export_service._text")


def to_markdown(svc, state: StoryState) -> str:
    """Export story as markdown.

    Args:
        svc: ExportService instance.
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


def to_text(svc, state: StoryState) -> str:
    """Export story as plain text.

    Args:
        svc: ExportService instance.
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


def to_html(
    svc,
    state: StoryState,
    template: str | None = None,
    options: ExportOptions | None = None,
) -> str:
    """Export story as standalone HTML.

    Args:
        svc: ExportService instance.
        state: The story state to export.
        template: Template name to use. If None, uses ebook template.
        options: Custom export options. If None, uses template defaults.

    Returns:
        HTML formatted string.
    """
    logger.debug(f"Exporting story to HTML: {state.id}")

    # Get template and options
    tmpl = svc.get_template(template)
    opts = options or tmpl.options

    brief = state.brief
    title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")

    # Build default CSS with template options
    default_css = f"""
        body {{
            font-family: {opts.font_family};
            font-size: {opts.font_size}px;
            line-height: {opts.line_height};
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: {opts.chapter_spacing}em;
            margin-bottom: 1em;
        }}
        .meta {{
            color: #666;
            font-style: italic;
            margin-bottom: 30px;
        }}
        p {{
            margin-bottom: {opts.paragraph_spacing}em;
        }}
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{html.escape(title)}</title>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        default_css,
        opts.custom_css,  # Add custom CSS from template
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
            chapter_header = svc._format_chapter_header(chapter.number, chapter.title, opts)
            html_parts.append(f"<h2>{html.escape(chapter_header)}</h2>")

            for para in chapter.content.split("\n\n"):
                if para.strip():
                    safe_para = html.escape(para.strip())
                    html_parts.append(f"<p>{safe_para}</p>")

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)
