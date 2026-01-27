"""PDF export function."""

import html
import logging
from io import BytesIO

from src.memory.story_state import StoryState
from src.utils.validation import validate_not_none, validate_type

from ._types import ExportOptions

logger = logging.getLogger("src.services.export_service._pdf")


def to_pdf(
    svc,
    state: StoryState,
    template: str | None = None,
    options: ExportOptions | None = None,
) -> bytes:
    """Export story as PDF.

    Args:
        svc: ExportService instance.
        state: The story state to export.
        template: Template name to use. If None, uses ebook template.
        options: Custom export options. If None, uses template defaults.

    Returns:
        PDF file as bytes.
    """
    validate_not_none(state, "state")
    validate_type(state, "state", StoryState)
    logger.info(f"Exporting story to PDF: {state.id}")
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    # Get template and options
    tmpl = svc.get_template(template)
    opts = options or tmpl.options

    buffer = BytesIO()

    # Use margin from options
    margin = opts.page_margin_inches * inch
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
    )

    # Calculate line spacing (leading = font_size * line_height)
    leading = opts.font_size * opts.line_height
    if opts.double_spaced:
        leading = opts.font_size * 2.0

    chapter_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading2"],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=opts.chapter_spacing * opts.font_size,
    )

    # Map font family to ReportLab font
    font_map = {
        "Courier New, monospace": "Courier",
        "Georgia, serif": "Times-Roman",
    }
    font_name = font_map.get(opts.font_family, "Times-Roman")

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=opts.font_size,
        leading=leading,
        spaceAfter=opts.paragraph_spacing * opts.font_size,
    )

    story_elements: list[Flowable] = []

    # Title page
    brief = state.brief
    title = state.project_name or (brief.premise[:50] if brief else "Untitled Story")
    story_elements.append(Paragraph(title, title_style))

    if brief:
        story_elements.append(Paragraph(f"<i>{brief.genre} | {brief.tone}</i>", styles["Normal"]))
        story_elements.append(Spacer(1, 0.5 * inch))

    story_elements.append(PageBreak())

    # Chapters
    for ch in state.chapters:
        if ch.content:
            chapter_header = svc._format_chapter_header(ch.number, ch.title, opts)
            story_elements.append(Paragraph(html.escape(chapter_header), chapter_style))

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
