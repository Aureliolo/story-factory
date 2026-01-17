"""Unit tests for export service templates and formatting options."""

import pytest

from memory.story_state import Chapter, StoryBrief, StoryState
from services.export_service import (
    EXPORT_TEMPLATES,
    ExportOptions,
    ExportService,
    ExportTemplate,
)
from settings import Settings


class TestExportTemplates:
    """Test export template registry and configuration."""

    def test_template_registry_has_all_templates(self):
        """Test that template registry contains all expected templates."""
        assert "manuscript" in EXPORT_TEMPLATES
        assert "ebook" in EXPORT_TEMPLATES
        assert "web_serial" in EXPORT_TEMPLATES

    def test_manuscript_template_configuration(self):
        """Test manuscript template has correct configuration."""
        template = EXPORT_TEMPLATES["manuscript"]
        assert template.name == "manuscript"
        assert "Courier" in template.options.font_family
        assert template.options.font_size == 12
        assert template.options.double_spaced is True
        assert template.options.line_height == 2.0
        assert template.options.page_margin_inches == 1.0

    def test_ebook_template_configuration(self):
        """Test ebook template has correct configuration."""
        template = EXPORT_TEMPLATES["ebook"]
        assert template.name == "ebook"
        assert "Georgia" in template.options.font_family
        assert template.options.font_size == 14
        assert template.options.double_spaced is False
        assert template.options.custom_css != ""

    def test_web_serial_template_configuration(self):
        """Test web serial template has correct configuration."""
        template = EXPORT_TEMPLATES["web_serial"]
        assert template.name == "web_serial"
        assert template.options.font_size == 16
        assert template.options.custom_css != ""
        assert "max-width" in template.options.custom_css


class TestExportServiceTemplates:
    """Test ExportService template methods."""

    def test_get_template_with_valid_name(self):
        """Test getting template by name."""
        service = ExportService(Settings())
        template = service.get_template("manuscript")
        assert template.name == "manuscript"
        assert isinstance(template, ExportTemplate)

    def test_get_template_with_none_returns_ebook(self):
        """Test that None template name returns ebook template."""
        service = ExportService(Settings())
        template = service.get_template(None)
        assert template.name == "ebook"

    def test_get_template_with_invalid_name_raises_error(self):
        """Test that invalid template name raises ValueError."""
        service = ExportService(Settings())
        with pytest.raises(ValueError, match="Unknown template"):
            service.get_template("nonexistent")

    def test_format_chapter_header_with_numbers(self):
        """Test chapter header formatting with numbers."""
        service = ExportService(Settings())
        options = ExportOptions(
            include_chapter_numbers=True,
            chapter_number_format="Chapter {number}",
            chapter_separator=": ",
        )
        result = service._format_chapter_header(1, "The Beginning", options)
        assert result == "Chapter 1: The Beginning"

    def test_format_chapter_header_without_numbers(self):
        """Test chapter header formatting without numbers."""
        service = ExportService(Settings())
        options = ExportOptions(include_chapter_numbers=False)
        result = service._format_chapter_header(1, "The Beginning", options)
        assert result == "The Beginning"

    def test_format_chapter_header_custom_format(self):
        """Test chapter header with custom format."""
        service = ExportService(Settings())
        options = ExportOptions(
            include_chapter_numbers=True,
            chapter_number_format="Ch. {number}",
            chapter_separator=" - ",
        )
        result = service._format_chapter_header(5, "Epic Battle", options)
        assert result == "Ch. 5 - Epic Battle"


class TestHTMLExportWithTemplates:
    """Test HTML export with templates."""

    def test_html_export_with_manuscript_template(self):
        """Test HTML export using manuscript template."""
        service = ExportService(Settings())
        state = _create_test_state("html-manuscript")

        html = service.to_html(state, template="manuscript")

        assert "<!DOCTYPE html>" in html
        assert "Courier New" in html
        assert "font-size: 12px" in html
        assert "line-height: 2.0" in html

    def test_html_export_with_ebook_template(self):
        """Test HTML export using ebook template."""
        service = ExportService(Settings())
        state = _create_test_state("html-ebook")

        html = service.to_html(state, template="ebook")

        assert "<!DOCTYPE html>" in html
        assert "Georgia" in html
        assert "font-size: 14px" in html
        assert "text-indent" in html  # Custom CSS from ebook template

    def test_html_export_with_web_serial_template(self):
        """Test HTML export using web serial template."""
        service = ExportService(Settings())
        state = _create_test_state("html-web")

        html = service.to_html(state, template="web_serial")

        assert "<!DOCTYPE html>" in html
        assert "font-size: 16px" in html
        assert "max-width: 650px" in html  # Web serial custom CSS
        assert "prefers-color-scheme: dark" in html  # Dark mode support

    def test_html_export_with_custom_options(self):
        """Test HTML export with custom options overriding template."""
        service = ExportService(Settings())
        state = _create_test_state("html-custom")

        custom_options = ExportOptions(
            font_family="Arial, sans-serif",
            font_size=18,
            line_height=2.5,
            custom_css="body { background: red; }",
        )

        html = service.to_html(state, options=custom_options)

        assert "Arial, sans-serif" in html
        assert "font-size: 18px" in html
        assert "line-height: 2.5" in html
        assert "background: red" in html

    def test_html_export_includes_viewport_meta(self):
        """Test HTML export includes viewport meta tag."""
        service = ExportService(Settings())
        state = _create_test_state("html-viewport")

        html = service.to_html(state)

        assert "viewport" in html
        assert "width=device-width" in html


class TestEPUBExportWithTemplates:
    """Test EPUB export with templates."""

    def test_epub_export_with_manuscript_template(self):
        """Test EPUB export using manuscript template."""
        service = ExportService(Settings())
        state = _create_test_state("epub-manuscript")

        epub_bytes = service.to_epub(state, template="manuscript")

        assert isinstance(epub_bytes, bytes)
        assert len(epub_bytes) > 0

    def test_epub_export_with_ebook_template(self):
        """Test EPUB export using ebook template."""
        service = ExportService(Settings())
        state = _create_test_state("epub-ebook")

        epub_bytes = service.to_epub(state, template="ebook")

        assert isinstance(epub_bytes, bytes)
        assert len(epub_bytes) > 0

    def test_epub_export_with_custom_css(self):
        """Test EPUB export with custom CSS."""
        service = ExportService(Settings())
        state = _create_test_state("epub-css")

        custom_options = ExportOptions(custom_css="p { color: blue; font-style: italic; }")

        epub_bytes = service.to_epub(state, options=custom_options)

        assert isinstance(epub_bytes, bytes)
        assert len(epub_bytes) > 0


class TestPDFExportWithTemplates:
    """Test PDF export with templates."""

    def test_pdf_export_with_manuscript_template(self):
        """Test PDF export using manuscript template."""
        service = ExportService(Settings())
        state = _create_test_state("pdf-manuscript")

        pdf_bytes = service.to_pdf(state, template="manuscript")

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")  # PDF magic number

    def test_pdf_export_with_ebook_template(self):
        """Test PDF export using ebook template."""
        service = ExportService(Settings())
        state = _create_test_state("pdf-ebook")

        pdf_bytes = service.to_pdf(state, template="ebook")

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_export_with_custom_margins(self):
        """Test PDF export with custom margin settings."""
        service = ExportService(Settings())
        state = _create_test_state("pdf-margins")

        custom_options = ExportOptions(page_margin_inches=0.5)

        pdf_bytes = service.to_pdf(state, options=custom_options)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0


class TestDOCXExportWithTemplates:
    """Test DOCX export with templates."""

    def test_docx_export_with_manuscript_template(self):
        """Test DOCX export using manuscript template."""
        service = ExportService(Settings())
        state = _create_test_state("docx-manuscript")

        docx_bytes = service.to_docx(state, template="manuscript")

        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0
        assert docx_bytes[:4] == b"PK\x03\x04"  # ZIP magic number (DOCX is ZIP)

    def test_docx_export_with_ebook_template(self):
        """Test DOCX export using ebook template."""
        service = ExportService(Settings())
        state = _create_test_state("docx-ebook")

        docx_bytes = service.to_docx(state, template="ebook")

        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0

    def test_docx_export_with_custom_spacing(self):
        """Test DOCX export with custom spacing settings."""
        service = ExportService(Settings())
        state = _create_test_state("docx-spacing")

        custom_options = ExportOptions(
            paragraph_spacing=2.0,
            line_height=1.8,
        )

        docx_bytes = service.to_docx(state, options=custom_options)

        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0


class TestSaveToFileWithTemplates:
    """Test save_to_file with templates."""

    def test_save_html_with_template(self, tmp_path):
        """Test saving HTML with template."""
        service = ExportService(Settings())
        state = _create_test_state("save-html")
        output_file = tmp_path / "story.html"

        result = service.save_to_file(state, "html", output_file, template="manuscript")

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Courier New" in content

    def test_save_epub_with_template(self, tmp_path):
        """Test saving EPUB with template."""
        service = ExportService(Settings())
        state = _create_test_state("save-epub")
        output_file = tmp_path / "story.epub"

        result = service.save_to_file(state, "epub", output_file, template="ebook")

        assert result.exists()
        assert result.stat().st_size > 0

    def test_save_pdf_with_custom_options(self, tmp_path):
        """Test saving PDF with custom options."""
        service = ExportService(Settings())
        state = _create_test_state("save-pdf")
        output_file = tmp_path / "story.pdf"

        custom_options = ExportOptions(
            font_size=14,
            double_spaced=True,
        )

        result = service.save_to_file(state, "pdf", output_file, options=custom_options)

        assert result.exists()
        assert result.stat().st_size > 0

    def test_save_docx_with_template(self, tmp_path):
        """Test saving DOCX with template."""
        service = ExportService(Settings())
        state = _create_test_state("save-docx")
        output_file = tmp_path / "story.docx"

        result = service.save_to_file(state, "docx", output_file, template="web_serial")

        assert result.exists()
        assert result.stat().st_size > 0


class TestChapterHeaderFormatting:
    """Test chapter header formatting across all export formats."""

    def test_chapter_headers_in_html(self):
        """Test custom chapter header formatting in HTML."""
        service = ExportService(Settings())
        state = _create_test_state("header-html")

        options = ExportOptions(
            chapter_number_format="Part {number}",
            chapter_separator=" | ",
        )

        html = service.to_html(state, options=options)
        assert "Part 1 | Chapter One" in html

    def test_chapter_headers_without_numbers_in_html(self):
        """Test chapter headers without numbers in HTML."""
        service = ExportService(Settings())
        state = _create_test_state("no-numbers-html")

        options = ExportOptions(include_chapter_numbers=False)

        html = service.to_html(state, options=options)
        assert "<h2>Chapter One</h2>" in html
        assert "Chapter 1" not in html


def _create_test_state(state_id: str) -> StoryState:
    """Helper function to create test story state."""
    brief = StoryBrief(
        premise="Test story premise",
        genre="Fantasy",
        tone="Epic",
        setting_time="Medieval",
        setting_place="Kingdom",
        target_length="short_story",
        content_rating="none",
    )
    state = StoryState(id=state_id, project_name="Test Story", brief=brief)
    state.chapters = [
        Chapter(
            number=1,
            title="Chapter One",
            outline="Outline",
            content="Chapter content here.\n\nSecond paragraph of content.",
        )
    ]
    return state
