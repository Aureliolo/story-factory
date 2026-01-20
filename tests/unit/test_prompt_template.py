"""Unit tests for prompt template system."""

import tempfile
from pathlib import Path

import pytest

from utils.prompt_template import PromptTemplate, PromptTemplateError


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_create_basic_template(self):
        """Test creating a basic template."""
        template = PromptTemplate(
            name="test_template",
            version="1.0",
            description="Test template",
            agent="writer",
            task="write_test",
            template="Hello {{ name }}!",
            required_variables=["name"],
            optional_variables=[],
        )
        assert template.name == "test_template"
        assert template.version == "1.0"
        assert template.agent == "writer"
        assert template.task == "write_test"

    def test_render_with_required_vars(self):
        """Test rendering template with required variables."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Write about {{ topic }} in {{ language }}.",
            required_variables=["topic", "language"],
        )
        result = template.render(topic="dragons", language="English")
        assert result == "Write about dragons in English."

    def test_render_with_optional_vars(self):
        """Test rendering template with optional variables."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Topic: {{ topic }}{% if subtitle %} - {{ subtitle }}{% endif %}",
            required_variables=["topic"],
            optional_variables=["subtitle"],
        )
        # Without optional var
        result1 = template.render(topic="dragons")
        assert result1 == "Topic: dragons"

        # With optional var
        result2 = template.render(topic="dragons", subtitle="fire breathers")
        assert result2 == "Topic: dragons - fire breathers"

    def test_render_missing_required_raises(self):
        """Test that missing required variables raise an error."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }} from {{ place }}!",
            required_variables=["name", "place"],
        )
        with pytest.raises(PromptTemplateError) as exc_info:
            template.render(name="Alice")
        assert "place" in str(exc_info.value)

    def test_hash_consistency(self):
        """Test that hash is consistent for same template."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }}!",
            required_variables=["name"],
        )
        hash1 = template.get_hash()
        hash2 = template.get_hash()
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_hash_changes_with_template(self):
        """Test that hash changes when template content changes."""
        template1 = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }}!",
            required_variables=["name"],
        )
        template2 = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Goodbye {{ name }}!",
            required_variables=["name"],
        )
        assert template1.get_hash() != template2.get_hash()

    def test_hash_changes_with_version(self):
        """Test that hash changes when version changes."""
        template1 = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }}!",
            required_variables=["name"],
        )
        template2 = PromptTemplate(
            name="test",
            version="2.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }}!",
            required_variables=["name"],
        )
        assert template1.get_hash() != template2.get_hash()

    def test_jinja2_conditionals(self):
        """Test Jinja2 conditional logic."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="{% if verbose %}Verbose: {% endif %}{{ message }}",
            required_variables=["message"],
            optional_variables=["verbose"],
        )
        result1 = template.render(message="Hello", verbose=True)
        assert result1 == "Verbose: Hello"

        result2 = template.render(message="Hello", verbose=False)
        assert result2 == "Hello"

    def test_jinja2_loops(self):
        """Test Jinja2 loop logic."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Items: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}",
            required_variables=["items"],
        )
        result = template.render(items=["apple", "banana", "cherry"])
        assert result == "Items: apple, banana, cherry"

    def test_jinja2_filters(self):
        """Test Jinja2 filter support."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Items: {{ items | join(', ') }}",
            required_variables=["items"],
        )
        result = template.render(items=["a", "b", "c"])
        assert result == "Items: a, b, c"

    def test_validate_success(self):
        """Test validation passes for valid template."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }}!",
            required_variables=["name"],
        )
        errors = template.validate()
        assert len(errors) == 0

    def test_validate_missing_fields(self):
        """Test validation catches missing required fields."""
        template = PromptTemplate(
            name="",
            version="",
            description="Test",
            agent="",
            task="",
            template="",
            required_variables=[],
        )
        errors = template.validate()
        assert len(errors) >= 4  # name, version, agent, task, template

    def test_validate_invalid_jinja_syntax(self):
        """Test validation catches invalid Jinja2 syntax."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="test",
            template="Hello {{ name }",  # Missing closing }}
            required_variables=["name"],
        )
        errors = template.validate()
        assert any("syntax" in e.lower() for e in errors)

    def test_from_yaml(self):
        """Test loading template from YAML file."""
        yaml_content = """
name: test_yaml
version: "1.0"
description: Test YAML template
agent: writer
task: test_yaml
is_system_prompt: false

template: |
  Write about {{ topic }}.
  Language: {{ language }}

variables:
  required:
    - topic
    - language
  optional:
    - style
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            template = PromptTemplate.from_yaml(yaml_path)
            assert template.name == "test_yaml"
            assert template.version == "1.0"
            assert template.agent == "writer"
            assert template.task == "test_yaml"
            assert "topic" in template.required_variables
            assert "language" in template.required_variables
            assert "style" in template.optional_variables
            assert not template.is_system_prompt

            # Test rendering
            result = template.render(topic="cats", language="English")
            assert "cats" in result
            assert "English" in result
        finally:
            yaml_path.unlink()

    def test_from_yaml_not_found(self):
        """Test error when YAML file not found."""
        with pytest.raises(PromptTemplateError) as exc_info:
            PromptTemplate.from_yaml(Path("/nonexistent/path.yaml"))
        assert "not found" in str(exc_info.value)

    def test_from_yaml_invalid_yaml(self):
        """Test error when YAML is invalid."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = Path(f.name)

        try:
            with pytest.raises(PromptTemplateError) as exc_info:
                PromptTemplate.from_yaml(yaml_path)
            assert "YAML" in str(exc_info.value)
        finally:
            yaml_path.unlink()

    def test_to_yaml(self):
        """Test saving template to YAML file."""
        template = PromptTemplate(
            name="save_test",
            version="2.0",
            description="Test saving",
            agent="editor",
            task="save_test",
            template="Edit: {{ content }}",
            required_variables=["content"],
            optional_variables=["style"],
            is_system_prompt=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_template.yaml"
            template.to_yaml(yaml_path)

            # Verify file exists
            assert yaml_path.exists()

            # Load and verify
            loaded = PromptTemplate.from_yaml(yaml_path)
            assert loaded.name == template.name
            assert loaded.version == template.version
            assert loaded.agent == template.agent
            assert loaded.task == template.task
            assert loaded.required_variables == template.required_variables

    def test_str_representation(self):
        """Test string representation."""
        template = PromptTemplate(
            name="test",
            version="1.0",
            description="Test",
            agent="writer",
            task="write_chapter",
            template="Hello",
            required_variables=[],
        )
        str_repr = str(template)
        assert "writer" in str_repr
        assert "write_chapter" in str_repr
        assert "1.0" in str_repr
