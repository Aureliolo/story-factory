"""Tests for prompt template management system."""


import pytest

from prompts.template_manager import PromptTemplate, PromptTemplateManager


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_render_with_valid_variables(self):
        """Should render prompts successfully with all variables provided."""
        template = PromptTemplate(
            version="1.0",
            agent="writer",
            task="test",
            system_prompt="You are a writer.",
            user_prompt_template="Write chapter {number}: {title}",
            variables={"number": "int", "title": "str"},
        )

        system, user = template.render(number=1, title="The Beginning")

        assert system == "You are a writer."
        assert user == "Write chapter 1: The Beginning"

    def test_render_raises_on_missing_variables(self):
        """Should raise ValueError when required variables are missing."""
        template = PromptTemplate(
            version="1.0",
            agent="writer",
            task="test",
            system_prompt="System",
            user_prompt_template="Write {title}",
            variables={"title": "str"},
        )

        with pytest.raises(ValueError, match="Missing required variables"):
            template.render()

    def test_add_examples_with_examples(self):
        """Should add examples to prompt."""
        template = PromptTemplate(
            version="1.0",
            agent="writer",
            task="test",
            system_prompt="System",
            user_prompt_template="Write",
            variables={},
            examples=[
                {"context": "Test context", "output": "Test output 1"},
                {"output": "Test output 2"},
            ],
        )

        result = template.add_examples("Write this", max_examples=2)

        assert "EXAMPLES OF DESIRED OUTPUT" in result
        assert "Test context" in result
        assert "Test output 1" in result
        assert "Test output 2" in result
        assert "Write this" in result

    def test_add_examples_with_no_examples(self):
        """Should return prompt unchanged when no examples exist."""
        template = PromptTemplate(
            version="1.0",
            agent="writer",
            task="test",
            system_prompt="System",
            user_prompt_template="Write",
            variables={},
            examples=None,
        )

        result = template.add_examples("Write this")
        assert result == "Write this"

    def test_get_validation_rules(self):
        """Should return validation rules."""
        validation = {"min_length": 100, "max_length": 1000}
        template = PromptTemplate(
            version="1.0",
            agent="writer",
            task="test",
            system_prompt="System",
            user_prompt_template="Write",
            variables={},
            validation=validation,
        )

        rules = template.get_validation_rules()
        assert rules == validation
        # Should return a copy
        rules["min_length"] = 200
        assert template.validation["min_length"] == 100


class TestPromptTemplateManager:
    """Tests for PromptTemplateManager class."""

    @pytest.fixture
    def template_dir(self, tmp_path):
        """Create a temporary templates directory with test templates."""
        templates_dir = tmp_path / "templates"
        writer_dir = templates_dir / "writer"
        writer_dir.mkdir(parents=True)

        # Create test template v1
        template_v1 = writer_dir / "write_chapter_v1.yaml"
        template_v1.write_text("""
version: "1.0"
agent: "writer"
task: "write_chapter"
system_prompt: "You are a writer v1."
user_prompt_template: "Write chapter {number}"
variables:
  number: "int"
validation:
  min_length: 100
""")

        # Create test template v2
        template_v2 = writer_dir / "write_chapter_v2.yaml"
        template_v2.write_text("""
version: "2.0"
agent: "writer"
task: "write_chapter"
system_prompt: "You are a writer v2."
user_prompt_template: "Write chapter {number}: {title}"
variables:
  number: "int"
  title: "str"
""")

        return templates_dir

    def test_load_latest_version(self, template_dir):
        """Should load the latest version when version='latest'."""
        manager = PromptTemplateManager(template_dir)

        template = manager.load("writer", "write_chapter", version="latest")

        assert template.version == "2.0"
        assert "v2" in template.system_prompt

    def test_load_specific_version(self, template_dir):
        """Should load specific version when requested."""
        manager = PromptTemplateManager(template_dir)

        template = manager.load("writer", "write_chapter", version="1")

        assert template.version == "1.0"
        assert "v1" in template.system_prompt

    def test_load_caches_templates(self, template_dir):
        """Should cache loaded templates."""
        manager = PromptTemplateManager(template_dir)

        template1 = manager.load("writer", "write_chapter")
        template2 = manager.load("writer", "write_chapter")

        # Should be the same object (cached)
        assert template1 is template2

    def test_load_raises_on_missing_agent(self, template_dir):
        """Should raise FileNotFoundError for missing agent directory."""
        manager = PromptTemplateManager(template_dir)

        with pytest.raises(FileNotFoundError, match="No templates found for agent"):
            manager.load("nonexistent", "task")

    def test_load_raises_on_missing_task(self, template_dir):
        """Should raise FileNotFoundError for missing task template."""
        manager = PromptTemplateManager(template_dir)

        with pytest.raises(FileNotFoundError, match="No templates found matching"):
            manager.load("writer", "nonexistent_task")

    def test_load_raises_on_invalid_yaml(self, tmp_path):
        """Should raise ValueError for invalid YAML."""
        templates_dir = tmp_path / "templates"
        writer_dir = templates_dir / "writer"
        writer_dir.mkdir(parents=True)

        template_file = writer_dir / "bad_v1.yaml"
        template_file.write_text("invalid: yaml: content: [")

        manager = PromptTemplateManager(templates_dir)

        with pytest.raises(ValueError, match="Invalid YAML"):
            manager.load("writer", "bad")

    def test_load_raises_on_missing_required_fields(self, tmp_path):
        """Should raise ValueError when required fields are missing."""
        templates_dir = tmp_path / "templates"
        writer_dir = templates_dir / "writer"
        writer_dir.mkdir(parents=True)

        template_file = writer_dir / "incomplete_v1.yaml"
        template_file.write_text("""
version: "1.0"
agent: "writer"
# Missing task, system_prompt, user_prompt_template
""")

        manager = PromptTemplateManager(templates_dir)

        with pytest.raises(ValueError, match="missing required fields"):
            manager.load("writer", "incomplete")

    def test_list_templates_all(self, template_dir):
        """Should list all available templates."""
        manager = PromptTemplateManager(template_dir)

        templates = manager.list_templates()

        assert len(templates) == 2
        assert all(t["agent"] == "writer" for t in templates)
        assert all(t["task"] == "write_chapter" for t in templates)

    def test_list_templates_by_agent(self, template_dir):
        """Should filter templates by agent."""
        manager = PromptTemplateManager(template_dir)

        templates = manager.list_templates(agent="writer")

        assert len(templates) == 2
        assert all(t["agent"] == "writer" for t in templates)

    def test_list_templates_empty_agent(self, template_dir):
        """Should return empty list for agent with no templates."""
        manager = PromptTemplateManager(template_dir)

        templates = manager.list_templates(agent="nonexistent")

        assert templates == []

    def test_reload_bypasses_cache(self, template_dir):
        """Should reload template from disk, ignoring cache."""
        manager = PromptTemplateManager(template_dir)

        # Load and cache
        template1 = manager.load("writer", "write_chapter", version="1")

        # Reload
        template2 = manager.reload("writer", "write_chapter", version="1")

        # Should be different objects (reloaded)
        assert template1 is not template2
        assert template1.version == template2.version

    def test_nonexistent_templates_dir(self, tmp_path):
        """Should handle nonexistent templates directory gracefully."""
        manager = PromptTemplateManager(tmp_path / "nonexistent")

        # Should not raise on init
        assert manager.templates_dir.name == "nonexistent"

        # Should raise on load
        with pytest.raises(FileNotFoundError):
            manager.load("writer", "task")
