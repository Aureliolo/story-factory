"""Unit tests for prompt registry."""

import tempfile
from pathlib import Path

import pytest

from utils.prompt_registry import PromptRegistry
from utils.prompt_template import PromptTemplateError


class TestPromptRegistry:
    """Tests for PromptRegistry class."""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create a temporary templates directory with test templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)

            # Create writer templates
            writer_dir = templates_dir / "writer"
            writer_dir.mkdir()

            (writer_dir / "system.yaml").write_text(
                """
name: writer_system
version: "1.0"
description: System prompt for writer
agent: writer
task: system
is_system_prompt: true

template: |
  You are a writer.

variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            (writer_dir / "write_chapter.yaml").write_text(
                """
name: write_chapter
version: "1.0"
description: Write a chapter
agent: writer
task: write_chapter

template: |
  Write Chapter {{ chapter_number }}: {{ title }}
  Language: {{ language }}

variables:
  required:
    - chapter_number
    - title
    - language
  optional:
    - style
""",
                encoding="utf-8",
            )

            # Create editor templates
            editor_dir = templates_dir / "editor"
            editor_dir.mkdir()

            (editor_dir / "system.yaml").write_text(
                """
name: editor_system
version: "1.0"
description: System prompt for editor
agent: editor
task: system
is_system_prompt: true

template: |
  You are an editor.

variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            (editor_dir / "edit_chapter.yaml").write_text(
                """
name: edit_chapter
version: "1.0"
description: Edit a chapter
agent: editor
task: edit_chapter

template: |
  Edit this chapter:
  {{ content }}

variables:
  required:
    - content
  optional:
    - focus
""",
                encoding="utf-8",
            )

            yield templates_dir

    def test_load_all_templates(self, temp_templates_dir):
        """Test loading all templates from directory."""
        registry = PromptRegistry(temp_templates_dir)
        assert len(registry) == 4  # 2 writer + 2 editor

    def test_get_existing_template(self, temp_templates_dir):
        """Test getting an existing template."""
        registry = PromptRegistry(temp_templates_dir)
        template = registry.get("writer", "write_chapter")

        assert template is not None
        assert template.name == "write_chapter"
        assert template.agent == "writer"
        assert template.task == "write_chapter"

    def test_get_missing_template_raises(self, temp_templates_dir):
        """Test that getting a missing template raises an error."""
        registry = PromptRegistry(temp_templates_dir)

        with pytest.raises(PromptTemplateError) as exc_info:
            registry.get("nonexistent", "template")
        assert "not found" in str(exc_info.value)

    def test_get_system_prompt(self, temp_templates_dir):
        """Test getting system prompt for an agent."""
        registry = PromptRegistry(temp_templates_dir)
        template = registry.get_system("writer")

        assert template is not None
        assert template.is_system_prompt
        assert "You are a writer" in template.template

    def test_render_template(self, temp_templates_dir):
        """Test rendering a template through the registry."""
        registry = PromptRegistry(temp_templates_dir)
        result = registry.render(
            "writer",
            "write_chapter",
            chapter_number=1,
            title="The Beginning",
            language="English",
        )

        assert "Chapter 1" in result
        assert "The Beginning" in result
        assert "English" in result

    def test_render_system(self, temp_templates_dir):
        """Test rendering system prompt."""
        registry = PromptRegistry(temp_templates_dir)
        result = registry.render_system("editor")
        assert "You are an editor" in result

    def test_has_template(self, temp_templates_dir):
        """Test checking if template exists."""
        registry = PromptRegistry(temp_templates_dir)

        assert registry.has_template("writer", "write_chapter")
        assert registry.has_template("editor", "system")
        assert not registry.has_template("nonexistent", "template")

    def test_has_system(self, temp_templates_dir):
        """Test checking if system prompt exists."""
        registry = PromptRegistry(temp_templates_dir)

        assert registry.has_system("writer")
        assert registry.has_system("editor")
        assert not registry.has_system("nonexistent")

    def test_list_templates(self, temp_templates_dir):
        """Test listing all templates."""
        registry = PromptRegistry(temp_templates_dir)
        templates = registry.list_templates()

        assert len(templates) == 4
        assert "writer/system" in templates
        assert "writer/write_chapter" in templates
        assert "editor/system" in templates
        assert "editor/edit_chapter" in templates

    def test_list_agents(self, temp_templates_dir):
        """Test listing all agents."""
        registry = PromptRegistry(temp_templates_dir)
        agents = registry.list_agents()

        assert "writer" in agents
        assert "editor" in agents

    def test_list_tasks(self, temp_templates_dir):
        """Test listing tasks for an agent."""
        registry = PromptRegistry(temp_templates_dir)
        tasks = registry.list_tasks("writer")

        assert "system" in tasks
        assert "write_chapter" in tasks

    def test_get_hash(self, temp_templates_dir):
        """Test getting template hash."""
        registry = PromptRegistry(temp_templates_dir)
        hash_value = registry.get_hash("writer", "write_chapter")

        assert len(hash_value) == 32  # MD5 hex length
        # Same hash on repeated calls
        assert hash_value == registry.get_hash("writer", "write_chapter")

    def test_reload(self, temp_templates_dir):
        """Test reloading templates."""
        registry = PromptRegistry(temp_templates_dir)
        initial_count = len(registry)

        # Add a new template file
        (temp_templates_dir / "writer" / "new_task.yaml").write_text(
            """
name: new_task
version: "1.0"
description: New task
agent: writer
task: new_task

template: New template

variables:
  required: []
  optional: []
""",
            encoding="utf-8",
        )

        # Reload
        registry.reload()
        assert len(registry) == initial_count + 1
        assert registry.has_template("writer", "new_task")

    def test_get_template_info(self, temp_templates_dir):
        """Test getting template metadata."""
        registry = PromptRegistry(temp_templates_dir)
        info = registry.get_template_info("writer", "write_chapter")

        assert info["name"] == "write_chapter"
        assert info["version"] == "1.0"
        assert info["agent"] == "writer"
        assert info["task"] == "write_chapter"
        assert "chapter_number" in info["required_variables"]
        assert "hash" in info

    def test_empty_directory(self):
        """Test handling empty templates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptRegistry(Path(tmpdir))
            assert len(registry) == 0
            assert registry.list_templates() == []

    def test_nonexistent_directory(self, tmp_path):
        """Test handling nonexistent templates directory."""
        # Use a path inside tmp_path that doesn't exist
        nonexistent_path = tmp_path / "does_not_exist" / "nested" / "path"
        registry = PromptRegistry(nonexistent_path)
        assert len(registry) == 0

    def test_nested_templates(self):
        """Test loading templates from nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)

            # Create nested structure like world_quality/character/create.yaml
            nested_dir = templates_dir / "world_quality" / "character"
            nested_dir.mkdir(parents=True)

            (nested_dir / "create.yaml").write_text(
                """
name: character_create
version: "1.0"
description: Create character
agent: world_quality
task: character_create

template: Create a character

variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            registry = PromptRegistry(templates_dir)
            assert registry.has_template("world_quality", "character_create")

    def test_repr(self, temp_templates_dir):
        """Test string representation."""
        registry = PromptRegistry(temp_templates_dir)
        repr_str = repr(registry)
        assert "4 templates" in repr_str

    def test_duplicate_template_warning(self):
        """Test warning when loading duplicate templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)

            # Create first template
            writer_dir = templates_dir / "writer"
            writer_dir.mkdir()
            (writer_dir / "write_chapter.yaml").write_text(
                """
name: write_chapter
version: "1.0"
description: First write chapter
agent: writer
task: write_chapter
template: First template
variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            # Create second template with same agent/task but different filename
            (writer_dir / "write_chapter_v2.yaml").write_text(
                """
name: write_chapter_v2
version: "2.0"
description: Second write chapter
agent: writer
task: write_chapter
template: Second template
variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            # Should load both, second overwrites first
            registry = PromptRegistry(templates_dir)
            # Only one template should exist for the key
            assert registry.has_template("writer", "write_chapter")

    def test_invalid_template_file_skipped(self):
        """Test that invalid template files are skipped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)

            # Create valid template
            writer_dir = templates_dir / "writer"
            writer_dir.mkdir()
            (writer_dir / "system.yaml").write_text(
                """
name: system
version: "1.0"
description: System prompt
agent: writer
task: system
template: You are a writer
variables:
  required: []
  optional: []
""",
                encoding="utf-8",
            )

            # Create invalid template (missing required fields)
            (writer_dir / "invalid.yaml").write_text(
                """
name: ""
version: ""
agent: ""
task: ""
template: ""
""",
                encoding="utf-8",
            )

            # Should load valid template and skip invalid one
            registry = PromptRegistry(templates_dir)
            assert len(registry) == 1
            assert registry.has_template("writer", "system")


class TestPromptRegistryIntegration:
    """Integration tests using actual templates."""

    def test_load_actual_templates(self):
        """Test loading actual templates from the project."""
        # Use the actual templates directory
        templates_dir = Path(__file__).parent.parent.parent / "prompts" / "templates"

        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        registry = PromptRegistry(templates_dir)

        # Should have loaded some templates
        assert len(registry) > 0

        # Check some known templates exist
        if registry.has_template("writer", "system"):
            template = registry.get_system("writer")
            assert template.is_system_prompt

    def test_all_templates_valid(self):
        """Test that all actual templates are valid."""
        templates_dir = Path(__file__).parent.parent.parent / "prompts" / "templates"

        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        registry = PromptRegistry(templates_dir)

        for template_key in registry.list_templates():
            agent, task = template_key.split("/")
            template = registry.get(agent, task)

            # Validate template structure
            errors = template.validate()
            assert len(errors) == 0, f"Template {template_key} has errors: {errors}"
