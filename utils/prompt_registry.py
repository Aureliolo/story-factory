"""Central registry for prompt templates."""

import logging
from pathlib import Path
from typing import Any

from utils.prompt_template import PromptTemplate, PromptTemplateError

logger = logging.getLogger(__name__)

# Default templates directory
DEFAULT_TEMPLATES_DIR = Path(__file__).parent.parent / "prompts" / "templates"


class PromptRegistry:
    """Central registry that loads all templates on startup.

    The registry provides:
    - Automatic discovery and loading of YAML templates
    - Template lookup by agent and task
    - System prompt retrieval
    - Template rendering with variable substitution

    Templates are organized in directories by agent:
    ```
    prompts/templates/
    ├── writer/
    │   ├── system.yaml
    │   ├── write_chapter.yaml
    │   └── write_scene.yaml
    ├── editor/
    │   ├── system.yaml
    │   └── edit_chapter.yaml
    └── world_quality/
        ├── character/
        │   ├── create.yaml
        │   └── judge.yaml
        └── shared/
            └── mini_description.yaml
    ```
    """

    def __init__(self, templates_dir: Path | str | None = None):
        """Initialize registry and load all templates.

        Args:
            templates_dir: Directory containing template YAML files.
                          Defaults to prompts/templates.
        """
        self.templates_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
        self._templates: dict[str, PromptTemplate] = {}
        self._load_all_templates()

    def _make_key(self, agent: str, task: str) -> str:
        """Create lookup key from agent and task."""
        return f"{agent}/{task}"

    def _load_all_templates(self) -> None:
        """Load all templates from the templates directory.

        Recursively scans for YAML files and loads each as a template.
        Templates are keyed by agent/task.
        """
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        yaml_files = list(self.templates_dir.rglob("*.yaml"))
        logger.info(f"Loading {len(yaml_files)} template files from {self.templates_dir}")

        loaded = 0
        errors = 0

        for yaml_file in yaml_files:
            try:
                template = PromptTemplate.from_yaml(yaml_file)
                key = self._make_key(template.agent, template.task)

                if key in self._templates:
                    logger.warning(f"Duplicate template key '{key}', overwriting with {yaml_file}")

                self._templates[key] = template
                loaded += 1

            except PromptTemplateError as e:
                logger.error(f"Failed to load template {yaml_file}: {e}")
                errors += 1

        logger.info(f"Loaded {loaded} templates successfully, {errors} errors")

    def get(self, agent: str, task: str) -> PromptTemplate:
        """Get a template by agent and task.

        Args:
            agent: Agent role (e.g., "writer", "editor").
            task: Task identifier (e.g., "write_chapter", "edit_passage").

        Returns:
            PromptTemplate instance.

        Raises:
            PromptTemplateError: If template not found.
        """
        key = self._make_key(agent, task)
        template = self._templates.get(key)

        if template is None:
            available = sorted(self._templates.keys())
            raise PromptTemplateError(
                f"Template not found: {key}. Available templates: {available[:10]}..."
            )

        return template

    def get_system(self, agent: str) -> PromptTemplate:
        """Get system prompt template for an agent.

        Convenience method for retrieving system prompts, which follow
        the convention of task="system".

        Args:
            agent: Agent role (e.g., "writer", "editor").

        Returns:
            PromptTemplate instance for the system prompt.

        Raises:
            PromptTemplateError: If system prompt not found.
        """
        return self.get(agent, "system")

    def render(self, agent: str, task: str, **kwargs: Any) -> str:
        """Render a template with variables.

        Convenience method that combines get() and render().

        Args:
            agent: Agent role.
            task: Task identifier.
            **kwargs: Variables to substitute into the template.

        Returns:
            Rendered prompt string.

        Raises:
            PromptTemplateError: If template not found or rendering fails.
        """
        template = self.get(agent, task)
        return template.render(**kwargs)

    def render_system(self, agent: str, **kwargs: Any) -> str:
        """Render system prompt template for an agent.

        Args:
            agent: Agent role.
            **kwargs: Variables to substitute (usually none for system prompts).

        Returns:
            Rendered system prompt string.

        Raises:
            PromptTemplateError: If system prompt not found or rendering fails.
        """
        return self.render(agent, "system", **kwargs)

    def has_template(self, agent: str, task: str) -> bool:
        """Check if a template exists.

        Args:
            agent: Agent role.
            task: Task identifier.

        Returns:
            True if template exists, False otherwise.
        """
        key = self._make_key(agent, task)
        return key in self._templates

    def has_system(self, agent: str) -> bool:
        """Check if an agent has a system prompt template.

        Args:
            agent: Agent role.

        Returns:
            True if system prompt exists, False otherwise.
        """
        return self.has_template(agent, "system")

    def get_hash(self, agent: str, task: str) -> str:
        """Get the hash of a template.

        Args:
            agent: Agent role.
            task: Task identifier.

        Returns:
            MD5 hash of the template content.

        Raises:
            PromptTemplateError: If template not found.
        """
        template = self.get(agent, task)
        return template.get_hash()

    def list_templates(self) -> list[str]:
        """List all loaded template keys.

        Returns:
            Sorted list of "agent/task" keys.
        """
        return sorted(self._templates.keys())

    def list_agents(self) -> list[str]:
        """List all agents with templates.

        Returns:
            Sorted list of unique agent names.
        """
        agents = {template.agent for template in self._templates.values()}
        return sorted(agents)

    def list_tasks(self, agent: str) -> list[str]:
        """List all tasks for a specific agent.

        Args:
            agent: Agent role.

        Returns:
            Sorted list of task names for the agent.
        """
        tasks = [template.task for template in self._templates.values() if template.agent == agent]
        return sorted(tasks)

    def reload(self) -> None:
        """Reload all templates from disk.

        Useful for development when templates are being edited.
        """
        logger.info("Reloading all templates")
        self._templates.clear()
        self._load_all_templates()

    def get_template_info(self, agent: str, task: str) -> dict[str, Any]:
        """Get metadata about a template.

        Args:
            agent: Agent role.
            task: Task identifier.

        Returns:
            Dictionary with template metadata.

        Raises:
            PromptTemplateError: If template not found.
        """
        template = self.get(agent, task)
        return {
            "name": template.name,
            "version": template.version,
            "description": template.description,
            "agent": template.agent,
            "task": template.task,
            "is_system_prompt": template.is_system_prompt,
            "required_variables": template.required_variables,
            "optional_variables": template.optional_variables,
            "hash": template.get_hash(),
        }

    def __len__(self) -> int:
        """Return number of loaded templates."""
        return len(self._templates)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromptRegistry({len(self._templates)} templates from {self.templates_dir})"
