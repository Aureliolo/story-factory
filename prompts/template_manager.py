"""Prompt template management with versioning and few-shot examples."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.validation import validate_not_empty, validate_not_none

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A versioned prompt template with validation and examples.

    Templates separate prompts from code, enabling versioning, A/B testing,
    and easier maintenance.
    """

    version: str
    agent: str
    task: str
    system_prompt: str
    user_prompt_template: str
    variables: dict[str, str]  # Variable name -> type description
    validation: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> tuple[str, str]:
        """Render system and user prompts with provided variables.

        Args:
            **kwargs: Variable values to substitute in templates

        Returns:
            Tuple of (system_prompt, user_prompt)

        Raises:
            ValueError: If required variables are missing
        """
        # Validate all required variables provided
        missing = set(self.variables.keys()) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables for {self.agent}/{self.task}: {missing}")

        # Render user prompt with variables
        try:
            user_prompt = self.user_prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}") from e

        return self.system_prompt, user_prompt

    def add_examples(self, prompt: str, max_examples: int = 3) -> str:
        """Add few-shot examples to prompt.

        Args:
            prompt: The base prompt to enhance with examples
            max_examples: Maximum number of examples to include

        Returns:
            Enhanced prompt with examples prepended
        """
        if not self.examples:
            return prompt

        examples_text = "\n\nEXAMPLES OF DESIRED OUTPUT:\n"
        for i, ex in enumerate(self.examples[:max_examples], 1):
            examples_text += f"\n--- Example {i} ---\n"
            if "context" in ex:
                examples_text += f"Context: {ex['context']}\n"
            examples_text += f"Output:\n{ex['output']}\n"

        return examples_text + "\n" + prompt

    def get_validation_rules(self) -> dict[str, Any]:
        """Get validation rules for this template's outputs.

        Returns:
            Dictionary of validation parameters
        """
        return self.validation.copy()


class PromptTemplateManager:
    """Manages prompt templates with versioning and caching."""

    def __init__(self, templates_dir: Path):
        """Initialize the template manager.

        Args:
            templates_dir: Directory containing template YAML files
        """
        validate_not_none(templates_dir, "templates_dir")
        self.templates_dir = Path(templates_dir)
        self._cache: dict[str, PromptTemplate] = {}

        if not self.templates_dir.exists():
            logger.warning(
                f"Templates directory does not exist: {self.templates_dir}. "
                "Templates will not be available."
            )

    def load(self, agent: str, task: str, version: str = "latest") -> PromptTemplate:
        """Load a prompt template.

        Args:
            agent: Agent name (e.g., "writer", "editor")
            task: Task name (e.g., "write_chapter", "edit_prose")
            version: Template version or "latest" for most recent

        Returns:
            Loaded prompt template

        Raises:
            FileNotFoundError: If template not found
            ValueError: If template format is invalid
        """
        validate_not_empty(agent, "agent")
        validate_not_empty(task, "task")
        validate_not_empty(version, "version")

        # Check cache
        cache_key = f"{agent}/{task}/{version}"
        if cache_key in self._cache:
            logger.debug(f"Template cache hit: {cache_key}")
            return self._cache[cache_key]

        # Find template file
        agent_dir = self.templates_dir / agent
        if not agent_dir.exists():
            raise FileNotFoundError(f"No templates found for agent '{agent}' in {agent_dir}")

        pattern = f"{task}_v*.yaml"
        templates = list(agent_dir.glob(pattern))

        if not templates:
            raise FileNotFoundError(
                f"No templates found matching pattern '{pattern}' in {agent_dir}"
            )

        if version == "latest":
            # Get highest version number by parsing v1, v2, etc.
            def extract_version(path: Path) -> int:
                try:
                    version_str = path.stem.split("_v")[-1]
                    return int(version_str)
                except (ValueError, IndexError):
                    return 0

            template_file = max(templates, key=extract_version)
            logger.debug(f"Selected latest template: {template_file}")
        else:
            template_file = agent_dir / f"{task}_v{version}.yaml"
            if not template_file.exists():
                raise FileNotFoundError(f"Template not found: {template_file}")

        # Load and parse YAML
        try:
            with open(template_file) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in template {template_file}: {e}") from e

        # Validate required fields
        required_fields = {"version", "agent", "task", "system_prompt", "user_prompt_template"}
        missing = required_fields - set(data.keys())
        if missing:
            raise ValueError(f"Template {template_file} missing required fields: {missing}")

        # Create template instance
        template = PromptTemplate(
            version=str(data["version"]),
            agent=str(data["agent"]),
            task=str(data["task"]),
            system_prompt=str(data["system_prompt"]),
            user_prompt_template=str(data["user_prompt_template"]),
            variables=data.get("variables", {}),
            validation=data.get("validation", {}),
            examples=data.get("examples"),
            metadata=data.get("metadata", {}),
        )

        # Cache and return
        self._cache[cache_key] = template
        logger.info(f"Loaded template: {agent}/{task} v{template.version}")
        return template

    def list_templates(self, agent: str | None = None) -> list[dict[str, str]]:
        """List available templates.

        Args:
            agent: Optional agent name to filter by

        Returns:
            List of template metadata dictionaries
        """
        templates = []

        if agent:
            agent_dir = self.templates_dir / agent
            if not agent_dir.exists():
                return []
            search_dirs = [agent_dir]
        else:
            search_dirs = [d for d in self.templates_dir.iterdir() if d.is_dir()]

        for dir_path in search_dirs:
            for template_file in dir_path.glob("*_v*.yaml"):
                try:
                    with open(template_file) as f:
                        data = yaml.safe_load(f)
                    templates.append(
                        {
                            "agent": str(data.get("agent", dir_path.name)),
                            "task": str(data.get("task", template_file.stem.split("_v")[0])),
                            "version": str(data.get("version", "unknown")),
                            "file": str(template_file.relative_to(self.templates_dir)),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read template {template_file}: {e}")

        return templates

    def reload(self, agent: str, task: str, version: str = "latest"):
        """Reload a template from disk, bypassing cache.

        Args:
            agent: Agent name
            task: Task name
            version: Template version

        Returns:
            Reloaded template
        """
        cache_key = f"{agent}/{task}/{version}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        return self.load(agent, task, version)
