"""YAML-based prompt template system with Jinja2 rendering."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError

from utils.exceptions import StoryFactoryError

logger = logging.getLogger(__name__)


class PromptTemplateError(StoryFactoryError):
    """Error related to prompt template operations."""


@dataclass
class PromptTemplate:
    """A YAML-based prompt template with Jinja2 rendering.

    Attributes:
        name: Unique template name (e.g., "write_chapter").
        version: Template version for tracking changes.
        description: Human-readable description of the template's purpose.
        agent: Agent role this template belongs to (e.g., "writer", "editor").
        task: Task identifier (e.g., "write_chapter", "edit_passage").
        template: Jinja2 template string.
        required_variables: List of variables that must be provided.
        optional_variables: List of variables that may be provided.
        is_system_prompt: Whether this is a system prompt template.
    """

    name: str
    version: str
    description: str
    agent: str
    task: str
    template: str
    required_variables: list[str] = field(default_factory=list)
    optional_variables: list[str] = field(default_factory=list)
    is_system_prompt: bool = False

    # Cached hash value
    _hash: str | None = field(default=None, repr=False, compare=False)

    # Jinja2 environment configured for strict variable checking
    _jinja_env: Environment = field(
        default_factory=lambda: Environment(undefined=StrictUndefined),
        repr=False,
        compare=False,
    )

    def render(self, **kwargs: Any) -> str:
        """Render template with variables using Jinja2.

        Args:
            **kwargs: Variables to substitute into the template.

        Returns:
            Rendered prompt string.

        Raises:
            PromptTemplateError: If required variables are missing or rendering fails.
        """
        # Check for missing required variables
        missing = set(self.required_variables) - set(kwargs.keys())
        if missing:
            raise PromptTemplateError(
                f"Missing required variables for template '{self.name}': {sorted(missing)}"
            )

        # Set defaults for optional variables
        for var in self.optional_variables:
            if var not in kwargs:
                kwargs[var] = None

        # Render template
        try:
            jinja_template = self._jinja_env.from_string(self.template)
            rendered = jinja_template.render(**kwargs)
            logger.debug(f"Rendered template '{self.name}' v{self.version} ({len(rendered)} chars)")
            return rendered
        except UndefinedError as e:
            raise PromptTemplateError(f"Undefined variable in template '{self.name}': {e}") from e
        except TemplateSyntaxError as e:
            raise PromptTemplateError(f"Syntax error in template '{self.name}': {e}") from e
        except Exception as e:
            raise PromptTemplateError(f"Render error in template '{self.name}': {e}") from e

    def get_hash(self) -> str:
        """Generate MD5 hash of template content for metrics tracking.

        The hash is computed from the template string and version,
        allowing tracking of which template version produced which results.

        Returns:
            MD5 hash string (32 hex characters).
        """
        if self._hash is None:
            content = f"{self.version}:{self.template}"
            self._hash = hashlib.md5(content.encode()).hexdigest()
        return self._hash

    def validate(self) -> list[str]:
        """Validate template structure and syntax.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check required fields
        if not self.name:
            errors.append("Template name is required")
        if not self.version:
            errors.append("Template version is required")
        if not self.agent:
            errors.append("Agent role is required")
        if not self.task:
            errors.append("Task name is required")
        if not self.template:
            errors.append("Template content is required")

        # Validate Jinja2 syntax
        try:
            self._jinja_env.parse(self.template)
        except TemplateSyntaxError as e:
            errors.append(f"Invalid Jinja2 syntax: {e}")

        return errors

    @classmethod
    def from_yaml(cls, path: Path) -> PromptTemplate:
        """Load template from YAML file.

        Expected YAML structure:
        ```yaml
        name: write_chapter
        version: "1.0"
        description: "Writes a complete chapter from outline"
        agent: writer
        task: write_chapter
        is_system_prompt: false

        template: |
          Write Chapter {{ chapter_number }}: "{{ chapter_title }}"
          ...

        variables:
          required:
            - chapter_number
            - chapter_title
          optional:
            - revision_feedback
        ```

        Args:
            path: Path to YAML file.

        Returns:
            PromptTemplate instance.

        Raises:
            PromptTemplateError: If file cannot be read or parsed.
        """
        if not path.exists():
            raise PromptTemplateError(f"Template file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PromptTemplateError(f"Invalid YAML in {path}: {e}") from e
        except OSError as e:
            raise PromptTemplateError(f"Cannot read template file {path}: {e}") from e

        if not isinstance(data, dict):
            raise PromptTemplateError(f"Invalid template format in {path}: expected dict")

        # Extract and validate variables section
        variables = data.get("variables", {})
        if not isinstance(variables, dict):
            raise PromptTemplateError(
                f"Invalid 'variables' in {path}: expected dict, got {type(variables).__name__}"
            )
        required_vars = variables.get("required", [])
        optional_vars = variables.get("optional", [])
        if not isinstance(required_vars, list):
            raise PromptTemplateError(f"Invalid 'variables.required' in {path}: expected list")
        if not isinstance(optional_vars, list):
            raise PromptTemplateError(f"Invalid 'variables.optional' in {path}: expected list")

        # Require version explicitly for proper metrics tracking
        if "version" not in data:
            raise PromptTemplateError(f"Missing required 'version' field in {path}")

        # Create template instance
        template = cls(
            name=data.get("name", path.stem),
            version=str(data["version"]),
            description=data.get("description", ""),
            agent=data.get("agent", ""),
            task=data.get("task", path.stem),
            template=data.get("template", ""),
            required_variables=required_vars,
            optional_variables=optional_vars,
            is_system_prompt=data.get("is_system_prompt", False),
        )

        # Validate template
        errors = template.validate()
        if errors:
            raise PromptTemplateError(f"Invalid template in {path}: {'; '.join(errors)}")

        logger.debug(f"Loaded template '{template.name}' v{template.version} from {path}")
        return template

    def to_yaml(self, path: Path) -> None:
        """Save template to YAML file.

        Args:
            path: Path to save YAML file.

        Raises:
            PromptTemplateError: If file cannot be written.
        """
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "agent": self.agent,
            "task": self.task,
            "is_system_prompt": self.is_system_prompt,
            "template": self.template,
            "variables": {
                "required": self.required_variables,
                "optional": self.optional_variables,
            },
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
                )
            logger.debug(f"Saved template '{self.name}' to {path}")
        except OSError as e:
            raise PromptTemplateError(f"Cannot write template file {path}: {e}") from e

    def __str__(self) -> str:
        """Return string representation."""
        return f"PromptTemplate({self.agent}/{self.task} v{self.version})"
