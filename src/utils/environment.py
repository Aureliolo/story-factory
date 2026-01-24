"""Environment validation utilities.

This module checks that Python version and dependencies meet requirements
before the rest of the application loads. Import this module early in main.py.
"""

import sys
from pathlib import Path


def check_environment() -> None:
    """Check Python version and dependencies meet requirements.

    Reads Python version and dependencies from pyproject.toml,
    then validates the current environment meets all requirements.

    Raises:
        SystemExit: If Python version or dependencies are insufficient.
    """
    # Go up from utils/ to src/ to project root
    project_root = Path(__file__).parent.parent.parent

    _check_python_version(project_root)
    _check_dependencies(project_root)


def _check_python_version(project_root: Path) -> None:
    """Check Python version against pyproject.toml requirements."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return

    try:
        import tomllib

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        required_version_str = config.get("tool", {}).get("mypy", {}).get("python_version")
        if not required_version_str:
            return

        parts = required_version_str.split(".")
        required_major = int(parts[0])
        required_minor = int(parts[1]) if len(parts) > 1 else 0

        current_major = sys.version_info.major
        current_minor = sys.version_info.minor

        if (current_major, current_minor) < (required_major, required_minor):
            print(
                f"Error: Python {required_version_str}+ is required, "
                f"but you are running Python {current_major}.{current_minor}",
                file=sys.stderr,
            )
            print(
                f"Please upgrade Python or use a virtual environment with Python {required_version_str}+",
                file=sys.stderr,
            )
            sys.exit(1)

    except ImportError:
        print(
            f"Error: Python 3.11+ is required to parse pyproject.toml, "
            f"but you are running Python {sys.version_info.major}.{sys.version_info.minor}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (KeyError, ValueError, OSError) as e:
        print(f"Warning: Could not parse Python version from pyproject.toml: {e}", file=sys.stderr)


def _check_dependencies(project_root: Path) -> None:
    """Check installed packages against pyproject.toml dependencies."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return

    try:
        import tomllib
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        print("Warning: Cannot check dependencies (packaging not installed)", file=sys.stderr)
        return

    try:
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        project_section = config.get("project")
        if project_section is None:
            print("Warning: No [project] section in pyproject.toml", file=sys.stderr)
            return
        dependencies = project_section.get("dependencies")
        if dependencies is None:
            print("Warning: No dependencies in pyproject.toml [project] section", file=sys.stderr)
            return
    except (KeyError, ValueError, OSError) as e:
        print(f"Warning: Could not parse dependencies from pyproject.toml: {e}", file=sys.stderr)
        return

    missing = []
    version_mismatch = []

    for dep_string in dependencies:
        try:
            req = Requirement(dep_string)
        except InvalidRequirement:
            # Not a valid requirement string, skip
            continue

        package_name = req.name

        try:
            installed_version = get_version(package_name)
            if not req.specifier.contains(installed_version):
                version_mismatch.append(
                    f"  {package_name}: installed {installed_version}, requires {req.specifier}"
                )
        except PackageNotFoundError:
            missing.append(f"  {package_name}{req.specifier}")

    if missing or version_mismatch:
        print("Error: Missing or outdated dependencies:", file=sys.stderr)
        if missing:
            print("\nMissing packages:", file=sys.stderr)
            print("\n".join(missing), file=sys.stderr)
        if version_mismatch:
            print("\nOutdated packages:", file=sys.stderr)
            print("\n".join(version_mismatch), file=sys.stderr)
        print("\nRun: pip install .", file=sys.stderr)
        sys.exit(1)
