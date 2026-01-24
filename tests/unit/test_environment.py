"""Tests for the environment validation module."""

import sys
from unittest.mock import patch

import pytest

from src.utils.environment import (
    _check_dependencies,
    _check_python_version,
    check_environment,
)


class TestCheckEnvironment:
    """Tests for check_environment function."""

    def test_check_environment_calls_both_checks(self, tmp_path, monkeypatch):
        """Test that check_environment calls both validation functions."""
        with (
            patch("src.utils.environment._check_python_version") as mock_python,
            patch("src.utils.environment._check_dependencies") as mock_deps,
        ):
            check_environment()

            mock_python.assert_called_once()
            mock_deps.assert_called_once()


class TestCheckPythonVersion:
    """Tests for _check_python_version function."""

    def test_skips_when_no_pyproject(self, tmp_path):
        """Test that check is skipped when pyproject.toml doesn't exist."""
        # Should not raise
        _check_python_version(tmp_path)

    def test_skips_when_no_python_version_in_config(self, tmp_path):
        """Test that check is skipped when python_version not in config."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.mypy]\nstrict = true\n")

        # Should not raise
        _check_python_version(tmp_path)

    def test_passes_when_version_sufficient(self, tmp_path):
        """Test that check passes when Python version is sufficient."""
        pyproject = tmp_path / "pyproject.toml"
        # Use a version lower than current
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        pyproject.write_text(f'[tool.mypy]\npython_version = "{current}"\n')

        # Should not raise
        _check_python_version(tmp_path)

    def test_exits_when_version_insufficient(self, tmp_path, capsys):
        """Test that check exits when Python version is too low."""
        pyproject = tmp_path / "pyproject.toml"
        # Use a version higher than any current Python
        pyproject.write_text('[tool.mypy]\npython_version = "99.0"\n')

        with pytest.raises(SystemExit) as exc_info:
            _check_python_version(tmp_path)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Python 99.0+ is required" in captured.err

    def test_handles_tomllib_import_error(self, tmp_path, capsys, monkeypatch):
        """Test that check handles tomllib ImportError gracefully."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.mypy]\npython_version = "3.14"\n')

        # Mock tomllib import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Raise ImportError for tomllib while allowing other imports."""
            if name == "tomllib":
                raise ImportError("No module named 'tomllib'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(SystemExit) as exc_info:
            _check_python_version(tmp_path)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Python 3.11+ is required to parse pyproject.toml" in captured.err

    def test_warns_on_parse_error(self, tmp_path, capsys):
        """Test that check warns on parse errors but doesn't exit."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid toml {{{")

        # Should not raise (warns but continues)
        _check_python_version(tmp_path)

        captured = capsys.readouterr()
        assert "Warning: Could not parse Python version" in captured.err

    def test_handles_version_with_only_major(self, tmp_path):
        """Test that check handles version with only major number."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.mypy]\npython_version = "3"\n')

        # Should not raise (3.0 is less than current)
        _check_python_version(tmp_path)


class TestCheckDependencies:
    """Tests for _check_dependencies function."""

    def test_skips_when_no_pyproject(self, tmp_path):
        """Test that check is skipped when pyproject.toml doesn't exist."""
        # Should not raise
        _check_dependencies(tmp_path)

    def test_skips_empty_dependencies(self, tmp_path):
        """Test that check skips when dependencies list is empty."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\ndependencies = []\n")

        # Should not raise
        _check_dependencies(tmp_path)

    def test_warns_when_no_project_section(self, tmp_path, capsys):
        """Test that check warns when [project] section is missing."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.ruff]\nline-length = 100\n")

        # Should not raise (warns but continues)
        _check_dependencies(tmp_path)

        captured = capsys.readouterr()
        assert "No [project] section" in captured.err

    def test_warns_when_no_dependencies_key(self, tmp_path, capsys):
        """Test that check warns when dependencies key is missing."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')

        # Should not raise (warns but continues)
        _check_dependencies(tmp_path)

        captured = capsys.readouterr()
        assert "No dependencies in pyproject.toml" in captured.err

    def test_passes_when_all_dependencies_installed(self, tmp_path):
        """Test that check passes when all dependencies are installed."""
        pyproject = tmp_path / "pyproject.toml"
        # Use a package that's definitely installed (pytest) with >= specifier
        pyproject.write_text('[project]\ndependencies = ["pytest>=9.0.0"]\n')

        # Should not raise - dependency is satisfied
        _check_dependencies(tmp_path)

    def test_exits_when_dependency_missing(self, tmp_path, capsys):
        """Test that check exits when a dependency is missing."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["nonexistent-package-12345==1.0.0"]\n')

        with pytest.raises(SystemExit) as exc_info:
            _check_dependencies(tmp_path)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Missing packages" in captured.err
        assert "nonexistent-package-12345" in captured.err

    def test_exits_when_dependency_outdated(self, tmp_path, capsys, monkeypatch):
        """Test that check exits when a dependency is outdated."""
        pyproject = tmp_path / "pyproject.toml"
        # Require a version higher than installed
        pyproject.write_text('[project]\ndependencies = ["pytest==999.0.0"]\n')

        with pytest.raises(SystemExit) as exc_info:
            _check_dependencies(tmp_path)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Outdated packages" in captured.err or "Missing packages" in captured.err

    def test_handles_packaging_import_error(self, tmp_path, capsys, monkeypatch):
        """Test that check handles packaging ImportError gracefully."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["pytest>=9.0.0"]\n')

        # Mock packaging import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Raise ImportError for packaging.requirements while allowing other imports."""
            if name == "packaging.requirements":
                raise ImportError("No module named 'packaging'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Should not raise (warns but continues)
        _check_dependencies(tmp_path)

        captured = capsys.readouterr()
        assert "Cannot check dependencies" in captured.err

    def test_handles_various_version_specifiers(self, tmp_path):
        """Test that check handles various version specifiers correctly."""
        pyproject = tmp_path / "pyproject.toml"
        # Use installed packages with version specifiers they satisfy
        pyproject.write_text('[project]\ndependencies = ["pytest>=9.0.0", "pytest-cov>=7.0.0"]\n')

        # Should not raise (installed versions satisfy these specifiers)
        _check_dependencies(tmp_path)

    def test_shows_pip_install_hint(self, tmp_path, capsys):
        """Test that check shows pip install hint on failure."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["nonexistent-package-12345==1.0.0"]\n')

        with pytest.raises(SystemExit):
            _check_dependencies(tmp_path)

        captured = capsys.readouterr()
        assert "pip install ." in captured.err

    def test_handles_parse_error(self, tmp_path, capsys):
        """Test that check warns on parse errors but doesn't exit."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid toml {{{")

        # Should not raise (warns but continues)
        _check_dependencies(tmp_path)

        captured = capsys.readouterr()
        assert "Warning: Could not parse dependencies" in captured.err

    def test_skips_invalid_requirement_strings(self, tmp_path):
        """Test that check skips invalid requirement strings."""
        pyproject = tmp_path / "pyproject.toml"
        # Include an invalid requirement string alongside valid ones
        pyproject.write_text('[project]\ndependencies = ["!!!invalid!!!", "pytest>=9.0.0"]\n')

        # Should not raise - skips invalid, checks valid
        _check_dependencies(tmp_path)
