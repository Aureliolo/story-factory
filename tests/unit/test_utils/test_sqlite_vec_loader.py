"""Tests for the sqlite-vec extension loader utility."""

import logging
import sys
import types
from unittest.mock import MagicMock, call

from src.utils.sqlite_vec_loader import load_vec_extension


class TestLoadVecExtensionSuccess:
    """Tests for the successful loading path."""

    def test_load_vec_extension_success(self, monkeypatch):
        """Loading sqlite-vec successfully returns True and logs info."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        result = load_vec_extension(conn)

        assert result is True
        fake_sqlite_vec.load.assert_called_once_with(conn)

    def test_load_vec_extension_calls_enable_load_extension(self, monkeypatch):
        """Verify enable_load_extension is called with True before load and False after."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        load_vec_extension(conn)

        expected_calls = [call(True), call(False)]
        assert conn.enable_load_extension.call_args_list == expected_calls

    def test_load_vec_extension_success_logs_info(self, monkeypatch, caplog):
        """Successful load logs an info message about sqlite-vec being loaded."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        with caplog.at_level(logging.INFO, logger="src.utils.sqlite_vec_loader"):
            load_vec_extension(conn)

        assert any("sqlite-vec extension loaded successfully" in r.message for r in caplog.records)


class TestLoadVecExtensionImportError:
    """Tests for the ImportError (sqlite-vec not installed) path."""

    def test_load_vec_extension_import_error(self, monkeypatch):
        """When sqlite_vec is not importable, returns False without raising."""
        monkeypatch.delitem(sys.modules, "sqlite_vec", raising=False)

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Raise ImportError for sqlite_vec, delegate everything else."""
            if name == "sqlite_vec":
                raise ImportError("No module named 'sqlite_vec'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        conn = MagicMock()
        result = load_vec_extension(conn)

        assert result is False
        conn.enable_load_extension.assert_not_called()

    def test_load_vec_extension_import_error_logs_warning(self, monkeypatch, caplog):
        """ImportError logs a warning about sqlite-vec not being installed."""
        monkeypatch.delitem(sys.modules, "sqlite_vec", raising=False)

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Raise ImportError for sqlite_vec, delegate everything else."""
            if name == "sqlite_vec":
                raise ImportError("No module named 'sqlite_vec'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        conn = MagicMock()
        with caplog.at_level(logging.WARNING, logger="src.utils.sqlite_vec_loader"):
            load_vec_extension(conn)

        assert any("sqlite-vec package not installed" in r.message for r in caplog.records)


class TestLoadVecExtensionLoadFailure:
    """Tests for the path where sqlite_vec imports but load() raises."""

    def test_load_vec_extension_load_failure(self, monkeypatch):
        """When sqlite_vec.load() raises, returns False without propagating."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Extension load failed")
        )
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        result = load_vec_extension(conn)

        assert result is False

    def test_load_vec_extension_load_failure_logs_warning(self, monkeypatch, caplog):
        """Load failure logs a warning with the exception details."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Extension load failed")
        )
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        with caplog.at_level(logging.WARNING, logger="src.utils.sqlite_vec_loader"):
            load_vec_extension(conn)

        assert any("Failed to load sqlite-vec extension" in r.message for r in caplog.records)

    def test_load_vec_extension_load_failure_disables_extension_loading(self, monkeypatch):
        """On load failure, enable_load_extension(False) is still called for cleanup."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Extension load failed")
        )
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        load_vec_extension(conn)

        # enable_load_extension(True) was called first, then (False) in the except cleanup
        expected_calls = [call(True), call(False)]
        assert conn.enable_load_extension.call_args_list == expected_calls

    def test_load_vec_extension_load_failure_cleanup_also_fails(self, monkeypatch):
        """When both load() and the cleanup enable_load_extension(False) fail, returns False."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Extension load failed")
        )
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        # First call to enable_load_extension(True) succeeds,
        # second call (False) in cleanup raises
        conn.enable_load_extension.side_effect = [None, OSError("Cannot disable")]
        result = load_vec_extension(conn)

        assert result is False

    def test_load_vec_extension_enable_extension_true_fails(self, monkeypatch):
        """When enable_load_extension(True) itself raises, returns False."""
        fake_sqlite_vec = types.ModuleType("sqlite_vec")
        fake_sqlite_vec.load = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

        conn = MagicMock()
        conn.enable_load_extension.side_effect = RuntimeError("Cannot enable extensions")
        result = load_vec_extension(conn)

        assert result is False
        fake_sqlite_vec.load.assert_not_called()
