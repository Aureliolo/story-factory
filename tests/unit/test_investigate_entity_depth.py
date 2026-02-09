"""Unit tests for scripts/investigate_entity_depth.py.

Tests the entity depth analysis functions without requiring actual world databases.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.investigate_entity_depth import (
    EntityDepthRecord,
    _is_populated,
    _normalize_words,
    compute_cross_type_comparison,
    compute_type_depth_stats,
    find_world_databases,
    read_world_database,
)


# =====================================================================
# EntityDepthRecord tests
# =====================================================================
class TestEntityDepthRecord:
    """Tests for the EntityDepthRecord data class."""

    def test_total_text_includes_description(self):
        """Total text should include the description."""
        r = EntityDepthRecord("faction", "Guild", "A powerful guild.", {})
        assert "A powerful guild." in r.total_text

    def test_total_text_includes_string_attributes(self):
        """Total text should include string attribute values."""
        r = EntityDepthRecord(
            "faction",
            "Guild",
            "A guild.",
            {"leader": "Guildmaster Aldric", "base_location": "Capital City"},
        )
        assert "Guildmaster Aldric" in r.total_text
        assert "Capital City" in r.total_text

    def test_total_text_includes_list_attributes(self):
        """Total text should include items from list attributes."""
        r = EntityDepthRecord(
            "faction",
            "Guild",
            "A guild.",
            {"goals": ["Expand trade", "Maintain monopoly"]},
        )
        assert "Expand trade" in r.total_text
        assert "Maintain monopoly" in r.total_text

    def test_total_text_length(self):
        """Total text length should count all characters."""
        r = EntityDepthRecord("faction", "Guild", "Hello world", {})
        assert r.total_text_length == len("Hello world")

    def test_word_count(self):
        """Word count should count space-separated words."""
        r = EntityDepthRecord("faction", "Guild", "one two three four five", {})
        assert r.word_count == 5

    def test_vocabulary_diversity_all_unique(self):
        """All unique words should give diversity of 1.0."""
        r = EntityDepthRecord("faction", "Guild", "alpha beta gamma delta", {})
        assert r.vocabulary_diversity == 1.0

    def test_vocabulary_diversity_all_same(self):
        """All identical words should give low diversity."""
        r = EntityDepthRecord("faction", "Guild", "word word word word", {})
        assert r.vocabulary_diversity == pytest.approx(0.25)

    def test_vocabulary_diversity_empty(self):
        """Empty text should give 0.0 diversity."""
        r = EntityDepthRecord("faction", "Guild", "", {})
        assert r.vocabulary_diversity == 0.0

    def test_populated_fields_empty_entity(self):
        """Entity with no content should have 0 populated fields."""
        r = EntityDepthRecord("faction", "Guild", "", {})
        assert r.populated_fields == 0

    def test_populated_fields_with_content(self):
        """Entity with content should count populated fields."""
        r = EntityDepthRecord(
            "faction",
            "Guild",
            "A guild.",
            {"leader": "Aldric", "goals": ["trade"], "values": []},
        )
        # description (1) + leader (1) + goals (1) = 3 (values is empty list)
        assert r.populated_fields == 3

    def test_field_completeness_full(self):
        """Entity with all expected fields should have high completeness."""
        r = EntityDepthRecord(
            "faction",
            "Guild",
            "A powerful guild.",
            {
                "name": "Guild",
                "type": "faction",
                "leader": "Aldric",
                "goals": ["trade"],
                "values": ["profit"],
                "base_location": "Capital",
            },
        )
        assert r.field_completeness > 0.8

    def test_field_completeness_minimal(self):
        """Entity with only description should have low completeness."""
        r = EntityDepthRecord("faction", "Guild", "A guild.", {})
        # faction expects 5 depth fields, only description populated = 1/5 = 0.2
        assert r.field_completeness < 0.3

    def test_expected_field_count_known_type(self):
        """Known entity type should use EXPECTED_FIELDS."""
        r = EntityDepthRecord("faction", "Guild", "desc", {})
        assert (
            r.expected_field_count == 5
        )  # faction: description, leader, goals, values, base_location

    def test_expected_field_count_character(self):
        """Character should have 8 expected fields."""
        r = EntityDepthRecord("character", "Hero", "desc", {})
        assert (
            r.expected_field_count == 8
        )  # role, description, personality_traits, goals, relationships, arc_notes, arc_type, arc_progress

    def test_expected_field_count_unknown_type(self):
        """Unknown entity type should fall back to description + attribute count."""
        r = EntityDepthRecord("widget", "Gizmo", "desc", {"foo": "bar", "baz": "qux"})
        assert r.expected_field_count == 3  # 1 (description) + 2 attributes


# =====================================================================
# Helper function tests
# =====================================================================
class TestIsPopulated:
    """Tests for the _is_populated helper."""

    def test_none_is_not_populated(self):
        """None should not be populated."""
        assert _is_populated(None) is False

    def test_empty_string_is_not_populated(self):
        """Empty string should not be populated."""
        assert _is_populated("") is False
        assert _is_populated("   ") is False

    def test_non_empty_string_is_populated(self):
        """Non-empty string should be populated."""
        assert _is_populated("hello") is True

    def test_empty_list_is_not_populated(self):
        """Empty list should not be populated."""
        assert _is_populated([]) is False

    def test_non_empty_list_is_populated(self):
        """Non-empty list should be populated."""
        assert _is_populated(["item"]) is True

    def test_empty_dict_is_not_populated(self):
        """Empty dict should not be populated."""
        assert _is_populated({}) is False

    def test_non_empty_dict_is_populated(self):
        """Non-empty dict should be populated."""
        assert _is_populated({"key": "val"}) is True

    def test_numeric_is_populated(self):
        """Numbers should be considered populated."""
        assert _is_populated(0) is True
        assert _is_populated(42) is True


class TestNormalizeWords:
    """Tests for word normalization."""

    def test_basic_words(self):
        """Basic words should be lowercased."""
        result = _normalize_words("Hello World")
        assert result == ["hello", "world"]

    def test_strips_punctuation(self):
        """Punctuation should be stripped."""
        result = _normalize_words("Hello, world! How's it going?")
        assert "hello" in result
        assert "world" in result
        assert "," not in "".join(result)

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert _normalize_words("") == []

    def test_numbers_excluded(self):
        """Pure numbers should be excluded (only alpha words)."""
        result = _normalize_words("test 123 word")
        assert result == ["test", "word"]


# =====================================================================
# Database reading tests
# =====================================================================
class TestReadWorldDatabase:
    """Tests for reading SQLite world databases."""

    def _create_test_db(self, tmp_path: Path, entities: list[tuple[str, str, str, str]]) -> Path:
        """Create a temporary SQLite database with test entities.

        Args:
            tmp_path: Pytest-provided temporary directory.
            entities: List of (type, name, description, attributes_json) tuples.

        Returns:
            Path to the temporary database file.
        """
        db_path = tmp_path / "test_world.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE entities (
                id TEXT PRIMARY KEY, type TEXT, name TEXT,
                description TEXT, attributes TEXT,
                created_at TEXT, updated_at TEXT
            )"""
        )
        for i, (etype, name, desc, attrs) in enumerate(entities):
            conn.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"id-{i}", etype, name, desc, attrs, "2024-01-01", "2024-01-01"),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_read_basic_entities(self, tmp_path: Path):
        """Read entities from a basic database."""
        db_path = self._create_test_db(
            tmp_path,
            [
                ("faction", "The Guild", "A trading guild.", '{"leader": "Aldric"}'),
                ("location", "Castle", "A stone castle.", "{}"),
            ],
        )
        records = read_world_database(db_path)
        assert len(records) == 2
        types = {r.entity_type for r in records}
        assert types == {"faction", "location"}

    def test_read_with_complex_attributes(self, tmp_path: Path):
        """Read entities with complex JSON attributes."""
        attrs = json.dumps(
            {
                "goals": ["Expand trade", "Gain power"],
                "values": ["Profit", "Loyalty"],
                "leader": "Guildmaster",
            }
        )
        db_path = self._create_test_db(
            tmp_path,
            [
                ("faction", "Guild", "Trading guild.", attrs),
            ],
        )
        records = read_world_database(db_path)
        assert len(records) == 1
        assert records[0].attributes["leader"] == "Guildmaster"
        assert len(records[0].attributes["goals"]) == 2

    def test_read_empty_database(self, tmp_path: Path):
        """Empty database should return no records."""
        db_path = self._create_test_db(tmp_path, [])
        records = read_world_database(db_path)
        assert records == []

    def test_read_no_entities_table(self, tmp_path: Path):
        """Database without entities table should return empty list."""
        db_path = tmp_path / "no_entities.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other_table (id TEXT)")
        conn.commit()
        conn.close()

        records = read_world_database(db_path)
        assert records == []

    def test_read_invalid_json_attributes(self, tmp_path: Path):
        """Invalid JSON in attributes should be handled gracefully."""
        db_path = tmp_path / "invalid_json.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE entities (
                id TEXT PRIMARY KEY, type TEXT, name TEXT,
                description TEXT, attributes TEXT,
                created_at TEXT, updated_at TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id-1", "faction", "Guild", "desc", "not-valid-json", "2024-01-01", "2024-01-01"),
        )
        conn.commit()
        conn.close()

        records = read_world_database(db_path)
        assert len(records) == 1
        assert records[0].attributes == {}

    def test_read_non_dict_json_attributes(self, tmp_path: Path):
        """Non-dict JSON (e.g. list) in attributes should default to empty dict."""
        db_path = tmp_path / "list_attrs.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE entities (
                id TEXT PRIMARY KEY, type TEXT, name TEXT,
                description TEXT, attributes TEXT,
                created_at TEXT, updated_at TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id-1", "faction", "Guild", "desc", "[1, 2, 3]", "2024-01-01", "2024-01-01"),
        )
        conn.commit()
        conn.close()

        records = read_world_database(db_path)
        assert len(records) == 1
        assert records[0].attributes == {}


class TestFindWorldDatabases:
    """Tests for finding world database files."""

    def test_find_db_files(self, tmp_path: Path):
        """Find .db files in a directory."""
        (tmp_path / "world1.db").write_bytes(b"")
        (tmp_path / "world2.db").write_bytes(b"")
        (tmp_path / "readme.txt").write_bytes(b"")

        dbs = find_world_databases(tmp_path)
        assert len(dbs) == 2

    def test_find_sqlite_files(self, tmp_path: Path):
        """Find .sqlite and .sqlite3 files."""
        (tmp_path / "world.sqlite").write_bytes(b"")
        (tmp_path / "world.sqlite3").write_bytes(b"")

        dbs = find_world_databases(tmp_path)
        assert len(dbs) == 2

    def test_nonexistent_directory(self, tmp_path: Path):
        """Nonexistent directory should return empty list."""
        dbs = find_world_databases(tmp_path / "does_not_exist")
        assert dbs == []

    def test_file_instead_of_directory(self, tmp_path: Path):
        """Regular file instead of directory should return empty list."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        dbs = find_world_databases(file_path)
        assert dbs == []


# =====================================================================
# Analysis function tests
# =====================================================================
class TestComputeTypeDepthStats:
    """Tests for per-type depth statistics."""

    def test_single_type(self):
        """Stats for a single entity type."""
        records = [
            EntityDepthRecord("faction", "A", "Short description.", {"leader": "Chief"}),
            EntityDepthRecord(
                "faction", "B", "Another longer description with more words here.", {}
            ),
        ]
        stats = compute_type_depth_stats(records)
        assert "faction" in stats
        assert stats["faction"]["entity_count"] == 2
        assert stats["faction"]["word_count"]["mean"] > 0

    def test_multiple_types(self):
        """Stats should be separate for each entity type."""
        records = [
            EntityDepthRecord("faction", "A", "Short.", {}),
            EntityDepthRecord("location", "B", "A much longer description with many words.", {}),
        ]
        stats = compute_type_depth_stats(records)
        assert "faction" in stats
        assert "location" in stats
        # Location should have higher word count
        assert stats["location"]["word_count"]["mean"] > stats["faction"]["word_count"]["mean"]

    def test_empty_records(self):
        """Empty records should return empty stats."""
        stats = compute_type_depth_stats([])
        assert stats == {}


class TestComputeCrossTypeComparison:
    """Tests for cross-type comparison."""

    def test_detects_word_count_imbalance(self):
        """Should flag word count imbalances > 2x."""
        type_stats = {
            "faction": {
                "entity_count": 5,
                "word_count": {"mean": 30.0, "std_dev": 5.0, "min": 20.0, "max": 40.0},
                "vocabulary_diversity": {"mean": 0.7, "std_dev": 0.1, "min": 0.6, "max": 0.8},
                "field_completeness": {"mean": 0.8, "std_dev": 0.1, "min": 0.7, "max": 0.9},
                "text_length": {"mean": 200.0, "std_dev": 50.0, "min": 150.0, "max": 250.0},
            },
            "character": {
                "entity_count": 5,
                "word_count": {"mean": 120.0, "std_dev": 20.0, "min": 80.0, "max": 160.0},
                "vocabulary_diversity": {"mean": 0.6, "std_dev": 0.1, "min": 0.5, "max": 0.7},
                "field_completeness": {"mean": 0.9, "std_dev": 0.05, "min": 0.85, "max": 0.95},
                "text_length": {"mean": 800.0, "std_dev": 100.0, "min": 600.0, "max": 1000.0},
            },
        }
        comparison = compute_cross_type_comparison(type_stats)
        assert len(comparison["imbalances"]) >= 1
        assert "imbalance" in comparison["imbalances"][0].lower()

    def test_no_imbalance_when_similar(self):
        """Should not flag when types have similar depth."""
        type_stats = {
            "faction": {
                "entity_count": 5,
                "word_count": {"mean": 80.0, "std_dev": 10.0, "min": 60.0, "max": 100.0},
                "vocabulary_diversity": {"mean": 0.6, "std_dev": 0.1, "min": 0.5, "max": 0.7},
                "field_completeness": {"mean": 0.8, "std_dev": 0.1, "min": 0.7, "max": 0.9},
                "text_length": {"mean": 500.0, "std_dev": 50.0, "min": 400.0, "max": 600.0},
            },
            "location": {
                "entity_count": 5,
                "word_count": {"mean": 90.0, "std_dev": 15.0, "min": 60.0, "max": 120.0},
                "vocabulary_diversity": {"mean": 0.65, "std_dev": 0.1, "min": 0.55, "max": 0.75},
                "field_completeness": {"mean": 0.75, "std_dev": 0.1, "min": 0.65, "max": 0.85},
                "text_length": {"mean": 550.0, "std_dev": 60.0, "min": 400.0, "max": 700.0},
            },
        }
        comparison = compute_cross_type_comparison(type_stats)
        assert comparison["imbalances"] == []

    def test_empty_stats(self):
        """Empty stats should produce empty comparison."""
        comparison = compute_cross_type_comparison({})
        assert comparison["rankings"] == {}
        assert comparison["imbalances"] == []

    def test_rankings_present(self):
        """Rankings should be present for all metrics."""
        type_stats = {
            "faction": {
                "entity_count": 3,
                "word_count": {"mean": 50.0, "std_dev": 10.0, "min": 30.0, "max": 70.0},
                "vocabulary_diversity": {"mean": 0.6, "std_dev": 0.1, "min": 0.5, "max": 0.7},
                "field_completeness": {"mean": 0.7, "std_dev": 0.1, "min": 0.6, "max": 0.8},
                "text_length": {"mean": 300.0, "std_dev": 50.0, "min": 200.0, "max": 400.0},
            },
        }
        comparison = compute_cross_type_comparison(type_stats)
        assert "by_word_count" in comparison["rankings"]
        assert "by_vocabulary_diversity" in comparison["rankings"]
        assert "by_field_completeness" in comparison["rankings"]
