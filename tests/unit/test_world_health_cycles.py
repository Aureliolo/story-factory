"""Tests for world health accepted cycles and cycle hash computation."""

from unittest.mock import MagicMock

import pytest

from src.memory.world_database import WorldDatabase


class TestCycleHashComputation:
    """Tests for deterministic cycle hash computation."""

    def test_hash_is_deterministic(self):
        """Same edges should produce the same hash."""
        edges = [("A", "leads", "B"), ("B", "leads", "C"), ("C", "leads", "A")]
        hash1 = WorldDatabase.compute_cycle_hash(edges)
        hash2 = WorldDatabase.compute_cycle_hash(edges)
        assert hash1 == hash2

    def test_hash_independent_of_traversal_start(self):
        """Different traversal orders of the same cycle should produce the same hash."""
        # Same cycle, different start points
        edges1 = [("A", "leads", "B"), ("B", "leads", "C"), ("C", "leads", "A")]
        edges2 = [("B", "leads", "C"), ("C", "leads", "A"), ("A", "leads", "B")]
        edges3 = [("C", "leads", "A"), ("A", "leads", "B"), ("B", "leads", "C")]
        hash1 = WorldDatabase.compute_cycle_hash(edges1)
        hash2 = WorldDatabase.compute_cycle_hash(edges2)
        hash3 = WorldDatabase.compute_cycle_hash(edges3)
        assert hash1 == hash2 == hash3

    def test_different_cycles_produce_different_hashes(self):
        """Different cycles should produce different hashes."""
        edges1 = [("A", "leads", "B"), ("B", "leads", "A")]
        edges2 = [("A", "leads", "C"), ("C", "leads", "A")]
        hash1 = WorldDatabase.compute_cycle_hash(edges1)
        hash2 = WorldDatabase.compute_cycle_hash(edges2)
        assert hash1 != hash2

    def test_hash_sensitive_to_relation_type(self):
        """Same entities but different relation types should produce different hashes."""
        edges_leads = [("A", "leads", "B"), ("B", "leads", "A")]
        edges_hates = [("A", "hates", "B"), ("B", "hates", "A")]
        hash_leads = WorldDatabase.compute_cycle_hash(edges_leads)
        hash_hates = WorldDatabase.compute_cycle_hash(edges_hates)
        assert hash_leads != hash_hates

    def test_hash_is_16_chars(self):
        """Hash should be exactly 16 hex characters."""
        edges = [("X", "knows", "Y"), ("Y", "knows", "X")]
        h = WorldDatabase.compute_cycle_hash(edges)
        assert len(h) == 16
        # Verify it's valid hex
        int(h, 16)

    def test_empty_cycle_produces_hash(self):
        """Empty edge list should still produce a valid hash."""
        h = WorldDatabase.compute_cycle_hash([])
        assert len(h) == 16


class TestAcceptedCyclesRoundTrip:
    """Tests for accept_cycle / get_accepted_cycles / remove_accepted_cycle."""

    @pytest.fixture
    def world_db(self, tmp_path):
        """Create and yield a WorldDatabase, closing it after the test."""
        db = WorldDatabase(tmp_path / "test.db")
        yield db
        db.close()

    def test_accept_and_get_cycle(self, world_db):
        """Accepted cycle should appear in get_accepted_cycles."""
        cycle_hash = "abcdef0123456789"
        world_db.accept_cycle(cycle_hash)

        accepted = world_db.get_accepted_cycles()
        assert cycle_hash in accepted

    def test_get_empty_accepted_cycles(self, world_db):
        """Empty database should return empty set."""
        accepted = world_db.get_accepted_cycles()
        assert accepted == set()

    def test_accept_duplicate_is_idempotent(self, world_db):
        """Accepting the same cycle twice should not cause errors."""
        cycle_hash = "1234567890abcdef"
        world_db.accept_cycle(cycle_hash)
        world_db.accept_cycle(cycle_hash)

        accepted = world_db.get_accepted_cycles()
        assert cycle_hash in accepted
        assert len(accepted) == 1

    def test_remove_accepted_cycle(self, world_db):
        """Removed cycle should no longer appear in accepted set."""
        cycle_hash = "fedcba9876543210"
        world_db.accept_cycle(cycle_hash)
        assert cycle_hash in world_db.get_accepted_cycles()

        removed = world_db.remove_accepted_cycle(cycle_hash)
        assert removed is True
        assert cycle_hash not in world_db.get_accepted_cycles()

    def test_remove_nonexistent_cycle_returns_false(self, world_db):
        """Removing a cycle that doesn't exist should return False."""
        removed = world_db.remove_accepted_cycle("0000000000000000")
        assert removed is False

    def test_multiple_accepted_cycles(self, world_db):
        """Multiple different cycles can be accepted."""
        hashes = ["aaaa000000000000", "bbbb000000000000", "cccc000000000000"]
        for h in hashes:
            world_db.accept_cycle(h)

        accepted = world_db.get_accepted_cycles()
        assert len(accepted) == 3
        for h in hashes:
            assert h in accepted

    def test_accept_rejects_invalid_hash(self, world_db):
        """accept_cycle should reject hashes that aren't 16 hex chars."""
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("tooshort")

        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("")

        # Right length but not hex
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("ghijklmnopqrstuv")

    def test_remove_rejects_invalid_hash(self, world_db):
        """remove_accepted_cycle should reject hashes that aren't 16 hex chars."""
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.remove_accepted_cycle("tooshort")

        # Right length but not hex
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.remove_accepted_cycle("ghijklmnopqrstuv")


class TestHealthMetricsFilterAcceptedCycles:
    """Tests for filtering accepted cycles from health metrics."""

    def test_accepted_cycles_excluded_from_health(self, tmp_path):
        """Accepted cycles should be excluded from health metrics computation."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_filter.db")
        try:
            # Create entities forming a cycle: A -> B -> C -> A
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            id_c = db.add_entity("character", "Carol", "Character C")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_c, "leads", "B leads C")
            db.add_relationship(id_c, id_a, "leads", "C leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # First: get metrics without accepting the cycle
            metrics_before = get_world_health_metrics(svc, db)
            assert metrics_before.circular_count > 0, "Should detect circular chain"

            # Accept the cycle
            cycles = db.find_circular_relationships()
            assert len(cycles) > 0
            cycle_hash = db.compute_cycle_hash(cycles[0])
            db.accept_cycle(cycle_hash)

            # After accepting: cycle should be filtered out
            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == 0, "Accepted cycle should be excluded"

        finally:
            db.close()

    def test_unaccepted_cycles_still_shown(self, tmp_path):
        """Cycles that are not accepted should still appear in health metrics."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_partial.db")
        try:
            # Create two separate cycles
            # Cycle 1: A -> B -> A
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            # Cycle 2: C -> D -> C
            id_c = db.add_entity("character", "Carol", "Character C")
            id_d = db.add_entity("character", "Dave", "Character D")
            db.add_relationship(id_c, id_d, "leads", "C leads D")
            db.add_relationship(id_d, id_c, "leads", "D leads C")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            metrics = get_world_health_metrics(svc, db)
            initial_count = metrics.circular_count
            assert initial_count >= 2, "Should detect at least 2 circular chains"

            # Accept only the first cycle
            cycles = db.find_circular_relationships()
            cycle_hash = db.compute_cycle_hash(cycles[0])
            db.accept_cycle(cycle_hash)

            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == initial_count - 1

        finally:
            db.close()


class TestDismissHandlerHashConsistency:
    """Test that the dismiss handler builds hashes consistent with health metrics."""

    def test_hash_from_edge_dicts_matches_raw_tuples(self, tmp_path):
        """Hash computed from edge dicts (UI dismiss path) must match raw tuple hash
        (health metrics path) for the same cycle."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_consistency.db")
        try:
            # Create a cycle
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # Get health metrics (which returns dicts with edge data)
            metrics = get_world_health_metrics(svc, db)
            assert metrics.circular_count > 0

            # Simulate what handle_dismiss_circular does: build tuples from edge dicts
            cycle_dict = metrics.circular_relationships[0]
            edges = cycle_dict.get("edges", [])
            dismiss_tuples = [
                (edge.get("source", ""), edge.get("type", ""), edge.get("target", ""))
                for edge in edges
            ]
            dismiss_hash = db.compute_cycle_hash(dismiss_tuples)

            # Get the raw cycle tuples (as health metrics filtering uses them)
            raw_cycles = db.find_circular_relationships()
            raw_hash = db.compute_cycle_hash(raw_cycles[0])

            # These must match for the dismiss → filter chain to work
            assert dismiss_hash == raw_hash, (
                f"Hash mismatch: dismiss handler produces {dismiss_hash}, "
                f"health metrics uses {raw_hash}"
            )

            # Verify it actually filters when accepted
            db.accept_cycle(dismiss_hash)
            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == 0
        finally:
            db.close()


class TestImportExportAcceptedCycles:
    """Tests for accepted_cycles round-trip through export/import."""

    def test_import_with_accepted_cycles(self, tmp_path):
        """Importing data with accepted_cycles should persist them."""
        db = WorldDatabase(tmp_path / "test_import.db")
        try:
            # Create entity data for a valid export structure
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            # Accept a cycle
            cycle_hash = db.compute_cycle_hash(
                [("Alice", "leads", "Bob"), ("Bob", "leads", "Alice")]
            )
            db.accept_cycle(cycle_hash)

            # Export
            data = db.export_to_json()
            assert cycle_hash in data["accepted_cycles"]

            # Import into fresh database
            new_db = WorldDatabase(tmp_path / "imported.db")
            try:
                new_db.import_from_json(data)
                accepted = new_db.get_accepted_cycles()
                assert cycle_hash in accepted
            finally:
                new_db.close()
        finally:
            db.close()

    def test_import_skips_invalid_cycle_hashes(self, tmp_path):
        """Import should skip invalid cycle hashes (wrong length, non-hex)."""
        db = WorldDatabase(tmp_path / "test_skip.db")
        try:
            valid_hash = "abcdef0123456789"
            data = {
                "entities": [],
                "relationships": [],
                "events": [],
                "accepted_cycles": [
                    valid_hash,
                    "tooshort",
                    "ghijklmnopqrstuv",
                    "",
                ],
            }
            db.import_from_json(data)
            accepted = db.get_accepted_cycles()
            assert accepted == {valid_hash}
        finally:
            db.close()


class TestHealthMetricsFallbackOnAcceptedCyclesError:
    """Tests for graceful fallback when accepted cycles lookup fails."""

    def test_health_metrics_shows_all_cycles_when_accepted_cycles_fails(self, tmp_path):
        """If get_accepted_cycles() raises, health metrics should still show all cycles."""
        from unittest.mock import patch

        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_fallback.db")
        try:
            # Create a cycle
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # Patch get_accepted_cycles to raise
            with patch.object(db, "get_accepted_cycles", side_effect=RuntimeError("DB corrupted")):
                metrics = get_world_health_metrics(svc, db)
                # Cycles should still be reported (fallback to empty accepted set)
                assert metrics.circular_count > 0
        finally:
            db.close()
