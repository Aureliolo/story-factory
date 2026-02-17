"""Tests for world health accepted cycles and cycle hash computation."""

from unittest.mock import MagicMock

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

    def test_accept_and_get_cycle(self, tmp_path):
        """Accepted cycle should appear in get_accepted_cycles."""
        db = WorldDatabase(tmp_path / "test_accept.db")
        try:
            cycle_hash = "abcdef0123456789"
            db.accept_cycle(cycle_hash)

            accepted = db.get_accepted_cycles()
            assert cycle_hash in accepted
        finally:
            db.close()

    def test_get_empty_accepted_cycles(self, tmp_path):
        """Empty database should return empty set."""
        db = WorldDatabase(tmp_path / "test_empty.db")
        try:
            accepted = db.get_accepted_cycles()
            assert accepted == set()
        finally:
            db.close()

    def test_accept_duplicate_is_idempotent(self, tmp_path):
        """Accepting the same cycle twice should not cause errors."""
        db = WorldDatabase(tmp_path / "test_dup.db")
        try:
            cycle_hash = "1234567890abcdef"
            db.accept_cycle(cycle_hash)
            db.accept_cycle(cycle_hash)

            accepted = db.get_accepted_cycles()
            assert cycle_hash in accepted
            assert len(accepted) == 1
        finally:
            db.close()

    def test_remove_accepted_cycle(self, tmp_path):
        """Removed cycle should no longer appear in accepted set."""
        db = WorldDatabase(tmp_path / "test_remove.db")
        try:
            cycle_hash = "fedcba9876543210"
            db.accept_cycle(cycle_hash)
            assert cycle_hash in db.get_accepted_cycles()

            removed = db.remove_accepted_cycle(cycle_hash)
            assert removed is True
            assert cycle_hash not in db.get_accepted_cycles()
        finally:
            db.close()

    def test_remove_nonexistent_cycle_returns_false(self, tmp_path):
        """Removing a cycle that doesn't exist should return False."""
        db = WorldDatabase(tmp_path / "test_noremove.db")
        try:
            removed = db.remove_accepted_cycle("does_not_exist_000")
            assert removed is False
        finally:
            db.close()

    def test_multiple_accepted_cycles(self, tmp_path):
        """Multiple different cycles can be accepted."""
        db = WorldDatabase(tmp_path / "test_multi.db")
        try:
            hashes = ["aaaa000000000000", "bbbb000000000000", "cccc000000000000"]
            for h in hashes:
                db.accept_cycle(h)

            accepted = db.get_accepted_cycles()
            assert len(accepted) == 3
            for h in hashes:
                assert h in accepted
        finally:
            db.close()


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
