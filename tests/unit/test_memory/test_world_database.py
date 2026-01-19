"""Tests for WorldDatabase."""

from collections.abc import Generator
from unittest.mock import patch

import pytest

from memory.world_database import (
    MAX_ATTRIBUTES_SIZE_BYTES,
    WorldDatabase,
    _validate_attributes,
)


@pytest.fixture
def db(tmp_path) -> Generator[WorldDatabase]:
    """Create a test database that auto-closes after each test."""
    database = WorldDatabase(tmp_path / "test.db")
    yield database
    database.close()


class TestWorldDatabase:
    """Tests for WorldDatabase."""

    def test_create_database(self, tmp_path):
        """Test creating a new database."""
        db_path = tmp_path / "test.db"
        with WorldDatabase(db_path) as db:
            assert db_path.exists()
            assert db.count_entities() == 0

    def test_add_entity(self, db):
        """Test adding an entity."""

        entity_id = db.add_entity(
            entity_type="character",
            name="Test Character",
            description="A test character",
        )

        assert entity_id is not None
        assert db.count_entities("character") == 1

    def test_get_entity(self, db):
        """Test retrieving an entity."""
        entity_id = db.add_entity(
            entity_type="location",
            name="Test Location",
            description="A test location",
            attributes={"climate": "temperate"},
        )

        entity = db.get_entity(entity_id)

        assert entity is not None
        assert entity.name == "Test Location"
        assert entity.type == "location"
        assert entity.attributes["climate"] == "temperate"

    def test_update_entity(self, db):
        """Test updating an entity."""
        entity_id = db.add_entity(
            entity_type="character",
            name="Original Name",
            description="Original description",
        )

        result = db.update_entity(
            entity_id,
            name="Updated Name",
            description="Updated description",
        )

        assert result is True

        # Verify the update
        updated = db.get_entity(entity_id)
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.description == "Updated description"

    def test_delete_entity(self, db):
        """Test deleting an entity."""
        entity_id = db.add_entity(
            entity_type="item",
            name="Test Item",
        )

        assert db.count_entities("item") == 1

        result = db.delete_entity(entity_id)

        assert result is True
        assert db.count_entities("item") == 0

    def test_search_entities(self, db):
        """Test searching entities."""
        db.add_entity("character", "Alice", "A brave warrior")
        db.add_entity("character", "Bob", "A wise wizard")
        db.add_entity("location", "Alice's Tower", "Where Alice lives")

        # Search by name
        results = db.search_entities("Alice")
        assert len(results) == 2

        # Search with type filter
        results = db.search_entities("Alice", entity_type="character")
        assert len(results) == 1
        assert results[0].name == "Alice"

    def test_add_relationship(self, db):
        """Test adding a relationship."""
        char1_id = db.add_entity("character", "Alice")
        char2_id = db.add_entity("character", "Bob")

        rel_id = db.add_relationship(
            source_id=char1_id,
            target_id=char2_id,
            relation_type="knows",
            description="They are friends",
        )

        assert rel_id is not None

        relationships = db.get_relationships(char1_id)
        assert len(relationships) == 1
        assert relationships[0].relation_type == "knows"

    def test_find_path(self, sample_world_db):
        """Test finding path between entities."""
        # Get entity IDs
        alice = sample_world_db.search_entities("Alice")[0]
        dark_lord = sample_world_db.search_entities("Dark Lord")[0]

        path = sample_world_db.find_path(alice.id, dark_lord.id)

        assert path is not None
        assert alice.id in path
        assert dark_lord.id in path

    def test_get_connected_entities(self, sample_world_db):
        """Test getting connected entities."""
        alice = sample_world_db.search_entities("Alice")[0]

        connected = sample_world_db.get_connected_entities(alice.id, max_depth=1)

        # Alice is connected to Bob and Dark Lord
        assert len(connected) >= 2

    def test_get_most_connected(self, sample_world_db):
        """Test getting most connected entities."""
        most_connected = sample_world_db.get_most_connected(limit=5)

        assert len(most_connected) > 0
        # Each result is (entity, degree)
        for _entity, degree in most_connected:
            assert degree >= 0

    def test_export_import_json(self, tmp_path, sample_world_db):
        """Test exporting and importing JSON."""
        # Export
        json_data = sample_world_db.export_to_json()
        assert "entities" in json_data
        assert "relationships" in json_data

        # Create new database and import
        new_db = WorldDatabase(tmp_path / "imported.db")
        new_db.import_from_json(json_data)

        # Verify data was imported
        assert new_db.count_entities() == sample_world_db.count_entities()

    def test_get_context_for_agents(self, sample_world_db):
        """Test generating context for agents."""
        context = sample_world_db.get_context_for_agents()

        # Context is a dict with characters, locations, etc.
        assert "characters" in context
        assert len(context["characters"]) >= 2

        # Check for Alice and Bob in characters
        char_names = [c["name"] for c in context["characters"]]
        assert "Alice" in char_names
        assert "Bob" in char_names

    def test_add_entity_validation_empty_name(self, db):
        """Test that empty entity names are rejected."""
        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            db.add_entity("character", "")

        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            db.add_entity("character", "   ")

    def test_add_entity_validation_long_name(self, db):
        """Test that overly long entity names are rejected."""
        with pytest.raises(ValueError, match="cannot exceed 200 characters"):
            db.add_entity("character", "x" * 201)

    def test_add_entity_validation_long_description(self, db):
        """Test that overly long descriptions are rejected."""
        with pytest.raises(ValueError, match="cannot exceed 5000 characters"):
            db.add_entity("character", "Test", "x" * 5001)

    def test_update_entity_validation_empty_name(self, db):
        """Test that empty names are rejected in updates."""
        entity_id = db.add_entity("character", "Valid Name")

        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            db.update_entity(entity_id, name="")

    def test_update_entity_validation_strips_whitespace(self, db):
        """Test that entity names are trimmed."""
        entity_id = db.add_entity("character", "  Spaced Name  ")

        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.name == "Spaced Name"

    def test_context_manager_closes_connection(self, tmp_path):
        """Test that context manager properly closes the database connection."""
        db_path = tmp_path / "context_test.db"

        # Use context manager
        with WorldDatabase(db_path) as db:
            entity_id = db.add_entity("character", "Test Character")
            assert entity_id is not None
            assert db.count_entities() == 1

        # After exiting context, connection should be closed
        # Verify by opening a new connection and checking data persisted
        db2 = WorldDatabase(db_path)
        assert db2.count_entities() == 1
        db2.close()

    def test_context_manager_does_not_suppress_exceptions(self, tmp_path):
        """Test that context manager does not suppress exceptions."""
        db_path = tmp_path / "exception_test.db"

        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            with WorldDatabase(db_path) as db:
                db.add_entity("character", "")  # Should raise ValueError

    def test_description_whitespace_stripped_on_add(self, db):
        """Test that description whitespace is stripped when adding entity."""
        entity_id = db.add_entity("character", "Test", "  Spaced description  ")

        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.description == "Spaced description"

    def test_description_whitespace_stripped_on_update(self, db):
        """Test that description whitespace is stripped when updating entity."""
        entity_id = db.add_entity("character", "Test", "Original")

        db.update_entity(entity_id, description="  Updated description  ")

        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.description == "Updated description"

    def test_get_entity_not_found(self, db):
        """Test get_entity returns None for non-existent entity."""
        result = db.get_entity("non-existent-id")

        assert result is None

    def test_get_entity_by_name(self, db):
        """Test get_entity_by_name retrieves entity correctly."""
        db.add_entity("character", "Alice", "A brave character")
        db.add_entity("location", "Alice's House", "Where Alice lives")

        # Get by name only
        entity = db.get_entity_by_name("Alice")
        assert entity is not None
        assert entity.name == "Alice"

        # Get by name and type
        entity = db.get_entity_by_name("Alice", entity_type="character")
        assert entity is not None
        assert entity.type == "character"

        # Get by name with wrong type
        entity = db.get_entity_by_name("Alice", entity_type="location")
        assert entity is None

    def test_get_entity_by_name_not_found(self, db):
        """Test get_entity_by_name returns None for non-existent name."""
        result = db.get_entity_by_name("NonExistent")

        assert result is None

    def test_update_entity_not_found(self, db):
        """Test update_entity returns False for non-existent entity."""
        result = db.update_entity("non-existent-id", name="New Name")

        assert result is False

    def test_update_entity_invalid_field(self, db):
        """Test update_entity ignores invalid fields."""
        entity_id = db.add_entity("character", "Test")

        # invalid_field should be ignored (not in allowed fields)
        result = db.update_entity(entity_id, name="Updated", invalid_field="ignored")

        assert result is True
        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.name == "Updated"

    def test_update_entity_attributes(self, db):
        """Test update_entity can update attributes."""
        entity_id = db.add_entity("character", "Test", attributes={"age": 25})

        result = db.update_entity(entity_id, attributes={"age": 30, "rank": "Captain"})

        assert result is True
        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.attributes["age"] == 30
        assert entity.attributes["rank"] == "Captain"

    def test_update_entity_type(self, db):
        """Test update_entity can change entity type."""
        entity_id = db.add_entity("character", "Test")

        result = db.update_entity(entity_id, type="faction")

        assert result is True
        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.type == "faction"

    def test_delete_entity_not_found(self, db):
        """Test delete_entity returns False for non-existent entity."""
        result = db.delete_entity("non-existent-id")

        assert result is False

    def test_delete_entity_cleans_up_graph(self, db):
        """Test delete_entity removes entity from graph."""
        char1 = db.add_entity("character", "Alice")

        # Entity should be in graph
        graph = db.get_graph()
        assert char1 in graph

        # Delete Alice
        db.delete_entity(char1)

        # Entity should be gone
        assert db.get_entity(char1) is None

    def test_list_entities_with_type_filter(self, db):
        """Test list_entities with type filter."""
        db.add_entity("character", "Alice")
        db.add_entity("character", "Bob")
        db.add_entity("location", "Town")

        chars = db.list_entities(entity_type="character")
        locs = db.list_entities(entity_type="location")

        assert len(chars) == 2
        assert len(locs) == 1

    def test_list_entities_all(self, db):
        """Test list_entities without filter."""
        db.add_entity("character", "Alice")
        db.add_entity("location", "Town")

        all_entities = db.list_entities()

        assert len(all_entities) == 2

    def test_get_relationship_between(self, db):
        """Test get_relationship_between returns relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        db.add_relationship(char1, char2, "knows")

        rel = db.get_relationship_between(char1, char2)

        assert rel is not None
        assert rel.relation_type == "knows"

    def test_get_relationship_between_not_found(self, db):
        """Test get_relationship_between returns None when not found."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")

        rel = db.get_relationship_between(char1, char2)

        assert rel is None

    def test_delete_relationship(self, db):
        """Test delete_relationship removes relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "knows")

        result = db.delete_relationship(rel_id)

        assert result is True
        assert db.get_relationship_between(char1, char2) is None

    def test_delete_relationship_not_found(self, db):
        """Test delete_relationship returns False when not found."""
        result = db.delete_relationship("non-existent-id")

        assert result is False

    def test_update_relationship(self, db):
        """Test update_relationship modifies relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "knows", description="Acquaintances")

        result = db.update_relationship(rel_id, relation_type="friend", description="Best friends")

        assert result is True
        rel = db.get_relationship_between(char1, char2)
        assert rel is not None
        assert rel.relation_type == "friend"
        assert rel.description == "Best friends"

    def test_update_relationship_not_found(self, db):
        """Test update_relationship returns False when not found."""
        result = db.update_relationship("non-existent-id", relation_type="friend")

        assert result is False

    def test_list_relationships(self, db):
        """Test list_relationships returns all relationships."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        char3 = db.add_entity("character", "Charlie")
        db.add_relationship(char1, char2, "knows")
        db.add_relationship(char2, char3, "knows")

        rels = db.list_relationships()

        assert len(rels) == 2

    def test_close_database(self, tmp_path):
        """Test close properly closes the connection."""
        db = WorldDatabase(tmp_path / "test.db")
        db.add_entity("character", "Test")

        db.close()

        # Connection should be closed
        assert db._closed is True

    def test_close_twice_is_safe(self, tmp_path):
        """Test closing twice doesn't raise error."""
        db = WorldDatabase(tmp_path / "test.db")
        db.close()
        db.close()  # Should not raise


class TestWorldDatabaseEvents:
    """Tests for event-related methods."""

    def test_add_event(self, db):
        """Test adding a world event."""
        char_id = db.add_entity("character", "Alice")

        # Participants are tuples of (entity_id, role)
        event_id = db.add_event(
            description="Alice discovers a secret",
            participants=[(char_id, "protagonist")],
            chapter_number=1,
        )

        assert event_id is not None

    def test_get_events_for_entity(self, db):
        """Test getting events for a specific entity."""
        char_id = db.add_entity("character", "Alice")
        db.add_event(
            description="First event",
            participants=[(char_id, "protagonist")],
            chapter_number=1,
        )

        events = db.get_events_for_entity(char_id)

        assert len(events) == 1
        assert "First event" in events[0].description

    def test_get_events_for_chapter(self, db):
        """Test getting events for a specific chapter."""
        char_id = db.add_entity("character", "Alice")
        db.add_event("Event 1", [(char_id, "main")], chapter_number=1)
        db.add_event("Event 2", [(char_id, "main")], chapter_number=1)
        db.add_event("Event 3", [(char_id, "main")], chapter_number=2)

        events = db.get_events_for_chapter(1)

        assert len(events) == 2

    def test_get_event_participants(self, db):
        """Test getting participants for an event."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        event_id = db.add_event(
            description="A battle",
            participants=[(char1, "hero"), (char2, "villain")],
            chapter_number=1,
        )

        participants = db.get_event_participants(event_id)

        assert len(participants) == 2

    def test_list_events_with_limit(self, db):
        """Test list_events with limit."""
        char_id = db.add_entity("character", "Alice")
        for i in range(5):
            db.add_event(f"Event {i}", [(char_id, "main")], chapter_number=1)

        events = db.list_events(limit=3)

        assert len(events) == 3

    def test_list_events_no_limit(self, db):
        """Test list_events without limit."""
        char_id = db.add_entity("character", "Alice")
        for i in range(3):
            db.add_event(f"Event {i}", [(char_id, "main")], chapter_number=1)

        events = db.list_events()

        assert len(events) == 3

    def test_add_event_with_consequences(self, db):
        """Test adding event with consequences."""
        char_id = db.add_entity("character", "Alice")

        db.add_event(
            description="Major event",
            participants=[(char_id, "main")],
            chapter_number=1,
            consequences=["Changed the world", "Awakened powers"],
        )

        events = db.list_events()
        assert len(events) == 1
        assert events[0].consequences == ["Changed the world", "Awakened powers"]

    def test_add_event_no_participants(self, db):
        """Test adding event without participants."""
        event_id = db.add_event(
            description="Natural disaster",
            chapter_number=1,
        )

        assert event_id is not None
        events = db.list_events()
        assert len(events) == 1


class TestWorldDatabaseGraphOperations:
    """Tests for graph-related operations."""

    def test_get_graph_returns_digraph(self, sample_world_db):
        """Test get_graph returns a NetworkX DiGraph."""
        import networkx as nx

        graph = sample_world_db.get_graph()

        assert isinstance(graph, nx.DiGraph)

    def test_find_path_not_found(self, db):
        """Test find_path returns empty when no path exists."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        # No relationship between them

        path = db.find_path(char1, char2)

        assert path == []

    def test_find_all_paths(self, sample_world_db):
        """Test find_all_paths returns multiple paths."""
        alice = sample_world_db.search_entities("Alice")[0]
        dark_lord = sample_world_db.search_entities("Dark Lord")[0]

        paths = sample_world_db.find_all_paths(alice.id, dark_lord.id)

        # Should have at least one path
        assert len(paths) >= 1

    def test_find_all_paths_with_max_length(self, sample_world_db):
        """Test find_all_paths respects max_length."""
        alice = sample_world_db.search_entities("Alice")[0]
        dark_lord = sample_world_db.search_entities("Dark Lord")[0]

        paths = sample_world_db.find_all_paths(alice.id, dark_lord.id, max_length=1)

        # With max_length of 1, may not find a path if distance is > 1
        # Just verify it doesn't crash and returns a list
        assert isinstance(paths, list)

    def test_find_all_paths_no_path(self, db):
        """Test find_all_paths returns empty when no path exists."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        # No relationship between them

        paths = db.find_all_paths(char1, char2)

        assert paths == []

    def test_find_all_paths_node_not_found(self, db):
        """Test find_all_paths handles non-existent node."""
        char1 = db.add_entity("character", "Alice")

        paths = db.find_all_paths(char1, "non-existent")

        assert paths == []

    def test_get_communities(self, sample_world_db):
        """Test get_communities returns community structure."""
        communities = sample_world_db.get_communities()

        # Should return list of communities (each is a list of entity IDs)
        assert isinstance(communities, list)
        if communities:
            assert isinstance(communities[0], list)

    def test_get_communities_empty_graph(self, db):
        """Test get_communities returns empty for empty graph."""
        communities = db.get_communities()

        assert communities == []

    def test_get_entity_centrality(self, sample_world_db):
        """Test get_entity_centrality returns centrality scores."""
        centrality = sample_world_db.get_entity_centrality()

        # Should return dict of entity_id -> centrality score
        assert isinstance(centrality, dict)
        for entity_id, score in centrality.items():
            assert isinstance(entity_id, str)
            assert isinstance(score, float)

    def test_get_entity_centrality_empty_graph(self, db):
        """Test get_entity_centrality returns empty for empty graph."""
        centrality = db.get_entity_centrality()

        assert centrality == {}

    def test_get_connected_entities_nonexistent(self, db):
        """Test get_connected_entities with non-existent entity."""
        connected = db.get_connected_entities("non-existent-id")

        assert connected == []

    def test_get_connected_entities_no_connections(self, db):
        """Test get_connected_entities with entity that has no connections."""
        char_id = db.add_entity("character", "Lonely")

        connected = db.get_connected_entities(char_id)

        assert connected == []

    def test_get_important_relationships(self, sample_world_db):
        """Test _get_important_relationships returns formatted relationships."""
        rels = sample_world_db._get_important_relationships(limit=5)

        assert isinstance(rels, list)
        if rels:
            # Uses 'from' and 'to' keys, not 'source' and 'target'
            assert "from" in rels[0]
            assert "to" in rels[0]
            assert "type" in rels[0]


class TestValidateAttributes:
    """Tests for _validate_attributes function."""

    def test_valid_attributes(self):
        """Test valid attributes pass validation."""
        attrs = {"name": "Test", "age": 25, "nested": {"key": "value"}}
        _validate_attributes(attrs)  # Should not raise

    def test_attributes_exceed_size(self):
        """Test attributes exceeding size limit raise error."""
        # Create attrs larger than MAX_ATTRIBUTES_SIZE_BYTES
        large_value = "x" * (MAX_ATTRIBUTES_SIZE_BYTES + 1000)
        attrs = {"large": large_value}

        with pytest.raises(ValueError, match="exceed maximum size"):
            _validate_attributes(attrs)

    def test_attributes_exceed_depth(self):
        """Test attributes exceeding depth limit raise error."""
        # Create deeply nested attrs (more than MAX_ATTRIBUTES_DEPTH)
        attrs = {"l1": {"l2": {"l3": {"l4": {"l5": "too deep"}}}}}

        with pytest.raises(ValueError, match="exceed maximum nesting depth"):
            _validate_attributes(attrs)

    def test_attributes_list_depth(self):
        """Test deeply nested lists also checked."""
        # Lists at each level
        attrs = {"l1": [{"l2": [{"l3": [{"l4": "too deep"}]}]}]}

        with pytest.raises(ValueError, match="exceed maximum nesting depth"):
            _validate_attributes(attrs)


class TestWorldDatabaseDestructor:
    """Tests for __del__ destructor."""

    def test_destructor_closes_connection(self, tmp_path):
        """Test destructor closes connection safely."""
        db = WorldDatabase(tmp_path / "test.db")
        db.add_entity("character", "Test")

        # Manually call __del__ to test (normally called by GC)
        db.__del__()

        # Should be closed
        assert db._closed is True

    def test_destructor_handles_exception(self, tmp_path):
        """Test destructor handles exceptions gracefully."""
        db = WorldDatabase(tmp_path / "test.db")

        # Mock close to raise an exception
        with patch.object(db, "close", side_effect=Exception("Test error")):
            # Should not raise, just ignore the error
            db.__del__()


class TestWorldDatabaseAddEntityValidation:
    """Tests for add_entity validation."""

    def test_add_entity_invalid_type(self, db):
        """Test add_entity rejects invalid entity type."""
        with pytest.raises(ValueError, match="Invalid entity type"):
            db.add_entity("invalid_type", "Test")

    def test_add_entity_with_attributes_validation(self, db):
        """Test add_entity validates attributes."""
        # Create deeply nested attributes that exceed depth limit
        deep_attrs = {"l1": {"l2": {"l3": {"l4": {"l5": "too deep"}}}}}

        with pytest.raises(ValueError, match="exceed maximum nesting depth"):
            db.add_entity("character", "Test", attributes=deep_attrs)


class TestWorldDatabaseRelationshipFeatures:
    """Tests for relationship features."""

    def test_add_relationship_bidirectional(self, db):
        """Test adding bidirectional relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")

        db.add_relationship(char1, char2, "friends", bidirectional=True, strength=0.9)

        rel = db.get_relationship_between(char1, char2)
        assert rel is not None
        assert rel.bidirectional is True
        assert rel.strength == 0.9

    def test_add_relationship_with_attributes(self, db):
        """Test adding relationship with attributes."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")

        db.add_relationship(char1, char2, "knows", attributes={"since": "childhood", "trust": 10})

        rel = db.get_relationship_between(char1, char2)
        assert rel is not None
        assert rel.attributes["since"] == "childhood"
        assert rel.attributes["trust"] == 10

    def test_update_relationship_strength(self, db):
        """Test updating relationship strength."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "knows", strength=0.5)

        result = db.update_relationship(rel_id, strength=0.9)

        assert result is True
        rel = db.get_relationship_between(char1, char2)
        assert rel is not None
        assert rel.strength == 0.9

    def test_update_relationship_attributes(self, db):
        """Test updating relationship attributes."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "knows")

        result = db.update_relationship(rel_id, attributes={"status": "close"})

        assert result is True
        rel = db.get_relationship_between(char1, char2)
        assert rel is not None
        assert rel.attributes["status"] == "close"


class TestWorldDatabaseGetRelationships:
    """Tests for get_relationships with different directions."""

    def test_get_relationships_outgoing(self, db):
        """Test get_relationships with outgoing direction."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        char3 = db.add_entity("character", "Charlie")
        db.add_relationship(char1, char2, "knows")  # Alice -> Bob
        db.add_relationship(char3, char1, "knows")  # Charlie -> Alice

        rels = db.get_relationships(char1, direction="outgoing")

        assert len(rels) == 1
        assert rels[0].target_id == char2

    def test_get_relationships_incoming(self, db):
        """Test get_relationships with incoming direction."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        char3 = db.add_entity("character", "Charlie")
        db.add_relationship(char1, char2, "knows")  # Alice -> Bob
        db.add_relationship(char3, char1, "knows")  # Charlie -> Alice

        rels = db.get_relationships(char1, direction="incoming")

        assert len(rels) == 1
        assert rels[0].source_id == char3

    def test_get_relationships_both(self, db):
        """Test get_relationships with both direction (default)."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        char3 = db.add_entity("character", "Charlie")
        db.add_relationship(char1, char2, "knows")  # Alice -> Bob
        db.add_relationship(char3, char1, "knows")  # Charlie -> Alice

        rels = db.get_relationships(char1, direction="both")

        assert len(rels) == 2


class TestWorldDatabaseUpdateValidation:
    """Tests for update_entity validation edge cases."""

    def test_update_entity_skips_updated_at(self, db):
        """Test update_entity ignores explicit updated_at."""
        entity_id = db.add_entity("character", "Test")

        # Try to set updated_at explicitly (should be ignored)
        result = db.update_entity(entity_id, updated_at="2000-01-01", name="New Name")

        assert result is True
        entity = db.get_entity(entity_id)
        assert entity is not None
        assert entity.name == "New Name"

    def test_update_entity_no_valid_fields(self, db):
        """Test update_entity with no valid fields returns False."""
        entity_id = db.add_entity("character", "Test")

        # Only provide invalid/ignored fields
        result = db.update_entity(entity_id, invalid_field="value")

        # Should return False since no valid updates
        assert result is False

    def test_update_entity_long_name(self, db):
        """Test update_entity rejects long name."""
        entity_id = db.add_entity("character", "Test")

        with pytest.raises(ValueError, match="cannot exceed 200 characters"):
            db.update_entity(entity_id, name="x" * 201)

    def test_update_entity_long_description(self, db):
        """Test update_entity rejects long description."""
        entity_id = db.add_entity("character", "Test")

        with pytest.raises(ValueError, match="cannot exceed 5000 characters"):
            db.update_entity(entity_id, description="x" * 5001)

    def test_update_entity_empty_type(self, db):
        """Test update_entity rejects empty type."""
        entity_id = db.add_entity("character", "Test")

        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            db.update_entity(entity_id, type="")

    def test_update_entity_invalid_type(self, db):
        """Test update_entity rejects invalid type."""
        entity_id = db.add_entity("character", "Test")

        with pytest.raises(ValueError, match="Invalid entity type"):
            db.update_entity(entity_id, type="invalid_type")


class TestWorldDatabaseAddEntityValidationExtra:
    """Additional tests for add_entity validation."""

    def test_add_entity_empty_type(self, db):
        """Test add_entity rejects empty type."""
        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            db.add_entity("", "Test Name")

    def test_add_entity_whitespace_type(self, db):
        """Test add_entity rejects whitespace-only type."""
        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            db.add_entity("   ", "Test Name")


class TestWorldDatabaseGraphInternal:
    """Tests for internal graph operations."""

    def test_graph_updates_on_add_entity(self, db):
        """Test graph is updated when adding entity after graph is built."""
        char1 = db.add_entity("character", "Alice")

        # Build graph first
        graph = db.get_graph()
        assert char1 in graph

        # Add another entity - should update graph
        char2 = db.add_entity("character", "Bob")
        graph = db.get_graph()
        assert char2 in graph

    def test_graph_updates_on_update_entity(self, db):
        """Test graph is updated when updating entity."""
        char_id = db.add_entity("character", "Alice")

        # Build graph
        graph = db.get_graph()
        assert graph.nodes[char_id]["name"] == "Alice"

        # Update entity
        db.update_entity(char_id, name="Alicia")
        graph = db.get_graph()
        assert graph.nodes[char_id]["name"] == "Alicia"

    def test_graph_updates_on_delete_entity(self, db):
        """Test graph is updated when deleting entity."""
        char_id = db.add_entity("character", "Alice")

        # Build graph
        graph = db.get_graph()
        assert char_id in graph

        # Delete entity
        db.delete_entity(char_id)
        graph = db.get_graph()
        # Graph should be invalidated and rebuilt without the entity
        assert char_id not in graph

    def test_graph_updates_on_add_relationship(self, db):
        """Test graph is updated when adding relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")

        # Build graph
        graph = db.get_graph()
        assert not graph.has_edge(char1, char2)

        # Add relationship
        db.add_relationship(char1, char2, "knows")
        graph = db.get_graph()
        assert graph.has_edge(char1, char2)

    def test_graph_updates_on_add_bidirectional_relationship(self, db):
        """Test graph is updated for bidirectional relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")

        # Build graph first
        graph = db.get_graph()

        # Add bidirectional relationship
        db.add_relationship(char1, char2, "friends", bidirectional=True)
        graph = db.get_graph()

        # Should have edges both ways
        assert graph.has_edge(char1, char2)
        assert graph.has_edge(char2, char1)

    def test_graph_updates_on_delete_relationship(self, db):
        """Test graph is updated when deleting relationship."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "knows")

        # Build graph
        graph = db.get_graph()
        assert graph.has_edge(char1, char2)

        # Delete relationship
        db.delete_relationship(rel_id)
        graph = db.get_graph()
        # Graph is invalidated and rebuilt without the edge
        assert not graph.has_edge(char1, char2)


class TestWorldDatabaseImportExportEvents:
    """Tests for import/export with events."""

    def test_export_import_with_events(self, tmp_path):
        """Test export and import preserves events."""
        db = WorldDatabase(tmp_path / "test.db")
        char_id = db.add_entity("character", "Alice")
        db.add_event(
            "Test event",
            participants=[(char_id, "main")],
            chapter_number=1,
            consequences=["Result 1"],
        )

        # Export
        data = db.export_to_json()
        assert "events" in data
        assert len(data["events"]) == 1

        # Import to new database
        new_db = WorldDatabase(tmp_path / "imported.db")
        new_db.import_from_json(data)

        # Verify events imported
        events = new_db.list_events()
        assert len(events) == 1
        assert events[0].description == "Test event"

    def test_import_events_with_participants(self, tmp_path):
        """Test import preserves event participants."""
        db = WorldDatabase(tmp_path / "test.db")
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        db.add_event(
            "Group event",
            participants=[(char1, "hero"), (char2, "sidekick")],
            chapter_number=1,
        )

        # Export
        data = db.export_to_json()

        # Import to new database
        new_db = WorldDatabase(tmp_path / "imported.db")
        new_db.import_from_json(data)

        # Verify participants imported
        events = new_db.list_events()
        assert len(events) == 1

        # Check participants by looking at events for entities
        alice_events = new_db.get_events_for_entity(char1)
        assert len(alice_events) == 1


class TestWorldDatabaseGetMostConnected:
    """Tests for get_most_connected method."""

    def test_get_most_connected_empty(self, db):
        """Test get_most_connected with empty database."""
        result = db.get_most_connected(limit=5)

        assert result == []

    def test_get_most_connected_ordering(self, db):
        """Test get_most_connected returns correctly ordered."""
        # Create hub entity connected to many
        hub = db.add_entity("character", "Hub")
        chars = [db.add_entity("character", f"Char{i}") for i in range(5)]
        for char in chars:
            db.add_relationship(hub, char, "knows")

        result = db.get_most_connected(limit=3)

        # Hub should be first (most connected)
        assert len(result) <= 3
        if result:
            assert result[0][0].name == "Hub"


class TestWorldDatabaseGraphInternalExtra:
    """Additional tests for internal graph operations."""

    def test_graph_updates_on_update_entity_type(self, db):
        """Test graph is updated when updating entity type."""
        char_id = db.add_entity("character", "Alice")

        # Build graph
        graph = db.get_graph()
        assert graph.nodes[char_id]["type"] == "character"

        # Update type
        db.update_entity(char_id, type="faction")
        graph = db.get_graph()
        assert graph.nodes[char_id]["type"] == "faction"

    def test_graph_updates_on_update_entity_description(self, db):
        """Test graph is updated when updating entity description."""
        char_id = db.add_entity("character", "Alice", "Original description")

        # Build graph
        graph = db.get_graph()
        assert graph.nodes[char_id]["description"] == "Original description"

        # Update description
        db.update_entity(char_id, description="New description")
        graph = db.get_graph()
        assert graph.nodes[char_id]["description"] == "New description"

    def test_graph_updates_on_update_entity_attributes(self, db):
        """Test graph is updated when updating entity attributes."""
        char_id = db.add_entity("character", "Alice", attributes={"age": 25})

        # Build graph
        graph = db.get_graph()
        assert graph.nodes[char_id]["attributes"]["age"] == 25

        # Update attributes
        db.update_entity(char_id, attributes={"age": 30, "rank": "Captain"})
        graph = db.get_graph()
        assert graph.nodes[char_id]["attributes"]["age"] == 30
        assert graph.nodes[char_id]["attributes"]["rank"] == "Captain"

    def test_graph_deletes_bidirectional_relationship(self, db):
        """Test deleting bidirectional relationship removes both edges."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        rel_id = db.add_relationship(char1, char2, "friends", bidirectional=True)

        # Build graph and verify both edges
        graph = db.get_graph()
        assert graph.has_edge(char1, char2)
        assert graph.has_edge(char2, char1)

        # Delete relationship
        db.delete_relationship(rel_id)

        # Both edges should be gone
        graph = db.get_graph()
        assert not graph.has_edge(char1, char2)
        assert not graph.has_edge(char2, char1)

    def test_graph_rebuild_with_bidirectional_relationship(self, db):
        """Test graph rebuild includes bidirectional relationships."""
        char1 = db.add_entity("character", "Alice")
        char2 = db.add_entity("character", "Bob")
        db.add_relationship(char1, char2, "friends", bidirectional=True)

        # Force graph rebuild by invalidating and getting
        db._invalidate_graph()
        graph = db.get_graph()

        # Both edges should exist
        assert graph.has_edge(char1, char2)
        assert graph.has_edge(char2, char1)


class TestWorldDatabaseFindAllPathsNodeNotFound:
    """Tests for find_all_paths with non-existent nodes."""

    def test_find_all_paths_source_not_found(self, db):
        """Test find_all_paths with non-existent source node."""
        char1 = db.add_entity("character", "Alice")

        # Source doesn't exist
        paths = db.find_all_paths("non-existent", char1)

        assert paths == []

    def test_find_all_paths_target_not_found(self, db):
        """Test find_all_paths with non-existent target node."""
        char1 = db.add_entity("character", "Alice")

        # Target doesn't exist
        paths = db.find_all_paths(char1, "non-existent")

        assert paths == []


class TestWorldDatabaseGraphEdgeCases:
    """Tests for edge cases in graph operations."""

    def test_update_entity_in_graph_entity_not_in_graph(self, db):
        """Test _update_entity_in_graph handles entity not in graph."""
        char_id = db.add_entity("character", "Alice")

        # Build graph
        graph = db.get_graph()
        assert char_id in graph

        # Manually remove entity from graph (simulating inconsistency)
        graph.remove_node(char_id)
        assert char_id not in graph

        # Now try to update - should handle gracefully
        db._update_entity_in_graph(char_id, name="New Name")

        # No error should occur, and entity should still not be in graph
        assert char_id not in graph

    def test_invalidate_graph_cache(self, db):
        """Test invalidate_graph_cache forces graph rebuild on next access."""
        char_id = db.add_entity("character", "Alice")

        # Build graph first
        graph = db.get_graph()
        assert char_id in graph

        # Invalidate cache
        db.invalidate_graph_cache()

        # Graph should be None after invalidation
        assert db._graph is None

        # Getting graph again should rebuild it
        graph = db.get_graph()
        assert graph is not None
        assert char_id in graph
