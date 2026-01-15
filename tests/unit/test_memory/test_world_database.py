"""Tests for WorldDatabase."""

from memory.world_database import WorldDatabase


class TestWorldDatabase:
    """Tests for WorldDatabase."""

    def test_create_database(self, tmp_path):
        """Test creating a new database."""
        db_path = tmp_path / "test.db"
        db = WorldDatabase(db_path)

        assert db_path.exists()
        assert db.count_entities() == 0

    def test_add_entity(self, tmp_path):
        """Test adding an entity."""
        db = WorldDatabase(tmp_path / "test.db")

        entity_id = db.add_entity(
            entity_type="character",
            name="Test Character",
            description="A test character",
        )

        assert entity_id is not None
        assert db.count_entities("character") == 1

    def test_get_entity(self, tmp_path):
        """Test retrieving an entity."""
        db = WorldDatabase(tmp_path / "test.db")

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

    def test_update_entity(self, tmp_path):
        """Test updating an entity."""
        db = WorldDatabase(tmp_path / "test.db")

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

    def test_delete_entity(self, tmp_path):
        """Test deleting an entity."""
        db = WorldDatabase(tmp_path / "test.db")

        entity_id = db.add_entity(
            entity_type="item",
            name="Test Item",
        )

        assert db.count_entities("item") == 1

        result = db.delete_entity(entity_id)

        assert result is True
        assert db.count_entities("item") == 0

    def test_search_entities(self, tmp_path):
        """Test searching entities."""
        db = WorldDatabase(tmp_path / "test.db")

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

    def test_add_relationship(self, tmp_path):
        """Test adding a relationship."""
        db = WorldDatabase(tmp_path / "test.db")

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
