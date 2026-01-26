"""Tests for world page filter and sort logic - Issue #182."""

from src.memory.entities import Entity


class TestQualityFilter:
    """Tests for quality-based entity filtering."""

    def _make_entity_with_quality(self, entity_id: str, name: str, quality_avg: float) -> Entity:
        """Create an entity with a specific quality score."""
        return Entity(
            id=entity_id,
            name=name,
            type="character",
            description=f"Entity with quality {quality_avg}",
            attributes={"quality_scores": {"average": quality_avg}},
        )

    def _filter_by_quality(self, entities: list[Entity], quality_filter: str) -> list[Entity]:
        """Replicate the filter_by_quality logic from world.py."""
        if quality_filter == "all":
            return entities

        result = []
        for entity in entities:
            scores = entity.attributes.get("quality_scores") if entity.attributes else None
            avg = scores.get("average", 0) if scores else 0

            if quality_filter == "high" and avg >= 8:
                result.append(entity)
            elif quality_filter == "medium" and 6 <= avg < 8:
                result.append(entity)
            elif quality_filter == "low" and avg < 6:
                result.append(entity)
        return result

    def test_filter_all_returns_all_entities(self):
        """Test that 'all' filter returns all entities."""
        entities = [
            self._make_entity_with_quality("1", "Low", 4.0),
            self._make_entity_with_quality("2", "Medium", 7.0),
            self._make_entity_with_quality("3", "High", 9.0),
        ]

        result = self._filter_by_quality(entities, "all")

        assert len(result) == 3

    def test_filter_high_quality(self):
        """Test that 'high' filter returns entities with avg >= 8."""
        entities = [
            self._make_entity_with_quality("1", "Low", 4.0),
            self._make_entity_with_quality("2", "Medium", 7.0),
            self._make_entity_with_quality("3", "High", 8.0),
            self._make_entity_with_quality("4", "VeryHigh", 9.5),
        ]

        result = self._filter_by_quality(entities, "high")

        assert len(result) == 2
        names = [e.name for e in result]
        assert "High" in names
        assert "VeryHigh" in names

    def test_filter_medium_quality(self):
        """Test that 'medium' filter returns entities with 6 <= avg < 8."""
        entities = [
            self._make_entity_with_quality("1", "Low", 4.0),
            self._make_entity_with_quality("2", "MediumLow", 6.0),
            self._make_entity_with_quality("3", "MediumHigh", 7.9),
            self._make_entity_with_quality("4", "High", 8.0),
        ]

        result = self._filter_by_quality(entities, "medium")

        assert len(result) == 2
        names = [e.name for e in result]
        assert "MediumLow" in names
        assert "MediumHigh" in names

    def test_filter_low_quality(self):
        """Test that 'low' filter returns entities with avg < 6."""
        entities = [
            self._make_entity_with_quality("1", "VeryLow", 2.0),
            self._make_entity_with_quality("2", "Low", 5.9),
            self._make_entity_with_quality("3", "Medium", 6.0),
            self._make_entity_with_quality("4", "High", 9.0),
        ]

        result = self._filter_by_quality(entities, "low")

        assert len(result) == 2
        names = [e.name for e in result]
        assert "VeryLow" in names
        assert "Low" in names

    def test_filter_handles_missing_quality_scores(self):
        """Test that entities without quality scores are treated as 0."""
        entities = [
            Entity(
                id="1",
                name="NoScores",
                type="character",
                description="No quality scores",
                attributes={},
            ),
            self._make_entity_with_quality("2", "HasScores", 7.0),
        ]

        # Entities without scores should appear in low filter
        result = self._filter_by_quality(entities, "low")
        assert len(result) == 1
        assert result[0].name == "NoScores"

    def test_filter_handles_empty_attributes(self):
        """Test that entities with empty attributes are handled."""
        entities = [
            Entity(
                id="1",
                name="EmptyAttrs",
                type="character",
                description="Empty attributes",
                attributes={},
            ),
            self._make_entity_with_quality("2", "HasAttrs", 7.0),
        ]

        # Entities with empty attributes should appear in low filter (treated as 0)
        result = self._filter_by_quality(entities, "low")
        assert len(result) == 1
        assert result[0].name == "EmptyAttrs"


class TestEntitySorting:
    """Tests for entity sorting logic."""

    def _make_entities(self) -> list[Entity]:
        """Create test entities for sorting."""
        return [
            Entity(
                id="1",
                name="Zara",
                type="location",
                description="Location Zara",
                attributes={"quality_scores": {"average": 8.0}},
            ),
            Entity(
                id="2",
                name="Alice",
                type="character",
                description="Character Alice",
                attributes={"quality_scores": {"average": 6.0}},
            ),
            Entity(
                id="3",
                name="Bob",
                type="character",
                description="Character Bob",
                attributes={"quality_scores": {"average": 9.0}},
            ),
            Entity(
                id="4",
                name="Middle",
                type="item",
                description="Item Middle",
                attributes={"quality_scores": {"average": 7.0}},
            ),
        ]

    def _sort_entities(
        self, entities: list[Entity], sort_by: str, descending: bool
    ) -> list[Entity]:
        """Replicate the sort_entities logic from world.py."""
        if sort_by == "name":
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)
        elif sort_by == "type":
            return sorted(entities, key=lambda e: e.type, reverse=descending)
        elif sort_by == "quality":
            return sorted(
                entities,
                key=lambda e: (e.attributes.get("quality_scores") or {}).get("average", 0),
                reverse=descending,
            )
        else:
            return sorted(entities, key=lambda e: e.name.lower(), reverse=descending)

    def test_sort_by_name_ascending(self):
        """Test sorting by name in ascending order."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "name", descending=False)

        names = [e.name for e in result]
        assert names == ["Alice", "Bob", "Middle", "Zara"]

    def test_sort_by_name_descending(self):
        """Test sorting by name in descending order."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "name", descending=True)

        names = [e.name for e in result]
        assert names == ["Zara", "Middle", "Bob", "Alice"]

    def test_sort_by_type_ascending(self):
        """Test sorting by type in ascending order."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "type", descending=False)

        types = [e.type for e in result]
        # character < item < location (alphabetically)
        assert types == ["character", "character", "item", "location"]

    def test_sort_by_quality_ascending(self):
        """Test sorting by quality in ascending order."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "quality", descending=False)

        # Quality: Alice 6.0, Middle 7.0, Zara 8.0, Bob 9.0
        names = [e.name for e in result]
        assert names == ["Alice", "Middle", "Zara", "Bob"]

    def test_sort_by_quality_descending(self):
        """Test sorting by quality in descending order."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "quality", descending=True)

        # Quality descending: Bob 9.0, Zara 8.0, Middle 7.0, Alice 6.0
        names = [e.name for e in result]
        assert names == ["Bob", "Zara", "Middle", "Alice"]

    def test_sort_handles_missing_quality(self):
        """Test that entities without quality are sorted as 0."""
        entities = [
            Entity(
                id="1",
                name="WithQuality",
                type="character",
                description="Has quality",
                attributes={"quality_scores": {"average": 5.0}},
            ),
            Entity(
                id="2",
                name="WithoutQuality",
                type="character",
                description="No quality",
                attributes={},
            ),
        ]

        result = self._sort_entities(entities, "quality", descending=False)

        # WithoutQuality (0) should come before WithQuality (5.0)
        names = [e.name for e in result]
        assert names == ["WithoutQuality", "WithQuality"]

    def test_unknown_sort_key_defaults_to_name(self):
        """Test that unknown sort key defaults to name sort."""
        entities = self._make_entities()
        result = self._sort_entities(entities, "unknown_key", descending=False)

        names = [e.name for e in result]
        assert names == ["Alice", "Bob", "Middle", "Zara"]


class TestSearchScope:
    """Tests for search scope (names/descriptions) filtering."""

    def _make_entities_for_search(self) -> list[Entity]:
        """Create test entities for search testing."""
        return [
            Entity(
                id="1",
                name="Dragon",
                type="character",
                description="A fire-breathing creature",
                attributes={},
            ),
            Entity(
                id="2",
                name="Castle",
                type="location",
                description="Where the dragon lives",
                attributes={},
            ),
            Entity(
                id="3",
                name="Sword",
                type="item",
                description="A weapon of steel",
                attributes={},
            ),
        ]

    def _search_entities(
        self,
        entities: list[Entity],
        query: str,
        search_names: bool,
        search_descriptions: bool,
    ) -> list[Entity]:
        """Replicate the search logic from world.py."""
        if not query:
            return entities

        query = query.lower()
        filtered = []
        for e in entities:
            match = False
            if search_names and query in e.name.lower():
                match = True
            if search_descriptions and query in e.description.lower():
                match = True
            if match:
                filtered.append(e)
        return filtered

    def test_search_names_only(self):
        """Test searching only in names."""
        entities = self._make_entities_for_search()

        # "dragon" is in name of first entity and description of second
        result = self._search_entities(
            entities, "dragon", search_names=True, search_descriptions=False
        )

        assert len(result) == 1
        assert result[0].name == "Dragon"

    def test_search_descriptions_only(self):
        """Test searching only in descriptions."""
        entities = self._make_entities_for_search()

        # "dragon" is in name of first entity and description of second
        result = self._search_entities(
            entities, "dragon", search_names=False, search_descriptions=True
        )

        assert len(result) == 1
        assert result[0].name == "Castle"

    def test_search_both_names_and_descriptions(self):
        """Test searching in both names and descriptions."""
        entities = self._make_entities_for_search()

        # "dragon" is in name of first entity and description of second
        result = self._search_entities(
            entities, "dragon", search_names=True, search_descriptions=True
        )

        assert len(result) == 2
        names = [e.name for e in result]
        assert "Dragon" in names
        assert "Castle" in names

    def test_search_case_insensitive(self):
        """Test that search is case-insensitive."""
        entities = self._make_entities_for_search()

        result = self._search_entities(
            entities, "DRAGON", search_names=True, search_descriptions=True
        )

        assert len(result) == 2

    def test_search_no_results(self):
        """Test search with no matching results."""
        entities = self._make_entities_for_search()

        result = self._search_entities(
            entities, "unicorn", search_names=True, search_descriptions=True
        )

        assert len(result) == 0

    def test_empty_search_returns_all(self):
        """Test that empty search query returns all entities."""
        entities = self._make_entities_for_search()

        result = self._search_entities(entities, "", search_names=True, search_descriptions=True)

        assert len(result) == 3

    def test_search_neither_scope_returns_empty(self):
        """Test that searching with both scopes disabled returns no results."""
        entities = self._make_entities_for_search()

        result = self._search_entities(
            entities, "dragon", search_names=False, search_descriptions=False
        )

        assert len(result) == 0


class TestAppStateFilters:
    """Tests for AppState filter/sort fields."""

    def test_default_quality_filter(self):
        """Test that default quality filter is 'all'."""
        from src.ui.state import AppState

        state = AppState()
        assert state.entity_quality_filter == "all"

    def test_default_sort_by(self):
        """Test that default sort is by name."""
        from src.ui.state import AppState

        state = AppState()
        assert state.entity_sort_by == "name"

    def test_default_sort_direction(self):
        """Test that default sort direction is ascending."""
        from src.ui.state import AppState

        state = AppState()
        assert state.entity_sort_descending is False

    def test_default_search_scopes(self):
        """Test that default search scopes are both enabled."""
        from src.ui.state import AppState

        state = AppState()
        assert state.entity_search_names is True
        assert state.entity_search_descriptions is True
