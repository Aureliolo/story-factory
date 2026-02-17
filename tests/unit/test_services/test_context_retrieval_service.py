"""Unit tests for ContextRetrievalService.

Tests the RAG-based context retrieval service that provides semantically
relevant world context for LLM prompts. Covers vector retrieval, fallback
to legacy context, graph expansion, token budgeting, deduplication,
similarity threshold filtering, and content type filtering.
"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from src.memory.entities import Entity, Relationship
from src.services.context_retrieval_service import (
    ContextItem,
    ContextRetrievalService,
    RetrievedContext,
)
from src.settings import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR = [0.1] * 1024


def _make_search_result(
    source_id: str,
    content_type: str = "entity",
    display_text: str = "Test entity description",
    distance: float = 0.2,
    entity_type: str = "character",
    chapter_number: int | None = None,
) -> dict:
    """Build a dict matching the shape returned by WorldDatabase.search_similar."""
    return {
        "source_id": source_id,
        "content_type": content_type,
        "entity_type": entity_type,
        "chapter_number": chapter_number,
        "display_text": display_text,
        "distance": distance,
    }


def _make_entity(entity_id: str, name: str, description: str = "A test entity") -> Entity:
    """Build a minimal Entity instance for testing."""
    return Entity(
        id=entity_id,
        type="character",
        name=name,
        description=description,
    )


def _make_relationship(
    rel_id: str,
    source_id: str,
    target_id: str,
    relation_type: str = "knows",
    description: str = "A test relationship",
) -> Relationship:
    """Build a minimal Relationship instance for testing."""
    return Relationship(
        id=rel_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        description=description,
    )


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance with RAG enabled and sensible test defaults."""
    settings = Settings()
    settings.rag_context_enabled = True
    settings.rag_context_max_tokens = 2000
    settings.rag_context_max_items = 20
    settings.rag_context_similarity_threshold = 0.3
    settings.rag_context_graph_expansion = False
    settings.rag_context_graph_depth = 1
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _make_embedding_service(available: bool = True, embed_result: list | None = None):
    """Create a mock EmbeddingService."""
    mock = MagicMock()
    type(mock).is_available = PropertyMock(return_value=available)
    mock.embed_text.return_value = embed_result if embed_result is not None else FAKE_VECTOR
    return mock


def _make_world_db(
    vec_available: bool = True,
    search_results: list | None = None,
    context_for_agents: dict | None = None,
    connected_entities: list | None = None,
    relationships: list | None = None,
    entity_map: dict | None = None,
):
    """Create a mock WorldDatabase with configurable return values."""
    mock = MagicMock()
    type(mock).vec_available = PropertyMock(return_value=vec_available)
    mock.search_similar.return_value = search_results if search_results is not None else []
    mock.get_context_for_agents.return_value = context_for_agents or {
        "characters": [],
        "locations": [],
        "items": [],
        "factions": [],
        "key_relationships": [],
        "recent_events": [],
        "entity_counts": {
            "characters": 0,
            "locations": 0,
            "items": 0,
            "factions": 0,
            "concepts": 0,
        },
    }
    mock.get_connected_entities.return_value = connected_entities or []
    mock.get_relationships.return_value = relationships or []

    if entity_map:
        mock.get_entity.side_effect = lambda eid: entity_map.get(eid)
    else:
        mock.get_entity.return_value = None

    return mock


# ---------------------------------------------------------------------------
# Tests for ContextItem dataclass
# ---------------------------------------------------------------------------


class TestContextItem:
    """Tests for the ContextItem dataclass."""

    def test_context_item_token_estimate(self) -> None:
        """Token estimate is auto-calculated from text length divided by 4."""
        text = "A" * 100  # 100 chars => 25 tokens
        item = ContextItem(
            source_type="entity",
            source_id="ent-1",
            relevance_score=0.9,
            text=text,
        )
        assert item.token_estimate == 25

    def test_context_item_token_estimate_minimum_one(self) -> None:
        """Token estimate is at least 1, even for very short text."""
        item = ContextItem(
            source_type="entity",
            source_id="ent-1",
            relevance_score=0.9,
            text="Hi",
        )
        # len("Hi") == 2, 2 // 4 == 0, max(1, 0) == 1
        assert item.token_estimate == 1

    def test_context_item_explicit_token_estimate_preserved(self) -> None:
        """When token_estimate is explicitly provided and non-zero, it is kept."""
        item = ContextItem(
            source_type="entity",
            source_id="ent-1",
            relevance_score=0.9,
            text="Hello world",
            token_estimate=999,
        )
        assert item.token_estimate == 999


# ---------------------------------------------------------------------------
# Tests for RetrievedContext dataclass
# ---------------------------------------------------------------------------


class TestRetrievedContext:
    """Tests for the RetrievedContext dataclass."""

    def test_retrieved_context_format_empty(self) -> None:
        """format_for_prompt returns empty string when no items exist."""
        ctx = RetrievedContext(items=[])
        assert ctx.format_for_prompt() == ""

    def test_retrieved_context_format_groups_by_type(self) -> None:
        """format_for_prompt groups items by source_type with section headers."""
        items = [
            ContextItem(
                source_type="entity",
                source_id="e1",
                relevance_score=0.9,
                text="Alice: The protagonist",
            ),
            ContextItem(
                source_type="relationship",
                source_id="r1",
                relevance_score=0.8,
                text="Alice knows Bob: Best friends",
            ),
            ContextItem(
                source_type="entity",
                source_id="e2",
                relevance_score=0.7,
                text="Bob: Supporting character",
            ),
            ContextItem(
                source_type="event",
                source_id="ev1",
                relevance_score=0.6,
                text="The battle began",
            ),
        ]
        ctx = RetrievedContext(items=items)
        result = ctx.format_for_prompt()

        # Entities should be grouped under RELEVANT ENTITIES
        assert "RELEVANT ENTITIES:" in result
        assert "- Alice: The protagonist" in result
        assert "- Bob: Supporting character" in result

        # Relationships under their own section
        assert "RELEVANT RELATIONSHIPS:" in result
        assert "- Alice knows Bob: Best friends" in result

        # Events under their section
        assert "RELEVANT EVENTS:" in result
        assert "- The battle began" in result

    def test_retrieved_context_format_unknown_type_uses_uppercase(self) -> None:
        """format_for_prompt uses uppercased source_type when no title mapping exists."""
        items = [
            ContextItem(
                source_type="custom_thing",
                source_id="c1",
                relevance_score=0.5,
                text="Something custom",
            ),
        ]
        ctx = RetrievedContext(items=items)
        result = ctx.format_for_prompt()
        assert "CUSTOM_THING:" in result

    def test_retrieved_context_default_values(self) -> None:
        """RetrievedContext has correct default values."""
        ctx = RetrievedContext()
        assert ctx.items == []
        assert ctx.total_tokens == 0
        assert ctx.retrieval_method == "vector"


# ---------------------------------------------------------------------------
# Tests for ContextRetrievalService
# ---------------------------------------------------------------------------


class TestContextRetrievalService:
    """Tests for the ContextRetrievalService class."""

    def test_retrieve_context_rag_disabled(self, sample_story_state) -> None:
        """Returns empty context with method='disabled' when RAG is turned off."""
        settings = _make_settings(rag_context_enabled=False)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db()

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "disabled"
        assert result.items == []
        assert result.total_tokens == 0

    def test_retrieve_context_vector_path(self, sample_story_state) -> None:
        """Uses vector retrieval when embedding_service is available and db has vec."""
        search_results = [
            _make_search_result("ent-alice", "entity", "Alice: The protagonist", distance=0.1),
            _make_search_result("ent-bob", "entity", "Bob: A friend", distance=0.3),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=True, embed_result=FAKE_VECTOR)
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write about Alice and Bob",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "vector"
        embedding_svc.embed_text.assert_called_once_with("Write about Alice and Bob")
        world_db.search_similar.assert_called_once()

        # Should contain the search results plus project info
        source_ids = {item.source_id for item in result.items}
        assert "ent-alice" in source_ids
        assert "ent-bob" in source_ids
        assert "project:brief" in source_ids

    def test_retrieve_context_fallback_path(self, sample_story_state) -> None:
        """Falls back to legacy context when vec is not available."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=False)
        world_db = _make_world_db(
            vec_available=False,
            context_for_agents={
                "characters": [{"name": "Alice", "description": "Hero"}],
                "locations": [],
                "key_relationships": [],
                "recent_events": [],
            },
        )

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "fallback"
        world_db.get_context_for_agents.assert_called_once()

        # Should have the character item plus project info
        source_ids = {item.source_id for item in result.items}
        assert "fallback:char:Alice" in source_ids
        assert "project:brief" in source_ids

    def test_retrieve_context_fallback_legacy_data(self, sample_story_state) -> None:
        """Legacy data (characters, locations, relationships, events) is converted correctly."""
        legacy_data = {
            "characters": [
                {"name": "Alice", "description": "The protagonist"},
                {"name": "Bob", "description": "A friend"},
            ],
            "locations": [
                {"name": "Forest", "description": "A dark forest"},
            ],
            "key_relationships": [
                {
                    "from": "Alice",
                    "to": "Bob",
                    "type": "knows",
                    "description": "Best friends",
                },
            ],
            "recent_events": [
                {"chapter": 1, "description": "Alice discovered her powers"},
            ],
        }
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=False)
        world_db = _make_world_db(vec_available=False, context_for_agents=legacy_data)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 2",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "fallback"

        items_by_id = {item.source_id: item for item in result.items}

        # Characters converted
        alice_item = items_by_id["fallback:char:Alice"]
        assert alice_item.source_type == "entity"
        assert "Alice: The protagonist" in alice_item.text
        assert alice_item.relevance_score == 0.5

        bob_item = items_by_id["fallback:char:Bob"]
        assert "Bob: A friend" in bob_item.text

        # Location converted
        forest_item = items_by_id["fallback:loc:Forest"]
        assert forest_item.source_type == "entity"
        assert "Forest: A dark forest" in forest_item.text
        assert forest_item.relevance_score == 0.4

        # Relationship converted
        rel_item = items_by_id["fallback:rel:Alice:Bob"]
        assert rel_item.source_type == "relationship"
        assert "Alice knows Bob: Best friends" in rel_item.text
        assert rel_item.relevance_score == 0.4

        # Event converted
        evt_item = items_by_id["fallback:evt:Alice discovered her powers"]
        assert evt_item.source_type == "event"
        assert "Event (Ch.1): Alice discovered her powers" in evt_item.text
        assert evt_item.relevance_score == 0.3

    def test_retrieve_context_fallback_event_no_chapter(self, sample_story_state) -> None:
        """Legacy events without a chapter key display '?' as the chapter number."""
        legacy_data = {
            "characters": [],
            "locations": [],
            "key_relationships": [],
            "recent_events": [
                {"description": "Something happened"},
            ],
        }
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=False)
        world_db = _make_world_db(vec_available=False, context_for_agents=legacy_data)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Continue the story",
            world_db=world_db,
            story_state=sample_story_state,
        )

        event_items = [i for i in result.items if i.source_type == "event"]
        assert len(event_items) == 1
        assert "Event (Ch.?)" in event_items[0].text

    def test_retrieve_context_vector_with_graph_expansion(self, sample_story_state) -> None:
        """Graph expansion adds neighbors and relationships for entity results."""
        search_results = [
            _make_search_result("ent-alice", "entity", "Alice: The protagonist", distance=0.1),
        ]
        neighbor = _make_entity("ent-bob", "Bob", "Alice's friend")
        relationship = _make_relationship("rel-1", "ent-alice", "ent-bob", "knows", "Best friends")
        alice_entity = _make_entity("ent-alice", "Alice", "The protagonist")
        bob_entity = _make_entity("ent-bob", "Bob", "Alice's friend")

        settings = _make_settings(rag_context_graph_expansion=True, rag_context_graph_depth=1)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(
            vec_available=True,
            search_results=search_results,
            connected_entities=[neighbor],
            relationships=[relationship],
            entity_map={"ent-alice": alice_entity, "ent-bob": bob_entity},
        )

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write about Alice",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "vector"

        source_ids = {item.source_id for item in result.items}
        # Original entity
        assert "ent-alice" in source_ids
        # Expanded neighbor
        assert "ent-bob" in source_ids
        # Expanded relationship
        assert "rel-1" in source_ids

        # Neighbor relevance is 70% of parent's
        bob_item = next(i for i in result.items if i.source_id == "ent-bob")
        alice_item = next(i for i in result.items if i.source_id == "ent-alice")
        assert bob_item.relevance_score == pytest.approx(alice_item.relevance_score * 0.7)

        # Relationship text includes entity names
        rel_item = next(i for i in result.items if i.source_id == "rel-1")
        assert "Alice knows Bob: Best friends" in rel_item.text

    def test_retrieve_context_graph_expansion_skips_existing_neighbors(
        self, sample_story_state
    ) -> None:
        """Graph expansion does not overwrite items already found by vector search."""
        search_results = [
            _make_search_result("ent-alice", "entity", "Alice: The protagonist", distance=0.1),
            _make_search_result("ent-bob", "entity", "Bob: A friend", distance=0.15),
        ]
        # Bob is both a search result and a neighbor -- the search result should win
        neighbor = _make_entity("ent-bob", "Bob", "Expanded version")

        settings = _make_settings(rag_context_graph_expansion=True)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(
            vec_available=True,
            search_results=search_results,
            connected_entities=[neighbor],
            relationships=[],
        )

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write about Alice",
            world_db=world_db,
            story_state=sample_story_state,
        )

        bob_items = [i for i in result.items if i.source_id == "ent-bob"]
        assert len(bob_items) == 1
        # Should keep the original vector-retrieved text, not the expanded version
        assert "Bob: A friend" in bob_items[0].text

    def test_retrieve_context_graph_expansion_failure_handled(self, sample_story_state) -> None:
        """Graph expansion exceptions are caught and logged without crashing."""
        search_results = [
            _make_search_result("ent-alice", "entity", "Alice: The protagonist", distance=0.1),
        ]
        settings = _make_settings(rag_context_graph_expansion=True)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)
        world_db.get_connected_entities.side_effect = RuntimeError("Graph error")

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write about Alice",
            world_db=world_db,
            story_state=sample_story_state,
        )

        # Should still return a result despite graph expansion failure
        assert result.retrieval_method == "vector"
        source_ids = {item.source_id for item in result.items}
        assert "ent-alice" in source_ids

    def test_retrieve_context_graph_expansion_relationship_missing_entity(
        self, sample_story_state
    ) -> None:
        """Relationships where source or target entity is missing are skipped."""
        search_results = [
            _make_search_result("ent-alice", "entity", "Alice: The protagonist", distance=0.1),
        ]
        relationship = _make_relationship(
            "rel-orphan", "ent-alice", "ent-missing", "knows", "Unknown"
        )
        alice_entity = _make_entity("ent-alice", "Alice", "The protagonist")

        settings = _make_settings(rag_context_graph_expansion=True)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(
            vec_available=True,
            search_results=search_results,
            connected_entities=[],
            relationships=[relationship],
            entity_map={"ent-alice": alice_entity},  # ent-missing not in map
        )

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write about Alice",
            world_db=world_db,
            story_state=sample_story_state,
        )

        # The orphan relationship should not appear because target entity is None
        source_ids = {item.source_id for item in result.items}
        assert "rel-orphan" not in source_ids

    def test_retrieve_context_token_budgeting(self, sample_story_state) -> None:
        """Items are packed greedily until max_tokens is reached, skipping large items."""
        # Create items whose total exceeds budget: each ~25 tokens (100 chars / 4)
        search_results = [
            _make_search_result(f"ent-{i}", "entity", "X" * 100, distance=0.1 + i * 0.01)
            for i in range(10)
        ]

        settings = _make_settings(rag_context_max_tokens=60)  # Budget for ~2 items + brief
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
            max_tokens=60,
        )

        # Total tokens should not exceed the budget
        assert result.total_tokens <= 60
        # Not all 10 items should fit
        assert len(result.items) < 10

    def test_retrieve_context_similarity_threshold_filtering(self, sample_story_state) -> None:
        """Items below the similarity threshold are filtered out."""
        search_results = [
            _make_search_result("ent-good", "entity", "Relevant entity", distance=0.1),
            _make_search_result("ent-bad", "entity", "Irrelevant entity", distance=0.9),
        ]
        # threshold = 0.3; relevance = 1 - distance
        # ent-good: relevance = 0.9 (passes)
        # ent-bad: relevance = 0.1 (filtered out)
        settings = _make_settings(rag_context_similarity_threshold=0.3)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Find relevant entities",
            world_db=world_db,
            story_state=sample_story_state,
        )

        source_ids = {item.source_id for item in result.items}
        assert "ent-good" in source_ids
        assert "ent-bad" not in source_ids

    def test_retrieve_context_deduplication(self, sample_story_state) -> None:
        """Duplicate source_ids keep the entry with the highest relevance score."""
        search_results = [
            _make_search_result("ent-dup", "entity", "First occurrence", distance=0.3),
            _make_search_result("ent-dup", "entity", "Second occurrence", distance=0.1),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Deduplicate test",
            world_db=world_db,
            story_state=sample_story_state,
        )

        dup_items = [i for i in result.items if i.source_id == "ent-dup"]
        assert len(dup_items) == 1
        # distance=0.1 => relevance=0.9, distance=0.3 => relevance=0.7
        # Should keep the higher relevance (0.9)
        assert dup_items[0].relevance_score == pytest.approx(0.9)

    def test_retrieve_context_deduplication_first_wins_text(self, sample_story_state) -> None:
        """When a duplicate has higher relevance, its score is updated but text stays."""
        # First occurrence seen: distance=0.3 -> relevance=0.7, text="First"
        # Second occurrence: distance=0.1 -> relevance=0.9, updates score only
        search_results = [
            _make_search_result("ent-dup", "entity", "First text", distance=0.3),
            _make_search_result("ent-dup", "entity", "Second text", distance=0.1),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Deduplicate text test",
            world_db=world_db,
            story_state=sample_story_state,
        )

        dup_items = [i for i in result.items if i.source_id == "ent-dup"]
        assert len(dup_items) == 1
        # Text from the first occurrence is kept (only score is updated)
        assert "First text" in dup_items[0].text

    def test_retrieve_context_project_info_always_included(self, sample_story_state) -> None:
        """Base project info (premise, genre, tone, setting) is always in the result."""
        search_results = [
            _make_search_result("ent-1", "entity", "Some entity", distance=0.2),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        project_items = [i for i in result.items if i.source_id == "project:brief"]
        assert len(project_items) == 1

        pi = project_items[0]
        assert pi.source_type == "project_info"
        assert pi.relevance_score == 1.0
        assert "Premise:" in pi.text
        assert "Genre: Fantasy" in pi.text
        assert "Tone: Epic" in pi.text
        assert "Setting: Kingdom of Eldoria" in pi.text
        assert "Time period: Medieval" in pi.text

    def test_retrieve_context_content_type_filter(self, sample_story_state) -> None:
        """When content_types is specified, search_similar is called per type."""
        entity_results = [
            _make_search_result("ent-1", "entity", "An entity", distance=0.1),
        ]
        event_results = [
            _make_search_result("evt-1", "event", "An event", distance=0.2),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True)
        # Return different results per call
        world_db.search_similar.side_effect = [entity_results, event_results]

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Find entities and events",
            world_db=world_db,
            story_state=sample_story_state,
            content_types=["entity", "event"],
        )

        # search_similar should be called once per content type
        assert world_db.search_similar.call_count == 2

        source_ids = {item.source_id for item in result.items}
        assert "ent-1" in source_ids
        assert "evt-1" in source_ids

    def test_retrieve_context_content_type_filter_with_entity_types(
        self, sample_story_state
    ) -> None:
        """Entity type filter is passed through when exactly one entity type is given."""
        search_results = [
            _make_search_result("ent-1", "entity", "A character", distance=0.1),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Find characters",
            world_db=world_db,
            story_state=sample_story_state,
            content_types=["entity"],
            entity_types=["character"],
        )

        call_kwargs = world_db.search_similar.call_args[1]
        assert call_kwargs["entity_type"] == "character"

    def test_retrieve_context_entity_types_defaults_content_types_to_entity(
        self, sample_story_state
    ) -> None:
        """Passing entity_types without content_types defaults content_types to ['entity'].

        This ensures entity_type filter is applied even when content_types is omitted.
        """
        search_results = [
            _make_search_result("ent-1", "entity", "A location", distance=0.1),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Find locations",
            world_db=world_db,
            story_state=sample_story_state,
            entity_types=["location"],
        )

        call_kwargs = world_db.search_similar.call_args[1]
        assert call_kwargs["content_type"] == "entity"
        assert call_kwargs["entity_type"] == "location"

    def test_retrieve_context_multiple_entity_types_no_filter(self, sample_story_state) -> None:
        """Multiple entity types result in entity_type=None in the query."""
        search_results: list[dict] = []
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Find everything",
            world_db=world_db,
            story_state=sample_story_state,
            entity_types=["character", "location"],
        )

        call_kwargs = world_db.search_similar.call_args[1]
        # Multiple entity_types → entity_type=None (can't filter to one)
        assert call_kwargs["entity_type"] is None

    def test_retrieve_context_embed_failure_fallback(self, sample_story_state) -> None:
        """Falls back to legacy context when embed_text returns empty list."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=True, embed_result=[])
        world_db = _make_world_db(
            vec_available=True,
            context_for_agents={
                "characters": [{"name": "Alice", "description": "Fallback hero"}],
                "locations": [],
                "key_relationships": [],
                "recent_events": [],
            },
        )

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "fallback"
        source_ids = {item.source_id for item in result.items}
        assert "fallback:char:Alice" in source_ids

    def test_retrieve_context_vector_search_exception_fallback(self, sample_story_state) -> None:
        """Falls back to legacy context when search_similar raises an exception."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=True, embed_result=FAKE_VECTOR)
        world_db = _make_world_db(
            vec_available=True,
            context_for_agents={
                "characters": [{"name": "Bob", "description": "Fallback char"}],
                "locations": [],
                "key_relationships": [],
                "recent_events": [],
            },
        )
        world_db.search_similar.side_effect = RuntimeError("vec table corrupted")

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "fallback"
        source_ids = {item.source_id for item in result.items}
        assert "fallback:char:Bob" in source_ids

    def test_get_project_info_no_brief(self) -> None:
        """_get_project_info_items returns empty list when story state has no brief."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        service = ContextRetrievalService(settings, embedding_svc)

        # No story state at all
        result = service._get_project_info_items(None)
        assert result == []

        # Story state without brief attribute
        mock_state = MagicMock(spec=[])
        result = service._get_project_info_items(mock_state)
        assert result == []

        # Story state with brief=None
        mock_state = MagicMock()
        mock_state.brief = None
        result = service._get_project_info_items(mock_state)
        assert result == []

    def test_get_project_info_partial_brief(self) -> None:
        """_get_project_info_items handles a brief with only some fields populated."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        service = ContextRetrievalService(settings, embedding_svc)

        mock_state = MagicMock()
        mock_brief = MagicMock()
        mock_brief.premise = "A mystery"
        mock_brief.genre = ""
        mock_brief.tone = ""
        mock_brief.setting_place = ""
        mock_brief.setting_time = ""
        mock_state.brief = mock_brief

        result = service._get_project_info_items(mock_state)
        assert len(result) == 1
        assert "Premise: A mystery" in result[0].text
        assert "Genre:" not in result[0].text

    def test_get_project_info_no_setting_attributes(self) -> None:
        """_get_project_info_items works when brief has no setting_place/setting_time attrs."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        service = ContextRetrievalService(settings, embedding_svc)

        mock_state = MagicMock()
        mock_brief = MagicMock(spec=["premise", "genre", "tone"])
        mock_brief.premise = "A story"
        mock_brief.genre = "Sci-Fi"
        mock_brief.tone = "Dark"
        mock_state.brief = mock_brief

        result = service._get_project_info_items(mock_state)
        assert len(result) == 1
        assert "Premise: A story" in result[0].text
        assert "Genre: Sci-Fi" in result[0].text
        assert "Tone: Dark" in result[0].text
        # setting_place and setting_time not present
        assert "Setting:" not in result[0].text
        assert "Time period:" not in result[0].text

    def test_get_project_info_empty_parts(self) -> None:
        """_get_project_info_items returns empty list when all brief fields are empty."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        service = ContextRetrievalService(settings, embedding_svc)

        mock_state = MagicMock()
        mock_brief = MagicMock(spec=["premise", "genre", "tone"])
        mock_brief.premise = ""
        mock_brief.genre = ""
        mock_brief.tone = ""
        mock_state.brief = mock_brief

        result = service._get_project_info_items(mock_state)
        assert result == []

    def test_retrieve_context_uses_settings_defaults(self, sample_story_state) -> None:
        """max_tokens and k default to settings values when not provided."""
        settings = _make_settings(rag_context_max_tokens=500, rag_context_max_items=5)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=[])

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Test defaults",
            world_db=world_db,
            story_state=sample_story_state,
        )

        call_kwargs = world_db.search_similar.call_args[1]
        assert call_kwargs["k"] == 5

    def test_retrieve_context_custom_max_tokens_and_k(self, sample_story_state) -> None:
        """Explicit max_tokens and k parameters override settings defaults."""
        settings = _make_settings(rag_context_max_tokens=500, rag_context_max_items=5)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=[])

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Test overrides",
            world_db=world_db,
            story_state=sample_story_state,
            max_tokens=1000,
            k=50,
        )

        call_kwargs = world_db.search_similar.call_args[1]
        assert call_kwargs["k"] == 50

    def test_retrieve_context_chapter_number_passed(self, sample_story_state) -> None:
        """chapter_number is passed through to search_similar."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=[])

        service = ContextRetrievalService(settings, embedding_svc)
        service.retrieve_context(
            task_description="Write chapter 3",
            world_db=world_db,
            story_state=sample_story_state,
            chapter_number=3,
        )

        call_kwargs = world_db.search_similar.call_args[1]
        assert call_kwargs["chapter_number"] == 3

    def test_retrieve_context_negative_relevance_clamped(self, sample_story_state) -> None:
        """Distances greater than 1.0 produce relevance 0.0, which is below threshold."""
        search_results = [
            _make_search_result("ent-far", "entity", "Very far away", distance=1.5),
        ]
        settings = _make_settings(rag_context_similarity_threshold=0.3)
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Test clamping",
            world_db=world_db,
            story_state=sample_story_state,
        )

        # The entity with distance > 1.0 should be filtered (relevance = max(0, 1-1.5) = 0.0)
        entity_ids = {i.source_id for i in result.items if i.source_type == "entity"}
        assert "ent-far" not in entity_ids

    def test_retrieve_fallback_legacy_exception(self, sample_story_state) -> None:
        """Fallback returns empty context when get_context_for_agents raises."""
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=False)
        world_db = _make_world_db(vec_available=False)
        world_db.get_context_for_agents.side_effect = RuntimeError("DB error")

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        assert result.retrieval_method == "fallback"
        assert result.items == []
        assert result.total_tokens == 0

    def test_retrieve_context_fallback_token_budgeting(self, sample_story_state) -> None:
        """Fallback respects token budget, skipping items that would exceed it."""
        # Each character text "Name: Description" is about 5-10 tokens
        # Create enough characters to exceed a tiny budget
        chars = [{"name": f"Char{i}", "description": "A" * 200} for i in range(20)]
        legacy_data = {
            "characters": chars,
            "locations": [],
            "key_relationships": [],
            "recent_events": [],
        }
        settings = _make_settings()
        embedding_svc = _make_embedding_service(available=False)
        world_db = _make_world_db(vec_available=False, context_for_agents=legacy_data)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
            max_tokens=100,
        )

        assert result.total_tokens <= 100
        assert len(result.items) < 20

    def test_retrieve_context_display_text_none_handled(self, sample_story_state) -> None:
        """Search results with display_text=None are converted to empty string."""
        search_results = [
            {
                "source_id": "ent-null",
                "content_type": "entity",
                "entity_type": "character",
                "display_text": None,
                "distance": 0.1,
            },
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Handle None text",
            world_db=world_db,
            story_state=sample_story_state,
        )

        null_items = [i for i in result.items if i.source_id == "ent-null"]
        assert len(null_items) == 1
        assert null_items[0].text == ""

    def test_retrieve_context_project_info_not_duplicated(self, sample_story_state) -> None:
        """Project info source_id 'project:brief' is not duplicated when already in items."""
        # This covers the branch where project_info source_id is already in items_by_id
        # We simulate it by having a search result with source_id='project:brief'
        search_results = [
            _make_search_result(
                "project:brief", "project_info", "Existing brief info", distance=0.05
            ),
        ]
        settings = _make_settings()
        embedding_svc = _make_embedding_service()
        world_db = _make_world_db(vec_available=True, search_results=search_results)

        service = ContextRetrievalService(settings, embedding_svc)
        result = service.retrieve_context(
            task_description="Write chapter 1",
            world_db=world_db,
            story_state=sample_story_state,
        )

        brief_items = [i for i in result.items if i.source_id == "project:brief"]
        assert len(brief_items) == 1

    def test_retrieve_context_format_known_section_titles(self) -> None:
        """All known section titles in _SECTION_TITLES are used correctly."""
        from src.services.context_retrieval_service import _SECTION_TITLES

        items = []
        for content_type, _title in _SECTION_TITLES.items():
            items.append(
                ContextItem(
                    source_type=content_type,
                    source_id=f"test-{content_type}",
                    relevance_score=0.5,
                    text=f"Test {content_type} content",
                )
            )

        ctx = RetrievedContext(items=items)
        result = ctx.format_for_prompt()

        for content_type, title in _SECTION_TITLES.items():
            assert f"{title}:" in result
            assert f"- Test {content_type} content" in result
