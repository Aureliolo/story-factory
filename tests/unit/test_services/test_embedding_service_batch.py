"""Tests for the EmbeddingService — batch, story state, and callback methods.

Covers embed_story_state_data, embed_all_world_data, check_and_reembed_if_needed,
attach_to_database, and the bug fixes for content length validation (Bug 3)
and batch skip/retry logic (Bug 2).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.memory.entities import Entity, Relationship, WorldEvent
from src.services.embedding_service import EmbeddingService
from src.settings import Settings


@pytest.fixture
def embedding_settings():
    """Create a Settings instance with RAG context enabled and a fake embedding model."""
    settings = Settings()
    settings.rag_context_enabled = True
    settings.embedding_model = "test-embed:latest"
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_generate_timeout = 30.0
    return settings


@pytest.fixture
def disabled_settings():
    """Create a Settings instance with RAG context disabled."""
    settings = Settings()
    settings.rag_context_enabled = False
    settings.embedding_model = "test-embed:latest"
    return settings


@pytest.fixture
def service(embedding_settings):
    """Create an EmbeddingService with RAG enabled."""
    return EmbeddingService(embedding_settings)


@pytest.fixture
def disabled_service(disabled_settings):
    """Create an EmbeddingService with RAG disabled."""
    return EmbeddingService(disabled_settings)


@pytest.fixture
def mock_db():
    """Create a mock WorldDatabase with common methods stubbed."""
    db = MagicMock()
    db.upsert_embedding.return_value = True
    db.list_entities.return_value = []
    db.list_relationships.return_value = []
    db.list_events.return_value = []
    db.get_entity.return_value = None
    db.needs_reembedding.return_value = False
    db.get_embedded_source_ids.return_value = set()
    db.attach_content_changed_callback = MagicMock()
    db.attach_content_deleted_callback = MagicMock()
    db.recreate_vec_table = MagicMock()
    db.clear_all_embeddings = MagicMock()
    db.delete_embedding = MagicMock()
    return db


@pytest.fixture
def sample_entity():
    """Create a sample Entity for embedding tests."""
    return Entity(
        id="ent-001",
        type="character",
        name="Alice",
        description="A brave adventurer who seeks the truth",
    )


@pytest.fixture
def sample_relationship():
    """Create a sample Relationship for embedding tests."""
    return Relationship(
        id="rel-001",
        source_id="ent-001",
        target_id="ent-002",
        relation_type="allied_with",
        description="They have been allies since childhood",
    )


@pytest.fixture
def sample_event():
    """Create a sample WorldEvent without a chapter number."""
    return WorldEvent(
        id="evt-001",
        description="The great battle of the northern plains",
    )


@pytest.fixture
def sample_event_with_chapter():
    """Create a sample WorldEvent with a chapter number."""
    return WorldEvent(
        id="evt-002",
        description="The hero confronts the villain",
        chapter_number=3,
    )


FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


class TestEmbedStoryStateData:
    """Tests for the embed_story_state_data method."""

    def test_embed_story_state_data_with_facts(self, service, mock_db):
        """Embeds established facts from story state and counts them."""
        story_state = SimpleNamespace(
            established_facts=["The world is flat.", "Magic is real."],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 2
        assert mock_db.upsert_embedding.call_count == 2
        # Verify the first fact has a source_id starting with "fact:"
        first_call_kwargs = mock_db.upsert_embedding.call_args_list[0].kwargs
        assert first_call_kwargs["source_id"].startswith("fact:")
        assert first_call_kwargs["content_type"] == "fact"
        assert "The world is flat" in first_call_kwargs["text"]

    def test_embed_story_state_data_with_world_rules(self, service, mock_db):
        """Embeds world rules from story state."""
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=["No magic after midnight."],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 1
        call_kwargs = mock_db.upsert_embedding.call_args_list[0].kwargs
        assert call_kwargs["source_id"].startswith("rule:")
        assert call_kwargs["content_type"] == "rule"
        assert "No magic after midnight" in call_kwargs["text"]

    def test_embed_story_state_data_with_chapters(self, service, mock_db):
        """Embeds chapter outlines from story state."""
        chapter = SimpleNamespace(
            number=1,
            title="The Beginning",
            outline="The hero sets out on a journey.",
            scenes=[],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 1
        call_kwargs = mock_db.upsert_embedding.call_args_list[0].kwargs
        assert call_kwargs["source_id"] == "chapter:1"
        assert call_kwargs["content_type"] == "chapter_outline"
        assert "'The Beginning'" in call_kwargs["text"]
        assert call_kwargs["chapter_number"] == 1

    def test_embed_story_state_data_with_scenes(self, service, mock_db):
        """Embeds scene outlines within chapters."""
        scene = SimpleNamespace(
            title="Opening Scene",
            outline="A dark and stormy night.",
        )
        chapter = SimpleNamespace(
            number=2,
            title="",
            outline="The middle of the story.",
            scenes=[scene],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        # 1 chapter outline + 1 scene outline
        assert count == 2
        scene_call = mock_db.upsert_embedding.call_args_list[1].kwargs
        assert scene_call["source_id"] == "scene:2:0"
        assert scene_call["content_type"] == "scene_outline"
        assert "'Opening Scene'" in scene_call["text"]

    def test_embed_story_state_data_chapter_no_title(self, service, mock_db):
        """Embeds chapter outline without a title (no title part in text)."""
        chapter = SimpleNamespace(
            number=1,
            title="",
            outline="An outline without a title.",
            scenes=[],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 1
        call_kwargs = mock_db.upsert_embedding.call_args_list[0].kwargs
        assert "'" not in call_kwargs["text"]
        assert "Chapter 1:" in call_kwargs["text"]

    def test_embed_story_state_data_scene_no_title(self, service, mock_db):
        """Embeds scene outline where scene has an empty title."""
        scene = SimpleNamespace(
            outline="Something happens here.",
            title="",
        )
        chapter = SimpleNamespace(
            number=1,
            title="Ch1",
            outline="Chapter one outline.",
            scenes=[scene],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        # 1 chapter + 1 scene
        assert count == 2
        scene_call = mock_db.upsert_embedding.call_args_list[1].kwargs
        # No title in scene text since scene title is empty
        assert "'" not in scene_call["text"]

    def test_embed_story_state_data_chapter_no_outline(self, service, mock_db):
        """Skips chapters without an outline."""
        chapter = SimpleNamespace(
            number=1,
            title="Empty",
            outline="",
            scenes=[],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_story_state_data_scene_no_outline(self, service, mock_db):
        """Skips scenes without an outline."""
        scene = SimpleNamespace(outline="")
        chapter = SimpleNamespace(
            number=1,
            title="Ch1",
            outline="Has an outline that is long enough.",
            scenes=[scene],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        # Only the chapter outline, scene skipped
        assert count == 1

    def test_embed_story_state_data_embed_text_fails(self, service, mock_db):
        """Returns 0 and records failure when embed_text returns empty vectors."""
        story_state = SimpleNamespace(
            established_facts=["A fact that fails to embed."],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=[]):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()
        assert len(service._failed_source_ids) == 1

    def test_embed_story_state_data_not_available(self, disabled_service, mock_db):
        """Returns 0 when the embedding service is not available."""
        story_state = SimpleNamespace(
            established_facts=["Some fact that is long enough."],
            world_rules=[],
            chapters=[],
        )

        with patch.object(disabled_service, "embed_text", return_value=[]):
            count = disabled_service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_story_state_data_no_facts_or_rules_or_chapters(self, service, mock_db):
        """Handles story state with no embeddable content attributes."""
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0

    def test_embed_story_state_data_upsert_returns_false(self, service, mock_db):
        """Does not increment count when upsert_embedding returns False (unchanged content)."""
        story_state = SimpleNamespace(
            established_facts=["Already embedded fact."],
            world_rules=[],
            chapters=[],
        )
        mock_db.upsert_embedding.return_value = False

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0

    def test_empty_fact_skipped(self, service, mock_db, caplog):
        """Empty string fact is not embedded."""
        story_state = SimpleNamespace(
            established_facts=[""],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()
        assert any("too-short" in msg.lower() for msg in caplog.messages)

    def test_short_fact_skipped(self, service, mock_db, caplog):
        """A 5-character fact is below _MIN_CONTENT_LENGTH and not embedded."""
        story_state = SimpleNamespace(
            established_facts=["Hello"],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()

    def test_empty_world_rule_skipped(self, service, mock_db, caplog):
        """Empty rule is not embedded."""
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[""],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()
        assert any("too-short" in msg.lower() for msg in caplog.messages)

    def test_short_chapter_outline_skipped_and_stale_deleted(self, service, mock_db, caplog):
        """Short chapter outline is skipped and stale embedding is deleted."""
        chapter = SimpleNamespace(
            number=1,
            title="Ch1",
            outline="Short",
            scenes=[],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()
        mock_db.delete_embedding.assert_called_once_with("chapter:1")
        assert any("too-short" in msg.lower() for msg in caplog.messages)

    def test_short_scene_outline_skipped_and_stale_deleted(self, service, mock_db, caplog):
        """Short scene outline is skipped and stale embedding is deleted."""
        scene = SimpleNamespace(
            title="S1",
            outline="Tiny",
        )
        chapter = SimpleNamespace(
            number=1,
            title="Ch1",
            outline="A sufficiently long chapter outline.",
            scenes=[scene],
        )
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[chapter],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            count = service.embed_story_state_data(mock_db, story_state)

        # Only the chapter outline is embedded, scene is skipped
        assert count == 1
        mock_db.delete_embedding.assert_called_once_with("scene:1:0")
        assert any("too-short" in msg.lower() for msg in caplog.messages)

    def test_minimum_length_content_passes(self, service, mock_db):
        """Content with exactly _MIN_CONTENT_LENGTH (10) characters IS embedded."""
        story_state = SimpleNamespace(
            established_facts=["0123456789"],  # exactly 10 chars
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 1
        mock_db.upsert_embedding.assert_called_once()

    def test_mixed_valid_and_invalid_facts(self, service, mock_db):
        """Only valid-length facts are embedded; short ones are skipped."""
        story_state = SimpleNamespace(
            established_facts=[
                "",  # too short
                "Short",  # too short (5 chars)
                "A valid fact that should be embedded.",  # valid
                "12345",  # too short (5 chars)
                "Another valid fact for embedding!",  # valid
            ],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 2
        assert mock_db.upsert_embedding.call_count == 2


class TestEmbedAllWorldData:
    """Tests for the embed_all_world_data method."""

    def test_embed_all_world_data(self, service, mock_db, sample_entity, sample_relationship):
        """Embeds all world content and returns counts by type."""
        entity2 = Entity(id="ent-002", type="location", name="Castle", description="A dark castle")
        source_entity = Entity(id="ent-001", type="character", name="Alice", description="The hero")
        target_entity = Entity(
            id="ent-002", type="location", name="Castle", description="A dark castle"
        )
        event = WorldEvent(id="evt-001", description="The siege begins")

        mock_db.list_entities.return_value = [sample_entity, entity2]
        mock_db.list_relationships.return_value = [sample_relationship]
        mock_db.list_events.return_value = [event]
        mock_db.get_entity.side_effect = lambda eid: {
            "ent-001": source_entity,
            "ent-002": target_entity,
        }.get(eid)

        story_state = SimpleNamespace(
            established_facts=["A fact that is long enough."],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["entity"] == 2
        assert counts["relationship"] == 1
        assert counts["event"] == 1
        assert counts["story_state"] == 1

    def test_embed_all_world_data_empty_world(self, service, mock_db):
        """Returns zero counts when the world has no entities, relationships, or events."""
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["entity"] == 0
        assert counts["relationship"] == 0
        assert counts["event"] == 0
        assert counts["story_state"] == 0

    def test_embed_all_world_data_with_progress_callback(self, service, mock_db, sample_entity):
        """Calls progress callback for each entity during batch embedding."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )
        progress_cb = MagicMock()

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_all_world_data(mock_db, story_state, progress_callback=progress_cb)

        progress_cb.assert_called_once_with(0, 1, "Embedding entity: Alice")

    def test_embed_all_world_data_relationship_missing_entity(
        self, service, mock_db, sample_relationship, caplog
    ):
        """Logs warning and tracks failure for relationships with missing entities."""
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = [sample_relationship]
        mock_db.list_events.return_value = []
        mock_db.get_entity.return_value = None

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.WARNING),
        ):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["relationship"] == 0
        assert any("missing" in msg.lower() for msg in caplog.messages)
        assert sample_relationship.id in service._failed_source_ids

    def test_batch_skips_already_embedded_items(self, service, mock_db, sample_entity):
        """Skips items that already have embeddings for the current model."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []
        # Mark the entity as already embedded
        mock_db.get_embedded_source_ids.return_value = {sample_entity.id}

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        # embed_entity should NOT have been called (entity was skipped)
        assert counts["entity"] == 0
        # embed_text should not have been called for entity embedding
        # (it may still be called by _get_model but not for embed_entity)
        mock_db.upsert_embedding.assert_not_called()

    def test_batch_retries_failed_inline_items(self, service, mock_db, sample_entity):
        """Items in the failure set are retried even if they have existing embeddings."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []
        # Entity is both "already embedded" AND in the failure set
        mock_db.get_embedded_source_ids.return_value = {sample_entity.id}
        service._record_failure(sample_entity.id)

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        # Should have been retried and succeeded
        assert counts["entity"] == 1

    def test_batch_logs_still_failed_items(self, service, mock_db, sample_entity, caplog):
        """Logs a warning listing items that still fail after retry."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        # embed_text returns [] so embed_entity fails
        with (
            patch.object(service, "embed_text", return_value=[]),
            caplog.at_level(logging.WARNING),
        ):
            service.embed_all_world_data(mock_db, story_state)

        assert any("still without embeddings" in msg.lower() for msg in caplog.messages)

    def test_batch_logs_skipped_when_all_embedded(self, service, mock_db, sample_entity, caplog):
        """Logs that items were skipped when all are already embedded."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []
        mock_db.get_embedded_source_ids.return_value = {sample_entity.id}

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            caplog.at_level(logging.INFO),
        ):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["entity"] == 0
        assert any("skipped" in msg.lower() for msg in caplog.messages)

    def test_batch_skips_already_embedded_relationships(self, service, mock_db):
        """Skips relationships that already have embeddings for the current model."""
        rel = Relationship(
            id="rel-001",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="allied_with",
            description="Allies since childhood",
        )
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = [rel]
        mock_db.list_events.return_value = []
        mock_db.get_embedded_source_ids.return_value = {"rel-001"}

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["relationship"] == 0

    def test_batch_skips_already_embedded_events(self, service, mock_db):
        """Skips events that already have embeddings for the current model."""
        event = WorldEvent(id="evt-001", description="The siege begins")
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = [event]
        mock_db.get_embedded_source_ids.return_value = {"evt-001"}

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["event"] == 0

    def test_batch_tracks_still_failed_relationships(self, service, mock_db, caplog):
        """Relationships that fail embedding are tracked in still_failed."""
        source = Entity(id="ent-001", type="character", name="Alice", description="Hero")
        target = Entity(id="ent-002", type="character", name="Bob", description="Villain")
        rel = Relationship(
            id="rel-001",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="enemy_of",
            description="They are enemies",
        )
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = [rel]
        mock_db.list_events.return_value = []
        mock_db.get_entity.side_effect = lambda eid: {"ent-001": source, "ent-002": target}.get(eid)

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=[]),
            caplog.at_level(logging.WARNING),
        ):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["relationship"] == 0
        assert any("still without embeddings" in msg.lower() for msg in caplog.messages)

    def test_batch_tracks_still_failed_events(self, service, mock_db, caplog):
        """Events that fail embedding are tracked in still_failed."""
        event = WorldEvent(id="evt-001", description="The great event")
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = [event]

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with (
            patch.object(service, "embed_text", return_value=[]),
            caplog.at_level(logging.WARNING),
        ):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["event"] == 0
        assert any("still without embeddings" in msg.lower() for msg in caplog.messages)

    def test_batch_preserves_still_failed_in_failure_set(self, service, mock_db, sample_entity):
        """Items that fail during the batch run remain in the failure set."""
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=[]):
            service.embed_all_world_data(mock_db, story_state)

        assert sample_entity.id in service._failed_source_ids

    def test_batch_clears_succeeded_from_failure_set(self, service, mock_db, sample_entity):
        """Previously-failed items that succeed during batch are removed from the failure set."""
        service._record_failure(sample_entity.id)
        mock_db.list_entities.return_value = [sample_entity]
        mock_db.list_relationships.return_value = []
        mock_db.list_events.return_value = []

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_all_world_data(mock_db, story_state)

        assert sample_entity.id not in service._failed_source_ids


class TestCheckAndReembedIfNeeded:
    """Tests for the check_and_reembed_if_needed method."""

    def test_check_and_reembed_if_needed_model_changed(self, service, mock_db):
        """Triggers full re-embedding when the model has changed."""
        mock_db.needs_reembedding.return_value = True
        story_state = SimpleNamespace()

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            patch.object(service, "embed_all_world_data", return_value={"entity": 3}) as mock_batch,
        ):
            result = service.check_and_reembed_if_needed(mock_db, story_state)

        assert result is True
        mock_db.recreate_vec_table.assert_called_once_with(len(FAKE_EMBEDDING))
        mock_batch.assert_called_once_with(mock_db, story_state)

    def test_check_and_reembed_logs_warning_on_zero_items(self, service, mock_db, caplog):
        """Logs warning when re-embedding produces zero items."""
        mock_db.needs_reembedding.return_value = True
        story_state = SimpleNamespace()

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            patch.object(
                service,
                "embed_all_world_data",
                return_value={"entity": 0, "relationship": 0, "event": 0, "story_state": 0},
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = service.check_and_reembed_if_needed(mock_db, story_state)

        assert result is True
        assert any("zero items were embedded" in msg.lower() for msg in caplog.messages)

    def test_check_and_reembed_if_needed_model_changed_sample_fails(self, service, mock_db):
        """Keeps existing embeddings when model is unreachable (sample embed fails)."""
        mock_db.needs_reembedding.return_value = True
        story_state = SimpleNamespace()

        with (
            patch.object(service, "embed_text", return_value=[]),
            patch.object(service, "embed_all_world_data") as mock_batch,
        ):
            result = service.check_and_reembed_if_needed(mock_db, story_state)

        assert result is False
        mock_db.clear_all_embeddings.assert_not_called()
        mock_db.recreate_vec_table.assert_not_called()
        mock_batch.assert_not_called()

    def test_check_and_reembed_if_needed_not_needed(self, service, mock_db):
        """Returns False when the model has not changed."""
        mock_db.needs_reembedding.return_value = False
        story_state = SimpleNamespace()

        result = service.check_and_reembed_if_needed(mock_db, story_state)

        assert result is False
        mock_db.recreate_vec_table.assert_not_called()
        mock_db.clear_all_embeddings.assert_not_called()

    def test_check_and_reembed_if_needed_no_model(self, service, mock_db):
        """Raises ValueError when embedding model is cleared after construction."""
        service.settings.embedding_model = ""
        story_state = SimpleNamespace()

        with pytest.raises(ValueError, match="No embedding model configured"):
            service.check_and_reembed_if_needed(mock_db, story_state)


class TestAttachToDatabase:
    """Tests for the attach_to_database method."""

    def test_attach_to_database(self, service, mock_db):
        """Registers content changed and deleted callbacks on the database."""
        service.attach_to_database(mock_db)

        mock_db.attach_content_changed_callback.assert_called_once()
        mock_db.attach_content_deleted_callback.assert_called_once()

        # Verify the callbacks are callable
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]
        on_deleted = mock_db.attach_content_deleted_callback.call_args[0][0]
        assert callable(on_changed)
        assert callable(on_deleted)

    def test_attach_to_database_always_registers(self, disabled_service, mock_db):
        """Registers callbacks even when rag_context_enabled is False (embedding is mandatory)."""
        disabled_service.attach_to_database(mock_db)

        mock_db.attach_content_changed_callback.assert_called_once()
        mock_db.attach_content_deleted_callback.assert_called_once()

    def test_attached_on_content_changed_callback_embeds(self, service, mock_db):
        """The content changed callback embeds new content and upserts it."""
        entity = Entity(
            id="ent-test",
            type="character",
            name="Test",
            description="A test entity",
        )
        mock_db.get_entity.return_value = entity

        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            on_changed("ent-test", "entity", "Test: A test entity")

        mock_db.upsert_embedding.assert_called_once_with(
            source_id="ent-test",
            content_type="entity",
            text="Test: A test entity",
            embedding=FAKE_EMBEDDING,
            model="test-embed:latest",
            entity_type="character",
        )

    def test_attached_on_content_changed_non_entity_type(self, service, mock_db):
        """The content changed callback sets empty entity_type for non-entity content."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            on_changed("rel-001", "relationship", "Alice knows Bob")

        call_kwargs = mock_db.upsert_embedding.call_args.kwargs
        assert call_kwargs["entity_type"] == ""

    def test_attached_on_content_changed_embed_fails(self, service, mock_db):
        """The content changed callback does not upsert when embed_text returns empty."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", return_value=[]):
            on_changed("ent-test", "entity", "Some text")

        mock_db.upsert_embedding.assert_not_called()

    def test_attached_on_content_changed_records_failure(self, service, mock_db):
        """The content changed callback records failure when embed_text returns empty."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]
        with patch.object(service, "embed_text", return_value=[]):
            on_changed("ent-test", "entity", "Some text")
        assert "ent-test" in service._failed_source_ids

    def test_attached_on_content_changed_clears_failure_on_success(self, service, mock_db):
        """The content changed callback clears failure on successful embedding."""
        service._record_failure("ent-test")
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            on_changed("ent-test", "entity", "Some text")
        assert "ent-test" not in service._failed_source_ids

    def test_attached_on_content_changed_exception_handled(self, service, mock_db):
        """The content changed callback handles exceptions gracefully."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", side_effect=RuntimeError("Boom")):
            # Should not raise
            on_changed("ent-test", "entity", "Some text")

    def test_attached_on_content_changed_exception_records_failure(self, service, mock_db):
        """The content changed callback records failure when exception occurs."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", side_effect=RuntimeError("Boom")):
            on_changed("ent-test", "entity", "Some text")

        assert "ent-test" in service._failed_source_ids

    def test_attached_on_content_deleted_callback(self, service, mock_db):
        """The content deleted callback removes the embedding from the database."""
        service.attach_to_database(mock_db)
        on_deleted = mock_db.attach_content_deleted_callback.call_args[0][0]
        on_deleted("ent-test")
        mock_db.delete_embedding.assert_called_once_with("ent-test")

    def test_attached_on_content_deleted_exception_handled(self, service, mock_db):
        """The content deleted callback handles exceptions gracefully."""
        mock_db.delete_embedding.side_effect = RuntimeError("DB error")

        service.attach_to_database(mock_db)
        on_deleted = mock_db.attach_content_deleted_callback.call_args[0][0]

        # Should not raise
        on_deleted("ent-test")
