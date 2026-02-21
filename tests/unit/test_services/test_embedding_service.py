"""Tests for the EmbeddingService.

Covers embedding generation, entity/relationship/event embedding, story state
data embedding, batch re-embedding, model change detection, and database
callback attachment.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.memory.entities import Entity, Relationship, WorldEvent
from src.services.embedding_service import EmbeddingService
from src.settings import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# embed_text tests
# ---------------------------------------------------------------------------


class TestEmbedText:
    """Tests for the embed_text method."""

    def test_embed_text_success(self, service):
        """Successfully generates an embedding vector via the Ollama client."""
        mock_client = MagicMock()
        # Support both dict and attribute access per CLAUDE.md guidelines
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        response.embedding = FAKE_EMBEDDING
        mock_client.embeddings.return_value = response

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Hello world")

        assert result == FAKE_EMBEDDING
        mock_client.embeddings.assert_called_once()

    def test_embed_text_empty_input(self, service):
        """Returns empty list when given empty or whitespace-only text."""
        assert service.embed_text("") == []
        assert service.embed_text("   ") == []

    def test_embed_text_ollama_error(self, service):
        """Returns empty list when Ollama connection fails."""
        mock_client = MagicMock()
        mock_client.embeddings.side_effect = ConnectionError("Connection refused")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []

    def test_embed_text_no_model_configured(self, service):
        """Returns empty list when embedding model is cleared after construction (ValueError path)."""
        service.settings.embedding_model = ""
        result = service.embed_text("Some text")
        assert result == []

    def test_embed_text_timeout_error(self, service):
        """Returns empty list when Ollama request times out."""
        mock_client = MagicMock()
        mock_client.embeddings.side_effect = TimeoutError("Request timed out")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []

    def test_embed_text_empty_embedding_returned(self, service):
        """Returns empty list when Ollama returns an empty embedding vector."""
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": []}

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []

    def test_embed_text_unexpected_exception(self, service):
        """Returns empty list on unexpected errors (generic Exception path)."""
        mock_client = MagicMock()
        mock_client.embeddings.side_effect = RuntimeError("Unexpected failure")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []


# ---------------------------------------------------------------------------
# embed_text truncation tests
# ---------------------------------------------------------------------------


class TestEmbedTextTruncation:
    """Tests for embedding input truncation when text exceeds model context limit."""

    def test_text_truncated_when_exceeds_context_limit(self, service):
        """Truncates prompt when estimated tokens exceed model context size."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        long_text = "a" * 3000  # ~750 tokens, should exceed 512 limit

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
        ):
            result = service.embed_text(long_text)

        assert result == FAKE_EMBEDDING
        # Verify the prompt was truncated to exactly max_chars = (512 - 10) * 2 = 1004
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        assert len(actual_prompt) == (512 - 10) * 2
        # Verify prefix is preserved at start of truncated prompt
        from src.settings._model_registry import get_embedding_prefix

        expected_prefix = get_embedding_prefix("test-embed:latest")
        assert actual_prompt.startswith(expected_prefix)

    def test_text_not_truncated_when_within_limit(self, service):
        """Short text is passed through unchanged when within context limit."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        short_text = "Hello world"

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
        ):
            result = service.embed_text(short_text)

        assert result == FAKE_EMBEDDING
        # Prompt should contain the full text (with prefix)
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        assert short_text in actual_prompt

    def test_no_truncation_when_context_size_unavailable(self, service):
        """Text is passed as-is when model context size is unavailable (None)."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        long_text = "b" * 5000

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=None),
        ):
            result = service.embed_text(long_text)

        assert result == FAKE_EMBEDDING
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        assert long_text in actual_prompt

    def test_truncation_logs_warning(self, service, caplog):
        """Logs a warning when truncation occurs."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        long_text = "c" * 3000

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            caplog.at_level(logging.WARNING),
        ):
            service.embed_text(long_text)

        assert any("truncated" in msg.lower() for msg in caplog.messages)

    def test_truncation_accounts_for_embedding_prefix(self, service):
        """Truncation applies to the combined prefix+text prompt, not just raw text."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        fake_prefix = "search_document: "  # 18 chars
        # Text that fits alone but exceeds limit when prefix is included
        max_chars = (512 - 10) * 2  # 1004
        text_len = max_chars - len(fake_prefix) + 100  # 100 chars over with prefix
        long_text = "x" * text_len

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=fake_prefix),
        ):
            result = service.embed_text(long_text)

        assert result == FAKE_EMBEDDING
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        # Prompt should be truncated to max_chars and start with the prefix
        assert len(actual_prompt) == max_chars
        assert actual_prompt.startswith(fake_prefix)

    def test_tiny_context_limit_returns_empty(self, service, caplog):
        """Returns empty list and logs warning when context_limit <= 10."""
        mock_client = MagicMock()

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=5),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_text("Some text")

        assert result == []
        mock_client.embeddings.assert_not_called()
        assert any("invalid context size" in msg.lower() for msg in caplog.messages)

    def test_get_model_context_size_exception_returns_empty(self, service):
        """Falls back to empty list when get_model_context_size raises an unexpected exception."""
        mock_client = MagicMock()

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch(
                "src.services.embedding_service.get_model_context_size",
                side_effect=RuntimeError("API exploded"),
            ),
        ):
            result = service.embed_text("Some text")

        # Should be caught by the generic except Exception handler
        assert result == []

    def test_context_limit_exactly_10_returns_empty(self, service, caplog):
        """context_limit == 10 is treated as invalid and returns empty."""
        mock_client = MagicMock()

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=10),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_text("Some text")

        assert result == []
        mock_client.embeddings.assert_not_called()

    def test_context_limit_11_proceeds_with_embedding(self, service):
        """context_limit == 11 is valid and proceeds with embedding."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=11),
        ):
            result = service.embed_text("Hi")

        assert result == FAKE_EMBEDDING
        mock_client.embeddings.assert_called_once()


# ---------------------------------------------------------------------------
# embed_entity tests
# ---------------------------------------------------------------------------


class TestEmbedEntity:
    """Tests for the embed_entity method."""

    def test_embed_entity_success(self, service, mock_db, sample_entity):
        """Successfully embeds an entity and stores it in the database."""
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            result = service.embed_entity(mock_db, sample_entity)

        assert result is True
        mock_db.upsert_embedding.assert_called_once_with(
            source_id="ent-001",
            content_type="entity",
            text="Alice: A brave adventurer who seeks the truth",
            embedding=FAKE_EMBEDDING,
            model="test-embed:latest",
            entity_type="character",
        )

    def test_embed_entity_not_available(self, disabled_service, mock_db, sample_entity):
        """Returns False when the embedding service is not available."""
        result = disabled_service.embed_entity(mock_db, sample_entity)

        assert result is False
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_entity_embed_text_fails(self, service, mock_db, sample_entity):
        """Returns False when embed_text returns an empty vector."""
        with patch.object(service, "embed_text", return_value=[]):
            result = service.embed_entity(mock_db, sample_entity)

        assert result is False
        mock_db.upsert_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# embed_relationship tests
# ---------------------------------------------------------------------------


class TestEmbedRelationship:
    """Tests for the embed_relationship method."""

    def test_embed_relationship_success(self, service, mock_db, sample_relationship):
        """Successfully embeds a relationship and stores it in the database."""
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            result = service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert result is True
        mock_db.upsert_embedding.assert_called_once_with(
            source_id="rel-001",
            content_type="relationship",
            text="Alice allied_with Bob: They have been allies since childhood",
            embedding=FAKE_EMBEDDING,
            model="test-embed:latest",
        )

    def test_embed_relationship_not_available(self, disabled_service, mock_db, sample_relationship):
        """Returns False when the embedding service is not available."""
        result = disabled_service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert result is False
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_relationship_embed_text_fails(self, service, mock_db, sample_relationship):
        """Returns False when embed_text returns an empty vector."""
        with patch.object(service, "embed_text", return_value=[]):
            result = service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert result is False
        mock_db.upsert_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# embed_event tests
# ---------------------------------------------------------------------------


class TestEmbedEvent:
    """Tests for the embed_event method."""

    def test_embed_event_success(self, service, mock_db, sample_event):
        """Successfully embeds an event without chapter number."""
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            result = service.embed_event(mock_db, sample_event)

        assert result is True
        mock_db.upsert_embedding.assert_called_once_with(
            source_id="evt-001",
            content_type="event",
            text="Event: The great battle of the northern plains",
            embedding=FAKE_EMBEDDING,
            model="test-embed:latest",
            chapter_number=None,
        )

    def test_embed_event_with_chapter(self, service, mock_db, sample_event_with_chapter):
        """Successfully embeds an event with a chapter number appended."""
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            result = service.embed_event(mock_db, sample_event_with_chapter)

        assert result is True
        mock_db.upsert_embedding.assert_called_once_with(
            source_id="evt-002",
            content_type="event",
            text="Event: The hero confronts the villain (Chapter 3)",
            embedding=FAKE_EMBEDDING,
            model="test-embed:latest",
            chapter_number=3,
        )

    def test_embed_event_not_available(self, disabled_service, mock_db, sample_event):
        """Returns False when the embedding service is not available."""
        result = disabled_service.embed_event(mock_db, sample_event)

        assert result is False
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_event_embed_text_fails(self, service, mock_db, sample_event):
        """Returns False when embed_text returns an empty vector."""
        with patch.object(service, "embed_text", return_value=[]):
            result = service.embed_event(mock_db, sample_event)

        assert result is False
        mock_db.upsert_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# embed_story_state_data tests
# ---------------------------------------------------------------------------


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
            outline="Has an outline.",
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
        """Returns 0 when embed_text always returns empty vectors."""
        story_state = SimpleNamespace(
            established_facts=["A fact that fails to embed."],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=[]):
            count = service.embed_story_state_data(mock_db, story_state)

        assert count == 0
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_story_state_data_not_available(self, disabled_service, mock_db):
        """Returns 0 when the embedding service is not available."""
        story_state = SimpleNamespace(
            established_facts=["Some fact"],
            world_rules=[],
            chapters=[],
        )

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


# ---------------------------------------------------------------------------
# embed_all_world_data tests
# ---------------------------------------------------------------------------


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
            established_facts=["A fact."],
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
        self, service, mock_db, sample_relationship
    ):
        """Skips relationships where source or target entity is not found."""
        mock_db.list_entities.return_value = []
        mock_db.list_relationships.return_value = [sample_relationship]
        mock_db.list_events.return_value = []
        mock_db.get_entity.return_value = None

        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=[],
            chapters=[],
        )

        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            counts = service.embed_all_world_data(mock_db, story_state)

        assert counts["relationship"] == 0


# ---------------------------------------------------------------------------
# check_and_reembed_if_needed tests
# ---------------------------------------------------------------------------


class TestCheckAndReembedIfNeeded:
    """Tests for the check_and_reembed_if_needed method."""

    def test_check_and_reembed_if_needed_model_changed(self, service, mock_db):
        """Triggers full re-embedding when the model has changed."""
        mock_db.needs_reembedding.return_value = True
        story_state = SimpleNamespace()

        with (
            patch.object(service, "embed_text", return_value=FAKE_EMBEDDING),
            patch.object(service, "embed_all_world_data") as mock_batch,
        ):
            result = service.check_and_reembed_if_needed(mock_db, story_state)

        assert result is True
        mock_db.recreate_vec_table.assert_called_once_with(len(FAKE_EMBEDDING))
        mock_batch.assert_called_once_with(mock_db, story_state)

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


# ---------------------------------------------------------------------------
# attach_to_database tests
# ---------------------------------------------------------------------------


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

    def test_attached_on_content_changed_exception_handled(self, service, mock_db):
        """The content changed callback handles exceptions gracefully."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        with patch.object(service, "embed_text", side_effect=RuntimeError("Boom")):
            # Should not raise
            on_changed("ent-test", "entity", "Some text")

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


# ---------------------------------------------------------------------------
# _get_client / _get_model internal methods
# ---------------------------------------------------------------------------


class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_get_client_creates_once(self, service):
        """The Ollama client is lazily created and reused on subsequent calls."""
        with patch("src.services.embedding_service.ollama.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            service._client = None  # Reset to force creation
            client1 = service._get_client()
            client2 = service._get_client()
            assert client1 is client2
            mock_cls.assert_called_once()

    def test_get_model_returns_stripped_name(self, service):
        """Returns the embedding model name with whitespace stripped."""
        service.settings.embedding_model = "  test-embed:latest  "
        assert service._get_model() == "test-embed:latest"

    def test_get_model_raises_when_empty(self, service):
        """Raises ValueError when embedding model is cleared after construction."""
        service.settings.embedding_model = ""
        with pytest.raises(ValueError, match="No embedding model configured"):
            service._get_model()

    def test_constructor_fails_fast_on_empty_model(self):
        """Constructor raises ValueError when embedding_model is empty."""
        settings = Settings()
        settings.embedding_model = ""
        with pytest.raises(
            ValueError, match="EmbeddingService requires a configured embedding_model"
        ):
            EmbeddingService(settings)

    def test_constructor_fails_fast_on_whitespace_model(self):
        """Constructor raises ValueError when embedding_model is whitespace-only."""
        settings = Settings()
        settings.embedding_model = "   "
        with pytest.raises(
            ValueError, match="EmbeddingService requires a configured embedding_model"
        ):
            EmbeddingService(settings)
