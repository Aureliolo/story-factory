"""Tests for the EmbeddingService — core methods.

Covers embedding generation, truncation, entity/relationship/event embedding,
internal methods, context-length retry, and failure tracking.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ollama
import pytest

from src.memory.entities import Entity, Relationship, WorldEvent
from src.services import embedding_service as embedding_service_mod
from src.services.embedding_service import (
    _FALLBACK_CONTEXT_TOKENS,
    _MIN_CONTENT_LENGTH,
    EmbeddingService,
)
from src.settings import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_warned_context_models():
    """Clear the module-level warned set before each test."""
    embedding_service_mod._warned_context_models.clear()
    yield
    embedding_service_mod._warned_context_models.clear()


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

    def test_embed_text_no_model_configured(self, service, caplog):
        """Returns empty list when embedding model is cleared after construction (ValueError path)."""
        service.settings.embedding_model = ""
        with caplog.at_level(logging.WARNING):
            result = service.embed_text("Some text")
        assert result == []
        assert any("configuration error" in msg.lower() for msg in caplog.messages)

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

    def test_embed_text_malformed_response_key_error(self, service):
        """Returns empty list when Ollama response is missing 'embedding' key."""
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {}  # Missing 'embedding' key

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []

    def test_embed_text_unexpected_exception_propagates(self, service):
        """Unexpected errors (programming bugs) propagate to caller instead of being swallowed."""
        mock_client = MagicMock()
        mock_client.embeddings.side_effect = RuntimeError("Unexpected failure")

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="Unexpected failure"),
        ):
            service.embed_text("Some text")


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

        long_text = "a" * 3000  # ~1500 tokens at 2 chars/token, should exceed 512 limit
        fake_prefix = "search_document: "

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=fake_prefix),
        ):
            result = service.embed_text(long_text)

        assert result == FAKE_EMBEDDING
        # Verify the prompt was truncated to exactly max_chars = (512 - 10) * 2 = 1004
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        assert len(actual_prompt) == (512 - 10) * 2
        # Verify prefix is preserved at start of truncated prompt
        assert actual_prompt.startswith(fake_prefix)

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

    def test_fallback_context_limit_when_none(self, service):
        """Truncation uses _FALLBACK_CONTEXT_TOKENS when context size is unavailable."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        # Use text that exceeds the fallback limit
        long_text = "x" * 3000

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=None),
        ):
            result = service.embed_text(long_text)

        assert result == FAKE_EMBEDDING
        call_kwargs = mock_client.embeddings.call_args
        actual_prompt = call_kwargs.kwargs["prompt"]
        # Should be truncated to (_FALLBACK_CONTEXT_TOKENS - 10) * 2 = 1004
        assert len(actual_prompt) == (_FALLBACK_CONTEXT_TOKENS - 10) * 2

    def test_fallback_context_logs_warning(self, service, caplog):
        """Logs a warning when fallback context limit is used."""
        mock_client = MagicMock()
        response = MagicMock()
        response.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.return_value = response

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            service.embed_text("Some text")

        assert any("fallback" in msg.lower() for msg in caplog.messages)

    def test_fallback_context_warning_only_once(self, service, caplog):
        """Logs fallback warning only once per model, not on every call."""
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": FAKE_EMBEDDING}

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            service.embed_text("First call")
            first_count = sum(1 for msg in caplog.messages if "fallback" in msg.lower())
            service.embed_text("Second call")
            second_count = sum(1 for msg in caplog.messages if "fallback" in msg.lower())

        assert first_count == 1
        assert second_count == 1  # No additional warning on second call

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
        """Returns empty list and logs error when context_limit <= 10."""
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

    def test_get_model_context_size_exception_propagates(self, service):
        """Unexpected exceptions from get_model_context_size propagate to caller."""
        mock_client = MagicMock()

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch(
                "src.services.embedding_service.get_model_context_size",
                side_effect=RuntimeError("API exploded"),
            ),
            pytest.raises(RuntimeError, match="API exploded"),
        ):
            service.embed_text("Some text")

    def test_context_limit_too_small_for_prefix_returns_empty(self, service, caplog):
        """Returns empty list when max_chars is smaller than the prefix length."""
        mock_client = MagicMock()
        # context_limit=15 → max_chars = (15-10)*2 = 10, prefix_len=11 → guard triggers
        long_prefix = "x" * 11

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=15),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=long_prefix),
            caplog.at_level(logging.ERROR),
        ):
            result = service.embed_text("Some text")

        assert result == []
        mock_client.embeddings.assert_not_called()
        assert any("too small to embed" in msg.lower() for msg in caplog.messages)

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
# embed_text context-length retry tests
# ---------------------------------------------------------------------------


class TestEmbedTextContextRetry:
    """Tests for the context-length overflow retry logic in embed_text."""

    def test_context_length_error_retries_with_halved_input(self, service):
        """On context-length ResponseError, retries with halved prompt and succeeds."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("input length exceeds context length", status_code=500)
        response_ok = MagicMock()
        response_ok.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        # First call raises context-length error, second (retry) succeeds
        mock_client.embeddings.side_effect = [context_error, response_ok]

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text that is long enough")

        assert result == FAKE_EMBEDDING
        assert mock_client.embeddings.call_count == 2
        # The retry prompt should be shorter than the original
        retry_call = mock_client.embeddings.call_args_list[1]
        original_call = mock_client.embeddings.call_args_list[0]
        assert len(retry_call.kwargs["prompt"]) < len(original_call.kwargs["prompt"])

    def test_context_length_error_retry_also_fails(self, service):
        """When both the original and retry fail, returns empty list."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("input length exceeds context length", status_code=500)
        retry_error = ollama.ResponseError("still too long", status_code=500)
        mock_client.embeddings.side_effect = [context_error, retry_error]

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text that keeps failing")

        assert result == []
        assert mock_client.embeddings.call_count == 2

    def test_non_context_error_does_not_retry(self, service):
        """Non-context ResponseError does not trigger retry logic."""
        mock_client = MagicMock()
        generic_error = ollama.ResponseError("model not found", status_code=404)
        mock_client.embeddings.side_effect = generic_error

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []
        # Should only have been called once (no retry)
        mock_client.embeddings.assert_called_once()

    def test_context_length_retry_preserves_prefix(self, service):
        """Halved prompt still starts with the embedding prefix."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("context length exceeded", status_code=500)
        response_ok = MagicMock()
        response_ok.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.side_effect = [context_error, response_ok]

        fake_prefix = "search_document: "

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=fake_prefix),
        ):
            result = service.embed_text("a" * 500)

        assert result == FAKE_EMBEDDING
        retry_prompt = mock_client.embeddings.call_args_list[1].kwargs["prompt"]
        assert retry_prompt.startswith(fake_prefix)

    def test_context_length_retry_returns_empty_embedding(self, service):
        """When retry succeeds but returns empty embedding, returns empty list."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("input length exceeds context length", status_code=500)
        response_empty = MagicMock()
        response_empty.__getitem__ = lambda self, key: [] if key == "embedding" else None
        mock_client.embeddings.side_effect = [context_error, response_empty]

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == []
        assert mock_client.embeddings.call_count == 2

    def test_context_length_retry_unexpected_error_propagates(self, service):
        """Unexpected errors during retry propagate instead of being swallowed."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("input length exceeds context length", status_code=500)
        mock_client.embeddings.side_effect = [context_error, TypeError("bad type")]

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            pytest.raises(TypeError, match="bad type"),
        ):
            service.embed_text("Some text that triggers retry")

    def test_context_length_pattern_matching_case_insensitive(self, service):
        """Context-length error matching is case-insensitive on the error message."""
        mock_client = MagicMock()
        context_error = ollama.ResponseError("INPUT LENGTH exceeds CONTEXT LENGTH", status_code=500)
        response_ok = MagicMock()
        response_ok.__getitem__ = lambda self, key: FAKE_EMBEDDING if key == "embedding" else None
        mock_client.embeddings.side_effect = [context_error, response_ok]

        with patch.object(service, "_get_client", return_value=mock_client):
            result = service.embed_text("Some text")

        assert result == FAKE_EMBEDDING
        assert mock_client.embeddings.call_count == 2


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
        with patch.object(disabled_service, "embed_text", return_value=[]):
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
        with patch.object(disabled_service, "embed_text", return_value=[]):
            result = disabled_service.embed_relationship(
                mock_db, sample_relationship, "Alice", "Bob"
            )

        assert result is False
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_relationship_embed_text_fails(self, service, mock_db, sample_relationship):
        """Returns False when embed_text returns an empty vector."""
        with patch.object(service, "embed_text", return_value=[]):
            result = service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert result is False
        mock_db.upsert_embedding.assert_not_called()

    def test_embed_relationship_truncates_long_description(self, service, mock_db, caplog):
        """Description is truncated while preserving entity names when it exceeds budget."""
        long_rel = Relationship(
            id="rel-long",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="allied_with",
            description="x" * 5000,  # Very long description
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=128),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_relationship(mock_db, long_rel, "Alice", "Bob")

        assert result is True
        # The text passed to embed_text should be shorter than 5000 chars
        embedded_text = mock_embed_text.call_args[0][0]
        assert len(embedded_text) < 5000
        assert embedded_text.startswith("Alice allied_with Bob: ")
        assert any("Truncating relationship description" in r.message for r in caplog.records)

    def test_embed_relationship_empty_description_when_header_exceeds_budget(
        self, service, mock_db, caplog
    ):
        """Description is cleared when overhead equals budget but fits max_chars."""
        rel = Relationship(
            id="rel-overflow",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="allied_with",
            description="Some description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "A allied_with B: " (17 chars)
        # prefix = "x" * 43 → overhead = 43 + 17 = 60
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # available_for_desc = 60 - 60 = 0 → budget exhausted
        # overhead (60) == max_chars (60), NOT > → embed with empty description
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 43),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_relationship(mock_db, rel, "A", "B")

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text.endswith(": ")
        assert any("exceeds embedding budget" in r.message for r in caplog.records)

    def test_embed_relationship_skips_when_overhead_exceeds_max_chars(
        self, service, mock_db, caplog
    ):
        """Embedding is skipped when prefix + header exceeds max_chars."""
        rel = Relationship(
            id="rel-skip",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="allied_with",
            description="Some description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "A allied_with B: " (17 chars)
        # prefix = "x" * 50 → overhead = 50 + 17 = 67
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # overhead (67) > max_chars (60) → skip embedding
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 50),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_relationship(mock_db, rel, "A", "B")

        assert result is False
        mock_embed_text.assert_not_called()
        assert any("skipping embedding to avoid truncating" in r.message for r in caplog.records)

    def test_embed_relationship_skips_when_header_alone_exceeds_max_chars(
        self, service, mock_db, caplog
    ):
        """Embedding is skipped when header alone exceeds max_chars (even without prefix)."""
        rel = Relationship(
            id="rel-skip-header",
            source_id="ent-001",
            target_id="ent-002",
            relation_type="allied_with",
            description="Some description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            # Tiny context so header alone exceeds max_chars
            patch("src.services.embedding_service.get_model_context_size", return_value=1),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_relationship(mock_db, rel, "Alice", "Bob")

        assert result is False
        mock_embed_text.assert_not_called()
        assert any("skipping embedding to avoid truncating" in r.message for r in caplog.records)


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
        with patch.object(disabled_service, "embed_text", return_value=[]):
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


# ---------------------------------------------------------------------------
# Failure tracking tests
# ---------------------------------------------------------------------------


class TestFailureTracking:
    """Tests for the _record_failure / _clear_failure / _failed_source_ids infrastructure."""

    def test_record_and_clear_failure(self, service):
        """Recording a failure adds to the set; clearing removes it."""
        service._record_failure("ent-001")
        assert "ent-001" in service._failed_source_ids

        service._clear_failure("ent-001")
        assert "ent-001" not in service._failed_source_ids

    def test_embed_entity_failure_records_source_id(self, service, mock_db, sample_entity):
        """embed_entity records the entity ID in the failure set when embedding fails."""
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_entity(mock_db, sample_entity)

        assert sample_entity.id in service._failed_source_ids

    def test_embed_entity_success_clears_failure(self, service, mock_db, sample_entity):
        """embed_entity clears the entity ID from the failure set on success."""
        service._record_failure(sample_entity.id)
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_entity(mock_db, sample_entity)

        assert sample_entity.id not in service._failed_source_ids

    def test_embed_relationship_failure_records_source_id(
        self, service, mock_db, sample_relationship
    ):
        """embed_relationship records the rel ID in the failure set when embedding fails."""
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert sample_relationship.id in service._failed_source_ids

    def test_embed_event_failure_records_source_id(self, service, mock_db, sample_event):
        """embed_event records the event ID in the failure set when embedding fails."""
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_event(mock_db, sample_event)

        assert sample_event.id in service._failed_source_ids

    def test_embed_relationship_success_clears_failure(self, service, mock_db, sample_relationship):
        """embed_relationship clears the rel ID from the failure set on success."""
        service._record_failure(sample_relationship.id)
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert sample_relationship.id not in service._failed_source_ids

    def test_embed_event_success_clears_failure(self, service, mock_db, sample_event):
        """embed_event clears the event ID from the failure set on success."""
        service._record_failure(sample_event.id)
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_event(mock_db, sample_event)

        assert sample_event.id not in service._failed_source_ids

    def test_embed_entity_upsert_false_clears_failure(self, service, mock_db, sample_entity):
        """embed_entity clears failure even when upsert returns False (unchanged content)."""
        service._record_failure(sample_entity.id)
        mock_db.upsert_embedding.return_value = False
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_entity(mock_db, sample_entity)

        assert sample_entity.id not in service._failed_source_ids

    def test_embed_relationship_upsert_false_clears_failure(
        self, service, mock_db, sample_relationship
    ):
        """embed_relationship clears failure even when upsert returns False."""
        service._record_failure(sample_relationship.id)
        mock_db.upsert_embedding.return_value = False
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_relationship(mock_db, sample_relationship, "Alice", "Bob")

        assert sample_relationship.id not in service._failed_source_ids

    def test_embed_event_upsert_false_clears_failure(self, service, mock_db, sample_event):
        """embed_event clears failure even when upsert returns False."""
        service._record_failure(sample_event.id)
        mock_db.upsert_embedding.return_value = False
        with patch.object(service, "embed_text", return_value=FAKE_EMBEDDING):
            service.embed_event(mock_db, sample_event)

        assert sample_event.id not in service._failed_source_ids

    def test_clear_failure_logs_debug(self, service, caplog):
        """_clear_failure emits a debug log when removing a source_id."""
        service._record_failure("ent-001")
        with caplog.at_level(logging.DEBUG):
            service._clear_failure("ent-001")

        assert any("cleared failure marker" in msg.lower() for msg in caplog.messages)

    def test_constants_exported(self):
        """Verify new constants are accessible."""
        assert _FALLBACK_CONTEXT_TOKENS == 512
        assert _MIN_CONTENT_LENGTH == 10


class TestStoryStateFailureTracking:
    """Tests for failure tracking in embed_story_state_data edge cases."""

    def test_rule_embed_text_failure_records_failure(self, service, mock_db):
        """Records failure when embed_text returns empty for a world rule."""
        story_state = SimpleNamespace(
            established_facts=[],
            world_rules=["A valid world rule for testing."],
            chapters=[],
        )
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_story_state_data(mock_db, story_state)
        assert len(service._failed_source_ids) == 1

    def test_chapter_outline_embed_text_failure_records_failure(self, service, mock_db):
        """Records failure when embed_text returns empty for a chapter outline."""
        chapter = SimpleNamespace(
            number=1, title="Ch1", outline="A sufficiently long chapter outline.", scenes=[]
        )
        story_state = SimpleNamespace(established_facts=[], world_rules=[], chapters=[chapter])
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_story_state_data(mock_db, story_state)
        assert "chapter:1" in service._failed_source_ids

    def test_scene_outline_embed_text_failure_records_failure(self, service, mock_db):
        """Records failure when embed_text returns empty for a scene outline."""
        scene = SimpleNamespace(title="S1", outline="A sufficiently long scene outline.")
        chapter = SimpleNamespace(
            number=1,
            title="Ch1",
            outline="A sufficiently long chapter outline.",
            scenes=[scene],
        )
        story_state = SimpleNamespace(established_facts=[], world_rules=[], chapters=[chapter])
        with patch.object(service, "embed_text", return_value=[]):
            service.embed_story_state_data(mock_db, story_state)
        assert "scene:1:0" in service._failed_source_ids
