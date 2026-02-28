"""Tests for pre-truncation in embed_entity, embed_event, and callback text.

Covers the description pre-truncation logic added to embed_entity, embed_event,
and the _pre_truncate_callback_text helper in attach_to_database callbacks.
These mirror the existing embed_relationship truncation tests.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.memory.entities import Entity, WorldEvent
from src.services import embedding_service as embedding_service_mod
from src.services.embedding_service import (
    _FALLBACK_CONTEXT_TOKENS,
    EmbeddingService,
)
from src.settings import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture(autouse=True)
def clear_warned_context_models():
    """Clear the module-level warned set before each test."""
    embedding_service_mod._warned_context_models.clear()
    yield
    embedding_service_mod._warned_context_models.clear()


@pytest.fixture
def embedding_settings():
    """Create a Settings instance with a fake embedding model."""
    settings = Settings()
    settings.rag_context_enabled = True
    settings.embedding_model = "test-embed:latest"
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_generate_timeout = 30.0
    return settings


@pytest.fixture
def service(embedding_settings):
    """Create an EmbeddingService with RAG enabled."""
    return EmbeddingService(embedding_settings)


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


# ---------------------------------------------------------------------------
# embed_entity pre-truncation tests
# ---------------------------------------------------------------------------


class TestEmbedEntityTruncation:
    """Tests for pre-truncation in embed_entity."""

    def test_embed_entity_truncates_long_description(self, service, mock_db, caplog):
        """Description is truncated while preserving entity name when it exceeds budget."""
        long_entity = Entity(
            id="ent-long",
            type="character",
            name="Alice",
            description="x" * 5000,
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
            result = service.embed_entity(mock_db, long_entity)

        assert result is True
        # The text passed to embed_text should be shorter than 5000 chars
        embedded_text = mock_embed_text.call_args[0][0]
        assert len(embedded_text) < 5000
        assert embedded_text.startswith("Alice: ")
        assert any("Truncating entity description" in r.message for r in caplog.records)

    def test_embed_entity_empty_description_when_header_meets_budget(
        self, service, mock_db, caplog
    ):
        """Description is cleared when overhead equals budget but fits max_chars."""
        entity = Entity(
            id="ent-overflow",
            type="location",
            name="A",
            description="Some description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "A: " (3 chars)
        # prefix = "x" * 57 → overhead = 57 + 3 = 60
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # available_for_desc = 60 - 60 = 0 → budget exhausted
        # overhead (60) == max_chars (60), NOT > → embed with empty description
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 57),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_entity(mock_db, entity)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text.endswith(": ")
        assert any("meets embedding budget" in r.message for r in caplog.records)

    def test_embed_entity_skips_when_overhead_exceeds_max_chars(self, service, mock_db, caplog):
        """Embedding is skipped when prefix + header exceeds max_chars."""
        entity = Entity(
            id="ent-skip",
            type="character",
            name="Alice",
            description="Some description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "Alice: " (7 chars)
        # prefix = "x" * 60 → overhead = 60 + 7 = 67
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # overhead (67) > max_chars (60) → skip embedding
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 60),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_entity(mock_db, entity)

        assert result is False
        mock_embed_text.assert_not_called()
        assert any("overhead exceeds max chars" in r.message for r in caplog.records)

    def test_embed_entity_skips_records_failure(self, service, mock_db):
        """Records failure when overhead exceeds max_chars and embedding is skipped."""
        entity = Entity(
            id="ent-skip-fail",
            type="character",
            name="Alice",
            description="Some description",
        )
        mock_client = MagicMock()
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 60),
            patch.object(service, "embed_text", MagicMock(return_value=FAKE_EMBEDDING)),
        ):
            service.embed_entity(mock_db, entity)

        assert "ent-skip-fail" in service._failed_source_ids

    def test_embed_entity_uses_fallback_when_context_size_none(self, service, mock_db, caplog):
        """embed_entity uses _FALLBACK_CONTEXT_TOKENS when context size is None."""
        entity = Entity(
            id="ent-fallback",
            type="character",
            name="Alice",
            description="A short description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        embedding_service_mod._warned_context_models.discard(service.settings.embedding_model)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch(
                "src.services.embedding_service.get_model_context_size",
                return_value=None,
            ),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="pfx:"),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_entity(mock_db, entity)

        assert result is True
        mock_embed_text.assert_called_once()
        fallback_value = str(_FALLBACK_CONTEXT_TOKENS)
        assert any(
            "falling back to" in r.message and fallback_value in r.message for r in caplog.records
        )

    def test_embed_entity_no_truncation_when_fits(self, service, mock_db, caplog):
        """Description is not truncated when it fits within the context budget."""
        entity = Entity(
            id="ent-fits",
            type="character",
            name="Alice",
            description="Short desc",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_entity(mock_db, entity)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text == "Alice: Short desc"
        # No truncation warnings
        assert not any("Truncating" in r.message for r in caplog.records)

    def test_embed_entity_truncation_uses_sentence_boundary(self, service, mock_db):
        """Truncation prefers sentence boundaries when available in the second half."""
        # Build text with clear sentence boundaries that exceeds budget.
        sentences = "First sentence. Second sentence. Third sentence. Fourth sentence. "
        text = sentences * 20  # ~1300 chars, plenty to exceed a small budget
        entity = Entity(
            id="ent-sent",
            type="character",
            name="A",
            description=text,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "A: " (3 chars), prefix = "" → overhead = 3
        # context_limit = 60, margin = 10 → max_chars = (60 - 10) * 2 = 100
        # available_for_desc = 100 - 3 = 97
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=60),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
        ):
            service.embed_entity(mock_db, entity)

        embedded_text = mock_embed_text.call_args[0][0]
        # Description part should end at a sentence boundary (period)
        desc_part = embedded_text[len("A: ") :]
        assert desc_part.endswith(".")
        assert len(desc_part) <= 97

    def test_embed_entity_truncation_preserves_exact_budget(self, service, mock_db):
        """Truncated description length matches the available character budget exactly."""
        entity = Entity(
            id="ent-exact",
            type="character",
            name="A",
            description="x" * 300,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "A: " (3 chars), prefix = "" → overhead = 3
        # context_limit = 60, margin = 10 → max_chars = (60 - 10) * 2 = 100
        # available_for_desc = 100 - 3 = 97
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=60),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
        ):
            service.embed_entity(mock_db, entity)

        embedded_text = mock_embed_text.call_args[0][0]
        # header (3) + description (97) = 100 chars total
        assert len(embedded_text) == 100
        assert embedded_text.startswith("A: ")


# ---------------------------------------------------------------------------
# embed_event pre-truncation tests
# ---------------------------------------------------------------------------


class TestEmbedEventTruncation:
    """Tests for pre-truncation in embed_event."""

    def test_embed_event_truncates_long_description(self, service, mock_db, caplog):
        """Description is truncated while preserving event structure when it exceeds budget."""
        long_event = WorldEvent(
            id="evt-long",
            description="x" * 5000,
            chapter_number=3,
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
            result = service.embed_event(mock_db, long_event)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert len(embedded_text) < 5000
        assert embedded_text.startswith("Event: ")
        assert embedded_text.endswith(" (Chapter 3)")
        assert any("Truncating event description" in r.message for r in caplog.records)

    def test_embed_event_empty_description_when_overhead_meets_budget(
        self, service, mock_db, caplog
    ):
        """Description is cleared when overhead equals budget but fits max_chars."""
        event = WorldEvent(
            id="evt-overflow",
            description="Some description",
            chapter_number=1,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "Event: " (7 chars), chapter_part = " (Chapter 1)" (12 chars)
        # prefix = "x" * 41 → overhead = 41 + 7 + 12 = 60
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # available_for_desc = 60 - 60 = 0 → budget exhausted
        # overhead (60) == max_chars (60), NOT > → embed with empty description
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 41),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_event(mock_db, event)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text == "Event:  (Chapter 1)"
        assert any("meets embedding budget" in r.message for r in caplog.records)

    def test_embed_event_skips_when_overhead_exceeds_max_chars(self, service, mock_db, caplog):
        """Embedding is skipped when prefix + header + chapter_part exceeds max_chars."""
        event = WorldEvent(
            id="evt-skip",
            description="Some description",
            chapter_number=1,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        # header = "Event: " (7 chars), chapter_part = " (Chapter 1)" (12 chars)
        # prefix = "x" * 50 → overhead = 50 + 7 + 12 = 69
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # overhead (69) > max_chars (60) → skip embedding
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 50),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_event(mock_db, event)

        assert result is False
        mock_embed_text.assert_not_called()
        assert any("overhead exceeds max chars" in r.message for r in caplog.records)

    def test_embed_event_skips_records_failure(self, service, mock_db):
        """Records failure when overhead exceeds max_chars and embedding is skipped."""
        event = WorldEvent(
            id="evt-skip-fail",
            description="Some description",
            chapter_number=1,
        )
        mock_client = MagicMock()
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 50),
            patch.object(service, "embed_text", MagicMock(return_value=FAKE_EMBEDDING)),
        ):
            service.embed_event(mock_db, event)

        assert "evt-skip-fail" in service._failed_source_ids

    def test_embed_event_uses_fallback_when_context_size_none(self, service, mock_db, caplog):
        """embed_event uses _FALLBACK_CONTEXT_TOKENS when context size is None."""
        event = WorldEvent(
            id="evt-fallback",
            description="A short event description",
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        embedding_service_mod._warned_context_models.discard(service.settings.embedding_model)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch(
                "src.services.embedding_service.get_model_context_size",
                return_value=None,
            ),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="pfx:"),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_event(mock_db, event)

        assert result is True
        mock_embed_text.assert_called_once()
        fallback_value = str(_FALLBACK_CONTEXT_TOKENS)
        assert any(
            "falling back to" in r.message and fallback_value in r.message for r in caplog.records
        )

    def test_embed_event_no_truncation_when_fits(self, service, mock_db, caplog):
        """Description is not truncated when it fits within the context budget."""
        event = WorldEvent(
            id="evt-fits",
            description="Short event",
            chapter_number=2,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            result = service.embed_event(mock_db, event)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text == "Event: Short event (Chapter 2)"
        assert not any("Truncating" in r.message for r in caplog.records)

    def test_embed_event_without_chapter_truncation(self, service, mock_db):
        """Truncation works correctly for events without a chapter number."""
        event = WorldEvent(
            id="evt-no-ch",
            description="x" * 5000,
        )
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=128),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
        ):
            result = service.embed_event(mock_db, event)

        assert result is True
        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text.startswith("Event: ")
        # No chapter suffix
        assert "(Chapter" not in embedded_text
        assert len(embedded_text) < 5000


# ---------------------------------------------------------------------------
# _pre_truncate_callback_text tests
# ---------------------------------------------------------------------------


class TestPreTruncateCallbackText:
    """Tests for _pre_truncate_callback_text in attach_to_database callback."""

    def test_callback_truncates_long_text(self, service, mock_db, caplog):
        """Callback text is truncated when it exceeds the context budget."""
        entity = Entity(
            id="ent-cb-long",
            type="character",
            name="Test",
            description="A test entity",
        )
        mock_db.get_entity.return_value = entity

        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        long_text = "x" * 5000
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=128),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            on_changed("ent-cb-long", "entity", long_text)

        # embed_text should have been called with truncated text
        embedded_text = mock_embed_text.call_args[0][0]
        assert len(embedded_text) < 5000
        assert any("Truncating callback text" in r.message for r in caplog.records)

    def test_callback_returns_empty_when_overhead_exceeds_budget(self, service, mock_db, caplog):
        """Callback text becomes empty when prefix overhead exceeds budget."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=[])
        # prefix = "x" * 70 → overhead = 70
        # context_limit = 40, margin = 10 → max_chars = (40 - 10) * 2 = 60
        # available_for_text = 60 - 70 = -10 → empty
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=40),
            patch("src.services.embedding_service.get_embedding_prefix", return_value="x" * 70),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            on_changed("ent-cb-overflow", "entity", "Some text")

        # embed_text is called with empty string (which returns [])
        assert any("overhead exceeds max chars" in r.message for r in caplog.records)

    def test_callback_no_truncation_when_fits(self, service, mock_db, caplog):
        """Callback text is not truncated when it fits within budget."""
        entity = Entity(
            id="ent-cb-fits",
            type="character",
            name="Test",
            description="A test entity",
        )
        mock_db.get_entity.return_value = entity

        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        short_text = "Test: A test entity"
        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=512),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            on_changed("ent-cb-fits", "entity", short_text)

        embedded_text = mock_embed_text.call_args[0][0]
        assert embedded_text == short_text
        assert not any("Truncating" in r.message for r in caplog.records)

    def test_callback_truncation_fallback_on_error(self, service, mock_db, caplog):
        """Callback proceeds with original text when pre-truncation encounters an error."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", side_effect=ConnectionError("Connection refused")),
            patch.object(service, "embed_text", mock_embed_text),
            caplog.at_level(logging.WARNING),
        ):
            on_changed("ent-cb-err", "entity", "Some text")

        assert any("Could not pre-truncate" in r.message for r in caplog.records)

    def test_callback_uses_fallback_context_when_none(self, service, mock_db):
        """Callback uses fallback context tokens when get_model_context_size returns None."""
        service.attach_to_database(mock_db)
        on_changed = mock_db.attach_content_changed_callback.call_args[0][0]

        mock_client = MagicMock()
        mock_embed_text = MagicMock(return_value=FAKE_EMBEDDING)
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("src.services.embedding_service.get_model_context_size", return_value=None),
            patch("src.services.embedding_service.get_embedding_prefix", return_value=""),
            patch.object(service, "embed_text", mock_embed_text),
        ):
            # Short text should pass through without truncation with 512 fallback
            on_changed("ent-cb-none", "entity", "Short text")

        mock_embed_text.assert_called_once_with("Short text")
