"""Embedding service for generating and managing vector embeddings.

Provides methods to embed world content (entities, relationships, events, facts,
rules, chapter/scene outlines) into the WorldDatabase vec_embeddings table via
Ollama's embedding API. Supports incremental embedding after every meaningful
change and batch re-embedding when the model changes.
"""

import hashlib
import logging
import threading
from collections.abc import Callable

import ollama

from src.memory.entities import Entity, Relationship, WorldEvent
from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.llm_client import get_model_context_size
from src.settings import Settings
from src.settings._model_registry import get_embedding_prefix

logger = logging.getLogger(__name__)

# Embedding context-limit constants
_MIN_USABLE_CONTEXT_TOKENS = 10  # Below this, context window is unusable
_EMBEDDING_CONTEXT_MARGIN_TOKENS = 10  # Reserve tokens for model overhead
_CHARS_PER_TOKEN_ESTIMATE = 2  # Approximate chars/token for Latin scripts; may undercount CJK/emoji


class EmbeddingService:
    """Generates and manages vector embeddings for world content.

    Thread-safe implementation using RLock for concurrent embedding operations.
    Individual embedding calls handle Ollama connection failures gracefully,
    but a configured embedding model is required (raises ValueError otherwise).

    Attributes:
        settings: Application settings instance.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the embedding service.

        Args:
            settings: Application settings for Ollama connection and model config.

        Raises:
            ValueError: If no embedding model is configured in settings.
        """
        if not settings.embedding_model.strip():
            raise ValueError(
                "EmbeddingService requires a configured embedding_model. "
                "Set embedding_model in Settings to a valid model name "
                "(e.g. 'mxbai-embed-large')."
            )
        self.settings = settings
        self._lock = threading.RLock()
        self._client: ollama.Client | None = None
        logger.info(
            "EmbeddingService initialized with model '%s'",
            settings.embedding_model,
        )

    def _get_client(self) -> ollama.Client:
        """Get or create the Ollama client.

        Note: This method is not thread-safe on its own. It must be called
        within self._lock to avoid TOCTOU races on self._client.

        Returns:
            An initialized Ollama client.
        """
        if self._client is None:
            self._client = ollama.Client(
                host=self.settings.ollama_url,
                timeout=self.settings.ollama_generate_timeout,
            )
        return self._client

    def _get_model(self) -> str:
        """Get the configured embedding model name.

        Returns:
            The embedding model identifier.

        Raises:
            ValueError: If no embedding model is configured.
        """
        model = self.settings.embedding_model.strip()
        if not model:
            raise ValueError(
                "No embedding model configured. "
                "Set embedding_model in Settings to use vector search."
            )
        return model

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Applies the model-specific prompt prefix from the registry for optimal
        embedding quality. If the combined prefix + text exceeds the model's
        context window, the prompt is truncated with a warning log. Returns an
        empty list if the model reports an unusably small context size
        (<= _MIN_USABLE_CONTEXT_TOKENS tokens).

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats. Returns empty list on failure.
        """
        if not text or not text.strip():
            return []

        try:
            model = self._get_model()
            prefix = get_embedding_prefix(model)
            prompt = f"{prefix}{text}"

            with self._lock:
                client = self._get_client()
                context_limit = get_model_context_size(client, model)
                if context_limit is not None:
                    if context_limit <= _MIN_USABLE_CONTEXT_TOKENS:
                        logger.error(
                            "Invalid context size %d for model '%s'; skipping embedding. "
                            "Check 'ollama show %s' for model metadata issues.",
                            context_limit,
                            model,
                            model,
                        )
                        return []
                    max_chars = (
                        context_limit - _EMBEDDING_CONTEXT_MARGIN_TOKENS
                    ) * _CHARS_PER_TOKEN_ESTIMATE
                    prefix_len = len(prefix)
                    if max_chars <= prefix_len:
                        logger.error(
                            "Context limit %d too small to embed any content beyond "
                            "prefix for model '%s'; skipping embedding",
                            context_limit,
                            model,
                        )
                        return []
                    if len(prompt) > max_chars:
                        logger.warning(
                            "Embedding input truncated for model '%s' (est. %d tokens > "
                            "limit %d). Preview: '%s...'",
                            model,
                            len(prompt) // _CHARS_PER_TOKEN_ESTIMATE,
                            context_limit,
                            text[:60],
                        )
                        prompt = prompt[:max_chars]
                response = client.embeddings(model=model, prompt=prompt)

            embedding: list[float] = response["embedding"]
            if not embedding:
                logger.warning("Empty embedding returned for text: '%s...'", text[:40])
                return []

            logger.debug(
                "Generated embedding for '%s...' (%d dimensions)",
                text[:40],
                len(embedding),
            )
            return embedding

        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning("Failed to generate embedding: %s", e)
            return []
        except ValueError as e:
            logger.debug("Embedding skipped: %s", e)
            return []
        except Exception as e:
            logger.exception("Unexpected error generating embedding: %s", e)
            return []

    def embed_entity(self, db: WorldDatabase, entity: Entity) -> bool:
        """Embed an entity and store it in the database.

        Args:
            db: WorldDatabase instance with vec support.
            entity: The entity to embed.

        Returns:
            True if the embedding was stored, False otherwise.
        """
        model = self._get_model()
        text = f"{entity.name}: {entity.description}"
        embedding = self.embed_text(text)
        if not embedding:
            return False

        return db.upsert_embedding(
            source_id=entity.id,
            content_type="entity",
            text=text,
            embedding=embedding,
            model=model,
            entity_type=entity.type,
        )

    def embed_relationship(
        self,
        db: WorldDatabase,
        rel: Relationship,
        source_name: str,
        target_name: str,
    ) -> bool:
        """Embed a relationship and store it in the database.

        Args:
            db: WorldDatabase instance with vec support.
            rel: The relationship to embed.
            source_name: Name of the source entity.
            target_name: Name of the target entity.

        Returns:
            True if the embedding was stored, False otherwise.
        """
        model = self._get_model()
        text = f"{source_name} {rel.relation_type} {target_name}: {rel.description}"
        embedding = self.embed_text(text)
        if not embedding:
            return False

        return db.upsert_embedding(
            source_id=rel.id,
            content_type="relationship",
            text=text,
            embedding=embedding,
            model=model,
        )

    def embed_event(self, db: WorldDatabase, event: WorldEvent) -> bool:
        """Embed an event and store it in the database.

        Args:
            db: WorldDatabase instance with vec support.
            event: The event to embed.

        Returns:
            True if the embedding was stored, False otherwise.
        """
        model = self._get_model()
        chapter_part = f" (Chapter {event.chapter_number})" if event.chapter_number else ""
        text = f"Event: {event.description}{chapter_part}"
        embedding = self.embed_text(text)
        if not embedding:
            return False

        return db.upsert_embedding(
            source_id=event.id,
            content_type="event",
            text=text,
            embedding=embedding,
            model=model,
            chapter_number=event.chapter_number,
        )

    def embed_story_state_data(self, db: WorldDatabase, story_state: StoryState) -> int:
        """Embed established facts, world rules, chapter outlines, and scene outlines.

        Uses synthetic source IDs to track these non-entity content types.

        Args:
            db: WorldDatabase instance with vec support.
            story_state: StoryState with brief, chapters, and established_facts.

        Returns:
            Number of items successfully embedded.
        """
        embedded_count = 0
        model = self._get_model()

        # Embed established facts
        if story_state.established_facts:
            for fact in story_state.established_facts:
                fact_hash = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:12]
                source_id = f"fact:{fact_hash}"
                text = f"Established fact: {fact}"
                embedding = self.embed_text(text)
                if embedding:
                    if db.upsert_embedding(
                        source_id=source_id,
                        content_type="fact",
                        text=text,
                        embedding=embedding,
                        model=model,
                    ):
                        embedded_count += 1

        # Embed world rules
        if story_state.world_rules:
            for rule in story_state.world_rules:
                rule_hash = hashlib.sha256(rule.encode("utf-8")).hexdigest()[:12]
                source_id = f"rule:{rule_hash}"
                text = f"World rule: {rule}"
                embedding = self.embed_text(text)
                if embedding:
                    if db.upsert_embedding(
                        source_id=source_id,
                        content_type="rule",
                        text=text,
                        embedding=embedding,
                        model=model,
                    ):
                        embedded_count += 1

        # Embed chapter outlines
        if story_state.chapters:
            for chapter in story_state.chapters:
                if chapter.outline:
                    source_id = f"chapter:{chapter.number}"
                    title_part = f" '{chapter.title}'" if chapter.title else ""
                    text = f"Chapter {chapter.number}{title_part}: {chapter.outline}"
                    embedding = self.embed_text(text)
                    if embedding:
                        if db.upsert_embedding(
                            source_id=source_id,
                            content_type="chapter_outline",
                            text=text,
                            embedding=embedding,
                            model=model,
                            chapter_number=chapter.number,
                        ):
                            embedded_count += 1

                # Embed scene outlines if available
                if chapter.scenes:
                    for scene_idx, scene in enumerate(chapter.scenes):
                        if scene.outline:
                            source_id = f"scene:{chapter.number}:{scene_idx}"
                            scene_title = f" '{scene.title}'" if scene.title else ""
                            text = (
                                f"Chapter {chapter.number} Scene {scene_idx + 1}"
                                f"{scene_title}: {scene.outline}"
                            )
                            embedding = self.embed_text(text)
                            if embedding:
                                if db.upsert_embedding(
                                    source_id=source_id,
                                    content_type="scene_outline",
                                    text=text,
                                    embedding=embedding,
                                    model=model,
                                    chapter_number=chapter.number,
                                ):
                                    embedded_count += 1

        logger.info("Embedded %d story state items", embedded_count)
        return embedded_count

    def embed_all_world_data(
        self,
        db: WorldDatabase,
        story_state: StoryState,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Batch-embed all world content: entities, relationships, events, and story state.

        Args:
            db: WorldDatabase instance.
            story_state: StoryState with brief and chapter data.
            progress_callback: Optional callable(current, total, message) for progress updates.

        Returns:
            Dict mapping content_type to count of items embedded.
        """
        counts: dict[str, int] = {}

        # Embed all entities
        entities = db.list_entities()
        entity_count = 0
        for i, entity in enumerate(entities):
            if progress_callback:
                progress_callback(i, len(entities), f"Embedding entity: {entity.name}")
            if self.embed_entity(db, entity):
                entity_count += 1
        counts["entity"] = entity_count

        # Embed all relationships
        relationships = db.list_relationships()
        rel_count = 0
        for rel in relationships:
            source = db.get_entity(rel.source_id)
            target = db.get_entity(rel.target_id)
            if source and target:
                if self.embed_relationship(db, rel, source.name, target.name):
                    rel_count += 1
        counts["relationship"] = rel_count

        # Embed all events
        events = db.list_events()
        event_count = 0
        for event in events:
            if self.embed_event(db, event):
                event_count += 1
        counts["event"] = event_count

        # Embed story state data (facts, rules, outlines)
        state_count = self.embed_story_state_data(db, story_state)
        counts["story_state"] = state_count

        total = sum(counts.values())
        logger.info("Batch embedding complete: %d items total (%s)", total, counts)
        return counts

    def check_and_reembed_if_needed(
        self,
        db: WorldDatabase,
        story_state: StoryState,
    ) -> bool:
        """Check if re-embedding is needed and perform it if so.

        Triggers full re-embedding when the model has changed (detected by
        comparing stored model names with current setting). Skips if the
        model has not changed since the last embedding run.

        Args:
            db: WorldDatabase instance.
            story_state: StoryState for embedding story data.

        Returns:
            True if re-embedding was performed, False if not needed.
        """
        model = self._get_model()

        if db.needs_reembedding(model):
            logger.info("Model changed, clearing and re-embedding all content")
            # Get a sample embedding to detect dimension changes
            sample = self.embed_text("dimension check")
            if not sample:
                logger.warning(
                    "Cannot re-embed: embedding model '%s' is unreachable. "
                    "Keeping existing embeddings until model becomes available.",
                    model,
                )
                return False

            db.recreate_vec_table(len(sample))
            self.embed_all_world_data(db, story_state)
            return True

        return False

    def attach_to_database(self, db: WorldDatabase) -> None:
        """Register embedding callbacks on a WorldDatabase instance.

        After calling this method, entity/relationship/event CRUD operations
        will automatically trigger embedding updates via the registered callbacks.

        Args:
            db: WorldDatabase instance to attach callbacks to.
        """

        def on_content_changed(source_id: str, content_type: str, text: str) -> None:
            """Handle content changes by embedding the new/updated content."""
            try:
                embedding = self.embed_text(text)
                if embedding:
                    entity_type = ""
                    if content_type == "entity":
                        entity = db.get_entity(source_id)
                        if entity:
                            entity_type = entity.type
                    db.upsert_embedding(
                        source_id=source_id,
                        content_type=content_type,
                        text=text,
                        embedding=embedding,
                        model=self._get_model(),
                        entity_type=entity_type,
                    )
            except Exception as e:
                logger.warning("Failed to embed content change for %s: %s", source_id, e)

        def on_content_deleted(source_id: str) -> None:
            """Handle content deletions by removing the embedding."""
            try:
                db.delete_embedding(source_id)
            except Exception as e:
                logger.warning("Failed to delete embedding for %s: %s", source_id, e)

        db.attach_content_changed_callback(on_content_changed)
        db.attach_content_deleted_callback(on_content_deleted)
        logger.info("Embedding callbacks attached to WorldDatabase")
