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
_FALLBACK_CONTEXT_TOKENS = 512  # Fallback when model context size is unavailable
_CONTEXT_LENGTH_ERROR_PATTERNS = (
    "context length",
    "input length",
)  # Ollama context overflow markers
_MIN_CONTENT_LENGTH = 10  # Minimum content length to be worth embedding

# Track models for which we've already logged "context size unavailable" warning.
# Protected by _warned_context_models_lock for thread-safety across instances.
_warned_context_models: set[str] = set()
_warned_context_models_lock = threading.Lock()


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
        self._failed_source_ids: set[str] = set()
        self._failed_source_ids_lock = threading.Lock()
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

    def _record_failure(self, source_id: str) -> None:
        """Record a source_id that failed to embed for later retry.

        Args:
            source_id: Identifier of the content that failed embedding.
        """
        with self._failed_source_ids_lock:
            self._failed_source_ids.add(source_id)
        logger.debug("Recorded embedding failure for %s", source_id)

    def _clear_failure(self, source_id: str) -> None:
        """Remove a source_id from the failure set after successful embedding.

        Args:
            source_id: Identifier of the content that succeeded embedding.
        """
        with self._failed_source_ids_lock:
            self._failed_source_ids.discard(source_id)
        logger.debug("Cleared failure marker for %s", source_id)

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
                if context_limit is None:
                    with _warned_context_models_lock:
                        if model not in _warned_context_models:
                            logger.warning(
                                "Context size unavailable for model '%s'; "
                                "using fallback %d tokens for all subsequent "
                                "embeddings (this warning will not repeat)",
                                model,
                                _FALLBACK_CONTEXT_TOKENS,
                            )
                            _warned_context_models.add(model)
                    context_limit = _FALLBACK_CONTEXT_TOKENS
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
            if isinstance(e, ollama.ResponseError) and any(
                pattern in str(e.error).lower() for pattern in _CONTEXT_LENGTH_ERROR_PATTERNS
            ):
                logger.warning(
                    "Context length overflow for model '%s', retrying with halved input: %s",
                    model,
                    e,
                )
                try:
                    halved_prompt = prompt[: len(prompt) // 2]
                    with self._lock:
                        client = self._get_client()
                        response = client.embeddings(model=model, prompt=halved_prompt)
                    embedding = response["embedding"]
                    if embedding:
                        logger.info(
                            "Retry with halved input succeeded for '%s...' (%d dims)",
                            text[:40],
                            len(embedding),
                        )
                        return embedding
                    logger.warning(
                        "Retry with halved input returned empty embedding for model "
                        "'%s' (halved_len=%d, response=%r)",
                        model,
                        len(halved_prompt),
                        response,
                    )
                    return []
                except (ollama.ResponseError, ConnectionError, TimeoutError) as retry_err:
                    logger.warning("Retry with halved input also failed: %s", retry_err)
                    return []
            logger.warning("Failed to generate embedding: %s", e)
            return []
        except ValueError as e:
            logger.warning("Embedding skipped due to configuration error: %s", e)
            return []
        except KeyError as e:
            logger.exception("Malformed embedding response (missing key): %s", e)
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
            self._record_failure(entity.id)
            return False

        result = db.upsert_embedding(
            source_id=entity.id,
            content_type="entity",
            text=text,
            embedding=embedding,
            model=model,
            entity_type=entity.type,
        )
        # Clear failure regardless — embed_text succeeded, embedding exists in DB
        self._clear_failure(entity.id)
        return result

    def embed_relationship(
        self,
        db: WorldDatabase,
        rel: Relationship,
        source_name: str,
        target_name: str,
    ) -> bool:
        """Embed a relationship and store it in the database.

        Pre-truncates the description to fit within the embedding model's
        context budget after accounting for entity names and relation type,
        so that entity names are always preserved intact.

        Args:
            db: WorldDatabase instance with vec support.
            rel: The relationship to embed.
            source_name: Name of the source entity.
            target_name: Name of the target entity.

        Returns:
            True if the embedding was stored, False otherwise.
        """
        model = self._get_model()
        prefix = get_embedding_prefix(model)

        # Build the structural portion (always kept intact)
        header = f"{source_name} {rel.relation_type} {target_name}: "
        overhead_chars = len(prefix) + len(header)

        # Estimate max chars from model context
        with self._lock:
            client = self._get_client()
            context_limit = get_model_context_size(client, model)
        if context_limit is None:
            logger.debug(
                "Context size unavailable for embedding model in embed_relationship, "
                "falling back to %d tokens",
                _FALLBACK_CONTEXT_TOKENS,
            )
            context_limit = _FALLBACK_CONTEXT_TOKENS
        max_chars = (context_limit - _EMBEDDING_CONTEXT_MARGIN_TOKENS) * _CHARS_PER_TOKEN_ESTIMATE

        description = rel.description
        available_for_desc = max_chars - overhead_chars
        if available_for_desc <= 0:
            if len(header) > max_chars:
                # Header alone exceeds the budget — embed_text would truncate
                # entity names, defeating the "names preserved" guarantee.
                logger.warning(
                    "Relationship header exceeds max chars (%d > %d); "
                    "skipping embedding to avoid truncating entity names: "
                    "%s...%s...%s",
                    len(header),
                    max_chars,
                    source_name,
                    rel.relation_type,
                    target_name,
                )
                self._record_failure(rel.id)
                return False
            logger.warning(
                "Relationship header exceeds embedding budget (%d > %d chars); "
                "embedding with empty description to preserve entity names: %s...%s...%s",
                overhead_chars,
                max_chars,
                source_name,
                rel.relation_type,
                target_name,
            )
            description = ""
        elif len(description) > available_for_desc:
            logger.warning(
                "Truncating relationship description for embedding "
                "(names preserved, desc %d -> %d chars): %s...%s...%s",
                len(description),
                available_for_desc,
                source_name,
                rel.relation_type,
                target_name,
            )
            description = description[:available_for_desc]

        text = f"{header}{description}"
        embedding = self.embed_text(text)
        if not embedding:
            self._record_failure(rel.id)
            return False

        result = db.upsert_embedding(
            source_id=rel.id,
            content_type="relationship",
            text=text,
            embedding=embedding,
            model=model,
        )
        # Clear failure regardless — embed_text succeeded, embedding exists in DB
        self._clear_failure(rel.id)
        return result

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
            self._record_failure(event.id)
            return False

        result = db.upsert_embedding(
            source_id=event.id,
            content_type="event",
            text=text,
            embedding=embedding,
            model=model,
            chapter_number=event.chapter_number,
        )
        # Clear failure regardless — embed_text succeeded, embedding exists in DB
        self._clear_failure(event.id)
        return result

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
                if len(fact.strip()) < _MIN_CONTENT_LENGTH:
                    logger.warning(
                        "Skipping fact with too-short content (%d chars): '%s'",
                        len(fact.strip()),
                        fact[:40],
                    )
                    db.delete_embedding(source_id)
                    self._clear_failure(source_id)
                    continue
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
                    self._clear_failure(source_id)
                else:
                    self._record_failure(source_id)

        # Embed world rules
        if story_state.world_rules:
            for rule in story_state.world_rules:
                rule_hash = hashlib.sha256(rule.encode("utf-8")).hexdigest()[:12]
                source_id = f"rule:{rule_hash}"
                if len(rule.strip()) < _MIN_CONTENT_LENGTH:
                    logger.warning(
                        "Skipping rule with too-short content (%d chars): '%s'",
                        len(rule.strip()),
                        rule[:40],
                    )
                    db.delete_embedding(source_id)
                    self._clear_failure(source_id)
                    continue
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
                    self._clear_failure(source_id)
                else:
                    self._record_failure(source_id)

        # Embed chapter outlines
        if story_state.chapters:
            for chapter in story_state.chapters:
                if chapter.outline:
                    source_id = f"chapter:{chapter.number}"
                    if len(chapter.outline.strip()) < _MIN_CONTENT_LENGTH:
                        logger.warning(
                            "Skipping chapter %d outline with too-short content (%d chars)",
                            chapter.number,
                            len(chapter.outline.strip()),
                        )
                        db.delete_embedding(source_id)
                        self._clear_failure(source_id)
                    else:
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
                            self._clear_failure(source_id)
                        else:
                            self._record_failure(source_id)

                # Embed scene outlines if available
                if chapter.scenes:
                    for scene_idx, scene in enumerate(chapter.scenes):
                        if scene.outline:
                            source_id = f"scene:{chapter.number}:{scene_idx}"
                            if len(scene.outline.strip()) < _MIN_CONTENT_LENGTH:
                                logger.warning(
                                    "Skipping chapter %d scene %d outline with "
                                    "too-short content (%d chars)",
                                    chapter.number,
                                    scene_idx + 1,
                                    len(scene.outline.strip()),
                                )
                                db.delete_embedding(source_id)
                                self._clear_failure(source_id)
                                continue
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
                                self._clear_failure(source_id)
                            else:
                                self._record_failure(source_id)

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
        model = self._get_model()
        counts: dict[str, int] = {}
        skipped = 0
        still_failed: list[str] = []

        # Snapshot already-embedded IDs and the failure set
        already_embedded = db.get_embedded_source_ids(model)
        with self._failed_source_ids_lock:
            failed_snapshot = frozenset(self._failed_source_ids)

        # Embed all entities
        entities = db.list_entities()
        entity_count = 0
        for i, entity in enumerate(entities):
            if progress_callback:
                progress_callback(i, len(entities), f"Embedding entity: {entity.name}")
            if entity.id in already_embedded and entity.id not in failed_snapshot:
                skipped += 1
                continue
            if self.embed_entity(db, entity):
                entity_count += 1
            else:
                still_failed.append(entity.id)
        counts["entity"] = entity_count

        # Embed all relationships
        relationships = db.list_relationships()
        rel_count = 0
        for rel in relationships:
            if rel.id in already_embedded and rel.id not in failed_snapshot:
                skipped += 1
                continue
            source = db.get_entity(rel.source_id)
            target = db.get_entity(rel.target_id)
            if source and target:
                if self.embed_relationship(db, rel, source.name, target.name):
                    rel_count += 1
                else:
                    still_failed.append(rel.id)
            else:
                logger.warning(
                    "Skipping relationship %s: source entity %s (%s) or "
                    "target entity %s (%s) not found",
                    rel.id,
                    rel.source_id,
                    "found" if source else "missing",
                    rel.target_id,
                    "found" if target else "missing",
                )
                still_failed.append(rel.id)
        counts["relationship"] = rel_count

        # Embed all events
        events = db.list_events()
        event_count = 0
        for event in events:
            if event.id in already_embedded and event.id not in failed_snapshot:
                skipped += 1
                continue
            if self.embed_event(db, event):
                event_count += 1
            else:
                still_failed.append(event.id)
        counts["event"] = event_count

        # Embed story state data (facts, rules, outlines)
        state_count = self.embed_story_state_data(db, story_state)
        counts["story_state"] = state_count

        total = sum(counts.values())
        logger.info(
            "Batch embedding complete: %d embedded, %d skipped (already embedded), "
            "%d still failed (%s)",
            total,
            skipped,
            len(still_failed),
            counts,
        )
        if still_failed:
            logger.warning(
                "Items still without embeddings after retry: %s",
                still_failed,
            )

        # Update failure set: remove previously-failed items that succeeded,
        # keep items that are still failing
        with self._failed_source_ids_lock:
            succeeded = failed_snapshot - frozenset(still_failed)
            self._failed_source_ids -= succeeded
            self._failed_source_ids.update(still_failed)

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
            counts = self.embed_all_world_data(db, story_state)
            total = sum(counts.values())
            if total == 0:
                logger.warning(
                    "Re-embedding completed but zero items were embedded. "
                    "Check Ollama connectivity and model availability.",
                )
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
                    self._clear_failure(source_id)
                else:
                    self._record_failure(source_id)
            except Exception as e:
                self._record_failure(source_id)
                logger.exception("Failed to embed content change for %s: %s", source_id, e)

        def on_content_deleted(source_id: str) -> None:
            """Handle content deletions by removing the embedding."""
            try:
                db.delete_embedding(source_id)
            except Exception as e:
                logger.exception("Failed to delete embedding for %s: %s", source_id, e)

        db.attach_content_changed_callback(on_content_changed)
        db.attach_content_deleted_callback(on_content_deleted)
        logger.info("Embedding callbacks attached to WorldDatabase")
