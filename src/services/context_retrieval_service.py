"""Context retrieval service for smart RAG-based context injection.

Uses vector-similarity-based retrieval via the sqlite-vec + embedding pipeline
to enrich writing-agent prompts with relevant world context.
"""

import logging
import sqlite3
import struct
from dataclasses import dataclass, replace
from typing import Literal

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.embedding_service import EmbeddingService
from src.settings import Settings

logger = logging.getLogger(__name__)

# Human-readable section titles for each content type
_SECTION_TITLES: dict[str, str] = {
    "entity": "RELEVANT ENTITIES",
    "relationship": "RELEVANT RELATIONSHIPS",
    "event": "RELEVANT EVENTS",
    "fact": "ESTABLISHED FACTS",
    "rule": "WORLD RULES",
    "chapter_outline": "CHAPTER OUTLINES",
    "scene_outline": "SCENE OUTLINES",
}


@dataclass(frozen=True)
class ContextItem:
    """A single piece of retrieved context for prompt injection.

    Frozen dataclass — instances are immutable after creation.  Use
    ``dataclasses.replace()`` to derive a new item with different field values.

    Attributes:
        source_type: Content type (entity, relationship, event, fact, rule, etc.).
        source_id: Unique identifier of the source content.
        relevance_score: Relevance score from 0.0 to 1.0 (1.0 = most relevant).
        text: Formatted text for inclusion in the prompt.
        token_estimate: Rough token count estimate (len(text) // 4).
    """

    source_type: str
    source_id: str
    relevance_score: float
    text: str
    token_estimate: int = 0

    def __post_init__(self) -> None:
        """Clamp relevance_score to [0.0, 1.0] and calculate token estimate.

        Uses ``object.__setattr__`` to bypass the frozen constraint during
        initialization (the standard pattern for post-init fixups on frozen
        dataclasses).
        """
        clamped = max(0.0, min(1.0, self.relevance_score))
        if clamped != self.relevance_score:
            logger.debug(
                "Clamped relevance_score %.4f -> %.4f for %s",
                self.relevance_score,
                clamped,
                self.source_id,
            )
            object.__setattr__(self, "relevance_score", clamped)
        if not self.token_estimate:
            object.__setattr__(self, "token_estimate", max(1, len(self.text) // 4))


@dataclass(frozen=True)
class RetrievedContext:
    """Container for context items retrieved via vector search.

    Frozen dataclass — constructed once with final values.

    Attributes:
        items: Tuple of retrieved context items (immutable after construction).
        total_tokens: Sum of token estimates across all items.
        retrieval_method: How context was retrieved ('vector' or 'disabled').
    """

    items: tuple[ContextItem, ...] = ()
    total_tokens: int = 0
    retrieval_method: Literal["vector", "disabled"] = "vector"

    def __post_init__(self) -> None:
        """Normalize items to a tuple to guarantee immutability at runtime.

        Callers may pass a list at runtime despite the type annotation; this
        ensures the stored value is always a tuple.
        """
        object.__setattr__(self, "items", tuple(self.items))

    def format_for_prompt(self) -> str:
        """Format all context items as a structured text block for prompt injection.

        Groups items by source_type and formats them with section headers.

        Returns:
            Formatted context string ready for prompt inclusion.
        """
        if not self.items:
            return ""

        # Group items by source type
        grouped: dict[str, list[ContextItem]] = {}
        for item in self.items:
            grouped.setdefault(item.source_type, []).append(item)

        sections = []
        for content_type, items in grouped.items():
            title = _SECTION_TITLES.get(content_type, content_type.upper())
            lines = [f"- {item.text}" for item in items]
            sections.append(f"{title}:\n" + "\n".join(lines))

        context_body = "\n\n".join(sections)
        return f"<retrieved-context>\n{context_body}\n</retrieved-context>"


class ContextRetrievalService:
    """Retrieves semantically relevant world context for LLM prompts.

    Uses vector similarity search with graph expansion to include related
    entities.

    Attributes:
        settings: Application settings.
        embedding_service: Service for generating query embeddings.
    """

    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        """Initialize the context retrieval service.

        Args:
            settings: Application settings for RAG configuration.
            embedding_service: Embedding service for query vector generation.
        """
        self.settings = settings
        self.embedding_service = embedding_service
        logger.info("ContextRetrievalService initialized")

    def retrieve_context(
        self,
        task_description: str,
        world_db: WorldDatabase,
        story_state: StoryState,
        max_tokens: int | None = None,
        content_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        chapter_number: int | None = None,
        k: int | None = None,
    ) -> RetrievedContext:
        """Retrieve semantically relevant context for an LLM prompt.

        Algorithm:
        1. Embed the task_description via EmbeddingService
        2. KNN search in vec_embeddings with optional filters (deduplicates
           inline by source_id, keeping highest relevance)
        3. Graph expansion: for entity results, fetch neighbors (depth from settings)
        4. Always include base project info (premise, genre, tone, setting)
        5. Token budgeting: sort by relevance, pack greedily

        Args:
            task_description: Description of what the agent is about to do.
            world_db: WorldDatabase with potential vec_embeddings.
            story_state: Current StoryState for brief/chapter info.
            max_tokens: Maximum tokens for context (defaults to settings).
            content_types: Filter to specific content types.
            entity_types: Filter to specific entity types.
            chapter_number: Filter to specific chapter.
            k: Number of neighbors to retrieve (defaults to settings).

        Returns:
            RetrievedContext with items sorted by relevance.
        """
        if not self.settings.rag_context_enabled:
            logger.debug("RAG context disabled, returning empty context")
            return RetrievedContext(retrieval_method="disabled")

        effective_max_tokens = (
            self.settings.rag_context_max_tokens if max_tokens is None else max_tokens
        )
        effective_k = self.settings.rag_context_max_items if k is None else k

        return self._retrieve_vector(
            task_description=task_description,
            world_db=world_db,
            story_state=story_state,
            max_tokens=effective_max_tokens,
            content_types=content_types,
            entity_types=entity_types,
            chapter_number=chapter_number,
            k=effective_k,
        )

    def _retrieve_vector(
        self,
        task_description: str,
        world_db: WorldDatabase,
        story_state: StoryState,
        max_tokens: int,
        content_types: list[str] | None,
        entity_types: list[str] | None,
        chapter_number: int | None,
        k: int,
    ) -> RetrievedContext:
        """Perform vector-based context retrieval with graph expansion.

        Args:
            task_description: The query text to embed.
            world_db: WorldDatabase with vec_embeddings.
            story_state: Current story state.
            max_tokens: Token budget for context.
            content_types: Optional content type filter.
            entity_types: Optional entity type filter.
            chapter_number: Optional chapter filter.
            k: Number of nearest neighbors.

        Returns:
            RetrievedContext with vector-retrieved items.
        """
        # Step 1: Embed the task description
        query_embedding = self.embedding_service.embed_text(task_description)
        if not query_embedding:
            logger.warning("Failed to embed task description, returning empty context")
            return RetrievedContext(retrieval_method="vector")

        # Step 2: KNN search with optional content type filter
        all_results: list[dict] = []
        try:
            # Default to entity content type when entity_types filter is provided
            if not content_types and entity_types:
                content_types = ["entity"]
            if content_types:
                for ct in content_types:
                    results = world_db.search_similar(
                        query_embedding=query_embedding,
                        k=k,
                        content_type=ct,
                        # Only apply entity_type filter for entity content type
                        entity_type=entity_types[0]
                        if entity_types and len(entity_types) == 1 and ct == "entity"
                        else None,
                        chapter_number=chapter_number,
                    )
                    all_results.extend(results)
            else:
                all_results = world_db.search_similar(
                    query_embedding=query_embedding,
                    k=k,
                    chapter_number=chapter_number,
                )
        except (sqlite3.Error, struct.error) as e:
            logger.warning("Vector search failed: %s", e)
            return RetrievedContext(retrieval_method="vector")

        # Convert to ContextItems with relevance scores
        items_by_id: dict[str, ContextItem] = {}
        threshold = self.settings.rag_context_similarity_threshold

        for result in all_results:
            distance = result["distance"]
            # Convert distance to relevance score (0-1, higher = more relevant)
            # sqlite-vec returns L2 distance; invert and normalize
            relevance = max(0.0, 1.0 - distance)

            if relevance < threshold:
                continue

            source_id = result["source_id"]
            if source_id in items_by_id:
                # Keep highest relevance
                if relevance > items_by_id[source_id].relevance_score:
                    items_by_id[source_id] = replace(
                        items_by_id[source_id], relevance_score=relevance
                    )
                continue

            items_by_id[source_id] = ContextItem(
                source_type=result["content_type"],
                source_id=source_id,
                relevance_score=relevance,
                text=result["display_text"] or "",
            )

        # Step 3: Graph expansion for entity results
        if self.settings.rag_context_graph_expansion:
            expansion_depth = self.settings.rag_context_graph_depth
            entity_items = [item for item in items_by_id.values() if item.source_type == "entity"]
            for item in entity_items:
                try:
                    neighbors = world_db.get_connected_entities(
                        item.source_id, max_depth=expansion_depth
                    )
                    for neighbor in neighbors:
                        if neighbor.id not in items_by_id:
                            # Add neighbor with reduced relevance
                            items_by_id[neighbor.id] = ContextItem(
                                source_type="entity",
                                source_id=neighbor.id,
                                relevance_score=item.relevance_score * 0.7,
                                text=f"{neighbor.name}: {neighbor.description}",
                            )

                    # Also add relationships for the original search result entity
                    relationships = world_db.get_relationships(item.source_id)
                    for rel in relationships:
                        if rel.id not in items_by_id:
                            source = world_db.get_entity(rel.source_id)
                            target = world_db.get_entity(rel.target_id)
                            if source and target:
                                items_by_id[rel.id] = ContextItem(
                                    source_type="relationship",
                                    source_id=rel.id,
                                    relevance_score=item.relevance_score * 0.7,
                                    text=f"{source.name} {rel.relation_type} {target.name}: {rel.description}",
                                )
                except Exception as e:
                    logger.warning("Graph expansion failed for %s: %s", item.source_id, e)

        # Step 4: Add base project info (always included)
        project_items = self._get_project_info_items(story_state)
        for pi in project_items:
            if pi.source_id not in items_by_id:
                items_by_id[pi.source_id] = pi

        # Step 5: Token budgeting — sort by relevance, pack greedily
        sorted_items = sorted(items_by_id.values(), key=lambda x: x.relevance_score, reverse=True)
        packed_items: list[ContextItem] = []
        total_tokens = 0

        for item in sorted_items:
            if total_tokens + item.token_estimate > max_tokens:
                continue
            packed_items.append(item)
            total_tokens += item.token_estimate

        logger.info(
            "Vector retrieval: %d items, ~%d tokens (from %d candidates)",
            len(packed_items),
            total_tokens,
            len(items_by_id),
        )

        return RetrievedContext(
            items=tuple(packed_items),
            total_tokens=total_tokens,
            retrieval_method="vector",
        )

    def _get_project_info_items(self, story_state: StoryState | None) -> list[ContextItem]:
        """Extract base project info as high-priority context items.

        Always included regardless of vector search results. Provides
        fundamental story parameters (premise, genre, tone, setting).

        Args:
            story_state: Current story state with brief.

        Returns:
            List of ContextItem instances for project info.
        """
        items: list[ContextItem] = []

        if not story_state or not hasattr(story_state, "brief") or not story_state.brief:
            return items

        brief = story_state.brief
        parts = []
        if brief.premise:
            parts.append(f"Premise: {brief.premise}")
        if brief.genre:
            parts.append(f"Genre: {brief.genre}")
        if brief.tone:
            parts.append(f"Tone: {brief.tone}")
        if hasattr(brief, "setting_place") and brief.setting_place:
            parts.append(f"Setting: {brief.setting_place}")
        if hasattr(brief, "setting_time") and brief.setting_time:
            parts.append(f"Time period: {brief.setting_time}")

        if parts:
            text = "; ".join(parts)
            items.append(
                ContextItem(
                    source_type="project_info",
                    source_id="project:brief",
                    relevance_score=1.0,  # Always most relevant
                    text=text,
                )
            )

        return items
