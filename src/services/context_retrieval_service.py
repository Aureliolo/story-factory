"""Context retrieval service for smart RAG-based context injection.

Replaces the existing dumb context slicing (get_context_for_agents) with
vector-similarity-based retrieval for all LLM calls. Falls back to the
legacy method when vector search is unavailable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class ContextItem:
    """A single piece of retrieved context for prompt injection.

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
        """Calculate token estimate from text length."""
        if not self.token_estimate:
            self.token_estimate = max(1, len(self.text) // 4)


@dataclass
class RetrievedContext:
    """Container for context items retrieved via vector search or fallback.

    Attributes:
        items: List of retrieved context items.
        total_tokens: Sum of token estimates across all items.
        retrieval_method: How context was retrieved ('vector' or 'fallback').
    """

    items: list[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    retrieval_method: str = "vector"

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

        return "\n\n".join(sections)


class ContextRetrievalService:
    """Retrieves semantically relevant world context for LLM prompts.

    Uses vector similarity search when available, with graph expansion to
    include related entities. Falls back to the legacy get_context_for_agents()
    method when vectors are unavailable.

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
        story_state: Any,
        max_tokens: int | None = None,
        content_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        chapter_number: int | None = None,
        k: int | None = None,
    ) -> RetrievedContext:
        """Retrieve semantically relevant context for an LLM prompt.

        Algorithm:
        1. Embed the task_description via EmbeddingService
        2. KNN search in vec_embeddings with optional filters
        3. Graph expansion: for entity results, fetch 1-hop neighbors
        4. Deduplicate by source_id (keep highest relevance)
        5. Always include base project info (premise, genre, tone, setting)
        6. Token budgeting: sort by relevance, pack greedily

        Falls back to legacy context when vectors are unavailable.

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

        effective_max_tokens = max_tokens or self.settings.rag_context_max_tokens
        effective_k = k or self.settings.rag_context_max_items

        # Try vector retrieval first
        if self.embedding_service.is_available and world_db.vec_available:
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

        # Fallback to legacy context
        return self._retrieve_fallback(world_db, story_state, effective_max_tokens)

    def _retrieve_vector(
        self,
        task_description: str,
        world_db: WorldDatabase,
        story_state: Any,
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
            logger.warning("Failed to embed task description, falling back")
            return self._retrieve_fallback(world_db, story_state, max_tokens)

        # Step 2: KNN search with optional content type filter
        all_results: list[dict] = []
        if content_types:
            for ct in content_types:
                results = world_db.search_similar(
                    query_embedding=query_embedding,
                    k=k,
                    content_type=ct,
                    entity_type=entity_types[0]
                    if entity_types and len(entity_types) == 1
                    else None,
                    chapter_number=chapter_number,
                )
                all_results.extend(results)
        else:
            all_results = world_db.search_similar(
                query_embedding=query_embedding,
                k=k,
                entity_type=entity_types[0] if entity_types and len(entity_types) == 1 else None,
                chapter_number=chapter_number,
            )

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
                    items_by_id[source_id].relevance_score = relevance
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

                    # Also add relationships for expanded entities
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
                    logger.debug("Graph expansion failed for %s: %s", item.source_id, e)

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
            items=packed_items,
            total_tokens=total_tokens,
            retrieval_method="vector",
        )

    def _retrieve_fallback(
        self,
        world_db: WorldDatabase,
        story_state: Any,
        max_tokens: int,
    ) -> RetrievedContext:
        """Retrieve context using the legacy get_context_for_agents method.

        Args:
            world_db: WorldDatabase instance.
            story_state: Current story state.
            max_tokens: Token budget for context.

        Returns:
            RetrievedContext with fallback-retrieved items.
        """
        items: list[ContextItem] = []
        total_tokens = 0

        # Get legacy context
        try:
            context = world_db.get_context_for_agents()
        except Exception as e:
            logger.warning("Failed to get legacy context: %s", e)
            return RetrievedContext(retrieval_method="fallback")

        # Convert characters
        for char in context.get("characters", []):
            text = f"{char['name']}: {char['description']}"
            item = ContextItem(
                source_type="entity",
                source_id=f"fallback:char:{char['name']}",
                relevance_score=0.5,
                text=text,
            )
            if total_tokens + item.token_estimate <= max_tokens:
                items.append(item)
                total_tokens += item.token_estimate

        # Convert locations
        for loc in context.get("locations", []):
            text = f"{loc['name']}: {loc['description']}"
            item = ContextItem(
                source_type="entity",
                source_id=f"fallback:loc:{loc['name']}",
                relevance_score=0.4,
                text=text,
            )
            if total_tokens + item.token_estimate <= max_tokens:
                items.append(item)
                total_tokens += item.token_estimate

        # Convert relationships
        for rel in context.get("key_relationships", []):
            text = f"{rel['from']} {rel['type']} {rel['to']}: {rel['description']}"
            item = ContextItem(
                source_type="relationship",
                source_id=f"fallback:rel:{rel['from']}:{rel['to']}",
                relevance_score=0.4,
                text=text,
            )
            if total_tokens + item.token_estimate <= max_tokens:
                items.append(item)
                total_tokens += item.token_estimate

        # Convert events
        for evt in context.get("recent_events", []):
            text = f"Event (Ch.{evt.get('chapter', '?')}): {evt['description']}"
            item = ContextItem(
                source_type="event",
                source_id=f"fallback:evt:{evt['description'][:30]}",
                relevance_score=0.3,
                text=text,
            )
            if total_tokens + item.token_estimate <= max_tokens:
                items.append(item)
                total_tokens += item.token_estimate

        # Add project info
        project_items = self._get_project_info_items(story_state)
        for pi in project_items:
            if total_tokens + pi.token_estimate <= max_tokens:
                items.append(pi)
                total_tokens += pi.token_estimate

        logger.info("Fallback retrieval: %d items, ~%d tokens", len(items), total_tokens)
        return RetrievedContext(
            items=items,
            total_tokens=total_tokens,
            retrieval_method="fallback",
        )

    def _get_project_info_items(self, story_state: Any) -> list[ContextItem]:
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
