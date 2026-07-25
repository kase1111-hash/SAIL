"""
Base knowledge domain with shared query, scoring, and file-loading logic.

Eliminates ~2,000 lines of duplication across the 4 domain files.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from src.knowledge.base import (
    KnowledgeCategory,
    KnowledgeDomain,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeRelevance,
    KnowledgeResult,
)

logger = logging.getLogger(__name__)

# Words that carry no retrieval signal. Every knowledge item contains most of
# these, so scoring them lets an unrelated item outrank the right one purely on
# filler-word overlap.
STOP_WORDS = frozenset({
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "but", "by", "can", "did", "do", "does", "for", "from", "get", "had",
    "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "my", "no", "not", "of", "on",
    "or", "our", "out", "she", "should", "so", "some", "than", "that", "the",
    "their", "them", "then", "there", "they", "this", "to", "up", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_query(text: str) -> list[str]:
    """Split query text into meaningful lowercase search tokens.

    Strips punctuation (so ``"choking,"`` matches ``"choking"``) and drops
    stop words and single characters, which otherwise dominate keyword scores.
    """
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


class BaseKnowledgeDomain(KnowledgeDomain):
    """Shared implementation for all knowledge domains.

    Subclasses override:
    - ``_load_items()`` to load domain-specific data
    - ``_apply_domain_boost()`` (optional) for domain-specific scoring
    - ``_filter_jurisdiction()`` (optional) for jurisdiction filtering
    """

    def __init__(self, domain_type: KnowledgeDomainType, data_path: Path | None = None):
        super().__init__(domain_type, data_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    async def load(self) -> bool:
        """Load items via ``_load_items`` and then from JSON files."""
        try:
            self._load_items()
            await self._load_from_files()
            self._loaded = True
            logger.info(f"Loaded {self.item_count} {self.domain_type.value} knowledge items")
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.domain_type.value} knowledge: {e}")
            return False

    def _load_items(self) -> None:
        """Override to load domain-specific data dictionaries."""
        raise NotImplementedError

    async def _load_from_files(self) -> None:
        """Load additional knowledge from JSON files on disk."""
        if not self.data_path.exists():
            return

        for json_file in self.data_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item_data in data:
                        self.add_item(KnowledgeItem.from_dict(item_data))
                elif isinstance(data, dict):
                    self.add_item(KnowledgeItem.from_dict(data))

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Query the domain for relevant knowledge items."""
        start_time = time.time()

        # Filter by category
        candidates = list(self._items.values())
        if query.category:
            candidates = [i for i in candidates if i.category == query.category]

        # Optional jurisdiction filtering (Legal domain overrides this)
        jurisdiction_filtered = False
        filtered = self._filter_jurisdiction(candidates, query)
        if filtered is not None:
            candidates = filtered
            jurisdiction_filtered = True

        # Score items
        query_words = tokenize_query(query.query_text)
        scored_items: list[tuple[KnowledgeItem, float]] = []
        for item in candidates:
            score = self._calculate_relevance_score(item, query_words, query)
            if score > 0:
                scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        top_items = scored_items[: query.max_results]
        items = [i for i, _ in top_items]
        scores = [s for _, s in top_items]

        return KnowledgeResult(
            query_id=query.query_id,
            items=items,
            scores=scores,
            total_found=len(scored_items),
            domain_searched=self.domain_type,
            jurisdiction_filtered=jurisdiction_filtered,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            sources=[item.source for item in items if item.source],
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_relevance_score(
        self,
        item: KnowledgeItem,
        query_words: list[str],
        query: KnowledgeQuery,
    ) -> float:
        """Compute relevance score with shared algorithm + domain boosts."""
        score = 0.0
        item_text = f"{item.title} {item.content} {item.summary} {' '.join(item.keywords)}".lower()

        # Keyword matching
        for word in query_words:
            if word in item_text:
                score += 1.0
            if word in [k.lower() for k in item.keywords]:
                score += 2.0

        # Title match boost
        title_lower = item.title.lower()
        for word in query_words:
            if word in title_lower:
                score += 1.5

        # Relevance level boost
        relevance_boosts = {
            KnowledgeRelevance.CRITICAL: 3.0,
            KnowledgeRelevance.HIGH: 2.0,
            KnowledgeRelevance.MEDIUM: 1.0,
            KnowledgeRelevance.LOW: 0.5,
            KnowledgeRelevance.INFORMATIONAL: 0.25,
        }
        score *= relevance_boosts.get(item.relevance, 1.0)

        # Domain-specific boost (subclass hook)
        score = self._apply_domain_boost(item, query_words, query, score)

        return score

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _filter_jurisdiction(
        self,
        candidates: list[KnowledgeItem],
        query: KnowledgeQuery,
    ) -> list[KnowledgeItem] | None:
        """Override to filter candidates by jurisdiction.

        Return the filtered list, or ``None`` if no filtering was applied.
        """
        return None

    def _apply_domain_boost(
        self,
        item: KnowledgeItem,
        query_words: list[str],
        query: KnowledgeQuery,
        score: float,
    ) -> float:
        """Override to apply domain-specific scoring adjustments."""
        return score
