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

# Per-token scoring weights. Tuned against the relevance cases in
# tests/integration/test_retrieval_relevance.py -- raising the title weight
# further was measured to make ranking worse, not better, because several
# correct answers do not name the topic in their title.
TEXT_MATCH_SCORE = 1.0
KEYWORD_MATCH_SCORE = 2.0
TITLE_MATCH_SCORE = 1.5

# Relevance-level multipliers. Deliberately gentle: a large spread lets a
# CRITICAL item that barely matches the query outrank the HIGH item that
# actually answers it.
RELEVANCE_BOOSTS = {
    KnowledgeRelevance.CRITICAL: 1.3,
    KnowledgeRelevance.HIGH: 1.15,
    KnowledgeRelevance.MEDIUM: 1.0,
    KnowledgeRelevance.LOW: 0.9,
    KnowledgeRelevance.INFORMATIONAL: 0.8,
}

# Best score a single query token can contribute: a body-text hit that is also
# a keyword and appears in the title, at the largest relevance multiplier. Used
# to normalize raw tallies into a query-length-independent ratio. Subclass
# boosts can push a score past this, so callers clamp.
MAX_TOKEN_SCORE = (
    TEXT_MATCH_SCORE + KEYWORD_MATCH_SCORE + TITLE_MATCH_SCORE
) * max(RELEVANCE_BOOSTS.values())

# Suffixes stripped when reducing a token to its stem. Deliberately excludes
# "es": stripping it turns "quotes" into "quot" while "quote" stays put, so the
# two forms stop matching. Plain "s" plurals dominate this corpus anyway.
_SUFFIXES = ("ing", "ed", "s")

# Minimum length of the remainder after stripping a suffix. Without this,
# "is" -> "" and "thing" -> "th".
_MIN_STEM_LEN = 3


def _collapse_doubled_consonant(stem: str) -> str:
    """Collapse a trailing doubled consonant ("stopp" -> "stop").

    Applied to every stem rather than only to suffix-stripped ones. Doing it
    on both sides is what keeps the pairs consistent: "stopped" -> "stopp" ->
    "stop" needs the collapse, and "called" -> "call" -> "cal" only matches
    "call" because "call" is collapsed to "cal" too.
    """
    if len(stem) > _MIN_STEM_LEN and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        return stem[:-1]
    return stem


def stem_token(token: str) -> str:
    """Reduce a token to a crude stem so word forms match each other.

    Handles the plural/participle forms that dominate this corpus:
    ``scams`` -> ``scam``, ``rights`` -> ``right``, ``stopped`` -> ``stop``.
    Deliberately conservative -- it is only ever compared against other stems,
    so under-stemming costs a match while over-stemming invents one.
    """
    if token.endswith("ies") and len(token) - 3 >= _MIN_STEM_LEN - 1:
        return token[:-3] + "y"

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
            return _collapse_doubled_consonant(token[: -len(suffix)])

    return _collapse_doubled_consonant(token)


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


def tokenize_stems(text: str) -> set[str]:
    """Tokenize text into a set of stems for word-boundary matching.

    Matching against this set instead of doing a substring test on the raw
    text is what stops ``"quit"`` from scoring a hit on ``"quite"``.
    """
    return {stem_token(token) for token in _TOKEN_RE.findall(text.lower())}


class BaseKnowledgeDomain(KnowledgeDomain):
    """Shared implementation for all knowledge domains.

    Subclasses override:
    - ``_load_items()`` to load domain-specific data
    - ``_apply_domain_boost()`` (optional) for domain-specific scoring
    - ``_filter_jurisdiction()`` (optional) for jurisdiction filtering
    """

    def __init__(self, domain_type: KnowledgeDomainType, data_path: Path | None = None):
        super().__init__(domain_type, data_path)
        # item_id -> (text stems, keyword stems, title stems). Items are static
        # once loaded, so stemming every item on every query is pure waste.
        self._stem_cache: dict[str, tuple[set[str], set[str], set[str]]] = {}

    def add_item(self, item: KnowledgeItem) -> None:
        """Add an item, dropping any stale cached stems for that item id."""
        super().add_item(item)
        self._stem_cache.pop(item.item_id, None)

    def _item_stems(self, item: KnowledgeItem) -> tuple[set[str], set[str], set[str]]:
        """Return cached (text, keyword, title) stem sets for an item."""
        cached = self._stem_cache.get(item.item_id)
        if cached is None:
            keyword_text = " ".join(item.keywords)
            cached = (
                tokenize_stems(
                    f"{item.title} {item.content} {item.summary} {keyword_text}"
                ),
                # Keywords are tokenized rather than compared whole, so the
                # query word "traffic" matches the keyword "traffic stop".
                tokenize_stems(keyword_text),
                tokenize_stems(item.title),
            )
            self._stem_cache[item.item_id] = cached
        return cached

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
        text_stems, keyword_stems, title_stems = self._item_stems(item)

        # Keyword matching. Stems are compared as whole tokens, so "quit" no
        # longer scores a hit against "quite" while "scams" still matches
        # "scam".
        for word in query_words:
            stem = stem_token(word)
            if stem in text_stems:
                score += TEXT_MATCH_SCORE
            if stem in keyword_stems:
                score += KEYWORD_MATCH_SCORE
            if stem in title_stems:
                score += TITLE_MATCH_SCORE

        score *= RELEVANCE_BOOSTS.get(item.relevance, 1.0)

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
