"""
Retrieval relevance regression tests.

These cover end-to-end knowledge retrieval quality: that a query is answered
from the domain that actually covers it, and that ranking reflects relevance.

Regression coverage for three defects that made every query return the same
items from whichever domain happened to be registered first:
- ``query_with_context`` discarded domain relevance scores, substituting a
  constant, then truncated without ranking.
- Query tokens kept their punctuation, so ``"choking,"`` never matched
  ``"choking"``.
- Stop words scored as keyword hits, letting filler-word overlap outrank
  genuine topic matches.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.knowledge.base import KnowledgeDomainType
from src.knowledge.domains.base_domain import tokenize_query
from src.knowledge.manager import create_initialized_knowledge_manager
from src.knowledge.retrieval import LocalEmbeddingProvider

# (query, expected domain, expected top-result title)
RELEVANCE_CASES = [
    (
        "My friend is choking, what do I do?",
        KnowledgeDomainType.EMERGENCY,
        "Choking Response for Adults",
    ),
    (
        "Someone called saying I must wire money immediately or be arrested",
        KnowledgeDomainType.FINANCIAL,
        "Wire Transfer Fraud Warning Signs",
    ),
    (
        "What are my rights during a traffic stop?",
        KnowledgeDomainType.LEGAL,
        "General Traffic Stop Guidelines",
    ),
    (
        "How do I de-escalate an angry confrontation?",
        KnowledgeDomainType.SAFETY,
        "Verbal De-escalation Techniques",
    ),
    (
        "I'm having chest pain and shortness of breath",
        KnowledgeDomainType.EMERGENCY,
        "Heart Attack Recognition and Response",
    ),
    (
        "A caller wants my social security number to verify my account",
        KnowledgeDomainType.FINANCIAL,
        "Government and Authority Impersonation Scams",
    ),
    (
        "I'm meeting a stranger from the internet, how do I stay safe?",
        KnowledgeDomainType.SAFETY,
        "Safe Meeting Protocol for Online Dating",
    ),
]


@pytest.fixture
async def knowledge_manager():
    manager = await create_initialized_knowledge_manager()
    yield manager
    await manager.stop()


class TestRetrievalRelevance:
    """End-to-end relevance of the knowledge retrieval pipeline."""

    @pytest.mark.parametrize(("query", "expected_domain", "expected_title"), RELEVANCE_CASES)
    async def test_query_retrieves_from_correct_domain(
        self, knowledge_manager, query, expected_domain, expected_title
    ):
        """The top result comes from the domain that actually covers the query."""
        result = await knowledge_manager.query(query, max_results=3)

        assert result.items, f"no results for {query!r}"
        assert result.items[0].domain == expected_domain
        assert result.items[0].title == expected_title

    async def test_results_are_ranked_by_descending_score(self, knowledge_manager):
        """Merged results are ordered by relevance, not by domain registration order."""
        result = await knowledge_manager.query(
            "My friend is choking, what do I do?", max_results=5
        )

        assert len(result.scores) >= 2
        assert result.scores == sorted(result.scores, reverse=True)

    async def test_scores_are_not_a_flat_constant(self, knowledge_manager):
        """Real relevance scores are preserved rather than replaced by a placeholder."""
        result = await knowledge_manager.query(
            "My friend is choking, what do I do?", max_results=5
        )

        assert len(set(result.scores)) > 1, f"scores are flat: {result.scores}"

    async def test_different_queries_retrieve_different_items(self, knowledge_manager):
        """Distinct topics must not collapse onto one fixed set of items."""
        choking = await knowledge_manager.query("My friend is choking", max_results=3)
        traffic = await knowledge_manager.query("traffic stop rights", max_results=3)

        choking_ids = {i.item_id for i in choking.items}
        traffic_ids = {i.item_id for i in traffic.items}

        assert choking_ids
        assert traffic_ids
        assert choking_ids != traffic_ids

    async def test_emergency_query_does_not_return_only_legal_items(
        self, knowledge_manager
    ):
        """The original bug: the first-registered domain answered everything."""
        result = await knowledge_manager.query(
            "I'm having chest pain and shortness of breath", max_results=3
        )

        domains = {item.domain for item in result.items}
        assert domains != {KnowledgeDomainType.LEGAL}


class TestQueryTokenization:
    """Tokenization behaviour that keyword scoring depends on."""

    def test_strips_trailing_punctuation(self):
        assert "choking" in tokenize_query("My friend is choking, what do I do?")

    def test_drops_stop_words(self):
        tokens = tokenize_query("What are my rights during a traffic stop?")

        assert "traffic" in tokens
        assert "rights" in tokens
        for filler in ("what", "are", "my", "a"):
            assert filler not in tokens

    def test_drops_single_characters(self):
        assert tokenize_query("I do a b c") == []

    def test_lowercases_and_splits_on_apostrophes(self):
        tokens = tokenize_query("I'm having CHEST pain")

        assert "chest" in tokens
        assert "pain" in tokens

    def test_empty_query_yields_no_tokens(self):
        assert tokenize_query("") == []
        assert tokenize_query("?!.,") == []


class TestEmbeddingSideEffects:
    """The fallback embedder must not disturb process-global state."""

    async def test_embed_does_not_reseed_global_numpy_rng(self):
        """Seeding np.random globally per word would make unrelated code deterministic."""
        provider = LocalEmbeddingProvider()
        await provider.initialize()

        np.random.seed(1234)
        expected = np.random.rand()

        np.random.seed(1234)
        await provider.embed("some query text to embed")
        actual = np.random.rand()

        assert actual == expected

    async def test_embedding_is_deterministic(self):
        """Same text must always embed identically, despite the local RNG."""
        provider = LocalEmbeddingProvider()
        await provider.initialize()

        first = await provider.embed("wire transfer fraud")
        second = await provider.embed("wire transfer fraud")

        assert first == second

    async def test_different_text_embeds_differently(self):
        provider = LocalEmbeddingProvider()
        await provider.initialize()

        first = await provider.embed("wire transfer fraud")
        second = await provider.embed("choking response")

        assert first != second
