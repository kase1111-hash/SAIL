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

import sys
import types

import numpy as np
import pytest
from src.knowledge.base import KnowledgeDomainType
from src.knowledge.domains.base_domain import stem_token, tokenize_query
from src.knowledge.manager import create_initialized_knowledge_manager
from src.knowledge.retrieval import (
    FALLBACK_SIMILARITY_THRESHOLD,
    LocalEmbeddingProvider,
)

# (query, expected domain, acceptable top-result titles)
#
# Most queries have exactly one right answer in the bundled corpus. Where more
# than one title is listed, the corpus genuinely contains several items that
# answer the question equally well and the retriever is not expected to pick
# between them -- see the note on the wire-transfer case below.
RELEVANCE_CASES = [
    (
        "My friend is choking, what do I do?",
        KnowledgeDomainType.EMERGENCY,
        ("Choking Response for Adults",),
    ),
    (
        # Unsolicited call + demand to wire money + threat of arrest. Three
        # corpus items each cover part of that and none covers all of it: the
        # wire-transfer item owns the payment method, the impersonation item
        # owns the arrest threat, and the family-emergency item owns the
        # caller-with-an-urgent-story script. All three give the user the same
        # protective advice, and both retrieval halves independently rank the
        # family-emergency item top because the query shares its most
        # distinctive terms ("wire money", "arrested"). Asserting one exact
        # title here would be pinning down a distinction the corpus does not
        # actually support.
        "Someone called saying I must wire money immediately or be arrested",
        KnowledgeDomainType.FINANCIAL,
        (
            "Wire Transfer Fraud Warning Signs",
            "Government and Authority Impersonation Scams",
            "Grandparent and Family Emergency Scams",
        ),
    ),
    (
        "What are my rights during a traffic stop?",
        KnowledgeDomainType.LEGAL,
        ("General Traffic Stop Guidelines",),
    ),
    (
        "How do I de-escalate an angry confrontation?",
        KnowledgeDomainType.SAFETY,
        ("Verbal De-escalation Techniques",),
    ),
    (
        "I'm having chest pain and shortness of breath",
        KnowledgeDomainType.EMERGENCY,
        ("Heart Attack Recognition and Response",),
    ),
    (
        "A caller wants my social security number to verify my account",
        KnowledgeDomainType.FINANCIAL,
        ("Government and Authority Impersonation Scams",),
    ),
    (
        "I'm meeting a stranger from the internet, how do I stay safe?",
        KnowledgeDomainType.SAFETY,
        ("Safe Meeting Protocol for Online Dating",),
    ),
]

# Queries that no corpus item answers. These must produce no citations at all
# rather than whatever item they happen to graze.
IRRELEVANT_QUERIES = ["quit", "hello", "thanks", "what is the weather", "asdfghjkl"]


@pytest.fixture
async def knowledge_manager():
    manager = await create_initialized_knowledge_manager()
    yield manager
    await manager.stop()


class TestRetrievalRelevance:
    """End-to-end relevance of the knowledge retrieval pipeline."""

    @pytest.mark.parametrize(
        ("query", "expected_domain", "expected_titles"), RELEVANCE_CASES
    )
    async def test_query_retrieves_from_correct_domain(
        self, knowledge_manager, query, expected_domain, expected_titles
    ):
        """The top result comes from the domain that actually covers the query."""
        result = await knowledge_manager.query(query, max_results=3)

        assert result.items, f"no results for {query!r}"
        assert result.items[0].domain == expected_domain
        assert result.items[0].title in expected_titles

    async def test_results_are_ranked_by_descending_score(self, knowledge_manager):
        """Merged results are ordered by relevance, not by domain registration order."""
        result = await knowledge_manager.query(
            "someone is having a heart attack", max_results=5
        )

        assert len(result.scores) >= 2
        assert result.scores == sorted(result.scores, reverse=True)

    async def test_scores_are_not_a_flat_constant(self, knowledge_manager):
        """Real relevance scores are preserved rather than replaced by a placeholder."""
        result = await knowledge_manager.query(
            "someone is having a heart attack", max_results=5
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


class TestIrrelevantQueries:
    """A query nothing covers must produce no citations.

    Before the relevance floor, every query returned whatever it grazed with
    a score of exactly 1.0 -- "quit" was answered with FTC/FBI scam sources.
    """

    @pytest.mark.parametrize("query", IRRELEVANT_QUERIES)
    async def test_irrelevant_query_returns_no_items(self, knowledge_manager, query):
        result = await knowledge_manager.query(query, max_results=3)

        assert result.items == [], (
            f"{query!r} cited {[i.title for i in result.items]}"
        )

    async def test_scores_are_not_pinned_to_one(self, knowledge_manager):
        """Relative normalization made the best hit exactly 1.0 for every query."""
        result = await knowledge_manager.query(
            "My friend is choking, what do I do?", max_results=5
        )

        assert result.scores
        assert max(result.scores) < 1.0

    async def test_weak_match_scores_below_strong_match(self, knowledge_manager):
        """Confidence is absolute, so scores are comparable across queries."""
        strong = await knowledge_manager.query(
            "I'm having chest pain and shortness of breath", max_results=1
        )
        weaker = await knowledge_manager.query(
            "how do I stay safe walking at night", max_results=1
        )

        assert strong.scores, "expected a hit for the chest-pain query"
        if weaker.scores:
            assert strong.scores[0] > weaker.scores[0]


class TestSemanticSearch:
    """The RAG half of retrieval has to actually contribute.

    The fallback embeddings produce cosines far below the sentence-transformer
    threshold of 0.5, so vector search returned nothing for every query and
    all results came from keyword matching alone.
    """

    @pytest.mark.parametrize(
        "query", ["traffic stop police rights", "heart attack", "choking"]
    )
    async def test_vector_search_returns_results(self, knowledge_manager, query):
        result = await knowledge_manager._rag_pipeline.retrieve(
            query=query, max_results=3
        )

        assert result.items, f"semantic search returned nothing for {query!r}"

    async def test_vector_search_rejects_nonsense(self, knowledge_manager):
        result = await knowledge_manager._rag_pipeline.retrieve(
            query="asdfghjkl", max_results=3
        )

        assert result.items == []

    async def test_idf_weights_are_fitted_at_index_time(self, knowledge_manager):
        """Without fitted IDF every token weighs the same and documents blur together."""
        provider = knowledge_manager._rag_pipeline._embedding_provider

        assert provider._idf, "IDF weights were never built"
        # Keyed by stem, the same form the keyword scorer uses.
        # "scam" appears across many items; "choking" in very few, so it must
        # carry more discriminative weight.
        assert provider._idf[stem_token("choking")] > provider._idf[stem_token("scam")]

    async def test_threshold_is_reachable(self, knowledge_manager):
        """The active threshold must be reachable by the active embeddings.

        This is the shape of the original bug: a threshold calibrated for one
        embedding backend, applied to another that never reaches it. Asserted
        for whichever backend is installed rather than skipped, since either
        one being unreachable disables semantic search.
        """
        rag = knowledge_manager._rag_pipeline
        provider = rag._embedding_provider

        if provider._use_sentence_transformers:
            expected = knowledge_manager.config.similarity_threshold
        else:
            expected = FALLBACK_SIMILARITY_THRESHOLD
        assert rag._similarity_threshold() == expected

        item = next(
            i for i in rag._item_index.values() if i.title == "Choking Response for Adults"
        )
        query_embedding = np.array(await provider.embed("my friend is choking"))
        cosine = float(query_embedding @ np.array(item.embedding))

        assert cosine >= expected, (
            f"on-topic cosine {cosine:.3f} is below the {expected} threshold, "
            "so semantic search returns nothing"
        )


class TestEmbeddingInitialization:
    """Falling back must be a terminal decision, not retried per call."""

    @staticmethod
    def _install_failing_model(monkeypatch, calls):
        """Make `from sentence_transformers import SentenceTransformer` succeed
        but constructing the model fail, as it does offline with no cached copy.
        """
        fake = types.ModuleType("sentence_transformers")

        def boom(*args, **kwargs):
            calls.append(1)
            raise OSError("no cached model and no network")

        fake.SentenceTransformer = boom
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    async def test_model_load_failure_falls_back(self, monkeypatch):
        calls: list[int] = []
        self._install_failing_model(monkeypatch, calls)

        provider = LocalEmbeddingProvider()

        assert await provider.initialize() is True
        assert provider._initialized is True
        assert provider._use_sentence_transformers is False
        assert len(calls) == 1

    async def test_failed_load_is_not_retried_on_every_embed(self, monkeypatch):
        """A doomed model load used to run again on each embed() call."""
        calls: list[int] = []
        self._install_failing_model(monkeypatch, calls)

        provider = LocalEmbeddingProvider()
        await provider.initialize()
        attempts_after_init = len(calls)

        for _ in range(5):
            await provider.embed("my friend is choking")

        assert len(calls) == attempts_after_init

    async def test_fallback_embeddings_still_usable_after_failure(self, monkeypatch):
        calls: list[int] = []
        self._install_failing_model(monkeypatch, calls)

        provider = LocalEmbeddingProvider()
        await provider.initialize()
        embedding = await provider.embed("my friend is choking")

        assert len(embedding) == 384
        assert any(value != 0.0 for value in embedding)


class TestStemming:
    """Keyword matching is on whole-word stems, not substrings."""

    def test_substring_is_not_a_match(self):
        """"quit" scored a hit against "quite", so "quit" retrieved scam warnings."""
        assert stem_token("quit") != stem_token("quite")

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("scams", "scam"),
            ("rights", "right"),
            ("stopped", "stop"),
            ("called", "call"),
            ("emergencies", "emergency"),
        ],
    )
    def test_word_forms_match(self, first, second):
        assert stem_token(first) == stem_token(second)

    @pytest.mark.parametrize(
        ("first", "second"),
        [("act", "contact"), ("ice", "police"), ("police", "policy")],
    )
    def test_unrelated_words_do_not_match(self, first, second):
        assert stem_token(first) != stem_token(second)


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
