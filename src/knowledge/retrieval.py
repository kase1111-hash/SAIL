"""
SAIL Knowledge Retrieval System

Provides Retrieval-Augmented Generation (RAG) capabilities for the knowledge
domain system including:
- Vector embeddings for semantic search
- Similarity-based retrieval
- Citation tracking
- Knowledge update mechanisms
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.base import (
    Jurisdiction,
    KnowledgeConfig,
    KnowledgeDomain,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeResult,
)
from src.knowledge.domains.base_domain import (
    MAX_TOKEN_SCORE,
    stem_token,
    tokenize_query,
)

logger = logging.getLogger(__name__)


# Cosine threshold for the TF-IDF fallback embeddings. Measured against the
# bundled corpus: on-topic query/document pairs score 0.17-0.42, while the best
# score any nonsense query reaches is 0.16. The bands nearly touch, so this sits
# at the conservative end -- RAG only supplements keyword matching here, and a
# confidently wrong citation costs more than a missed supplementary one.
#
# The sentence-transformer default of 0.5 is above anything these embeddings
# produce, which is what left semantic search returning nothing for every query.
FALLBACK_SIMILARITY_THRESHOLD = 0.25


def normalize_domain_score(score: float, query_token_count: int) -> float:
    """Map a raw keyword tally into [0, 1] as a fraction of the best possible.

    Absolute rather than relative: the previous ``score / max_score`` made the
    best hit exactly 1.0 for *every* query, so a query that merely grazed one
    irrelevant item was reported with the same confidence as an exact match.

    Dividing by the per-token maximum keeps the result comparable across
    queries of different lengths -- a two-word query can only ever accumulate
    a fraction of what a six-word query does, which would otherwise make short
    questions look uniformly irrelevant.
    """
    if score <= 0 or query_token_count <= 0:
        return 0.0
    return min(score / (MAX_TOKEN_SCORE * query_token_count), 1.0)


# ============================================================================
# Embedding Interface
# ============================================================================


class EmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        raise NotImplementedError

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Falls back to simple TF-IDF-like embeddings if sentence-transformers
    is not available.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self._use_sentence_transformers = False
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        # IDF for a token absent from the corpus. A query word no document
        # contains is maximally distinctive, so it gets the highest weight.
        self._default_idf: float = 1.0
        self._word_vectors: dict[str, np.ndarray] = {}

    @property
    def uses_semantic_model(self) -> bool:
        """Whether the trained encoder is actually loaded.

        False both when the package is missing and when it is installed but the
        model could not be loaded, which are different causes with the same
        consequence. Only meaningful after ``initialize()``.
        """
        return self._use_sentence_transformers

    @property
    def similarity_threshold(self) -> float:
        """Cosine threshold appropriate to the embeddings actually in use.

        The fallback embeddings are a random projection of a TF-IDF bag of
        words, whose cosines sit far below what a trained sentence encoder
        produces. Applying the sentence-transformer threshold to them rejects
        every result, which silently disabled semantic search entirely.
        """
        return 0.5 if self._use_sentence_transformers else FALLBACK_SIMILARITY_THRESHOLD

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_sentence_transformers = True
            self._initialized = True
            logger.info(f"Initialized sentence-transformers with {self.model_name}")
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using simple embeddings"
            )
            self._use_sentence_transformers = False
            self._initialized = True
            return True
        except Exception as e:
            # The package is installed but the model could not be loaded --
            # typically no cached copy and no network, which is the normal
            # state for an offline install. Fall back exactly as the missing
            # package does. Reporting failure here instead left _initialized
            # False, so every embed() call retried the model load and stalled
            # on a doomed network fetch: two per query, indefinitely.
            logger.warning(
                f"Could not load embedding model {self.model_name} ({e}); "
                "using simple embeddings"
            )
            self._use_sentence_transformers = False
            self._initialized = True
            return True

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        if self._use_sentence_transformers:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            return self._simple_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if self._use_sentence_transformers:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [self._simple_embed(text) for text in texts]

    def _word_vector(self, token: str, dim: int) -> np.ndarray:
        """Deterministic random unit vector for a token, cached."""
        cached = self._word_vectors.get(token)
        if cached is None or cached.shape[0] != dim:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            # Use a local generator: seeding the global np.random state here
            # would reset it for the whole process on every embedded word.
            rng = np.random.default_rng(token_hash % (2**32))
            cached = rng.standard_normal(dim)
            self._word_vectors[token] = cached
        return cached

    def _simple_embed(self, text: str, dim: int = 384) -> list[float]:
        """
        Generate a TF-IDF weighted hash embedding.

        This is a fallback when sentence-transformers is not available. It is a
        random projection of the text's TF-IDF bag of words, which approximately
        preserves the cosine similarity of the underlying sparse vectors.

        Tokens are stop-word filtered and stemmed with the same helpers the
        keyword scorer uses, so the two retrieval halves agree on what a word
        is: without this, filler words dominated every vector and made all
        documents look alike.
        """
        tokens = [stem_token(token) for token in tokenize_query(text)]

        embedding = np.zeros(dim)
        for token in tokens:
            weight = self._idf.get(token, self._default_idf)
            embedding += weight * self._word_vector(token, dim)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def build_vocabulary(self, texts: list[str]) -> None:
        """Build IDF weights from the corpus for the TF-IDF fallback.

        Called at index time. Without it every token carries equal weight and
        common corpus words ("scam", "call", "safety") drown out the ones that
        actually discriminate between documents.
        """
        word_doc_count: dict[str, int] = {}
        total_docs = len(texts)
        if total_docs == 0:
            return

        for text in texts:
            words = {stem_token(token) for token in tokenize_query(text)}
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        # Smoothed IDF, always positive. The unsmoothed log(N / (df + 1)) goes
        # negative for any word appearing in most documents, which would flip
        # that word's contribution and push matching documents apart.
        for word, count in word_doc_count.items():
            self._idf[word] = float(np.log((1 + total_docs) / (1 + count)) + 1.0)

        self._default_idf = float(np.log(1 + total_docs) + 1.0)

        # Weights changed, so any vectors built under the old ones are stale.
        self._word_vectors.clear()

        # Build vocabulary index
        self._vocabulary = {word: i for i, word in enumerate(sorted(word_doc_count.keys()))}


# ============================================================================
# Vector Store
# ============================================================================


@dataclass
class VectorEntry:
    """Entry in the vector store."""

    item_id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Simple in-memory vector store for knowledge items.

    For production, this could be replaced with FAISS, Chroma,
    or another vector database.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._entries: dict[str, VectorEntry] = {}
        self._embeddings: np.ndarray | None = None
        self._item_ids: list[str] = []

    def add(self, item_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        """Add a vector entry to the store."""
        entry = VectorEntry(
            item_id=item_id,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._entries[item_id] = entry
        self._embeddings = None  # Invalidate cache

    def remove(self, item_id: str) -> bool:
        """Remove a vector entry from the store."""
        if item_id in self._entries:
            del self._entries[item_id]
            self._embeddings = None
            return True
        return False

    def _build_index(self) -> None:
        """Build the index for fast similarity search."""
        if self._embeddings is not None:
            return

        self._item_ids = list(self._entries.keys())
        if not self._item_ids:
            self._embeddings = np.array([])
            return

        embeddings_list = [self._entries[item_id].embedding for item_id in self._item_ids]
        self._embeddings = np.array(embeddings_list)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (item_id, similarity_score) tuples
        """
        self._build_index()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Convert query to numpy
        query_vec = np.array(query_embedding)

        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        query_vec = query_vec / query_norm

        # Normalize stored embeddings
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = self._embeddings / norms

        # Compute similarities
        similarities = np.dot(normalized, query_vec)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self._item_ids[idx], score))

        return results

    def save(self, path: Path) -> None:
        """Save vector store to disk."""
        data = {
            "dimension": self.dimension,
            "entries": {
                item_id: {
                    "embedding": entry.embedding,
                    "metadata": entry.metadata,
                }
                for item_id, entry in self._entries.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        """Load vector store from disk."""
        with open(path) as f:
            data = json.load(f)

        store = cls(dimension=data.get("dimension", 384))

        for item_id, entry_data in data.get("entries", {}).items():
            store.add(
                item_id=item_id,
                embedding=entry_data["embedding"],
                metadata=entry_data.get("metadata", {}),
            )

        return store

    @property
    def count(self) -> int:
        """Get number of entries in the store."""
        return len(self._entries)


# ============================================================================
# RAG Pipeline
# ============================================================================


@dataclass
class RAGResult:
    """Result from RAG retrieval."""

    query: str
    items: list[KnowledgeItem]
    scores: list[float]
    augmented_context: str
    citations: list[str]
    retrieval_time_ms: float


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for knowledge retrieval.

    Combines semantic search with knowledge domain filtering to
    provide relevant, contextual information.
    """

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.config = config or KnowledgeConfig()
        self._embedding_provider = embedding_provider or LocalEmbeddingProvider(
            self.config.embedding_model
        )
        self._vector_store = VectorStore()
        self._item_index: dict[str, KnowledgeItem] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the RAG pipeline."""
        try:
            success = await self._embedding_provider.initialize()
            self._initialized = success
            return success
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False

    async def index_items(self, items: list[KnowledgeItem]) -> int:
        """
        Index knowledge items for semantic search.

        Args:
            items: Knowledge items to index

        Returns:
            Number of items indexed
        """
        if not self._initialized:
            await self.initialize()

        indexed = 0

        # Prepare texts for embedding
        texts = []
        valid_items = []

        for item in items:
            # Create searchable text from item
            text = self._item_to_text(item)
            texts.append(text)
            valid_items.append(item)

        # Fit IDF weights on the corpus before embedding it. Query embeddings
        # generated later reuse these weights, so both sides of the comparison
        # live in the same space.
        build_vocabulary = getattr(self._embedding_provider, "build_vocabulary", None)
        if callable(build_vocabulary):
            build_vocabulary(texts)

        # Generate embeddings in batch
        embeddings = await self._embedding_provider.embed_batch(texts)

        # Add to vector store and item index. strict: a provider returning the
        # wrong number of embeddings would otherwise silently leave the tail of
        # the corpus unindexed and unsearchable.
        for item, embedding in zip(valid_items, embeddings, strict=True):
            self._vector_store.add(
                item_id=item.item_id,
                embedding=embedding,
                metadata={
                    "domain": item.domain.value,
                    "category": item.category.value if item.category else None,
                    "relevance": item.relevance.value,
                },
            )
            self._item_index[item.item_id] = item
            item.embedding = embedding
            indexed += 1

        logger.info(f"Indexed {indexed} items for semantic search")
        return indexed

    def _similarity_threshold(self) -> float:
        """Cosine threshold to apply, deferring to the embedding provider.

        ``KnowledgeConfig.similarity_threshold`` is calibrated for a trained
        sentence encoder. When the provider reports its own threshold -- as the
        TF-IDF fallback does, on a much lower cosine scale -- honour that
        instead, so semantic search does not silently return nothing.
        """
        provider_threshold = getattr(
            self._embedding_provider, "similarity_threshold", None
        )
        if provider_threshold is None:
            return self.config.similarity_threshold
        return float(provider_threshold)

    def _item_to_text(self, item: KnowledgeItem) -> str:
        """Convert a knowledge item to searchable text."""
        parts = [item.title, item.summary, item.content]
        if item.keywords:
            parts.append(" ".join(item.keywords))
        return " ".join(parts)

    async def retrieve(
        self,
        query: str,
        domain: KnowledgeDomainType | None = None,
        jurisdiction: Jurisdiction | None = None,
        max_results: int = 5,
    ) -> RAGResult:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: Search query
            domain: Optional domain filter
            jurisdiction: Optional jurisdiction filter
            max_results: Maximum number of results

        Returns:
            RAGResult with relevant items and context
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self._embedding_provider.embed(query)

        # Search vector store
        results = self._vector_store.search(
            query_embedding,
            top_k=max_results * 2,  # Get extra for filtering
            threshold=self._similarity_threshold(),
        )

        # Filter and rank results
        items: list[KnowledgeItem] = []
        scores: list[float] = []

        for item_id, score in results:
            item = self._item_index.get(item_id)
            if not item:
                continue

            # Filter by domain
            if domain and item.domain != domain:
                continue

            # Filter by jurisdiction
            if jurisdiction and not item.applies_to_jurisdiction(jurisdiction):
                continue

            items.append(item)
            scores.append(score)

            if len(items) >= max_results:
                break

        # Build augmented context
        augmented_context = self._build_context(items, query)

        # Extract citations
        citations = [
            f"{item.source}" + (f" ({item.source_url})" if item.source_url else "")
            for item in items
            if item.source
        ]

        retrieval_time = (time.time() - start_time) * 1000

        return RAGResult(
            query=query,
            items=items,
            scores=scores,
            augmented_context=augmented_context,
            citations=citations,
            retrieval_time_ms=retrieval_time,
        )

    def _build_context(self, items: list[KnowledgeItem], query: str) -> str:
        """Build augmented context from retrieved items."""
        if not items:
            return ""

        parts = [f"Relevant information for: {query}\n"]

        for i, item in enumerate(items, 1):
            parts.append(f"\n[{i}] {item.title}")
            if item.summary:
                parts.append(item.summary)
            else:
                # Use truncated content
                content = item.content[:500]
                if len(item.content) > 500:
                    content += "..."
                parts.append(content)

            if item.source:
                parts.append(f"Source: {item.source}")

        return "\n".join(parts)

    async def query_with_context(
        self,
        query: KnowledgeQuery,
        domains: list[KnowledgeDomain] | None = None,
    ) -> KnowledgeResult:
        """
        Query with context-aware retrieval.

        Combines RAG semantic search with domain-specific querying.

        Args:
            query: Knowledge query
            domains: Optional list of domains to search

        Returns:
            KnowledgeResult with combined results
        """
        start_time = time.time()

        # Get RAG results
        rag_result = await self.retrieve(
            query=query.query_text,
            domain=query.domain,
            jurisdiction=query.jurisdiction,
            max_results=query.max_results,
        )

        # If domains provided, also query them directly
        domain_scored: list[tuple[KnowledgeItem, float]] = []
        if domains:
            for domain in domains:
                if query.domain and domain.domain_type != query.domain:
                    continue

                domain_result = await domain.query(query)
                domain_scored.extend(
                    zip(domain_result.items, domain_result.scores, strict=False)
                )

        # Merge on item_id, keeping the strongest signal for each item. RAG
        # results are already gated by ``similarity_threshold``; domain hits
        # carry their own floor, applied below.
        scored: dict[str, tuple[KnowledgeItem, float]] = {
            item.item_id: (item, score)
            for item, score in zip(rag_result.items, rag_result.scores, strict=False)
        }
        query_token_count = len(tokenize_query(query.query_text))
        for item, score in domain_scored:
            normalized = normalize_domain_score(score, query_token_count)
            if normalized < self.config.min_domain_relevance:
                continue
            existing = scored.get(item.item_id)
            if existing is None or normalized > existing[1]:
                scored[item.item_id] = (item, normalized)

        # Rank by relevance, then trim. Ranking here is what keeps the
        # first-registered domain from monopolizing the results.
        ranked = sorted(scored.values(), key=lambda pair: pair[1], reverse=True)
        ranked = ranked[: query.max_results]
        merged_items = [item for item, _ in ranked]
        merged_scores = [score for _, score in ranked]

        result = KnowledgeResult(
            query_id=query.query_id,
            items=merged_items,
            scores=merged_scores,
            total_found=len(merged_items),
            domain_searched=query.domain,
            jurisdiction_filtered=query.jurisdiction is not None,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            sources=rag_result.citations,
        )

        return result

    def save_index(self, path: Path) -> None:
        """Save the vector index to disk."""
        self._vector_store.save(path)

    def load_index(self, path: Path) -> bool:
        """Load the vector index from disk."""
        try:
            self._vector_store = VectorStore.load(path)
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    @property
    def indexed_count(self) -> int:
        """Get number of indexed items."""
        return self._vector_store.count


# ============================================================================
# Factory Functions
# ============================================================================


def create_rag_pipeline(
    config: KnowledgeConfig | None = None,
) -> RAGPipeline:
    """Create a RAG pipeline instance."""
    return RAGPipeline(config)


async def create_initialized_rag_pipeline(
    config: KnowledgeConfig | None = None,
) -> RAGPipeline:
    """Create and initialize a RAG pipeline."""
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    return pipeline
