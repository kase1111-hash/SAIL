"""
SAIL Knowledge Domain System

Provides jurisdiction-aware knowledge retrieval across multiple domains:
- Legal: Rights, obligations, and procedures by jurisdiction
- Safety: Personal safety protocols and de-escalation techniques
- Financial: Scam detection and financial protection
- Emergency: Medical and emergency procedures
"""

from src.knowledge.base import (
    # Data models
    Jurisdiction,
    JurisdictionLevel,
    KnowledgeCategory,
    # Configuration
    KnowledgeConfig,
    # Base class
    KnowledgeDomain,
    # Domain types
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeRelevance,
    KnowledgeResult,
)
from src.knowledge.manager import (
    KnowledgeEvent,
    KnowledgeEventType,
    KnowledgeManager,
    create_initialized_knowledge_manager,
    create_knowledge_manager,
)
from src.knowledge.retrieval import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    RAGPipeline,
    RAGResult,
    VectorStore,
    create_rag_pipeline,
)

__all__ = [
    # Domain types
    "KnowledgeDomainType",
    "KnowledgeCategory",
    "KnowledgeRelevance",
    "JurisdictionLevel",
    # Data models
    "Jurisdiction",
    "KnowledgeItem",
    "KnowledgeQuery",
    "KnowledgeResult",
    # Base class
    "KnowledgeDomain",
    # Configuration
    "KnowledgeConfig",
    # Manager
    "KnowledgeManager",
    "KnowledgeEvent",
    "KnowledgeEventType",
    "create_knowledge_manager",
    "create_initialized_knowledge_manager",
    # RAG
    "RAGPipeline",
    "RAGResult",
    "VectorStore",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "create_rag_pipeline",
]
