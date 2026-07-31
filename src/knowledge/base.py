"""
SAIL Knowledge Domain Base Module

Provides base types, abstractions, and configuration for the knowledge
domain system. Supports jurisdiction-aware knowledge retrieval with
multiple specialized domains.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Knowledge Domain Types
# ============================================================================


class KnowledgeDomainType(str, Enum):
    """Types of knowledge domains."""

    LEGAL = "legal"                     # Legal rights and obligations
    SAFETY = "safety"                   # Personal safety protocols
    FINANCIAL = "financial"             # Financial protection
    EMERGENCY = "emergency"             # Emergency procedures
    MEDICAL = "medical"                 # Basic medical guidance
    GENERAL = "general"                 # General knowledge


class KnowledgeCategory(str, Enum):
    """Categories within knowledge domains."""

    # Legal categories
    TRAFFIC_STOP = "traffic_stop"
    POLICE_INTERACTION = "police_interaction"
    TENANT_RIGHTS = "tenant_rights"
    EMPLOYMENT_RIGHTS = "employment_rights"
    CONSUMER_RIGHTS = "consumer_rights"
    CONSENT_LAWS = "consent_laws"

    # Safety categories
    DE_ESCALATION = "de_escalation"
    SITUATIONAL_AWARENESS = "situational_awareness"
    SAFE_MEETING = "safe_meeting"
    ONLINE_SAFETY = "online_safety"
    PERSONAL_SECURITY = "personal_security"

    # Financial categories
    SCAM_DETECTION = "scam_detection"
    PHISHING = "phishing"
    SOCIAL_ENGINEERING = "social_engineering"
    WIRE_FRAUD = "wire_fraud"
    IDENTITY_THEFT = "identity_theft"

    # Emergency categories
    MEDICAL_EMERGENCY = "medical_emergency"
    ACCIDENT = "accident"
    FIRE = "fire"
    NATURAL_DISASTER = "natural_disaster"
    CRIME_IN_PROGRESS = "crime_in_progress"


class KnowledgeRelevance(str, Enum):
    """Relevance level of knowledge items."""

    CRITICAL = "critical"       # Must be communicated immediately
    HIGH = "high"               # Very relevant to current situation
    MEDIUM = "medium"           # Moderately relevant
    LOW = "low"                 # Background information
    INFORMATIONAL = "informational"  # Nice to know


class JurisdictionLevel(str, Enum):
    """Level of jurisdiction specificity."""

    FEDERAL = "federal"         # Country-wide (e.g., US federal law)
    STATE = "state"             # State/province level
    LOCAL = "local"             # City/county level
    UNIVERSAL = "universal"     # Applies everywhere


# ============================================================================
# Jurisdiction Data Model
# ============================================================================


@dataclass
class Jurisdiction:
    """
    Represents a legal jurisdiction for location-aware knowledge.
    """

    country: str = "US"                 # ISO country code
    state: str | None = None         # State/province code
    county: str | None = None        # County/district
    city: str | None = None          # City/municipality

    def __post_init__(self) -> None:
        """Normalize jurisdiction codes."""
        self.country = self.country.upper()
        if self.state:
            self.state = self.state.upper()

    @property
    def full_code(self) -> str:
        """Get full jurisdiction code (e.g., 'US-CA' or 'US-CA-LA')."""
        parts = [self.country]
        if self.state:
            parts.append(self.state)
        if self.county:
            parts.append(self.county)
        if self.city:
            parts.append(self.city)
        return "-".join(parts)

    @property
    def state_code(self) -> str:
        """Get state-level code (e.g., 'US-CA')."""
        if self.state:
            return f"{self.country}-{self.state}"
        return self.country

    def matches(self, other: Jurisdiction, level: JurisdictionLevel) -> bool:
        """Check if this jurisdiction matches another at the given level."""
        if level == JurisdictionLevel.UNIVERSAL:
            return True
        if level == JurisdictionLevel.FEDERAL:
            return self.country == other.country
        if level == JurisdictionLevel.STATE:
            return self.country == other.country and self.state == other.state
        if level == JurisdictionLevel.LOCAL:
            return (
                self.country == other.country
                and self.state == other.state
                and (self.county == other.county or self.city == other.city)
            )
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "country": self.country,
            "state": self.state,
            "county": self.county,
            "city": self.city,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Jurisdiction:
        """Create from dictionary."""
        return cls(
            country=data.get("country", "US"),
            state=data.get("state"),
            county=data.get("county"),
            city=data.get("city"),
        )

    @classmethod
    def from_code(cls, code: str) -> Jurisdiction:
        """Create from jurisdiction code (e.g., 'US-CA-LA')."""
        parts = code.split("-")
        return cls(
            country=parts[0] if len(parts) > 0 else "US",
            state=parts[1] if len(parts) > 1 else None,
            county=parts[2] if len(parts) > 2 else None,
            city=parts[3] if len(parts) > 3 else None,
        )


# ============================================================================
# Knowledge Item Data Model
# ============================================================================


@dataclass
class KnowledgeItem:
    """
    A single piece of knowledge in the knowledge base.
    """

    item_id: str = field(default_factory=lambda: str(uuid4()))
    domain: KnowledgeDomainType = KnowledgeDomainType.GENERAL
    category: KnowledgeCategory | None = None

    # Content
    title: str = ""
    content: str = ""
    summary: str = ""                   # Brief summary for quick retrieval

    # Jurisdiction applicability
    jurisdiction_level: JurisdictionLevel = JurisdictionLevel.UNIVERSAL
    applicable_jurisdictions: list[str] = field(default_factory=list)

    # Relevance and context
    relevance: KnowledgeRelevance = KnowledgeRelevance.MEDIUM
    keywords: list[str] = field(default_factory=list)
    related_items: list[str] = field(default_factory=list)

    # Source and verification
    source: str | None = None
    source_url: str | None = None
    last_verified: datetime | None = None
    is_verified: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Embedding for vector search (populated by RAG system)
    embedding: list[float] | None = None

    def applies_to_jurisdiction(self, jurisdiction: Jurisdiction) -> bool:
        """Check if this item applies to the given jurisdiction."""
        if self.jurisdiction_level == JurisdictionLevel.UNIVERSAL:
            return True

        if not self.applicable_jurisdictions:
            return True

        # Check if any applicable jurisdiction matches
        for code in self.applicable_jurisdictions:
            applicable = Jurisdiction.from_code(code)
            if jurisdiction.matches(applicable, self.jurisdiction_level):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "item_id": self.item_id,
            "domain": self.domain.value,
            "category": self.category.value if self.category else None,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "jurisdiction_level": self.jurisdiction_level.value,
            "applicable_jurisdictions": self.applicable_jurisdictions,
            "relevance": self.relevance.value,
            "keywords": self.keywords,
            "related_items": self.related_items,
            "source": self.source,
            "source_url": self.source_url,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeItem:
        """Create from dictionary."""
        return cls(
            item_id=data.get("item_id", str(uuid4())),
            domain=KnowledgeDomainType(data.get("domain", "general")),
            category=KnowledgeCategory(data["category"]) if data.get("category") else None,
            title=data.get("title", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            jurisdiction_level=JurisdictionLevel(
                data.get("jurisdiction_level", "universal")
            ),
            applicable_jurisdictions=data.get("applicable_jurisdictions", []),
            relevance=KnowledgeRelevance(data.get("relevance", "medium")),
            keywords=data.get("keywords", []),
            related_items=data.get("related_items", []),
            source=data.get("source"),
            source_url=data.get("source_url"),
            last_verified=datetime.fromisoformat(data["last_verified"])
            if data.get("last_verified")
            else None,
            is_verified=data.get("is_verified", False),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Knowledge Query and Result
# ============================================================================


@dataclass
class KnowledgeQuery:
    """
    A query for knowledge retrieval.
    """

    query_id: str = field(default_factory=lambda: str(uuid4()))
    query_text: str = ""
    domain: KnowledgeDomainType | None = None
    category: KnowledgeCategory | None = None
    jurisdiction: Jurisdiction | None = None

    # Query constraints
    max_results: int = 5
    min_relevance: KnowledgeRelevance = KnowledgeRelevance.LOW
    require_verified: bool = False

    # Context from situational awareness
    context: dict[str, Any] = field(default_factory=dict)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "domain": self.domain.value if self.domain else None,
            "category": self.category.value if self.category else None,
            "jurisdiction": self.jurisdiction.to_dict() if self.jurisdiction else None,
            "max_results": self.max_results,
            "min_relevance": self.min_relevance.value,
            "require_verified": self.require_verified,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class KnowledgeResult:
    """
    Result from a knowledge query.
    """

    query_id: str
    items: list[KnowledgeItem] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)   # Relevance scores
    total_found: int = 0

    # Retrieval metadata
    domain_searched: KnowledgeDomainType | None = None
    jurisdiction_filtered: bool = False
    retrieval_time_ms: float = 0.0

    # Citation tracking
    sources: list[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def get_best_item(self) -> KnowledgeItem | None:
        """Get the highest scoring item."""
        if self.items:
            return self.items[0]
        return None

    def get_summary(self) -> str:
        """Get a combined summary of all results."""
        if not self.items:
            return "No relevant knowledge found."

        summaries = []
        for item in self.items[:3]:  # Top 3
            if item.summary:
                summaries.append(item.summary)
            elif item.content:
                summaries.append(item.content[:200] + "..." if len(item.content) > 200 else item.content)

        return "\n\n".join(summaries)

    def get_citations(self) -> list[str]:
        """Get citations for all items."""
        citations = []
        for item in self.items:
            if item.source:
                citation = item.source
                if item.source_url:
                    citation += f" ({item.source_url})"
                citations.append(citation)
        return citations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "items": [item.to_dict() for item in self.items],
            "scores": self.scores,
            "total_found": self.total_found,
            "domain_searched": self.domain_searched.value if self.domain_searched else None,
            "jurisdiction_filtered": self.jurisdiction_filtered,
            "retrieval_time_ms": self.retrieval_time_ms,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Knowledge Domain Base Class
# ============================================================================


class KnowledgeDomain(ABC):
    """
    Abstract base class for knowledge domains.

    Each domain specializes in a specific type of knowledge
    (legal, safety, financial, etc.) and provides retrieval
    methods for that knowledge.
    """

    def __init__(
        self,
        domain_type: KnowledgeDomainType,
        data_path: Path | None = None,
    ):
        self.domain_type = domain_type
        self.data_path = data_path or Path("data/knowledge_base") / domain_type.value
        self._items: dict[str, KnowledgeItem] = {}
        self._loaded = False

    @property
    def item_count(self) -> int:
        """Get number of items in the domain."""
        return len(self._items)

    @abstractmethod
    async def load(self) -> bool:
        """
        Load knowledge items for this domain.

        Returns:
            True if loading was successful
        """
        pass

    @abstractmethod
    async def query(
        self,
        query: KnowledgeQuery,
    ) -> KnowledgeResult:
        """
        Query this domain for relevant knowledge.

        Args:
            query: The knowledge query

        Returns:
            KnowledgeResult with matching items
        """
        pass

    def add_item(self, item: KnowledgeItem) -> None:
        """Add a knowledge item to this domain."""
        item.domain = self.domain_type
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> KnowledgeItem | None:
        """Get a specific item by ID."""
        return self._items.get(item_id)

    def get_items_by_category(
        self,
        category: KnowledgeCategory,
    ) -> list[KnowledgeItem]:
        """Get all items in a category."""
        return [
            item for item in self._items.values()
            if item.category == category
        ]

    def get_items_for_jurisdiction(
        self,
        jurisdiction: Jurisdiction,
    ) -> list[KnowledgeItem]:
        """Get all items applicable to a jurisdiction."""
        return [
            item for item in self._items.values()
            if item.applies_to_jurisdiction(jurisdiction)
        ]

    def search_keywords(
        self,
        keywords: list[str],
        max_results: int = 10,
    ) -> list[KnowledgeItem]:
        """Simple keyword-based search."""
        results: list[tuple[KnowledgeItem, int]] = []

        for item in self._items.values():
            score = 0
            item_text = f"{item.title} {item.content} {' '.join(item.keywords)}".lower()

            for keyword in keywords:
                if keyword.lower() in item_text:
                    score += 1
                if keyword.lower() in [k.lower() for k in item.keywords]:
                    score += 2  # Boost for explicit keyword match

            if score > 0:
                results.append((item, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:max_results]]

    def to_dict(self) -> dict[str, Any]:
        """Convert domain to dictionary."""
        return {
            "domain_type": self.domain_type.value,
            "data_path": str(self.data_path),
            "item_count": self.item_count,
            "loaded": self._loaded,
        }


# ============================================================================
# Knowledge Configuration
# ============================================================================


@dataclass
class KnowledgeConfig:
    """Configuration for the knowledge system."""

    # Data paths
    base_data_path: Path = field(default_factory=lambda: Path("data/knowledge_base"))
    jurisdiction_data_path: Path = field(default_factory=lambda: Path("data/jurisdictions"))

    # Default jurisdiction
    default_jurisdiction: Jurisdiction = field(default_factory=Jurisdiction)

    # Retrieval settings
    default_max_results: int = 5
    enable_rag: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.5

    # Minimum normalized keyword score for a domain hit to be cited. Filters
    # incidental one-word overlap ("what is the weather" brushing against a
    # car-accident item) that would otherwise be presented as a source.
    min_domain_relevance: float = 0.25

    # Domain settings
    enabled_domains: list[KnowledgeDomainType] = field(
        default_factory=lambda: [
            KnowledgeDomainType.LEGAL,
            KnowledgeDomainType.SAFETY,
            KnowledgeDomainType.FINANCIAL,
            KnowledgeDomainType.EMERGENCY,
        ]
    )

    # Citation settings
    require_citations: bool = True
    include_source_urls: bool = True

    # Update settings
    auto_update_check: bool = False
    update_interval_hours: int = 168  # Weekly

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_data_path": str(self.base_data_path),
            "jurisdiction_data_path": str(self.jurisdiction_data_path),
            "default_jurisdiction": self.default_jurisdiction.to_dict(),
            "default_max_results": self.default_max_results,
            "enable_rag": self.enable_rag,
            "embedding_model": self.embedding_model,
            "similarity_threshold": self.similarity_threshold,
            "min_domain_relevance": self.min_domain_relevance,
            "enabled_domains": [d.value for d in self.enabled_domains],
            "require_citations": self.require_citations,
            "include_source_urls": self.include_source_urls,
            "auto_update_check": self.auto_update_check,
            "update_interval_hours": self.update_interval_hours,
        }
