"""
SAIL Knowledge Manager

Orchestrates the complete knowledge domain system including:
- Multi-domain management
- Jurisdiction-aware queries
- Integration with intervention engine
- Knowledge update coordination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.events import EventEmitter

from src.knowledge.base import (
    Jurisdiction,
    KnowledgeCategory,
    KnowledgeConfig,
    KnowledgeDomain,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeResult,
)
from src.knowledge.domains.emergency import EmergencyKnowledgeDomain
from src.knowledge.domains.financial import FinancialKnowledgeDomain
from src.knowledge.domains.legal import LegalKnowledgeDomain
from src.knowledge.domains.safety import SafetyKnowledgeDomain
from src.knowledge.retrieval import RAGPipeline, create_rag_pipeline

logger = logging.getLogger(__name__)


# ============================================================================
# Knowledge Manager Events
# ============================================================================


@dataclass
class KnowledgeEvent:
    """Event from the knowledge system."""

    event_type: str
    domain: KnowledgeDomainType | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeEventType:
    """Types of knowledge events."""

    QUERY_EXECUTED = "query_executed"
    KNOWLEDGE_LOADED = "knowledge_loaded"
    SCAM_DETECTED = "scam_detected"
    EMERGENCY_TRIGGERED = "emergency_triggered"
    JURISDICTION_CHANGED = "jurisdiction_changed"


# ============================================================================
# Knowledge Manager
# ============================================================================


class KnowledgeManager(EventEmitter[KnowledgeEvent]):
    """
    Central manager for the knowledge domain system.

    Coordinates multiple knowledge domains, handles jurisdiction-aware
    queries, and integrates with the intervention engine.
    """

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
    ):
        self.config = config or KnowledgeConfig()

        # Knowledge domains
        self._domains: dict[KnowledgeDomainType, KnowledgeDomain] = {}

        # RAG pipeline
        self._rag_pipeline: RAGPipeline | None = None
        self._rag_enabled = self.config.enable_rag

        # Current jurisdiction
        self._jurisdiction: Jurisdiction = self.config.default_jurisdiction

        # State
        self._initialized = False
        self._running = False

        self.__init_emitter__()

        # Statistics
        self._query_count = 0
        self._last_query_time: datetime | None = None

    @property
    def jurisdiction(self) -> Jurisdiction:
        """Get current jurisdiction."""
        return self._jurisdiction

    @jurisdiction.setter
    def jurisdiction(self, value: Jurisdiction) -> None:
        """Set current jurisdiction."""
        old_jurisdiction = self._jurisdiction
        self._jurisdiction = value

        if old_jurisdiction.full_code != value.full_code:
            self._emit_event(
                KnowledgeEventType.JURISDICTION_CHANGED,
                data={
                    "old": old_jurisdiction.to_dict(),
                    "new": value.to_dict(),
                },
            )
            logger.info(f"Jurisdiction changed: {old_jurisdiction.full_code} -> {value.full_code}")

    @property
    def domains(self) -> dict[KnowledgeDomainType, KnowledgeDomain]:
        """Get all loaded domains."""
        return self._domains

    async def initialize(self) -> bool:
        """
        Initialize the knowledge manager and load all domains.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing knowledge manager...")

        try:
            # Create domain instances
            self._domains = {
                KnowledgeDomainType.LEGAL: LegalKnowledgeDomain(
                    self.config.base_data_path / "legal"
                ),
                KnowledgeDomainType.SAFETY: SafetyKnowledgeDomain(
                    self.config.base_data_path / "safety"
                ),
                KnowledgeDomainType.FINANCIAL: FinancialKnowledgeDomain(
                    self.config.base_data_path / "financial"
                ),
                KnowledgeDomainType.EMERGENCY: EmergencyKnowledgeDomain(
                    self.config.base_data_path / "emergency"
                ),
            }

            # Load domains
            for domain_type in self.config.enabled_domains:
                if domain_type in self._domains:
                    domain = self._domains[domain_type]
                    success = await domain.load()
                    if success:
                        logger.info(f"Loaded {domain.item_count} items from {domain_type.value} domain")
                        self._emit_event(
                            KnowledgeEventType.KNOWLEDGE_LOADED,
                            domain=domain_type,
                            data={"item_count": domain.item_count},
                        )
                    else:
                        logger.warning(f"Failed to load {domain_type.value} domain")

            # Initialize RAG pipeline
            if self._rag_enabled:
                self._rag_pipeline = create_rag_pipeline(self.config)
                await self._rag_pipeline.initialize()

                # Index all items
                all_items = []
                for domain in self._domains.values():
                    all_items.extend(list(domain._items.values()))

                if all_items:
                    indexed = await self._rag_pipeline.index_items(all_items)
                    logger.info(f"Indexed {indexed} items for semantic search")

            self._initialized = True
            logger.info("Knowledge manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize knowledge manager: {e}")
            return False

    async def start(self) -> None:
        """Start the knowledge manager."""
        if not self._initialized:
            await self.initialize()
        self._running = True
        logger.info("Knowledge manager started")

    async def stop(self) -> None:
        """Stop the knowledge manager."""
        self._running = False
        logger.info("Knowledge manager stopped")

    async def query(
        self,
        query_text: str,
        domain: KnowledgeDomainType | None = None,
        category: KnowledgeCategory | None = None,
        jurisdiction: Jurisdiction | None = None,
        max_results: int = 5,
        context: dict[str, Any] | None = None,
    ) -> KnowledgeResult:
        """
        Query the knowledge base.

        Args:
            query_text: Natural language query
            domain: Optional domain to search
            category: Optional category filter
            jurisdiction: Optional jurisdiction (uses current if not specified)
            max_results: Maximum number of results
            context: Optional situational context

        Returns:
            KnowledgeResult with matching items
        """
        # Use current jurisdiction if not specified
        effective_jurisdiction = jurisdiction or self._jurisdiction

        # Build query
        query = KnowledgeQuery(
            query_text=query_text,
            domain=domain,
            category=category,
            jurisdiction=effective_jurisdiction,
            max_results=max_results,
            context=context or {},
        )

        # Use RAG if available and enabled
        if self._rag_pipeline and self._rag_enabled:
            domains_to_search = []
            if domain:
                if domain in self._domains:
                    domains_to_search.append(self._domains[domain])
            else:
                domains_to_search = list(self._domains.values())

            result = await self._rag_pipeline.query_with_context(
                query, domains_to_search
            )
        else:
            # Fall back to direct domain querying
            result = await self._query_domains(query)

        # Update statistics
        self._query_count += 1
        self._last_query_time = datetime.now()

        # Emit event
        self._emit_event(
            KnowledgeEventType.QUERY_EXECUTED,
            domain=domain,
            data={
                "query": query_text,
                "results_count": len(result.items),
                "retrieval_time_ms": result.retrieval_time_ms,
            },
        )

        return result

    async def _query_domains(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Query domains directly without RAG."""
        all_items: list[KnowledgeItem] = []
        all_scores: list[float] = []

        domains_to_search = []
        if query.domain:
            if query.domain in self._domains:
                domains_to_search.append(self._domains[query.domain])
        else:
            domains_to_search = [
                self._domains[dt]
                for dt in self.config.enabled_domains
                if dt in self._domains
            ]

        for domain in domains_to_search:
            result = await domain.query(query)
            all_items.extend(result.items)
            all_scores.extend(result.scores)

        # Sort by score and limit
        combined = list(zip(all_items, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:query.max_results]

        items = [item for item, _ in combined]
        scores = [score for _, score in combined]

        return KnowledgeResult(
            query_id=query.query_id,
            items=items,
            scores=scores,
            total_found=len(all_items),
            domain_searched=query.domain,
            jurisdiction_filtered=query.jurisdiction is not None,
            sources=[item.source for item in items if item.source],
        )

    async def get_traffic_stop_guidance(
        self,
        jurisdiction: Jurisdiction | None = None,
    ) -> KnowledgeResult:
        """
        Get traffic stop guidance for a jurisdiction.

        Convenience method for common use case.

        Args:
            jurisdiction: Jurisdiction to get guidance for

        Returns:
            KnowledgeResult with traffic stop information
        """
        return await self.query(
            query_text="traffic stop rights police",
            domain=KnowledgeDomainType.LEGAL,
            category=KnowledgeCategory.TRAFFIC_STOP,
            jurisdiction=jurisdiction,
            max_results=3,
        )

    async def get_emergency_guidance(
        self,
        emergency_type: str,
    ) -> KnowledgeResult:
        """
        Get emergency guidance for a specific type of emergency.

        Args:
            emergency_type: Type of emergency (e.g., "heart attack", "fire", "choking")

        Returns:
            KnowledgeResult with emergency procedures
        """
        return await self.query(
            query_text=emergency_type,
            domain=KnowledgeDomainType.EMERGENCY,
            max_results=3,
        )

    async def analyze_for_scams(
        self,
        text: str,
    ) -> tuple[float, list[str], KnowledgeResult]:
        """
        Analyze text for potential scam indicators.

        Args:
            text: Text to analyze (message, email, call transcript)

        Returns:
            Tuple of (risk_score, detected_indicators, relevant_knowledge)
        """
        # Get financial domain for scam analysis
        financial_domain = self._domains.get(KnowledgeDomainType.FINANCIAL)

        if isinstance(financial_domain, FinancialKnowledgeDomain):
            risk_score, indicators, items = financial_domain.analyze_for_scams(text)

            if risk_score > 0.3:
                self._emit_event(
                    KnowledgeEventType.SCAM_DETECTED,
                    domain=KnowledgeDomainType.FINANCIAL,
                    data={
                        "risk_score": risk_score,
                        "indicators": indicators,
                    },
                )

            # Build result
            result = KnowledgeResult(
                query_id="scam_analysis",
                items=items,
                scores=[risk_score] * len(items),
                total_found=len(items),
                domain_searched=KnowledgeDomainType.FINANCIAL,
            )

            return risk_score, indicators, result

        # Fallback if domain not loaded
        return 0.0, [], KnowledgeResult(query_id="scam_analysis", items=[])

    async def get_de_escalation_guidance(self) -> KnowledgeResult:
        """
        Get de-escalation techniques.

        Returns:
            KnowledgeResult with de-escalation guidance
        """
        return await self.query(
            query_text="de-escalation calm angry conflict",
            domain=KnowledgeDomainType.SAFETY,
            category=KnowledgeCategory.DE_ESCALATION,
            max_results=3,
        )

    def update_jurisdiction_from_location(
        self,
        country: str,
        state: str | None = None,
        county: str | None = None,
        city: str | None = None,
    ) -> None:
        """
        Update jurisdiction based on location data.

        Args:
            country: Country code (e.g., "US")
            state: State/province code (e.g., "CA")
            county: County/district
            city: City/municipality
        """
        self.jurisdiction = Jurisdiction(
            country=country,
            state=state,
            county=county,
            city=city,
        )

    def _emit_event(
        self,
        event_type: str,
        domain: KnowledgeDomainType | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a knowledge event."""
        self._emit(KnowledgeEvent(event_type=event_type, domain=domain, data=data or {}))

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge manager statistics."""
        domain_stats = {}
        total_items = 0

        for domain_type, domain in self._domains.items():
            domain_stats[domain_type.value] = {
                "item_count": domain.item_count,
                "loaded": domain._loaded,
            }
            total_items += domain.item_count

        return {
            "initialized": self._initialized,
            "running": self._running,
            "total_items": total_items,
            "domains": domain_stats,
            "current_jurisdiction": self._jurisdiction.full_code,
            "rag_enabled": self._rag_enabled,
            "rag_indexed_count": self._rag_pipeline.indexed_count if self._rag_pipeline else 0,
            "query_count": self._query_count,
            "last_query_time": self._last_query_time.isoformat() if self._last_query_time else None,
        }

    async def __aenter__(self) -> KnowledgeManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Factory Functions
# ============================================================================


def create_knowledge_manager(
    config: KnowledgeConfig | None = None,
) -> KnowledgeManager:
    """Create a knowledge manager instance."""
    return KnowledgeManager(config)


async def create_initialized_knowledge_manager(
    config: KnowledgeConfig | None = None,
) -> KnowledgeManager:
    """Create and initialize a knowledge manager."""
    manager = KnowledgeManager(config)
    await manager.initialize()
    return manager
