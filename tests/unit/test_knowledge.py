"""
Unit tests for the Knowledge Domain System (Phase 8).

Tests cover:
- Base types and data models
- Jurisdiction system
- Knowledge domains (Legal, Safety, Financial, Emergency)
- RAG pipeline
- Knowledge manager
"""


import pytest
from src.knowledge.base import (
    Jurisdiction,
    JurisdictionLevel,
    KnowledgeCategory,
    KnowledgeConfig,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeRelevance,
    KnowledgeResult,
)

# ============================================================================
# Jurisdiction Tests
# ============================================================================


class TestJurisdiction:
    """Tests for Jurisdiction data model."""

    def test_jurisdiction_creation(self):
        """Test creating a jurisdiction."""
        jurisdiction = Jurisdiction(
            country="US",
            state="CA",
            county="Los Angeles",
        )

        assert jurisdiction.country == "US"
        assert jurisdiction.state == "CA"
        assert jurisdiction.county == "Los Angeles"

    def test_jurisdiction_normalization(self):
        """Test jurisdiction codes are normalized to uppercase."""
        jurisdiction = Jurisdiction(
            country="us",
            state="ca",
        )

        assert jurisdiction.country == "US"
        assert jurisdiction.state == "CA"

    def test_full_code(self):
        """Test full jurisdiction code generation."""
        jurisdiction = Jurisdiction(
            country="US",
            state="CA",
            county="LA",
        )

        assert jurisdiction.full_code == "US-CA-LA"

    def test_state_code(self):
        """Test state-level code generation."""
        jurisdiction = Jurisdiction(country="US", state="TX")
        assert jurisdiction.state_code == "US-TX"

    def test_from_code(self):
        """Test creating jurisdiction from code string."""
        jurisdiction = Jurisdiction.from_code("US-NY-NYC")

        assert jurisdiction.country == "US"
        assert jurisdiction.state == "NY"
        assert jurisdiction.county == "NYC"

    def test_matches_federal(self):
        """Test federal level matching."""
        us_ca = Jurisdiction(country="US", state="CA")
        us_tx = Jurisdiction(country="US", state="TX")
        ca_on = Jurisdiction(country="CA", state="ON")

        assert us_ca.matches(us_tx, JurisdictionLevel.FEDERAL)
        assert not us_ca.matches(ca_on, JurisdictionLevel.FEDERAL)

    def test_matches_state(self):
        """Test state level matching."""
        ca1 = Jurisdiction(country="US", state="CA", county="LA")
        ca2 = Jurisdiction(country="US", state="CA", county="SF")
        tx = Jurisdiction(country="US", state="TX")

        assert ca1.matches(ca2, JurisdictionLevel.STATE)
        assert not ca1.matches(tx, JurisdictionLevel.STATE)

    def test_matches_universal(self):
        """Test universal always matches."""
        us = Jurisdiction(country="US")
        uk = Jurisdiction(country="UK")

        assert us.matches(uk, JurisdictionLevel.UNIVERSAL)

    def test_to_dict(self):
        """Test converting jurisdiction to dictionary."""
        jurisdiction = Jurisdiction(country="US", state="CA")
        data = jurisdiction.to_dict()

        assert data["country"] == "US"
        assert data["state"] == "CA"

    def test_from_dict(self):
        """Test creating jurisdiction from dictionary."""
        data = {"country": "US", "state": "NY", "city": "NYC"}
        jurisdiction = Jurisdiction.from_dict(data)

        assert jurisdiction.country == "US"
        assert jurisdiction.state == "NY"
        assert jurisdiction.city == "NYC"


# ============================================================================
# Knowledge Item Tests
# ============================================================================


class TestKnowledgeItem:
    """Tests for KnowledgeItem data model."""

    def test_item_creation(self):
        """Test creating a knowledge item."""
        item = KnowledgeItem(
            title="Test Item",
            content="Test content",
            domain=KnowledgeDomainType.LEGAL,
            category=KnowledgeCategory.TRAFFIC_STOP,
        )

        assert item.title == "Test Item"
        assert item.domain == KnowledgeDomainType.LEGAL
        assert item.category == KnowledgeCategory.TRAFFIC_STOP
        assert item.item_id is not None

    def test_item_jurisdiction_universal(self):
        """Test item with universal jurisdiction."""
        item = KnowledgeItem(
            title="Universal Item",
            jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        )

        us = Jurisdiction(country="US", state="CA")
        uk = Jurisdiction(country="UK")

        assert item.applies_to_jurisdiction(us)
        assert item.applies_to_jurisdiction(uk)

    def test_item_jurisdiction_state(self):
        """Test item with state-level jurisdiction."""
        item = KnowledgeItem(
            title="California Item",
            jurisdiction_level=JurisdictionLevel.STATE,
            applicable_jurisdictions=["US-CA"],
        )

        ca = Jurisdiction(country="US", state="CA")
        tx = Jurisdiction(country="US", state="TX")

        assert item.applies_to_jurisdiction(ca)
        assert not item.applies_to_jurisdiction(tx)

    def test_item_to_dict(self):
        """Test converting item to dictionary."""
        item = KnowledgeItem(
            title="Test",
            content="Content",
            keywords=["test", "example"],
        )

        data = item.to_dict()

        assert data["title"] == "Test"
        assert data["content"] == "Content"
        assert "test" in data["keywords"]

    def test_item_from_dict(self):
        """Test creating item from dictionary."""
        data = {
            "item_id": "test-123",
            "title": "Test Item",
            "domain": "legal",
            "category": "traffic_stop",
            "content": "Content",
            "relevance": "high",
        }

        item = KnowledgeItem.from_dict(data)

        assert item.item_id == "test-123"
        assert item.title == "Test Item"
        assert item.domain == KnowledgeDomainType.LEGAL
        assert item.relevance == KnowledgeRelevance.HIGH


# ============================================================================
# Knowledge Query Tests
# ============================================================================


class TestKnowledgeQuery:
    """Tests for KnowledgeQuery data model."""

    def test_query_creation(self):
        """Test creating a knowledge query."""
        query = KnowledgeQuery(
            query_text="traffic stop rights",
            domain=KnowledgeDomainType.LEGAL,
            max_results=5,
        )

        assert query.query_text == "traffic stop rights"
        assert query.domain == KnowledgeDomainType.LEGAL
        assert query.max_results == 5

    def test_query_with_jurisdiction(self):
        """Test query with jurisdiction filter."""
        jurisdiction = Jurisdiction(country="US", state="CA")
        query = KnowledgeQuery(
            query_text="tenant rights",
            jurisdiction=jurisdiction,
        )

        assert query.jurisdiction is not None
        assert query.jurisdiction.state == "CA"

    def test_query_to_dict(self):
        """Test converting query to dictionary."""
        query = KnowledgeQuery(
            query_text="test query",
            domain=KnowledgeDomainType.SAFETY,
        )

        data = query.to_dict()

        assert data["query_text"] == "test query"
        assert data["domain"] == "safety"


# ============================================================================
# Knowledge Result Tests
# ============================================================================


class TestKnowledgeResult:
    """Tests for KnowledgeResult data model."""

    def test_result_creation(self):
        """Test creating a knowledge result."""
        items = [
            KnowledgeItem(title="Item 1", content="Content 1"),
            KnowledgeItem(title="Item 2", content="Content 2"),
        ]

        result = KnowledgeResult(
            query_id="test-query",
            items=items,
            scores=[0.9, 0.7],
            total_found=2,
        )

        assert len(result.items) == 2
        assert result.total_found == 2

    def test_get_best_item(self):
        """Test getting the best item from results."""
        items = [
            KnowledgeItem(title="Best", content="Best content"),
            KnowledgeItem(title="Second", content="Second content"),
        ]

        result = KnowledgeResult(
            query_id="test",
            items=items,
            scores=[0.95, 0.7],
        )

        best = result.get_best_item()
        assert best is not None
        assert best.title == "Best"

    def test_get_summary(self):
        """Test getting combined summary."""
        items = [
            KnowledgeItem(title="Item 1", summary="Summary 1"),
            KnowledgeItem(title="Item 2", summary="Summary 2"),
        ]

        result = KnowledgeResult(
            query_id="test",
            items=items,
            scores=[0.9, 0.8],
        )

        summary = result.get_summary()
        assert "Summary 1" in summary
        assert "Summary 2" in summary

    def test_empty_result(self):
        """Test empty result handling."""
        result = KnowledgeResult(query_id="test", items=[])

        assert result.get_best_item() is None
        assert result.get_summary() == "No relevant knowledge found."


# ============================================================================
# Legal Domain Tests
# ============================================================================


class TestLegalDomain:
    """Tests for LegalKnowledgeDomain."""

    @pytest.mark.asyncio
    async def test_domain_creation(self):
        """Test creating legal domain."""
        from src.knowledge.domains.legal import LegalKnowledgeDomain

        domain = LegalKnowledgeDomain()
        assert domain.domain_type == KnowledgeDomainType.LEGAL

    @pytest.mark.asyncio
    async def test_domain_load(self):
        """Test loading legal domain knowledge."""
        from src.knowledge.domains.legal import LegalKnowledgeDomain

        domain = LegalKnowledgeDomain()
        success = await domain.load()

        assert success
        assert domain.item_count > 0

    @pytest.mark.asyncio
    async def test_traffic_stop_query(self):
        """Test querying traffic stop rights."""
        from src.knowledge.domains.legal import LegalKnowledgeDomain

        domain = LegalKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="traffic stop rights",
            category=KnowledgeCategory.TRAFFIC_STOP,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0
        assert result.domain_searched == KnowledgeDomainType.LEGAL

    @pytest.mark.asyncio
    async def test_jurisdiction_filtering(self):
        """Test jurisdiction-specific query."""
        from src.knowledge.domains.legal import LegalKnowledgeDomain

        domain = LegalKnowledgeDomain()
        await domain.load()

        ca_jurisdiction = Jurisdiction(country="US", state="CA")
        query = KnowledgeQuery(
            query_text="traffic stop",
            jurisdiction=ca_jurisdiction,
            max_results=5,
        )

        result = await domain.query(query)

        # Should have California-specific results
        assert result.jurisdiction_filtered

    @pytest.mark.asyncio
    async def test_get_traffic_stop_guidance(self):
        """Test convenience method for traffic stop guidance."""
        from src.knowledge.domains.legal import LegalKnowledgeDomain

        domain = LegalKnowledgeDomain()
        await domain.load()

        ca_jurisdiction = Jurisdiction(country="US", state="CA")
        items = domain.get_traffic_stop_guidance(ca_jurisdiction)

        assert len(items) > 0


# ============================================================================
# Safety Domain Tests
# ============================================================================


class TestSafetyDomain:
    """Tests for SafetyKnowledgeDomain."""

    @pytest.mark.asyncio
    async def test_domain_creation(self):
        """Test creating safety domain."""
        from src.knowledge.domains.safety import SafetyKnowledgeDomain

        domain = SafetyKnowledgeDomain()
        assert domain.domain_type == KnowledgeDomainType.SAFETY

    @pytest.mark.asyncio
    async def test_domain_load(self):
        """Test loading safety domain knowledge."""
        from src.knowledge.domains.safety import SafetyKnowledgeDomain

        domain = SafetyKnowledgeDomain()
        success = await domain.load()

        assert success
        assert domain.item_count > 0

    @pytest.mark.asyncio
    async def test_de_escalation_query(self):
        """Test querying de-escalation techniques."""
        from src.knowledge.domains.safety import SafetyKnowledgeDomain

        domain = SafetyKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="de-escalation calm angry",
            category=KnowledgeCategory.DE_ESCALATION,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_safe_meeting_query(self):
        """Test querying safe meeting protocols."""
        from src.knowledge.domains.safety import SafetyKnowledgeDomain

        domain = SafetyKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="online dating meeting safety",
            category=KnowledgeCategory.SAFE_MEETING,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0


# ============================================================================
# Financial Domain Tests
# ============================================================================


class TestFinancialDomain:
    """Tests for FinancialKnowledgeDomain."""

    @pytest.mark.asyncio
    async def test_domain_creation(self):
        """Test creating financial domain."""
        from src.knowledge.domains.financial import FinancialKnowledgeDomain

        domain = FinancialKnowledgeDomain()
        assert domain.domain_type == KnowledgeDomainType.FINANCIAL

    @pytest.mark.asyncio
    async def test_domain_load(self):
        """Test loading financial domain knowledge."""
        from src.knowledge.domains.financial import FinancialKnowledgeDomain

        domain = FinancialKnowledgeDomain()
        success = await domain.load()

        assert success
        assert domain.item_count > 0

    @pytest.mark.asyncio
    async def test_scam_query(self):
        """Test querying scam detection knowledge."""
        from src.knowledge.domains.financial import FinancialKnowledgeDomain

        domain = FinancialKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="gift card scam urgency",
            category=KnowledgeCategory.SCAM_DETECTION,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_analyze_scam_text(self):
        """Test analyzing text for scam indicators."""
        from src.knowledge.domains.financial import FinancialKnowledgeDomain

        domain = FinancialKnowledgeDomain()
        await domain.load()

        scam_text = "ACT NOW! Send gift cards immediately or your account will be suspended!"
        risk_score, indicators, items = domain.analyze_for_scams(scam_text)

        assert risk_score > 0.3
        assert len(indicators) > 0
        assert "urgency" in indicators or "gift card" in indicators

    @pytest.mark.asyncio
    async def test_analyze_normal_text(self):
        """Test that normal text doesn't trigger scam detection."""
        from src.knowledge.domains.financial import FinancialKnowledgeDomain

        domain = FinancialKnowledgeDomain()
        await domain.load()

        normal_text = "Let's meet for coffee tomorrow at 3pm at the usual place."
        risk_score, indicators, _ = domain.analyze_for_scams(normal_text)

        assert risk_score < 0.3
        assert len(indicators) == 0


class TestScamIndicators:
    """Tests for scam indicator analysis."""

    def test_analyze_scam_risk(self):
        """Test scam risk analysis function."""
        from src.knowledge.domains.financial import analyze_scam_risk

        # Text with multiple scam indicators
        text = "Act now! Wire transfer needed urgently. Don't tell anyone!"
        risk_score, indicators = analyze_scam_risk(text)

        assert risk_score > 0.5
        assert "act now" in indicators or "urgency" in indicators

    def test_clean_text_low_risk(self):
        """Test clean text has low risk score."""
        from src.knowledge.domains.financial import analyze_scam_risk

        text = "Thank you for your order. Your package will arrive next week."
        risk_score, indicators = analyze_scam_risk(text)

        assert risk_score < 0.3


# ============================================================================
# Emergency Domain Tests
# ============================================================================


class TestEmergencyDomain:
    """Tests for EmergencyKnowledgeDomain."""

    @pytest.mark.asyncio
    async def test_domain_creation(self):
        """Test creating emergency domain."""
        from src.knowledge.domains.emergency import EmergencyKnowledgeDomain

        domain = EmergencyKnowledgeDomain()
        assert domain.domain_type == KnowledgeDomainType.EMERGENCY

    @pytest.mark.asyncio
    async def test_domain_load(self):
        """Test loading emergency domain knowledge."""
        from src.knowledge.domains.emergency import EmergencyKnowledgeDomain

        domain = EmergencyKnowledgeDomain()
        success = await domain.load()

        assert success
        assert domain.item_count > 0

    @pytest.mark.asyncio
    async def test_cpr_query(self):
        """Test querying CPR procedures."""
        from src.knowledge.domains.emergency import EmergencyKnowledgeDomain

        domain = EmergencyKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="CPR heart attack not breathing",
            category=KnowledgeCategory.MEDICAL_EMERGENCY,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_fire_query(self):
        """Test querying fire safety procedures."""
        from src.knowledge.domains.emergency import EmergencyKnowledgeDomain

        domain = EmergencyKnowledgeDomain()
        await domain.load()

        query = KnowledgeQuery(
            query_text="house fire escape",
            category=KnowledgeCategory.FIRE,
            max_results=3,
        )

        result = await domain.query(query)

        assert len(result.items) > 0


# ============================================================================
# RAG Pipeline Tests
# ============================================================================


class TestVectorStore:
    """Tests for VectorStore."""

    def test_store_creation(self):
        """Test creating vector store."""
        from src.knowledge.retrieval import VectorStore

        store = VectorStore(dimension=384)
        assert store.count == 0

    def test_add_and_search(self):
        """Test adding and searching vectors."""
        import numpy as np
        from src.knowledge.retrieval import VectorStore

        store = VectorStore(dimension=10)

        # Add some vectors
        vec1 = np.random.randn(10).tolist()
        vec2 = np.random.randn(10).tolist()

        store.add("item1", vec1, {"category": "test1"})
        store.add("item2", vec2, {"category": "test2"})

        assert store.count == 2

        # Search
        results = store.search(vec1, top_k=2)
        assert len(results) > 0
        assert results[0][0] == "item1"  # Most similar should be itself

    def test_remove(self):
        """Test removing vectors."""
        import numpy as np
        from src.knowledge.retrieval import VectorStore

        store = VectorStore(dimension=10)
        vec = np.random.randn(10).tolist()

        store.add("item1", vec)
        assert store.count == 1

        store.remove("item1")
        assert store.count == 0


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider."""

    @pytest.mark.asyncio
    async def test_provider_creation(self):
        """Test creating embedding provider."""
        from src.knowledge.retrieval import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()
        success = await provider.initialize()

        assert success

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Test embedding a single text."""
        from src.knowledge.retrieval import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()
        await provider.initialize()

        embedding = await provider.embed("Test text for embedding")

        assert len(embedding) > 0
        assert isinstance(embedding[0], float)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test embedding multiple texts."""
        from src.knowledge.retrieval import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()
        await provider.initialize()

        texts = ["First text", "Second text", "Third text"]
        embeddings = await provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == len(embeddings[1])


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_creation(self):
        """Test creating RAG pipeline."""
        from src.knowledge.retrieval import RAGPipeline

        pipeline = RAGPipeline()
        success = await pipeline.initialize()

        assert success

    @pytest.mark.asyncio
    async def test_index_items(self):
        """Test indexing knowledge items."""
        from src.knowledge.retrieval import RAGPipeline

        pipeline = RAGPipeline()
        await pipeline.initialize()

        items = [
            KnowledgeItem(title="Traffic Stop", content="Your rights during a traffic stop"),
            KnowledgeItem(title="CPR", content="How to perform CPR"),
        ]

        indexed = await pipeline.index_items(items)

        assert indexed == 2
        assert pipeline.indexed_count == 2

    @pytest.mark.asyncio
    async def test_retrieve(self):
        """Test retrieving knowledge."""
        from src.knowledge.retrieval import RAGPipeline

        pipeline = RAGPipeline()
        await pipeline.initialize()

        items = [
            KnowledgeItem(
                title="Traffic Stop Rights",
                content="You have the right to remain silent during a traffic stop",
                keywords=["traffic", "stop", "rights", "police"],
            ),
            KnowledgeItem(
                title="CPR Basics",
                content="Push hard and fast on the center of the chest",
                keywords=["cpr", "heart", "chest", "emergency"],
            ),
        ]

        await pipeline.index_items(items)

        result = await pipeline.retrieve("traffic stop police rights", max_results=2)

        assert len(result.items) > 0
        assert result.retrieval_time_ms >= 0


# ============================================================================
# Knowledge Manager Tests
# ============================================================================


class TestKnowledgeManager:
    """Tests for KnowledgeManager."""

    @pytest.mark.asyncio
    async def test_manager_creation(self):
        """Test creating knowledge manager."""
        from src.knowledge.manager import KnowledgeManager

        manager = KnowledgeManager()
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_manager_initialize(self):
        """Test initializing knowledge manager."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)  # Disable RAG for faster test
        manager = KnowledgeManager(config)

        success = await manager.initialize()

        assert success
        assert manager._initialized
        assert len(manager._domains) > 0

    @pytest.mark.asyncio
    async def test_basic_query(self):
        """Test basic knowledge query."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        result = await manager.query("traffic stop rights")

        assert isinstance(result, KnowledgeResult)

    @pytest.mark.asyncio
    async def test_domain_specific_query(self):
        """Test domain-specific query."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        result = await manager.query(
            "de-escalation techniques",
            domain=KnowledgeDomainType.SAFETY,
        )

        assert result.domain_searched == KnowledgeDomainType.SAFETY

    @pytest.mark.asyncio
    async def test_jurisdiction_update(self):
        """Test jurisdiction update."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        events = []
        manager.register_event_callback(lambda e: events.append(e))

        manager.update_jurisdiction_from_location("US", "CA")

        assert manager.jurisdiction.state == "CA"
        assert any(e.event_type == "jurisdiction_changed" for e in events)

    @pytest.mark.asyncio
    async def test_traffic_stop_guidance(self):
        """Test traffic stop guidance convenience method."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        result = await manager.get_traffic_stop_guidance()

        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_emergency_guidance(self):
        """Test emergency guidance convenience method."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        result = await manager.get_emergency_guidance("heart attack")

        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_scam_analysis(self):
        """Test scam analysis method."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        risk, indicators, result = await manager.analyze_for_scams(
            "Send gift cards now! This is urgent!"
        )

        assert risk > 0.3
        assert len(indicators) > 0

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting manager statistics."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        stats = manager.get_stats()

        assert "total_items" in stats
        assert "domains" in stats
        assert stats["initialized"] is True

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)

        async with KnowledgeManager(config) as manager:
            assert manager._running
            result = await manager.query("test")
            assert result is not None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_knowledge_manager(self):
        """Test create_knowledge_manager factory."""
        from src.knowledge.manager import create_knowledge_manager

        manager = create_knowledge_manager()
        assert manager is not None
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_create_initialized_knowledge_manager(self):
        """Test create_initialized_knowledge_manager factory."""
        from src.knowledge.manager import create_initialized_knowledge_manager

        config = KnowledgeConfig(enable_rag=False)
        manager = await create_initialized_knowledge_manager(config)

        assert manager is not None
        assert manager._initialized

    @pytest.mark.asyncio
    async def test_create_rag_pipeline(self):
        """Test create_rag_pipeline factory."""
        from src.knowledge.retrieval import create_rag_pipeline

        pipeline = create_rag_pipeline()
        assert pipeline is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestKnowledgeIntegration:
    """Integration tests for knowledge system."""

    @pytest.mark.asyncio
    async def test_full_query_flow(self):
        """Test complete query flow from manager through domains."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        # Set jurisdiction
        manager.update_jurisdiction_from_location("US", "CA")

        # Query with jurisdiction context
        result = await manager.query(
            "traffic stop rights california",
            domain=KnowledgeDomainType.LEGAL,
            max_results=3,
        )

        assert len(result.items) > 0
        assert result.domain_searched == KnowledgeDomainType.LEGAL

    @pytest.mark.asyncio
    async def test_multi_domain_query(self):
        """Test querying across multiple domains."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        # Query without domain filter
        result = await manager.query(
            "safety help emergency",
            max_results=10,
        )

        # Should potentially return results from multiple domains
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_scam_detection_integration(self):
        """Test scam detection with knowledge retrieval."""
        from src.knowledge.manager import KnowledgeManager

        config = KnowledgeConfig(enable_rag=False)
        manager = KnowledgeManager(config)
        await manager.initialize()

        # Analyze scam text
        scam_text = """
        Congratulations! You've won $1,000,000!
        To claim your prize, send gift cards worth $500 immediately.
        This offer expires in 24 hours. Act now! Don't tell anyone!
        """

        risk_score, indicators, knowledge = await manager.analyze_for_scams(scam_text)

        assert risk_score > 0.5
        assert "gift card" in indicators or "act now" in indicators
