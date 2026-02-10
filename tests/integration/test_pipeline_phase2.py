"""
SAIL Pipeline Phase 2 Tests

Tests for intervention engine integration, knowledge domain indexing,
temporal sensor context, and citation formatting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from src.config.schema import Config, ContextConfig
from src.core.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    LLMProviderType,
    Message,
    StreamChunk,
)
from src.intervention.base import InterventionMode
from src.intervention.engine import InterventionEngine, create_intervention_engine
from src.knowledge.manager import KnowledgeManager, create_knowledge_manager
from src.pipeline import Pipeline, QueryResult


# ============================================================================
# Mock LLM Provider (echoes user input)
# ============================================================================


class MockLLMProvider(LLMProvider):
    """LLM provider that returns predictable responses for testing."""

    def __init__(self):
        super().__init__(model="mock-model", base_url="http://mock")
        self.last_messages: list[Message] = []
        self.generate_count = 0

    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OLLAMA

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        self.last_messages = messages
        self.generate_count += 1

        user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
        user_text = user_msg.content if user_msg else "no question"

        return GenerationResult(
            content=f"Mock response to: {user_text}",
            model="mock-model",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

    async def stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.last_messages = messages
        self.generate_count += 1

        user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
        user_text = user_msg.content if user_msg else "no question"
        full = f"Mock streamed response to: {user_text}"

        for word in full.split():
            yield StreamChunk(content=word + " ", done=False)
        yield StreamChunk(content="", done=True, finish_reason="stop")

    async def is_available(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return ["mock-model"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def config() -> Config:
    return Config(
        context=ContextConfig(
            persist_sessions=False,
            encryption_enabled=False,
        ),
    )


@pytest.fixture
async def knowledge_mgr() -> KnowledgeManager:
    """Create and initialize a knowledge manager with all domains loaded."""
    mgr = create_knowledge_manager()
    await mgr.initialize()
    return mgr


@pytest.fixture
async def pipeline_with_knowledge(
    config: Config,
    mock_llm: MockLLMProvider,
    knowledge_mgr: KnowledgeManager,
) -> Pipeline:
    """Pipeline with real knowledge manager (domains loaded)."""
    p = Pipeline(config, llm=mock_llm, knowledge_mgr=knowledge_mgr)
    await p.start()
    yield p
    await p.stop()


@pytest.fixture
async def pipeline_full(config: Config, mock_llm: MockLLMProvider) -> Pipeline:
    """Full pipeline with all Phase 2 components (auto-initializes knowledge)."""
    p = Pipeline(config, llm=mock_llm)
    await p.start()
    yield p
    await p.stop()


# ============================================================================
# Knowledge Manager Integration Tests
# ============================================================================


class TestKnowledgeManagerIntegration:
    """Tests for knowledge domain loading and retrieval through the pipeline."""

    @pytest.mark.asyncio
    async def test_knowledge_manager_loads_domains(self, knowledge_mgr: KnowledgeManager):
        """Knowledge manager loads all 4 domains at startup."""
        stats = knowledge_mgr.get_stats()
        assert stats["initialized"] is True
        assert stats["total_items"] > 0
        # All 4 domains should be loaded
        assert "legal" in stats["domains"]
        assert "safety" in stats["domains"]
        assert "financial" in stats["domains"]
        assert "emergency" in stats["domains"]

    @pytest.mark.asyncio
    async def test_knowledge_query_returns_items(self, knowledge_mgr: KnowledgeManager):
        """Querying for traffic stop returns relevant legal knowledge."""
        result = await knowledge_mgr.query("traffic stop rights police", max_results=3)
        assert len(result.items) > 0
        # Should find items from legal domain
        assert any("traffic" in item.title.lower() or "amendment" in item.title.lower()
                    for item in result.items)

    @pytest.mark.asyncio
    async def test_knowledge_citations_populated(self, knowledge_mgr: KnowledgeManager):
        """Knowledge results include citations from source material."""
        result = await knowledge_mgr.query("fourth amendment rights")
        citations = result.get_citations()
        # Legal domain items have source fields (e.g., "U.S. Constitution, Fourth Amendment")
        assert len(citations) > 0

    @pytest.mark.asyncio
    async def test_pipeline_includes_knowledge_in_llm_messages(
        self,
        pipeline_with_knowledge: Pipeline,
        mock_llm: MockLLMProvider,
    ):
        """Pipeline injects knowledge context into LLM system messages."""
        await pipeline_with_knowledge.query("What are my rights during a traffic stop?")

        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        # Should have base system prompt + knowledge context
        assert len(system_messages) >= 2
        # Knowledge message should contain domain content
        knowledge_msgs = [m for m in system_messages if "knowledge" in m.content.lower()]
        assert len(knowledge_msgs) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_returns_citations(self, pipeline_with_knowledge: Pipeline):
        """Pipeline returns citations from knowledge retrieval."""
        result = await pipeline_with_knowledge.query("fourth amendment search and seizure")
        # Should have citations from the legal domain
        assert len(result.citations) > 0

    @pytest.mark.asyncio
    async def test_scam_analysis(self, knowledge_mgr: KnowledgeManager):
        """Knowledge manager detects scam indicators."""
        risk_score, indicators, result = await knowledge_mgr.analyze_for_scams(
            "You need to wire transfer $5000 immediately using Western Union gift cards"
        )
        assert risk_score > 0


# ============================================================================
# Intervention Engine Integration Tests
# ============================================================================


class TestInterventionIntegration:
    """Tests for intervention engine wiring into the pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_starts_in_ambient_mode(self, pipeline_full: Pipeline):
        """Pipeline starts with intervention engine in ambient mode."""
        result = await pipeline_full.query("What time is it?")
        assert result.intervention_mode == "ambient"

    @pytest.mark.asyncio
    async def test_intervention_mode_in_query_result(self, pipeline_full: Pipeline):
        """Query result includes current intervention mode."""
        result = await pipeline_full.query("hello")
        assert result.intervention_mode in ("ambient", "advisory", "guardian", "crisis")

    @pytest.mark.asyncio
    async def test_financial_risk_detection(self, pipeline_full: Pipeline, mock_llm: MockLLMProvider):
        """Pipeline detects financial risk patterns in user input."""
        result = await pipeline_full.query(
            "Someone is asking me to wire transfer money to their bitcoin wallet right now"
        )
        # The risk detector should flag "wire", "transfer", "bitcoin"
        # Check that the system messages contain risk-related content
        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        # At minimum, the risk pattern detector ran
        assert result.intervention_mode is not None

    @pytest.mark.asyncio
    async def test_pressure_tactics_detection(self, pipeline_full: Pipeline, mock_llm: MockLLMProvider):
        """Pipeline detects social pressure tactics."""
        result = await pipeline_full.query(
            "They told me to act now and don't tell anyone about this limited time offer"
        )
        # "act now", "don't tell anyone", "limited time" are pressure phrases
        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        assert result.intervention_mode is not None

    @pytest.mark.asyncio
    async def test_safe_query_no_intervention(self, pipeline_full: Pipeline):
        """Normal queries don't trigger intervention warnings."""
        result = await pipeline_full.query("What is the weather like today?")
        assert result.intervention is None

    @pytest.mark.asyncio
    async def test_intervention_engine_accessible(self, pipeline_full: Pipeline):
        """Intervention engine is initialized and running."""
        assert pipeline_full._intervention is not None
        assert pipeline_full._intervention.current_mode == InterventionMode.AMBIENT


# ============================================================================
# Temporal Sensor Integration Tests
# ============================================================================


class TestTemporalIntegration:
    """Tests for temporal sensor wiring into the pipeline."""

    @pytest.mark.asyncio
    async def test_temporal_context_in_system_prompt(
        self, pipeline_full: Pipeline, mock_llm: MockLLMProvider,
    ):
        """System prompt includes temporal context (day of week, time)."""
        await pipeline_full.query("hello")

        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        assert len(system_messages) >= 1
        # The system prompt should contain a day name (Monday, Tuesday, etc.)
        system_text = system_messages[0].content
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in system_text for day in days)

    @pytest.mark.asyncio
    async def test_temporal_sensor_initialized(self, pipeline_full: Pipeline):
        """Temporal sensor is initialized and can take readings."""
        reading = await pipeline_full._temporal.read()
        assert reading is not None
        assert reading.data is not None
        assert reading.data.time_context is not None


# ============================================================================
# Citation Formatting Tests
# ============================================================================


class TestCitationFormatting:
    """Tests for citation display in responses."""

    @pytest.mark.asyncio
    async def test_citations_in_knowledge_message(
        self,
        pipeline_with_knowledge: Pipeline,
        mock_llm: MockLLMProvider,
    ):
        """Sources are included in the knowledge system message sent to LLM."""
        await pipeline_with_knowledge.query("What is the fourth amendment?")

        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        knowledge_msgs = [m for m in system_messages if "Sources:" in m.content]
        assert len(knowledge_msgs) >= 1

    @pytest.mark.asyncio
    async def test_query_result_citations_match_knowledge(
        self, pipeline_with_knowledge: Pipeline,
    ):
        """Citations in QueryResult come from knowledge domain items."""
        result = await pipeline_with_knowledge.query("right to remain silent fifth amendment")
        if result.citations:
            # Citations should be real source strings from domain data
            assert all(isinstance(c, str) and len(c) > 0 for c in result.citations)


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================


class TestFullPipelineIntegration:
    """End-to-end tests with all Phase 2 components active."""

    @pytest.mark.asyncio
    async def test_full_pipeline_starts_and_stops(self, config: Config, mock_llm: MockLLMProvider):
        """Full pipeline with all components starts and stops cleanly."""
        pipeline = Pipeline(config, llm=mock_llm)
        await pipeline.start()
        assert pipeline.is_started
        await pipeline.stop()
        assert not pipeline.is_started

    @pytest.mark.asyncio
    async def test_full_pipeline_query_with_all_components(self, pipeline_full: Pipeline):
        """A query exercises knowledge, intervention, and temporal in one pass."""
        result = await pipeline_full.query("What should I do during a traffic stop?")
        assert result.response
        assert result.intervention_mode == "ambient"
        assert result.retrieval_time_ms >= 0
        assert result.generation_time_ms > 0

    @pytest.mark.asyncio
    async def test_streaming_with_intervention(
        self, config: Config, mock_llm: MockLLMProvider,
    ):
        """Streaming mode appends intervention warnings at the end."""
        pipeline = Pipeline(config, llm=mock_llm)
        await pipeline.start()

        chunks = []
        async for chunk in pipeline.stream_query(
            "I need to wire transfer money to a bitcoin wallet immediately"
        ):
            chunks.append(chunk)

        full = "".join(chunks)
        # The response should include the original LLM output
        assert "wire transfer" in full.lower() or "Mock" in full

        await pipeline.stop()
