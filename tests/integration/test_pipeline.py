"""
SAIL Pipeline Integration Tests

Tests the end-to-end query pipeline: user input -> context -> knowledge -> LLM -> response.
Uses a mock LLM provider to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from src.config.schema import Config, ContextConfig, LLMConfig
from src.context.base import EntryType
from src.context.manager import ContextBufferManager
from src.core.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    LLMProviderType,
    Message,
    StreamChunk,
)
from src.core.llm.prompts import PromptLibrary
from src.knowledge.retrieval import RAGPipeline
from src.pipeline import Pipeline, QueryResult


# ============================================================================
# Mock LLM Provider
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

        # Build a response that echoes the user's question
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
    """Minimal config for pipeline testing."""
    return Config(
        context=ContextConfig(
            persist_sessions=False,
            encryption_enabled=False,
        ),
    )


@pytest.fixture
async def pipeline(config: Config, mock_llm: MockLLMProvider) -> Pipeline:
    """Create and start a pipeline with mock LLM."""
    p = Pipeline(config, llm=mock_llm)
    await p.start()
    yield p
    await p.stop()


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipelineLifecycle:
    """Tests for pipeline start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self, config: Config, mock_llm: MockLLMProvider):
        """Test that pipeline starts and stops cleanly."""
        pipeline = Pipeline(config, llm=mock_llm)

        assert not pipeline.is_started
        await pipeline.start()
        assert pipeline.is_started
        await pipeline.stop()
        assert not pipeline.is_started

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, config: Config, mock_llm: MockLLMProvider):
        """Starting an already-started pipeline is a no-op."""
        pipeline = Pipeline(config, llm=mock_llm)
        await pipeline.start()
        await pipeline.start()  # Should not raise
        assert pipeline.is_started
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, config: Config, mock_llm: MockLLMProvider):
        """Stopping an already-stopped pipeline is a no-op."""
        pipeline = Pipeline(config, llm=mock_llm)
        await pipeline.start()
        await pipeline.stop()
        await pipeline.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, config: Config, mock_llm: MockLLMProvider):
        """Test async context manager protocol."""
        async with Pipeline(config, llm=mock_llm) as p:
            assert p.is_started
        assert not p.is_started

    @pytest.mark.asyncio
    async def test_query_before_start_raises(self, config: Config, mock_llm: MockLLMProvider):
        """Querying before start raises RuntimeError."""
        pipeline = Pipeline(config, llm=mock_llm)
        with pytest.raises(RuntimeError, match="not started"):
            await pipeline.query("hello")


class TestPipelineQuery:
    """Tests for the core query flow."""

    @pytest.mark.asyncio
    async def test_basic_query_returns_response(self, pipeline: Pipeline):
        """A simple query returns a non-empty response."""
        result = await pipeline.query("What are my rights during a traffic stop?")

        assert isinstance(result, QueryResult)
        assert result.response
        assert "traffic stop" in result.response

    @pytest.mark.asyncio
    async def test_query_invokes_llm(self, pipeline: Pipeline, mock_llm: MockLLMProvider):
        """Verify the LLM was actually called."""
        await pipeline.query("test question")

        assert mock_llm.generate_count == 1
        assert any(m.role == "user" and "test question" in m.content for m in mock_llm.last_messages)

    @pytest.mark.asyncio
    async def test_query_includes_system_prompt(self, pipeline: Pipeline, mock_llm: MockLLMProvider):
        """Verify the system prompt is included in LLM messages."""
        await pipeline.query("hello")

        system_messages = [m for m in mock_llm.last_messages if m.role == "system"]
        assert len(system_messages) >= 1
        assert "SAIL" in system_messages[0].content

    @pytest.mark.asyncio
    async def test_query_records_context(self, pipeline: Pipeline):
        """Verify that user input and assistant response are recorded in context."""
        result = await pipeline.query("What is a scam?")

        entries = pipeline._context_mgr.get_conversation(max_turns=10)
        user_entries = [e for e in entries if e.entry_type == EntryType.USER_INPUT]
        assistant_entries = [e for e in entries if e.entry_type == EntryType.ASSISTANT_RESPONSE]

        assert any("What is a scam?" in e.content for e in user_entries)
        assert any(result.response in e.content for e in assistant_entries)

    @pytest.mark.asyncio
    async def test_conversation_history_sent_to_llm(self, pipeline: Pipeline, mock_llm: MockLLMProvider):
        """Multi-turn conversation includes prior context in LLM messages."""
        await pipeline.query("first question")
        await pipeline.query("follow-up question")

        # The second call should include history from the first exchange
        messages = mock_llm.last_messages
        non_system = [m for m in messages if m.role != "system"]
        # Should have: prior user, prior assistant, current user
        assert len(non_system) >= 3

    @pytest.mark.asyncio
    async def test_query_timing_populated(self, pipeline: Pipeline):
        """Verify timing fields are populated."""
        result = await pipeline.query("timing test")

        assert result.generation_time_ms > 0
        assert result.retrieval_time_ms >= 0

    @pytest.mark.asyncio
    async def test_rag_result_attached(self, pipeline: Pipeline):
        """Verify RAG result is attached to query result."""
        result = await pipeline.query("What are my legal rights?")

        # RAG pipeline is initialized but has no indexed items,
        # so it should return an empty result (not None)
        assert result.rag_result is not None


class TestPipelineStreaming:
    """Tests for streaming query mode."""

    @pytest.mark.asyncio
    async def test_stream_query_yields_chunks(self, pipeline: Pipeline):
        """Streaming query yields text chunks."""
        chunks = []
        async for chunk in pipeline.stream_query("stream test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "stream test" in full_text

    @pytest.mark.asyncio
    async def test_stream_query_records_context(self, pipeline: Pipeline):
        """Streaming query also records full response in context."""
        chunks = []
        async for chunk in pipeline.stream_query("context test"):
            chunks.append(chunk)

        entries = pipeline._context_mgr.get_conversation(max_turns=10)
        assistant_entries = [e for e in entries if e.entry_type == EntryType.ASSISTANT_RESPONSE]
        assert len(assistant_entries) >= 1


class TestPipelineResilience:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_input(self, pipeline: Pipeline):
        """Pipeline handles empty input gracefully."""
        result = await pipeline.query("")
        assert isinstance(result, QueryResult)

    @pytest.mark.asyncio
    async def test_long_input(self, pipeline: Pipeline):
        """Pipeline handles very long input."""
        long_text = "word " * 1000
        result = await pipeline.query(long_text)
        assert isinstance(result, QueryResult)
        assert result.response

    @pytest.mark.asyncio
    async def test_rag_failure_doesnt_block_response(self, config: Config, mock_llm: MockLLMProvider):
        """If RAG pipeline fails, the query still returns an LLM response."""
        # Create a RAG pipeline that always raises
        broken_rag = RAGPipeline()
        broken_rag._initialized = True
        broken_rag.retrieve = AsyncMock(side_effect=RuntimeError("RAG broken"))

        pipeline = Pipeline(config, llm=mock_llm, rag=broken_rag)
        await pipeline.start()

        result = await pipeline.query("This should still work")
        assert result.response
        assert "This should still work" in result.response
        assert result.rag_result is None

        await pipeline.stop()
