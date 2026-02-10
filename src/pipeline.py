"""
SAIL Pipeline

Connects LLM, context, knowledge retrieval, and voice I/O into a single
end-to-end query-handling pipeline. This is the integration layer that
was missing — all components existed, none were wired together.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.context.base import EntryType
from src.context.manager import ContextBufferManager, create_context_manager
from src.core.llm.base import GenerationConfig, LLMProvider, Message
from src.core.llm.factory import LLMProviderFactory
from src.core.llm.prompts import PromptLibrary, get_prompt_library
from src.knowledge.retrieval import RAGPipeline, RAGResult, create_rag_pipeline

if TYPE_CHECKING:
    from src.config.schema import Config

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from processing a user query through the full pipeline."""

    response: str
    rag_result: RAGResult | None = None
    citations: list[str] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class Pipeline:
    """
    End-to-end query pipeline connecting context, knowledge, and LLM.

    Usage:
        pipeline = Pipeline(config)
        await pipeline.start()
        result = await pipeline.query("What are my rights during a traffic stop?")
        await pipeline.stop()
    """

    def __init__(
        self,
        config: Config,
        llm: LLMProvider | None = None,
        context_mgr: ContextBufferManager | None = None,
        rag: RAGPipeline | None = None,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self._config = config
        self._llm = llm or LLMProviderFactory.from_config(config.llm)
        self._context_mgr = context_mgr
        self._rag = rag
        self._prompts = prompts or get_prompt_library()
        self._config_context = config.context
        self._started = False

    async def start(self) -> None:
        """Initialize all pipeline components."""
        if self._started:
            return

        # Context manager
        if self._context_mgr is None:
            self._context_mgr = await create_context_manager(self._config_context)
        await self._context_mgr.start()

        # RAG pipeline
        if self._rag is None:
            self._rag = create_rag_pipeline()
        await self._rag.initialize()

        self._started = True
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Shut down pipeline components."""
        if not self._started:
            return
        if self._context_mgr:
            await self._context_mgr.stop()
        self._started = False
        logger.info("Pipeline stopped")

    async def query(self, user_input: str) -> QueryResult:
        """
        Process a user query through context -> knowledge -> LLM.

        Args:
            user_input: The user's question or statement.

        Returns:
            QueryResult with the LLM response, citations, and timing.
        """
        if not self._started:
            raise RuntimeError("Pipeline not started. Call await pipeline.start() first.")

        import time

        # 1. Record user input in context buffer
        self._context_mgr.add_user_input(user_input)

        # 2. Retrieve relevant knowledge
        rag_result: RAGResult | None = None
        retrieval_start = time.time()
        try:
            rag_result = await self._rag.retrieve(user_input, max_results=3)
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed (continuing without): {e}")
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # 3. Build messages for the LLM
        messages = self._build_messages(user_input, rag_result)

        # 4. Generate response
        gen_start = time.time()
        gen_config = GenerationConfig(
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
        )
        result = await self._llm.generate(messages, gen_config)
        generation_time_ms = (time.time() - gen_start) * 1000

        response_text = result.content

        # 5. Append citations if available
        citations: list[str] = []
        if rag_result and rag_result.citations:
            citations = rag_result.citations

        # 6. Record assistant response in context buffer
        self._context_mgr.add_assistant_response(response_text)

        return QueryResult(
            response=response_text,
            rag_result=rag_result,
            citations=citations,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )

    async def stream_query(self, user_input: str) -> AsyncIterator[str]:
        """
        Stream a response token-by-token for real-time voice output.

        Yields text chunks as they arrive from the LLM.
        """
        if not self._started:
            raise RuntimeError("Pipeline not started.")

        self._context_mgr.add_user_input(user_input)

        # Retrieve knowledge (blocking, but fast)
        rag_result: RAGResult | None = None
        try:
            rag_result = await self._rag.retrieve(user_input, max_results=3)
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")

        messages = self._build_messages(user_input, rag_result)
        gen_config = GenerationConfig(
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
        )

        full_response_parts: list[str] = []
        async for chunk in self._llm.stream(messages, gen_config):
            full_response_parts.append(chunk.content)
            yield chunk.content

        full_response = "".join(full_response_parts)
        self._context_mgr.add_assistant_response(full_response)

    def _build_messages(
        self,
        user_input: str,
        rag_result: RAGResult | None,
    ) -> list[Message]:
        """Build the LLM message list from context, knowledge, and user input."""
        messages: list[Message] = []

        # System prompt
        now = datetime.now()
        system_prompt = self._prompts.render(
            "system_base",
            user_name="User",
            location_context="Unknown",
            time_context=now.strftime("%A %I:%M %p"),
            intervention_mode="ambient",
        )
        messages.append(Message(role="system", content=system_prompt))

        # Inject retrieved knowledge as additional system context
        if rag_result and rag_result.augmented_context:
            messages.append(Message(
                role="system",
                content=f"Relevant knowledge from local database:\n\n{rag_result.augmented_context}",
            ))

        # Conversation history from context buffer
        history = self._context_mgr.get_conversation(max_turns=6)
        for entry in history:
            if entry.entry_type == EntryType.USER_INPUT:
                # Skip the current input — we append it explicitly below
                if entry.content == user_input:
                    continue
                messages.append(Message(role="user", content=entry.content))
            elif entry.entry_type == EntryType.ASSISTANT_RESPONSE:
                messages.append(Message(role="assistant", content=entry.content))

        # Current user input
        messages.append(Message(role="user", content=user_input))

        return messages

    @property
    def is_started(self) -> bool:
        return self._started

    async def __aenter__(self) -> Pipeline:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
