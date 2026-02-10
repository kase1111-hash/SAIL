"""
SAIL Pipeline

Connects LLM, context, knowledge retrieval, intervention engine, and
temporal awareness into a single end-to-end query-handling pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.context.base import EntryType
from src.context.manager import ContextBufferManager, create_context_manager
from src.core.llm.base import GenerationConfig, LLMProvider, Message
from src.core.llm.factory import LLMProviderFactory
from src.core.llm.prompts import PromptLibrary, get_prompt_library
from src.intervention.base import Intervention, InterventionMode
from src.intervention.engine import InterventionEngine, create_intervention_engine
from src.intervention.risk import RiskPatternDetector
from src.knowledge.manager import KnowledgeManager, create_knowledge_manager
from src.knowledge.retrieval import RAGPipeline, RAGResult
from src.sensors.temporal import TemporalAnalysis, TemporalSensor, get_time_context

if TYPE_CHECKING:
    from src.config.schema import Config

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from processing a user query through the full pipeline."""

    response: str
    rag_result: RAGResult | None = None
    citations: list[str] = field(default_factory=list)
    intervention: Intervention | None = None
    intervention_mode: str = "ambient"
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class Pipeline:
    """
    End-to-end query pipeline connecting context, knowledge, intervention, and LLM.

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
        knowledge_mgr: KnowledgeManager | None = None,
        intervention_engine: InterventionEngine | None = None,
        prompts: PromptLibrary | None = None,
        # Kept for backward-compat with Phase 1 tests
        rag: RAGPipeline | None = None,
    ) -> None:
        self._config = config
        self._llm = llm or LLMProviderFactory.from_config(config.llm)
        self._context_mgr = context_mgr
        self._knowledge_mgr = knowledge_mgr
        self._intervention = intervention_engine
        self._prompts = prompts or get_prompt_library()
        self._temporal = TemporalSensor()
        self._risk_detector = RiskPatternDetector()
        self._config_context = config.context
        self._started = False

        # Backward-compat: if raw RAG passed (Phase 1 tests), wrap it
        self._legacy_rag = rag

    async def start(self) -> None:
        """Initialize all pipeline components."""
        if self._started:
            return

        # Context manager
        if self._context_mgr is None:
            self._context_mgr = await create_context_manager(self._config_context)
        await self._context_mgr.start()

        # Knowledge manager (loads domains + indexes into RAG)
        if self._knowledge_mgr is None and self._legacy_rag is None:
            self._knowledge_mgr = create_knowledge_manager()
            await self._knowledge_mgr.initialize()
        elif self._legacy_rag is not None:
            # Phase 1 backward-compat path
            await self._legacy_rag.initialize()

        # Intervention engine
        if self._intervention is None:
            self._intervention = create_intervention_engine()
        await self._intervention.start()

        # Temporal sensor
        await self._temporal.initialize()

        self._started = True
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Shut down pipeline components."""
        if not self._started:
            return
        if self._context_mgr:
            await self._context_mgr.stop()
        if self._intervention:
            await self._intervention.stop()
        if self._knowledge_mgr:
            await self._knowledge_mgr.stop()
        self._started = False
        logger.info("Pipeline stopped")

    async def query(self, user_input: str) -> QueryResult:
        """
        Process a user query through the full pipeline:
        temporal context -> knowledge retrieval -> intervention check -> LLM -> response.
        """
        if not self._started:
            raise RuntimeError("Pipeline not started. Call await pipeline.start() first.")

        # 1. Record user input in context buffer
        self._context_mgr.add_user_input(user_input)

        # 2. Get temporal context (no hardware needed)
        temporal = await self._read_temporal()

        # 3. Retrieve relevant knowledge
        rag_result: RAGResult | None = None
        citations: list[str] = []
        retrieval_start = time.time()
        try:
            if self._knowledge_mgr:
                kr = await self._knowledge_mgr.query(user_input, max_results=3)
                rag_result = RAGResult(
                    query=user_input,
                    items=kr.items,
                    scores=kr.scores,
                    augmented_context=kr.get_summary(),
                    citations=kr.get_citations(),
                    retrieval_time_ms=kr.retrieval_time_ms,
                )
                citations = kr.get_citations()
            elif self._legacy_rag:
                rag_result = await self._legacy_rag.retrieve(user_input, max_results=3)
                if rag_result and rag_result.citations:
                    citations = rag_result.citations
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed (continuing without): {e}")
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # 4. Run risk pattern detection on user input
        intervention_result = await self._evaluate_intervention(user_input, temporal)

        # 5. Build messages for the LLM
        messages = self._build_messages(user_input, rag_result, temporal, intervention_result)

        # 6. Generate response
        gen_start = time.time()
        gen_config = GenerationConfig(
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
        )
        result = await self._llm.generate(messages, gen_config)
        generation_time_ms = (time.time() - gen_start) * 1000

        response_text = result.content

        # 7. Append intervention warning to response if triggered
        if intervention_result:
            response_text = self._format_intervention(response_text, intervention_result)

        # 8. Record assistant response in context buffer
        self._context_mgr.add_assistant_response(response_text)

        return QueryResult(
            response=response_text,
            rag_result=rag_result,
            citations=citations,
            intervention=intervention_result,
            intervention_mode=self._intervention.current_mode.value,
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

        temporal = await self._read_temporal()

        rag_result: RAGResult | None = None
        try:
            if self._knowledge_mgr:
                kr = await self._knowledge_mgr.query(user_input, max_results=3)
                rag_result = RAGResult(
                    query=user_input,
                    items=kr.items,
                    scores=kr.scores,
                    augmented_context=kr.get_summary(),
                    citations=kr.get_citations(),
                    retrieval_time_ms=kr.retrieval_time_ms,
                )
            elif self._legacy_rag:
                rag_result = await self._legacy_rag.retrieve(user_input, max_results=3)
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")

        intervention_result = await self._evaluate_intervention(user_input, temporal)
        messages = self._build_messages(user_input, rag_result, temporal, intervention_result)

        gen_config = GenerationConfig(
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
        )

        full_response_parts: list[str] = []
        async for chunk in self._llm.stream(messages, gen_config):
            full_response_parts.append(chunk.content)
            yield chunk.content

        full_response = "".join(full_response_parts)

        if intervention_result:
            warning = f"\n\n[{intervention_result.type.value.upper()}] {intervention_result.message}"
            yield warning
            full_response += warning

        self._context_mgr.add_assistant_response(full_response)

    async def _read_temporal(self) -> TemporalAnalysis | None:
        """Read temporal sensor for time context."""
        try:
            reading = await self._temporal.read()
            if reading and reading.data:
                return reading.data
        except Exception as e:
            logger.debug(f"Temporal sensor read failed: {e}")
        return None

    async def _evaluate_intervention(
        self,
        user_input: str,
        temporal: TemporalAnalysis | None,
    ) -> Intervention | None:
        """Evaluate user input for risk patterns and run intervention engine."""
        context: dict[str, Any] = {"query": user_input}

        # Add temporal risk signals
        if temporal:
            context["is_late_night"] = temporal.is_late_night
            context["time_context"] = temporal.time_context.value

        # Detect text-based risk patterns
        detected_factors = []

        pressure = self._risk_detector.detect_pressure_tactics(user_input)
        if pressure:
            detected_factors.append(pressure)

        financial = self._risk_detector.detect_financial_risk(user_input)
        if financial:
            detected_factors.append(financial)

        if detected_factors:
            context["detected_factors"] = detected_factors

        # Run the intervention engine
        try:
            return await self._intervention.evaluate(context)
        except Exception as e:
            logger.warning(f"Intervention evaluation failed: {e}")
            return None

    def _build_messages(
        self,
        user_input: str,
        rag_result: RAGResult | None,
        temporal: TemporalAnalysis | None,
        intervention: Intervention | None,
    ) -> list[Message]:
        """Build the LLM message list from context, knowledge, time, and user input."""
        messages: list[Message] = []

        # Build time context string
        if temporal:
            time_str = temporal.local_time.strftime("%A %I:%M %p")
            if temporal.is_late_night:
                time_str += " (late night)"
            elif temporal.is_early_morning:
                time_str += " (early morning)"
            if temporal.is_weekend:
                time_str += " (weekend)"
            if temporal.is_holiday and temporal.holiday_name:
                time_str += f" ({temporal.holiday_name})"
        else:
            time_str = datetime.now().strftime("%A %I:%M %p")

        # System prompt with temporal context
        system_prompt = self._prompts.render(
            "system_base",
            user_name="User",
            location_context="Unknown",
            time_context=time_str,
            intervention_mode=self._intervention.current_mode.value,
        )
        messages.append(Message(role="system", content=system_prompt))

        # Inject retrieved knowledge
        if rag_result and rag_result.augmented_context:
            knowledge_msg = f"Relevant knowledge from local database:\n\n{rag_result.augmented_context}"
            if rag_result.citations:
                knowledge_msg += f"\n\nSources: {'; '.join(rag_result.citations)}"
            messages.append(Message(role="system", content=knowledge_msg))

        # Inject intervention guidance if engine is in elevated mode
        if intervention or self._intervention.current_mode != InterventionMode.AMBIENT:
            mode = self._intervention.current_mode
            mode_guidance = {
                InterventionMode.ADVISORY: "You are in advisory mode. Gently mention relevant safety considerations if appropriate.",
                InterventionMode.GUARDIAN: "You are in guardian mode. Actively warn the user about detected risks.",
                InterventionMode.CRISIS: "You are in crisis mode. Provide clear, calm, step-by-step guidance. Keep responses concise and actionable.",
            }
            if mode in mode_guidance:
                messages.append(Message(role="system", content=mode_guidance[mode]))

            if intervention:
                messages.append(Message(
                    role="system",
                    content=f"RISK ALERT: {intervention.message}"
                    + (f"\nDetails: {intervention.details}" if intervention.details else ""),
                ))

        # Conversation history from context buffer
        history = self._context_mgr.get_conversation(max_turns=6)
        for entry in history:
            if entry.entry_type == EntryType.USER_INPUT:
                if entry.content == user_input:
                    continue
                messages.append(Message(role="user", content=entry.content))
            elif entry.entry_type == EntryType.ASSISTANT_RESPONSE:
                messages.append(Message(role="assistant", content=entry.content))

        # Current user input
        messages.append(Message(role="user", content=user_input))

        return messages

    @staticmethod
    def _format_intervention(response: str, intervention: Intervention) -> str:
        """Append intervention warning to the LLM response."""
        label = intervention.type.value.upper()
        warning = f"\n\n[{label}] {intervention.message}"
        if intervention.actions:
            warning += "\nRecommended actions:"
            for action in intervention.actions:
                warning += f"\n  - {action}"
        return response + warning

    @property
    def is_started(self) -> bool:
        return self._started

    async def __aenter__(self) -> Pipeline:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
