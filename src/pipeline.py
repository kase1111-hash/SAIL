"""
SAIL Pipeline

Connects LLM, context, knowledge retrieval, intervention engine,
sensor fusion, user management, and hub coordination into a single
end-to-end query-handling pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

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
from src.sensors.base import SituationalState
from src.sensors.fusion import SensorFusionManager, create_default_fusion_manager
from src.sensors.temporal import TemporalAnalysis, TemporalSensor, get_time_context
# Import user types directly from base module (not via __init__)
# to avoid triggering cryptography import from isolation.py
from src.users.base import AccessLevel, User, UserRole

if TYPE_CHECKING:
    from src.config.schema import Config
    from src.users.manager import UserManager

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
    user_name: str | None = None
    guardian_alerts: list[Any] = field(default_factory=list)


class Pipeline:
    """
    End-to-end query pipeline connecting context, knowledge, intervention,
    sensor fusion, user management, and LLM.

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
        user_manager: UserManager | None = None,  # noqa: F821
        sensor_fusion: SensorFusionManager | None = None,
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

        # Phase 4: Multi-user support
        self._user_manager = user_manager
        self._users_enabled = bool(config.users)

        # Phase 4: Sensor fusion (replaces temporal-only path when enabled)
        self._sensor_fusion = sensor_fusion
        self._situational_state: SituationalState | None = None

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

        # Temporal sensor (always available as fallback)
        await self._temporal.initialize()

        # Phase 4: Sensor fusion
        if self._sensor_fusion is not None:
            await self._sensor_fusion.initialize()
            await self._sensor_fusion.start()
            logger.info("Sensor fusion started")

        # Phase 4: User management (lazy import to avoid cryptography dep at module level)
        if self._users_enabled and self._user_manager is None:
            try:
                from src.users.manager import UserManager, create_user_manager
                self._user_manager = create_user_manager()
                await self._user_manager.start()
                self._register_config_users()
                logger.info("User manager started")
            except Exception as e:
                logger.warning(f"User manager initialization failed: {e}")
                self._user_manager = None
        elif self._user_manager is not None:
            if not self._user_manager._running:
                await self._user_manager.start()

        self._started = True
        logger.info("Pipeline started")

    def _register_config_users(self) -> None:
        """Register users from the config file into UserManager."""
        for user_cfg in self._config.users:
            existing = self._user_manager.get_user_by_name(user_cfg.name)
            if existing:
                continue
            guardian_name = user_cfg.guardian if hasattr(user_cfg, "guardian") else None
            self._user_manager.register_user(
                name=user_cfg.name,
                role=UserRole(user_cfg.role.value),
                access_level=AccessLevel(user_cfg.access.value),
                restrictions=list(user_cfg.restrictions) if user_cfg.restrictions else None,
                guardian_name=guardian_name,
                custom_briefings=list(user_cfg.custom_briefings) if user_cfg.custom_briefings else None,
            )
            logger.debug(f"Registered user from config: {user_cfg.name}")

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
        if self._sensor_fusion:
            await self._sensor_fusion.stop()
        if self._user_manager:
            await self._user_manager.stop()
        self._started = False
        logger.info("Pipeline stopped")

    async def identify_speaker(
        self,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        explicit_name: str | None = None,
    ) -> User | None:
        """
        Identify a user by voice audio or explicit name.

        Returns the identified User or None if no match.
        """
        if not self._user_manager:
            return None
        user, confidence = await self._user_manager.identify_user(
            audio=audio,
            sample_rate=sample_rate,
            explicit_name=explicit_name,
        )
        if user:
            logger.info(f"Identified user: {user.name} (confidence={confidence:.2f})")
        return user

    async def set_active_user(self, name: str) -> User | None:
        """Set the active user by name (text-mode switch, no voice verification)."""
        if not self._user_manager:
            return None
        user = self._user_manager.get_user_by_name(name)
        if user:
            from src.users.base import IdentificationMethod
            await self._user_manager._start_session(
                user, IdentificationMethod.EXPLICIT, 1.0,
            )
            logger.info(f"Active user set to: {user.name}")
        return user

    def get_current_user(self) -> User | None:
        """Get the currently active user."""
        if not self._user_manager:
            return None
        return self._user_manager.get_current_user()

    async def read_sensors(self) -> SituationalState | None:
        """Read current sensor fusion state."""
        if self._sensor_fusion:
            self._situational_state = self._sensor_fusion.situational_state
            return self._situational_state
        return None

    async def query(self, user_input: str) -> QueryResult:
        """
        Process a user query through the full pipeline:
        sensors -> user context -> knowledge retrieval -> intervention check -> LLM -> response.
        """
        if not self._started:
            raise RuntimeError("Pipeline not started. Call await pipeline.start() first.")

        # 0. Get current user (if user management is active)
        current_user = self.get_current_user()
        user_name = current_user.name if current_user else "User"

        # 1. Record user input in context buffer
        self._context_mgr.add_user_input(user_input)

        # 2. Read sensor state (fusion or temporal-only)
        temporal = await self._read_temporal()
        situational = await self.read_sensors() if self._sensor_fusion else None

        # Build location context string from sensors
        location_context = self._build_location_context(situational)

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
        intervention_result = await self._evaluate_intervention(
            user_input, temporal, situational
        )

        # 5. Build messages for the LLM
        messages = self._build_messages(
            user_input, rag_result, temporal, intervention_result,
            user_name=user_name, location_context=location_context,
            situational=situational,
        )

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

        # 8. Check guardian alerts (if user is a minor with a guardian)
        guardian_alerts = self._check_guardian_alerts(
            current_user, intervention_result, temporal, situational
        )

        # 9. Record assistant response in context buffer
        self._context_mgr.add_assistant_response(response_text)

        return QueryResult(
            response=response_text,
            rag_result=rag_result,
            citations=citations,
            intervention=intervention_result,
            intervention_mode=self._intervention.current_mode.value,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            user_name=user_name if current_user else None,
            guardian_alerts=guardian_alerts,
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
        situational: SituationalState | None = None,
    ) -> Intervention | None:
        """Evaluate user input for risk patterns and run intervention engine."""
        context: dict[str, Any] = {"query": user_input}

        # Add temporal risk signals
        if temporal:
            context["is_late_night"] = temporal.is_late_night
            context["time_context"] = temporal.time_context.value

        # Add sensor fusion signals (override temporal if available)
        if situational:
            context["is_late_night"] = situational.is_late_night
            context["motion_state"] = situational.motion_state.value
            context["location_context"] = situational.location_context.value
            context["audio_environment"] = situational.audio_environment.value
            context["stress_level"] = situational.stress_level.value
            if situational.threat_cues:
                context["threat_cues"] = situational.threat_cues
            if situational.risk_factors:
                context["risk_factors"] = situational.risk_factors
            if situational.jurisdiction:
                context["jurisdiction"] = {
                    "country": situational.jurisdiction.country,
                    "state": situational.jurisdiction.state,
                }

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

    def _build_location_context(self, situational: SituationalState | None) -> str:
        """Build location context string from sensor fusion state."""
        if not situational:
            return "Unknown"

        parts = []

        if situational.jurisdiction:
            if situational.jurisdiction.state:
                parts.append(situational.jurisdiction.state)
            if situational.jurisdiction.country:
                parts.append(situational.jurisdiction.country)

        if situational.location_context.value != "unknown":
            parts.append(f"({situational.location_context.value})")

        if situational.motion_state.value not in ("unknown", "stationary"):
            parts.append(f"[{situational.motion_state.value}]")

        return " ".join(parts) if parts else "Unknown"

    def _build_messages(
        self,
        user_input: str,
        rag_result: RAGResult | None,
        temporal: TemporalAnalysis | None,
        intervention: Intervention | None,
        user_name: str = "User",
        location_context: str = "Unknown",
        situational: SituationalState | None = None,
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

        # System prompt with user-aware and location-aware context
        system_prompt = self._prompts.render(
            "system_base",
            user_name=user_name,
            location_context=location_context,
            time_context=time_str,
            intervention_mode=self._intervention.current_mode.value,
        )
        messages.append(Message(role="system", content=system_prompt))

        # Inject sensor fusion context if available
        if situational:
            sensor_parts = []
            if situational.motion_state.value not in ("unknown", "stationary"):
                sensor_parts.append(f"Motion: {situational.motion_state.value}")
            if situational.audio_environment.value != "quiet":
                sensor_parts.append(f"Environment: {situational.audio_environment.value}")
            if situational.stress_level.value not in ("unknown", "low", "normal"):
                sensor_parts.append(f"Stress level: {situational.stress_level.value}")
            if situational.threat_cues:
                sensor_parts.append(f"Threats detected: {', '.join(situational.threat_cues)}")
            if sensor_parts:
                messages.append(Message(
                    role="system",
                    content=f"Situational awareness: {'; '.join(sensor_parts)}",
                ))

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

    def _check_guardian_alerts(
        self,
        user: User | None,
        intervention: Intervention | None,
        temporal: TemporalAnalysis | None,
        situational: SituationalState | None,
    ) -> list[Any]:
        """Check for guardian alerts if the current user is a minor."""
        if not self._user_manager or not user or not user.has_guardian:
            return []

        context: dict[str, Any] = {}
        if intervention:
            context["intervention_mode"] = self._intervention.current_mode.value
            if self._intervention.current_mode == InterventionMode.CRISIS:
                context["intervention_mode"] = "crisis"
        if temporal:
            context["is_late_night"] = temporal.is_late_night
        if situational:
            context["is_late_night"] = situational.is_late_night
            context["location_context"] = situational.location_context.value

        try:
            alerts = self._user_manager.check_guardian_triggers(context)
            if alerts:
                logger.info(f"Guardian alerts triggered: {len(alerts)}")
            return alerts
        except Exception as e:
            logger.warning(f"Guardian alert check failed: {e}")
            return []

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

    @property
    def user_manager(self) -> UserManager | None:
        return self._user_manager

    @property
    def sensor_fusion(self) -> SensorFusionManager | None:
        return self._sensor_fusion

    async def __aenter__(self) -> Pipeline:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
