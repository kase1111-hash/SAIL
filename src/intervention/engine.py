"""
Intervention engine that orchestrates all intervention modes.

Manages mode transitions, evaluates context, and coordinates
responses across the intervention framework.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .base import (
    Intervention,
    InterventionConfig,
    InterventionEvent,
    InterventionEventType,
    InterventionMode,
    InterventionResponse,
    ModeTransition,
    RiskAssessment,
)
from .modes import (
    AdvisoryModeHandler,
    AmbientModeHandler,
    CrisisModeHandler,
    GuardianModeHandler,
)
from .risk import RiskAssessmentEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Intervention Queue
# ============================================================================


class InterventionQueue:
    """Priority queue for pending interventions."""

    def __init__(self, max_size: int = 10):
        self._queue: list[Intervention] = []
        self._max_size = max_size

    def add(self, intervention: Intervention) -> None:
        """Add intervention to queue."""
        # Remove expired interventions first
        self._cleanup_expired()

        # Add new intervention
        self._queue.append(intervention)

        # Sort by priority (higher priority first)
        self._queue.sort(key=lambda i: i.priority.get_value(), reverse=True)

        # Trim to max size
        if len(self._queue) > self._max_size:
            self._queue = self._queue[:self._max_size]

    def get_next(self) -> Intervention | None:
        """Get and remove highest priority intervention."""
        self._cleanup_expired()

        if self._queue:
            return self._queue.pop(0)
        return None

    def peek(self) -> Intervention | None:
        """View highest priority intervention without removing."""
        self._cleanup_expired()

        if self._queue:
            return self._queue[0]
        return None

    def remove(self, intervention_id: str) -> bool:
        """Remove specific intervention by ID."""
        original_len = len(self._queue)
        self._queue = [i for i in self._queue if i.id != intervention_id]
        return len(self._queue) < original_len

    def _cleanup_expired(self) -> None:
        """Remove expired interventions."""
        self._queue = [i for i in self._queue if not i.is_expired()]

    def clear(self) -> None:
        """Clear all interventions."""
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        """Get count of pending interventions."""
        self._cleanup_expired()
        return len(self._queue)


# ============================================================================
# Intervention Engine
# ============================================================================


class InterventionEngine:
    """
    Main intervention engine that orchestrates all modes.

    Manages:
    - Mode transitions based on risk assessment
    - Intervention generation and delivery
    - Response handling and escalation
    - Event callbacks and logging
    """

    def __init__(self, config: InterventionConfig | None = None):
        self.config = config or InterventionConfig()

        # Current mode
        self._current_mode = InterventionMode.AMBIENT
        self._mode_start_time = datetime.now()

        # Risk assessment
        self._risk_engine = RiskAssessmentEngine(self.config)
        self._last_assessment: RiskAssessment | None = None

        # Mode handlers
        self._handlers: dict[InterventionMode, Any] = {
            InterventionMode.AMBIENT: AmbientModeHandler(self.config),
            InterventionMode.ADVISORY: AdvisoryModeHandler(self.config),
            InterventionMode.GUARDIAN: GuardianModeHandler(self.config),
            InterventionMode.CRISIS: CrisisModeHandler(self.config),
        }

        # Activate initial mode
        self._handlers[self._current_mode].activate()

        # Intervention tracking
        self._intervention_queue = InterventionQueue(self.config.max_pending_interventions)
        self._delivered_interventions: dict[str, Intervention] = {}
        self._intervention_history: deque[Intervention] = deque(maxlen=100)

        # Mode transition tracking
        self._mode_history: deque[ModeTransition] = deque(maxlen=50)
        self._pending_escalation: InterventionMode | None = None
        self._pending_deescalation: InterventionMode | None = None
        self._escalation_timer: datetime | None = None

        # Callbacks
        self._event_callbacks: list[Callable[[InterventionEvent], None]] = []
        self._intervention_callbacks: list[Callable[[Intervention], None]] = []

        # Running state
        self._running = False

    def _safe_create_task(self, coro: Any, name: str | None = None) -> asyncio.Task[Any]:
        """Create a task with error handling to prevent silent failures."""
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """Handle task completion, logging any exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Background task {task.get_name()} failed: {exc}")

    @property
    def current_mode(self) -> InterventionMode:
        """Get current intervention mode."""
        return self._current_mode

    @property
    def last_assessment(self) -> RiskAssessment | None:
        """Get last risk assessment."""
        return self._last_assessment

    def register_event_callback(self, callback: Callable[[InterventionEvent], None]) -> None:
        """Register callback for intervention events."""
        self._event_callbacks.append(callback)

    def register_intervention_callback(self, callback: Callable[[Intervention], None]) -> None:
        """Register callback for new interventions."""
        self._intervention_callbacks.append(callback)

    async def start(self) -> None:
        """Start the intervention engine."""
        self._running = True
        logger.info("Intervention engine started")

    async def stop(self) -> None:
        """Stop the intervention engine."""
        self._running = False
        logger.info("Intervention engine stopped")

    async def evaluate(self, context: dict[str, Any]) -> Intervention | None:
        """
        Evaluate context and potentially generate an intervention.

        Args:
            context: Current situational context from sensor fusion

        Returns:
            Intervention if one should be delivered, None otherwise
        """
        # Assess risk
        assessment = self._risk_engine.assess(context)
        self._last_assessment = assessment

        # Check for mode transition
        await self._check_mode_transition(assessment)

        # Get current handler
        handler = self._handlers[self._current_mode]

        # Evaluate for intervention
        intervention = await handler.evaluate(context, assessment)

        if intervention:
            self._queue_intervention(intervention)
            self._emit_event(InterventionEventType.INTERVENTION_CREATED, intervention)

        # Return next pending intervention
        return self._get_next_intervention()

    async def _check_mode_transition(self, assessment: RiskAssessment) -> None:
        """Check if mode should transition based on assessment."""
        recommended = assessment.recommended_mode
        current = self._current_mode

        if recommended == current:
            # Clear any pending transitions
            self._pending_escalation = None
            self._pending_deescalation = None
            self._escalation_timer = None
            return

        # Determine if escalating or de-escalating
        is_escalation = recommended.get_priority() > current.get_priority()

        if is_escalation:
            await self._handle_escalation(recommended, assessment)
        else:
            await self._handle_deescalation(recommended, assessment)

    async def _handle_escalation(
        self,
        target_mode: InterventionMode,
        assessment: RiskAssessment,
    ) -> None:
        """Handle potential mode escalation."""
        # Crisis mode escalates immediately
        if target_mode == InterventionMode.CRISIS:
            await self._transition_to(target_mode, "Critical risk detected", assessment)
            return

        # Other escalations use delay
        if self._pending_escalation != target_mode:
            self._pending_escalation = target_mode
            self._escalation_timer = datetime.now()
            return

        # Check if delay has passed
        if self._escalation_timer:
            elapsed = (datetime.now() - self._escalation_timer).total_seconds()
            if elapsed >= self.config.mode_escalation_delay:
                await self._transition_to(target_mode, "Sustained elevated risk", assessment)
                self._pending_escalation = None
                self._escalation_timer = None

    async def _handle_deescalation(
        self,
        target_mode: InterventionMode,
        assessment: RiskAssessment,
    ) -> None:
        """Handle potential mode de-escalation."""
        # Don't de-escalate from crisis too quickly
        if self._current_mode == InterventionMode.CRISIS:
            # Require longer stability before leaving crisis
            required_delay = self.config.mode_deescalation_delay * 2
        else:
            required_delay = self.config.mode_deescalation_delay

        if self._pending_deescalation != target_mode:
            self._pending_deescalation = target_mode
            self._escalation_timer = datetime.now()
            return

        # Check if delay has passed
        if self._escalation_timer:
            elapsed = (datetime.now() - self._escalation_timer).total_seconds()
            if elapsed >= required_delay:
                await self._transition_to(target_mode, "Risk level reduced", assessment)
                self._pending_deescalation = None
                self._escalation_timer = None

    async def _transition_to(
        self,
        new_mode: InterventionMode,
        reason: str,
        assessment: RiskAssessment | None = None,
    ) -> None:
        """Execute mode transition."""
        old_mode = self._current_mode

        # Deactivate old handler
        self._handlers[old_mode].deactivate()

        # Update mode
        self._current_mode = new_mode
        self._mode_start_time = datetime.now()

        # Activate new handler
        self._handlers[new_mode].activate()

        # Record transition
        transition = ModeTransition(
            from_mode=old_mode,
            to_mode=new_mode,
            reason=reason,
            risk_assessment=assessment,
        )
        self._mode_history.append(transition)

        # Emit event
        self._emit_event(InterventionEventType.MODE_CHANGED, {
            "from_mode": old_mode,
            "to_mode": new_mode,
            "reason": reason,
        })

        logger.info(f"Mode transition: {old_mode.value} -> {new_mode.value} ({reason})")

    def _queue_intervention(self, intervention: Intervention) -> None:
        """Add intervention to queue."""
        self._intervention_queue.add(intervention)

    def _get_next_intervention(self) -> Intervention | None:
        """Get next intervention to deliver."""
        intervention = self._intervention_queue.get_next()

        if intervention:
            self._delivered_interventions[intervention.id] = intervention
            self._intervention_history.append(intervention)

            # Notify callbacks
            for callback in self._intervention_callbacks:
                try:
                    callback(intervention)
                except Exception as e:
                    logger.error(f"Intervention callback error: {e}")

            self._emit_event(InterventionEventType.INTERVENTION_DELIVERED, intervention)

        return intervention

    async def handle_response(
        self,
        intervention_id: str,
        response: InterventionResponse,
    ) -> Intervention | None:
        """
        Handle user response to an intervention.

        Args:
            intervention_id: ID of the intervention being responded to
            response: User's response

        Returns:
            Follow-up intervention if needed
        """
        intervention = self._delivered_interventions.get(intervention_id)

        if not intervention:
            logger.warning(f"Response for unknown intervention: {intervention_id}")
            return None

        # Emit acknowledgment event
        if response.acknowledged:
            self._emit_event(InterventionEventType.INTERVENTION_ACKNOWLEDGED, {
                "intervention": intervention,
                "response": response,
            })
        else:
            self._emit_event(InterventionEventType.INTERVENTION_DISMISSED, {
                "intervention": intervention,
                "response": response,
            })

            # Track dismissal for false positive mitigation
            if intervention.metadata.get("factor"):
                self._risk_engine.record_dismissal(intervention.metadata["factor"])

        # Get handler for the intervention's mode
        handler = self._handlers[intervention.mode]

        # Let handler process response
        follow_up = await handler.handle_response(intervention, response)

        # Check for escalation signal
        if follow_up and follow_up.metadata.get("escalate_to_crisis"):
            await self._transition_to(
                InterventionMode.CRISIS,
                "User requested emergency assistance",
                self._last_assessment,
            )

        if follow_up:
            self._queue_intervention(follow_up)
            return self._get_next_intervention()

        return None

    def set_addressed(self, addressed: bool = True) -> None:
        """Signal that the user has addressed the system (for ambient mode)."""
        ambient_handler = self._handlers[InterventionMode.AMBIENT]
        if isinstance(ambient_handler, AmbientModeHandler):
            ambient_handler.set_addressed(addressed)

    def force_mode(self, mode: InterventionMode, reason: str = "Manual override") -> None:
        """Force transition to a specific mode."""
        self._safe_create_task(self._transition_to(mode, reason), name="force_mode_transition")

    def _emit_event(self, event_type: InterventionEventType, data: Any) -> None:
        """Emit an intervention event."""
        event = InterventionEvent(event_type=event_type, data=data)

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_mode_duration(self) -> float:
        """Get duration in current mode (seconds)."""
        return (datetime.now() - self._mode_start_time).total_seconds()

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "current_mode": self._current_mode.value,
            "mode_duration_seconds": self.get_mode_duration(),
            "pending_interventions": self._intervention_queue.pending_count,
            "total_interventions": len(self._intervention_history),
            "mode_transitions": len(self._mode_history),
            "last_assessment": self._last_assessment.to_dict() if self._last_assessment else None,
            "risk_engine_stats": self._risk_engine.get_stats(),
        }

    def get_recent_interventions(self, count: int = 10) -> list[Intervention]:
        """Get recent interventions."""
        return list(self._intervention_history)[-count:]

    def get_mode_history(self, count: int = 10) -> list[ModeTransition]:
        """Get recent mode transitions."""
        return list(self._mode_history)[-count:]

    async def __aenter__(self) -> "InterventionEngine":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Factory Function
# ============================================================================


def create_intervention_engine(
    config: InterventionConfig | None = None,
) -> InterventionEngine:
    """
    Create and configure an intervention engine.

    Args:
        config: Optional configuration

    Returns:
        Configured InterventionEngine
    """
    return InterventionEngine(config)
