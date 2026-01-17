"""
Intervention mode handlers.

Implements the four intervention modes:
- Ambient: Passive listening, addressed-only responses
- Advisory: Gentle suggestions for detected risks
- Guardian: Active alerts for high-risk patterns
- Crisis: Hands-free walkthrough for acute situations
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from .base import (
    Intervention,
    InterventionConfig,
    InterventionMode,
    InterventionPriority,
    InterventionResponse,
    InterventionType,
    ModeHandler,
    RiskAssessment,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def generate_intervention_id() -> str:
    """Generate unique intervention ID."""
    return f"int_{uuid.uuid4().hex[:12]}"


# ============================================================================
# Ambient Mode Handler
# ============================================================================


class AmbientModeHandler(ModeHandler):
    """
    Ambient mode: Passive listening state.

    - Responds only when directly addressed
    - Minimal proactive monitoring
    - Gathers context silently
    """

    def __init__(self, config: InterventionConfig | None = None):
        super().__init__(InterventionMode.AMBIENT, config)
        self._addressed = False

    def set_addressed(self, addressed: bool) -> None:
        """Set whether the user has addressed the system."""
        self._addressed = addressed

    async def evaluate(
        self,
        context: dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> Intervention | None:
        """
        In ambient mode, only respond if directly addressed.
        Does not proactively intervene regardless of risk.
        """
        if not self._active:
            return None

        # Ambient mode only responds when addressed
        if not self._addressed:
            return None

        # Reset addressed flag after responding
        self._addressed = False

        # Even when addressed, just provide info, not warnings
        return Intervention(
            id=generate_intervention_id(),
            type=InterventionType.INFO,
            priority=InterventionPriority.NORMAL,
            mode=self.mode,
            message="I'm here and listening. How can I help?",
            requires_response=False,
        )

    async def handle_response(
        self,
        intervention: Intervention,
        response: InterventionResponse,
    ) -> Intervention | None:
        """Handle user response in ambient mode."""
        # In ambient mode, responses are simple acknowledgments
        return None


# ============================================================================
# Advisory Mode Handler
# ============================================================================


class AdvisoryModeHandler(ModeHandler):
    """
    Advisory mode: Gentle unprompted suggestions.

    - Detects low-to-moderate risk situations
    - Provides non-intrusive suggestions
    - Respects cooldown periods
    """

    def __init__(self, config: InterventionConfig | None = None):
        super().__init__(InterventionMode.ADVISORY, config)
        self._recent_suggestions: dict[str, datetime] = {}
        self._suggestion_cooldown = timedelta(minutes=10)

    async def evaluate(
        self,
        context: dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> Intervention | None:
        """
        Evaluate context and provide gentle suggestions for detected risks.
        """
        if not self._active:
            return None

        # Check if we can intervene (cooldown)
        if not self.can_intervene():
            return None

        # Only act on low-to-moderate risks in advisory mode
        if risk_assessment.level not in (RiskLevel.LOW, RiskLevel.MODERATE):
            return None

        # Generate suggestion based on top risk factors
        suggestion = self._generate_suggestion(risk_assessment, context)

        if suggestion:
            self.record_intervention()
            return suggestion

        return None

    def _generate_suggestion(
        self,
        risk_assessment: RiskAssessment,
        context: dict[str, Any],
    ) -> Intervention | None:
        """Generate a gentle suggestion based on risk factors."""
        if not risk_assessment.factors:
            return None

        # Get the primary risk factor
        primary_factor = max(risk_assessment.factors, key=lambda f: f.weighted_score)

        # Check suggestion cooldown for this specific factor
        factor_key = primary_factor.name
        if factor_key in self._recent_suggestions:
            elapsed = datetime.now() - self._recent_suggestions[factor_key]
            if elapsed < self._suggestion_cooldown:
                return None

        # Generate appropriate suggestion
        message, details = self._get_suggestion_content(primary_factor, context)

        if not message:
            return None

        # Record this suggestion
        self._recent_suggestions[factor_key] = datetime.now()

        return Intervention(
            id=generate_intervention_id(),
            type=InterventionType.SUGGESTION,
            priority=InterventionPriority.LOW,
            mode=self.mode,
            message=message,
            details=details,
            requires_response=False,
            timeout_seconds=60.0,
            metadata={"factor": primary_factor.name},
        )

    def _get_suggestion_content(
        self,
        factor,
        context: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Get suggestion message and details for a risk factor."""
        suggestions = {
            "unfamiliar_area": (
                "You're in an unfamiliar area.",
                "Consider sharing your location with someone you trust, or noting nearby landmarks.",
            ),
            "late_night": (
                "It's getting late.",
                "Stay aware of your surroundings and consider letting someone know your plans.",
            ),
            "late_night_travel": (
                "Traveling late at night.",
                "Stay on well-lit main roads when possible. Keep your phone charged.",
            ),
            "elevated_stress": (
                "I notice you might be feeling stressed.",
                "Take a moment to breathe if you can. I'm here if you need anything.",
            ),
            "loud_environment": (
                "It's quite loud around you.",
                "You may want to move somewhere quieter if you need to focus.",
            ),
        }

        return suggestions.get(factor.name, (None, None))

    async def handle_response(
        self,
        intervention: Intervention,
        response: InterventionResponse,
    ) -> Intervention | None:
        """Handle user response to advisory suggestion."""
        if response.response_type == "dismissed":
            # User dismissed the suggestion - note for future
            factor_name = intervention.metadata.get("factor")
            if factor_name:
                # Extend cooldown for this factor
                self._recent_suggestions[factor_name] = datetime.now()

        return None


# ============================================================================
# Guardian Mode Handler
# ============================================================================


class GuardianModeHandler(ModeHandler):
    """
    Guardian mode: Active alerts for high-risk patterns.

    - Detects high-risk behavioral patterns
    - Provides verification prompts
    - More insistent but still respectful
    """

    def __init__(self, config: InterventionConfig | None = None):
        super().__init__(InterventionMode.GUARDIAN, config)
        self._pending_verifications: dict[str, Intervention] = {}
        self._alert_escalation_count: dict[str, int] = {}

    async def evaluate(
        self,
        context: dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> Intervention | None:
        """
        Evaluate context and provide alerts for high-risk situations.
        """
        if not self._active:
            return None

        # Check cooldown
        if not self.can_intervene():
            return None

        # Act on moderate-to-high risks
        if risk_assessment.level not in (RiskLevel.MODERATE, RiskLevel.HIGH):
            return None

        # Generate alert based on risk factors
        alert = self._generate_alert(risk_assessment, context)

        if alert:
            self.record_intervention()
            return alert

        return None

    def _generate_alert(
        self,
        risk_assessment: RiskAssessment,
        context: dict[str, Any],
    ) -> Intervention | None:
        """Generate an alert for high-risk factors."""
        if not risk_assessment.factors:
            return None

        # Get high-scoring factors
        high_factors = [f for f in risk_assessment.factors if f.weighted_score > 0.4]
        if not high_factors:
            return None

        primary_factor = max(high_factors, key=lambda f: f.weighted_score)

        # Generate appropriate alert
        message, details, actions = self._get_alert_content(primary_factor, context)

        if not message:
            return None

        # Determine if verification is needed
        requires_verification = primary_factor.name in [
            "threat_sounds",
            "high_stress",
            "pressure_detected",
            "wire_transfer",
        ]

        intervention_type = (
            InterventionType.VERIFICATION if requires_verification
            else InterventionType.WARNING
        )

        intervention = Intervention(
            id=generate_intervention_id(),
            type=intervention_type,
            priority=InterventionPriority.HIGH,
            mode=self.mode,
            message=message,
            details=details,
            actions=actions,
            requires_response=requires_verification,
            timeout_seconds=120.0,
            metadata={"factor": primary_factor.name, "risk_level": risk_assessment.level.value},
        )

        if requires_verification:
            self._pending_verifications[intervention.id] = intervention

        return intervention

    def _get_alert_content(
        self,
        factor,
        context: dict[str, Any],
    ) -> tuple[str | None, str | None, list[str]]:
        """Get alert message, details, and actions for a risk factor."""
        alerts = {
            "threat_sounds": (
                "I detected sounds that concern me. Are you okay?",
                "I heard what sounded like raised voices or distress.",
                ["I'm fine", "I need help", "Call emergency"],
            ),
            "high_stress": (
                "You sound very stressed. Is everything alright?",
                "Your voice indicates high stress levels. I'm here to help.",
                ["I'm okay", "I could use support", "It's an emergency"],
            ),
            "pressure_detected": (
                "I'm noticing pressure tactics in this conversation.",
                "Someone may be trying to pressure you into something. Take your time.",
                ["Thanks for the heads up", "Help me respond", "End this call"],
            ),
            "wire_transfer": (
                "Careful - wire transfers are often irreversible.",
                "Scammers frequently request wire transfers. Verify this is legitimate.",
                ["I verified it's safe", "Help me verify", "Stop the transfer"],
            ),
            "late_night_travel": (
                "You're traveling quite late. Just checking in.",
                "Late night travel can carry elevated risks. Stay alert.",
                ["I'm being careful", "Share my location", "Call someone"],
            ),
            "erratic_driving": (
                "Your driving pattern seems unusual. Everything okay?",
                "Consider pulling over safely if you need a moment.",
                ["I'm fine", "Finding a place to stop", "Need assistance"],
            ),
        }

        return alerts.get(factor.name, (None, None, []))

    async def handle_response(
        self,
        intervention: Intervention,
        response: InterventionResponse,
    ) -> Intervention | None:
        """Handle user response to guardian alert."""
        # Remove from pending if it was a verification
        self._pending_verifications.pop(intervention.id, None)

        # Handle specific responses
        if "emergency" in response.response_type.lower() or "help" in (response.response_text or "").lower():
            # Escalate to crisis mode signal
            return Intervention(
                id=generate_intervention_id(),
                type=InterventionType.ALERT,
                priority=InterventionPriority.URGENT,
                mode=self.mode,
                message="Understood. Let me help you right away.",
                details="Preparing crisis assistance.",
                requires_response=False,
                metadata={"escalate_to_crisis": True},
            )

        if response.response_type == "dismissed":
            # Track dismissals for pattern learning
            factor_name = intervention.metadata.get("factor")
            if factor_name:
                self._alert_escalation_count[factor_name] = 0

        return None


# ============================================================================
# Crisis Mode Handler
# ============================================================================


class CrisisModeHandler(ModeHandler):
    """
    Crisis mode: Calm, hands-free walkthrough for acute situations.

    - Provides step-by-step guidance
    - Hands-free operation
    - Can trigger emergency protocols
    - Locks context to prevent data loss
    """

    def __init__(self, config: InterventionConfig | None = None):
        super().__init__(InterventionMode.CRISIS, config)
        self._active_walkthrough: str | None = None
        self._walkthrough_step: int = 0
        self._emergency_contacts_notified: bool = False

    async def evaluate(
        self,
        context: dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> Intervention | None:
        """
        Provide crisis guidance for critical situations.
        """
        if not self._active:
            return None

        # Crisis mode activates on high/critical risk
        if risk_assessment.level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return None

        # If already in a walkthrough, continue it
        if self._active_walkthrough:
            return self._continue_walkthrough(context)

        # Start new crisis intervention
        return self._start_crisis_intervention(risk_assessment, context)

    def _start_crisis_intervention(
        self,
        risk_assessment: RiskAssessment,
        context: dict[str, Any],
    ) -> Intervention:
        """Start a new crisis intervention."""
        # Determine crisis type from factors
        crisis_type = self._determine_crisis_type(risk_assessment.factors)

        # Get initial guidance
        message, details, walkthrough_id = self._get_crisis_guidance(crisis_type, context)

        self._active_walkthrough = walkthrough_id
        self._walkthrough_step = 0

        return Intervention(
            id=generate_intervention_id(),
            type=InterventionType.GUIDANCE,
            priority=InterventionPriority.CRITICAL,
            mode=self.mode,
            message=message,
            details=details,
            requires_response=True,
            timeout_seconds=None,  # No timeout in crisis
            metadata={
                "crisis_type": crisis_type,
                "walkthrough_id": walkthrough_id,
                "step": 0,
            },
        )

    def _determine_crisis_type(self, factors) -> str:
        """Determine the type of crisis from risk factors."""
        factor_names = [f.name for f in factors]

        if "threat_sounds" in factor_names or "confrontation_detected" in factor_names:
            return "safety_threat"
        elif "high_stress" in factor_names:
            return "emotional_crisis"
        elif "erratic_driving" in factor_names:
            return "driving_emergency"
        elif "wire_transfer" in factor_names or "urgency_pressure" in factor_names:
            return "scam_in_progress"
        else:
            return "general_emergency"

    def _get_crisis_guidance(
        self,
        crisis_type: str,
        context: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Get crisis-specific guidance."""
        guidance = {
            "safety_threat": (
                "I'm here with you. First, let's make sure you're safe.",
                "If you can, move to a safe location. If you can't speak freely, just say 'red'.",
                "safety_walkthrough",
            ),
            "emotional_crisis": (
                "I hear that you're going through something difficult.",
                "Let's take this one step at a time. You're not alone. Can you take a slow breath with me?",
                "emotional_support",
            ),
            "driving_emergency": (
                "Let's get you safely stopped.",
                "If you can, signal and pull over to a safe spot. Turn on your hazard lights.",
                "driving_safety",
            ),
            "scam_in_progress": (
                "Stop. Don't send anything yet.",
                "This has signs of a scam. Let's verify before you proceed. Hang up if you're on a call.",
                "scam_prevention",
            ),
            "general_emergency": (
                "I'm here to help you through this.",
                "Tell me what's happening, or just let me know if you need me to call for help.",
                "general_crisis",
            ),
        }

        return guidance.get(crisis_type, guidance["general_emergency"])

    def _continue_walkthrough(self, context: dict[str, Any]) -> Intervention:
        """Continue an active walkthrough."""
        self._walkthrough_step += 1

        # Get walkthrough steps (simplified - would be more comprehensive in production)
        walkthroughs = {
            "safety_walkthrough": [
                ("Are you in a safe place now?", "If not, your safety is the priority."),
                ("Good. Take a moment. Is anyone with you who can help?", ""),
                ("Would you like me to notify someone you trust?", ""),
            ],
            "emotional_support": [
                ("How are you feeling right now?", "There's no wrong answer."),
                ("Thank you for sharing that. What would help most right now?", ""),
                ("I'm here as long as you need.", ""),
            ],
            "driving_safety": [
                ("Are you safely stopped?", "Keep your hazards on."),
                ("Good. Take a moment. Do you need roadside assistance?", ""),
                ("Would you like me to call someone for you?", ""),
            ],
            "scam_prevention": [
                ("Have you sent any money or shared any personal information?", ""),
                ("Let's verify this together. Who contacted you and what did they claim?", ""),
                ("This is a known scam pattern. I recommend ending contact with them.", ""),
            ],
        }

        steps = walkthroughs.get(self._active_walkthrough, [])

        if self._walkthrough_step >= len(steps):
            # Walkthrough complete
            self._active_walkthrough = None
            self._walkthrough_step = 0
            return Intervention(
                id=generate_intervention_id(),
                type=InterventionType.GUIDANCE,
                priority=InterventionPriority.HIGH,
                mode=self.mode,
                message="We've gone through the immediate steps. How are you doing now?",
                details="I'll stay in heightened awareness mode for a while.",
                requires_response=False,
                metadata={"walkthrough_complete": True},
            )

        message, details = steps[self._walkthrough_step]

        return Intervention(
            id=generate_intervention_id(),
            type=InterventionType.WALKTHROUGH,
            priority=InterventionPriority.CRITICAL,
            mode=self.mode,
            message=message,
            details=details,
            requires_response=True,
            metadata={
                "walkthrough_id": self._active_walkthrough,
                "step": self._walkthrough_step,
            },
        )

    async def handle_response(
        self,
        intervention: Intervention,
        response: InterventionResponse,
    ) -> Intervention | None:
        """Handle user response in crisis mode."""
        # Check for emergency keywords
        response_text = (response.response_text or "").lower()

        if any(word in response_text for word in ["help", "emergency", "911", "call"]):
            return Intervention(
                id=generate_intervention_id(),
                type=InterventionType.EMERGENCY,
                priority=InterventionPriority.CRITICAL,
                mode=self.mode,
                message="I understand. Getting help now.",
                details="Initiating emergency protocol.",
                requires_response=False,
                metadata={"emergency_protocol": True},
            )

        # Check for "red" safety word
        if "red" in response_text:
            return Intervention(
                id=generate_intervention_id(),
                type=InterventionType.EMERGENCY,
                priority=InterventionPriority.CRITICAL,
                mode=self.mode,
                message="Understood.",
                details="Silent emergency protocol activated.",
                requires_response=False,
                metadata={"silent_emergency": True},
            )

        # Continue walkthrough
        if self._active_walkthrough:
            return self._continue_walkthrough({})

        return None

    def reset_walkthrough(self) -> None:
        """Reset the current walkthrough."""
        self._active_walkthrough = None
        self._walkthrough_step = 0
