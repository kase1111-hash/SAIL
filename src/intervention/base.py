"""
Base types and abstractions for the intervention framework.

Provides intervention modes, risk levels, and response types
for proportional situational guidance.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Intervention Mode Enumeration
# ============================================================================


class InterventionMode(str, Enum):
    """
    Intervention modes representing different levels of system engagement.

    AMBIENT: Passive listening; responds only when directly addressed
    ADVISORY: Gentle unprompted suggestions for detected risks
    GUARDIAN: Active alerts for high-risk behavioral patterns
    CRISIS: Calm, hands-free walkthrough for acute situations
    """

    AMBIENT = "ambient"
    ADVISORY = "advisory"
    GUARDIAN = "guardian"
    CRISIS = "crisis"

    def get_priority(self) -> int:
        """Get mode priority (higher = more urgent)."""
        priorities = {
            InterventionMode.AMBIENT: 0,
            InterventionMode.ADVISORY: 1,
            InterventionMode.GUARDIAN: 2,
            InterventionMode.CRISIS: 3,
        }
        return priorities[self]


# ============================================================================
# Risk Levels
# ============================================================================


class RiskLevel(str, Enum):
    """Risk levels for situation assessment."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def get_score(self) -> int:
        """Get numeric score for risk level."""
        scores = {
            RiskLevel.NONE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return scores[self]

    @classmethod
    def from_score(cls, score: int) -> "RiskLevel":
        """Get risk level from numeric score."""
        if score <= 0:
            return cls.NONE
        elif score == 1:
            return cls.LOW
        elif score == 2:
            return cls.MODERATE
        elif score == 3:
            return cls.HIGH
        else:
            return cls.CRITICAL


# ============================================================================
# Intervention Types
# ============================================================================


class InterventionType(str, Enum):
    """Types of interventions the system can make."""

    # Informational
    INFO = "info"
    REMINDER = "reminder"
    TIP = "tip"

    # Advisory
    SUGGESTION = "suggestion"
    RECOMMENDATION = "recommendation"
    CAUTION = "caution"

    # Guardian
    WARNING = "warning"
    ALERT = "alert"
    VERIFICATION = "verification"

    # Crisis
    GUIDANCE = "guidance"
    WALKTHROUGH = "walkthrough"
    EMERGENCY = "emergency"


class InterventionPriority(str, Enum):
    """Priority levels for interventions."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

    def get_value(self) -> int:
        """Get numeric value for priority."""
        values = {
            InterventionPriority.LOW: 1,
            InterventionPriority.NORMAL: 2,
            InterventionPriority.HIGH: 3,
            InterventionPriority.URGENT: 4,
            InterventionPriority.CRITICAL: 5,
        }
        return values[self]


# ============================================================================
# Risk Factor Categories
# ============================================================================


class RiskCategory(str, Enum):
    """Categories of risk factors."""

    LOCATION = "location"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    PHYSIOLOGICAL = "physiological"
    SOCIAL = "social"
    FINANCIAL = "financial"
    LEGAL = "legal"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RiskFactor:
    """A single risk factor contributing to overall risk assessment."""

    category: RiskCategory
    name: str
    description: str
    weight: float  # 0.0 to 1.0
    score: float  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def weighted_score(self) -> float:
        """Get confidence-weighted score."""
        return self.score * self.weight * self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "score": self.score,
            "confidence": self.confidence,
            "weighted_score": self.weighted_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment from multiple factors."""

    level: RiskLevel
    score: float  # 0.0 to 1.0
    factors: list[RiskFactor]
    recommended_mode: InterventionMode
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "score": self.score,
            "factors": [f.to_dict() for f in self.factors],
            "recommended_mode": self.recommended_mode.value,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Intervention:
    """An intervention action to be taken."""

    id: str
    type: InterventionType
    priority: InterventionPriority
    mode: InterventionMode
    message: str
    details: str | None = None
    actions: list[str] = field(default_factory=list)
    requires_response: bool = False
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if intervention has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "mode": self.mode.value,
            "message": self.message,
            "details": self.details,
            "actions": self.actions,
            "requires_response": self.requires_response,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class InterventionResponse:
    """User's response to an intervention."""

    intervention_id: str
    acknowledged: bool
    response_type: str  # "accepted", "dismissed", "deferred", "custom"
    response_text: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intervention_id": self.intervention_id,
            "acknowledged": self.acknowledged,
            "response_type": self.response_type,
            "response_text": self.response_text,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModeTransition:
    """Record of a mode transition."""

    from_mode: InterventionMode
    to_mode: InterventionMode
    reason: str
    risk_assessment: RiskAssessment | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "reason": reason,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class InterventionConfig:
    """Configuration for the intervention framework."""

    # Mode thresholds (risk score)
    ambient_threshold: float = 0.0
    advisory_threshold: float = 0.25
    guardian_threshold: float = 0.5
    crisis_threshold: float = 0.75

    # Cooldown settings (seconds)
    advisory_cooldown: float = 300.0  # 5 minutes
    guardian_cooldown: float = 60.0   # 1 minute
    crisis_cooldown: float = 0.0      # No cooldown for crisis

    # Mode transition settings
    mode_escalation_delay: float = 5.0  # Seconds before escalating
    mode_deescalation_delay: float = 30.0  # Seconds before de-escalating

    # Intervention settings
    max_pending_interventions: int = 10
    intervention_timeout: float = 300.0  # 5 minutes default

    # Risk assessment settings
    risk_history_window: float = 3600.0  # 1 hour
    false_positive_threshold: int = 3  # Dismissals before suppressing


# ============================================================================
# Abstract Mode Handler
# ============================================================================


class ModeHandler(ABC):
    """Abstract base class for intervention mode handlers."""

    def __init__(self, mode: InterventionMode, config: InterventionConfig | None = None):
        self.mode = mode
        self.config = config or InterventionConfig()
        self._active = False
        self._last_intervention_time: datetime | None = None
        self._intervention_count = 0

    @property
    def is_active(self) -> bool:
        """Check if mode is active."""
        return self._active

    def activate(self) -> None:
        """Activate this mode."""
        self._active = True
        logger.info(f"Mode {self.mode.value} activated")

    def deactivate(self) -> None:
        """Deactivate this mode."""
        self._active = False
        logger.info(f"Mode {self.mode.value} deactivated")

    def can_intervene(self) -> bool:
        """Check if intervention is allowed (cooldown check)."""
        if self._last_intervention_time is None:
            return True

        cooldowns = {
            InterventionMode.AMBIENT: float("inf"),  # No proactive interventions
            InterventionMode.ADVISORY: self.config.advisory_cooldown,
            InterventionMode.GUARDIAN: self.config.guardian_cooldown,
            InterventionMode.CRISIS: self.config.crisis_cooldown,
        }

        elapsed = (datetime.now() - self._last_intervention_time).total_seconds()
        return elapsed >= cooldowns.get(self.mode, 0)

    def record_intervention(self) -> None:
        """Record that an intervention was made."""
        self._last_intervention_time = datetime.now()
        self._intervention_count += 1

    @abstractmethod
    async def evaluate(
        self,
        context: dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> Intervention | None:
        """
        Evaluate context and determine if intervention is needed.

        Args:
            context: Current situational context
            risk_assessment: Current risk assessment

        Returns:
            Intervention if one should be made, None otherwise
        """
        pass

    @abstractmethod
    async def handle_response(
        self,
        intervention: Intervention,
        response: InterventionResponse,
    ) -> Intervention | None:
        """
        Handle user response to an intervention.

        Args:
            intervention: The intervention that was responded to
            response: User's response

        Returns:
            Follow-up intervention if needed, None otherwise
        """
        pass


# ============================================================================
# Events
# ============================================================================


class InterventionEventType(str, Enum):
    """Types of intervention events."""

    MODE_CHANGED = "mode_changed"
    INTERVENTION_CREATED = "intervention_created"
    INTERVENTION_DELIVERED = "intervention_delivered"
    INTERVENTION_ACKNOWLEDGED = "intervention_acknowledged"
    INTERVENTION_DISMISSED = "intervention_dismissed"
    INTERVENTION_EXPIRED = "intervention_expired"
    RISK_ELEVATED = "risk_elevated"
    RISK_REDUCED = "risk_reduced"


@dataclass
class InterventionEvent:
    """An event from the intervention system."""

    event_type: InterventionEventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data_dict = self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
        return {
            "event_type": self.event_type.value,
            "data": data_dict,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
