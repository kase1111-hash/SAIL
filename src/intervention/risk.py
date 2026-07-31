"""
Risk assessment engine for the intervention framework.

Provides multi-factor risk scoring, history tracking, and
proportional mode recommendations.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base import (
    InterventionConfig,
    InterventionMode,
    RiskAssessment,
    RiskCategory,
    RiskFactor,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# Speed above which driving is treated as a risk factor. Deliberately duplicated
# rather than imported from src.sensors.fusion: this module takes a plain dict
# context so the intervention layer stays independent of the sensor types.
# test_risk_speed_threshold_matches_fusion keeps the two values in step.
HIGH_SPEED_KMH = 120.0


# ============================================================================
# Risk Factor Definitions
# ============================================================================


# Default risk factor weights by category
DEFAULT_CATEGORY_WEIGHTS: dict[RiskCategory, float] = {
    RiskCategory.LOCATION: 0.8,
    RiskCategory.TEMPORAL: 0.6,
    RiskCategory.BEHAVIORAL: 0.9,
    RiskCategory.ENVIRONMENTAL: 0.7,
    RiskCategory.PHYSIOLOGICAL: 0.85,
    RiskCategory.SOCIAL: 0.5,
    RiskCategory.FINANCIAL: 0.7,
    RiskCategory.LEGAL: 0.75,
}


# Known risk patterns with their base scores
RISK_PATTERNS: dict[str, dict[str, Any]] = {
    # Location risks
    "unfamiliar_area": {
        "category": RiskCategory.LOCATION,
        "weight": 0.6,
        "description": "User is in an unfamiliar location",
    },
    "high_crime_area": {
        "category": RiskCategory.LOCATION,
        "weight": 0.8,
        "description": "Location has elevated crime statistics",
    },
    "isolated_location": {
        "category": RiskCategory.LOCATION,
        "weight": 0.7,
        "description": "User is in an isolated area with limited help access",
    },

    # Temporal risks
    "late_night": {
        "category": RiskCategory.TEMPORAL,
        "weight": 0.5,
        "description": "It is late at night (midnight - 5 AM)",
    },
    "late_night_travel": {
        "category": RiskCategory.TEMPORAL,
        "weight": 0.7,
        "description": "Traveling during late night hours",
    },
    "unusual_hours": {
        "category": RiskCategory.TEMPORAL,
        "weight": 0.4,
        "description": "Activity during unusual hours for user",
    },

    # Behavioral risks
    "erratic_driving": {
        "category": RiskCategory.BEHAVIORAL,
        "weight": 0.9,
        "description": "Driving patterns indicate distraction or impairment",
    },
    "high_speed_driving": {
        "category": RiskCategory.BEHAVIORAL,
        "weight": 0.7,
        "description": "Driving well above typical highway speeds",
    },
    "sudden_route_change": {
        "category": RiskCategory.BEHAVIORAL,
        "weight": 0.5,
        "description": "Unexpected deviation from planned route",
    },
    "extended_stop": {
        "category": RiskCategory.BEHAVIORAL,
        "weight": 0.4,
        "description": "Unexpectedly stopped for extended period",
    },

    # Environmental risks
    "loud_environment": {
        "category": RiskCategory.ENVIRONMENTAL,
        "weight": 0.3,
        "description": "Environment is very loud",
    },
    "threat_sounds": {
        "category": RiskCategory.ENVIRONMENTAL,
        "weight": 0.9,
        "description": "Potential threat sounds detected (screaming, crashes)",
    },
    "confrontation_detected": {
        "category": RiskCategory.ENVIRONMENTAL,
        "weight": 0.85,
        "description": "Signs of verbal confrontation detected",
    },

    # Physiological risks
    "elevated_stress": {
        "category": RiskCategory.PHYSIOLOGICAL,
        "weight": 0.6,
        "description": "Voice stress indicators detected",
    },
    "high_stress": {
        "category": RiskCategory.PHYSIOLOGICAL,
        "weight": 0.85,
        "description": "High stress level detected in voice",
    },
    "rapid_heartrate": {
        "category": RiskCategory.PHYSIOLOGICAL,
        "weight": 0.7,
        "description": "Heart rate significantly elevated",
    },

    # Social risks
    "meeting_stranger": {
        "category": RiskCategory.SOCIAL,
        "weight": 0.5,
        "description": "Meeting someone for the first time",
    },
    "pressure_detected": {
        "category": RiskCategory.SOCIAL,
        "weight": 0.7,
        "description": "Social pressure tactics detected in conversation",
    },

    # Financial risks
    "large_transaction": {
        "category": RiskCategory.FINANCIAL,
        "weight": 0.6,
        "description": "Large financial transaction being discussed",
    },
    "wire_transfer": {
        "category": RiskCategory.FINANCIAL,
        "weight": 0.8,
        "description": "Wire transfer being initiated",
    },
    "urgency_pressure": {
        "category": RiskCategory.FINANCIAL,
        "weight": 0.75,
        "description": "Urgency language in financial context",
    },

    # Legal risks
    "traffic_stop": {
        "category": RiskCategory.LEGAL,
        "weight": 0.6,
        "description": "Potential traffic stop situation",
    },
    "authority_interaction": {
        "category": RiskCategory.LEGAL,
        "weight": 0.5,
        "description": "Interaction with authority figures",
    },
}


# ============================================================================
# Risk History Entry
# ============================================================================


@dataclass
class RiskHistoryEntry:
    """Entry in risk history."""

    assessment: RiskAssessment
    context_snapshot: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Risk Assessment Engine
# ============================================================================


class RiskAssessmentEngine:
    """
    Engine for assessing situational risk from multiple factors.

    Combines risk factors from location, temporal, behavioral, environmental,
    and other sources into a unified risk score with mode recommendations.
    """

    def __init__(self, config: InterventionConfig | None = None):
        self.config = config or InterventionConfig()

        # Category weights (can be customized)
        self._category_weights = DEFAULT_CATEGORY_WEIGHTS.copy()

        # Risk history for trend analysis
        self._history: deque[RiskHistoryEntry] = deque(maxlen=100)

        # False positive tracking
        self._dismissed_patterns: dict[str, int] = {}
        self._suppressed_patterns: set[str] = set()

    def set_category_weight(self, category: RiskCategory, weight: float) -> None:
        """Set weight for a risk category."""
        self._category_weights[category] = max(0.0, min(1.0, weight))

    def assess(
        self,
        context: dict[str, Any],
        detected_factors: list[RiskFactor] | None = None,
    ) -> RiskAssessment:
        """
        Assess risk from context and detected factors.

        Args:
            context: Current situational context (from sensor fusion)
            detected_factors: Pre-detected risk factors (optional)

        Returns:
            Complete risk assessment
        """
        factors: list[RiskFactor] = []

        # Use provided factors or detect from context
        if detected_factors:
            factors.extend(detected_factors)
        else:
            factors.extend(self._detect_factors(context))

        # Filter suppressed patterns
        factors = [
            f for f in factors
            if f.name not in self._suppressed_patterns
        ]

        # Calculate overall risk score
        if factors:
            total_weighted = sum(f.weighted_score for f in factors)
            total_weight = sum(f.weight * f.confidence for f in factors)
            score = total_weighted / total_weight if total_weight > 0 else 0.0
        else:
            score = 0.0

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Determine risk level
        level = self._score_to_level(score)

        # Determine recommended mode
        recommended_mode = self._determine_mode(score)

        # Generate summary
        summary = self._generate_summary(level, factors)

        assessment = RiskAssessment(
            level=level,
            score=score,
            factors=factors,
            recommended_mode=recommended_mode,
            summary=summary,
        )

        # Record in history
        self._history.append(RiskHistoryEntry(
            assessment=assessment,
            context_snapshot=context.copy() if context else {},
        ))

        return assessment

    def _detect_factors(self, context: dict[str, Any]) -> list[RiskFactor]:
        """Detect risk factors from context."""
        factors: list[RiskFactor] = []

        if not context:
            return factors

        # Location-based factors
        location_context = context.get("location_context", "")
        if location_context == "unfamiliar":
            factors.append(self._create_factor("unfamiliar_area", 0.7))

        # Temporal factors
        is_late_night = context.get("is_late_night", False)
        if is_late_night:
            factors.append(self._create_factor("late_night", 0.6))

            # Late night + unfamiliar = elevated
            if location_context == "unfamiliar":
                factors.append(self._create_factor("late_night_travel", 0.8))

        # Motion-based factors
        motion_state = context.get("motion_state", "")
        speed_mps = context.get("speed_mps")

        if motion_state == "driving" and is_late_night:
            factors.append(self._create_factor("late_night_travel", 0.7))

        # Speed threshold matches SensorFusion._calculate_risk, so the two
        # risk paths agree on what counts as high speed.
        if motion_state == "driving" and speed_mps:
            speed_kmh = speed_mps * 3.6
            if speed_kmh > HIGH_SPEED_KMH:
                factors.append(self._create_factor("high_speed_driving", 0.8, {
                    "speed_kmh": round(speed_kmh, 1),
                }))

        # Environmental factors
        threat_cues = context.get("threat_cues", [])
        if threat_cues:
            factors.append(self._create_factor("threat_sounds", 0.9, {
                "detected_cues": threat_cues,
            }))

        audio_environment = context.get("audio_environment", "")
        if audio_environment == "loud":
            factors.append(self._create_factor("loud_environment", 0.4))

        # Stress factors
        stress_level = context.get("stress_level", "")
        if stress_level == "elevated":
            factors.append(self._create_factor("elevated_stress", 0.6))
        elif stress_level == "high":
            factors.append(self._create_factor("high_stress", 0.85))

        return factors

    def _create_factor(
        self,
        pattern_name: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> RiskFactor:
        """Create a risk factor from a known pattern."""
        pattern = RISK_PATTERNS.get(pattern_name, {})

        return RiskFactor(
            category=pattern.get("category", RiskCategory.BEHAVIORAL),
            name=pattern_name,
            description=pattern.get("description", pattern_name),
            weight=pattern.get("weight", 0.5),
            score=score,
            metadata=metadata or {},
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score < 0.1:
            return RiskLevel.NONE
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.MODERATE
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _determine_mode(self, score: float) -> InterventionMode:
        """Determine recommended intervention mode from score."""
        if score >= self.config.crisis_threshold:
            return InterventionMode.CRISIS
        elif score >= self.config.guardian_threshold:
            return InterventionMode.GUARDIAN
        elif score >= self.config.advisory_threshold:
            return InterventionMode.ADVISORY
        else:
            return InterventionMode.AMBIENT

    def _generate_summary(self, level: RiskLevel, factors: list[RiskFactor]) -> str:
        """Generate human-readable summary of risk assessment."""
        if not factors:
            return "No significant risk factors detected."

        level_descriptions = {
            RiskLevel.NONE: "Situation appears normal",
            RiskLevel.LOW: "Minor risk factors present",
            RiskLevel.MODERATE: "Moderate risk factors detected",
            RiskLevel.HIGH: "Elevated risk situation",
            RiskLevel.CRITICAL: "Critical risk situation requiring immediate attention",
        }

        summary = level_descriptions.get(level, "Risk assessment complete")

        # Add top factors
        top_factors = sorted(factors, key=lambda f: f.weighted_score, reverse=True)[:3]
        if top_factors:
            factor_names = [f.description for f in top_factors]
            summary += ". Key factors: " + "; ".join(factor_names)

        return summary

    def record_dismissal(self, pattern_name: str) -> None:
        """Record that a pattern was dismissed by user (false positive tracking)."""
        self._dismissed_patterns[pattern_name] = (
            self._dismissed_patterns.get(pattern_name, 0) + 1
        )

        # Suppress pattern if dismissed too many times
        if self._dismissed_patterns[pattern_name] >= self.config.false_positive_threshold:
            self._suppressed_patterns.add(pattern_name)
            logger.info(f"Pattern '{pattern_name}' suppressed due to repeated dismissals")

    def reset_suppression(self, pattern_name: str) -> None:
        """Reset suppression for a pattern."""
        self._suppressed_patterns.discard(pattern_name)
        self._dismissed_patterns.pop(pattern_name, None)

    def get_risk_trend(self, window_minutes: int = 30) -> str:
        """
        Analyze risk trend over recent history.

        Returns: "increasing", "stable", "decreasing"
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [e for e in self._history if e.timestamp > cutoff]

        if len(recent) < 2:
            return "stable"

        # Compare first half to second half
        mid = len(recent) // 2
        first_half_avg = sum(e.assessment.score for e in recent[:mid]) / mid
        second_half_avg = sum(e.assessment.score for e in recent[mid:]) / (len(recent) - mid)

        diff = second_half_avg - first_half_avg

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"

    def get_recent_assessments(self, count: int = 10) -> list[RiskAssessment]:
        """Get recent risk assessments."""
        return [e.assessment for e in list(self._history)[-count:]]

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        recent = list(self._history)[-20:] if self._history else []

        return {
            "history_count": len(self._history),
            "suppressed_patterns": list(self._suppressed_patterns),
            "dismissed_counts": self._dismissed_patterns.copy(),
            "average_recent_score": (
                sum(e.assessment.score for e in recent) / len(recent)
                if recent else 0.0
            ),
            "risk_trend": self.get_risk_trend(),
        }


# ============================================================================
# Risk Pattern Detector
# ============================================================================


class RiskPatternDetector:
    """Detects specific risk patterns from behavioral data."""

    def __init__(self):
        self._behavior_history: deque[dict[str, Any]] = deque(maxlen=100)

    def add_observation(self, observation: dict[str, Any]) -> None:
        """Add a behavioral observation."""
        observation["timestamp"] = datetime.now()
        self._behavior_history.append(observation)

    def detect_erratic_driving(self) -> RiskFactor | None:
        """Detect erratic driving patterns."""
        # Look for rapid speed changes, unusual stopping patterns, etc.
        recent = [
            o for o in self._behavior_history
            if o.get("type") == "motion"
            and (datetime.now() - o["timestamp"]).total_seconds() < 300
        ]

        if len(recent) < 5:
            return None

        speeds = [o.get("speed_mps", 0) for o in recent if o.get("speed_mps") is not None]
        if len(speeds) < 3:
            return None

        # Calculate speed variance
        avg_speed = sum(speeds) / len(speeds)
        variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)

        # High variance while driving indicates erratic behavior
        if variance > 25 and avg_speed > 5:  # High variance at driving speeds
            return RiskFactor(
                category=RiskCategory.BEHAVIORAL,
                name="erratic_driving",
                description="Erratic driving patterns detected",
                weight=0.9,
                score=min(1.0, variance / 50),
                metadata={"variance": variance, "avg_speed": avg_speed},
            )

        return None

    def detect_pressure_tactics(self, conversation_text: str) -> RiskFactor | None:
        """Detect pressure tactics in conversation."""
        pressure_phrases = [
            "act now",
            "limited time",
            "don't tell anyone",
            "keep this between us",
            "urgent",
            "immediately",
            "right now",
            "don't hang up",
            "stay on the line",
            "wire transfer",
            "gift cards",
            "don't call the police",
        ]

        text_lower = conversation_text.lower()
        detected = [p for p in pressure_phrases if p in text_lower]

        if detected:
            score = min(1.0, len(detected) * 0.25)
            return RiskFactor(
                category=RiskCategory.SOCIAL,
                name="pressure_detected",
                description="Pressure tactics detected in conversation",
                weight=0.7,
                score=score,
                metadata={"detected_phrases": detected},
            )

        return None

    def detect_financial_risk(self, conversation_text: str) -> RiskFactor | None:
        """Detect financial risk indicators."""
        financial_phrases = [
            "wire",
            "transfer",
            "bitcoin",
            "gift card",
            "western union",
            "moneygram",
            "social security",
            "bank account",
            "routing number",
            "pin number",
        ]

        text_lower = conversation_text.lower()
        detected = [p for p in financial_phrases if p in text_lower]

        if detected:
            score = min(1.0, len(detected) * 0.3)
            return RiskFactor(
                category=RiskCategory.FINANCIAL,
                name="large_transaction",
                description="Financial transaction indicators detected",
                weight=0.7,
                score=score,
                metadata={"detected_phrases": detected},
            )

        return None
