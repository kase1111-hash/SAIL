"""
Intervention Framework for SAIL.

Provides four intervention modes with proportional responses:
- Ambient: Passive listening, addressed-only responses
- Advisory: Gentle suggestions for detected risks
- Guardian: Active alerts for high-risk patterns
- Crisis: Hands-free walkthrough for acute situations
"""

from .base import (
    # Enums
    InterventionMode,
    RiskLevel,
    RiskCategory,
    InterventionType,
    InterventionPriority,
    InterventionEventType,
    # Data classes
    RiskFactor,
    RiskAssessment,
    Intervention,
    InterventionResponse,
    ModeTransition,
    InterventionEvent,
    InterventionConfig,
    # Abstract base
    ModeHandler,
)

from .risk import (
    RiskAssessmentEngine,
    RiskPatternDetector,
    RISK_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
)

from .modes import (
    AmbientModeHandler,
    AdvisoryModeHandler,
    GuardianModeHandler,
    CrisisModeHandler,
    generate_intervention_id,
)

from .engine import (
    InterventionEngine,
    InterventionQueue,
    create_intervention_engine,
)

__all__ = [
    # Enums
    "InterventionMode",
    "RiskLevel",
    "RiskCategory",
    "InterventionType",
    "InterventionPriority",
    "InterventionEventType",
    # Data classes
    "RiskFactor",
    "RiskAssessment",
    "Intervention",
    "InterventionResponse",
    "ModeTransition",
    "InterventionEvent",
    "InterventionConfig",
    # Abstract base
    "ModeHandler",
    # Risk engine
    "RiskAssessmentEngine",
    "RiskPatternDetector",
    "RISK_PATTERNS",
    "DEFAULT_CATEGORY_WEIGHTS",
    # Mode handlers
    "AmbientModeHandler",
    "AdvisoryModeHandler",
    "GuardianModeHandler",
    "CrisisModeHandler",
    "generate_intervention_id",
    # Engine
    "InterventionEngine",
    "InterventionQueue",
    "create_intervention_engine",
]
