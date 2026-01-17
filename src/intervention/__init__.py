"""
Intervention Framework for SAIL.

Provides four intervention modes with proportional responses:
- Ambient: Passive listening, addressed-only responses
- Advisory: Gentle suggestions for detected risks
- Guardian: Active alerts for high-risk patterns
- Crisis: Hands-free walkthrough for acute situations
"""

from .base import (
    Intervention,
    InterventionConfig,
    InterventionEvent,
    InterventionEventType,
    # Enums
    InterventionMode,
    InterventionPriority,
    InterventionResponse,
    InterventionType,
    # Abstract base
    ModeHandler,
    ModeTransition,
    RiskAssessment,
    RiskCategory,
    # Data classes
    RiskFactor,
    RiskLevel,
)
from .engine import (
    InterventionEngine,
    InterventionQueue,
    create_intervention_engine,
)
from .modes import (
    AdvisoryModeHandler,
    AmbientModeHandler,
    CrisisModeHandler,
    GuardianModeHandler,
    generate_intervention_id,
)
from .risk import (
    DEFAULT_CATEGORY_WEIGHTS,
    RISK_PATTERNS,
    RiskAssessmentEngine,
    RiskPatternDetector,
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
