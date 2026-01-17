"""
SAIL Knowledge Domains

Specialized knowledge domains for different areas of expertise.
"""

from src.knowledge.domains.emergency import (
    EmergencyKnowledgeDomain,
    create_emergency_domain,
)
from src.knowledge.domains.financial import (
    SCAM_INDICATORS,
    FinancialKnowledgeDomain,
    analyze_scam_risk,
    create_financial_domain,
)
from src.knowledge.domains.legal import (
    LegalKnowledgeDomain,
    create_legal_domain,
)
from src.knowledge.domains.safety import (
    SafetyKnowledgeDomain,
    create_safety_domain,
)

__all__ = [
    # Legal
    "LegalKnowledgeDomain",
    "create_legal_domain",
    # Safety
    "SafetyKnowledgeDomain",
    "create_safety_domain",
    # Financial
    "FinancialKnowledgeDomain",
    "create_financial_domain",
    "analyze_scam_risk",
    "SCAM_INDICATORS",
    # Emergency
    "EmergencyKnowledgeDomain",
    "create_emergency_domain",
]
