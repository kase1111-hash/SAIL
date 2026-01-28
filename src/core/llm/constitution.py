"""
SAIL Constitutional Constraints

Enforces SAIL's behavioral principles through response filtering
and validation. Ensures all outputs comply with the constitution.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ConstitutionalPrinciple(str, Enum):
    """SAIL's constitutional principles."""

    SOVEREIGNTY = "sovereignty"
    RESPECT = "respect"
    HUMILITY = "humility"
    NON_JUDGMENT = "non_judgment"
    PROPORTIONALITY = "proportionality"
    TRANSPARENCY = "transparency"
    FAMILY_BOUNDARIES = "family_boundaries"


@dataclass
class ConstitutionalViolation:
    """Represents a detected constitutional violation."""

    principle: ConstitutionalPrinciple
    description: str
    severity: str  # "warning", "violation", "critical"
    suggested_fix: str | None = None


@dataclass
class ConstitutionalCheck:
    """A single constitutional check."""

    principle: ConstitutionalPrinciple
    name: str
    description: str
    check_fn: Callable[[str], ConstitutionalViolation | None]


class Constitution:
    """
    SAIL's constitutional constraint system.

    Validates responses against SAIL's core principles:
    1. SOVEREIGNTY: No data leaves user hardware
    2. RESPECT: Honor user boundaries immediately
    3. HUMILITY: State uncertainty, recommend professionals
    4. NON-JUDGMENT: Guide without shame
    5. PROPORTIONALITY: Match intensity to risk
    6. TRANSPARENCY: Explain reasoning when asked
    7. FAMILY BOUNDARIES: Respect privacy within family
    """

    def __init__(self):
        """Initialize the constitution with default checks."""
        self._checks: list[ConstitutionalCheck] = []
        self._load_default_checks()

    def _load_default_checks(self) -> None:
        """Load the default constitutional checks."""

        # SOVEREIGNTY checks
        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.SOVEREIGNTY,
                name="no_external_data_sharing",
                description="Detect suggestions to send data externally",
                check_fn=self._check_no_external_sharing,
            )
        )

        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.SOVEREIGNTY,
                name="no_cloud_services",
                description="Detect recommendations for cloud services",
                check_fn=self._check_no_cloud_services,
            )
        )

        # HUMILITY checks
        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.HUMILITY,
                name="medical_disclaimer",
                description="Ensure medical advice includes professional recommendation",
                check_fn=self._check_medical_disclaimer,
            )
        )

        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.HUMILITY,
                name="legal_disclaimer",
                description="Ensure legal advice includes professional recommendation",
                check_fn=self._check_legal_disclaimer,
            )
        )

        # NON-JUDGMENT checks
        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.NON_JUDGMENT,
                name="no_shaming_language",
                description="Detect judgmental or shaming language",
                check_fn=self._check_no_shaming,
            )
        )

        # PROPORTIONALITY checks
        self.add_check(
            ConstitutionalCheck(
                principle=ConstitutionalPrinciple.PROPORTIONALITY,
                name="no_catastrophizing",
                description="Detect unnecessarily alarming language",
                check_fn=self._check_no_catastrophizing,
            )
        )

    def add_check(self, check: ConstitutionalCheck) -> None:
        """Add a constitutional check."""
        self._checks.append(check)

    def validate(self, response: str, context: dict | None = None) -> list[ConstitutionalViolation]:
        """
        Validate a response against all constitutional checks.

        Args:
            response: The LLM response to validate
            context: Optional context for context-aware checks

        Returns:
            List of detected violations (empty if compliant)
        """
        violations = []

        for check in self._checks:
            violation = check.check_fn(response)
            if violation:
                violations.append(violation)

        return violations

    def is_compliant(self, response: str, context: dict | None = None) -> bool:
        """
        Check if a response is constitutionally compliant.

        Args:
            response: The LLM response to check
            context: Optional context

        Returns:
            True if compliant, False otherwise
        """
        violations = self.validate(response, context)
        # Only fail on violations and critical, not warnings
        return not any(v.severity in ("violation", "critical") for v in violations)

    def filter_response(self, response: str, context: dict | None = None) -> str:
        """
        Filter a response to be constitutionally compliant.

        Attempts to fix minor issues while preserving meaning.

        Args:
            response: The LLM response to filter
            context: Optional context

        Returns:
            Filtered response
        """
        filtered = response

        # Apply automatic fixes
        filtered = self._add_disclaimers_if_needed(filtered)
        filtered = self._soften_language(filtered)

        return filtered

    # --- Check implementations ---

    def _check_no_external_sharing(self, response: str) -> ConstitutionalViolation | None:
        """Check for suggestions to share data externally."""
        patterns = [
            r"send\s+(your\s+)?(data|information|details)\s+to",
            r"upload\s+(to|your)",
            r"share\s+(with|to)\s+(the\s+)?(cloud|server|service)",
            r"connect\s+to\s+(external|remote|cloud)",
            r"transmit\s+(your\s+)?(data|information)",
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.SOVEREIGNTY,
                    description="Response suggests sending data externally",
                    severity="violation",
                    suggested_fix="Remove external data sharing suggestion",
                )

        return None

    def _check_no_cloud_services(self, response: str) -> ConstitutionalViolation | None:
        """Check for cloud service recommendations."""
        # Only flag if recommending using cloud for SAIL functions
        patterns = [
            r"use\s+(a\s+)?cloud\s+(service|provider|platform)",
            r"sign\s+up\s+for",
            r"create\s+an?\s+account\s+(with|on)",
            r"subscribe\s+to",
        ]

        # Context: only flag if in context of SAIL functionality
        sail_context_patterns = [
            r"(store|save|backup)\s+(your\s+)?(data|context|history)",
            r"sync\s+(across|between)",
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                for ctx_pattern in sail_context_patterns:
                    if re.search(ctx_pattern, response, re.IGNORECASE):
                        return ConstitutionalViolation(
                            principle=ConstitutionalPrinciple.SOVEREIGNTY,
                            description="Response recommends cloud services for SAIL data",
                            severity="violation",
                            suggested_fix="Suggest local alternatives only",
                        )

        return None

    def _check_medical_disclaimer(self, response: str) -> ConstitutionalViolation | None:
        """Check that medical advice includes appropriate disclaimers."""
        medical_indicators = [
            r"(take|prescribe|dosage|medication|medicine|drug|treatment)",
            r"(diagnos|symptom|condition|disease|illness)",
            r"(medical|health)\s+(advice|guidance|recommendation)",
        ]

        has_medical_content = any(
            re.search(p, response, re.IGNORECASE) for p in medical_indicators
        )

        if not has_medical_content:
            return None

        disclaimer_patterns = [
            r"(consult|see|contact|call)\s+(a\s+)?(doctor|physician|medical\s+professional|healthcare)",
            r"(not\s+)?(medical\s+advice|substitute\s+for\s+professional)",
            r"seek\s+(professional\s+)?(medical\s+)?(help|attention|care)",
            r"professional\s+medical\s+(advice|guidance)",
        ]

        has_disclaimer = any(
            re.search(p, response, re.IGNORECASE) for p in disclaimer_patterns
        )

        if not has_disclaimer:
            return ConstitutionalViolation(
                principle=ConstitutionalPrinciple.HUMILITY,
                description="Medical content without professional recommendation",
                severity="warning",
                suggested_fix="Add recommendation to consult a medical professional",
            )

        return None

    def _check_legal_disclaimer(self, response: str) -> ConstitutionalViolation | None:
        """Check that legal advice includes appropriate disclaimers."""
        legal_indicators = [
            r"(legal|law|lawyer|attorney|court|sue|lawsuit)",
            r"(rights|liability|contract|agreement)",
            r"(illegal|unlawful|criminal|felony|misdemeanor)",
        ]

        has_legal_content = any(
            re.search(p, response, re.IGNORECASE) for p in legal_indicators
        )

        if not has_legal_content:
            return None

        disclaimer_patterns = [
            r"(consult|see|contact|call)\s+(a\s+)?(lawyer|attorney|legal\s+professional)",
            r"(not\s+)?(legal\s+advice|substitute\s+for\s+legal)",
            r"seek\s+(legal\s+)?(counsel|advice|help)",
            r"professional\s+legal\s+(advice|guidance)",
            r"(varies|depend)\s+(by|on)\s+(state|jurisdiction|location)",
        ]

        has_disclaimer = any(
            re.search(p, response, re.IGNORECASE) for p in disclaimer_patterns
        )

        if not has_disclaimer:
            return ConstitutionalViolation(
                principle=ConstitutionalPrinciple.HUMILITY,
                description="Legal content without professional recommendation",
                severity="warning",
                suggested_fix="Add recommendation to consult a legal professional",
            )

        return None

    def _check_no_shaming(self, response: str) -> ConstitutionalViolation | None:
        """Check for judgmental or shaming language."""
        shaming_patterns = [
            r"you\s+should\s+(have\s+)?(known|realized)",
            r"(stupid|dumb|foolish|idiotic)\s+(of\s+you|thing|mistake)",
            r"how\s+(could|can)\s+you\s+(not\s+)?know",
            r"everyone\s+knows\s+(that|this)",
            r"(obviously|clearly)\s+you\s+(should|shouldn't)",
            r"(shame\s+on|ashamed\s+of)\s+you",
            r"what\s+were\s+you\s+thinking",
        ]

        for pattern in shaming_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.NON_JUDGMENT,
                    description="Response contains judgmental language",
                    severity="violation",
                    suggested_fix="Rephrase without judgment",
                )

        return None

    def _check_no_catastrophizing(self, response: str) -> ConstitutionalViolation | None:
        """Check for unnecessarily alarming language."""
        catastrophizing_patterns = [
            r"you('re|\s+are)\s+(definitely|certainly)\s+going\s+to\s+(die|be\s+killed)",
            r"(worst|terrible|horrible|devastating)\s+thing\s+that\s+could\s+happen",
            r"(no\s+hope|hopeless|doomed)",
            r"(your\s+life|everything)\s+is\s+(over|ruined)",
        ]

        for pattern in catastrophizing_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.PROPORTIONALITY,
                    description="Response contains unnecessarily alarming language",
                    severity="warning",
                    suggested_fix="Use measured, proportionate language",
                )

        return None

    # --- Filtering helpers ---

    def _add_disclaimers_if_needed(self, response: str) -> str:
        """Add necessary disclaimers to response."""
        modified = response

        # Check for medical content without disclaimer
        violation = self._check_medical_disclaimer(response)
        if violation and violation.severity == "warning":
            # Add medical disclaimer if missing
            medical_disclaimer = "\n\nPlease consult a medical professional for personalized advice."
            if medical_disclaimer.strip() not in response:
                modified += medical_disclaimer

        # Check for legal content without disclaimer
        violation = self._check_legal_disclaimer(response)
        if violation and violation.severity == "warning":
            # Add legal disclaimer if missing
            legal_disclaimer = "\n\nLaws vary by jurisdiction. Consider consulting a legal professional for your specific situation."
            if legal_disclaimer.strip() not in response:
                modified += legal_disclaimer

        return modified

    def _soften_language(self, response: str) -> str:
        """Soften harsh or judgmental language."""
        # Simple replacements for common harsh phrases
        replacements = [
            (r"\byou\s+must\b", "you may want to"),
            (r"\byou\s+have\s+to\b", "it would be good to"),
            (r"\bnever\s+do\b", "try to avoid"),
            (r"\balways\s+do\b", "consider"),
        ]

        result = response
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


class UncertaintyDetector:
    """
    Detects when the LLM expresses uncertainty.

    Used to identify when SAIL should recommend professional help
    or acknowledge limitations.
    """

    UNCERTAINTY_PATTERNS = [
        r"i('m|\s+am)\s+not\s+(sure|certain)",
        r"i\s+don't\s+know",
        r"(it's|that's)\s+(unclear|uncertain|unknown)",
        r"(may|might|could)\s+be",
        r"(possibly|perhaps|maybe)",
        r"i\s+(can't|cannot)\s+(say|tell)\s+for\s+(sure|certain)",
        r"(this|that)\s+depends\s+on",
        r"(varies|different)\s+(by|depending)",
        r"i\s+would\s+need\s+more\s+(information|details|context)",
    ]

    HIGH_UNCERTAINTY_PATTERNS = [
        r"i\s+(really\s+)?don't\s+know",
        r"(beyond|outside)\s+(my|the)\s+(knowledge|scope|expertise)",
        r"you\s+should\s+(really\s+)?(ask|consult|see)\s+(a\s+)?(professional|expert|specialist)",
        r"i\s+(can't|cannot)\s+(help|assist)\s+(with\s+)?(that|this)",
    ]

    @classmethod
    def detect_uncertainty(cls, response: str) -> float:
        """
        Detect uncertainty level in a response.

        Args:
            response: The response to analyze

        Returns:
            Uncertainty score from 0.0 (certain) to 1.0 (very uncertain)
        """
        response_lower = response.lower()

        # Count uncertainty indicators
        uncertainty_count = sum(
            1 for p in cls.UNCERTAINTY_PATTERNS if re.search(p, response_lower)
        )

        high_uncertainty_count = sum(
            1 for p in cls.HIGH_UNCERTAINTY_PATTERNS if re.search(p, response_lower)
        )

        # Calculate score
        # High uncertainty patterns count double
        total_score = uncertainty_count + (high_uncertainty_count * 2)

        # Normalize to 0-1 range (cap at 5 indicators)
        return min(total_score / 5.0, 1.0)

    @classmethod
    def should_recommend_professional(cls, response: str, threshold: float = 0.6) -> bool:
        """
        Check if uncertainty level warrants professional recommendation.

        Args:
            response: The response to analyze
            threshold: Uncertainty threshold (default 0.6)

        Returns:
            True if professional help should be recommended
        """
        return cls.detect_uncertainty(response) >= threshold


# Global constitution instance
_constitution: Constitution | None = None


def get_constitution() -> Constitution:
    """Get the global constitution instance."""
    global _constitution
    if _constitution is None:
        _constitution = Constitution()
    return _constitution
