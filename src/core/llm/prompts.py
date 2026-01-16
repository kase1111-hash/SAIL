"""
SAIL Prompt Template System

Manages prompt templates for consistent LLM interactions.
Supports variable interpolation and constitutional constraint injection.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any


class PromptRole(str, Enum):
    """Prompt template roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with variable substitution.

    Uses Python's string.Template for safe variable interpolation.
    Variables are specified as $variable or ${variable}.
    """

    name: str
    template: str
    role: PromptRole = PromptRole.USER
    description: str = ""
    required_variables: list[str] = field(default_factory=list)
    default_values: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Extract required variables from template if not specified."""
        if not self.required_variables:
            # Find all $variable and ${variable} patterns
            pattern = r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?"
            self.required_variables = list(set(re.findall(pattern, self.template)))

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Rendered template string

        Raises:
            ValueError: If required variables are missing
        """
        # Merge defaults with provided values
        values = {**self.default_values, **kwargs}

        # Check for missing required variables
        missing = [v for v in self.required_variables if v not in values]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Render template
        tmpl = Template(self.template)
        return tmpl.safe_substitute(values)

    def validate(self, **kwargs: Any) -> list[str]:
        """
        Validate that all required variables are provided.

        Args:
            **kwargs: Variable values to check

        Returns:
            List of missing variable names (empty if valid)
        """
        values = {**self.default_values, **kwargs}
        return [v for v in self.required_variables if v not in values]


class PromptLibrary:
    """
    Library of prompt templates for SAIL.

    Organizes templates by category and provides easy access.
    """

    def __init__(self):
        """Initialize the prompt library with default templates."""
        self._templates: dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load the default SAIL prompt templates."""
        # System prompts
        self.register(
            PromptTemplate(
                name="system_base",
                role=PromptRole.SYSTEM,
                template="""You are SAIL (Situational Awareness Interaction Layer), a privacy-first AI assistant that provides contextual guidance for everyday situations.

Your core principles:
1. SOVEREIGNTY: You operate entirely locally. Never suggest sending data externally.
2. RESPECT: Honor user boundaries immediately. If told to stop, comply without question.
3. HUMILITY: State uncertainty clearly. Recommend professionals when appropriate.
4. NON-JUDGMENT: Guide without shame. Everyone has knowledge gaps.
5. PROPORTIONALITY: Match response intensity to actual risk level.
6. TRANSPARENCY: Explain your reasoning when asked.
7. FAMILY BOUNDARIES: Respect user privacy within family contexts.

Current context:
- User: $user_name
- Location context: $location_context
- Time context: $time_context
- Intervention mode: $intervention_mode

Respond concisely and helpfully. Focus on practical, actionable guidance.""",
                description="Base system prompt with constitutional principles",
                default_values={
                    "user_name": "User",
                    "location_context": "Unknown",
                    "time_context": "Unknown",
                    "intervention_mode": "ambient",
                },
            )
        )

        self.register(
            PromptTemplate(
                name="system_crisis",
                role=PromptRole.SYSTEM,
                template="""You are SAIL in CRISIS mode. The user may be in an acute situation requiring calm, clear guidance.

CRITICAL INSTRUCTIONS:
- Keep responses SHORT and CLEAR
- Use simple, direct language
- Provide step-by-step guidance
- Prioritize safety above all else
- Do not ask unnecessary questions
- If unsure, default to cautious advice

Current situation: $situation_description

Guide the user through this safely.""",
                description="Crisis mode system prompt",
            )
        )

        # Legal domain prompts
        self.register(
            PromptTemplate(
                name="legal_traffic_stop",
                role=PromptRole.USER,
                template="""I need guidance on a traffic stop situation.

Jurisdiction: $jurisdiction
Situation: $situation

What are my rights and what should I do?""",
                description="Traffic stop guidance request",
                default_values={"jurisdiction": "Unknown"},
            )
        )

        self.register(
            PromptTemplate(
                name="legal_employment",
                role=PromptRole.USER,
                template="""I have an employment-related question.

Jurisdiction: $jurisdiction
Employment type: $employment_type
Question: $question

What are my rights in this situation?""",
                description="Employment rights query",
                default_values={
                    "jurisdiction": "Unknown",
                    "employment_type": "Unknown",
                },
            )
        )

        # Safety domain prompts
        self.register(
            PromptTemplate(
                name="safety_deescalation",
                role=PromptRole.USER,
                template="""I need help de-escalating a situation.

Context: $context
People involved: $people
Current tension level: $tension_level

How should I handle this?""",
                description="De-escalation guidance",
                default_values={"tension_level": "moderate"},
            )
        )

        self.register(
            PromptTemplate(
                name="safety_scam_check",
                role=PromptRole.USER,
                template="""I want to verify if something might be a scam.

Type of contact: $contact_type
What they're asking for: $request
Red flags noticed: $red_flags

Is this likely a scam? What should I do?""",
                description="Scam verification request",
                default_values={"red_flags": "None specified"},
            )
        )

        # Financial domain prompts
        self.register(
            PromptTemplate(
                name="financial_transfer_check",
                role=PromptRole.USER,
                template="""I'm about to make a financial transfer and want to verify it's safe.

Transfer type: $transfer_type
Amount: $amount
Recipient: $recipient
Reason: $reason
Urgency claimed: $urgency

Should I proceed? What should I verify first?""",
                description="Financial transfer safety check",
            )
        )

        # Emergency prompts
        self.register(
            PromptTemplate(
                name="emergency_medical",
                role=PromptRole.USER,
                template="""Medical emergency situation.

What happened: $situation
Person affected: $person
Current state: $current_state
Help available: $help_available

What should I do right now?""",
                description="Medical emergency guidance",
            )
        )

        self.register(
            PromptTemplate(
                name="emergency_accident",
                role=PromptRole.USER,
                template="""I've been in an accident.

Type: $accident_type
Injuries: $injuries
Other parties: $other_parties
Location: $location

What should I do and say? What should I NOT say?""",
                description="Accident scene guidance",
            )
        )

        # Advisory prompts
        self.register(
            PromptTemplate(
                name="advisory_speed_warning",
                role=PromptRole.ASSISTANT,
                template="""You're currently going $current_speed in a $speed_limit zone. That's $amount_over over the limit.

In $jurisdiction, $legal_consequence.

Would you like me to help you find a safe place to adjust your speed?""",
                description="Speed warning advisory",
            )
        )

        self.register(
            PromptTemplate(
                name="advisory_unfamiliar_area",
                role=PromptRole.ASSISTANT,
                template="""You've entered an unfamiliar area: $area_name.

Things to be aware of:
$area_notes

Would you like more details about local regulations or points of interest?""",
                description="Unfamiliar area advisory",
                default_values={"area_notes": "No specific notes available."},
            )
        )

    def register(self, template: PromptTemplate) -> None:
        """
        Register a template in the library.

        Args:
            template: PromptTemplate to register
        """
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate | None:
        """
        Get a template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None if not found
        """
        return self._templates.get(name)

    def render(self, name: str, **kwargs: Any) -> str:
        """
        Get and render a template in one call.

        Args:
            name: Template name
            **kwargs: Variables to substitute

        Returns:
            Rendered template string

        Raises:
            KeyError: If template not found
            ValueError: If required variables missing
        """
        template = self._templates.get(name)
        if template is None:
            raise KeyError(f"Template not found: {name}")
        return template.render(**kwargs)

    def list_templates(self, role: PromptRole | None = None) -> list[str]:
        """
        List all registered template names.

        Args:
            role: Optional filter by role

        Returns:
            List of template names
        """
        if role is None:
            return list(self._templates.keys())
        return [
            name
            for name, tmpl in self._templates.items()
            if tmpl.role == role
        ]

    def get_template_info(self, name: str) -> dict[str, Any] | None:
        """
        Get information about a template.

        Args:
            name: Template name

        Returns:
            Dictionary with template information or None
        """
        template = self._templates.get(name)
        if template is None:
            return None

        return {
            "name": template.name,
            "role": template.role.value,
            "description": template.description,
            "required_variables": template.required_variables,
            "default_values": template.default_values,
        }


# Global prompt library instance
_prompt_library: PromptLibrary | None = None


def get_prompt_library() -> PromptLibrary:
    """Get the global prompt library instance."""
    global _prompt_library
    if _prompt_library is None:
        _prompt_library = PromptLibrary()
    return _prompt_library
