"""
SAIL Response Parsing Utilities

Utilities for parsing and processing LLM responses.
Extracts structured data, handles formatting, and normalizes output.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResponseType(str, Enum):
    """Types of parsed responses."""

    TEXT = "text"
    STEPS = "steps"
    LIST = "list"
    STRUCTURED = "structured"
    GUIDANCE = "guidance"


@dataclass
class ParsedResponse:
    """A parsed and structured LLM response."""

    type: ResponseType
    content: Any
    raw: str
    metadata: dict | None = None

    def as_text(self) -> str:
        """Get response as plain text."""
        if self.type == ResponseType.TEXT:
            return self.content
        elif self.type == ResponseType.STEPS:
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.content))
        elif self.type == ResponseType.LIST:
            return "\n".join(f"• {item}" for item in self.content)
        elif self.type == ResponseType.GUIDANCE:
            guidance = self.content
            parts = []
            if guidance.get("summary"):
                parts.append(guidance["summary"])
            if guidance.get("steps"):
                parts.append("\n".join(f"{i+1}. {s}" for i, s in enumerate(guidance["steps"])))
            if guidance.get("warning"):
                parts.append(f"⚠️ {guidance['warning']}")
            return "\n\n".join(parts)
        else:
            return str(self.content)


class ResponseParser:
    """
    Parser for LLM responses.

    Extracts structured information and normalizes formatting.
    """

    @staticmethod
    def parse(response: str) -> ParsedResponse:
        """
        Parse an LLM response into structured format.

        Automatically detects response type and extracts structure.

        Args:
            response: Raw LLM response text

        Returns:
            ParsedResponse with extracted structure
        """
        response = response.strip()

        # Try to detect and parse different formats
        if ResponseParser._looks_like_json(response):
            return ResponseParser._parse_json(response)

        if ResponseParser._looks_like_numbered_steps(response):
            return ResponseParser._parse_steps(response)

        if ResponseParser._looks_like_bullet_list(response):
            return ResponseParser._parse_list(response)

        # Default to text
        return ParsedResponse(
            type=ResponseType.TEXT,
            content=response,
            raw=response,
        )

    @staticmethod
    def parse_as_guidance(response: str) -> ParsedResponse:
        """
        Parse response as guidance/advice format.

        Extracts summary, steps, warnings, and recommendations.

        Args:
            response: Raw LLM response

        Returns:
            ParsedResponse with guidance structure
        """
        guidance = {
            "summary": None,
            "steps": [],
            "warnings": [],
            "recommendations": [],
            "disclaimers": [],
        }

        lines = response.strip().split("\n")
        current_section = "summary"
        summary_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section markers
            lower_line = line.lower()

            # Numbered lists are steps
            if re.match(r"^\d+[\.\)]\s*", line):
                current_section = "steps"

            if any(marker in lower_line for marker in ["warning", "caution", "important", "⚠"]):
                current_section = "warnings"

            if any(marker in lower_line for marker in ["recommend", "suggest", "consider"]):
                current_section = "recommendations"

            if any(marker in lower_line for marker in ["disclaimer", "note:", "please note"]):
                current_section = "disclaimers"

            # Extract content based on current section
            if current_section == "summary" and not re.match(r"^\d+[\.\)]\s*", line):
                summary_lines.append(line)
            elif current_section == "steps":
                step = re.sub(r"^\d+[\.\)]\s*", "", line)
                if step:
                    guidance["steps"].append(step)
            elif current_section == "warnings":
                warning = re.sub(r"^[⚠️\-\*]\s*", "", line)
                if warning and "warning" not in warning.lower()[:10]:
                    guidance["warnings"].append(warning)
            elif current_section == "recommendations":
                rec = re.sub(r"^[\-\*]\s*", "", line)
                guidance["recommendations"].append(rec)
            elif current_section == "disclaimers":
                disc = re.sub(r"^[\-\*]\s*", "", line)
                guidance["disclaimers"].append(disc)

        guidance["summary"] = " ".join(summary_lines) if summary_lines else None

        return ParsedResponse(
            type=ResponseType.GUIDANCE,
            content=guidance,
            raw=response,
        )

    @staticmethod
    def extract_steps(response: str) -> list[str]:
        """
        Extract numbered steps from a response.

        Args:
            response: Raw response text

        Returns:
            List of step strings
        """
        steps = []

        # Match various numbering formats: "1.", "1)", "Step 1:", etc.
        patterns = [
            r"(?:^|\n)\s*(\d+)[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\n\n|$)",
            r"(?:^|\n)\s*[Ss]tep\s*(\d+)[:\.]?\s*(.+?)(?=\n\s*[Ss]tep|\n\n|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Sort by step number and extract text
                sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                steps = [m[1].strip() for m in sorted_matches]
                break

        return steps

    @staticmethod
    def extract_bullet_points(response: str) -> list[str]:
        """
        Extract bullet points from a response.

        Args:
            response: Raw response text

        Returns:
            List of bullet point strings
        """
        # Match various bullet formats: -, *, •, ▪, etc.
        pattern = r"(?:^|\n)\s*[\-\*•▪◦]\s*(.+?)(?=\n\s*[\-\*•▪◦]|\n\n|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    @staticmethod
    def extract_json(response: str) -> dict | list | None:
        """
        Extract JSON from a response.

        Handles JSON embedded in markdown code blocks.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON or None if not found
        """
        # Try to find JSON in code blocks first
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(code_block_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Try to parse the whole response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in text
        json_patterns = [
            r"\{[\s\S]*\}",  # JSON object
            r"\[[\s\S]*\]",  # JSON array
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        return None

    @staticmethod
    def clean_response(response: str) -> str:
        """
        Clean and normalize a response.

        Removes unnecessary whitespace, normalizes line endings, etc.

        Args:
            response: Raw response text

        Returns:
            Cleaned response
        """
        # Normalize line endings
        response = response.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (more than 2)
        response = re.sub(r"\n{3,}", "\n\n", response)

        # Strip leading/trailing whitespace
        response = response.strip()

        # Remove common LLM artifacts
        artifacts = [
            r"^(Sure|Certainly|Of course|Absolutely)[,!]?\s*",
            r"^(Here'?s?|I'?ll|Let me)\s+.{0,30}:\s*\n",
        ]

        for artifact in artifacts:
            response = re.sub(artifact, "", response, flags=re.IGNORECASE)

        return response.strip()

    @staticmethod
    def truncate_response(response: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate a response to a maximum length.

        Tries to truncate at sentence or word boundaries.

        Args:
            response: Response to truncate
            max_length: Maximum character length
            suffix: Suffix to append if truncated

        Returns:
            Truncated response
        """
        if len(response) <= max_length:
            return response

        # Account for suffix length
        target_length = max_length - len(suffix)

        # Try to find a sentence boundary
        truncated = response[:target_length]
        last_sentence = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
        )

        if last_sentence > target_length * 0.5:  # At least half the content
            return response[: last_sentence + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > target_length * 0.7:
            return response[:last_space] + suffix

        # Hard truncate
        return response[:target_length] + suffix

    # --- Private helper methods ---

    @staticmethod
    def _looks_like_json(response: str) -> bool:
        """Check if response appears to be JSON."""
        stripped = response.strip()
        return (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
            or "```json" in response
        )

    @staticmethod
    def _looks_like_numbered_steps(response: str) -> bool:
        """Check if response appears to contain numbered steps."""
        # Look for at least 2 numbered items
        pattern = r"(?:^|\n)\s*\d+[\.\)]\s*\S"
        matches = re.findall(pattern, response)
        return len(matches) >= 2

    @staticmethod
    def _looks_like_bullet_list(response: str) -> bool:
        """Check if response appears to be a bullet list."""
        # Look for at least 2 bullet points
        pattern = r"(?:^|\n)\s*[\-\*•]\s*\S"
        matches = re.findall(pattern, response)
        return len(matches) >= 2

    @staticmethod
    def _parse_json(response: str) -> ParsedResponse:
        """Parse response as JSON."""
        data = ResponseParser.extract_json(response)
        if data is not None:
            return ParsedResponse(
                type=ResponseType.STRUCTURED,
                content=data,
                raw=response,
            )
        # Fallback to text if JSON parsing fails
        return ParsedResponse(
            type=ResponseType.TEXT,
            content=response,
            raw=response,
        )

    @staticmethod
    def _parse_steps(response: str) -> ParsedResponse:
        """Parse response as numbered steps."""
        steps = ResponseParser.extract_steps(response)
        return ParsedResponse(
            type=ResponseType.STEPS,
            content=steps,
            raw=response,
        )

    @staticmethod
    def _parse_list(response: str) -> ParsedResponse:
        """Parse response as bullet list."""
        items = ResponseParser.extract_bullet_points(response)
        return ParsedResponse(
            type=ResponseType.LIST,
            content=items,
            raw=response,
        )


class IntentExtractor:
    """
    Extracts user intent from queries.

    Identifies what the user is asking about to route to
    appropriate knowledge domains.
    """

    INTENT_PATTERNS = {
        "legal_traffic": [
            r"(traffic|pulled over|cop|police|officer|speeding|ticket)",
            r"(search|warrant|rights)\s+(my\s+)?(car|vehicle)",
        ],
        "legal_employment": [
            r"(job|work|employer|boss|fired|quit|wages|salary|overtime)",
            r"(break|lunch|hours|schedule)\s+(at\s+)?work",
        ],
        "legal_contract": [
            r"(sign|signed|contract|agreement|lease|terms)",
            r"(cancel|cooling\s*off|void)\s+(a\s+)?(contract|agreement)",
        ],
        "safety_deescalation": [
            r"(calm|deescalate|angry|aggressive|confrontation)",
            r"(argument|fight|conflict)\s+(with|between)",
        ],
        "safety_scam": [
            r"(scam|fraud|phishing|suspicious|too good to be true)",
            r"(verify|check|legitimate)\s+(this|if)",
        ],
        "financial_transfer": [
            r"(wire|transfer|send)\s+(money|funds|payment)",
            r"(bank|account)\s+(details|information|number)",
        ],
        "emergency_medical": [
            r"(medical|health)\s+emergency",
            r"(hurt|injured|bleeding|unconscious|not breathing)",
            r"(call|need)\s+(911|ambulance|doctor)",
        ],
        "emergency_accident": [
            r"(car|vehicle)\s+(accident|crash|collision)",
            r"(hit|struck)\s+(by|someone|something)",
        ],
    }

    @classmethod
    def extract_intent(cls, query: str) -> list[tuple[str, float]]:
        """
        Extract possible intents from a query.

        Args:
            query: User query text

        Returns:
            List of (intent, confidence) tuples, sorted by confidence
        """
        query_lower = query.lower()
        results = []

        for intent, patterns in cls.INTENT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1

            if matches > 0:
                # Confidence based on number of pattern matches
                confidence = min(matches / len(patterns) + 0.3, 1.0)
                results.append((intent, confidence))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @classmethod
    def get_primary_intent(cls, query: str) -> str | None:
        """
        Get the most likely intent for a query.

        Args:
            query: User query text

        Returns:
            Intent string or None if no intent detected
        """
        intents = cls.extract_intent(query)
        if intents and intents[0][1] >= 0.4:  # Minimum confidence threshold
            return intents[0][0]
        return None

    @classmethod
    def get_domain(cls, intent: str) -> str | None:
        """
        Get the knowledge domain for an intent.

        Args:
            intent: Intent string

        Returns:
            Domain string (legal, safety, financial, emergency)
        """
        domain_map = {
            "legal_traffic": "legal",
            "legal_employment": "legal",
            "legal_contract": "legal",
            "safety_deescalation": "safety",
            "safety_scam": "safety",
            "financial_transfer": "financial",
            "emergency_medical": "emergency",
            "emergency_accident": "emergency",
        }
        return domain_map.get(intent)
