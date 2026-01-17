"""
SAIL Custom Briefings System

Provides role-specific and age-appropriate knowledge content
tailored to individual users based on their custom briefings
configuration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.users.base import AccessLevel, User, UserManagerConfig, UserRole

logger = logging.getLogger(__name__)


# ============================================================================
# Briefing Types
# ============================================================================


class BriefingCategory(str, Enum):
    """Categories of custom briefings."""

    DRIVING = "driving"             # New driver content
    EMPLOYMENT = "employment"       # First job, workplace rights
    EDUCATION = "education"         # School-related content
    DATING = "dating"               # Safe dating practices
    FINANCIAL = "financial"         # Age-appropriate financial literacy
    LEGAL = "legal"                 # Legal rights and responsibilities
    SAFETY = "safety"               # Safety protocols
    DIGITAL = "digital"             # Online safety


class AgeAppropriateness(str, Enum):
    """Age appropriateness levels."""

    CHILD = "child"                 # Under 13
    TEEN = "teen"                   # 13-17
    YOUNG_ADULT = "young_adult"     # 18-25
    ADULT = "adult"                 # 18+
    ALL = "all"                     # All ages


# ============================================================================
# Briefing Content
# ============================================================================


@dataclass
class BriefingContent:
    """
    A piece of briefing content.
    """

    briefing_id: str
    title: str
    content: str
    summary: str = ""

    # Categorization
    category: BriefingCategory = BriefingCategory.SAFETY
    tags: list[str] = field(default_factory=list)

    # Access control
    age_appropriate: AgeAppropriateness = AgeAppropriateness.ALL
    min_access_level: AccessLevel = AccessLevel.LIMITED

    # Metadata
    source: str | None = None
    last_updated: datetime = field(default_factory=datetime.now)

    def is_accessible_by(self, user: User) -> bool:
        """Check if this content is accessible by a user."""
        # Check access level
        access_order = [AccessLevel.LIMITED, AccessLevel.STANDARD, AccessLevel.FULL]
        user_access_idx = access_order.index(user.access_level)
        min_access_idx = access_order.index(self.min_access_level)

        if user_access_idx < min_access_idx and not user.is_admin:
            return False

        # Check age appropriateness for minors
        if user.role == UserRole.MINOR:
            if self.age_appropriate == AgeAppropriateness.ADULT:
                return False
            if self.age_appropriate == AgeAppropriateness.YOUNG_ADULT:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "briefing_id": self.briefing_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "category": self.category.value,
            "tags": self.tags,
            "age_appropriate": self.age_appropriate.value,
            "min_access_level": self.min_access_level.value,
            "source": self.source,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BriefingContent:
        """Create from dictionary."""
        return cls(
            briefing_id=data["briefing_id"],
            title=data["title"],
            content=data["content"],
            summary=data.get("summary", ""),
            category=BriefingCategory(data.get("category", "safety")),
            tags=data.get("tags", []),
            age_appropriate=AgeAppropriateness(data.get("age_appropriate", "all")),
            min_access_level=AccessLevel(data.get("min_access_level", "limited")),
            source=data.get("source"),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
        )


# ============================================================================
# Built-in Briefings
# ============================================================================


BUILTIN_BRIEFINGS: dict[str, BriefingContent] = {
    "new_driver_protocol": BriefingContent(
        briefing_id="new_driver_protocol",
        title="New Driver Safety Protocol",
        summary="Essential safety guidelines for new drivers",
        content="""New Driver Safety Guidelines:

BEFORE DRIVING:
- Adjust mirrors and seat before starting
- Minimize distractions (phone on Do Not Disturb)
- Plan your route before departing
- Check fuel, lights, and tire pressure weekly

WHILE DRIVING:
- Keep both hands on the wheel
- Maintain safe following distance (3-4 seconds)
- Check mirrors every 5-8 seconds
- Signal before changing lanes or turning
- Never use phone while driving

IF PULLED OVER:
- Pull over safely to the right
- Turn off engine, roll down window
- Keep hands visible on steering wheel
- Wait for officer to approach
- Be polite, provide license/registration when asked
- You have the right to remain silent beyond ID

IF IN AN ACCIDENT:
- Check for injuries first
- Move to safety if possible
- Call 911 if anyone is hurt
- Exchange information with other driver
- Take photos of damage and scene
- Don't admit fault

PASSENGER RULES:
- All passengers must wear seatbelts
- Limit the number of teen passengers
- No loud music or disruptive behavior
- Focus on the road, not conversations""",
        category=BriefingCategory.DRIVING,
        tags=["driving", "new driver", "safety", "traffic"],
        age_appropriate=AgeAppropriateness.TEEN,
        source="Safe driving best practices",
    ),

    "first_job_rights": BriefingContent(
        briefing_id="first_job_rights",
        title="First Job Workplace Rights",
        summary="Know your rights as a new employee",
        content="""Your Rights as a New Employee:

BASIC RIGHTS:
- Minimum wage (check your state's rate)
- Overtime pay for hours over 40/week
- Safe working conditions
- Breaks (requirements vary by state)
- Protection from discrimination

PAY:
- You must be paid at least minimum wage
- Overtime is 1.5x regular rate for hours over 40/week
- Tips cannot be deducted from your wages
- Employer must provide pay stubs
- You can request your time records

SCHEDULING:
- Review your schedule in advance
- Document any hours you work
- Note if asked to work off the clock (this is illegal)
- Know your state's break requirements

SAFETY:
- You can refuse clearly dangerous work
- Report safety violations without retaliation
- Employer must provide safety training
- You have right to know about workplace hazards

IF THERE'S A PROBLEM:
- Document incidents with dates and witnesses
- Report to supervisor or HR
- If unresolved, contact Department of Labor
- Retaliation for complaints is illegal

WHAT TO KEEP:
- Copy of your W-4 and I-9
- Pay stubs
- Any employment documents
- Record of hours worked""",
        category=BriefingCategory.EMPLOYMENT,
        tags=["job", "employment", "rights", "workplace", "first job"],
        age_appropriate=AgeAppropriateness.TEEN,
        source="Department of Labor, FLSA",
    ),

    "dating_safety": BriefingContent(
        briefing_id="dating_safety",
        title="Safe Dating Practices",
        summary="Stay safe while meeting new people",
        content="""Safe Dating Guidelines:

BEFORE MEETING:
- Video chat first to verify identity
- Research the person (social media, mutual friends)
- Don't share personal details (home address, schedule)
- Tell a friend who you're meeting

FIRST MEETINGS:
- Always meet in a public place
- During daytime if possible
- Drive yourself or use your own transportation
- Tell someone where you're going and when
- Arrange a check-in text or call

DURING THE DATE:
- Stay in public areas
- Don't leave drinks unattended
- Trust your instincts
- It's okay to leave if uncomfortable
- Limit alcohol consumption

RED FLAGS:
- Pressure to leave public areas
- Won't take no for an answer
- Gets angry at boundaries
- Tries to isolate you from friends
- Demands too much too fast

AFTER:
- Text your friend that you're safe
- You don't owe a second date
- Block if they won't respect boundaries
- Trust your gut feeling

ONLINE SAFETY:
- Don't share explicit photos
- Be wary of profiles with few photos/friends
- Watch for catfishing signs
- Never send money to online contacts""",
        category=BriefingCategory.DATING,
        tags=["dating", "safety", "online dating", "relationships"],
        age_appropriate=AgeAppropriateness.TEEN,
        source="Personal safety best practices",
    ),

    "contract_basics": BriefingContent(
        briefing_id="contract_basics",
        title="Understanding Contracts",
        summary="What to know before signing anything",
        content="""Contract Basics for Young Adults:

WHAT IS A CONTRACT:
- A legally binding agreement
- Can be written or verbal (written is safer)
- Creates obligations on both sides
- Breaking it can have consequences

BEFORE SIGNING ANYTHING:
- Read the ENTIRE document
- Ask questions about anything unclear
- Take time - don't sign under pressure
- Get a copy for your records

KEY THINGS TO LOOK FOR:
- Length of agreement
- Payment terms
- Cancellation policy
- Automatic renewal clauses
- Penalties and fees
- What you're agreeing to do

COMMON CONTRACTS YOU'LL SEE:
- Employment agreements
- Apartment leases
- Cell phone contracts
- Gym memberships
- Subscription services
- Car financing

WARNING SIGNS:
- Pressure to sign immediately
- "Don't worry about reading it"
- Complex language with no explanation
- Empty blanks that could be filled later
- No copy provided

YOU CAN:
- Ask for time to review
- Ask for clarification
- Negotiate terms
- Walk away
- Ask a trusted adult for advice

WHEN TO GET HELP:
- Large financial commitments
- Long-term obligations (1+ years)
- Anything involving property
- If you don't understand something""",
        category=BriefingCategory.LEGAL,
        tags=["contracts", "legal", "signing", "agreements"],
        age_appropriate=AgeAppropriateness.TEEN,
        source="Consumer protection guidance",
    ),

    "online_privacy": BriefingContent(
        briefing_id="online_privacy",
        title="Protecting Your Online Privacy",
        summary="Keep your personal information safe online",
        content="""Online Privacy Protection:

INFORMATION TO PROTECT:
- Full name (use nickname in casual settings)
- Home address and school
- Phone number
- Photos of your home/school
- Daily schedule
- Passwords

SOCIAL MEDIA:
- Review privacy settings regularly
- Don't post real-time location
- Be careful with location-tagged photos
- Limit who can see your posts
- Be selective about friend requests

PASSWORDS:
- Use different passwords for each account
- Make them long (12+ characters)
- Use a password manager
- Enable two-factor authentication
- Never share passwords

WHAT NOT TO SHARE:
- Vacation plans (post after you return)
- Explicit photos (never)
- Screenshots of IDs or cards
- Answers to security questions
- Private conversations

IF SOMETHING GOES WRONG:
- Change passwords immediately
- Tell a trusted adult
- Report to the platform
- Document what happened

REMEMBER:
- Nothing online is truly private
- Screenshots can be taken
- Things posted can resurface later
- Online reputation matters""",
        category=BriefingCategory.DIGITAL,
        tags=["privacy", "online", "social media", "digital safety"],
        age_appropriate=AgeAppropriateness.TEEN,
        source="Digital safety best practices",
    ),
}


# ============================================================================
# Briefing System
# ============================================================================


class CustomBriefingSystem:
    """
    System for managing and delivering custom briefings to users.
    """

    def __init__(self, config: UserManagerConfig | None = None):
        self.config = config or UserManagerConfig()

        # Briefing content storage
        self._briefings: dict[str, BriefingContent] = {}

        # Load built-in briefings
        self._briefings.update(BUILTIN_BRIEFINGS)

    async def load_custom_briefings(self, path: Path | None = None) -> int:
        """
        Load additional briefings from a directory.

        Args:
            path: Path to briefings directory

        Returns:
            Number of briefings loaded
        """
        if path is None:
            path = self.config.data_path / "briefings"

        if not path.exists():
            return 0

        loaded = 0

        for json_file in path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        briefing = BriefingContent.from_dict(item)
                        self._briefings[briefing.briefing_id] = briefing
                        loaded += 1
                elif isinstance(data, dict):
                    briefing = BriefingContent.from_dict(data)
                    self._briefings[briefing.briefing_id] = briefing
                    loaded += 1

            except Exception as e:
                logger.warning(f"Failed to load briefing {json_file}: {e}")

        logger.info(f"Loaded {loaded} custom briefings")
        return loaded

    def get_briefing(self, briefing_id: str) -> BriefingContent | None:
        """Get a specific briefing by ID."""
        return self._briefings.get(briefing_id)

    def get_user_briefings(self, user: User) -> list[BriefingContent]:
        """
        Get all briefings assigned to a user.

        Args:
            user: The user

        Returns:
            List of accessible briefings for the user
        """
        briefings = []

        for briefing_id in user.custom_briefings:
            briefing = self._briefings.get(briefing_id)
            if briefing and briefing.is_accessible_by(user):
                briefings.append(briefing)

        return briefings

    def get_briefings_by_category(
        self,
        category: BriefingCategory,
        user: User | None = None,
    ) -> list[BriefingContent]:
        """
        Get briefings by category, optionally filtered for user access.

        Args:
            category: Briefing category
            user: Optional user for access filtering

        Returns:
            List of briefings
        """
        briefings = [
            b for b in self._briefings.values()
            if b.category == category
        ]

        if user:
            briefings = [b for b in briefings if b.is_accessible_by(user)]

        return briefings

    def get_age_appropriate_briefings(
        self,
        user: User,
    ) -> list[BriefingContent]:
        """
        Get all age-appropriate briefings for a user.

        Args:
            user: The user

        Returns:
            List of accessible briefings
        """
        return [
            b for b in self._briefings.values()
            if b.is_accessible_by(user)
        ]

    def search_briefings(
        self,
        query: str,
        user: User | None = None,
    ) -> list[BriefingContent]:
        """
        Search briefings by keyword.

        Args:
            query: Search query
            user: Optional user for access filtering

        Returns:
            List of matching briefings
        """
        query_lower = query.lower()
        results = []

        for briefing in self._briefings.values():
            if user and not briefing.is_accessible_by(user):
                continue

            # Search in title, content, tags
            if (
                query_lower in briefing.title.lower()
                or query_lower in briefing.content.lower()
                or any(query_lower in tag.lower() for tag in briefing.tags)
            ):
                results.append(briefing)

        return results

    def add_briefing(self, briefing: BriefingContent) -> None:
        """Add a briefing to the system."""
        self._briefings[briefing.briefing_id] = briefing

    def get_recommended_briefings(self, user: User) -> list[str]:
        """
        Get recommended briefing IDs for a user based on their role.

        Args:
            user: The user

        Returns:
            List of recommended briefing IDs
        """
        recommendations = []

        # Recommendations for minors/teens
        if user.role == UserRole.MINOR:
            recommendations.extend([
                "online_privacy",
                "dating_safety",
            ])

        # Check existing briefings
        existing = set(user.custom_briefings)

        # Add relevant ones not already assigned
        for rec in recommendations:
            if rec not in existing and rec in self._briefings:
                existing.add(rec)

        return list(existing)

    def get_stats(self) -> dict[str, Any]:
        """Get briefing system statistics."""
        categories = {}
        for briefing in self._briefings.values():
            cat = briefing.category.value
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_briefings": len(self._briefings),
            "builtin_briefings": len(BUILTIN_BRIEFINGS),
            "categories": categories,
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_briefing_system(
    config: UserManagerConfig | None = None,
) -> CustomBriefingSystem:
    """Create a custom briefing system."""
    return CustomBriefingSystem(config)
