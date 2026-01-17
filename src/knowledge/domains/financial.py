"""
SAIL Financial Protection Knowledge Domain

Provides financial safety knowledge including:
- Scam detection patterns
- Phishing identification
- Social engineering recognition
- Wire fraud warnings
- Identity theft protection
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.knowledge.base import (
    KnowledgeDomain,
    KnowledgeDomainType,
    KnowledgeCategory,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeRelevance,
    Jurisdiction,
    JurisdictionLevel,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Scam Detection Knowledge
# ============================================================================


SCAM_PATTERNS = {
    "urgency_scam": KnowledgeItem(
        item_id="financial-scam-urgency",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.SCAM_DETECTION,
        title="Urgency and Pressure Scam Tactics",
        content="""Recognizing urgency and pressure scam tactics:

COMMON URGENCY PHRASES:
- "Act now or lose this opportunity"
- "This offer expires in 24 hours"
- "Immediate action required"
- "Your account will be suspended"
- "Final notice before legal action"
- "Limited time offer - today only"
- "Don't tell anyone about this opportunity"

WHY SCAMMERS USE URGENCY:
- Prevents you from thinking clearly
- Stops you from consulting others
- Creates fear of missing out (FOMO)
- Bypasses your normal decision-making process

RED FLAGS:
- Any request to act immediately
- Pressure not to consult family, friends, or advisors
- Threats of negative consequences for delay
- "Secret" opportunities you can't discuss
- Countdown timers or expiration threats

THE RULE: LEGITIMATE OPPORTUNITIES DON'T DISAPPEAR:
- Real opportunities allow time for due diligence
- Legitimate companies don't pressure you to decide instantly
- Government agencies send official letters, not urgent calls
- Real emergencies from family can be verified

WHAT TO DO:
- Say: "I need time to think about this"
- Tell them: "I need to discuss this with my family/advisor first"
- Hang up and call back using a number you find independently
- Never make major financial decisions under pressure
- If truly urgent, take time to verify through official channels""",
        summary="Scammers create false urgency. Legitimate opportunities allow time to think. Never make instant decisions under pressure.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["urgency", "pressure", "scam", "act now", "limited time", "immediate"],
        source="FTC, FBI IC3",
        is_verified=True,
    ),
    "impersonation_scam": KnowledgeItem(
        item_id="financial-scam-impersonation",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.SCAM_DETECTION,
        title="Government and Authority Impersonation Scams",
        content="""Recognizing impersonation scams:

COMMON IMPERSONATIONS:
- IRS/Tax agency
- Social Security Administration
- FBI/Police/Law enforcement
- Court/Legal system (jury duty scams)
- Utility companies
- Tech support (Microsoft, Apple)
- Bank fraud department
- Amazon/Major retailers

RED FLAGS:
- Calling to demand immediate payment
- Threatening arrest, deportation, or license suspension
- Requesting payment via gift cards, wire transfer, or cryptocurrency
- Asking for Social Security numbers over the phone
- Claiming you owe money you didn't know about
- Caller ID showing official-looking number (can be spoofed)

WHAT REAL AGENCIES DO:
- IRS: Sends letters first, never demands gift cards
- SSA: Never threatens to suspend your number
- Police: Don't call to collect fines over the phone
- Courts: Send official mail for jury duty
- Banks: Don't ask for full passwords or PINs by phone

VERIFICATION STEPS:
1. Hang up the call
2. Find the official number independently (website, card, statement)
3. Call that number to verify
4. Never call back the number they give you
5. Never click links in suspicious emails/texts

WHAT TO SAY:
- "I'll call you back at your official number"
- "Please send me this information in writing"
- "I need to verify this with [agency] directly"

REPORT IMPERSONATION:
- FTC: ReportFraud.ftc.gov
- IRS impersonation: TIGTA.gov
- SSA impersonation: oig.ssa.gov""",
        summary="Government agencies don't call demanding gift cards or threatening arrest. Always verify by calling official numbers you find yourself.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["impersonation", "irs", "government", "scam", "police", "social security", "tech support"],
        source="FTC, IRS, SSA official warnings",
        is_verified=True,
    ),
    "romance_scam": KnowledgeItem(
        item_id="financial-scam-romance",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.SCAM_DETECTION,
        title="Romance Scam Detection",
        content="""Recognizing romance scams:

TYPICAL SCAMMER PROFILE:
- Claims to be overseas (military, oil rig, international business)
- Attractive photos (often stolen from models/influencers)
- Falls in love very quickly
- Always has reasons they can't video chat
- Story involves wealth or prestigious career

THE SCAM PROGRESSION:
1. Make contact on dating site or social media
2. Move to private messaging quickly
3. Build intense emotional connection
4. Share fabricated life story with travel or hardship
5. Eventually ask for money for "emergency"

COMMON MONEY REQUESTS:
- Travel costs to come visit you
- Medical emergency
- Business investment opportunity
- Customs fees to release inheritance/package
- Bail money or legal fees
- Money to buy a phone/computer

RED FLAGS:
- Never able to meet in person
- Always has excuse for video chat
- Declares love very quickly
- Asks for money for any reason
- Gets defensive when questioned
- Story has inconsistencies
- Claims to be wealthy but needs your money

VERIFICATION:
- Reverse image search their photos
- Ask questions about their claimed profession/location
- Insist on video chat
- Search their story elements online (common scam scripts)

THE RULE:
- NEVER send money to someone you haven't met in person
- Real romantic partners don't need you to wire money
- Love doesn't require financial transactions

IF VICTIMIZED:
- Stop all contact
- Report to the platform
- Report to FTC and IC3
- Don't feel ashamed - these are sophisticated criminals""",
        summary="Never send money to someone you've only met online. Real love doesn't require financial transactions. Verify via video chat.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["romance scam", "dating", "love scam", "online dating", "catfish", "money"],
        source="FTC, FBI IC3",
        is_verified=True,
    ),
    "grandparent_scam": KnowledgeItem(
        item_id="financial-scam-grandparent",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.SCAM_DETECTION,
        title="Grandparent and Family Emergency Scams",
        content="""Recognizing family emergency scams:

THE SCRIPT:
- Caller claims to be grandchild, niece, or nephew
- Says they're in trouble (arrested, accident, stranded)
- Urgently needs money for bail, medical bills, or travel
- Asks you not to tell other family members
- May hand phone to "lawyer" or "officer" for legitimacy

KEY PHRASES:
- "Grandma/Grandpa, it's me" (waits for you to say a name)
- "Don't tell mom and dad"
- "I'm in jail and need bail money"
- "I was in a car accident in [foreign country]"
- "I need money wired right away"

RED FLAGS:
- They want you to guess who's calling
- Pressure to keep it secret from other family
- Request for gift cards, wire transfer, or cash
- Unusual payment methods
- Voice doesn't quite sound right (they'll claim bad connection)
- They can't answer personal questions correctly

VERIFICATION STEPS:
1. Don't say any names - make them identify themselves first
2. Ask a question only the real person would know
3. Say "Let me call you back" and hang up
4. Call the family member directly at their known number
5. Call another family member to verify

WHAT TO SAY:
- "What's my pet's name?"
- "What did we do for your last birthday?"
- "Let me call you back at your number"
- "I need to verify this with [other family member]"

PROTECT FAMILY:
- Establish a family code word for emergencies
- Tell older relatives about this scam
- Make sure family knows to verify before sending money""",
        summary="If a 'grandchild' calls needing urgent money, hang up and call them directly at their known number. Use family code words.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["grandparent scam", "family emergency", "bail", "arrested", "wire money", "grandchild"],
        source="FTC, FBI IC3",
        is_verified=True,
    ),
}


# ============================================================================
# Phishing Detection
# ============================================================================


PHISHING_PATTERNS = {
    "email_phishing": KnowledgeItem(
        item_id="financial-phishing-email",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.PHISHING,
        title="Email Phishing Detection",
        content="""Detecting phishing emails:

CHECK THE SENDER:
- Hover over sender name to see actual email address
- Watch for misspellings: "amaz0n.com", "paypa1.com"
- Legitimate companies use their actual domain
- Be suspicious of personal email addresses for business matters

CHECK THE CONTENT:
- Generic greetings ("Dear Customer" instead of your name)
- Spelling and grammar errors
- Threats or urgent language
- Requests for sensitive information
- Claims about account problems you didn't know about

CHECK THE LINKS:
- Hover over links to see actual URL (don't click)
- Watch for misspelled domains
- Look for extra words or numbers in URLs
- Legitimate sites use HTTPS, but so can fake ones

COMMON PHISHING SUBJECTS:
- "Your account has been compromised"
- "Verify your information"
- "Package delivery failed"
- "You've won a prize"
- "Invoice attached"
- "Payment problem"
- "Reset your password" (that you didn't request)

WHAT LEGITIMATE COMPANIES DON'T DO:
- Ask for passwords via email
- Ask for full Social Security numbers
- Send links to enter login credentials
- Threaten immediate account closure
- Request payment via gift cards

SAFE PRACTICES:
- Don't click links in suspicious emails
- Go directly to the company's website by typing the URL
- Use bookmarks for frequently visited sites
- When in doubt, call the company using a number from their official website
- Enable two-factor authentication on important accounts""",
        summary="Check sender's actual email address, hover over links before clicking, go directly to websites instead of clicking links.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["phishing", "email", "scam", "link", "password", "account"],
        source="CISA, FTC",
        is_verified=True,
    ),
    "smishing": KnowledgeItem(
        item_id="financial-phishing-smishing",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.PHISHING,
        title="SMS/Text Message Phishing (Smishing)",
        content="""Detecting text message scams (smishing):

COMMON SMISHING MESSAGES:
- Package delivery notifications with links
- Bank fraud alerts asking you to call a number or click
- IRS or government texts (they don't text you)
- Prize or lottery winning notifications
- Two-factor codes you didn't request
- "Problem with your payment" messages

RED FLAGS:
- Unexpected texts from unknown numbers
- Shortened URLs that hide the real destination
- Urgency ("respond within 24 hours")
- Requests to click links or call numbers in the text
- Messages claiming to be from companies that have your real number

THE PACKAGE SCAM:
- Claims package couldn't be delivered
- Asks you to click a link to reschedule
- Link leads to fake site requesting personal info or payment
- Real carriers won't text you out of the blue

BANK TEXT SCAMS:
- Claims suspicious activity on your account
- Provides number to call or link to click
- Real banks don't send unsolicited texts asking you to verify via link

SAFE PRACTICES:
- Never click links in unexpected texts
- Don't call numbers provided in suspicious texts
- Look up official numbers and call those instead
- Delete suspicious texts
- Don't reply to unknown numbers (confirms your number is active)
- Report spam texts to 7726 (SPAM)

IF YOU CLICKED A LINK:
- Don't enter any information
- Close the browser
- Run a security scan on your device
- Monitor accounts for suspicious activity
- Change passwords if you entered any""",
        summary="Don't click links in unexpected texts. Real companies don't text demanding action. Look up official numbers independently.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["smishing", "text scam", "sms", "phishing", "package", "delivery"],
        source="FCC, CISA",
        is_verified=True,
    ),
}


# ============================================================================
# Wire Fraud and Payment Scams
# ============================================================================


WIRE_FRAUD = {
    "wire_transfer_warning": KnowledgeItem(
        item_id="financial-wire-transfer",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.WIRE_FRAUD,
        title="Wire Transfer Fraud Warning Signs",
        content="""Wire transfer fraud warning signs:

WIRE TRANSFERS ARE LIKE CASH:
- Once sent, nearly impossible to recover
- No buyer protection like credit cards
- Scammers' preferred payment method
- Legitimate businesses rarely require wire transfers

WHEN SCAMMERS REQUEST WIRE TRANSFERS:
- Rental listings (apartment scams)
- Romance scams
- Lottery/prize scams
- Government impersonation
- Tech support scams
- Grandparent scams
- Business email compromise

BUSINESS EMAIL COMPROMISE (BEC):
- Scammer hacks or spoofs business email
- Sends fake invoice with new wire instructions
- Claims "our bank information has changed"
- Targets businesses, real estate, law firms
- Often involves large sums

REAL ESTATE WIRE FRAUD:
- Scammer monitors real estate transactions
- Sends fake "closing instructions" by email
- Wire goes to scammer's account instead of title company
- ALWAYS verify wire instructions by phone

PROTECTION:
- Never wire money to someone you haven't met
- Verify any wire instructions by calling a known number
- Be suspicious of last-minute changes to payment info
- Call the recipient to confirm before large transfers
- Use payment methods with buyer protection when possible

IF YOU WIRED MONEY TO A SCAMMER:
- Contact your bank IMMEDIATELY
- Ask them to recall the wire
- Report to FBI IC3 (ic3.gov)
- Report to FTC
- Time is critical - act within hours if possible""",
        summary="Wire transfers are irreversible. Never wire money to strangers. Always verify wire instructions by phone using known numbers.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["wire transfer", "wire fraud", "payment", "bank transfer", "closing", "real estate"],
        source="FBI IC3, FinCEN",
        is_verified=True,
    ),
    "gift_card_scam": KnowledgeItem(
        item_id="financial-gift-card-scam",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.WIRE_FRAUD,
        title="Gift Card Payment Scams",
        content="""Gift card scams and why scammers love them:

WHY SCAMMERS USE GIFT CARDS:
- Easy to buy anonymously
- Hard to trace
- Impossible to reverse once codes are shared
- Victim can buy them anywhere
- No bank involvement to flag fraud

THE RULE:
- NO legitimate business or government accepts gift cards as payment
- ONLY scammers ask for gift cards

COMMON GIFT CARD SCAMS:
- IRS demanding payment in gift cards (NEVER real)
- Utility company threatening shutoff unless paid in gift cards
- Tech support wanting gift cards for "repairs"
- Romance scammer needing gift cards for "emergency"
- Boss/CEO email asking you to buy gift cards (email compromise)

THE "BOSS" SCAM:
- Email appears to be from your boss or CEO
- Asks you to buy gift cards for "client gifts" or "prizes"
- Asks you to send the card numbers via email
- Your boss didn't send it - verify by phone or in person

WHAT TO DO IF ASKED FOR GIFT CARDS:
- Stop immediately - it's a scam
- No matter how convincing the story
- No matter who they claim to be
- Real emergencies don't require gift cards

IF YOU ALREADY SENT GIFT CARD NUMBERS:
- Contact the gift card company immediately
- Some may be able to freeze unused funds
- Report to FTC at ReportFraud.ftc.gov
- Keep the cards and receipts as evidence""",
        summary="NO legitimate entity accepts gift cards as payment - only scammers. If anyone asks for gift cards, it's a scam.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["gift card", "scam", "payment", "itunes", "google play", "amazon"],
        source="FTC",
        is_verified=True,
    ),
}


# ============================================================================
# Social Engineering Patterns
# ============================================================================


SOCIAL_ENGINEERING = {
    "pressure_tactics": KnowledgeItem(
        item_id="financial-social-engineering",
        domain=KnowledgeDomainType.FINANCIAL,
        category=KnowledgeCategory.SOCIAL_ENGINEERING,
        title="Social Engineering Tactics Recognition",
        content="""Recognizing social engineering tactics:

WHAT IS SOCIAL ENGINEERING:
- Manipulating people into giving up information or money
- Exploits trust, fear, helpfulness, or authority
- Targets human psychology, not technology

COMMON TACTICS:

1. AUTHORITY:
- Claiming to be from the government, police, or your employer
- Using official-sounding titles and jargon
- Counter: Verify through official channels

2. URGENCY:
- Creating time pressure
- Threatening consequences for delay
- Counter: Take time, verify independently

3. RECIPROCITY:
- Offering something first to create obligation
- "I'm helping you, now you help me"
- Counter: You don't owe strangers favors

4. SOCIAL PROOF:
- "Everyone is doing this"
- "Your neighbors have already paid"
- Counter: Verify independently

5. LIKING:
- Being overly friendly or complimentary
- Finding common ground quickly
- Counter: Stay objective, don't let flattery lower guard

6. SCARCITY:
- "Limited time offer"
- "Only a few left"
- Counter: Real opportunities allow time

7. FEAR:
- Threatening arrest, fines, or harm
- Creating panic to bypass rational thinking
- Counter: Pause, verify, don't act in fear

PROTECTION STRATEGIES:
- Pause before acting on any request
- Verify through channels you find, not they provide
- Consult trusted family or friends
- Remember: it's okay to say no
- Legitimate requests allow verification time""",
        summary="Scammers use authority, urgency, fear, and flattery to manipulate. Always pause, verify independently, and consult others.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["social engineering", "manipulation", "tactics", "psychology", "scam"],
        source="Social engineering security research",
        is_verified=True,
    ),
}


# ============================================================================
# Scam Pattern Detection
# ============================================================================


@dataclass
class ScamIndicator:
    """An indicator that a situation may be a scam."""

    indicator: str
    weight: float  # 0-1 importance
    category: str


# Common scam indicators for real-time detection
SCAM_INDICATORS = [
    ScamIndicator("urgency", 0.8, "pressure"),
    ScamIndicator("act now", 0.8, "pressure"),
    ScamIndicator("limited time", 0.7, "pressure"),
    ScamIndicator("don't tell anyone", 0.9, "secrecy"),
    ScamIndicator("keep this between us", 0.9, "secrecy"),
    ScamIndicator("gift card", 0.95, "payment"),
    ScamIndicator("wire transfer", 0.8, "payment"),
    ScamIndicator("bitcoin", 0.7, "payment"),
    ScamIndicator("cryptocurrency", 0.7, "payment"),
    ScamIndicator("account suspended", 0.8, "threat"),
    ScamIndicator("arrest warrant", 0.9, "threat"),
    ScamIndicator("legal action", 0.7, "threat"),
    ScamIndicator("verify your information", 0.7, "phishing"),
    ScamIndicator("click this link", 0.6, "phishing"),
    ScamIndicator("won a prize", 0.8, "lottery"),
    ScamIndicator("lottery winner", 0.9, "lottery"),
    ScamIndicator("inheritance from overseas", 0.95, "advance_fee"),
    ScamIndicator("nigerian prince", 0.99, "advance_fee"),
]


def analyze_scam_risk(text: str) -> tuple[float, list[str]]:
    """
    Analyze text for scam indicators.

    Returns:
        Tuple of (risk_score 0-1, list of detected indicators)
    """
    text_lower = text.lower()
    detected = []
    total_weight = 0.0

    for indicator in SCAM_INDICATORS:
        if indicator.indicator in text_lower:
            detected.append(indicator.indicator)
            total_weight += indicator.weight

    # Normalize score (cap at 1.0)
    risk_score = min(total_weight / 2.0, 1.0)  # Divide by 2 so it takes multiple indicators to max out

    return risk_score, detected


# ============================================================================
# Financial Knowledge Domain Class
# ============================================================================


class FinancialKnowledgeDomain(KnowledgeDomain):
    """
    Financial protection knowledge domain.
    """

    def __init__(self, data_path: Optional[Path] = None):
        super().__init__(KnowledgeDomainType.FINANCIAL, data_path)

    async def load(self) -> bool:
        """Load all financial knowledge items."""
        try:
            # Load scam patterns
            for item in SCAM_PATTERNS.values():
                self.add_item(item)

            # Load phishing patterns
            for item in PHISHING_PATTERNS.values():
                self.add_item(item)

            # Load wire fraud knowledge
            for item in WIRE_FRAUD.values():
                self.add_item(item)

            # Load social engineering
            for item in SOCIAL_ENGINEERING.values():
                self.add_item(item)

            # Try to load additional items from data files
            await self._load_from_files()

            self._loaded = True
            logger.info(f"Loaded {self.item_count} financial knowledge items")
            return True

        except Exception as e:
            logger.error(f"Failed to load financial knowledge: {e}")
            return False

    async def _load_from_files(self) -> None:
        """Load additional knowledge from JSON files."""
        if not self.data_path.exists():
            return

        for json_file in self.data_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item_data in data:
                        item = KnowledgeItem.from_dict(item_data)
                        self.add_item(item)

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    async def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        """Query the financial knowledge domain."""
        start_time = time.time()

        candidates = list(self._items.values())
        if query.category:
            candidates = [item for item in candidates if item.category == query.category]

        query_words = query.query_text.lower().split()
        scored_items: list[tuple[KnowledgeItem, float]] = []

        for item in candidates:
            score = self._calculate_relevance_score(item, query_words, query)
            if score > 0:
                scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        top_items = scored_items[:query.max_results]
        items = [item for item, _ in top_items]
        scores = [score for _, score in top_items]

        result = KnowledgeResult(
            query_id=query.query_id,
            items=items,
            scores=scores,
            total_found=len(scored_items),
            domain_searched=self.domain_type,
            jurisdiction_filtered=False,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            sources=[item.source for item in items if item.source],
        )

        return result

    def _calculate_relevance_score(
        self,
        item: KnowledgeItem,
        query_words: list[str],
        query: KnowledgeQuery,
    ) -> float:
        """Calculate relevance score for an item."""
        score = 0.0
        item_text = f"{item.title} {item.content} {item.summary} {' '.join(item.keywords)}".lower()

        for word in query_words:
            if word in item_text:
                score += 1.0
            if word in [k.lower() for k in item.keywords]:
                score += 2.0

        title_lower = item.title.lower()
        for word in query_words:
            if word in title_lower:
                score += 1.5

        relevance_boosts = {
            KnowledgeRelevance.CRITICAL: 3.0,
            KnowledgeRelevance.HIGH: 2.0,
            KnowledgeRelevance.MEDIUM: 1.0,
            KnowledgeRelevance.LOW: 0.5,
            KnowledgeRelevance.INFORMATIONAL: 0.25,
        }
        score *= relevance_boosts.get(item.relevance, 1.0)

        return score

    def analyze_for_scams(self, text: str) -> tuple[float, list[str], list[KnowledgeItem]]:
        """
        Analyze text for potential scam indicators.

        Args:
            text: Text to analyze (could be email, message, call transcript)

        Returns:
            Tuple of (risk_score, detected_indicators, relevant_knowledge_items)
        """
        risk_score, indicators = analyze_scam_risk(text)

        # Get relevant knowledge based on detected indicators
        relevant_items: list[KnowledgeItem] = []
        if indicators:
            for item in self._items.values():
                for indicator in indicators:
                    if indicator in item.keywords or indicator in item.content.lower():
                        if item not in relevant_items:
                            relevant_items.append(item)
                        break

        return risk_score, indicators, relevant_items[:5]


# ============================================================================
# Factory Function
# ============================================================================


def create_financial_domain(data_path: Optional[Path] = None) -> FinancialKnowledgeDomain:
    """Create a financial knowledge domain instance."""
    return FinancialKnowledgeDomain(data_path)
