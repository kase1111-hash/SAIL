"""
SAIL Safety Knowledge Domain

Provides personal safety knowledge including:
- De-escalation techniques and scripts
- Situational awareness guidelines
- Safe meeting protocols
- Online safety and verification
- Personal security best practices
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.knowledge.base import (
    JurisdictionLevel,
    KnowledgeCategory,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeRelevance,
)
from src.knowledge.domains.base_domain import BaseKnowledgeDomain

logger = logging.getLogger(__name__)


# ============================================================================
# De-escalation Knowledge
# ============================================================================


DE_ESCALATION_TECHNIQUES = {
    "verbal_de_escalation": KnowledgeItem(
        item_id="safety-de-escalation-verbal",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.DE_ESCALATION,
        title="Verbal De-escalation Techniques",
        content="""Verbal de-escalation techniques to calm tense situations:

VOICE AND TONE:
- Speak slowly and calmly
- Keep your voice low and steady
- Avoid raising your voice even if they do
- Use a respectful, non-judgmental tone

WORD CHOICE:
- Use "I" statements instead of "you" accusations
- "I understand this is frustrating" not "You need to calm down"
- Avoid commands like "calm down" which can escalate
- Use the person's name if you know it

ACTIVE LISTENING:
- Let them speak without interruption
- Acknowledge their feelings: "I can see you're upset"
- Paraphrase to show understanding: "So what you're saying is..."
- Ask open-ended questions to understand their perspective

EMPATHY PHRASES:
- "I understand why you feel that way"
- "That sounds really difficult"
- "Help me understand what happened"
- "I want to help resolve this"

THINGS TO AVOID:
- Do NOT say "calm down" or "relax"
- Do NOT argue facts during emotional peaks
- Do NOT use sarcasm or condescension
- Do NOT make threats or ultimatums
- Do NOT take insults personally

EXIT STRATEGY:
- If de-escalation isn't working, disengage safely
- Say: "I want to help, but I need to step away for a moment"
- Create physical distance if possible
- Have an exit route in mind""",
        summary="Stay calm, speak slowly, use 'I' statements, acknowledge feelings, avoid saying 'calm down', have exit strategy.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["de-escalation", "calm", "angry", "conflict", "verbal", "argument", "tense"],
        source="Crisis Intervention Training principles",
        is_verified=True,
    ),
    "physical_de_escalation": KnowledgeItem(
        item_id="safety-de-escalation-physical",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.DE_ESCALATION,
        title="Physical De-escalation and Body Language",
        content="""Non-verbal de-escalation techniques:

BODY POSITIONING:
- Stand at an angle, not directly face-to-face (less confrontational)
- Maintain 1.5-2 arm lengths distance
- Keep your hands visible and relaxed at your sides
- Don't cross your arms (appears defensive/closed)
- Keep feet shoulder-width apart for stability

FACIAL EXPRESSION:
- Maintain neutral, calm expression
- Make appropriate eye contact (not staring)
- Occasional nods to show listening
- Avoid expressions of contempt, eye-rolling, or smirking

AVOID THESE SIGNALS:
- Pointing fingers (aggressive)
- Hands on hips (authority/challenge)
- Invading personal space
- Blocking exits or cornering
- Sudden movements

ENVIRONMENT:
- Move to a quieter area if possible
- Remove audience/bystanders if you can
- Consider barriers (desk, counter) for safety
- Know your exits

WARNING SIGNS TO WATCH:
- Clenched fists
- Reddening face
- Loud, fast speech
- Pacing or inability to stay still
- Thousand-yard stare
- Target glancing (looking at potential weapons)

IF VIOLENCE SEEMS IMMINENT:
- Create distance immediately
- Put barriers between you
- Call for help
- Leave if safe to do so
- Your safety is priority""",
        summary="Angle your body, maintain distance, keep hands visible, watch for warning signs like clenched fists, know your exits.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["de-escalation", "body language", "physical", "warning signs", "violence", "safety"],
        source="Crisis Intervention Training principles",
        is_verified=True,
    ),
    "road_rage": KnowledgeItem(
        item_id="safety-de-escalation-road-rage",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.DE_ESCALATION,
        title="Handling Road Rage Encounters",
        content="""How to handle road rage situations:

IF YOU MADE A DRIVING MISTAKE:
- Briefly wave or mouth "sorry" to acknowledge
- Do not engage further
- Do not make eye contact
- Continue driving normally

IF SOMEONE IS AGGRESSIVE TOWARD YOU:
- DO NOT engage or respond to gestures
- DO NOT make eye contact
- DO NOT return gestures or yell back
- Keep your doors locked
- Leave space between cars to maneuver

IF BEING FOLLOWED:
- Do NOT drive home
- Drive to a police station, fire station, or busy public place
- Call 911 while driving (use hands-free)
- Stay in your vehicle with doors locked
- Honk horn to draw attention if needed

IF CONFRONTED:
- Stay in your vehicle
- Keep windows up and doors locked
- Call 911
- If they approach your vehicle, do NOT exit
- If they try to block you, carefully drive away if safe
- Use your horn

PREVENTION:
- Don't tailgate
- Use turn signals
- Let aggressive drivers pass
- Don't compete for lane position
- Give yourself extra time to reduce your own stress
- Apologize for mistakes with a wave

DOCUMENT:
- Note license plate, vehicle description
- Note time and location
- Report to police if seriously threatened""",
        summary="Do not engage or make eye contact. Stay in vehicle with doors locked. If followed, drive to police station. Call 911.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["road rage", "driving", "aggressive driver", "followed", "car", "traffic"],
        source="AAA Foundation for Traffic Safety, Police guidance",
        is_verified=True,
    ),
}


# ============================================================================
# Situational Awareness Knowledge
# ============================================================================


SITUATIONAL_AWARENESS = {
    "general_awareness": KnowledgeItem(
        item_id="safety-awareness-general",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.SITUATIONAL_AWARENESS,
        title="General Situational Awareness",
        content="""Situational awareness fundamentals:

COOPER'S COLOR CODE:
- WHITE: Unaware, relaxed (avoid in public)
- YELLOW: Relaxed alert, general awareness (default in public)
- ORANGE: Specific alert, potential threat identified
- RED: Ready to act, threat confirmed

WHAT TO NOTICE:
- Exits and entrances
- People who seem out of place
- Hands - what are people carrying/holding?
- Vehicles that don't fit the area
- Anyone paying unusual attention to you
- Changes in ambient noise level

THE OODA LOOP:
1. Observe: What's happening around you?
2. Orient: What does it mean?
3. Decide: What should you do?
4. Act: Execute your decision

AVOIDING "VICTIM SELECTION":
- Walk with purpose and confidence
- Keep your head up, scan the area
- Don't appear distracted (phone, headphones)
- Stay in well-lit, populated areas
- Trust your instincts - if something feels wrong, leave

IN PUBLIC SPACES:
- Sit facing the door in restaurants
- Know where exits are when you enter
- Notice groups forming or dispersing
- Be aware of choke points

WHEN SOMETHING FEELS WRONG:
- Trust your gut instinct
- It's okay to leave or change direction
- Cross the street, enter a store, change your route
- Better to be wrong and safe than polite and sorry""",
        summary="Stay in 'yellow' alert mode in public. Notice exits, hands, unusual attention. Trust your instincts.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["situational awareness", "safety", "alert", "aware", "danger", "threat"],
        source="Personal security training principles",
        is_verified=True,
    ),
    "late_night_safety": KnowledgeItem(
        item_id="safety-awareness-late-night",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.SITUATIONAL_AWARENESS,
        title="Late Night Safety Guidelines",
        content="""Safety guidelines for late night activities:

WALKING AT NIGHT:
- Stay in well-lit areas on main streets
- Walk facing traffic so cars can't approach from behind
- Keep one earbud out or volume low to hear surroundings
- Have your phone ready but not constantly looking at it
- Walk with confidence and purpose
- Vary your route if walking regularly

IF WALKING ALONE:
- Tell someone your route and expected arrival time
- Consider a personal safety app that tracks your location
- Keep keys ready (but not as a weapon - can hurt your hand)
- Be aware of people loitering or following

PARKING AT NIGHT:
- Park in well-lit areas near entrances
- Check around and under your car before approaching
- Have keys ready before reaching the car
- Lock doors immediately upon entering
- Check backseat before getting in

PUBLIC TRANSIT:
- Wait in well-lit areas with other people
- Sit near the driver or in a car with others
- Stay awake and alert
- Know your stop and be ready to exit
- Have someone meet you at your stop if possible

RIDE SHARE:
- Verify the car make, model, color, and license plate
- Verify driver's name and photo
- Share your trip with a friend
- Sit in the back seat
- If something feels wrong, ask to be let out in a safe area

GENERAL:
- Keep your phone charged
- Know numbers to call (local non-emergency police)
- Consider personal alarm/whistle
- Tell someone when you arrive safely""",
        summary="Stay in well-lit areas, tell someone your plans, verify ride shares, keep phone charged, trust instincts.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["night", "late", "dark", "walking", "safety", "alone", "ride share"],
        source="Personal security best practices",
        is_verified=True,
    ),
}


# ============================================================================
# Safe Meeting Protocols
# ============================================================================


SAFE_MEETING = {
    "online_date": KnowledgeItem(
        item_id="safety-meeting-online-date",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.SAFE_MEETING,
        title="Safe Meeting Protocol for Online Dating",
        content="""Safety protocol for meeting someone from online dating:

BEFORE MEETING:
- Video chat first to verify they are who they say they are
- Google their name, check social media
- Do a reverse image search on their photos
- Get their full name and phone number
- Trust your instincts - if something feels off, cancel

FIRST MEETING:
- Meet in a public place (coffee shop, restaurant)
- During daylight hours if possible
- Tell a friend where you're going and who you're meeting
- Share their photo and contact info with your friend
- Arrange a check-in call or text at a specific time
- Have a "safe word" to text if you need help

TRANSPORTATION:
- Drive yourself or use your own ride share
- Do NOT let them pick you up from home
- Do NOT go to their home
- Park where you can leave quickly

DURING THE DATE:
- Stay in public areas
- Do not leave your drink unattended
- Limit alcohol consumption
- Watch for red flags:
  - Pressuring you to leave with them
  - Getting too personal too fast
  - Anger when you set boundaries
  - Not respecting "no"

LEAVING:
- Leave when you planned to
- Trust your gut if you feel uncomfortable
- You do NOT owe them another date or explanation
- Walk to your car in public view
- Make sure you're not being followed

AFTER:
- Text your friend that you're safe
- If they showed concerning behavior, block them
- Report serious concerns to the platform""",
        summary="Video chat first, meet in public, tell a friend, drive yourself, don't leave drinks unattended, trust your gut.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["dating", "online", "meeting", "first date", "safety", "stranger"],
        source="Personal safety best practices",
        is_verified=True,
    ),
    "marketplace_transaction": KnowledgeItem(
        item_id="safety-meeting-marketplace",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.SAFE_MEETING,
        title="Safe Meeting for Buying/Selling Items",
        content="""Safety protocol for meeting to buy/sell items (Craigslist, Facebook Marketplace, etc.):

COMMUNICATION:
- Keep communications on the platform initially
- Don't share personal details (address, work, schedule)
- Be wary of overly eager buyers or too-good prices
- Trust your instincts - if communication feels off, cancel

MEETING LOCATION:
- Meet at a police station "safe exchange zone" (many have them)
- Or meet at a busy public place (shopping center, coffee shop)
- During daylight hours
- NEVER meet at your home or their home
- NEVER meet in isolated locations

TELL SOMEONE:
- Tell a friend or family member:
  - Where you're meeting
  - What time
  - Who you're meeting (name, phone, description)
  - What you're buying/selling
- Arrange a check-in time

DURING THE MEETING:
- Bring someone with you if possible
- Stay in public view
- Inspect items before exchanging money
- Use cash for small transactions
- For larger transactions, meet at the buyer's bank

VEHICLE TRANSACTIONS:
- Meet at a police station or busy public area
- Never go for a test drive alone with a stranger
- Bring a friend for test drives
- Verify VIN and title match
- Use secure payment methods

RED FLAGS:
- Won't meet in public
- Pressure to complete quickly
- Story keeps changing
- Wants to come to your house
- Asks for personal information
- Payment issues (overpayment scam, wants wire transfer)""",
        summary="Meet at police safe zones or busy public places. Never at homes. Tell someone. Bring a friend if possible.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["marketplace", "craigslist", "facebook", "buying", "selling", "meeting", "transaction"],
        source="Law enforcement safe exchange recommendations",
        is_verified=True,
    ),
}


# ============================================================================
# Online Safety Knowledge
# ============================================================================


ONLINE_SAFETY = {
    "catfish_detection": KnowledgeItem(
        item_id="safety-online-catfish",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.ONLINE_SAFETY,
        title="Detecting Catfish and Fake Profiles",
        content="""How to identify fake profiles and catfishing:

WARNING SIGNS:
- Only a few photos, all professional-looking or model-quality
- Won't video chat or always has excuses
- Story inconsistencies (job, location, family details change)
- Relationship escalates very quickly ("love bombing")
- Claims to be overseas, military, working on oil rig, etc.
- Asks for money, gift cards, or financial help
- Avoids answering direct questions
- Profile is very new with few friends/connections

VERIFICATION STEPS:
1. Reverse image search their photos (Google Images, TinEye)
2. Check if their social media looks real (history, friends, posts over time)
3. Insist on video chat (not pre-recorded video)
4. Ask questions about their claimed location or profession
5. See if their photos show consistent background, style, aging

RED FLAGS DURING COMMUNICATION:
- Grammar/spelling suggests non-native speaker when they claim otherwise
- Available at unusual hours for their claimed timezone
- Photos don't match video calls
- Avoids talking on phone or video
- Gets defensive when questioned

WHAT REAL PEOPLE DO:
- Have varied photos (casual, with friends, different settings)
- Have social media history going back years
- Happy to video chat
- Consistent stories and details
- Don't ask for money from strangers

IF YOU SUSPECT CATFISHING:
- Do not send money or gifts
- Do not share personal/financial information
- Report the profile to the platform
- Consider searching their photos on romance scam databases""",
        summary="Reverse image search photos, insist on video chat, watch for inconsistencies, never send money to online-only contacts.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["catfish", "fake profile", "scam", "online", "dating", "romance scam", "verification"],
        source="FTC, IC3 romance scam warnings",
        is_verified=True,
    ),
    "personal_info_protection": KnowledgeItem(
        item_id="safety-online-personal-info",
        domain=KnowledgeDomainType.SAFETY,
        category=KnowledgeCategory.ONLINE_SAFETY,
        title="Protecting Personal Information Online",
        content="""Protecting your personal information online:

INFORMATION TO PROTECT:
- Full legal name (use nickname in casual settings)
- Home address
- Phone number
- Workplace/school name and location
- Daily schedule and routines
- Financial information
- Social Security Number / ID numbers
- Photos of your home, workplace, or kids

SOCIAL MEDIA BEST PRACTICES:
- Review privacy settings regularly
- Don't post real-time location (post after you leave)
- Be careful with location-tagged photos
- Don't share vacation plans publicly
- Limit who can see your friends list
- Don't accept friend requests from strangers
- Be cautious about quizzes and apps that access your data

INFORMATION THAT REVEALS LOCATION:
- Photos can contain GPS metadata
- Background details in photos (visible addresses, landmarks)
- Check-ins and location tags
- Regular posting patterns ("always at this coffee shop")

ACCOUNT SECURITY:
- Use unique, strong passwords for each account
- Enable two-factor authentication
- Don't reuse passwords
- Use a password manager
- Review login activity regularly
- Be suspicious of password reset emails you didn't request

IF YOUR INFORMATION IS EXPOSED:
- Change passwords immediately
- Enable additional security on financial accounts
- Consider credit freeze if serious
- Monitor for identity theft
- Report to relevant platforms""",
        summary="Protect address, phone, workplace. Don't post real-time location. Use strong unique passwords and 2FA.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.MEDIUM,
        keywords=["privacy", "personal information", "online", "social media", "security", "stalking"],
        source="Privacy and security best practices",
        is_verified=True,
    ),
}


# ============================================================================
# Safety Knowledge Domain Class
# ============================================================================


class SafetyKnowledgeDomain(BaseKnowledgeDomain):
    """Safety knowledge domain providing personal safety guidance."""

    def __init__(self, data_path: Path | None = None):
        super().__init__(KnowledgeDomainType.SAFETY, data_path)

    def _load_items(self) -> None:
        for item in DE_ESCALATION_TECHNIQUES.values():
            self.add_item(item)
        for item in SITUATIONAL_AWARENESS.values():
            self.add_item(item)
        for item in SAFE_MEETING.values():
            self.add_item(item)
        for item in ONLINE_SAFETY.values():
            self.add_item(item)

    def _apply_domain_boost(self, item, query_words, query, score):
        context = query.context
        if context:
            if context.get("stress_level") == "high" and item.category == KnowledgeCategory.DE_ESCALATION:
                score *= 1.5
            item_text = f"{item.title} {item.content}".lower()
            if (context.get("is_late_night") and "late" in item_text) or "night" in item_text:
                score *= 1.3
        return score

    def get_de_escalation_guidance(self) -> list[KnowledgeItem]:
        """Get all de-escalation guidance."""
        return self.get_items_by_category(KnowledgeCategory.DE_ESCALATION)

    def get_safe_meeting_protocols(self) -> list[KnowledgeItem]:
        """Get all safe meeting protocols."""
        return self.get_items_by_category(KnowledgeCategory.SAFE_MEETING)


# ============================================================================
# Factory Function
# ============================================================================


def create_safety_domain(data_path: Path | None = None) -> SafetyKnowledgeDomain:
    """Create a safety knowledge domain instance."""
    return SafetyKnowledgeDomain(data_path)
