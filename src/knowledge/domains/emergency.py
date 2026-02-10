"""
SAIL Emergency Knowledge Domain

Provides emergency procedure guidance including:
- Medical emergency walkthroughs
- Accident scene procedures
- Fire safety protocols
- Natural disaster response
- Emergency contact management
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
# Medical Emergency Knowledge
# ============================================================================


MEDICAL_EMERGENCIES = {
    "calling_911": KnowledgeItem(
        item_id="emergency-call-911",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="How to Call 911 Effectively",
        content="""How to call 911 effectively in an emergency:

WHEN TO CALL 911:
- Life-threatening situations
- Medical emergencies (unconscious, difficulty breathing, severe bleeding)
- Crimes in progress
- Fire or smoke
- Car accidents with injuries
- When someone's life, health, or property is in immediate danger

WHAT TO DO:
1. Stay calm - take a breath
2. Dial 911
3. Stay on the line - don't hang up
4. Answer all questions - the dispatcher may be sending help while asking

INFORMATION TO PROVIDE:
1. LOCATION: Address, cross streets, landmarks
   - "I'm at 123 Main Street, between Oak and Elm"
   - "I'm at the gas station on the corner of 5th and Broadway"
   - "I'm on Highway 101, mile marker 45"

2. WHAT'S HAPPENING: Type of emergency
   - "Someone is having a heart attack"
   - "There's a car accident with injuries"
   - "There's a fire in a house"

3. WHO: How many people involved/injured
   - "One person is injured"
   - "Three cars involved"

4. YOUR NAME: So they can call back if disconnected

IMPORTANT:
- Don't hang up until told to
- Follow any instructions given
- If safe, stay with the victim
- Send someone to wave down emergency vehicles

IF YOU CAN'T SPEAK:
- Stay on the line - dispatcher may hear background
- On mobile, text 911 if available in your area
- Press buttons in response to questions if instructed

ACCIDENTAL CALL:
- Don't hang up - tell them it was accidental
- Hanging up may cause them to send help anyway""",
        summary="Call 911 for emergencies. Stay calm. Give location first, then describe the emergency. Stay on the line.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        applicable_jurisdictions=["US", "CA"],
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["911", "emergency", "call", "help", "ambulance", "police", "fire"],
        source="National 911 Program",
        is_verified=True,
    ),
    "cpr_basics": KnowledgeItem(
        item_id="emergency-cpr",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="Hands-Only CPR Guide",
        content="""Hands-only CPR for adults (when someone collapses):

WHEN TO USE:
- Person is unresponsive
- Person is not breathing or only gasping
- For teens and adults
- When you haven't been trained in full CPR

STEPS:

1. CHECK FOR RESPONSE
   - Tap shoulders firmly
   - Shout "Are you okay?"
   - If no response, proceed

2. CALL 911
   - Call yourself or tell someone specific: "You, call 911"
   - Put on speaker if alone
   - Send someone for an AED if available

3. CHECK FOR BREATHING
   - Look at chest for rise and fall
   - No normal breathing = start CPR
   - Gasping is NOT normal breathing

4. START COMPRESSIONS
   - Place heel of one hand on center of chest
   - Place other hand on top, interlace fingers
   - Keep arms straight, shoulders over hands
   - Push hard and fast
   - At least 2 inches deep
   - Rate: 100-120 per minute (beat of "Stayin' Alive")
   - Let chest fully rise between compressions

5. DON'T STOP
   - Continue until EMS arrives
   - Or person shows signs of life
   - Or you're too exhausted to continue
   - If tired, switch with someone every 2 minutes

HAND POSITION:
- Center of chest, between the nipples
- Use heel of palm, not fingers
- Keep fingers off the ribs

IMPORTANT:
- Push HARD - ribs may crack, that's okay
- Push FAST - 100-120 per minute
- Don't stop to check for pulse
- Any CPR is better than no CPR

FOR INFANTS/CHILDREN:
- Use different technique - consider taking a CPR class
- For children, two fingers for compressions
- Cover mouth AND nose for rescue breaths""",
        summary="For unconscious adult: Call 911, push hard and fast center of chest (100-120/min), don't stop until help arrives.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["cpr", "heart attack", "unconscious", "not breathing", "cardiac arrest", "compressions"],
        source="American Heart Association",
        is_verified=True,
    ),
    "choking_adult": KnowledgeItem(
        item_id="emergency-choking-adult",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="Choking Response for Adults",
        content="""What to do when an adult is choking:

SIGNS OF CHOKING:
- Cannot speak, cry out, or breathe
- Hands clutching throat (universal choking sign)
- Skin turning blue
- Weak cough or no cough
- High-pitched sounds when trying to inhale

IF PERSON CAN COUGH OR SPEAK:
- Encourage continued coughing
- Do NOT interfere
- Stay with them
- Be ready to act if it worsens

IF PERSON CANNOT BREATHE/SPEAK (HEIMLICH MANEUVER):

1. Call 911 or have someone call

2. Stand behind the person

3. Make a fist with one hand
   - Thumb side against stomach
   - Above navel, below ribcage

4. Grasp fist with other hand

5. Give quick upward thrusts
   - Pull inward and upward
   - Quick, forceful movements
   - Continue until object comes out or person becomes unconscious

IF PERSON BECOMES UNCONSCIOUS:
- Lower them to the ground
- Call 911 if not done
- Begin CPR
- Check mouth before each breath for visible object
- Do NOT do blind finger sweeps

SELF-CHOKING (ALONE):
- Call 911 immediately if possible
- Make a fist above navel
- Thrust upward with other hand
- Or use a chair/counter edge
- Lean over and push abdomen against the edge

PREGNANT OR OBESE PERSON:
- Place fist on center of breastbone
- Thrust straight back (chest thrusts)""",
        summary="Stand behind, fist above navel, quick upward thrusts until object dislodges. If unconscious, start CPR.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["choking", "heimlich", "can't breathe", "stuck in throat", "abdominal thrust"],
        source="American Red Cross",
        is_verified=True,
    ),
    "severe_bleeding": KnowledgeItem(
        item_id="emergency-bleeding",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="Controlling Severe Bleeding",
        content="""How to control severe bleeding:

SIGNS OF SEVERE BLEEDING:
- Blood spurting or flowing rapidly
- Blood pooling on ground
- Blood soaking through clothing
- Victim confused, pale, or losing consciousness
- Partial or complete amputation

STEPS:

1. ENSURE SAFETY
   - Put on gloves if available
   - Any barrier between you and blood (plastic bag, clothing)

2. CALL 911

3. APPLY DIRECT PRESSURE
   - Place clean cloth on wound
   - Press down firmly with both hands
   - Don't lift to check - add more cloth on top if soaking through
   - Maintain constant pressure

4. IF LIMB INJURY:
   - Keep pressing
   - If possible, elevate the limb above heart
   - Continue pressure

5. IF PRESSURE ISN'T WORKING:
   - Pack the wound (stuff cloth into it)
   - Apply tourniquet if available and trained
   - Tourniquet placement: 2-3 inches above wound
   - Tighten until bleeding stops
   - Note the time

TOURNIQUET USE (LIFE-THREATENING BLEEDING):
- Place 2-3 inches above wound (not on joint)
- Tighten until bleeding stops
- Once applied, DO NOT remove
- Write the time on victim's forehead or tourniquet

DON'T:
- Don't remove objects embedded in wound
- Don't remove blood-soaked cloths (add more on top)
- Don't use tourniquet on neck
- Don't give up - keep pressure until help arrives

SHOCK PREVENTION:
- Keep victim lying down
- Keep them warm (blanket, coat)
- Talk to them, keep them calm""",
        summary="Apply firm direct pressure with cloth. Don't lift to check. Call 911. Use tourniquet for life-threatening limb bleeding.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["bleeding", "blood", "wound", "tourniquet", "pressure", "cut"],
        source="Stop the Bleed, American Red Cross",
        is_verified=True,
    ),
    "stroke_recognition": KnowledgeItem(
        item_id="emergency-stroke",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="Stroke Recognition and Response (BE-FAST)",
        content="""Recognizing and responding to stroke:

TIME IS CRITICAL:
- Brain cells die every minute during a stroke
- Treatment within 3 hours dramatically improves outcomes
- Call 911 immediately if you suspect stroke

BE-FAST RECOGNITION:

B - BALANCE
- Sudden loss of balance or coordination
- Difficulty walking, dizziness

E - EYES
- Sudden vision problems
- Trouble seeing in one or both eyes
- Double vision

F - FACE
- Face drooping on one side
- Ask them to smile - is it uneven?
- Numbness on one side

A - ARMS
- Arm weakness or numbness
- Ask them to raise both arms - does one drift down?

S - SPEECH
- Slurred speech
- Difficulty speaking or understanding
- Ask them to repeat a simple sentence

T - TIME
- Note when symptoms started
- Call 911 immediately
- Time of onset is crucial for treatment

WHAT TO DO:
1. Call 911 immediately
2. Note the time symptoms started
3. Help them lie down with head slightly elevated
4. Do not give them anything to eat or drink
5. If unconscious, place on side (recovery position)
6. Stay with them and keep them calm
7. Be ready to perform CPR if they become unresponsive

DON'T:
- Don't drive them yourself - call ambulance
- Don't give aspirin (could be bleeding stroke)
- Don't let them "sleep it off"
- Don't delay - minutes matter""",
        summary="BE-FAST: Balance, Eyes, Face drooping, Arm weakness, Speech difficulty = stroke. Call 911 immediately. Note symptom start time.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["stroke", "face drooping", "arm weakness", "speech", "fast", "be fast"],
        source="American Stroke Association",
        is_verified=True,
    ),
    "heart_attack": KnowledgeItem(
        item_id="emergency-heart-attack",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.MEDICAL_EMERGENCY,
        title="Heart Attack Recognition and Response",
        content="""Recognizing and responding to a heart attack:

WARNING SIGNS:
- Chest discomfort (pressure, squeezing, pain)
- Pain spreading to shoulders, neck, jaw, or arms
- Shortness of breath
- Cold sweat
- Nausea or vomiting
- Light-headedness

WOMEN MAY EXPERIENCE:
- Less typical chest pain
- More likely: shortness of breath, nausea
- Jaw or back pain
- Unusual fatigue

WHAT TO DO:

1. CALL 911 IMMEDIATELY
   - Don't drive yourself
   - Ambulance can start treatment en route

2. CHEW ASPIRIN (if not allergic)
   - 325mg regular aspirin, or
   - 4 baby aspirin (81mg each)
   - Chewing works faster than swallowing whole

3. REST IN COMFORTABLE POSITION
   - Sitting up or semi-reclined often easiest for breathing
   - Loosen tight clothing

4. STAY CALM
   - Try to remain calm
   - Help is on the way

5. BE READY FOR CPR
   - If person becomes unconscious and stops breathing
   - Start hands-only CPR

IF YOU'RE ALONE:
- Call 911 first
- Unlock your door so paramedics can enter
- Don't exert yourself
- Chew aspirin if available and not allergic

DON'T:
- Don't ignore symptoms hoping they'll pass
- Don't drive yourself to hospital
- Don't wait to see if symptoms get worse
- Don't take someone else's prescription medications""",
        summary="Chest pain/pressure, pain in arm/jaw, shortness of breath = possible heart attack. Call 911 immediately. Chew aspirin if not allergic.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["heart attack", "chest pain", "cardiac", "heart", "aspirin"],
        source="American Heart Association",
        is_verified=True,
    ),
}


# ============================================================================
# Accident Procedures
# ============================================================================


ACCIDENT_PROCEDURES = {
    "car_accident": KnowledgeItem(
        item_id="emergency-car-accident",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.ACCIDENT,
        title="Car Accident Response Procedure",
        content="""What to do after a car accident:

IMMEDIATELY AFTER:

1. CHECK YOURSELF AND PASSENGERS
   - Are you injured?
   - Can you move safely?

2. ASSESS THE SCENE
   - Is there fire or smoke?
   - Is traffic a danger?
   - Is everyone conscious?

3. CALL 911 IF:
   - Anyone is injured
   - There's significant damage
   - Other driver appears impaired
   - Other driver is uncooperative

4. MOVE TO SAFETY (IF POSSIBLE)
   - If minor accident and cars are drivable
   - Move to shoulder or parking lot
   - Turn on hazard lights
   - If you can't move the car, exit and move away from traffic

5. DON'T ADMIT FAULT
   - Exchange information without apologizing or admitting fault
   - "I'm sorry" can be used against you legally
   - Stick to facts

INFORMATION TO EXCHANGE:
- Name and contact information
- Insurance company and policy number
- Driver's license number
- License plate number
- Make, model, and color of vehicles
- Location of accident

DOCUMENT THE SCENE:
- Take photos of all vehicles and damage
- Photo license plates
- Photo the overall scene
- Photo any injuries
- Note time, weather, road conditions
- Get contact info for witnesses

WHEN POLICE ARRIVE:
- Provide your license, registration, insurance
- Answer factual questions
- Don't speculate or admit fault
- Get the police report number

AFTER THE SCENE:
- File insurance claim promptly
- See a doctor even if you feel fine (injuries can appear later)
- Keep records of everything
- Don't post on social media about the accident""",
        summary="Check safety, call 911 if injuries, exchange info, document with photos, don't admit fault, see a doctor.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["car accident", "crash", "collision", "fender bender", "hit"],
        source="Insurance Institute for Highway Safety",
        is_verified=True,
    ),
}


# ============================================================================
# Fire Safety
# ============================================================================


FIRE_SAFETY = {
    "house_fire": KnowledgeItem(
        item_id="emergency-fire-house",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.FIRE,
        title="House Fire Response",
        content="""What to do in a house fire:

IF YOU DISCOVER A FIRE:

1. GET OUT IMMEDIATELY
   - Don't try to fight large fires
   - Alert others as you leave
   - Yell "FIRE!" repeatedly

2. FEEL DOORS BEFORE OPENING
   - Use back of hand on door and knob
   - If hot, don't open - use alternate route
   - If cool, open slowly and be ready to close

3. STAY LOW
   - Smoke rises, cleaner air is near the floor
   - Crawl if necessary
   - Cover mouth and nose with cloth if available

4. GET OUT AND STAY OUT
   - Go to your meeting point
   - Do NOT go back inside for anything
   - Not for pets, not for phones, not for valuables

5. CALL 911
   - From outside, from a neighbor's phone
   - Give address clearly

IF CLOTHES CATCH FIRE:
- STOP - don't run (fans flames)
- DROP - to the ground
- ROLL - back and forth to smother flames
- Cover face with hands while rolling

IF TRAPPED:
- Close the door
- Seal cracks with cloth or tape
- Call 911 and give your exact location
- Go to window and signal for help
- Open window slightly at top for air if needed

BEFORE A FIRE (PREVENTION):
- Have working smoke detectors on every floor
- Test smoke detectors monthly
- Have a fire extinguisher in kitchen
- Create and practice an escape plan
- Know two ways out of every room
- Designate a meeting point outside""",
        summary="Get out immediately, stay low, don't go back inside, call 911 from outside. If clothes catch fire: stop, drop, roll.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["fire", "house fire", "smoke", "flames", "escape", "burn"],
        source="National Fire Protection Association",
        is_verified=True,
    ),
}


# ============================================================================
# Natural Disasters
# ============================================================================


NATURAL_DISASTERS = {
    "earthquake": KnowledgeItem(
        item_id="emergency-earthquake",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.NATURAL_DISASTER,
        title="Earthquake Response",
        content="""What to do during an earthquake:

DURING THE SHAKING:

IF INDOORS:
- DROP to your hands and knees
- Take COVER under sturdy table or desk
- HOLD ON until shaking stops
- If no cover, get against interior wall
- Stay away from windows, outside walls, heavy furniture

DROP, COVER, HOLD ON:
- Drop: Get down on hands and knees
- Cover: Get under sturdy furniture, protect head/neck
- Hold On: Hold furniture leg, be ready to move with it

IF IN BED:
- Stay in bed
- Protect head with pillow
- Stay away from windows

IF OUTDOORS:
- Move to open area away from buildings
- Avoid power lines, trees, buildings
- Drop and cover your head

IF IN A CAR:
- Pull over away from bridges, overpasses, buildings
- Stay in car with seatbelt on
- Don't stop under overpasses

IF IN A BUILDING:
- DO NOT run outside during shaking
- Stay away from elevators
- Don't stand in doorways (myth)

AFTER THE EARTHQUAKE:
- Expect aftershocks
- Check yourself and others for injuries
- Check for gas leaks (smell, hissing) - leave if suspected
- Check for structural damage before entering buildings
- Use text messages (phone networks may be overloaded)
- Do not use elevators""",
        summary="DROP to knees, take COVER under sturdy furniture, HOLD ON until shaking stops. Don't run outside during shaking.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["earthquake", "shaking", "tremor", "quake", "drop cover hold"],
        source="FEMA, USGS",
        is_verified=True,
    ),
    "tornado": KnowledgeItem(
        item_id="emergency-tornado",
        domain=KnowledgeDomainType.EMERGENCY,
        category=KnowledgeCategory.NATURAL_DISASTER,
        title="Tornado Safety",
        content="""What to do during a tornado:

TORNADO WATCH vs WARNING:
- WATCH: Conditions are favorable - be prepared
- WARNING: Tornado spotted or on radar - take shelter NOW

WHEN A WARNING IS ISSUED:

IF IN A BUILDING:
- Go to lowest floor (basement is best)
- Go to interior room with no windows
- Bathroom or closet in center of building
- Get under sturdy furniture
- Protect head and neck with arms
- Cover yourself with mattress or blankets

IF NO BASEMENT:
- Interior room on lowest floor
- Stay away from windows, doors, outside walls
- Get under heavy furniture
- Protect your head

IF IN A MOBILE HOME:
- GET OUT - mobile homes are not safe
- Go to designated shelter
- Or lie flat in low area (ditch) away from mobile home

IF IN A CAR:
- Do NOT try to outrun tornado
- If possible, drive to sturdy shelter
- If no shelter available:
  - Park car, stay in with seatbelt on
  - Duck below windows, protect head
  - OR if debris flying, get to low area outside car

IF OUTDOORS:
- Get to lowest area possible (ditch, culvert)
- Lie flat and cover head
- Stay away from trees, cars, overpasses

SIGNS OF TORNADO:
- Dark, greenish sky
- Large hail
- Loud roar (like freight train)
- Rotating clouds""",
        summary="Go to lowest floor, interior room, no windows. Protect head. Mobile homes not safe - evacuate. Don't outrun in car.",
        jurisdiction_level=JurisdictionLevel.UNIVERSAL,
        relevance=KnowledgeRelevance.HIGH,
        keywords=["tornado", "twister", "warning", "shelter", "funnel cloud"],
        source="National Weather Service, FEMA",
        is_verified=True,
    ),
}


# ============================================================================
# Emergency Knowledge Domain Class
# ============================================================================


class EmergencyKnowledgeDomain(BaseKnowledgeDomain):
    """Emergency protocols knowledge domain."""

    def __init__(self, data_path: Path | None = None):
        super().__init__(KnowledgeDomainType.EMERGENCY, data_path)

    def _load_items(self) -> None:
        for item in MEDICAL_EMERGENCIES.values():
            self.add_item(item)
        for item in ACCIDENT_PROCEDURES.values():
            self.add_item(item)
        for item in FIRE_SAFETY.values():
            self.add_item(item)
        for item in NATURAL_DISASTERS.values():
            self.add_item(item)

    def get_medical_procedures(self) -> list[KnowledgeItem]:
        """Get all medical emergency procedures."""
        return self.get_items_by_category(KnowledgeCategory.MEDICAL_EMERGENCY)

    def get_disaster_procedures(self) -> list[KnowledgeItem]:
        """Get all natural disaster procedures."""
        return self.get_items_by_category(KnowledgeCategory.NATURAL_DISASTER)


# ============================================================================
# Factory Function
# ============================================================================


def create_emergency_domain(data_path: Path | None = None) -> EmergencyKnowledgeDomain:
    """Create an emergency knowledge domain instance."""
    return EmergencyKnowledgeDomain(data_path)
