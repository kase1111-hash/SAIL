"""
SAIL Legal Knowledge Domain

Provides jurisdiction-aware legal knowledge including:
- Traffic stop rights and procedures
- Police interaction guidelines
- Tenant rights
- Employment rights
- Consumer protection
- Consent laws
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.knowledge.base import (
    Jurisdiction,
    JurisdictionLevel,
    KnowledgeCategory,
    KnowledgeDomainType,
    KnowledgeItem,
    KnowledgeQuery,
    KnowledgeRelevance,
)
from src.knowledge.domains.base_domain import BaseKnowledgeDomain

logger = logging.getLogger(__name__)


# ============================================================================
# Legal Knowledge Data
# ============================================================================


# US Federal rights that apply everywhere
US_FEDERAL_RIGHTS = {
    "fourth_amendment": KnowledgeItem(
        item_id="legal-us-fed-4th-amendment",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.POLICE_INTERACTION,
        title="Fourth Amendment Rights",
        content="""The Fourth Amendment protects you from unreasonable searches and seizures.

Key points:
1. Police generally need a warrant to search your home
2. You can refuse consent to search your vehicle during a traffic stop
3. Saying "I do not consent to this search" is your right
4. If police search anyway, do not physically resist - object verbally

Note: There are exceptions including plain view, exigent circumstances, and search incident to arrest.""",
        summary="The Fourth Amendment protects against unreasonable searches. You can refuse consent to searches.",
        jurisdiction_level=JurisdictionLevel.FEDERAL,
        applicable_jurisdictions=["US"],
        relevance=KnowledgeRelevance.HIGH,
        keywords=["search", "seizure", "warrant", "consent", "fourth amendment", "4th amendment"],
        source="U.S. Constitution, Fourth Amendment",
        is_verified=True,
    ),
    "fifth_amendment": KnowledgeItem(
        item_id="legal-us-fed-5th-amendment",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.POLICE_INTERACTION,
        title="Fifth Amendment Rights - Right to Remain Silent",
        content="""The Fifth Amendment protects your right to remain silent.

Key points:
1. You have the right to remain silent during police questioning
2. You must clearly invoke this right by saying "I am invoking my right to remain silent"
3. Simply remaining silent without invoking may not be protected
4. You cannot be compelled to testify against yourself
5. Silence cannot be used as evidence of guilt in court

How to invoke:
- "I am invoking my Fifth Amendment right to remain silent"
- "I wish to remain silent"
- "I am not going to answer questions without a lawyer"

After invoking, stop talking except to ask if you are free to leave.""",
        summary="You have the right to remain silent. Invoke it clearly by stating 'I am invoking my right to remain silent.'",
        jurisdiction_level=JurisdictionLevel.FEDERAL,
        applicable_jurisdictions=["US"],
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["silence", "remain silent", "fifth amendment", "5th amendment", "miranda", "self-incrimination"],
        source="U.S. Constitution, Fifth Amendment",
        is_verified=True,
    ),
    "sixth_amendment": KnowledgeItem(
        item_id="legal-us-fed-6th-amendment",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.POLICE_INTERACTION,
        title="Sixth Amendment Rights - Right to an Attorney",
        content="""The Sixth Amendment guarantees your right to an attorney.

Key points:
1. You have the right to an attorney during criminal proceedings
2. If you cannot afford one, one will be appointed (public defender)
3. You can request a lawyer during police questioning
4. Once you request a lawyer, police must stop questioning until one is present

How to invoke:
- "I want a lawyer"
- "I am not answering questions without an attorney present"
- "I want to speak to an attorney before answering any questions"

Important: Be clear and unambiguous. "Maybe I should get a lawyer" may not be sufficient.""",
        summary="You have the right to an attorney. Clearly state 'I want a lawyer' to invoke this right.",
        jurisdiction_level=JurisdictionLevel.FEDERAL,
        applicable_jurisdictions=["US"],
        relevance=KnowledgeRelevance.CRITICAL,
        keywords=["attorney", "lawyer", "sixth amendment", "6th amendment", "public defender", "counsel"],
        source="U.S. Constitution, Sixth Amendment",
        is_verified=True,
    ),
}


# Traffic stop procedures by state
TRAFFIC_STOP_BY_STATE = {
    "US-CA": KnowledgeItem(
        item_id="legal-traffic-stop-ca",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TRAFFIC_STOP,
        title="California Traffic Stop Rights and Procedures",
        content="""During a traffic stop in California:

YOUR OBLIGATIONS:
1. Pull over safely to the right when signaled
2. Keep hands visible (steering wheel is best)
3. Provide driver's license, registration, and insurance when asked
4. Exit the vehicle if ordered by the officer

YOUR RIGHTS:
1. You do NOT have to consent to a vehicle search
2. You can record the interaction (California is a two-party consent state for private conversations, but police interactions in public are generally recordable)
3. You can ask "Am I free to leave?" or "Am I being detained?"
4. Passengers can ask if they are free to leave

CALIFORNIA SPECIFIC:
- AB 953: Officers must provide a business card with name and badge number
- You can request a supervisor at the scene
- Officers cannot search your phone without a warrant (Riley v. California)
- DUI checkpoint rules: You must stop, but do not have to answer questions beyond ID

IF CITED:
- Signing a ticket is NOT an admission of guilt
- Signing is a promise to appear in court
- Refusing to sign can result in arrest""",
        summary="In CA: Provide license/registration/insurance. You can refuse searches, record interactions, and ask if you're being detained.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-CA"],
        relevance=KnowledgeRelevance.HIGH,
        keywords=["traffic stop", "california", "police", "pulled over", "rights", "search", "dui"],
        source="California Vehicle Code, California Penal Code",
        is_verified=True,
    ),
    "US-TX": KnowledgeItem(
        item_id="legal-traffic-stop-tx",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TRAFFIC_STOP,
        title="Texas Traffic Stop Rights and Procedures",
        content="""During a traffic stop in Texas:

YOUR OBLIGATIONS:
1. Pull over safely and promptly
2. Keep hands visible on the steering wheel
3. Provide driver's license, registration, and proof of insurance
4. Inform officer if you have a concealed handgun license and are carrying

YOUR RIGHTS:
1. You do NOT have to consent to a search
2. Texas is a one-party consent state for recording
3. You can ask "Am I free to leave?"
4. You do not have to answer questions beyond providing ID

TEXAS SPECIFIC:
- Failure to identify: You must provide your name if lawfully detained
- You must show your CHL/LTC if carrying a concealed handgun
- Officers can ask passengers for ID if they have reasonable suspicion
- "Sandra Bland Act" (2017): Officers must make reasonable accommodations for those with mental illness

IF CITED:
- You must sign the citation (promise to appear)
- Refusing to sign can result in arrest
- Signing is NOT an admission of guilt""",
        summary="In TX: Provide ID and inform if carrying with CHL/LTC. You can refuse searches and record interactions.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-TX"],
        relevance=KnowledgeRelevance.HIGH,
        keywords=["traffic stop", "texas", "police", "pulled over", "rights", "chl", "ltc", "concealed carry"],
        source="Texas Transportation Code, Texas Penal Code",
        is_verified=True,
    ),
    "US-NY": KnowledgeItem(
        item_id="legal-traffic-stop-ny",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TRAFFIC_STOP,
        title="New York Traffic Stop Rights and Procedures",
        content="""During a traffic stop in New York:

YOUR OBLIGATIONS:
1. Pull over safely when signaled
2. Provide license, registration, and insurance when requested
3. Exit the vehicle if ordered

YOUR RIGHTS:
1. You do NOT have to consent to a vehicle search
2. New York is a one-party consent state for recording
3. You can ask "Am I being detained or am I free to go?"
4. You do not have to answer questions beyond providing ID

NEW YORK SPECIFIC:
- Officers must have reasonable suspicion to detain you beyond the initial stop
- NYC has additional civilian complaint review board (CCRB)
- You have the right to refuse roadside sobriety tests (but there are license consequences)
- Implied consent: Refusing a chemical test can result in license revocation

IF CITED:
- Sign the ticket (promise to appear)
- You can contest in Traffic Violations Bureau or court""",
        summary="In NY: Provide license/registration/insurance. You can refuse searches. Recording is permitted.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-NY"],
        relevance=KnowledgeRelevance.HIGH,
        keywords=["traffic stop", "new york", "police", "pulled over", "rights", "nyc"],
        source="New York Vehicle and Traffic Law",
        is_verified=True,
    ),
    "US-FL": KnowledgeItem(
        item_id="legal-traffic-stop-fl",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TRAFFIC_STOP,
        title="Florida Traffic Stop Rights and Procedures",
        content="""During a traffic stop in Florida:

YOUR OBLIGATIONS:
1. Pull over safely and promptly
2. Keep hands visible
3. Provide license, registration, and insurance
4. Inform officer if you have a concealed weapon license and are carrying

YOUR RIGHTS:
1. You do NOT have to consent to a search
2. Florida is a two-party consent state for audio recording, but video-only is permitted
3. You can ask "Am I free to leave?"
4. You do not have to answer investigative questions

FLORIDA SPECIFIC:
- Must inform officer if carrying with a concealed weapon license
- "Move Over Law" - move over or slow down for emergency vehicles
- Implied consent for breath/urine/blood tests if arrested for DUI
- Refusing DUI test results in automatic license suspension

IF CITED:
- Sign the citation (not an admission of guilt)
- You have 30 days to respond""",
        summary="In FL: Provide ID and inform if carrying with CWL. Recording video is OK, but audio may require consent.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-FL"],
        relevance=KnowledgeRelevance.HIGH,
        keywords=["traffic stop", "florida", "police", "pulled over", "rights", "cwl", "concealed weapon"],
        source="Florida Statutes",
        is_verified=True,
    ),
}


# General traffic stop guidance (universal)
GENERAL_TRAFFIC_STOP = KnowledgeItem(
    item_id="legal-traffic-stop-general",
    domain=KnowledgeDomainType.LEGAL,
    category=KnowledgeCategory.TRAFFIC_STOP,
    title="General Traffic Stop Guidelines",
    content="""When pulled over by police, follow these general guidelines:

IMMEDIATE ACTIONS:
1. Pull over safely to the right side of the road
2. Turn off your engine
3. Turn on interior lights if dark
4. Roll down your window
5. Keep your hands visible - on the steering wheel is best
6. Wait for the officer to approach before reaching for documents

WHAT TO SAY:
- Be polite and calm
- Provide your license, registration, and insurance when asked
- You may ask: "Why was I stopped?"
- You may ask: "Am I being detained or am I free to go?"

YOUR RIGHTS (GENERAL):
- You do not have to consent to a search of your vehicle
- You have the right to remain silent beyond providing identification
- You can record the interaction in most states
- You can refuse to answer questions like "Do you know why I pulled you over?"

IF ASKED TO SEARCH:
- You can say: "I do not consent to searches"
- If they search anyway, do not physically resist
- Make your objection clear verbally

STAY CALM:
- Do not argue, resist, or flee
- Keep your hands visible at all times
- Follow lawful orders
- If you believe your rights were violated, document everything and file a complaint later""",
    summary="Stay calm, hands visible, provide ID. You can refuse searches and remain silent beyond identification.",
    jurisdiction_level=JurisdictionLevel.UNIVERSAL,
    applicable_jurisdictions=[],
    relevance=KnowledgeRelevance.HIGH,
    keywords=["traffic stop", "pulled over", "police", "rights", "general", "search"],
    source="ACLU Know Your Rights",
    is_verified=True,
)


# Tenant rights by state
TENANT_RIGHTS_BY_STATE = {
    "US-CA": KnowledgeItem(
        item_id="legal-tenant-rights-ca",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TENANT_RIGHTS,
        title="California Tenant Rights",
        content="""California Tenant Rights Overview:

RENT CONTROL:
- California has statewide rent control (AB 1482)
- Annual rent increases capped at 5% + local CPI (max 10%)
- Applies to buildings 15+ years old
- Some cities have additional local rent control

EVICTION PROTECTIONS:
- "Just cause" eviction required after 12 months
- 60-day notice required for rent increases over 10%
- 30-day notice for increases under 10%
- 3-day notice for non-payment of rent

SECURITY DEPOSITS:
- Maximum 2 months rent (unfurnished) or 3 months (furnished)
- Must be returned within 21 days of move-out
- Itemized statement required for any deductions

HABITABILITY:
- Landlord must maintain habitable conditions
- Tenant can repair and deduct for certain issues
- Tenant can withhold rent in extreme cases (with proper notice)

LANDLORD ENTRY:
- 24-hour notice required (except emergencies)
- Only during normal business hours
- Must have valid reason (repairs, showing, etc.)

RETALIATION PROTECTION:
- Landlord cannot retaliate for complaints or exercising rights
- 180-day presumption of retaliation after protected activity""",
        summary="CA: Rent control caps increases at 5%+CPI. 24-hr notice for entry. Security deposit max 2 months. Just cause eviction required.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-CA"],
        relevance=KnowledgeRelevance.MEDIUM,
        keywords=["tenant", "renter", "landlord", "rent control", "eviction", "california", "security deposit"],
        source="California Civil Code",
        is_verified=True,
    ),
    "US-NY": KnowledgeItem(
        item_id="legal-tenant-rights-ny",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.TENANT_RIGHTS,
        title="New York Tenant Rights",
        content="""New York Tenant Rights Overview:

RENT STABILIZATION (NYC):
- Applies to buildings with 6+ units built before 1974
- Annual rent increases set by Rent Guidelines Board
- Tenant has right to lease renewal
- Strong eviction protections

SECURITY DEPOSITS:
- Maximum 1 month rent (as of 2019 HSTPA)
- Must be returned within 14 days of move-out
- Must be held in interest-bearing account

EVICTION PROTECTIONS:
- Landlord must go through court process
- No self-help evictions allowed
- Right to remain until court order
- Good cause eviction in many areas

HABITABILITY:
- Warranty of habitability implied in all leases
- Landlord must provide heat, hot water, repairs
- Tenant can request rent abatement for violations

LANDLORD ENTRY:
- Reasonable notice required
- Tenant can deny entry for non-emergencies
- Lease may specify notice requirements

LEASE RENEWAL:
- 90-150 day notice required for non-renewal (units 2+ years)
- 30 day notice for tenancies under 1 year""",
        summary="NY: Security deposit max 1 month. Strong eviction protections. Rent stabilization in NYC for eligible buildings.",
        jurisdiction_level=JurisdictionLevel.STATE,
        applicable_jurisdictions=["US-NY"],
        relevance=KnowledgeRelevance.MEDIUM,
        keywords=["tenant", "renter", "landlord", "rent stabilization", "eviction", "new york", "nyc"],
        source="New York Real Property Law, HSTPA 2019",
        is_verified=True,
    ),
}


# Employment rights
EMPLOYMENT_RIGHTS = {
    "overtime": KnowledgeItem(
        item_id="legal-employment-overtime",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.EMPLOYMENT_RIGHTS,
        title="Overtime Rights (Federal FLSA)",
        content="""Federal Overtime Rights under the Fair Labor Standards Act:

BASIC RULE:
- Non-exempt employees must receive 1.5x regular rate for hours over 40/week
- Weekly threshold, not daily (federal law)
- Some states have daily overtime (CA: over 8 hours/day)

WHO IS COVERED:
- Most hourly employees
- Salaried employees under $684/week ($35,568/year)
- Does not apply to exempt employees (executive, administrative, professional)

EXEMPTIONS:
- Executive: Manage 2+ employees, hiring/firing authority
- Administrative: Office work, independent judgment
- Professional: Advanced knowledge, specialized education
- Computer: Programmers, analysts over $684/week
- Outside sales: Regular work away from employer's place of business

YOUR RIGHTS:
- Employer cannot waive your right to overtime
- You can file a complaint with DOL or sue
- 2-year statute of limitations (3 years if willful)
- Can recover unpaid wages plus liquidated damages""",
        summary="Non-exempt employees get 1.5x pay for hours over 40/week. Exemptions apply to certain salaried positions.",
        jurisdiction_level=JurisdictionLevel.FEDERAL,
        applicable_jurisdictions=["US"],
        relevance=KnowledgeRelevance.MEDIUM,
        keywords=["overtime", "wages", "employment", "flsa", "hours", "exempt", "non-exempt"],
        source="Fair Labor Standards Act (FLSA)",
        is_verified=True,
    ),
    "discrimination": KnowledgeItem(
        item_id="legal-employment-discrimination",
        domain=KnowledgeDomainType.LEGAL,
        category=KnowledgeCategory.EMPLOYMENT_RIGHTS,
        title="Workplace Discrimination Rights",
        content="""Federal Workplace Discrimination Protections:

PROTECTED CLASSES (Federal):
- Race and color
- National origin
- Sex (including pregnancy, sexual orientation, gender identity)
- Religion
- Age (40 and over)
- Disability
- Genetic information

COVERED EMPLOYERS:
- Title VII: 15+ employees
- ADEA (age): 20+ employees
- ADA (disability): 15+ employees

TYPES OF DISCRIMINATION:
- Disparate treatment: Intentional different treatment
- Disparate impact: Neutral policy with discriminatory effect
- Harassment: Hostile work environment or quid pro quo
- Retaliation: Punishment for reporting or opposing discrimination

YOUR RIGHTS:
- File EEOC complaint within 180 days (or 300 in some states)
- Request reasonable accommodations for disability/religion
- Protection from retaliation for complaints

WHAT TO DO:
1. Document incidents (dates, witnesses, details)
2. Report through internal channels if safe
3. File EEOC charge before lawsuit
4. EEOC will investigate or issue right-to-sue letter""",
        summary="Federal law prohibits discrimination based on race, sex, religion, age (40+), disability. File EEOC complaint within 180 days.",
        jurisdiction_level=JurisdictionLevel.FEDERAL,
        applicable_jurisdictions=["US"],
        relevance=KnowledgeRelevance.MEDIUM,
        keywords=["discrimination", "harassment", "workplace", "employment", "eeoc", "protected class", "civil rights"],
        source="Title VII, ADA, ADEA, EEOC",
        is_verified=True,
    ),
}


# ============================================================================
# Legal Knowledge Domain Class
# ============================================================================


class LegalKnowledgeDomain(BaseKnowledgeDomain):
    """Legal knowledge domain providing jurisdiction-aware legal information."""

    def __init__(self, data_path: Path | None = None):
        super().__init__(KnowledgeDomainType.LEGAL, data_path)

    def _load_items(self) -> None:
        for item in US_FEDERAL_RIGHTS.values():
            self.add_item(item)
        for item in TRAFFIC_STOP_BY_STATE.values():
            self.add_item(item)
        self.add_item(GENERAL_TRAFFIC_STOP)
        for item in TENANT_RIGHTS_BY_STATE.values():
            self.add_item(item)
        for item in EMPLOYMENT_RIGHTS.values():
            self.add_item(item)

    def _filter_jurisdiction(self, candidates, query):
        if not query.jurisdiction:
            return None
        filtered = [i for i in candidates if i.applies_to_jurisdiction(query.jurisdiction)]
        return filtered if filtered else None

    def _apply_domain_boost(self, item, query_words, query, score):
        if query.jurisdiction and item.applies_to_jurisdiction(query.jurisdiction):
            if item.jurisdiction_level == JurisdictionLevel.STATE:
                score *= 1.5
            elif item.jurisdiction_level == JurisdictionLevel.LOCAL:
                score *= 1.75
        return score

    def get_traffic_stop_guidance(
        self,
        jurisdiction: Jurisdiction | None = None,
    ) -> list[KnowledgeItem]:
        """Get traffic stop guidance for a jurisdiction."""
        items = self.get_items_by_category(KnowledgeCategory.TRAFFIC_STOP)

        if jurisdiction:
            # Get state-specific first
            state_items = [
                item for item in items
                if item.applies_to_jurisdiction(jurisdiction)
                and item.jurisdiction_level == JurisdictionLevel.STATE
            ]
            if state_items:
                # Add federal rights
                federal_items = self.get_items_by_category(KnowledgeCategory.POLICE_INTERACTION)
                return state_items + [item for item in federal_items if item.jurisdiction_level == JurisdictionLevel.FEDERAL]

        # Return general guidance with federal rights
        general = [item for item in items if item.jurisdiction_level == JurisdictionLevel.UNIVERSAL]
        federal = [
            item for item in self._items.values()
            if item.category == KnowledgeCategory.POLICE_INTERACTION
            and item.jurisdiction_level == JurisdictionLevel.FEDERAL
        ]
        return general + federal

    def get_tenant_rights(
        self,
        jurisdiction: Jurisdiction | None = None,
    ) -> list[KnowledgeItem]:
        """Get tenant rights for a jurisdiction."""
        items = self.get_items_by_category(KnowledgeCategory.TENANT_RIGHTS)

        if jurisdiction:
            state_items = [
                item for item in items
                if item.applies_to_jurisdiction(jurisdiction)
            ]
            if state_items:
                return state_items

        return items


# ============================================================================
# Factory Function
# ============================================================================


def create_legal_domain(data_path: Path | None = None) -> LegalKnowledgeDomain:
    """Create a legal knowledge domain instance."""
    return LegalKnowledgeDomain(data_path)
