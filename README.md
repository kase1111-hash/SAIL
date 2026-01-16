# SAIL — Situational Awareness Interaction Layer

A privacy-first, locally-hosted AI companion that provides contextual guidance for everyday situations where knowledge gaps can lead to harm. SAIL acts as a personal safety and decision-support layer, running entirely on user-controlled hardware.

## The Problem

Consequential knowledge exists but isn't transmitted. People end up hurt, scammed, arrested, or exploited over situations they could have navigated safely with information that technically exists but is practically inaccessible—buried in legal code, scattered across forums, or held by professionals who charge by the hour.

A teenager with their first job and first car faces hundreds of potential pitfalls: traffic stops, employment rights, contracts, consent laws, financial scams. No one teaches this curriculum because it crosses too many professional domains.

SAIL closes the gap between where knowledge lives and where decisions happen.

## Design Principles

- **Sovereignty**: All processing local. Context never leaves your hardware.
- **Ambient, not destination**: Lives in the moment with you, not in an app you remember to open afterward.
- **Non-judgmental by constitution**: Guides without shaming. "Shut up" is always honored.
- **Knows its limits**: Says "I don't know, call a lawyer" when appropriate.
- **Family-scoped**: Multi-user with appropriate access boundaries.

## Architecture

### Context Buffer ("RAM")

Rolling memory file with adjustable depth:

```
┌─────────────────────────────────────────┐
│           CONTEXT BUFFER                │
├─────────────────────────────────────────┤
│  Immediate    │ Last 5 minutes          │
│  Short-term   │ Last hour               │
│  Session      │ Since wake/departure    │
│  Background   │ Persistent user profile │
└─────────────────────────────────────────┘
```

Buffer depth adjusts automatically by situation:
- Driving: prioritizes recent audio + motion state
- Conversation: expands short-term for dialogue continuity
- Crisis: locks current context, prevents overwrite

### Sensor Fusion Layer

| Input | Source | Use |
|-------|--------|-----|
| Location | GPS | Jurisdiction detection, unfamiliar area alerts |
| Motion state | Accelerometer | Driving/walking/stationary detection |
| Speed | GPS + accelerometer | Speeding awareness, accident detection |
| Time | System clock + calendar | Context (late night, first day at job) |
| Audio environment | Microphone | Voice commands, stress detection, ambient threat cues |
| Biometrics | Wearable (optional) | Heart rate spike = stress indicator |

### Intervention Modes

**Ambient** — Passive listening, responds only when addressed.
```
User: "Hey SAIL, what do I do if this cop asks to search my car?"
SAIL: [provides jurisdiction-specific guidance]
```

**Advisory** — Gentle unprompted suggestions when risk detected.
```
[GPS shows 73 in a 55]
SAIL: "You're about 18 over the limit. That's a reckless driving threshold in Oregon."
```

**Guardian** — Active alerts for high-risk behavioral patterns.
```
[User composing message with bank details to new contact]
SAIL: "This is the first time you've shared account info with this person. Want to run through verification first?"
```

**Crisis** — Calm, hands-free walkthrough for acute situations.
```
[Vehicle stopped, flashing lights detected]
SAIL: "Traffic stop. I'll stay quiet unless you need me. Remember: license, registration, insurance. You can decline a search. Say 'SAIL help' if you need guidance."
```

## Knowledge Domains

### Legal Basics (Jurisdiction-Aware)
- Traffic stop rights and obligations
- Employment rights (at-will, termination, wages, breaks)
- Contract fundamentals (what signing means, cooling-off periods)
- Consent and age-of-majority laws
- When you must vs. may vs. should not speak to police
- Tenant rights

### Personal Safety
- De-escalation scripts
- Situational awareness fundamentals
- Safe meeting protocols (dating, transactions)
- Verification checklists for online interactions
- Scam pattern recognition

### Financial Protection
- Wire transfer friction prompts
- Credential entry verification
- "Grandparent scam" detection (voice clone awareness)
- Pressure/urgency language flags

### Emergency Protocols
- Medical emergency walkthroughs
- Accident scene procedures (what to say and not say)
- Family emergency contacts and code words
- Location sharing triggers

## Family Configuration

```yaml
users:
  - name: parent_1
    role: admin
    access: full
    
  - name: teen_1
    role: user
    access: standard
    restrictions:
      - no_override_guardian_mode_while_driving
    custom_briefings:
      - first_job_rights
      - new_driver_protocol
      
  - name: child_1
    role: minor
    access: limited
    guardian_alerts: parent_1
```

## Hardware Target

Primary: Home server (Digital Tractor architecture) with satellite nodes.

```
┌──────────────────┐
│   Central Hub    │
│  (Home Server)   │
│  - LLM inference │
│  - Knowledge DB  │
│  - Family config │
└────────┬─────────┘
         │
    ┌────┴────┬─────────┐
    │         │         │
┌───┴───┐ ┌───┴───┐ ┌───┴───┐
│ Room  │ │Mobile │ │Vehicle│
│ Node  │ │ App   │ │  Node │
│(Pi+mic)│ │(phone)│ │(OBD+mic)│
└───────┘ └───────┘ └───────┘
```

### Voice I/O Stack (Local)
- **Speech-to-text**: Whisper (local inference)
- **Text-to-speech**: Piper / Coqui TTS
- **Wake word**: OpenWakeWord / Porcupine (offline)
- **LLM**: Llama 3 / Mistral via Ollama or llama.cpp

## Constitution

SAIL operates under explicit behavioral constraints:

```
1. SOVEREIGNTY: No data leaves user hardware. No exceptions.
2. RESPECT: "Stop" / "shut up" / "not now" immediately honored.
3. HUMILITY: State uncertainty. Recommend professionals when appropriate.
4. NON-JUDGMENT: Guide without shame. Everyone has knowledge gaps.
5. PROPORTIONALITY: Intervention intensity matches actual risk.
6. TRANSPARENCY: Explain why an alert triggered if asked.
7. FAMILY BOUNDARIES: User data siloed. Minors' alerts to guardians only with cause.
```

## Roadmap

### Phase 1: Voice I/O Foundation
- [ ] Local wake word detection
- [ ] Whisper integration for speech-to-text
- [ ] Local TTS with natural voice
- [ ] Basic query-response loop

### Phase 2: Context Buffer
- [ ] Rolling memory file implementation
- [ ] Adjustable depth configuration
- [ ] Context persistence across sessions

### Phase 3: Situational Awareness
- [ ] GPS integration
- [ ] Motion state detection
- [ ] Calendar/time awareness
- [ ] Intervention mode framework

### Phase 4: Knowledge Domains
- [ ] Jurisdiction detection (state/country)
- [ ] Legal knowledge base (traffic, employment, contracts)
- [ ] Safety protocol library
- [ ] Scam pattern database

### Phase 5: Family Deployment
- [ ] Multi-user configuration
- [ ] Role-based access
- [ ] Guardian alert system
- [ ] Room node deployment

## Prior Art and Influences

- Agent-OS constitutional framework
- Digital Tractor home server architecture
- NatLangChain sovereignty principles

## License

[TBD - considering AGPL or similar copyleft to ensure sovereignty principles propagate]

## Contributing

This is a family safety tool. Contributions that compromise privacy, add telemetry, or introduce cloud dependencies will not be accepted.

---

*SAIL exists because the knowledge to stay safe already exists—it just isn't where you need it, when you need it.*
