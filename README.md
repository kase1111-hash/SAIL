# SAIL — Situational Awareness Interaction Layer

A privacy-first, locally-hosted AI companion that provides contextual guidance for everyday situations where knowledge gaps can lead to harm. SAIL acts as a personal safety and decision-support layer, running entirely on user-controlled hardware.

## The Problem

Consequential knowledge exists but isn't transmitted. People end up hurt, scammed, arrested, or exploited over situations they could have navigated safely with information that technically exists but is practically inaccessible—buried in legal code, scattered across forums, or held by professionals who charge by the hour.

A teenager with their first job and first car faces hundreds of potential pitfalls: traffic stops, employment rights, contracts, consent laws, financial scams. No one teaches this curriculum because it crosses too many professional domains.

SAIL closes the gap between where knowledge lives and where decisions happen.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally with a model (e.g., Llama 3 or Mistral)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sail.git
cd sail

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install SAIL with all optional dependencies
pip install -e ".[all]"

# Or install with just core dependencies (no voice hardware needed)
pip install -e ".[dev]"

# Initialize configuration
sail init

# Verify your setup
sail check
```

### Running SAIL

```bash
# Start Ollama in a separate terminal
ollama serve

# Run SAIL in text mode (no audio hardware needed)
sail run --text-mode

# Run SAIL with full voice I/O
sail run

# Run in hub mode (accepts connections from room nodes)
sail run --hub --text-mode
```

### Available Commands

```bash
sail run              # Start the assistant
sail init             # Create a configuration file
sail check            # Verify system requirements
sail config-show      # Display current configuration
sail users            # List configured users
```

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

## How It Works

All components are connected through a central `Pipeline` class (`src/pipeline.py`) that orchestrates query processing:

1. User input arrives (via text or voice)
2. Input is added to the context buffer
3. Knowledge domains are searched via RAG pipeline
4. Sensor context (time, location) enriches the query
5. The LLM generates a response with knowledge and context
6. The intervention engine evaluates risk level
7. Response is returned with citations and any interventions

The CLI (`src/cli.py`) provides text mode, voice mode, and hub mode entry points that all route through this pipeline.

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

### Phase 1-2: Core Foundation
- [x] Project structure with Python 3.11+ support
- [x] Configuration management with YAML schema validation
- [x] CLI interface with Click and Rich formatting
- [x] Docker development environment

### Phase 3-4: Voice I/O Foundation
- [x] Local wake word detection (OpenWakeWord, Porcupine)
- [x] Whisper integration for speech-to-text
- [x] Local TTS with natural voice (Piper, Coqui)
- [x] Basic query-response loop
- [x] Voice command handling

### Phase 5: Context Buffer
- [x] Rolling memory file implementation
- [x] Adjustable depth configuration
- [x] Context persistence across sessions
- [x] Crisis mode context locking

### Phase 6: Sensor Fusion
- [x] GPS integration with jurisdiction detection
- [x] Motion state detection
- [x] Calendar/time awareness
- [x] Audio environment analysis
- [x] Biometric integration framework

### Phase 7: Intervention Engine
- [x] Four intervention modes (Ambient/Advisory/Guardian/Crisis)
- [x] Risk assessment engine
- [x] Mode transition logic
- [x] Proportionality checking

### Phase 8: Knowledge Domains
- [x] Jurisdiction detection (state/country)
- [x] Legal knowledge base (traffic, employment, contracts)
- [x] Safety protocol library
- [x] Financial protection patterns
- [x] Emergency procedures

### Phase 9: Multi-User & Family System
- [x] Multi-user configuration
- [x] Role-based access control
- [x] Guardian alert system
- [x] Voice-based user identification
- [x] Per-user context isolation

### Phase 10: Node Deployment & Integration
- [x] Hub coordinator with request routing
- [x] Room node support (Raspberry Pi)
- [x] Mobile app integration framework
- [x] Vehicle node with OBD-II integration
- [x] Context synchronization

## Prior Art and Influences

- Agent-OS constitutional framework
- Digital Tractor home server architecture
- NatLangChain sovereignty principles

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

This is a family safety tool. Contributions that compromise privacy, add telemetry, or introduce cloud dependencies will not be accepted.

## Security

For information about reporting security vulnerabilities, please see our [Security Policy](SECURITY.md).

---

*SAIL exists because the knowledge to stay safe already exists—it just isn't where you need it, when you need it.*
