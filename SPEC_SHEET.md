# SAIL Specification Sheet

## Product Overview

| Attribute | Value |
|-----------|-------|
| **Name** | SAIL (Situational Awareness Interaction Layer) |
| **Type** | Privacy-first AI companion |
| **Deployment** | Self-hosted / local hardware |
| **Primary Purpose** | Contextual guidance for everyday safety and decision-support |
| **Target Users** | Families, individuals seeking privacy-respecting personal safety assistance |

---

## Core Design Principles

| Principle | Description |
|-----------|-------------|
| Sovereignty | All processing local; context never leaves user hardware |
| Ambient | Lives in the moment, not a destination app |
| Non-judgmental | Guides without shaming; respects user autonomy |
| Knows limits | Recommends professionals when appropriate |
| Family-scoped | Multi-user with appropriate access boundaries |

---

## System Architecture

### Hardware Requirements

#### Central Hub (Home Server)
| Component | Specification |
|-----------|---------------|
| **Type** | Home server (Digital Tractor architecture) |
| **Functions** | LLM inference, Knowledge DB, Family configuration |
| **Processing** | Local AI model execution |

#### Satellite Nodes
| Node Type | Hardware | Function |
|-----------|----------|----------|
| Room Node | Raspberry Pi + microphone | In-home voice interface |
| Mobile App | Smartphone | Portable access |
| Vehicle Node | OBD adapter + microphone | Automotive context awareness |

---

## Voice I/O Stack

| Component | Technology | Execution |
|-----------|------------|-----------|
| Speech-to-text | Whisper | Local inference |
| Text-to-speech | Piper / Coqui TTS | Local |
| Wake word detection | OpenWakeWord / Porcupine | Offline |
| LLM backend | Llama 3 / Mistral | Via Ollama or llama.cpp |

---

## Context Buffer System

### Memory Tiers

| Tier | Retention | Description |
|------|-----------|-------------|
| Immediate | Last 5 minutes | Recent context |
| Short-term | Last hour | Extended awareness |
| Session | Since wake/departure | Activity-based context |
| Background | Persistent | User profile data |

### Adaptive Depth Modes

| Situation | Buffer Priority |
|-----------|-----------------|
| Driving | Recent audio + motion state |
| Conversation | Expanded short-term for dialogue continuity |
| Crisis | Locked context, prevents overwrite |

---

## Sensor Fusion Layer

| Input | Source | Application |
|-------|--------|-------------|
| Location | GPS | Jurisdiction detection, unfamiliar area alerts |
| Motion state | Accelerometer | Driving/walking/stationary detection |
| Speed | GPS + accelerometer | Speeding awareness, accident detection |
| Time | System clock + calendar | Contextual awareness (late night, first day at job) |
| Audio environment | Microphone | Voice commands, stress detection, ambient threat cues |
| Biometrics | Wearable (optional) | Heart rate spike = stress indicator |

---

## Intervention Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Ambient** | User-initiated | Passive listening; responds only when addressed |
| **Advisory** | Risk detected | Gentle unprompted suggestions |
| **Guardian** | High-risk patterns | Active alerts for dangerous behavioral patterns |
| **Crisis** | Acute situations | Calm, hands-free walkthrough |

---

## Knowledge Domains

### Legal Basics (Jurisdiction-Aware)
- Traffic stop rights and obligations
- Employment rights (at-will, termination, wages, breaks)
- Contract fundamentals (signing implications, cooling-off periods)
- Consent and age-of-majority laws
- Police interaction guidelines
- Tenant rights

### Personal Safety
- De-escalation scripts
- Situational awareness fundamentals
- Safe meeting protocols (dating, transactions)
- Online interaction verification checklists
- Scam pattern recognition

### Financial Protection
- Wire transfer friction prompts
- Credential entry verification
- Voice clone awareness (grandparent scam detection)
- Pressure/urgency language flags

### Emergency Protocols
- Medical emergency walkthroughs
- Accident scene procedures
- Family emergency contacts and code words
- Location sharing triggers

---

## User Access Model

### Role Definitions

| Role | Access Level | Capabilities |
|------|--------------|--------------|
| Admin | Full | Complete system configuration and oversight |
| User | Standard | Normal SAIL interaction, some restrictions may apply |
| Minor | Limited | Restricted access; guardian alerts enabled |

### Configuration Example

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

---

## Constitutional Constraints

| # | Principle | Constraint |
|---|-----------|------------|
| 1 | SOVEREIGNTY | No data leaves user hardware. No exceptions. |
| 2 | RESPECT | "Stop" / "shut up" / "not now" immediately honored |
| 3 | HUMILITY | State uncertainty; recommend professionals when appropriate |
| 4 | NON-JUDGMENT | Guide without shame; everyone has knowledge gaps |
| 5 | PROPORTIONALITY | Intervention intensity matches actual risk |
| 6 | TRANSPARENCY | Explain why an alert triggered if asked |
| 7 | FAMILY BOUNDARIES | User data siloed; minors' alerts to guardians only with cause |

---

## Development Roadmap

### Phase 1-2: Core Foundation
- Project structure with Python 3.11+ support
- Configuration management with YAML schema validation (Pydantic)
- CLI interface using Click with Rich formatting
- Docker and docker-compose development environment
- Local LLM integration (Ollama, llama.cpp)

### Phase 3-4: Voice I/O
- Local wake word detection (OpenWakeWord, Porcupine)
- Whisper integration for speech-to-text
- Local TTS with natural voice (Piper, Coqui)
- Audio capture and preprocessing pipeline
- Voice command handling

### Phase 5: Context Buffer
- Multi-tier rolling memory architecture
- Adaptive depth modes (driving, conversation, crisis)
- Persistent encrypted storage with SQLite
- Crisis mode context locking

### Phase 6: Sensor Fusion
- GPS/location integration with jurisdiction detection
- Motion state detection
- Temporal awareness
- Audio environment analysis

### Phase 7: Intervention Engine
- Four intervention modes (Ambient, Advisory, Guardian, Crisis)
- Risk assessment engine
- Mode transition logic
- Proportionality checking

### Phase 8: Knowledge Domains
- Jurisdiction-aware legal knowledge base
- Personal safety protocols
- Financial protection patterns
- Emergency procedures
- Vector similarity search (RAG pipeline)

### Phase 9: Multi-User & Family System
- Voice-based user identification
- Role-based access control (Admin/User/Minor)
- Per-user context isolation
- Guardian alert system

### Phase 10: Node Deployment & Integration
- Hub coordinator with request routing
- Room node support (Raspberry Pi)
- Context synchronization between nodes
- End-to-end pipeline integration

---

## Technical Dependencies

| Category | Technologies |
|----------|--------------|
| Speech Recognition | Whisper |
| Text-to-Speech | Piper, Coqui TTS |
| Wake Word | OpenWakeWord, Porcupine |
| LLM Runtime | Ollama, llama.cpp |
| Models | Llama 3, Mistral |
| Hardware | Raspberry Pi (nodes), OBD adapter (vehicle) |

---

## Privacy & Compliance

| Requirement | Implementation |
|-------------|----------------|
| Data locality | All processing on user hardware |
| Cloud dependencies | None |
| Telemetry | Prohibited |
| Third-party data sharing | Prohibited |

---

## License

MIT License - See [LICENSE](LICENSE) for details.
