# Changelog

All notable changes to SAIL (Situational Awareness Interaction Layer) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project documentation (CONTRIBUTING.md, SECURITY.md, CHANGELOG.md)
- GitHub issue and pull request templates

## [0.1.0] - 2026-01-23

### Added

#### Phase 1-2: Core Foundation
- Project structure with Python 3.11+ support
- Configuration management with YAML schema validation (Pydantic)
- CLI interface using Click with Rich formatting
- Docker and docker-compose development environment
- Pre-commit hooks for code quality (ruff, black, mypy)

#### Phase 3-4: Voice I/O Foundation
- Wake word detection support (OpenWakeWord, Porcupine)
- Speech-to-text integration (Whisper, faster-whisper)
- Text-to-speech synthesis (Piper TTS, Coqui TTS)
- Audio capture and preprocessing pipeline
- Voice command handling system
- Voice I/O manager for orchestration

#### Phase 5: Context Buffer System
- Multi-tier rolling memory architecture:
  - Immediate context (last 5 minutes)
  - Short-term context (last hour)
  - Session context (since wake/departure)
  - Background context (persistent user profile)
- Adaptive depth modes (driving, conversation, crisis)
- Persistent encrypted storage with SQLite
- Crisis mode context locking
- Context lifecycle management

#### Phase 6: Sensor Fusion
- GPS/location integration with jurisdiction detection
- Motion state detection (driving/walking/stationary)
- Temporal awareness (time-based context)
- Audio environment analysis
- Biometric integration framework
- Sensor fusion engine for combined context

#### Phase 7: Intervention Engine
- Four intervention modes:
  - Ambient: Passive listening, responds when addressed
  - Advisory: Gentle unprompted suggestions on risk detection
  - Guardian: Active alerts for high-risk patterns
  - Crisis: Calm, hands-free walkthrough for acute situations
- Risk assessment engine
- Mode transition logic
- Proportionality checking
- Constitutional constraints enforcement

#### Phase 8: Knowledge Domains
- Legal knowledge base (jurisdiction-aware):
  - Traffic stop rights and obligations
  - Employment rights
  - Contract fundamentals
  - Consent and age-of-majority laws
- Personal safety protocols
- Financial protection patterns
- Emergency procedures
- Vector similarity search (RAG pipeline)
- Domain query routing

#### Phase 9: Multi-User & Family System
- Voice-based user identification
- Role-based access control (Admin/User/Minor)
- Per-user context isolation and data siloing
- Guardian alert system
- Custom user briefings
- Family configuration management

#### Phase 10: Node Deployment & Integration
- Hub coordinator with request routing
- Room node support (Raspberry Pi with microphone)
- Mobile app integration framework
- Vehicle node with OBD-II integration
- Context synchronization between nodes
- Node discovery and registration
- Deployment utilities

### Infrastructure
- Comprehensive test suite (unit and integration tests)
- Type hints throughout codebase
- Structured logging with structlog
- Async support with aiofiles
- LLM integration via Ollama

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-23 | Initial release with all 10 phases implemented |

## Upgrade Notes

### Upgrading to 0.1.0

This is the initial release. For new installations:

1. Ensure Python 3.11+ is installed
2. Install with: `pip install -e ".[all]"`
3. Initialize configuration: `sail init`
4. Run system check: `sail system-check`

See the README and DEVELOPMENT_GUIDE for detailed setup instructions.
