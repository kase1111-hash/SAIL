# SAIL - Claude.md

## Project Overview

**SAIL** (Situational Awareness Interaction Layer) is a privacy-first, locally-hosted AI companion designed to provide contextual guidance for everyday situations where knowledge gaps can lead to harm. It runs entirely on user-controlled hardware with zero cloud dependencies.

### Core Design Principles

1. **Sovereignty** - All data stays local; no external communication
2. **Ambient, Not Destination** - Lives in the moment, not an app you open
3. **Non-judgmental** - Guides without shaming; respects user autonomy
4. **Knows Its Limits** - Recommends professionals when appropriate
5. **Family-Scoped** - Multi-user with appropriate access boundaries

## Architecture

SAIL follows a **distributed hub-and-node architecture**:

```
Central Hub (Home Server)
├── LLM Inference Engine (Ollama/llama.cpp)
├── Knowledge Database (SQLite)
├── Family Configuration
├── Context Buffer Manager
└── Hub Coordinator

Connected Nodes:
├── Room Nodes (Raspberry Pi + microphone)
├── Mobile App (smartphone)
└── Vehicle Node (OBD adapter + microphone)
```

### Key Architectural Layers

- **Voice I/O Pipeline** - Audio capture → Wake word → STT → Response → TTS
- **Sensor Fusion Layer** - GPS, accelerometer, microphone, calendar, biometrics
- **Context Buffer System** - Multi-tier rolling memory (immediate, short-term, session, background)
- **Intervention Engine** - Four modes (Ambient, Advisory, Guardian, Crisis)
- **Knowledge Domains** - Legal, safety, financial, emergency protocols
- **User Management** - Voice identification, role-based access, family system

## Directory Structure

```
/home/user/SAIL/
├── src/                          # Main source code
│   ├── cli.py                    # Click CLI entry point
│   ├── config/                   # Configuration system (Pydantic)
│   ├── context/                  # 4-tier context buffer system
│   ├── core/llm/                 # LLM integration & constitutional constraints
│   ├── deployment/               # Health checks, OTA, provisioning
│   ├── hub/                      # Hub coordinator & routing
│   ├── intervention/             # Intervention modes & risk assessment
│   ├── knowledge/                # Knowledge base with 4 domains
│   │   └── domains/              # Legal, financial, safety, emergency
│   ├── nodes/                    # Satellite node implementations
│   ├── sensors/                  # Sensor fusion system
│   ├── users/                    # User management & family system
│   └── voice/                    # Voice I/O pipeline
│       ├── audio/                # Audio capture & VAD
│       ├── stt/                  # Speech-to-text (Whisper)
│       ├── tts/                  # Text-to-speech (Piper/Coqui)
│       └── wake_word/            # Wake word detection
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests (16 modules)
│   └── integration/              # Integration tests
├── config/
│   └── config.yaml               # Default configuration
├── scripts/                      # Setup and utility scripts
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Dev environment
└── pyproject.toml               # Python package configuration
```

## Technology Stack

- **Language:** Python 3.11+
- **LLM Runtime:** Ollama or llama.cpp (local inference)
- **Speech-to-Text:** OpenAI Whisper or faster-whisper
- **Text-to-Speech:** Piper TTS or Coqui TTS
- **Wake Word:** OpenWakeWord or Porcupine
- **Configuration:** Pydantic + YAML
- **Database:** SQLite (local persistence)
- **CLI:** Click + Rich
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **Code Quality:** ruff, black, mypy

## Key Entry Points

### CLI Commands

```bash
sail run          # Start the SAIL assistant
sail init         # Initialize configuration
sail system-check # Verify system readiness
```

### Programmatic Usage

```python
from sail import get_config, load_config, Config
from sail.voice import VoiceInputManager
from sail.intervention import InterventionEngine
from sail.users import UserManager
from sail.hub import HubCoordinator
```

## Development Guidelines

### Code Style

- **Type hints** are required throughout the codebase
- **Async/await** patterns for I/O operations
- **Pydantic models** for configuration and data validation
- Use `structlog` for logging

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Via Docker
docker-compose up --profile testing sail-test
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:
- ruff (linting)
- black (formatting)
- mypy (type checking)

### Docker Development

```bash
# Start development environment
docker-compose up sail-dev

# Start with Ollama
docker-compose up sail-dev ollama
```

## Core Components

### Configuration (`src/config/`)

All configuration uses Pydantic models with YAML loading. Main config file: `config/config.yaml`

### Context Buffer (`src/context/`)

4-tier memory system:
- **Immediate**: 5 minutes
- **Short-term**: 1 hour
- **Session**: Duration of interaction
- **Background**: Persistent patterns

### Intervention Engine (`src/intervention/`)

Four operating modes:
- **Ambient**: Passive monitoring
- **Advisory**: Proactive suggestions
- **Guardian**: Active protection (for minors)
- **Crisis**: Emergency response

### Knowledge Domains (`src/knowledge/domains/`)

- Legal (jurisdiction-aware)
- Financial (scam detection, contracts)
- Safety (emergency protocols)
- Emergency (crisis response)

### Voice Pipeline (`src/voice/`)

Complete voice I/O:
- `audio/` - Audio capture with VAD
- `stt/` - Speech-to-text (Whisper integration)
- `tts/` - Text-to-speech (Piper/Coqui)
- `wake_word/` - Wake word detection

### Hub Coordinator (`src/hub/`)

Manages distributed system:
- Node registration and routing
- Context synchronization
- Request distribution

## Important Patterns

### Constitutional Constraints

The LLM has built-in behavioral constraints (`src/core/llm/constitution.py`):
- Never provides illegal advice
- Recommends professionals when appropriate
- Respects user autonomy
- Age-appropriate responses

### Family System

Role-based access control:
- **Admin**: Full access, configuration
- **User**: Standard adult access
- **Minor**: Restricted, guardian alerts

### Error Handling

Use structured error types from each module. Critical operations should have fallbacks and graceful degradation.

## Environment Variables

- `SAIL_CONFIG_PATH` - Path to configuration file
- `SAIL_ENV` - Environment (development, testing, production)
- `PYTHONPATH` - Set to `/app/src` in Docker

## Related Documentation

- `README.md` - Project overview and roadmap
- `SPEC_SHEET.md` - Technical specifications
- `DEVELOPMENT_GUIDE.md` - 10-phase implementation guide
- `CONTRIBUTING.md` - Contributor guidelines
- `SECURITY.md` - Security policy
- `AUDIT_REPORT.md` - Security audit findings

## License

AGPL-3.0+ - All modifications must be open source
