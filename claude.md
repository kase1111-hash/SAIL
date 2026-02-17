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

SAIL uses a **pipeline architecture** with optional hub-and-node distribution:

```
Pipeline (src/pipeline.py) - Central orchestrator
├── LLM Provider (Ollama/llama.cpp) - Local inference
├── Context Buffer Manager - Multi-tier rolling memory
├── Knowledge Manager + RAG Pipeline - Domain-specific retrieval
├── Intervention Engine - Risk assessment and mode transitions
├── Sensor Fusion - Temporal, location, motion, audio context
└── User Manager (optional) - Multi-user with voice ID

Optional Distribution:
├── Hub Coordinator - Routes requests from satellite nodes
└── Room Nodes (Raspberry Pi + microphone) - Remote voice interfaces
```

### Key Architectural Layers

- **Pipeline** (`src/pipeline.py`) - End-to-end query processing connecting all components
- **Voice I/O Pipeline** - Audio capture → Wake word → STT → Response → TTS
- **Sensor Fusion Layer** - GPS, accelerometer, microphone, calendar context
- **Context Buffer System** - Multi-tier rolling memory (immediate, short-term, session, background)
- **Intervention Engine** - Four modes (Ambient, Advisory, Guardian, Crisis)
- **Knowledge Domains** - Legal, safety, financial, emergency protocols
- **User Management** - Voice identification, role-based access, family system

## Directory Structure

```
/home/user/SAIL/
├── src/                          # Main source code
│   ├── __init__.py               # Package init with __version__
│   ├── cli.py                    # Click CLI entry point
│   ├── pipeline.py               # End-to-end query pipeline (core integration)
│   ├── config/                   # Configuration system (Pydantic + YAML)
│   ├── context/                  # 4-tier context buffer system
│   ├── core/                     # Core application logic
│   │   ├── llm/                  # LLM integration (Ollama, llama.cpp)
│   │   ├── events.py             # Event system
│   │   └── stats.py              # Statistics tracking
│   ├── hub/                      # Hub coordinator & routing (for distributed mode)
│   ├── intervention/             # Intervention modes & risk assessment
│   ├── knowledge/                # Knowledge base with 4 domains
│   │   └── domains/              # Legal, financial, safety, emergency
│   ├── nodes/                    # Satellite node implementations (room node)
│   ├── sensors/                  # Sensor fusion system
│   ├── users/                    # User management & family system
│   └── voice/                    # Voice I/O pipeline
│       ├── audio/                # Audio capture & VAD
│       ├── stt/                  # Speech-to-text (Whisper)
│       ├── tts/                  # Text-to-speech (Piper/Coqui)
│       ├── wake_word/            # Wake word detection
│       ├── manager.py            # Voice input orchestration
│       └── commands.py           # Command detection
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── conftest.py               # Pytest configuration
├── config/
│   └── config.yaml               # Default configuration
├── scripts/                      # Setup and utility scripts
│   ├── run-tests.sh              # Test runner
│   └── setup-dev.sh              # Dev environment setup
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Dev environment
└── pyproject.toml                # Python package configuration
```

## Technology Stack

- **Language:** Python 3.11+
- **LLM Runtime:** Ollama or llama.cpp (local inference)
- **Speech-to-Text:** OpenAI Whisper or faster-whisper
- **Text-to-Speech:** Piper TTS or Coqui TTS
- **Wake Word:** OpenWakeWord or Porcupine
- **Configuration:** Pydantic + YAML
- **Database:** SQLite (local encrypted persistence)
- **CLI:** Click + Rich
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **Code Quality:** ruff, black, mypy

## Key Entry Points

### CLI Commands

```bash
sail run                  # Start the SAIL assistant (voice mode)
sail run --text-mode      # Start in text-only mode (no audio hardware needed)
sail run --hub            # Start in hub mode (accepts node connections)
sail run --verbose        # Enable verbose logging
sail init                 # Initialize a new configuration file
sail config-show          # Display current configuration
sail users                # List configured users
sail check                # Check system requirements and dependencies
```

### Programmatic Usage

The core integration point is the `Pipeline` class in `src/pipeline.py`:

```python
from src.pipeline import Pipeline, QueryResult
from src.config import load_config

config = load_config("config/config.yaml")
pipeline = Pipeline(config)
await pipeline.start()

result: QueryResult = await pipeline.query("What are my rights during a traffic stop?")
print(result.response)
print(result.citations)
print(result.intervention_mode)

await pipeline.stop()
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

### Pipeline (`src/pipeline.py`)

The central orchestrator that connects all subsystems. Processes user queries through context management, knowledge retrieval, intervention assessment, and LLM generation. Returns `QueryResult` with response text, citations, intervention mode, and guardian alerts.

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
- **Ambient**: Passive monitoring, responds when addressed
- **Advisory**: Proactive suggestions on risk detection
- **Guardian**: Active protection for high-risk patterns
- **Crisis**: Emergency response walkthrough

### Knowledge Domains (`src/knowledge/domains/`)

- Legal (jurisdiction-aware traffic, employment, contracts, consent)
- Financial (scam detection, wire transfer friction, credential verification)
- Safety (de-escalation, safe meetings, situational awareness)
- Emergency (medical emergencies, accident procedures, family protocols)

### Voice Pipeline (`src/voice/`)

Complete voice I/O:
- `audio/` - Audio capture with VAD
- `stt/` - Speech-to-text (Whisper integration)
- `tts/` - Text-to-speech (Piper/Coqui)
- `wake_word/` - Wake word detection

### Hub Coordinator (`src/hub/`)

Manages distributed operation (optional):
- Node registration and routing
- Context synchronization
- Request distribution from satellite nodes

### User Management (`src/users/`)

Multi-user family system:
- Voice-based user identification
- Role-based access control (Admin/User/Minor)
- Per-user context isolation
- Guardian alert system for minors

## Important Patterns

### Constitutional Constraints

The LLM has built-in behavioral constraints (`src/core/llm/constitution.py`):
- Never provides illegal advice
- Recommends professionals when appropriate
- Respects user autonomy ("shut up" is always honored)
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

- `README.md` - Project overview, architecture, and quickstart
- `SPEC_SHEET.md` - Technical specifications
- `DEVELOPMENT_GUIDE.md` - 10-phase implementation guide
- `CONTRIBUTING.md` - Contributor guidelines
- `SECURITY.md` - Security policy
- `AUDIT_REPORT.md` - Security audit findings (historical, some issues since resolved)
- `REFOCUS_PLAN.md` - Strategic refactoring plan (historical, largely completed)
- `REVIEW.md` - Initial project review (historical, pre-pipeline integration)

## License

MIT License - See [LICENSE](LICENSE) for details.
