# Contributing to SAIL

Thank you for your interest in contributing to SAIL (Situational Awareness Interaction Layer). This document provides guidelines and information for contributors.

## Core Principles

SAIL is a privacy-first, locally-hosted AI companion. **All contributions must align with these non-negotiable principles:**

1. **No Cloud Dependencies**: All processing must remain local. No external API calls for core functionality.
2. **No Telemetry**: No data collection, analytics, or phone-home features.
3. **No Data Exfiltration**: User context and conversations never leave the user's hardware.
4. **Sovereignty First**: Users must maintain complete control over their data and system.

Contributions that compromise these principles will not be accepted.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A local development environment

### Setting Up Your Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sail.git
   cd sail
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify your setup**
   ```bash
   pytest
   ```

### Using Docker (Alternative)

```bash
docker-compose up -d
docker-compose exec sail pytest
```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Code Style

SAIL uses automated code formatting and linting:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **mypy** for type checking

Run all checks locally:
```bash
# Format code
black src tests

# Run linter
ruff check src tests

# Type checking
mypy src
```

Pre-commit hooks will run these automatically on each commit.

### Type Hints

All new code must include type hints:

```python
def process_context(
    buffer: ContextBuffer,
    mode: InterventionMode,
    *,
    timeout: float = 30.0,
) -> ProcessingResult:
    """Process the context buffer with the given intervention mode."""
    ...
```

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest markers appropriately:
  - `@pytest.mark.unit` - Fast unit tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.slow` - Slow-running tests

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific markers
pytest -m unit
pytest -m "not slow"
```

## Making Changes

### 1. Create an Issue First

For significant changes, open an issue to discuss your proposal before starting work. This helps ensure your contribution aligns with the project's direction.

### 2. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages
- Follow existing code patterns and architecture
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the full test suite
pytest

# Run linting
ruff check src tests

# Run type checking
mypy src
```

### 4. Submit a Pull Request

- Fill out the pull request template completely
- Reference any related issues
- Ensure all CI checks pass
- Be responsive to review feedback

## Pull Request Guidelines

### What Makes a Good PR

- **Focused**: One logical change per PR
- **Tested**: Includes tests for new functionality
- **Documented**: Updates relevant documentation
- **Clean**: Passes all linting and type checks
- **Clear**: Has a descriptive title and explanation

### PR Checklist

Before submitting, ensure:

- [ ] Code follows the project style guide
- [ ] All tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes and motivation

## Areas for Contribution

### High Priority

- Knowledge domain expansion (legal, safety, financial information)
- Voice I/O improvements (wake word accuracy, TTS quality)
- Sensor integration for new platforms
- Documentation and tutorials

### Good First Issues

Look for issues labeled `good first issue` for beginner-friendly contributions.

### Documentation

Documentation improvements are always welcome:
- Clarifying existing docs
- Adding examples
- Fixing typos
- Translating documentation

## Architecture Overview

Understanding the codebase structure helps with contributions:

```
src/
├── cli.py              # Command-line interface
├── core/               # Core LLM and utilities
├── config/             # Configuration management
├── voice/              # Voice I/O (wake word, STT, TTS)
├── context/            # Context buffer system
├── sensors/            # Sensor fusion
├── users/              # Multi-user management
├── intervention/       # Intervention mode engine
├── knowledge/          # Knowledge domains
├── hub/                # Central hub coordinator
├── nodes/              # Satellite nodes (room, mobile, vehicle)
└── deployment/         # Deployment utilities
```

See `DEVELOPMENT_GUIDE.md` for detailed architecture documentation.

## Communication

- **Issues**: For bug reports, feature requests, and discussions
- **Pull Requests**: For code contributions

## License

By contributing to SAIL, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in release notes and the project's contributor list. Thank you for helping make SAIL better!

---

*Questions? Open an issue and we'll be happy to help.*
