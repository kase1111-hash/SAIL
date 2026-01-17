"""
SAIL Test Configuration and Fixtures
"""

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Return a sample configuration dictionary."""
    return {
        "sail": {
            "version": "0.1.0",
            "environment": "testing",
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434",
        },
        "voice": {
            "wake_word": "hey sail",
            "stt": {
                "engine": "whisper",
                "model": "base",
            },
            "tts": {
                "engine": "piper",
                "voice": "en_US-lessac-medium",
            },
        },
        "context": {
            "immediate_duration_seconds": 300,
            "short_term_duration_seconds": 3600,
        },
        "users": [
            {
                "name": "test_user",
                "role": "admin",
                "access": "full",
            }
        ],
    }


@pytest.fixture
def temp_config_file(tmp_path: Path, sample_config: dict[str, Any]) -> Path:
    """Create a temporary config file for testing."""
    import yaml

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path
