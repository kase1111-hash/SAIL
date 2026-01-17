"""
SAIL Text-to-Speech Base Module

Provides abstract base class for text-to-speech implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

import numpy as np


class TTSEngine(str, Enum):
    """Supported TTS engines."""

    PIPER = "piper"
    COQUI = "coqui"


class SpeechState(str, Enum):
    """State of speech synthesis."""

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    ERROR = "error"


class SpeechPriority(int, Enum):
    """Priority levels for speech output."""

    CRITICAL = 100      # Emergency alerts - cannot be interrupted
    HIGH = 75           # Important notifications
    NORMAL = 50         # Regular responses
    LOW = 25            # Background information
    AMBIENT = 10        # Ambient/optional speech


class SpeechMode(str, Enum):
    """Speech delivery modes for different contexts."""

    CALM = "calm"           # Soft, reassuring tone
    NORMAL = "normal"       # Standard conversational tone
    URGENT = "urgent"       # Faster, more emphatic
    CRISIS = "crisis"       # Clear, steady, authoritative


@dataclass
class VoiceProfile:
    """Voice characteristics profile."""

    name: str
    model_id: str
    language: str = "en"
    gender: str | None = None
    description: str = ""

    # Voice characteristics
    base_rate: float = 1.0          # Base speaking rate
    base_pitch: float = 1.0         # Base pitch adjustment
    base_volume: float = 1.0        # Base volume level

    # Mode-specific adjustments
    calm_rate: float = 0.9          # Slower for calm mode
    urgent_rate: float = 1.2        # Faster for urgent mode
    crisis_rate: float = 1.0        # Steady for crisis mode


@dataclass
class SpeechRequest:
    """A request to synthesize and speak text."""

    text: str
    priority: SpeechPriority = SpeechPriority.NORMAL
    mode: SpeechMode = SpeechMode.NORMAL

    # Optional overrides
    rate: float | None = None       # Override speaking rate
    pitch: float | None = None      # Override pitch
    volume: float | None = None     # Override volume

    # Control flags
    interruptible: bool = True      # Can be interrupted by stop command
    wait_for_completion: bool = False  # Wait for speech to finish

    # Metadata
    request_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class SpeechResult:
    """Result of speech synthesis."""

    audio_data: np.ndarray | None = None
    sample_rate: int = 22050
    duration_seconds: float = 0.0
    was_interrupted: bool = False
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """Check if synthesis was successful."""
        return self.error is None and self.audio_data is not None


@dataclass
class TTSConfig:
    """Base configuration for text-to-speech."""

    # Engine selection
    engine: TTSEngine = TTSEngine.PIPER

    # Voice settings
    voice: str = "en_US-lessac-medium"
    language: str = "en"

    # Audio settings
    sample_rate: int = 22050
    output_device: int | None = None

    # Speech characteristics
    rate: float = 1.0               # Speaking rate multiplier (0.5-2.0)
    pitch: float = 1.0              # Pitch adjustment (0.5-2.0)
    volume: float = 1.0             # Volume level (0.0-1.0)

    # Mode-specific rate adjustments
    calm_rate_multiplier: float = 0.9
    urgent_rate_multiplier: float = 1.2
    crisis_rate_multiplier: float = 1.0

    # Behavior settings
    sentence_pause_ms: int = 300    # Pause between sentences
    paragraph_pause_ms: int = 500   # Pause between paragraphs


class TTSProvider(ABC):
    """
    Abstract base class for text-to-speech providers.

    Implementations should synthesize text to speech audio using
    various TTS engines.
    """

    def __init__(self, config: TTSConfig | None = None) -> None:
        """
        Initialize TTS provider.

        Args:
            config: TTS configuration, uses defaults if None
        """
        self.config = config or TTSConfig()
        self._state = SpeechState.IDLE
        self._is_loaded = False
        self._current_voice: VoiceProfile | None = None

    @abstractmethod
    async def load_model(self, voice: str | None = None) -> None:
        """
        Load the TTS model.

        Args:
            voice: Voice model to load, uses config default if None
        """
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model to free resources."""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> SpeechResult:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            SpeechResult with audio data
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize text to audio with streaming output.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as numpy arrays
        """
        pass

    @abstractmethod
    async def list_voices(self) -> list[VoiceProfile]:
        """
        List available voices.

        Returns:
            List of VoiceProfile objects
        """
        pass

    @abstractmethod
    async def get_voice_info(self, voice_id: str) -> VoiceProfile | None:
        """
        Get information about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceProfile or None if not found
        """
        pass

    def get_rate_for_mode(self, mode: SpeechMode) -> float:
        """
        Get the speaking rate for a specific mode.

        Args:
            mode: Speech mode

        Returns:
            Rate multiplier
        """
        base_rate = self.config.rate

        if mode == SpeechMode.CALM:
            return base_rate * self.config.calm_rate_multiplier
        elif mode == SpeechMode.URGENT:
            return base_rate * self.config.urgent_rate_multiplier
        elif mode == SpeechMode.CRISIS:
            return base_rate * self.config.crisis_rate_multiplier
        else:
            return base_rate

    @property
    def state(self) -> SpeechState:
        """Get current synthesis state."""
        return self._state

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def current_voice(self) -> VoiceProfile | None:
        """Get current voice profile."""
        return self._current_voice

    async def __aenter__(self) -> TTSProvider:
        """Async context manager entry."""
        await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.unload_model()


def preprocess_text_for_speech(text: str) -> str:
    """
    Preprocess text for better speech synthesis.

    Handles common patterns that TTS engines may struggle with.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    import re

    # Expand common abbreviations
    abbreviations = {
        r"\bDr\.": "Doctor",
        r"\bMr\.": "Mister",
        r"\bMrs\.": "Misses",
        r"\bMs\.": "Miss",
        r"\bSt\.": "Street",
        r"\bAve\.": "Avenue",
        r"\bBlvd\.": "Boulevard",
        r"\betc\.": "etcetera",
        r"\bi\.e\.": "that is",
        r"\be\.g\.": "for example",
    }

    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Handle numbers with units
    text = re.sub(r"(\d+)%", r"\1 percent", text)
    text = re.sub(r"\$(\d+)", r"\1 dollars", text)

    # Handle time formats
    text = re.sub(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])", r"\1 \2 \3", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_text_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for streaming synthesis.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]
