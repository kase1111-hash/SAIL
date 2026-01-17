"""
SAIL Speech-to-Text Base Module

Provides abstract base class for speech-to-text implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class STTEngine(str, Enum):
    """Supported STT engines."""

    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"


class TranscriptionState(str, Enum):
    """State of transcription."""

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech."""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    language: str | None = None
    words: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""

    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    language_confidence: float = 0.0
    duration: float = 0.0
    processing_time: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if transcription is empty."""
        return not self.text.strip()


@dataclass
class STTConfig:
    """Base configuration for speech-to-text."""

    # Engine to use
    engine: STTEngine = STTEngine.WHISPER

    # Model size (tiny, base, small, medium, large, large-v2, large-v3)
    model: str = "base"

    # Language code (None for auto-detection)
    language: str | None = None

    # Device for inference (cpu, cuda, auto)
    device: str = "cpu"

    # Compute type for faster-whisper (int8, float16, float32)
    compute_type: str = "int8"

    # Sample rate (Whisper expects 16kHz)
    sample_rate: int = 16000

    # Temperature for sampling (0 = greedy, higher = more random)
    temperature: float = 0.0

    # Enable word-level timestamps
    word_timestamps: bool = False

    # Minimum confidence threshold
    min_confidence: float = 0.0

    # Maximum audio duration in seconds
    max_duration: float = 30.0


class STTProvider(ABC):
    """
    Abstract base class for speech-to-text providers.

    Implementations should transcribe audio to text using various
    speech recognition engines.
    """

    def __init__(self, config: STTConfig | None = None) -> None:
        """
        Initialize STT provider.

        Args:
            config: STT configuration, uses defaults if None
        """
        self.config = config or STTConfig()
        self._state = TranscriptionState.IDLE
        self._is_loaded = False

    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the speech recognition model.

        Should be called before transcribe() if model loading is deferred.
        """
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """
        Unload the model to free resources.
        """
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (int16 or float32)

        Returns:
            TranscriptionResult with transcribed text
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        Transcribe audio stream with streaming results.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            TranscriptionSegment as they become available
        """
        pass

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to float32 in range [-1, 1].

        Args:
            audio: Input audio array

        Returns:
            Normalized float32 audio
        """
        if audio.dtype == np.float32:
            # Ensure range is [-1, 1]
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                return audio / max_val
            return audio

        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0

        if audio.dtype == np.int32:
            return audio.astype(np.float32) / 2147483648.0

        # Fallback: convert and normalize
        audio_float = audio.astype(np.float32)
        max_val = np.abs(audio_float).max()
        if max_val > 0:
            return audio_float / max_val
        return audio_float

    @property
    def state(self) -> TranscriptionState:
        """Get current transcription state."""
        return self._state

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def __aenter__(self) -> STTProvider:
        """Async context manager entry."""
        await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.unload_model()
