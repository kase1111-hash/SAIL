"""
SAIL Wake Word Detection Base Module

Provides abstract base class for wake word detection implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sail.voice.audio.capture import AudioChunk


class WakeWordState(str, Enum):
    """State of wake word detection."""

    LISTENING = "listening"
    DETECTED = "detected"
    COOLDOWN = "cooldown"


@dataclass
class WakeWordResult:
    """Result of wake word detection on an audio chunk."""

    detected: bool
    confidence: float
    wake_word: str | None = None
    timestamp: float = 0.0

    def __bool__(self) -> bool:
        """Allow using result directly in boolean context."""
        return self.detected


@dataclass
class WakeWordConfig:
    """Base configuration for wake word detection."""

    # Wake word phrase to detect
    phrase: str = "hey sail"

    # Detection sensitivity (0.0-1.0, higher = more sensitive)
    sensitivity: float = 0.5

    # Cooldown period after detection (seconds)
    cooldown_seconds: float = 2.0

    # Sample rate (must match audio capture)
    sample_rate: int = 16000


class WakeWordDetector(ABC):
    """
    Abstract base class for wake word detection.

    Implementations should detect a specific wake word or phrase
    in audio streams with configurable sensitivity.
    """

    def __init__(self, config: WakeWordConfig | None = None) -> None:
        """
        Initialize wake word detector.

        Args:
            config: Detection configuration, uses defaults if None
        """
        self.config = config or WakeWordConfig()
        self._state = WakeWordState.LISTENING
        self._last_detection_time: float | None = None

    @abstractmethod
    def process(self, chunk: AudioChunk) -> WakeWordResult:
        """
        Process an audio chunk for wake word detection.

        Args:
            chunk: AudioChunk to process

        Returns:
            WakeWordResult with detection results
        """
        pass

    @abstractmethod
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        """
        Process raw audio data for wake word detection.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            WakeWordResult with detection results
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass

    def _check_cooldown(self, current_time: float) -> bool:
        """
        Check if detector is in cooldown period.

        Args:
            current_time: Current timestamp

        Returns:
            True if still in cooldown
        """
        if self._last_detection_time is None:
            return False

        elapsed = current_time - self._last_detection_time
        return elapsed < self.config.cooldown_seconds

    def _update_detection(self, current_time: float) -> None:
        """Update state after detection."""
        self._last_detection_time = current_time
        self._state = WakeWordState.COOLDOWN

    @property
    def state(self) -> WakeWordState:
        """Get current detector state."""
        return self._state

    @property
    def is_listening(self) -> bool:
        """Check if detector is actively listening (not in cooldown)."""
        return self._state == WakeWordState.LISTENING
