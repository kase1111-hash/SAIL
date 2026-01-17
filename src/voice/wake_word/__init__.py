"""
SAIL Wake Word Detection Module

Provides wake word detection capabilities for activating the voice assistant.
"""

from src.voice.wake_word.base import (
    WakeWordConfig,
    WakeWordDetector,
    WakeWordResult,
    WakeWordState,
)
from src.voice.wake_word.openwakeword import (
    OpenWakeWordConfig,
    OpenWakeWordDetector,
    OpenWakeWordError,
    SimpleWakeWordDetector,
)

__all__ = [
    "OpenWakeWordConfig",
    "OpenWakeWordDetector",
    "OpenWakeWordError",
    "SimpleWakeWordDetector",
    "WakeWordConfig",
    "WakeWordDetector",
    "WakeWordResult",
    "WakeWordState",
]
