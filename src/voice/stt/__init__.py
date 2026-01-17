"""
SAIL Speech-to-Text Module

Provides speech-to-text capabilities using various engines.
"""

from src.voice.stt.base import (
    STTConfig,
    STTEngine,
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionState,
)
from src.voice.stt.whisper import (
    FasterWhisperProvider,
    WhisperConfig,
    WhisperError,
    WhisperProvider,
    create_stt_provider,
)

__all__ = [
    "FasterWhisperProvider",
    "STTConfig",
    "STTEngine",
    "STTProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionState",
    "WhisperConfig",
    "WhisperError",
    "WhisperProvider",
    "create_stt_provider",
]
