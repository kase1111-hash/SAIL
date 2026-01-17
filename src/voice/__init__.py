"""
SAIL Voice Input Pipeline

This module provides the complete voice input pipeline for SAIL,
including audio capture, wake word detection, speech-to-text,
and command detection.

Phase 3 Implementation:
- Audio capture with sounddevice
- Voice Activity Detection (VAD) with WebRTC VAD
- Wake word detection with OpenWakeWord
- Speech-to-text with Whisper/faster-whisper
- Stop command detection for respect protocol
"""

from src.voice.commands import (
    CommandDetector,
    CommandPattern,
    CommandPriority,
    CommandResult,
    CommandType,
    QuickStopDetector,
)
from src.voice.manager import (
    VoiceInputEvent,
    VoiceInputManager,
    VoiceInputResult,
    VoiceInputState,
    create_voice_input_manager,
)

__all__ = [
    # Command detection
    "CommandDetector",
    "CommandPattern",
    "CommandPriority",
    "CommandResult",
    "CommandType",
    "QuickStopDetector",
    # Voice input manager
    "VoiceInputEvent",
    "VoiceInputManager",
    "VoiceInputResult",
    "VoiceInputState",
    "create_voice_input_manager",
]
