"""
SAIL Audio Capture and Processing

This module provides audio capture and processing capabilities for the voice pipeline.
"""

from src.voice.audio.capture import (
    AudioCapture,
    AudioCaptureError,
    AudioChunk,
    AudioDeviceInfo,
    DeviceNotFoundError,
)
from src.voice.audio.vad import VADProcessor, VADResult, VADState

__all__ = [
    "AudioCapture",
    "AudioCaptureError",
    "AudioChunk",
    "AudioDeviceInfo",
    "DeviceNotFoundError",
    "VADProcessor",
    "VADResult",
    "VADState",
]
