"""
SAIL Text-to-Speech Module

Provides text-to-speech capabilities for voice output.

Phase 4 Implementation:
- TTS provider abstraction with Piper integration
- Audio output management with priority queuing
- Interruptible speech system (respect protocol)
- Speech modes: calm, normal, urgent, crisis
"""

from src.voice.tts.base import (
    SpeechMode,
    SpeechPriority,
    SpeechRequest,
    SpeechResult,
    SpeechState,
    TTSConfig,
    TTSEngine,
    TTSProvider,
    VoiceProfile,
    preprocess_text_for_speech,
    split_text_into_sentences,
)
from src.voice.tts.manager import (
    SpeakResult,
    VoiceOutputEvent,
    VoiceOutputManager,
    VoiceOutputState,
    create_voice_output_manager,
)
from src.voice.tts.output import (
    AudioOutputManager,
    OutputState,
    PriorityAudioQueue,
    QueuedAudio,
)
from src.voice.tts.piper import (
    PiperConfig,
    PiperError,
    PiperProvider,
    create_tts_provider,
)

__all__ = [
    # Base types
    "SpeechMode",
    "SpeechPriority",
    "SpeechRequest",
    "SpeechResult",
    "SpeechState",
    "TTSConfig",
    "TTSEngine",
    "TTSProvider",
    "VoiceProfile",
    "preprocess_text_for_speech",
    "split_text_into_sentences",
    # Piper provider
    "PiperConfig",
    "PiperError",
    "PiperProvider",
    "create_tts_provider",
    # Audio output
    "AudioOutputManager",
    "OutputState",
    "PriorityAudioQueue",
    "QueuedAudio",
    # Voice output manager
    "SpeakResult",
    "VoiceOutputEvent",
    "VoiceOutputManager",
    "VoiceOutputState",
    "create_voice_output_manager",
]
