"""
SAIL Voice Activity Detection (VAD) Module

Provides voice activity detection using WebRTC VAD for detecting
speech segments in audio streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.voice.audio.capture import AudioChunk


class VADState(str, Enum):
    """Current state of voice activity detection."""

    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


@dataclass
class VADResult:
    """Result of VAD processing on an audio chunk."""

    is_speech: bool
    state: VADState
    confidence: float
    speech_duration_ms: float
    silence_duration_ms: float


@dataclass
class VADConfig:
    """Configuration for VAD processor."""

    # WebRTC VAD aggressiveness (0-3, higher = more aggressive filtering)
    aggressiveness: int = 2

    # Minimum speech duration to trigger speech_start (ms)
    min_speech_duration_ms: float = 250.0

    # Silence duration to trigger speech_end (ms)
    silence_threshold_ms: float = 500.0

    # Frame duration must be 10, 20, or 30 ms for WebRTC VAD
    frame_duration_ms: int = 30

    # Sample rate must be 8000, 16000, 32000, or 48000 for WebRTC VAD
    sample_rate: int = 16000


@dataclass
class SpeechSegment:
    """A detected speech segment."""

    start_time: float
    end_time: float | None = None
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    is_complete: bool = False

    @property
    def duration_ms(self) -> float:
        """Get the duration of the segment in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def get_audio(self) -> np.ndarray:
        """Combine all audio chunks into a single array."""
        if not self.audio_chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.audio_chunks)


class VADProcessor:
    """
    Voice Activity Detection processor using WebRTC VAD.

    Processes audio chunks and detects speech segments with
    state tracking for speech start/end transitions.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        """
        Initialize VAD processor.

        Args:
            config: VAD configuration, uses defaults if None
        """
        self.config = config or VADConfig()
        self._vad = None
        self._init_vad()

        # State tracking
        self._current_state = VADState.SILENCE
        self._speech_start_time: float | None = None
        self._silence_start_time: float | None = None
        self._speech_frame_count = 0
        self._silence_frame_count = 0

        # Current speech segment
        self._current_segment: SpeechSegment | None = None
        self._completed_segments: list[SpeechSegment] = []

        # Frame timing
        self._frames_for_min_speech = int(
            self.config.min_speech_duration_ms / self.config.frame_duration_ms
        )
        self._frames_for_silence_threshold = int(
            self.config.silence_threshold_ms / self.config.frame_duration_ms
        )

    def _init_vad(self) -> None:
        """Initialize the WebRTC VAD instance."""
        import webrtcvad

        self._vad = webrtcvad.Vad(self.config.aggressiveness)

    def reset(self) -> None:
        """Reset the VAD state."""
        self._current_state = VADState.SILENCE
        self._speech_start_time = None
        self._silence_start_time = None
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._current_segment = None
        self._completed_segments.clear()

    def _is_speech_frame(self, audio_data: bytes) -> bool:
        """
        Check if an audio frame contains speech.

        Args:
            audio_data: Audio data as bytes (int16 format)

        Returns:
            True if speech is detected
        """
        try:
            return self._vad.is_speech(audio_data, self.config.sample_rate)
        except Exception:
            return False

    def process(self, chunk: AudioChunk) -> VADResult:
        """
        Process an audio chunk and detect voice activity.

        Args:
            chunk: AudioChunk to process

        Returns:
            VADResult with detection results
        """
        # Convert to bytes for WebRTC VAD
        audio_bytes = chunk.to_bytes()

        # Calculate number of frames in this chunk
        bytes_per_frame = int(
            2 * self.config.sample_rate * self.config.frame_duration_ms / 1000
        )
        num_frames = len(audio_bytes) // bytes_per_frame

        # Process each frame
        speech_frames = 0
        for i in range(num_frames):
            frame = audio_bytes[i * bytes_per_frame : (i + 1) * bytes_per_frame]
            if len(frame) == bytes_per_frame and self._is_speech_frame(frame):
                speech_frames += 1

        # Determine if this chunk is speech (majority voting)
        is_speech = speech_frames > num_frames / 2 if num_frames > 0 else False

        # Update state tracking
        previous_state = self._current_state

        if is_speech:
            self._speech_frame_count += 1
            self._silence_frame_count = 0

            if self._current_state == VADState.SILENCE:
                # Potential speech start
                if self._speech_frame_count >= self._frames_for_min_speech:
                    self._current_state = VADState.SPEECH_START
                    self._speech_start_time = chunk.timestamp
                    self._current_segment = SpeechSegment(
                        start_time=chunk.timestamp
                    )
            elif self._current_state in (VADState.SPEECH_START, VADState.SPEECH_END):
                self._current_state = VADState.SPEECH

            # Add audio to current segment
            if self._current_segment is not None:
                self._current_segment.audio_chunks.append(chunk.to_int16())

        else:
            self._silence_frame_count += 1

            if self._current_state in (VADState.SPEECH_START, VADState.SPEECH):
                # Still in speech but silence detected
                if self._silence_frame_count >= self._frames_for_silence_threshold:
                    self._current_state = VADState.SPEECH_END

                    # Complete the current segment
                    if self._current_segment is not None:
                        self._current_segment.end_time = chunk.timestamp
                        self._current_segment.is_complete = True
                        self._completed_segments.append(self._current_segment)
                        self._current_segment = None

            elif self._current_state == VADState.SPEECH_END:
                self._current_state = VADState.SILENCE
                self._speech_frame_count = 0

        # Calculate durations
        speech_duration = 0.0
        silence_duration = 0.0

        if self._speech_start_time is not None:
            speech_duration = (
                self._speech_frame_count * self.config.frame_duration_ms
            )

        if self._silence_frame_count > 0:
            silence_duration = (
                self._silence_frame_count * self.config.frame_duration_ms
            )

        # Calculate confidence (ratio of speech frames)
        confidence = speech_frames / num_frames if num_frames > 0 else 0.0

        return VADResult(
            is_speech=is_speech,
            state=self._current_state,
            confidence=confidence,
            speech_duration_ms=speech_duration,
            silence_duration_ms=silence_duration,
        )

    def get_completed_segments(self) -> list[SpeechSegment]:
        """
        Get all completed speech segments and clear the buffer.

        Returns:
            List of completed SpeechSegment objects
        """
        segments = self._completed_segments.copy()
        self._completed_segments.clear()
        return segments

    def get_current_segment(self) -> SpeechSegment | None:
        """
        Get the current in-progress speech segment.

        Returns:
            Current SpeechSegment or None if no speech in progress
        """
        return self._current_segment

    @property
    def is_speaking(self) -> bool:
        """Check if speech is currently detected."""
        return self._current_state in (
            VADState.SPEECH_START,
            VADState.SPEECH,
        )

    @property
    def state(self) -> VADState:
        """Get current VAD state."""
        return self._current_state


class AudioAccumulator:
    """
    Accumulates audio until speech ends, then provides the complete utterance.

    Useful for collecting speech for transcription.
    """

    def __init__(
        self,
        vad: VADProcessor | None = None,
        max_duration_ms: float = 30000.0,
        pre_speech_buffer_ms: float = 300.0,
    ) -> None:
        """
        Initialize audio accumulator.

        Args:
            vad: VAD processor to use, creates new one if None
            max_duration_ms: Maximum duration to accumulate (30 seconds default)
            pre_speech_buffer_ms: Amount of audio to keep before speech start
        """
        self.vad = vad or VADProcessor()
        self.max_duration_ms = max_duration_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms

        self._pre_speech_buffer: list[np.ndarray] = []
        self._speech_buffer: list[np.ndarray] = []
        self._accumulated_duration_ms = 0.0
        self._is_accumulating = False

    def reset(self) -> None:
        """Reset accumulator state."""
        self.vad.reset()
        self._pre_speech_buffer.clear()
        self._speech_buffer.clear()
        self._accumulated_duration_ms = 0.0
        self._is_accumulating = False

    def process(self, chunk: AudioChunk) -> tuple[np.ndarray | None, bool]:
        """
        Process an audio chunk and accumulate speech.

        Args:
            chunk: AudioChunk to process

        Returns:
            Tuple of (accumulated_audio, is_complete)
            accumulated_audio is None until speech ends
        """
        result = self.vad.process(chunk)

        # Manage pre-speech buffer
        if not self._is_accumulating:
            self._pre_speech_buffer.append(chunk.to_int16())
            # Keep only recent pre-speech audio
            max_pre_chunks = int(
                self.pre_speech_buffer_ms / chunk.duration_ms
            )
            if len(self._pre_speech_buffer) > max_pre_chunks:
                self._pre_speech_buffer.pop(0)

        # Handle state transitions
        if result.state == VADState.SPEECH_START:
            self._is_accumulating = True
            # Include pre-speech buffer
            self._speech_buffer = self._pre_speech_buffer.copy()
            self._accumulated_duration_ms = sum(
                len(b) / chunk.sample_rate * 1000 for b in self._speech_buffer
            )

        if self._is_accumulating:
            self._speech_buffer.append(chunk.to_int16())
            self._accumulated_duration_ms += chunk.duration_ms

            # Check for max duration
            if self._accumulated_duration_ms >= self.max_duration_ms:
                audio = self._finalize()
                return audio, True

        if result.state == VADState.SPEECH_END:
            audio = self._finalize()
            return audio, True

        return None, False

    def _finalize(self) -> np.ndarray:
        """Finalize and return accumulated audio."""
        if not self._speech_buffer:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()
        self._pre_speech_buffer.clear()
        self._accumulated_duration_ms = 0.0
        self._is_accumulating = False
        return audio

    @property
    def is_accumulating(self) -> bool:
        """Check if currently accumulating speech."""
        return self._is_accumulating

    @property
    def accumulated_duration_ms(self) -> float:
        """Get duration of accumulated audio in milliseconds."""
        return self._accumulated_duration_ms
