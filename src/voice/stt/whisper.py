"""
SAIL Whisper Speech-to-Text Integration

Provides speech-to-text using OpenAI Whisper and faster-whisper.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from src.voice.stt.base import (
    STTConfig,
    STTEngine,
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionState,
)


class WhisperError(Exception):
    """Exception raised for Whisper-related errors."""


@dataclass
class WhisperConfig(STTConfig):
    """Configuration specific to Whisper STT."""

    # Initial prompt to guide transcription
    initial_prompt: str | None = None

    # Condition on previous text (for streaming)
    condition_on_previous_text: bool = True

    # Suppress blank outputs
    suppress_blank: bool = True

    # VAD filter for faster-whisper
    vad_filter: bool = True

    # Beam size for decoding
    beam_size: int = 5

    # Best of N for decoding
    best_of: int = 5


class WhisperProvider(STTProvider):
    """
    Speech-to-text provider using OpenAI Whisper.

    Whisper is an open-source speech recognition model that provides
    high-quality transcription across many languages.
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        """
        Initialize Whisper provider.

        Args:
            config: Whisper configuration, uses defaults if None
        """
        super().__init__(config or WhisperConfig())
        self.config: WhisperConfig = self.config  # Type narrowing

        self._model = None
        self._processor = None

    async def load_model(self) -> None:
        """Load the Whisper model."""
        if self._is_loaded:
            return

        self._state = TranscriptionState.PROCESSING

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            self._is_loaded = True
            self._state = TranscriptionState.IDLE
        except Exception as e:
            self._state = TranscriptionState.ERROR
            raise WhisperError(f"Failed to load Whisper model: {e}") from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading."""
        import whisper

        self._model = whisper.load_model(
            self.config.model,
            device=self.config.device,
        )

    async def unload_model(self) -> None:
        """Unload the model to free resources."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._state = TranscriptionState.IDLE

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array

        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._is_loaded:
            await self.load_model()

        self._state = TranscriptionState.PROCESSING
        start_time = time.perf_counter()

        try:
            # Normalize audio
            audio_normalized = self._normalize_audio(audio)

            # Run transcription in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_normalized,
            )

            processing_time = time.perf_counter() - start_time
            result.processing_time = processing_time

            self._state = TranscriptionState.COMPLETE
            return result

        except Exception as e:
            self._state = TranscriptionState.ERROR
            raise WhisperError(f"Transcription failed: {e}") from e

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptionResult:
        """Synchronous transcription."""
        import whisper

        # Pad/trim to 30 seconds
        audio = whisper.pad_or_trim(audio)

        # Transcribe
        options = {
            "language": self.config.language,
            "temperature": self.config.temperature,
            "initial_prompt": self.config.initial_prompt,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "word_timestamps": self.config.word_timestamps,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
        }

        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        result = self._model.transcribe(audio, **options)

        # Build segments
        segments = []
        for seg in result.get("segments", []):
            segment = TranscriptionSegment(
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
                confidence=seg.get("no_speech_prob", 0.0),
            )

            if self.config.word_timestamps and "words" in seg:
                segment.words = seg["words"]

            segments.append(segment)

        # Calculate duration
        duration = len(audio) / self.config.sample_rate

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
            language_confidence=1.0 - result.get("language_probability", 0.0),
            duration=duration,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        Transcribe audio stream with streaming results.

        Note: Whisper doesn't natively support streaming, so we buffer
        audio and transcribe in chunks.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            TranscriptionSegment as they become available
        """
        if not self._is_loaded:
            await self.load_model()

        buffer: list[np.ndarray] = []
        buffer_duration = 0.0
        chunk_duration = 5.0  # Transcribe every 5 seconds

        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            buffer_duration += len(audio_chunk) / self.config.sample_rate

            if buffer_duration >= chunk_duration:
                # Concatenate buffer
                audio = np.concatenate(buffer)
                buffer.clear()
                buffer_duration = 0.0

                # Transcribe
                result = await self.transcribe(audio)

                for segment in result.segments:
                    yield segment


class FasterWhisperProvider(STTProvider):
    """
    Speech-to-text provider using faster-whisper.

    faster-whisper is a reimplementation of Whisper using CTranslate2,
    providing faster inference with lower memory usage.
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        """
        Initialize faster-whisper provider.

        Args:
            config: Whisper configuration, uses defaults if None
        """
        config = config or WhisperConfig()
        config.engine = STTEngine.FASTER_WHISPER
        super().__init__(config)
        self.config: WhisperConfig = self.config  # Type narrowing

        self._model = None

    async def load_model(self) -> None:
        """Load the faster-whisper model."""
        if self._is_loaded:
            return

        self._state = TranscriptionState.PROCESSING

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            self._is_loaded = True
            self._state = TranscriptionState.IDLE
        except Exception as e:
            self._state = TranscriptionState.ERROR
            raise WhisperError(f"Failed to load faster-whisper model: {e}") from e

    def _load_model_sync(self) -> None:
        """Synchronous model loading."""
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    async def unload_model(self) -> None:
        """Unload the model to free resources."""
        self._model = None
        self._is_loaded = False
        self._state = TranscriptionState.IDLE

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array

        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._is_loaded:
            await self.load_model()

        self._state = TranscriptionState.PROCESSING
        start_time = time.perf_counter()

        try:
            # Normalize audio
            audio_normalized = self._normalize_audio(audio)

            # Run transcription in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_normalized,
            )

            processing_time = time.perf_counter() - start_time
            result.processing_time = processing_time

            self._state = TranscriptionState.COMPLETE
            return result

        except Exception as e:
            self._state = TranscriptionState.ERROR
            raise WhisperError(f"Transcription failed: {e}") from e

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptionResult:
        """Synchronous transcription."""
        # Transcribe
        segments_iter, info = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=self.config.temperature,
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
            vad_filter=self.config.vad_filter,
        )

        # Collect segments
        segments = []
        full_text = []

        for seg in segments_iter:
            segment = TranscriptionSegment(
                text=seg.text.strip(),
                start_time=seg.start,
                end_time=seg.end,
                confidence=1.0 - seg.no_speech_prob,
            )

            if self.config.word_timestamps and seg.words:
                segment.words = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]

            segments.append(segment)
            full_text.append(seg.text.strip())

        # Calculate duration
        duration = len(audio) / self.config.sample_rate

        return TranscriptionResult(
            text=" ".join(full_text),
            segments=segments,
            language=info.language,
            language_confidence=info.language_probability,
            duration=duration,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        Transcribe audio stream with streaming results.

        faster-whisper provides better streaming support with VAD.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            TranscriptionSegment as they become available
        """
        if not self._is_loaded:
            await self.load_model()

        buffer: list[np.ndarray] = []
        buffer_duration = 0.0
        min_chunk_duration = 1.0  # Minimum audio to transcribe

        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            buffer_duration += len(audio_chunk) / self.config.sample_rate

            if buffer_duration >= min_chunk_duration:
                # Concatenate buffer
                audio = np.concatenate(buffer)

                # Run transcription
                loop = asyncio.get_event_loop()
                segments_iter, _ = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        audio,
                        language=self.config.language,
                        beam_size=self.config.beam_size,
                        vad_filter=self.config.vad_filter,
                    ),
                )

                for seg in segments_iter:
                    yield TranscriptionSegment(
                        text=seg.text.strip(),
                        start_time=seg.start,
                        end_time=seg.end,
                        confidence=1.0 - seg.no_speech_prob,
                    )

                # Keep a small overlap for context
                overlap_samples = int(0.5 * self.config.sample_rate)
                if len(audio) > overlap_samples:
                    buffer = [audio[-overlap_samples:]]
                    buffer_duration = 0.5
                else:
                    buffer.clear()
                    buffer_duration = 0.0


def create_stt_provider(config: STTConfig | None = None) -> STTProvider:
    """
    Factory function to create the appropriate STT provider.

    Args:
        config: STT configuration

    Returns:
        STTProvider instance
    """
    config = config or STTConfig()

    if config.engine == STTEngine.FASTER_WHISPER:
        return FasterWhisperProvider(
            WhisperConfig(**{k: v for k, v in config.__dict__.items()})
        )
    else:
        return WhisperProvider(
            WhisperConfig(**{k: v for k, v in config.__dict__.items()})
        )
