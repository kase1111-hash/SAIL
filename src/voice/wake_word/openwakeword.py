"""
SAIL OpenWakeWord Integration

Provides wake word detection using the OpenWakeWord library.
Supports custom wake word models and built-in wake words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.voice.wake_word.base import (
    WakeWordConfig,
    WakeWordDetector,
    WakeWordResult,
    WakeWordState,
)

if TYPE_CHECKING:
    from src.voice.audio.capture import AudioChunk


class OpenWakeWordError(Exception):
    """Exception raised for OpenWakeWord-related errors."""


@dataclass
class OpenWakeWordConfig(WakeWordConfig):
    """Configuration for OpenWakeWord detector."""

    # Path to custom model file (None for built-in models)
    model_path: Path | None = None

    # Built-in model names to load (used if model_path is None)
    builtin_models: list[str] = field(
        default_factory=lambda: ["hey_jarvis_v0.1", "alexa_v0.1"]
    )

    # Inference framework ("onnx" or "tflite")
    inference_framework: str = "onnx"

    # Number of audio chunks to buffer for detection
    chunk_buffer_size: int = 5

    # Verification threshold (requires multiple consecutive detections)
    verification_threshold: int = 1


class OpenWakeWordDetector(WakeWordDetector):
    """
    Wake word detector using OpenWakeWord.

    OpenWakeWord is an open-source wake word detection library that
    provides both pre-trained models and support for custom wake words.
    """

    def __init__(self, config: OpenWakeWordConfig | None = None) -> None:
        """
        Initialize OpenWakeWord detector.

        Args:
            config: Detection configuration, uses defaults if None

        Raises:
            OpenWakeWordError: If OpenWakeWord cannot be initialized
        """
        super().__init__(config or OpenWakeWordConfig())
        self.config: OpenWakeWordConfig = self.config  # Type narrowing

        self._model = None
        self._model_names: list[str] = []
        self._audio_buffer: list[np.ndarray] = []
        self._consecutive_detections: dict[str, int] = {}

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the OpenWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            # Load models
            if self.config.model_path is not None:
                # Load custom model
                self._model = Model(
                    wakeword_models=[str(self.config.model_path)],
                    inference_framework=self.config.inference_framework,
                )
                self._model_names = [self.config.model_path.stem]
            else:
                # Load built-in models
                # Download models if not present
                openwakeword.utils.download_models(self.config.builtin_models)
                self._model = Model(
                    wakeword_models=self.config.builtin_models,
                    inference_framework=self.config.inference_framework,
                )
                self._model_names = self.config.builtin_models

            # Initialize detection counters
            self._consecutive_detections = dict.fromkeys(self._model_names, 0)

        except ImportError as e:
            raise OpenWakeWordError(
                "OpenWakeWord is not installed. Install with: pip install openwakeword"
            ) from e
        except Exception as e:
            raise OpenWakeWordError(f"Failed to initialize OpenWakeWord: {e}") from e

    def reset(self) -> None:
        """Reset detector state."""
        self._audio_buffer.clear()
        self._consecutive_detections = dict.fromkeys(self._model_names, 0)
        self._state = WakeWordState.LISTENING
        self._last_detection_time = None

        if self._model is not None:
            self._model.reset()

    def process(self, chunk: AudioChunk) -> WakeWordResult:
        """
        Process an audio chunk for wake word detection.

        Args:
            chunk: AudioChunk to process

        Returns:
            WakeWordResult with detection results
        """
        return self.process_audio(
            audio=chunk.to_int16(),
            sample_rate=chunk.sample_rate,
            timestamp=chunk.timestamp,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        timestamp: float = 0.0,
    ) -> WakeWordResult:
        """
        Process raw audio data for wake word detection.

        Args:
            audio: Audio data as numpy array (int16 format expected)
            sample_rate: Sample rate of the audio
            timestamp: Timestamp of the audio

        Returns:
            WakeWordResult with detection results
        """
        # Check cooldown
        if self._check_cooldown(timestamp):
            self._state = WakeWordState.COOLDOWN
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
            )

        self._state = WakeWordState.LISTENING

        if self._model is None:
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
            )

        # Convert to float32 if needed (OpenWakeWord expects float32 in range [-1, 1])
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample if needed (OpenWakeWord expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling - for production, use proper resampling
            ratio = 16000 / sample_rate
            new_length = int(len(audio_float) * ratio)
            indices = np.linspace(0, len(audio_float) - 1, new_length).astype(int)
            audio_float = audio_float[indices]

        # Run prediction
        predictions = self._model.predict(audio_float)

        # Find best detection
        best_model = None
        best_confidence = 0.0

        for model_name in self._model_names:
            if model_name in predictions:
                confidence = float(predictions[model_name])

                # Check against sensitivity threshold
                threshold = 1.0 - self.config.sensitivity
                if confidence > threshold:
                    self._consecutive_detections[model_name] += 1

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_model = model_name
                else:
                    self._consecutive_detections[model_name] = 0

        # Check if we have enough consecutive detections
        detected = False
        detected_model = None

        if best_model is not None:
            if self._consecutive_detections[best_model] >= self.config.verification_threshold:
                detected = True
                detected_model = best_model
                self._update_detection(timestamp)

                # Reset consecutive count after detection
                self._consecutive_detections[best_model] = 0

        return WakeWordResult(
            detected=detected,
            confidence=best_confidence,
            wake_word=detected_model,
            timestamp=timestamp,
        )

    def get_available_models(self) -> list[str]:
        """
        Get list of available wake word models.

        Returns:
            List of model names
        """
        return self._model_names.copy()


class SimpleWakeWordDetector(WakeWordDetector):
    """
    A simple wake word detector using keyword spotting in transcribed text.

    This is a fallback detector that uses STT output to detect wake words.
    Less efficient but works without specialized wake word models.
    """

    def __init__(self, config: WakeWordConfig | None = None) -> None:
        """
        Initialize simple wake word detector.

        Args:
            config: Detection configuration
        """
        super().__init__(config)
        self._wake_words = self._normalize_wake_words()

    def _normalize_wake_words(self) -> list[str]:
        """Normalize wake word phrase into searchable variants."""
        phrase = self.config.phrase.lower().strip()
        variants = [phrase]

        # Add common variations
        if " " in phrase:
            # "hey sail" -> ["hey sail", "haysail", "hey sale", etc.]
            variants.append(phrase.replace(" ", ""))

        return variants

    def reset(self) -> None:
        """Reset detector state."""
        self._state = WakeWordState.LISTENING
        self._last_detection_time = None

    def process(self, chunk: AudioChunk) -> WakeWordResult:
        """
        Process audio chunk - always returns not detected.

        Use detect_in_text() for actual detection with this detector.
        """
        return WakeWordResult(
            detected=False,
            confidence=0.0,
            timestamp=chunk.timestamp,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        timestamp: float = 0.0,
    ) -> WakeWordResult:
        """
        Process audio - always returns not detected.

        Use detect_in_text() for actual detection with this detector.
        """
        return WakeWordResult(
            detected=False,
            confidence=0.0,
            timestamp=timestamp,
        )

    def detect_in_text(self, text: str, timestamp: float = 0.0) -> WakeWordResult:
        """
        Detect wake word in transcribed text.

        Args:
            text: Transcribed text to search
            timestamp: Timestamp of the audio

        Returns:
            WakeWordResult with detection results
        """
        # Check cooldown
        if self._check_cooldown(timestamp):
            self._state = WakeWordState.COOLDOWN
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
            )

        self._state = WakeWordState.LISTENING

        text_lower = text.lower().strip()

        for variant in self._wake_words:
            if variant in text_lower:
                self._update_detection(timestamp)
                return WakeWordResult(
                    detected=True,
                    confidence=1.0,
                    wake_word=self.config.phrase,
                    timestamp=timestamp,
                )

        return WakeWordResult(
            detected=False,
            confidence=0.0,
            timestamp=timestamp,
        )
