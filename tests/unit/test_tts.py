"""
SAIL Voice Output Pipeline Tests

Comprehensive unit tests for the voice output (TTS) pipeline components.
"""

import numpy as np
from src.voice.tts.base import (
    SpeechMode,
    SpeechPriority,
    SpeechRequest,
    SpeechResult,
    SpeechState,
    TTSConfig,
    TTSEngine,
    VoiceProfile,
    preprocess_text_for_speech,
    split_text_into_sentences,
)
from src.voice.tts.manager import (
    SpeakResult,
    VoiceOutputEvent,
    VoiceOutputState,
)
from src.voice.tts.output import (
    AudioOutputManager,
    OutputState,
    PriorityAudioQueue,
    QueuedAudio,
)
from src.voice.tts.piper import PiperConfig

# ============================================================================
# TTS Config Tests
# ============================================================================


class TestTTSConfig:
    """Tests for TTS configuration."""

    def test_default_config(self):
        """Test default TTS configuration values."""
        config = TTSConfig()

        assert config.engine == TTSEngine.PIPER
        assert config.voice == "en_US-lessac-medium"
        assert config.rate == 1.0
        assert config.sample_rate == 22050

    def test_custom_config(self):
        """Test custom TTS configuration."""
        config = TTSConfig(
            voice="en_US-ryan-medium",
            rate=1.5,
            sample_rate=24000,
        )

        assert config.voice == "en_US-ryan-medium"
        assert config.rate == 1.5
        assert config.sample_rate == 24000

    def test_mode_rate_multipliers(self):
        """Test mode-specific rate multipliers."""
        config = TTSConfig()

        assert config.calm_rate_multiplier == 0.9
        assert config.urgent_rate_multiplier == 1.2
        assert config.crisis_rate_multiplier == 1.0


class TestPiperConfig:
    """Tests for Piper-specific configuration."""

    def test_default_piper_config(self):
        """Test default Piper configuration."""
        config = PiperConfig()

        assert config.engine == TTSEngine.PIPER
        assert config.use_cuda is False
        assert config.length_scale == 1.0
        assert config.noise_scale == 0.667

    def test_custom_piper_config(self):
        """Test custom Piper configuration."""
        config = PiperConfig(
            voice="custom-voice",
            use_cuda=True,
            length_scale=0.8,
        )

        assert config.voice == "custom-voice"
        assert config.use_cuda is True
        assert config.length_scale == 0.8


# ============================================================================
# Speech Types Tests
# ============================================================================


class TestSpeechPriority:
    """Tests for speech priority enumeration."""

    def test_priority_ordering(self):
        """Test priority ordering is correct."""
        assert SpeechPriority.CRITICAL > SpeechPriority.HIGH
        assert SpeechPriority.HIGH > SpeechPriority.NORMAL
        assert SpeechPriority.NORMAL > SpeechPriority.LOW
        assert SpeechPriority.LOW > SpeechPriority.AMBIENT

    def test_priority_values(self):
        """Test priority numeric values."""
        assert SpeechPriority.CRITICAL.value == 100
        assert SpeechPriority.HIGH.value == 75
        assert SpeechPriority.NORMAL.value == 50
        assert SpeechPriority.LOW.value == 25
        assert SpeechPriority.AMBIENT.value == 10


class TestSpeechMode:
    """Tests for speech mode enumeration."""

    def test_all_modes_defined(self):
        """Test all speech modes are defined."""
        expected_modes = ["CALM", "NORMAL", "URGENT", "CRISIS"]
        for mode in expected_modes:
            assert hasattr(SpeechMode, mode)

    def test_mode_values(self):
        """Test speech mode string values."""
        assert SpeechMode.CALM.value == "calm"
        assert SpeechMode.NORMAL.value == "normal"
        assert SpeechMode.URGENT.value == "urgent"
        assert SpeechMode.CRISIS.value == "crisis"


class TestSpeechState:
    """Tests for speech state enumeration."""

    def test_all_states_defined(self):
        """Test all speech states are defined."""
        expected_states = [
            "IDLE", "SYNTHESIZING", "SPEAKING",
            "PAUSED", "INTERRUPTED", "COMPLETE", "ERROR",
        ]
        for state in expected_states:
            assert hasattr(SpeechState, state)


class TestSpeechRequest:
    """Tests for speech request dataclass."""

    def test_default_request(self):
        """Test creating speech request with defaults."""
        request = SpeechRequest(text="Hello world")

        assert request.text == "Hello world"
        assert request.priority == SpeechPriority.NORMAL
        assert request.mode == SpeechMode.NORMAL
        assert request.interruptible is True
        assert request.wait_for_completion is False

    def test_custom_request(self):
        """Test creating custom speech request."""
        request = SpeechRequest(
            text="Emergency alert",
            priority=SpeechPriority.CRITICAL,
            mode=SpeechMode.URGENT,
            interruptible=False,
            rate=1.5,
        )

        assert request.text == "Emergency alert"
        assert request.priority == SpeechPriority.CRITICAL
        assert request.mode == SpeechMode.URGENT
        assert request.interruptible is False
        assert request.rate == 1.5


class TestSpeechResult:
    """Tests for speech result dataclass."""

    def test_success_result(self):
        """Test successful speech result."""
        audio = np.zeros(1000, dtype=np.int16)
        result = SpeechResult(
            audio_data=audio,
            sample_rate=22050,
            duration_seconds=1.0,
        )

        assert result.is_success is True
        assert result.error is None
        assert len(result.audio_data) == 1000

    def test_error_result(self):
        """Test error speech result."""
        result = SpeechResult(error="Synthesis failed")

        assert result.is_success is False
        assert result.error == "Synthesis failed"
        assert result.audio_data is None

    def test_interrupted_result(self):
        """Test interrupted speech result."""
        result = SpeechResult(
            audio_data=np.zeros(100, dtype=np.int16),
            sample_rate=22050,
            was_interrupted=True,
        )

        assert result.is_success is True
        assert result.was_interrupted is True


class TestVoiceProfile:
    """Tests for voice profile dataclass."""

    def test_default_profile(self):
        """Test creating voice profile with defaults."""
        profile = VoiceProfile(
            name="Test Voice",
            model_id="test-voice",
        )

        assert profile.name == "Test Voice"
        assert profile.model_id == "test-voice"
        assert profile.language == "en"
        assert profile.base_rate == 1.0

    def test_custom_profile(self):
        """Test creating custom voice profile."""
        profile = VoiceProfile(
            name="Custom Voice",
            model_id="custom-voice",
            language="es",
            gender="female",
            base_rate=0.9,
            calm_rate=0.8,
        )

        assert profile.language == "es"
        assert profile.gender == "female"
        assert profile.base_rate == 0.9
        assert profile.calm_rate == 0.8


# ============================================================================
# Text Preprocessing Tests
# ============================================================================


class TestTextPreprocessing:
    """Tests for text preprocessing functions."""

    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        text = "Dr. Smith lives on Oak St."
        result = preprocess_text_for_speech(text)

        assert "Doctor" in result
        assert "Street" in result

    def test_percent_expansion(self):
        """Test percentage expansion."""
        text = "The score was 85%"
        result = preprocess_text_for_speech(text)

        assert "85 percent" in result

    def test_dollar_expansion(self):
        """Test dollar sign expansion."""
        text = "The price is $50"
        result = preprocess_text_for_speech(text)

        assert "50 dollars" in result

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "Hello    world   test"
        result = preprocess_text_for_speech(text)

        assert "  " not in result
        assert result == "Hello world test"


class TestSentenceSplitting:
    """Tests for sentence splitting function."""

    def test_basic_splitting(self):
        """Test basic sentence splitting."""
        text = "Hello world. How are you? I am fine!"
        sentences = split_text_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine!"

    def test_single_sentence(self):
        """Test single sentence."""
        text = "Just one sentence here."
        sentences = split_text_into_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here."

    def test_empty_text(self):
        """Test empty text."""
        sentences = split_text_into_sentences("")
        assert len(sentences) == 0

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        text = "First.   Second.    Third."
        sentences = split_text_into_sentences(text)

        assert len(sentences) == 3
        for s in sentences:
            assert s == s.strip()


# ============================================================================
# Priority Queue Tests
# ============================================================================


class TestPriorityAudioQueue:
    """Tests for priority audio queue."""

    def test_queue_creation(self):
        """Test creating priority queue."""
        queue = PriorityAudioQueue()
        assert queue.is_empty is True
        assert queue.size == 0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        queue = PriorityAudioQueue()

        request = SpeechRequest(text="Test")
        item = QueuedAudio(
            audio_data=np.zeros(100, dtype=np.int16),
            sample_rate=22050,
            request=request,
        )

        queue.put(item)
        assert queue.is_empty is False
        assert queue.size == 1

        retrieved = queue.get(timeout=0.1)
        assert retrieved is not None
        assert retrieved.request.text == "Test"
        assert queue.is_empty is True

    def test_priority_ordering(self):
        """Test that higher priority items come first."""
        queue = PriorityAudioQueue()

        # Add low priority first
        low_request = SpeechRequest(text="Low", priority=SpeechPriority.LOW)
        low_item = QueuedAudio(
            audio_data=np.zeros(100, dtype=np.int16),
            sample_rate=22050,
            request=low_request,
        )
        queue.put(low_item)

        # Add high priority second
        high_request = SpeechRequest(text="High", priority=SpeechPriority.HIGH)
        high_item = QueuedAudio(
            audio_data=np.zeros(100, dtype=np.int16),
            sample_rate=22050,
            request=high_request,
        )
        queue.put(high_item)

        # High priority should come first
        first = queue.get(timeout=0.1)
        assert first is not None
        assert first.request.text == "High"

        second = queue.get(timeout=0.1)
        assert second is not None
        assert second.request.text == "Low"

    def test_clear(self):
        """Test clearing the queue."""
        queue = PriorityAudioQueue()

        for i in range(5):
            request = SpeechRequest(text=f"Item {i}")
            item = QueuedAudio(
                audio_data=np.zeros(100, dtype=np.int16),
                sample_rate=22050,
                request=request,
            )
            queue.put(item)

        assert queue.size == 5

        cleared = queue.clear()
        assert len(cleared) == 5
        assert queue.is_empty is True

    def test_clear_below_priority(self):
        """Test clearing items below a certain priority."""
        queue = PriorityAudioQueue()

        # Add items with different priorities
        for priority in [SpeechPriority.LOW, SpeechPriority.NORMAL, SpeechPriority.HIGH]:
            request = SpeechRequest(text=f"{priority.name}", priority=priority)
            item = QueuedAudio(
                audio_data=np.zeros(100, dtype=np.int16),
                sample_rate=22050,
                request=request,
            )
            queue.put(item)

        # Clear below HIGH priority
        cleared = queue.clear_below_priority(SpeechPriority.HIGH)

        assert len(cleared) == 2  # LOW and NORMAL cleared
        assert queue.size == 1     # Only HIGH remains


class TestQueuedAudio:
    """Tests for queued audio dataclass."""

    def test_creation(self):
        """Test creating queued audio."""
        request = SpeechRequest(text="Test")
        audio = np.zeros(1000, dtype=np.int16)

        item = QueuedAudio(
            audio_data=audio,
            sample_rate=22050,
            request=request,
        )

        assert len(item.audio_data) == 1000
        assert item.position == 0
        assert item.is_complete is False

    def test_remaining_samples(self):
        """Test remaining samples calculation."""
        request = SpeechRequest(text="Test")
        audio = np.zeros(1000, dtype=np.int16)

        item = QueuedAudio(
            audio_data=audio,
            sample_rate=22050,
            request=request,
        )

        assert item.remaining_samples == 1000

        item.position = 500
        assert item.remaining_samples == 500

    def test_is_complete(self):
        """Test completion detection."""
        request = SpeechRequest(text="Test")
        audio = np.zeros(1000, dtype=np.int16)

        item = QueuedAudio(
            audio_data=audio,
            sample_rate=22050,
            request=request,
        )

        assert item.is_complete is False

        item.position = 1000
        assert item.is_complete is True


# ============================================================================
# Output State Tests
# ============================================================================


class TestOutputState:
    """Tests for output state enumeration."""

    def test_all_states_defined(self):
        """Test all output states are defined."""
        expected_states = ["IDLE", "PLAYING", "PAUSED", "STOPPING", "STOPPED"]
        for state in expected_states:
            assert hasattr(OutputState, state)


class TestVoiceOutputState:
    """Tests for voice output state enumeration."""

    def test_all_states_defined(self):
        """Test all voice output states are defined."""
        expected_states = [
            "IDLE", "SYNTHESIZING", "SPEAKING",
            "PAUSED", "INTERRUPTED", "STOPPED", "ERROR",
        ]
        for state in expected_states:
            assert hasattr(VoiceOutputState, state)


# ============================================================================
# Voice Output Event Tests
# ============================================================================


class TestVoiceOutputEvent:
    """Tests for voice output event dataclass."""

    def test_event_creation(self):
        """Test creating voice output event."""
        event = VoiceOutputEvent(
            event_type="speaking",
            data={"text": "Hello"},
            timestamp=123.456,
        )

        assert event.event_type == "speaking"
        assert event.data["text"] == "Hello"
        assert event.timestamp == 123.456

    def test_default_data(self):
        """Test event with default data."""
        event = VoiceOutputEvent(event_type="stopped")

        assert event.event_type == "stopped"
        assert event.data == {}


# ============================================================================
# Speak Result Tests
# ============================================================================


class TestSpeakResult:
    """Tests for speak result dataclass."""

    def test_success_result(self):
        """Test successful speak result."""
        result = SpeakResult(
            success=True,
            queue_id="abc123",
            duration_seconds=2.5,
        )

        assert result.success is True
        assert result.queue_id == "abc123"
        assert result.duration_seconds == 2.5
        assert result.was_interrupted is False
        assert result.error is None

    def test_error_result(self):
        """Test error speak result."""
        result = SpeakResult(
            success=False,
            error="TTS not available",
        )

        assert result.success is False
        assert result.error == "TTS not available"

    def test_interrupted_result(self):
        """Test interrupted speak result."""
        result = SpeakResult(
            success=True,
            was_interrupted=True,
        )

        assert result.success is True
        assert result.was_interrupted is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestTTSEngine:
    """Tests for TTS engine enumeration."""

    def test_all_engines_defined(self):
        """Test all TTS engines are defined."""
        expected_engines = ["PIPER", "COQUI"]
        for engine in expected_engines:
            assert hasattr(TTSEngine, engine)

    def test_engine_values(self):
        """Test TTS engine string values."""
        assert TTSEngine.PIPER.value == "piper"
        assert TTSEngine.COQUI.value == "coqui"


class TestAudioOutputManagerCreation:
    """Tests for audio output manager creation."""

    def test_manager_creation(self):
        """Test creating audio output manager."""
        manager = AudioOutputManager(sample_rate=22050)

        assert manager.sample_rate == 22050
        assert manager.state == OutputState.IDLE
        assert manager.is_playing is False

    def test_manager_with_custom_settings(self):
        """Test manager with custom settings."""
        manager = AudioOutputManager(
            sample_rate=24000,
            output_device=1,
            chunk_size=2048,
        )

        assert manager.sample_rate == 24000
        assert manager.output_device == 1
        assert manager.chunk_size == 2048


class TestRespectProtocol:
    """Tests verifying respect protocol implementation."""

    def test_interruptible_default(self):
        """Test that speech is interruptible by default."""
        request = SpeechRequest(text="Test")
        assert request.interruptible is True

    def test_critical_priority_non_interruptible(self):
        """Test that critical alerts can be made non-interruptible."""
        request = SpeechRequest(
            text="Emergency",
            priority=SpeechPriority.CRITICAL,
            interruptible=False,
        )

        assert request.priority == SpeechPriority.CRITICAL
        assert request.interruptible is False

    def test_speech_modes_for_intervention(self):
        """Test speech modes suitable for intervention scenarios."""
        # Calm mode for de-escalation
        calm_request = SpeechRequest(
            text="Let's take a breath",
            mode=SpeechMode.CALM,
        )
        assert calm_request.mode == SpeechMode.CALM

        # Crisis mode for emergency guidance
        crisis_request = SpeechRequest(
            text="Stay calm, help is coming",
            mode=SpeechMode.CRISIS,
            priority=SpeechPriority.HIGH,
        )
        assert crisis_request.mode == SpeechMode.CRISIS
        assert crisis_request.priority == SpeechPriority.HIGH
