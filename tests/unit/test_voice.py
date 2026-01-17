"""
SAIL Voice Input Pipeline Tests

Comprehensive unit tests for the voice input pipeline components.
"""

import numpy as np
import pytest
from src.voice.audio.capture import AudioChunk, AudioDeviceInfo
from src.voice.audio.vad import (
    AudioAccumulator,
    SpeechSegment,
    VADConfig,
    VADProcessor,
    VADResult,
    VADState,
)
from src.voice.commands import (
    CommandDetector,
    CommandPriority,
    CommandResult,
    CommandType,
    QuickStopDetector,
)
from src.voice.stt.base import (
    STTConfig,
    STTEngine,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.voice.wake_word.base import (
    WakeWordConfig,
    WakeWordResult,
    WakeWordState,
)
from src.voice.wake_word.openwakeword import SimpleWakeWordDetector

# ============================================================================
# Audio Capture Tests
# ============================================================================


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_audio_chunk_creation(self):
        """Test creating an audio chunk."""
        data = np.zeros(480, dtype=np.int16)  # 30ms at 16kHz
        chunk = AudioChunk(
            data=data,
            sample_rate=16000,
            channels=1,
            timestamp=0.0,
        )

        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.duration_ms == 30.0

    def test_audio_chunk_to_int16(self):
        """Test converting audio chunk to int16."""
        # Float32 data
        data = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000, channels=1, timestamp=0.0)

        int16_data = chunk.to_int16()
        assert int16_data.dtype == np.int16
        assert int16_data[0] == 16383  # 0.5 * 32767
        assert int16_data[1] == -16383

    def test_audio_chunk_to_float32(self):
        """Test converting audio chunk to float32."""
        # Int16 data
        data = np.array([16384, -16384, 32767, -32768], dtype=np.int16)
        chunk = AudioChunk(data=data, sample_rate=16000, channels=1, timestamp=0.0)

        float_data = chunk.to_float32()
        assert float_data.dtype == np.float32
        assert abs(float_data[0] - 0.5) < 0.01
        assert abs(float_data[1] - (-0.5)) < 0.01

    def test_audio_chunk_to_bytes(self):
        """Test converting audio chunk to bytes."""
        data = np.array([1000, -1000], dtype=np.int16)
        chunk = AudioChunk(data=data, sample_rate=16000, channels=1, timestamp=0.0)

        audio_bytes = chunk.to_bytes()
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) == 4  # 2 samples * 2 bytes

    def test_audio_chunk_duration_calculation(self):
        """Test duration calculation for various sample counts."""
        # 160 samples at 16kHz = 10ms
        data = np.zeros(160, dtype=np.int16)
        chunk = AudioChunk(data=data, sample_rate=16000, channels=1, timestamp=0.0)
        assert chunk.duration_ms == 10.0

        # 8000 samples at 16kHz = 500ms
        data = np.zeros(8000, dtype=np.int16)
        chunk = AudioChunk(data=data, sample_rate=16000, channels=1, timestamp=0.0)
        assert chunk.duration_ms == 500.0


class TestAudioDeviceInfo:
    """Tests for AudioDeviceInfo dataclass."""

    def test_device_info_str(self):
        """Test string representation of device info."""
        device = AudioDeviceInfo(
            index=0,
            name="Test Microphone",
            channels=2,
            default_sample_rate=44100.0,
            is_default=False,
        )

        str_repr = str(device)
        assert "[0]" in str_repr
        assert "Test Microphone" in str_repr
        assert "2ch" in str_repr
        assert "44100" in str_repr

    def test_device_info_default_marker(self):
        """Test default device marker in string."""
        device = AudioDeviceInfo(
            index=0,
            name="Default Mic",
            channels=1,
            default_sample_rate=16000.0,
            is_default=True,
        )

        str_repr = str(device)
        assert "(default)" in str_repr


# ============================================================================
# VAD Tests
# ============================================================================


class TestVADConfig:
    """Tests for VAD configuration."""

    def test_default_config(self):
        """Test default VAD configuration values."""
        config = VADConfig()

        assert config.aggressiveness == 2
        assert config.min_speech_duration_ms == 250.0
        assert config.silence_threshold_ms == 500.0
        assert config.frame_duration_ms == 30
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom VAD configuration."""
        config = VADConfig(
            aggressiveness=3,
            min_speech_duration_ms=300.0,
            silence_threshold_ms=800.0,
        )

        assert config.aggressiveness == 3
        assert config.min_speech_duration_ms == 300.0
        assert config.silence_threshold_ms == 800.0


class TestVADResult:
    """Tests for VAD result dataclass."""

    def test_vad_result_creation(self):
        """Test creating a VAD result."""
        result = VADResult(
            is_speech=True,
            state=VADState.SPEECH,
            confidence=0.9,
            speech_duration_ms=500.0,
            silence_duration_ms=0.0,
        )

        assert result.is_speech is True
        assert result.state == VADState.SPEECH
        assert result.confidence == 0.9


class TestSpeechSegment:
    """Tests for speech segment dataclass."""

    def test_speech_segment_duration(self):
        """Test speech segment duration calculation."""
        segment = SpeechSegment(start_time=1.0, end_time=2.5)
        assert segment.duration_ms == 1500.0

    def test_speech_segment_no_end_time(self):
        """Test segment with no end time."""
        segment = SpeechSegment(start_time=1.0)
        assert segment.duration_ms == 0.0
        assert segment.is_complete is False

    def test_speech_segment_get_audio(self):
        """Test combining audio chunks."""
        segment = SpeechSegment(start_time=0.0)
        segment.audio_chunks = [
            np.array([1, 2, 3], dtype=np.int16),
            np.array([4, 5, 6], dtype=np.int16),
        ]

        audio = segment.get_audio()
        assert len(audio) == 6
        assert list(audio) == [1, 2, 3, 4, 5, 6]

    def test_speech_segment_empty_audio(self):
        """Test empty segment audio."""
        segment = SpeechSegment(start_time=0.0)
        audio = segment.get_audio()
        assert len(audio) == 0


class TestVADProcessor:
    """Tests for VAD processor."""

    def test_vad_processor_creation(self):
        """Test creating VAD processor."""
        pytest.importorskip("webrtcvad")
        processor = VADProcessor()
        assert processor.state == VADState.SILENCE
        assert processor.is_speaking is False

    def test_vad_processor_reset(self):
        """Test resetting VAD processor."""
        pytest.importorskip("webrtcvad")
        processor = VADProcessor()
        processor.reset()
        assert processor.state == VADState.SILENCE

    def test_vad_processor_states(self):
        """Test VAD state enumeration."""
        assert VADState.SILENCE.value == "silence"
        assert VADState.SPEECH_START.value == "speech_start"
        assert VADState.SPEECH.value == "speech"
        assert VADState.SPEECH_END.value == "speech_end"


class TestAudioAccumulator:
    """Tests for audio accumulator."""

    def test_accumulator_creation(self):
        """Test creating audio accumulator."""
        pytest.importorskip("webrtcvad")
        accumulator = AudioAccumulator()
        assert accumulator.is_accumulating is False
        assert accumulator.accumulated_duration_ms == 0.0

    def test_accumulator_reset(self):
        """Test resetting accumulator."""
        pytest.importorskip("webrtcvad")
        accumulator = AudioAccumulator()
        accumulator.reset()
        assert accumulator.is_accumulating is False
        assert accumulator.accumulated_duration_ms == 0.0


# ============================================================================
# Wake Word Tests
# ============================================================================


class TestWakeWordConfig:
    """Tests for wake word configuration."""

    def test_default_config(self):
        """Test default wake word configuration."""
        config = WakeWordConfig()

        assert config.phrase == "hey sail"
        assert config.sensitivity == 0.5
        assert config.cooldown_seconds == 2.0
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom wake word configuration."""
        config = WakeWordConfig(
            phrase="hello computer",
            sensitivity=0.7,
            cooldown_seconds=3.0,
        )

        assert config.phrase == "hello computer"
        assert config.sensitivity == 0.7
        assert config.cooldown_seconds == 3.0


class TestWakeWordResult:
    """Tests for wake word result dataclass."""

    def test_wake_word_result_detected(self):
        """Test wake word detected result."""
        result = WakeWordResult(
            detected=True,
            confidence=0.95,
            wake_word="hey sail",
            timestamp=1.0,
        )

        assert result.detected is True
        assert bool(result) is True
        assert result.confidence == 0.95
        assert result.wake_word == "hey sail"

    def test_wake_word_result_not_detected(self):
        """Test wake word not detected result."""
        result = WakeWordResult(detected=False, confidence=0.1)

        assert result.detected is False
        assert bool(result) is False


class TestSimpleWakeWordDetector:
    """Tests for simple wake word detector."""

    def test_detector_creation(self):
        """Test creating simple wake word detector."""
        config = WakeWordConfig(phrase="hey sail")
        detector = SimpleWakeWordDetector(config)

        assert detector.state == WakeWordState.LISTENING
        assert detector.is_listening is True

    def test_detect_in_text_positive(self):
        """Test detecting wake word in text."""
        config = WakeWordConfig(phrase="hey sail")
        detector = SimpleWakeWordDetector(config)

        result = detector.detect_in_text("Hey Sail, what's the weather?")
        assert result.detected is True
        assert result.confidence == 1.0

    def test_detect_in_text_negative(self):
        """Test not detecting wake word."""
        config = WakeWordConfig(phrase="hey sail")
        detector = SimpleWakeWordDetector(config)

        result = detector.detect_in_text("Hello there, how are you?")
        assert result.detected is False

    def test_detect_in_text_case_insensitive(self):
        """Test case insensitive detection."""
        config = WakeWordConfig(phrase="hey sail")
        detector = SimpleWakeWordDetector(config)

        result = detector.detect_in_text("HEY SAIL!")
        assert result.detected is True

    def test_detector_reset(self):
        """Test resetting detector."""
        config = WakeWordConfig(phrase="hey sail")
        detector = SimpleWakeWordDetector(config)

        detector.detect_in_text("hey sail")  # Should detect
        detector.reset()

        assert detector.state == WakeWordState.LISTENING


# ============================================================================
# Command Detection Tests
# ============================================================================


class TestCommandResult:
    """Tests for command result dataclass."""

    def test_command_result_detected(self):
        """Test detected command result."""
        result = CommandResult(
            detected=True,
            command_type=CommandType.STOP,
            priority=CommandPriority.CRITICAL,
            confidence=1.0,
            matched_phrase="stop",
            original_text="please stop",
        )

        assert result.detected is True
        assert bool(result) is True
        assert result.is_stop_command is True
        assert result.command_type == CommandType.STOP

    def test_command_result_not_detected(self):
        """Test not detected command result."""
        result = CommandResult(
            detected=False,
            command_type=CommandType.UNKNOWN,
            priority=CommandPriority.LOW,
            confidence=0.0,
        )

        assert result.detected is False
        assert bool(result) is False
        assert result.is_stop_command is False


class TestCommandDetector:
    """Tests for command detector."""

    def test_detector_creation(self):
        """Test creating command detector."""
        detector = CommandDetector()
        assert detector is not None

    def test_detect_stop_command(self):
        """Test detecting stop commands."""
        detector = CommandDetector()

        # Test various stop phrases
        stop_phrases = ["stop", "shut up", "be quiet", "silence", "enough"]

        for phrase in stop_phrases:
            result = detector.detect(phrase)
            assert result.detected is True, f"Should detect: {phrase}"
            assert result.command_type == CommandType.STOP
            assert result.priority == CommandPriority.CRITICAL

    def test_detect_stop_in_sentence(self):
        """Test detecting stop in a longer sentence."""
        detector = CommandDetector()

        result = detector.detect("Please just stop talking now")
        assert result.detected is True
        assert result.command_type == CommandType.STOP

    def test_detect_pause_command(self):
        """Test detecting pause commands."""
        detector = CommandDetector()

        result = detector.detect("pause for a moment")
        assert result.detected is True
        assert result.command_type == CommandType.PAUSE
        assert result.priority == CommandPriority.HIGH

    def test_detect_cancel_command(self):
        """Test detecting cancel commands."""
        detector = CommandDetector()

        result = detector.detect("cancel that")
        assert result.detected is True
        assert result.command_type == CommandType.CANCEL

    def test_detect_yes_command(self):
        """Test detecting yes/confirmation commands."""
        detector = CommandDetector()

        for phrase in ["yes", "yeah", "sure", "okay"]:
            result = detector.detect(phrase)
            assert result.detected is True, f"Should detect: {phrase}"
            assert result.command_type == CommandType.YES

    def test_detect_no_command(self):
        """Test detecting no/denial commands."""
        detector = CommandDetector()

        for phrase in ["no", "nope", "negative"]:
            result = detector.detect(phrase)
            assert result.detected is True, f"Should detect: {phrase}"
            assert result.command_type == CommandType.NO

    def test_detect_help_command(self):
        """Test detecting help commands."""
        detector = CommandDetector()

        result = detector.detect("help me")
        assert result.detected is True
        assert result.command_type == CommandType.HELP

    def test_detect_repeat_command(self):
        """Test detecting repeat commands."""
        detector = CommandDetector()

        result = detector.detect("say again")
        assert result.detected is True
        assert result.command_type == CommandType.REPEAT

    def test_no_command_detected(self):
        """Test when no command is detected."""
        detector = CommandDetector()

        result = detector.detect("What's the weather like today?")
        assert result.detected is False
        assert result.command_type == CommandType.UNKNOWN

    def test_empty_text(self):
        """Test with empty text."""
        detector = CommandDetector()

        result = detector.detect("")
        assert result.detected is False

        result = detector.detect("   ")
        assert result.detected is False

    def test_detect_stop_command_quick(self):
        """Test quick stop command detection."""
        detector = CommandDetector()

        assert detector.detect_stop_command("stop") is True
        assert detector.detect_stop_command("shut up") is True
        assert detector.detect_stop_command("hello world") is False
        assert detector.detect_stop_command("") is False

    def test_add_stop_phrase(self):
        """Test adding custom stop phrase."""
        detector = CommandDetector()

        # Initially shouldn't detect
        result = detector.detect("halt")
        assert result.command_type != CommandType.STOP

        # Add custom phrase
        detector.add_stop_phrase("halt")

        # Now should detect
        result = detector.detect("halt")
        assert result.detected is True
        assert result.command_type == CommandType.STOP

    def test_command_priority_order(self):
        """Test that higher priority commands are detected first."""
        detector = CommandDetector()

        # "stop" is CRITICAL, should be detected over NORMAL commands
        result = detector.detect("stop yes")
        assert result.command_type == CommandType.STOP

    def test_callback_registration(self):
        """Test registering command callbacks."""
        detector = CommandDetector()
        callback_called = False
        callback_result = None

        def on_stop(result: CommandResult):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result

        detector.on_command(CommandType.STOP, on_stop)
        detector.detect("stop")

        assert callback_called is True
        assert callback_result is not None
        assert callback_result.command_type == CommandType.STOP


class TestQuickStopDetector:
    """Tests for quick stop detector."""

    def test_detector_creation(self):
        """Test creating quick stop detector."""
        detector = QuickStopDetector()
        assert detector.is_enabled is True

    def test_detect_stop_words(self):
        """Test detecting stop words."""
        detector = QuickStopDetector()

        assert detector.detect("stop") is True
        assert detector.detect("Stop!") is True
        assert detector.detect("quiet") is True
        assert detector.detect("SILENCE") is True

    def test_detect_stop_phrases(self):
        """Test detecting two-word stop phrases."""
        detector = QuickStopDetector()

        assert detector.detect("shut up") is True
        assert detector.detect("be quiet") is True
        assert detector.detect("not now") is True
        assert detector.detect("go away") is True

    def test_detect_in_sentence(self):
        """Test detecting stop in longer text."""
        detector = QuickStopDetector()

        assert detector.detect("please stop talking") is True
        assert detector.detect("I need you to shut up") is True

    def test_no_stop_detected(self):
        """Test when no stop command present."""
        detector = QuickStopDetector()

        assert detector.detect("hello world") is False
        assert detector.detect("what time is it") is False
        assert detector.detect("") is False

    def test_enable_disable(self):
        """Test enabling and disabling detector."""
        detector = QuickStopDetector()

        detector.disable()
        assert detector.is_enabled is False
        assert detector.detect("stop") is False

        detector.enable()
        assert detector.is_enabled is True
        assert detector.detect("stop") is True


# ============================================================================
# STT Tests
# ============================================================================


class TestSTTConfig:
    """Tests for STT configuration."""

    def test_default_config(self):
        """Test default STT configuration."""
        config = STTConfig()

        assert config.engine == STTEngine.WHISPER
        assert config.model == "base"
        assert config.language is None
        assert config.device == "cpu"
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom STT configuration."""
        config = STTConfig(
            engine=STTEngine.FASTER_WHISPER,
            model="small",
            language="en",
            device="cuda",
        )

        assert config.engine == STTEngine.FASTER_WHISPER
        assert config.model == "small"
        assert config.language == "en"
        assert config.device == "cuda"


class TestTranscriptionResult:
    """Tests for transcription result dataclass."""

    def test_result_creation(self):
        """Test creating transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
        )

        assert result.text == "Hello world"
        assert result.is_empty is False
        assert result.language == "en"
        assert result.duration == 2.5

    def test_empty_result(self):
        """Test empty transcription result."""
        result = TranscriptionResult(text="", duration=0.0)
        assert result.is_empty is True

        result = TranscriptionResult(text="   ", duration=0.0)
        assert result.is_empty is True


class TestTranscriptionSegment:
    """Tests for transcription segment dataclass."""

    def test_segment_creation(self):
        """Test creating transcription segment."""
        segment = TranscriptionSegment(
            text="Hello",
            start_time=0.0,
            end_time=1.0,
            confidence=0.95,
        )

        assert segment.text == "Hello"
        assert segment.duration == 1.0
        assert segment.confidence == 0.95

    def test_segment_with_words(self):
        """Test segment with word-level timestamps."""
        segment = TranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            words=[
                {"word": "Hello", "start": 0.0, "end": 0.8},
                {"word": "world", "start": 1.0, "end": 2.0},
            ],
        )

        assert len(segment.words) == 2
        assert segment.words[0]["word"] == "Hello"


# ============================================================================
# Integration Tests
# ============================================================================


class TestCommandTypes:
    """Tests for command type enumeration."""

    def test_all_command_types(self):
        """Test all command types are defined."""
        expected_types = [
            "STOP", "PAUSE", "RESUME", "REPEAT",
            "HELP", "CANCEL", "YES", "NO", "UNKNOWN",
        ]

        for cmd_type in expected_types:
            assert hasattr(CommandType, cmd_type)

    def test_command_type_values(self):
        """Test command type string values."""
        assert CommandType.STOP.value == "stop"
        assert CommandType.PAUSE.value == "pause"
        assert CommandType.YES.value == "yes"
        assert CommandType.NO.value == "no"


class TestCommandPriorities:
    """Tests for command priority enumeration."""

    def test_priority_ordering(self):
        """Test priority ordering is correct."""
        assert CommandPriority.CRITICAL > CommandPriority.HIGH
        assert CommandPriority.HIGH > CommandPriority.NORMAL
        assert CommandPriority.NORMAL > CommandPriority.LOW

    def test_priority_values(self):
        """Test priority numeric values."""
        assert CommandPriority.CRITICAL.value == 100
        assert CommandPriority.HIGH.value == 75
        assert CommandPriority.NORMAL.value == 50
        assert CommandPriority.LOW.value == 25


class TestVoiceInputStates:
    """Tests for voice input states."""

    def test_all_states_defined(self):
        """Test all voice input states are defined."""
        from src.voice.manager import VoiceInputState

        expected_states = [
            "IDLE", "LISTENING_FOR_WAKE_WORD",
            "LISTENING_FOR_SPEECH", "PROCESSING",
            "STOPPED", "ERROR",
        ]

        for state in expected_states:
            assert hasattr(VoiceInputState, state)


class TestVoiceInputResult:
    """Tests for voice input result."""

    def test_result_with_text(self):
        """Test result with text."""
        from src.voice.manager import VoiceInputResult

        result = VoiceInputResult(
            text="Hello world",
            was_stopped=False,
            audio_duration_ms=1500.0,
        )

        assert result.has_text is True
        assert result.is_command is False
        assert result.was_stopped is False

    def test_result_with_command(self):
        """Test result with command."""
        from src.voice.manager import VoiceInputResult

        command = CommandResult(
            detected=True,
            command_type=CommandType.STOP,
            priority=CommandPriority.CRITICAL,
            confidence=1.0,
        )

        result = VoiceInputResult(
            text="stop",
            command=command,
            was_stopped=True,
        )

        assert result.is_command is True
        assert result.was_stopped is True

    def test_empty_result(self):
        """Test empty result."""
        from src.voice.manager import VoiceInputResult

        result = VoiceInputResult(text="", was_stopped=False)
        assert result.has_text is False
