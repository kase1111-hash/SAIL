"""
SAIL Voice Command Detection

Provides detection of voice commands including stop commands,
which are critical for respecting user boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class CommandType(str, Enum):
    """Types of detected voice commands."""

    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    REPEAT = "repeat"
    HELP = "help"
    CANCEL = "cancel"
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class CommandPriority(int, Enum):
    """Priority levels for commands."""

    CRITICAL = 100  # Stop commands - immediate action
    HIGH = 75       # Cancel, pause - urgent
    NORMAL = 50     # Yes/no, repeat - standard
    LOW = 25        # Help, resume - can wait


@dataclass
class CommandResult:
    """Result of command detection."""

    detected: bool
    command_type: CommandType
    priority: CommandPriority
    confidence: float
    matched_phrase: str | None = None
    original_text: str = ""

    def __bool__(self) -> bool:
        """Allow using result directly in boolean context."""
        return self.detected

    @property
    def is_stop_command(self) -> bool:
        """Check if this is a stop command."""
        return self.command_type == CommandType.STOP


@dataclass
class CommandPattern:
    """A pattern for detecting commands."""

    command_type: CommandType
    priority: CommandPriority
    patterns: list[str] = field(default_factory=list)
    regex_patterns: list[re.Pattern] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compile regex patterns."""
        if not self.regex_patterns and self.patterns:
            self.regex_patterns = [
                re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE)
                for p in self.patterns
            ]


class CommandDetector:
    """
    Detects voice commands in transcribed text.

    Implements SAIL's "respect" principle by immediately detecting
    and honoring stop commands.
    """

    # Default stop command phrases
    DEFAULT_STOP_PHRASES = [
        "stop",
        "shut up",
        "be quiet",
        "quiet",
        "silence",
        "enough",
        "that's enough",
        "stop talking",
        "hush",
        "shush",
        "not now",
        "go away",
        "leave me alone",
        "never mind",
        "nevermind",
        "forget it",
        "stop it",
        "cut it out",
    ]

    # Default pause phrases
    DEFAULT_PAUSE_PHRASES = [
        "pause",
        "hold on",
        "wait",
        "one moment",
        "just a moment",
        "one second",
        "just a second",
        "hang on",
    ]

    # Default resume phrases
    DEFAULT_RESUME_PHRASES = [
        "continue",
        "go on",
        "resume",
        "carry on",
        "keep going",
        "go ahead",
    ]

    # Default cancel phrases
    DEFAULT_CANCEL_PHRASES = [
        "cancel",
        "abort",
        "undo",
        "scratch that",
        "disregard",
    ]

    # Default confirmation phrases
    DEFAULT_YES_PHRASES = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "okay",
        "ok",
        "affirmative",
        "correct",
        "right",
        "that's right",
        "absolutely",
        "definitely",
        "of course",
        "please",
        "go for it",
    ]

    DEFAULT_NO_PHRASES = [
        "no",
        "nope",
        "nah",
        "negative",
        "wrong",
        "that's wrong",
        "incorrect",
        "not that",
        "don't",
        "do not",
    ]

    # Default help phrases
    DEFAULT_HELP_PHRASES = [
        "help",
        "help me",
        "what can you do",
        "what do you do",
        "how does this work",
        "instructions",
    ]

    # Default repeat phrases
    DEFAULT_REPEAT_PHRASES = [
        "repeat",
        "say again",
        "what did you say",
        "say that again",
        "repeat that",
        "come again",
        "pardon",
        "excuse me",
        "i didn't catch that",
        "one more time",
    ]

    def __init__(
        self,
        additional_stop_phrases: list[str] | None = None,
        custom_commands: list[CommandPattern] | None = None,
    ) -> None:
        """
        Initialize command detector.

        Args:
            additional_stop_phrases: Extra phrases to recognize as stop commands
            custom_commands: Custom command patterns to add
        """
        self._commands: list[CommandPattern] = []
        self._on_command_callbacks: dict[CommandType, list[Callable[[CommandResult], None]]] = {}

        # Initialize default commands
        self._init_default_commands(additional_stop_phrases)

        # Add custom commands
        if custom_commands:
            for cmd in custom_commands:
                self._commands.append(cmd)

        # Sort by priority (highest first)
        self._commands.sort(key=lambda c: c.priority.value, reverse=True)

    def _init_default_commands(self, additional_stop_phrases: list[str] | None) -> None:
        """Initialize default command patterns."""
        # Stop commands (highest priority)
        stop_phrases = self.DEFAULT_STOP_PHRASES.copy()
        if additional_stop_phrases:
            stop_phrases.extend(additional_stop_phrases)

        self._commands.append(CommandPattern(
            command_type=CommandType.STOP,
            priority=CommandPriority.CRITICAL,
            patterns=stop_phrases,
        ))

        # Cancel commands
        self._commands.append(CommandPattern(
            command_type=CommandType.CANCEL,
            priority=CommandPriority.HIGH,
            patterns=self.DEFAULT_CANCEL_PHRASES,
        ))

        # Pause commands
        self._commands.append(CommandPattern(
            command_type=CommandType.PAUSE,
            priority=CommandPriority.HIGH,
            patterns=self.DEFAULT_PAUSE_PHRASES,
        ))

        # Resume commands
        self._commands.append(CommandPattern(
            command_type=CommandType.RESUME,
            priority=CommandPriority.NORMAL,
            patterns=self.DEFAULT_RESUME_PHRASES,
        ))

        # Yes/No commands
        self._commands.append(CommandPattern(
            command_type=CommandType.YES,
            priority=CommandPriority.NORMAL,
            patterns=self.DEFAULT_YES_PHRASES,
        ))

        self._commands.append(CommandPattern(
            command_type=CommandType.NO,
            priority=CommandPriority.NORMAL,
            patterns=self.DEFAULT_NO_PHRASES,
        ))

        # Repeat commands
        self._commands.append(CommandPattern(
            command_type=CommandType.REPEAT,
            priority=CommandPriority.NORMAL,
            patterns=self.DEFAULT_REPEAT_PHRASES,
        ))

        # Help commands
        self._commands.append(CommandPattern(
            command_type=CommandType.HELP,
            priority=CommandPriority.LOW,
            patterns=self.DEFAULT_HELP_PHRASES,
        ))

    def detect(self, text: str) -> CommandResult:
        """
        Detect commands in text.

        Returns the highest priority command found.

        Args:
            text: Transcribed text to analyze

        Returns:
            CommandResult with detection results
        """
        if not text or not text.strip():
            return CommandResult(
                detected=False,
                command_type=CommandType.UNKNOWN,
                priority=CommandPriority.LOW,
                confidence=0.0,
                original_text=text,
            )

        text_lower = text.lower().strip()

        # Check commands in priority order
        for cmd_pattern in self._commands:
            for pattern in cmd_pattern.regex_patterns:
                match = pattern.search(text_lower)
                if match:
                    result = CommandResult(
                        detected=True,
                        command_type=cmd_pattern.command_type,
                        priority=cmd_pattern.priority,
                        confidence=1.0,
                        matched_phrase=match.group(),
                        original_text=text,
                    )

                    # Trigger callbacks
                    self._trigger_callbacks(result)

                    return result

        return CommandResult(
            detected=False,
            command_type=CommandType.UNKNOWN,
            priority=CommandPriority.LOW,
            confidence=0.0,
            original_text=text,
        )

    def detect_stop_command(self, text: str) -> bool:
        """
        Quick check for stop commands only.

        Optimized for fast detection of stop commands.

        Args:
            text: Text to check

        Returns:
            True if a stop command is detected
        """
        if not text:
            return False

        text_lower = text.lower().strip()

        # Find the stop command pattern
        for cmd_pattern in self._commands:
            if cmd_pattern.command_type == CommandType.STOP:
                for pattern in cmd_pattern.regex_patterns:
                    if pattern.search(text_lower):
                        return True
                break

        return False

    def on_command(
        self,
        command_type: CommandType,
        callback: Callable[[CommandResult], None],
    ) -> None:
        """
        Register a callback for a specific command type.

        Args:
            command_type: Command type to listen for
            callback: Function to call when command is detected
        """
        if command_type not in self._on_command_callbacks:
            self._on_command_callbacks[command_type] = []
        self._on_command_callbacks[command_type].append(callback)

    def _trigger_callbacks(self, result: CommandResult) -> None:
        """Trigger registered callbacks for a command result."""
        callbacks = self._on_command_callbacks.get(result.command_type, [])
        for callback in callbacks:
            try:
                callback(result)
            except Exception:
                # Don't let callback errors break detection
                pass

    def add_stop_phrase(self, phrase: str) -> None:
        """
        Add a new phrase to recognize as a stop command.

        Args:
            phrase: Phrase to add
        """
        for cmd_pattern in self._commands:
            if cmd_pattern.command_type == CommandType.STOP:
                cmd_pattern.patterns.append(phrase)
                cmd_pattern.regex_patterns.append(
                    re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
                )
                break

    def add_command_pattern(self, pattern: CommandPattern) -> None:
        """
        Add a custom command pattern.

        Args:
            pattern: CommandPattern to add
        """
        self._commands.append(pattern)
        self._commands.sort(key=lambda c: c.priority.value, reverse=True)


class QuickStopDetector:
    """
    Ultra-fast stop command detector for low-latency requirements.

    Uses a simplified approach for minimal processing overhead.
    """

    # Optimized stop word set for O(1) lookup
    STOP_WORDS = frozenset([
        "stop",
        "shut",
        "quiet",
        "silence",
        "enough",
        "hush",
        "shush",
    ])

    # Two-word phrases
    STOP_PHRASES = frozenset([
        "shut up",
        "be quiet",
        "stop talking",
        "not now",
        "go away",
        "forget it",
        "stop it",
    ])

    def __init__(self) -> None:
        """Initialize quick stop detector."""
        self._enabled = True

    def detect(self, text: str) -> bool:
        """
        Detect stop commands with minimal overhead.

        Args:
            text: Text to check

        Returns:
            True if stop command detected
        """
        if not self._enabled or not text:
            return False

        text_lower = text.lower().strip()

        # Check single words first (fastest)
        words = text_lower.split()
        for word in words:
            # Strip punctuation
            word_clean = word.strip(".,!?\"'")
            if word_clean in self.STOP_WORDS:
                return True

        # Check two-word phrases
        if len(words) >= 2:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i + 1]}".strip(".,!?\"'")
                if phrase in self.STOP_PHRASES:
                    return True

        return False

    def enable(self) -> None:
        """Enable the detector."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the detector."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self._enabled
