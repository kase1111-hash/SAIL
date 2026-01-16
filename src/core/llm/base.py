"""
SAIL LLM Provider Base

Abstract base class for LLM providers with common interface.
All LLM backends (Ollama, llama.cpp, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def usage(self) -> dict[str, int]:
        """Get token usage as dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens or 0,
            "completion_tokens": self.completion_tokens or 0,
            "total_tokens": self.total_tokens or 0,
        }


@dataclass
class StreamChunk:
    """A chunk from streaming generation."""

    content: str
    done: bool = False
    finish_reason: str | None = None


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMConnectionError(LLMProviderError):
    """Raised when connection to LLM service fails."""

    pass


class LLMGenerationError(LLMProviderError):
    """Raised when generation fails."""

    pass


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM backends must implement this interface to ensure
    consistent behavior across different inference engines.
    """

    def __init__(self, model: str, base_url: str | None = None):
        """
        Initialize the provider.

        Args:
            model: Model name/identifier to use
            base_url: Optional base URL for the API
        """
        self.model = model
        self.base_url = base_url

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a response from the model.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Returns:
            GenerationResult with the model's response

        Raises:
            LLMConnectionError: If connection fails
            LLMGenerationError: If generation fails
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response from the model.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Yields:
            StreamChunk objects as they arrive

        Raises:
            LLMConnectionError: If connection fails
            LLMGenerationError: If generation fails
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is reachable and ready
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models.

        Returns:
            List of model names/identifiers
        """
        pass

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """
        Convenience method for simple text generation.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            config: Optional generation configuration

        Returns:
            Generated text string
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        result = await self.generate(messages, config)
        return result.content

    async def stream_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Convenience method for simple streaming text generation.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            config: Optional generation configuration

        Yields:
            Text chunks as they arrive
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        async for chunk in self.stream(messages, config):
            yield chunk.content
