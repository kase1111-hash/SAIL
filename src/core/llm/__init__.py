"""
SAIL LLM Integration Module

Provides local LLM inference through multiple backends (Ollama, llama.cpp).
Includes prompt management, constitutional constraints, and response parsing.
"""

from .base import (
    GenerationConfig,
    GenerationResult,
    LLMConnectionError,
    LLMGenerationError,
    LLMProvider,
    LLMProviderError,
    LLMProviderType,
    Message,
    StreamChunk,
)
from .constitution import (
    Constitution,
    ConstitutionalCheck,
    ConstitutionalPrinciple,
    ConstitutionalViolation,
    UncertaintyDetector,
    get_constitution,
)
from .factory import (
    LLMProviderFactory,
    create_provider,
    create_provider_from_config,
)
from .llamacpp import LlamaCppProvider
from .ollama import OllamaProvider
from .parsing import (
    IntentExtractor,
    ParsedResponse,
    ResponseParser,
    ResponseType,
)
from .prompts import (
    PromptLibrary,
    PromptRole,
    PromptTemplate,
    get_prompt_library,
)

__all__ = [
    # Base classes and types
    "LLMProvider",
    "LLMProviderType",
    "LLMProviderError",
    "LLMConnectionError",
    "LLMGenerationError",
    "GenerationConfig",
    "GenerationResult",
    "Message",
    "StreamChunk",
    # Providers
    "OllamaProvider",
    "LlamaCppProvider",
    # Factory
    "LLMProviderFactory",
    "create_provider",
    "create_provider_from_config",
    # Prompts
    "PromptTemplate",
    "PromptRole",
    "PromptLibrary",
    "get_prompt_library",
    # Constitution
    "Constitution",
    "ConstitutionalPrinciple",
    "ConstitutionalViolation",
    "ConstitutionalCheck",
    "UncertaintyDetector",
    "get_constitution",
    # Parsing
    "ResponseParser",
    "ParsedResponse",
    "ResponseType",
    "IntentExtractor",
]
